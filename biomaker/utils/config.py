"""configuration and setup utils"""

import csv
import io
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, cast

import coolname
import rich
import shutup
from omegaconf import OmegaConf
from rich.console import Console
from rich.logging import RichHandler
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

from . import copytree, preproc_data, serialize
from .env import load_local_env
from .response import truncate_text

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


class RichLogHandler(RichHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = Console(file=sys.stderr, stderr=True)

    def emit(self, record):
        # Check if the message is a Rich object
        if isinstance(record.msg, (Tree, Text)):
            # Add padding to match the log message indentation (11 spaces)
            padded_content = Padding(record.msg, (0, 0, 0, 11))
            self.console.print(padded_content)
        else:
            # Use the default RichHandler behavior for regular messages
            super().emit(record)


shutup.mute_warnings()
logging.basicConfig(
    level="WARNING",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichLogHandler(show_time=True, show_level=True, show_path=False, markup=True)
    ],
)
logger = logging.getLogger("biomaker")
logger.setLevel(logging.WARNING)

""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str | None
    temp: float


@dataclass
class SearchConfig:
    max_debug_depth: int
    max_children_per_node: int
    num_bugs: int
    topk_improve: int


@dataclass
class AgentConfig:
    steps: int
    total_time_limit: int  # total time limit for all steps (second)

    k_fold_validation: int
    search: SearchConfig

    expose_prediction: bool
    data_preview: bool

    enable_monai_knowledge_base: bool
    enable_query2doc: bool
    enable_web_search: bool
    web_search_topk: int
    enable_node_tools: bool
    max_node_tool_rounds: int
    max_node_tool_calls: int

    code: StageConfig
    feedback: StageConfig
    knowledge_base: StageConfig


@dataclass
class ExecConfig:
    timeout: int
    parallel_workers: int
    max_generated_backlog: int
    max_draft_generated_backlog: int
    agent_file_name: str
    format_tb_ipython: bool
    enforce_environment: bool
    include_medical_packages: bool


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig

    model_for_all_agents: str | None
    log_level: str


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if (current_index := int(p.name.split("-")[0])) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def _is_usable_rag_index(index_file: Path) -> tuple[bool, str | None]:
    if not index_file.is_file():
        return False, "missing"
    if index_file.stat().st_size <= 0:
        return False, "empty"
    try:
        with open(index_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record.get("page_content"):
                    return False, "first record has no page_content"
                return True, None
    except Exception as exc:
        return False, f"invalid JSONL: {exc}"
    return False, "no non-empty records"


def prep_cfg(cfg: Config):
    load_local_env()

    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    data_dir = Path(cfg.data_dir)
    if str(data_dir).startswith("example_tasks/"):
        data_dir = Path(__file__).parent.parent / data_dir
    cfg.data_dir = data_dir.resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    main_model = (
        os.getenv("BIOMAKER_MAIN_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "deepseek-v4-pro"
    )
    helper_model = (
        os.getenv("BIOMAKER_HELPER_MODEL")
        or os.getenv("OPENAI_HELPER_MODEL")
        or "deepseek-v4-flash"
    )

    cfg.agent.code.model = cfg.agent.code.model or main_model
    cfg.agent.feedback.model = cfg.agent.feedback.model or main_model
    cfg.agent.knowledge_base.model = cfg.agent.knowledge_base.model or helper_model
    cfg.report.model = cfg.report.model or helper_model

    # if `model_for_all_agents` is set
    # override all model settings with the same model
    if cfg.model_for_all_agents is not None:
        cfg.agent.code.model = cfg.model_for_all_agents
        cfg.agent.feedback.model = cfg.model_for_all_agents
        cfg.agent.knowledge_base.model = cfg.model_for_all_agents
        cfg.report.model = cfg.model_for_all_agents

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    if cfg.agent.enable_web_search and not os.getenv("SERPER_API_KEY"):
        logger.warning(
            "Web search is enabled, but SERPER_API_KEY is not configured. "
            "Disabling web search for this run."
        )
        cfg.agent.enable_web_search = False
    elif cfg.agent.enable_web_search and not os.getenv("JINA_API_KEY"):
        logger.warning(
            "Web search is enabled, but JINA_API_KEY is not configured. "
            "Serper search will still run, but page reading will use snippets only."
        )

    rag_index = (
        Path(__file__).resolve().parent.parent
        / "monai_rag"
        / "monai_rag_db"
        / "chunks.jsonl"
    )
    rag_ok, rag_reason = _is_usable_rag_index(rag_index)
    if cfg.agent.enable_monai_knowledge_base and not rag_ok:
        logger.warning(
            "MONAI knowledge base is enabled, but the local index is not usable "
            f"({rag_reason}). "
            "Run `bash scripts/create_rag_db.sh` from the repository root to build it. "
            "Disabling MONAI RAG for this run."
        )
        cfg.agent.enable_monai_knowledge_base = False

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    if cfg.preprocess_data and not cfg.copy_data:
        logger.warning(
            "preprocess_data=True requires a copied input workspace; ignoring copy_data=False for this run."
        )
        cfg.copy_data = True
    use_symlinks = (not cfg.copy_data) and (not cfg.preprocess_data)
    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=use_symlinks)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def _replace_path_with_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target)


def prepare_node_workspace(cfg: Config, node) -> Path:
    """Create an isolated execution workspace for one node.

    The node workspace has its own `working/` and `runfile.py`, while `input/`
    and evaluator files are symlinked to the run workspace so large datasets are
    not copied.
    """
    node_dir = cfg.workspace_dir / "node_workspaces" / node.id[:8]
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "working").mkdir(parents=True, exist_ok=True)

    _replace_path_with_symlink(node_dir / "input", cfg.workspace_dir / "input")
    for filename in ["evaluator.py", "evaluator_prediction_spec.json"]:
        src = cfg.workspace_dir / filename
        if src.exists():
            _replace_path_with_symlink(node_dir / filename, src)

    node.execution_workspace = str(node_dir)
    return node_dir


def sync_node_working_to_run_workspace(cfg: Config, node_workspace: Path) -> None:
    """Mirror one node's working outputs to the run-level working directory."""
    src = node_workspace / "working"
    dst = cfg.workspace_dir / "working"
    if not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def _copy_submission_artifacts(
    submission_path: Path, workspace_dir: Path, log_dir: Path
):
    with open(submission_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        path_columns = [
            name
            for name in (reader.fieldnames or [])
            if "path" in name.lower() or "file" in name.lower()
        ]
        for row in reader:
            for column in path_columns:
                value = (row.get(column) or "").strip()
                if not value:
                    continue

                rel_path = Path(value)
                if rel_path.is_absolute():
                    src = rel_path
                    dst = log_dir / rel_path.name
                else:
                    normalized = Path(str(rel_path).removeprefix("./"))
                    if normalized.parts and normalized.parts[0] == "working":
                        src = workspace_dir / normalized
                        dst = log_dir / normalized
                    else:
                        src = workspace_dir / "working" / normalized
                        dst = log_dir / normalized

                if not src.is_file():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)


def _node_artifact_path(log_dir: Path, node) -> Path:
    step = "na" if node.step is None else f"{int(node.step):03d}"
    return log_dir / "node_artifacts" / f"{step}_{node.id[:8]}"


def _copy_csv_referenced_files(
    csv_path: Path,
    workspace_dir: Path,
    dst_dir: Path,
    file_columns: set[str] | None = None,
) -> None:
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if file_columns is None:
                path_columns = [
                    name
                    for name in (reader.fieldnames or [])
                    if "path" in name.lower() or "file" in name.lower()
                ]
            else:
                path_columns = [
                    name for name in (reader.fieldnames or []) if name in file_columns
                ]
            for row in reader:
                for column in path_columns:
                    value = (row.get(column) or "").strip()
                    if not value:
                        continue
                    raw_path = Path(value)
                    if raw_path.is_absolute():
                        src = raw_path
                    else:
                        normalized = Path(str(raw_path).removeprefix("./"))
                        candidates = [
                            workspace_dir / normalized,
                            workspace_dir / "working" / normalized,
                            workspace_dir / "input" / normalized,
                        ]
                        src = next(
                            (p for p in candidates if p.is_file()), candidates[0]
                        )
                    if not src.is_file():
                        continue
                    try:
                        rel = src.relative_to(workspace_dir / "working")
                    except ValueError:
                        rel = Path("external_files") / src.name
                    dst = dst_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
    except Exception as exc:
        logger.debug("Failed to copy CSV-referenced files from %s: %s", csv_path, exc)


def _copy_node_artifacts(cfg: Config, node) -> None:
    """Persist the candidate-owned artifacts for one node before ./working changes."""
    if node is None:
        return
    source_workspace = Path(node.execution_workspace or cfg.workspace_dir)
    working_dir = source_workspace / "working"
    if not working_dir.is_dir():
        return

    dst_dir = _node_artifact_path(cfg.log_dir, node)
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "runfile.py").write_text(node.code or "", encoding="utf-8")

    metadata = {
        "node_id": node.id,
        "step": node.step,
        "parent_id": node.parent.id if node.parent is not None else None,
        "status": node.node_status,
        "selection_reason": node.selection_reason,
        "metric": (
            None
            if node.metric is None
            else {
                "value": node.metric.value,
                "maximize": node.metric.maximize,
            }
        ),
        "outcome": node.outcome,
        "exec_time": node.exec_time,
        "resource_usage": node.resource_usage,
        "exc_type": node.exc_type,
        "exc_info": node.exc_info,
        "plan": node.plan,
        "analysis": node.analysis,
        "prompt_stats": node.prompt_stats,
        "prompt_messages": node.prompt_messages,
        "prompt": node.prompt,
    }
    (dst_dir / "node_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if node.artifact_report:
        (dst_dir / "artifact_report.json").write_text(
            json.dumps(node.artifact_report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    spec_path = Path(cfg.workspace_dir) / "evaluator_prediction_spec.json"
    if spec_path.is_file():
        shutil.copy2(spec_path, dst_dir / "evaluator_prediction_spec.json")
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception:
            spec = {}
    else:
        spec = {}

    prediction_columns = {
        str(column)
        for column in spec.get("prediction_columns", [])
        if isinstance(column, str)
    }
    file_reference_columns = {
        str(column)
        for column in spec.get("file_reference_columns", [])
        if isinstance(column, str)
    }
    prediction_file_columns = {
        column
        for column in prediction_columns | file_reference_columns
        if "path" in column.lower() or "file" in column.lower()
    }
    if not prediction_file_columns and not prediction_columns:
        prediction_file_columns = set()

    for csv_path in sorted(working_dir.glob("*.csv")):
        dst = dst_dir / csv_path.name
        shutil.copy2(csv_path, dst)
        _copy_csv_referenced_files(
            csv_path,
            source_workspace,
            dst_dir,
            file_columns=prediction_file_columns or None,
        )


def _artifact_alignment_value(report: dict | None, key: str):
    if not isinstance(report, dict):
        return None
    alignment = report.get("alignment")
    if isinstance(alignment, dict):
        return alignment.get(key)
    return None


def _write_results_table(log_dir: Path, journal) -> None:
    path = log_dir / "results.tsv"
    fieldnames = [
        "step",
        "node_id",
        "parent_id",
        "status",
        "metric",
        "maximize",
        "outcome",
        "delta_vs_parent",
        "delta_vs_best",
        "runtime_s",
        "peak_rss_mb",
        "avg_cpu_percent",
        "artifact_available",
        "alignment_method",
        "matched_rows",
        "exc_type",
        "error",
        "selection_reason",
        "plan",
        "analysis",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for node in journal.nodes:
            metric = node.metric.value if node.metric is not None else None
            resources = node.resource_usage or {}
            report = node.artifact_report
            writer.writerow(
                {
                    "step": node.step,
                    "node_id": node.id[:8],
                    "parent_id": node.parent.id[:8] if node.parent else "",
                    "status": node.node_status,
                    "metric": "" if metric is None else metric,
                    "maximize": (
                        "" if node.metric is None else bool(node.metric.maximize)
                    ),
                    "outcome": node.outcome or "",
                    "delta_vs_parent": (
                        "" if node.delta_vs_parent is None else node.delta_vs_parent
                    ),
                    "delta_vs_best": (
                        "" if node.delta_vs_best is None else node.delta_vs_best
                    ),
                    "runtime_s": "" if node.exec_time is None else node.exec_time,
                    "peak_rss_mb": resources.get("peak_rss_mb", ""),
                    "avg_cpu_percent": resources.get("avg_cpu_percent", ""),
                    "artifact_available": (
                        "" if not isinstance(report, dict) else report.get("available")
                    ),
                    "alignment_method": _artifact_alignment_value(report, "method")
                    or "",
                    "matched_rows": _artifact_alignment_value(report, "matched_rows")
                    or "",
                    "exc_type": node.exc_type or "",
                    "error": truncate_text(str(node.exc_info or ""), max_chars=300),
                    "selection_reason": truncate_text(
                        str(node.selection_reason or ""), max_chars=240
                    ),
                    "plan": truncate_text(str(node.plan or ""), max_chars=500),
                    "analysis": truncate_text(str(node.analysis or ""), max_chars=500),
                }
            )


def save_run(cfg: Config, journal, solution_tree: Tree = None):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    _write_results_table(cfg.log_dir, journal)
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    split_dir = Path(cfg.workspace_dir) / "input" / "biomaker_splits"
    if split_dir.is_dir():
        log_split_dir = cfg.log_dir / "biomaker_splits"
        if log_split_dir.exists():
            shutil.rmtree(log_split_dir)
        shutil.copytree(split_dir, log_split_dir)
    evaluator_path = Path(cfg.workspace_dir) / "evaluator.py"
    if evaluator_path.is_file():
        shutil.copy2(evaluator_path, cfg.log_dir / "evaluator.py")
    # create the tree + code visualization
    from . import tree_export

    tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # Save only a runnable best solution. If every node failed, journal.json remains
    # the source of truth instead of writing a misleading best_solution.py.
    best_node = journal.get_best_node(only_good=True)
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)

    latest_node = journal.nodes[-1] if journal.nodes else None
    if latest_node is not None:
        _copy_node_artifacts(cfg, latest_node)

    if latest_node is not None and latest_node is best_node:
        # Copy artifacts only when the latest candidate is the best runnable node.
        # This keeps best_solution.py, submission.csv, masks, and private_latest aligned.
        working_dir = Path(cfg.workspace_dir) / "working"
        submission_path = working_dir / "submission.csv"
        if submission_path.is_file():
            shutil.copy2(submission_path, cfg.log_dir / "submission.csv")
            _copy_submission_artifacts(submission_path, cfg.workspace_dir, cfg.log_dir)

        pred_masks_dir = working_dir / "pred_masks"
        if pred_masks_dir.is_dir():
            log_pred_masks_dir = cfg.log_dir / "pred_masks"
            if log_pred_masks_dir.exists():
                shutil.rmtree(log_pred_masks_dir)
            shutil.copytree(pred_masks_dir, log_pred_masks_dir)

        ckpt_path = Path(cfg.workspace_dir) / "working/best_ckpt.pth"
        if ckpt_path.is_file():
            shutil.copy2(ckpt_path, cfg.log_dir / "best_ckpt.pth")

    if solution_tree is not None:
        # save the solution tree
        console = Console(file=io.StringIO(), record=True)
        console.print(solution_tree)
        html_output = console.export_html(inline_styles=True)
        with open(cfg.log_dir / "tree_snapshot.html", "w") as f:
            f.write(html_output)
