import atexit
import copy
import csv
import json
import logging
import math
import shutil
import signal
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from omegaconf import OmegaConf
from rich.live import Live
from rich.tree import Tree

from . import backend
from .agent import BioMakerWorkflow, RoundNode, WorkflowAgent
from .agents.base_agent import RoundTask
from .agents.proposal_agent import ProposalAgent
from .interpreter import Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from .utils import tree_export
from .utils.artifact_validation import (
    normalize_working_outputs,
    validate_working_outputs,
)
from .utils.config import (
    SHARED_CONSOLE,
    Config,
    load_cfg,
    load_task_desc,
    prep_agent_workspace,
    prepare_node_workspace,
    save_run,
    sync_node_working_to_run_workspace,
)
from .utils.compute import ComputePool
from .utils.candidate_snapshot import write_candidate_snapshot
from .utils.evaluator_setup import (
    clean_candidate_outputs,
    prepare_evaluator,
    score_candidate_with_evaluator,
)
from .utils.events import append_event
from .utils.execution_record import (
    attach_artifact_validation_result,
    attach_evaluator_run_result,
)
from .utils.exp_guide import write_exp_guide
from .utils.experiment_contract import load_experiment_contract
from .utils.final_retrain import run_final_retrain
from .utils.private_evaluation import evaluate_private_outputs
from .utils.run_manifest import write_run_manifest
from .utils.split_protocol import prepare_split_protocol
from .utils.timer import Timer

# Add TRACE level
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")


def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)


logging.Logger.trace = trace


logger = logging.getLogger("biomaker")
logger.setLevel(logging.INFO)  # set default level


def _interpreter_kwargs(cfg):
    kwargs = dict(OmegaConf.to_container(cfg.exec))  # type: ignore
    for runtime_key in (
        "parallel_workers",
        "requested_parallel_workers",
        "effective_parallel_workers",
        "compute_slots",
    ):
        kwargs.pop(runtime_key, None)
    return kwargs


def _best_committed_node(nodes: list[Node]) -> Node | None:
    candidates = [
        node
        for node in nodes
        if node.step is not None and node.is_buggy is False and node.metric is not None
    ]
    return (
        sorted(candidates, key=lambda node: node.metric, reverse=True)[0]
        if candidates
        else None
    )


def journal_to_rich_tree(
    journal: Journal,
    mark_new: bool = True,
    title: str = "[bold blue]Solution tree",
    nodes: list[Node] | None = None,
):
    now = time.monotonic()

    def display_step(node: Node) -> int | None:
        if node.step is not None:
            return node.step
        value = node.scheduled_step
        return value if isinstance(value, int) else None

    display_nodes = journal.nodes if nodes is None else nodes
    completed_nodes = [node for node in display_nodes if node.step is not None]
    best_node = _best_committed_node(display_nodes)
    new_node = completed_nodes[-1] if mark_new and completed_nodes else None
    children_by_parent: dict[Node | None, list[Node]] = {}
    display_node_set = set(display_nodes)
    for node in display_nodes:
        parent = node.parent if node.parent in display_node_set else None
        children_by_parent.setdefault(parent, []).append(node)
    for children in children_by_parent.values():
        children.sort(
            key=lambda child: (
                display_step(child) if display_step(child) is not None else 10**9
            )
        )

    def append_rec(node: Node, tree):
        if node.step is None:
            step = display_step(node)
            step_prefix = "" if step is None else f"{step}: "
            status = node.tag or "pending"
            elapsed = ""
            if status in {"generating", "running"}:
                status_started_at = node.status_started_at
                if status_started_at is not None:
                    elapsed = f" ({max(0, int(now - status_started_at))}s)"
            color = (
                "cyan"
                if status == "completed"
                else "magenta" if status == "running" else "yellow"
            )
            if status == "completed" and node.is_buggy:
                s = f"[red]◍ {step_prefix}bug"
            elif status == "completed" and node.metric is not None:
                s = f"[cyan]● {step_prefix}{node.metric.value:.3f}"
                if node is best_node:
                    s += " (best)"
            elif status == "completed":
                s = f"[yellow]◌ {step_prefix}executed"
            else:
                s = f"[{color}]◌ {step_prefix}{status}{elapsed}"
        elif node.is_buggy:
            s = "[red]◍ bug"
        elif node.metric is None or node.metric.value is None:
            s = f"[yellow]◌ {node.node_status} n/a"
        else:
            style = "bold " if node is best_node else ""
            color = "blue" if node is new_node else "green"
            s = f"[{style}{color}]● {node.metric.value:.3f}"
            if node is best_node:
                s += " (best)"
        if node.step is not None and node is new_node:
            s += " (new)"

        subtree = tree.add(s)
        children = children_by_parent.get(node, [])
        for child in children:
            append_rec(child, subtree)

    tree = Tree(title)
    for n in children_by_parent.get(None, []):
        append_rec(n, tree)
    return tree


def _metric_payload(node: Node) -> dict:
    metric = node.metric
    if metric is None:
        return {"value": None, "maximize": None}
    return {
        "value": metric.value_npsafe,
        "maximize": metric.maximize,
    }


def _pending_round_node_count(round_nodes_by_round: dict) -> int:
    return sum(
        1
        for round_nodes in round_nodes_by_round.values()
        for round_node in round_nodes
        if getattr(round_node.node, "step", None) is None
    )


def _remaining_schedule_budget(
    *,
    total_steps: int,
    committed_steps: int,
    round_nodes_by_round: dict,
    generating_nodes: list[Node],
) -> int:
    """Return how many more nodes may be generated without exceeding the run plan."""
    live_count = _pending_round_node_count(round_nodes_by_round) + len(
        [node for node in generating_nodes if node.step is None]
    )
    return total_steps - committed_steps - live_count


@dataclass
class SchedulerState:
    """Mutable scheduler bookkeeping kept together to avoid split counter semantics."""

    future_to_round_node: dict = field(default_factory=dict)
    round_nodes_by_round: dict[int, list[RoundNode]] = field(default_factory=dict)
    raw_exec_results: dict = field(default_factory=dict)
    parsed_node_ids: set[str] = field(default_factory=set)
    result_appended_node_ids: set[str] = field(default_factory=set)
    generation_future_to_meta: dict = field(default_factory=dict)
    generating_nodes: list[Node] = field(default_factory=list)
    appended_node_ids_by_round: dict[int, list[str]] = field(default_factory=dict)
    pending_execution_queue: list[RoundNode] = field(default_factory=list)
    round_start_step_by_round: dict[int, int] = field(default_factory=dict)
    round_item_count_by_round: dict[int, int] = field(default_factory=dict)
    best_node_before_by_round: dict[int, Node | None] = field(default_factory=dict)

    def live_nodes(self) -> list[Node]:
        nodes = [
            round_node.node
            for round_nodes in self.round_nodes_by_round.values()
            for round_node in round_nodes
        ]
        return [node for node in nodes if node.step is None] + self.generating_nodes

    def generated_count(self) -> int:
        return sum(
            len(round_nodes) for round_nodes in self.round_nodes_by_round.values()
        )

    def remaining_budget(self, total_steps: int, committed_steps: int) -> int:
        return _remaining_schedule_budget(
            total_steps=total_steps,
            committed_steps=committed_steps,
            round_nodes_by_round=self.round_nodes_by_round,
            generating_nodes=self.generating_nodes,
        )


def _clone_node_view(node: Node) -> Node:
    clone = copy.copy(node)
    clone.parent = None
    clone.children = set()
    return clone


def _unique_nodes(nodes: list[Node | None]) -> list[Node]:
    result: list[Node] = []
    seen: set[int] = set()
    for node in nodes:
        if node is None:
            continue
        object_id = id(node)
        if object_id in seen:
            continue
        seen.add(object_id)
        result.append(node)
    return result


def _freeze_generation_inputs(
    *,
    task_parent: Node | None,
    pending_context: list[Node],
    journal_snapshot: list[Node],
) -> tuple[Node | None, list[Node], list[Node]]:
    """Freeze the minimal node view shown to a generation worker."""

    official_nodes = _unique_nodes(journal_snapshot)
    official_map = {id(node): _clone_node_view(node) for node in official_nodes}
    for node in official_nodes:
        clone = official_map[id(node)]
        if node.parent is not None and id(node.parent) in official_map:
            parent = official_map[id(node.parent)]
            clone.parent = parent
            parent.children.add(clone)

    pending_sources = _unique_nodes(pending_context)
    pending_nodes = [
        official_map.get(id(node)) or _clone_node_view(node) for node in pending_sources
    ]
    for source, clone in zip(pending_sources, pending_nodes):
        if source.parent is not None:
            clone.parent = official_map.get(id(source.parent))

    parent_clone = None
    if task_parent is not None:
        parent_clone = official_map.get(id(task_parent))
    return (
        parent_clone,
        pending_nodes,
        [official_map[id(node)] for node in journal_snapshot],
    )


def _count_csv_rows(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            return sum(1 for _row in reader)
    except Exception:
        return None


def _input_relative_path(workspace_dir: Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    path = Path(str(rel_path))
    if path.is_absolute():
        return path
    return workspace_dir / "input" / path


def _compact_task_description(task_desc, max_chars: int = 900) -> str:
    if isinstance(task_desc, dict):
        parts = []
        for key, value in task_desc.items():
            text = str(value).strip()
            if not text:
                continue
            parts.append(f"{key}: {text}")
        raw = "\n".join(parts)
    else:
        raw = str(task_desc or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _run_overview_text(
    task_desc, split_protocol, workspace_dir: Path, private_dir: Path
) -> tuple[str, dict]:
    contract = load_experiment_contract(workspace_dir, hidden_answers_dir=private_dir)
    train_ref = (
        contract.splits["train"].public_ref if "train" in contract.splits else None
    )
    score_ref = (
        contract.splits["score_val"].public_ref
        if "score_val" in contract.splits
        else None
    )
    test_ref = contract.splits["test"].public_ref if "test" in contract.splits else None
    train_rows = _count_csv_rows(contract.split_path("train"))
    score_rows = _count_csv_rows(contract.split_path("score_val"))
    test_rows = _count_csv_rows(contract.split_path("test"))

    private_sources = []
    private_test_rows = None
    test_answers = contract.hidden_answers.get("test")
    if test_answers is not None:
        source = test_answers.path.name
        private_sources.append(source)
        private_rows = _count_csv_rows(private_dir / source)
        if private_rows is not None:
            private_test_rows = private_rows
    if private_test_rows is None:
        private_test_rows = test_rows

    overview = {
        "train_rows": train_rows,
        "node_selection_val_rows": score_rows,
        "agent_test_rows": test_rows,
        "private_test_rows": private_test_rows,
        "train_path": train_ref,
        "node_selection_val_path": score_ref,
        "agent_test_path": test_ref,
        "private_sources": private_sources,
    }

    lines = ["Run overview:"]
    task_summary = _compact_task_description(task_desc)
    if task_summary:
        lines.append("Task description:")
        lines.extend(f"  {line}" for line in task_summary.splitlines())
    lines.append("Data split:")
    if train_ref:
        lines.append(
            f"- Training rows visible to generated code: {train_rows if train_rows is not None else 'unknown'} (`{train_ref}`)."
        )
    if score_ref:
        lines.append(
            f"- Node-selection validation rows, not for generated-code training: {score_rows if score_rows is not None else 'unknown'} (`{score_ref}`; labels hidden from generated code)."
        )
    if test_ref:
        lines.append(
            f"- Test input rows visible only for prediction: {test_rows if test_rows is not None else 'unknown'} (`{test_ref}`)."
        )
    if private_sources:
        source_text = ", ".join(f"`{source}`" for source in private_sources)
        lines.append(
            f"- Completely hidden/private test labels for user-only evaluation: {private_test_rows if private_test_rows is not None else 'unknown'} rows ({source_text})."
        )
    return "\n".join(lines), overview


def _append_private_metrics_entry(cfg, entry: dict) -> None:
    path = cfg.log_dir / "private_metrics.json"
    note = (
        "Private metrics are for user inspection only and are not passed to the agent."
    )
    payload = {
        "version": 1,
        "visible_to_agent": False,
        "note": note,
        "latest": None,
        "history": [],
    }
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload.update(
                    {
                        "version": existing.get("version", payload["version"]),
                        "visible_to_agent": False,
                        "note": existing.get("note", note),
                        "latest": existing.get("latest"),
                        "history": existing.get("history", []),
                    }
                )
        except json.JSONDecodeError:
            logger.warning(
                "Existing private metrics history is invalid JSON; rewriting."
            )

    if not isinstance(payload["history"], list):
        payload["history"] = []
    payload["history"].append(entry)
    payload["latest"] = entry
    payload["summary"] = _private_metrics_summary(payload["history"])
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _final_retrain_timeout_seconds(cfg: Config) -> int:
    return (cfg.exec.timeout * 3 + 1) // 2


def _finite_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _entry_val_metric(entry: dict) -> tuple[float | None, bool]:
    metric = entry.get("metric")
    if not isinstance(metric, dict):
        return None, True
    value = _finite_number(metric.get("value"))
    maximize = metric.get("maximize")
    return value, bool(maximize) if maximize is not None else True


def _entry_private_test_metric(entry: dict) -> tuple[float | None, bool, dict | None]:
    private_metrics = entry.get("private_metrics")
    if not isinstance(private_metrics, dict):
        return None, True, None
    results = private_metrics.get("results")
    if not isinstance(results, list):
        return None, True, None
    preferred = None
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("name") == "official_test":
            preferred = result
            break
        if preferred is None:
            preferred = result
    if preferred is None:
        return None, True, None
    value = _finite_number(preferred.get("value"))
    maximize = preferred.get("maximize")
    return value, bool(maximize) if maximize is not None else True, preferred


def _compact_private_summary_entry(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    val_value, _val_maximize = _entry_val_metric(entry)
    test_value, _test_maximize, test_result = _entry_private_test_metric(entry)
    test_metrics = None
    test_metric_name = None
    test_n = None
    if isinstance(test_result, dict):
        test_metrics = test_result.get("metrics")
        test_metric_name = test_result.get("metric_name")
        test_n = test_result.get("n")
    return {
        "workflow_step": entry.get("workflow_step"),
        "node_step": entry.get("node_step"),
        "node_id": entry.get("node_id"),
        "parent_id": entry.get("parent_id"),
        "stage": entry.get("stage"),
        "outcome": entry.get("outcome"),
        "is_buggy": entry.get("is_buggy"),
        "val_metric": val_value,
        "val_metric_maximize": _entry_val_metric(entry)[1],
        "private_test_metric": test_value,
        "private_test_metric_name": test_metric_name,
        "private_test_metrics": test_metrics,
        "private_test_n": test_n,
        "created_at": entry.get("created_at"),
    }


def _private_metrics_summary(history: list) -> dict:
    entries = [entry for entry in history if isinstance(entry, dict)]
    eligible_entries = [entry for entry in entries if entry.get("is_buggy") is not True]

    def better_by_val(entry: dict):
        value, maximize = _entry_val_metric(entry)
        if value is None:
            return None
        return value if maximize else -value

    def better_by_private_test(entry: dict):
        value, maximize, _result = _entry_private_test_metric(entry)
        if value is None:
            return None
        return value if maximize else -value

    val_candidates = [
        entry for entry in eligible_entries if better_by_val(entry) is not None
    ]
    test_candidates = [
        entry for entry in eligible_entries if better_by_private_test(entry) is not None
    ]
    val_best = (
        max(val_candidates, key=lambda entry: better_by_val(entry))
        if val_candidates
        else None
    )
    private_test_best = (
        max(test_candidates, key=lambda entry: better_by_private_test(entry))
        if test_candidates
        else None
    )
    return {
        "val_best": _compact_private_summary_entry(val_best),
        "private_test_best": _compact_private_summary_entry(private_test_best),
        "same_node": (
            bool(val_best is not None and private_test_best is not None)
            and val_best.get("node_id") == private_test_best.get("node_id")
        ),
        "history_count": len(entries),
        "eligible_history_count": len(eligible_entries),
    }


def _run(cfg: Config) -> int:
    requested_parallel_workers = cfg.exec.parallel_workers
    compute_pool = ComputePool.detect(requested_parallel_workers)
    cfg.exec.requested_parallel_workers = requested_parallel_workers
    cfg.exec.effective_parallel_workers = compute_pool.effective_workers
    cfg.exec.compute_slots = [slot.as_dict() for slot in compute_pool.slots]
    parallel_workers = compute_pool.effective_workers

    log_level = cfg.log_level.upper()
    level = getattr(logging, log_level, None)
    if log_level == "TRACE":
        level = TRACE_LEVEL_NUM

    if not isinstance(level, int):
        logger.warning(f"Invalid log level: {cfg.log_level}, defaulting to INFO")
        level = logging.INFO

    # 设置日志级别
    logger.setLevel(level)
    backend.logger.setLevel(level)

    # 创建文件处理器
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg.log_dir / f"{cfg.exp_name}.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # 添加文件处理器到logger
    logger.addHandler(file_handler)
    backend.logger.addHandler(file_handler)

    logger.info(f'Starting run "{cfg.exp_name}"')
    logger.info(f'Config: "{cfg}"')
    append_event(
        cfg.log_dir,
        "run_started",
        exp_name=cfg.exp_name,
        data_dir=cfg.data_dir,
        workspace_dir=cfg.workspace_dir,
        steps=cfg.agent.steps,
        per_node_timeout_seconds=cfg.exec.timeout,
        parallel_workers=parallel_workers,
        requested_parallel_workers=requested_parallel_workers,
        effective_parallel_workers=parallel_workers,
        compute_slots=[slot.as_dict() for slot in compute_pool.slots],
        total_time_limit_seconds=cfg.agent.total_time_limit,
    )
    compute_manifest_path = write_run_manifest(cfg, None)
    append_event(
        cfg.log_dir,
        "run_manifest_updated",
        path=compute_manifest_path,
        stage="compute_configured",
    )
    if compute_pool.warning_metadata is not None:
        logger.warning(compute_pool.warning_metadata["message"])
        append_event(cfg.log_dir, "run_warning", **compute_pool.warning_metadata)

    task_desc = load_task_desc(cfg)
    manifest_path = write_run_manifest(cfg, task_desc)
    append_event(
        cfg.log_dir,
        "run_manifest_updated",
        path=manifest_path,
        stage="task_loaded",
    )

    logger.info("Preparing agent workspace ...")
    prep_agent_workspace(cfg)
    append_event(cfg.log_dir, "workspace_prepared", workspace_dir=cfg.workspace_dir)
    private_dir = cfg.log_dir / "private_gt"
    split_protocol = prepare_split_protocol(
        cfg.workspace_dir / "input",
        private_dir=private_dir,
        task_desc=task_desc,
        model=cfg.agent.feedback.model,
        temperature=0.0,
        attempts=cfg.split_builder.attempts,
        cache_root=cfg.resource_dir / "split_builder",
    )
    append_event(
        cfg.log_dir,
        "split_protocol_prepared",
        private_dir=private_dir,
        prompt_chars=len(split_protocol.prompt_text),
    )
    overview_text, overview_payload = _run_overview_text(
        task_desc,
        split_protocol,
        workspace_dir=cfg.workspace_dir,
        private_dir=private_dir,
    )
    logger.info("%s", overview_text)
    append_event(cfg.log_dir, "run_overview", **overview_payload)
    logger.info("Split protocol:\n%s", split_protocol.prompt_text)

    interpreter = Interpreter(
        cfg.workspace_dir,
        **_interpreter_kwargs(cfg),
    )
    agent_task_desc = copy.deepcopy(task_desc)

    def refresh_exp_guide(stage: str, prediction_spec_data: dict | None = None) -> None:
        exp_guide_path = write_exp_guide(
            cfg.workspace_dir,
            prediction_spec=prediction_spec_data,
        )
        shutil.copy2(exp_guide_path, cfg.log_dir / exp_guide_path.name)
        append_event(
            cfg.log_dir,
            "exp_guide_written",
            path=exp_guide_path,
            stage=stage,
        )

    refresh_exp_guide("before_evaluator")

    logger.info("Preparing frozen evaluator ...")
    prepared_evaluator = prepare_evaluator(
        workspace_dir=cfg.workspace_dir,
        log_dir=cfg.log_dir,
        task_desc=task_desc,
        model=cfg.agent.feedback.model,
        temperature=0.0,
        timeout=min(180, cfg.exec.timeout),
        hidden_score_path=cfg.log_dir / "private_gt" / "score_val.csv",
        cache_root=(
            cfg.resource_dir / "evaluators" if cfg.evaluator_cache.enabled else None
        ),
    )
    append_event(
        cfg.log_dir,
        "evaluator_prepared",
        prediction_spec_chars=len(prepared_evaluator.prediction_spec),
    )
    try:
        prediction_spec = json.loads(prepared_evaluator.prediction_spec)
    except Exception:
        prediction_spec = {"raw": prepared_evaluator.prediction_spec}
    refresh_exp_guide(
        "after_evaluator",
        prediction_spec if isinstance(prediction_spec, dict) else None,
    )
    manifest_path = write_run_manifest(
        cfg,
        agent_task_desc,
        split_protocol_text=split_protocol.prompt_text,
        prediction_spec=prediction_spec,
    )
    append_event(
        cfg.log_dir,
        "run_manifest_updated",
        path=manifest_path,
        stage="evaluator_ready",
    )

    journal = Journal()

    agent = BioMakerWorkflow(
        task_desc=agent_task_desc,
        cfg=cfg,
        journal=journal,
        interpreter=interpreter,
    )

    total_steps = agent.update_steps_left()

    global_step = len(journal)

    def cleanup():
        if global_step == 0 and cfg.workspace_dir.exists():
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    def exec_callback(*args, **kwargs):
        node_start_time = time.monotonic()
        node = kwargs.pop("node", None)
        if node is not None:
            node.mark_status("running", node_start_time)
        execution_workspace = (
            prepare_node_workspace(cfg, node) if node is not None else cfg.workspace_dir
        )
        with compute_pool.lease() as slot:
            slot_payload = slot.as_dict()
            append_event(
                cfg.log_dir,
                "execution_started",
                code_chars=len(str(args[0])) if args else None,
                node_id=node.id if node is not None else None,
                execution_workspace=str(execution_workspace),
                timeout_seconds=cfg.exec.timeout,
                compute_backend=slot.backend,
                compute_slot=slot_payload,
            )
            try:
                clean_candidate_outputs(execution_workspace)
                node_interpreter = Interpreter(
                    execution_workspace,
                    **_interpreter_kwargs(cfg),
                    environment=slot.env_overrides,
                    output_label=node.id if node is not None else None,
                )
                res = node_interpreter.run(*args, **kwargs)
                if res.exc_type is None:
                    append_event(cfg.log_dir, "candidate_validation_started", node_id=node.id if node is not None else None)
                    normalization_messages = normalize_working_outputs(
                        execution_workspace, str(agent_task_desc)
                    )
                    if normalization_messages:
                        res.append_system_output(
                            "artifact_normalization",
                            "\n".join(normalization_messages),
                            details={"messages": normalization_messages},
                        )
                    artifact_check = validate_working_outputs(
                        execution_workspace, str(agent_task_desc)
                    )
                    attach_artifact_validation_result(
                        res,
                        artifact_check,
                        failure_reason="generated artifacts failed validation",
                    )
                if res.exc_type is None:
                    append_event(cfg.log_dir, "candidate_evaluation_started", node_id=node.id if node is not None else None)
                    evaluator_result = score_candidate_with_evaluator(
                        execution_workspace,
                        timeout=min(180, cfg.exec.timeout),
                        hidden_score_path=cfg.log_dir / "private_gt" / "score_val.csv",
                        env_overrides=slot.env_overrides,
                    )
                    attach_evaluator_run_result(
                        res,
                        evaluator_result,
                        failure_phase="evaluator_scoring",
                        failure_type="EvaluatorValidationError",
                        failure_message_prefix="EvaluatorValidationError",
                    )
                res.exec_time = time.monotonic() - node_start_time
                append_event(
                    cfg.log_dir,
                    "execution_finished",
                    node_id=node.id if node is not None else None,
                    execution_workspace=str(execution_workspace),
                    exec_time=res.exec_time,
                    exc_type=res.exc_type,
                    resource_usage=res.resource_usage,
                    compute_backend=slot.backend,
                    compute_slot=slot_payload,
                )
                return res
            except BaseException as exc:
                append_event(
                    cfg.log_dir,
                    "execution_finished",
                    node_id=node.id if node is not None else None,
                    execution_workspace=str(execution_workspace),
                    exec_time=time.monotonic() - node_start_time,
                    exc_type=type(exc).__name__,
                    error=str(exc),
                    compute_backend=slot.backend,
                    compute_slot=slot_payload,
                )
                raise

    def live_display_journal(extra_nodes: list[Node] | None = None) -> Journal:
        transient_nodes = [
            node
            for node in (extra_nodes or [])
            if node is not None and node.step is None
        ]
        return (
            Journal(nodes=[*journal.nodes, *transient_nodes])
            if transient_nodes
            else journal
        )

    presentation_warnings: set[str] = set()

    def record_presentation_warning(component: str, exc: BaseException) -> None:
        """Record a best-effort UI warning without risking the experiment."""
        if component in presentation_warnings:
            return
        presentation_warnings.add(component)
        try:
            append_event(
                cfg.log_dir,
                "run_warning",
                component=component,
                exc_type=type(exc).__name__,
                error=str(exc),
            )
        except Exception:
            # A warning sink is presentation infrastructure too. Its failure
            # must not turn the original display problem into a run failure.
            pass
        try:
            logger.warning("%s unavailable; continuing the run: %s", component, exc)
        except Exception:
            # The warning destination is itself presentation infrastructure.
            pass

    def save_live_tree(extra_nodes: list[Node] | None = None) -> Journal:
        """Best-effort tree export for completed and transient nodes."""
        display_journal = live_display_journal(extra_nodes)
        try:
            write_candidate_snapshot(cfg.log_dir, display_journal.nodes)
        except Exception as exc:
            record_presentation_warning("candidate_snapshot", exc)
        try:
            tree_export.generate(
                cfg,
                display_journal,
                cfg.log_dir / "tree_plot.html",
                live=display_journal is not journal,
            )
        except Exception as exc:
            record_presentation_warning("tree_export", exc)
        return display_journal

    def agent_loop_exit_handler(end_with_exception: bool):
        logger.info("Exiting agent loop...")

        if cfg.generate_report and not end_with_exception:
            append_event(cfg.log_dir, "report_started")
            report = journal2report(journal, agent_task_desc, cfg.report)
            report_file_path = cfg.log_dir / "report.md"
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(report)
            append_event(
                cfg.log_dir,
                "report_finished",
                path=report_file_path,
                chars=len(report),
            )
            try:
                print("Generating final report from journal...")
                print(report)
                print("Report written to file:", report_file_path)
            except Exception as exc:
                record_presentation_warning("report_console", exc)

    timeout_seconds = OmegaConf.select(cfg, "agent.total_time_limit", default=None)
    total_timer = Timer(timeout_seconds=timeout_seconds)

    logger.info("Starting BioMaker workflow")
    logger.info(f"Workflow:\n{agent.workflow_status}")
    total_timer.reset()
    # Continue to the next step only when both the step limit and time limit are not reached
    global_step = 0
    round_index = 0
    end_with_exception = False
    workflow_agent: WorkflowAgent = agent.workflow[0]
    proposal_agent = agent.proposal_agent
    scheduler = SchedulerState()
    future_to_round_node = scheduler.future_to_round_node
    round_nodes_by_round = scheduler.round_nodes_by_round
    raw_exec_results = scheduler.raw_exec_results
    parsed_node_ids = scheduler.parsed_node_ids
    result_appended_node_ids = scheduler.result_appended_node_ids
    generation_future_to_meta = scheduler.generation_future_to_meta
    generating_nodes = scheduler.generating_nodes
    appended_node_ids_by_round = scheduler.appended_node_ids_by_round
    pending_execution_queue = scheduler.pending_execution_queue
    round_start_step_by_round = scheduler.round_start_step_by_round
    round_item_count_by_round = scheduler.round_item_count_by_round
    best_node_before_by_round = scheduler.best_node_before_by_round
    live_tree_lock = RLock()
    live_tree_display = None
    live_tree_display_started = False
    live_tree_display_enabled = bool(SHARED_CONSOLE.is_terminal)
    if live_tree_display_enabled:
        try:
            live_tree_display = Live(
                console=SHARED_CONSOLE,
                auto_refresh=False,
                transient=False,
                vertical_overflow="visible",
            )
        except Exception as exc:
            live_tree_display_enabled = False
            record_presentation_warning("terminal_live_display", exc)

    def live_nodes() -> list[Node]:
        return scheduler.live_nodes()

    def live_display_nodes() -> list[Node]:
        return live_nodes()

    def live_tree_view(display_journal: Journal):
        return journal_to_rich_tree(display_journal)

    def refresh_live_tree(log_tree: bool = True) -> None:
        nonlocal live_tree_display_started, live_tree_display_enabled
        with live_tree_lock:
            display_journal = save_live_tree(live_display_nodes())
            if log_tree and live_tree_display_enabled:
                if not live_tree_display_started:
                    try:
                        assert live_tree_display is not None
                        live_tree_display.start()
                        live_tree_display_started = True
                    except Exception as exc:
                        live_tree_display_enabled = False
                        record_presentation_warning("terminal_live_display", exc)
                        return
                try:
                    assert live_tree_display is not None
                    live_tree_display.update(
                        live_tree_view(display_journal),
                        refresh=True,
                    )
                except Exception as exc:
                    live_tree_display_enabled = False
                    record_presentation_warning("terminal_live_display", exc)

    def remaining_schedule_budget() -> int:
        return scheduler.remaining_budget(total_steps, global_step)

    def submit_ready_nodes(executor: ThreadPoolExecutor) -> None:
        changed = False
        while len(future_to_round_node) < parallel_workers and pending_execution_queue:
            round_node = pending_execution_queue.pop(0)
            if round_node.node.step is not None:
                continue
            if round_node.node.tag == "completed":
                continue
            round_node.node.mark_status("running")
            future = executor.submit(
                exec_callback,
                round_node.node.code,
                True,
                node=round_node.node,
            )
            future_to_round_node[future] = round_node
            append_event(
                cfg.log_dir,
                "node_submitted",
                round_index=round_node.round_index,
                round_item_index=round_node.round_item_index,
                round_item_count=round_node.round_item_count,
                node_id=round_node.node.id,
                stage=round_node.node.node_status,
                parent_id=(
                    round_node.node.parent.id
                    if round_node.node.parent is not None
                    else None
                ),
            )
            changed = True
        if changed:
            refresh_live_tree(log_tree=True)

    def submit_generation_task(
        generation_executor: ThreadPoolExecutor,
        task: RoundTask,
        task_round_index: int,
        item_index: int,
        item_count: int,
        journal_snapshot: list[Node],
        scheduled_step: int | None = None,
    ) -> bool:
        if remaining_schedule_budget() <= 0:
            return False
        placeholder = Node(
            code="",
            plan="Code generation in progress.",
            parent=task.parent,
            selection_reason=task.selection_reason,
        )
        placeholder.agent_name = workflow_agent.name
        placeholder.assign_schedule(
            round_index=task_round_index,
            scheduled_step=scheduled_step,
        )
        placeholder.mark_status("generating", time.monotonic())
        if placeholder.parent is not None:
            placeholder.parent.children.discard(placeholder)
        generating_nodes.append(placeholder)
        append_event(
            cfg.log_dir,
            "node_generation_started",
            round_index=task_round_index,
            round_item_index=item_index,
            round_item_count=item_count,
            scheduled_step=scheduled_step,
            placeholder_id=placeholder.id,
            stage=placeholder.node_status,
            parent_id=placeholder.parent.id if placeholder.parent is not None else None,
            selection_reason=task.selection_reason,
        )
        refresh_live_tree()

        pending_context = []
        seen_pending_ids = {placeholder.id}
        for node in live_nodes():
            if node.id not in seen_pending_ids:
                pending_context.append(node)
                seen_pending_ids.add(node.id)
        for round_node in round_nodes_by_round.get(task_round_index, []):
            node = round_node.node
            if node.id in seen_pending_ids:
                continue
            pending_context.append(node)
            seen_pending_ids.add(node.id)
        frozen_task_parent, worker_pending_context, worker_journal_snapshot = (
            _freeze_generation_inputs(
                task_parent=task.parent,
                pending_context=pending_context,
                journal_snapshot=journal_snapshot,
            )
        )
        worker_task = RoundTask(
            parent=frozen_task_parent,
            selection_reason=task.selection_reason,
            draft_plan=task.draft_plan,
        )
        future = generation_executor.submit(
            generate_node_in_worker,
            worker_task,
            worker_pending_context,
            worker_journal_snapshot,
            item_index,
            item_count,
            scheduled_step,
        )
        generation_future_to_meta[future] = {
            "task": task,
            "placeholder": placeholder,
            "round_index": task_round_index,
            "item_index": item_index,
            "item_count": item_count,
            "scheduled_step": scheduled_step,
        }
        return True

    def generate_node_in_worker(
        task: RoundTask,
        pending_context: list[Node],
        journal_snapshot: list[Node],
        item_index: int,
        item_count: int,
        scheduled_step: int | None,
    ) -> Node:
        worker_agent = ProposalAgent(
            task_desc=agent_task_desc,
            cfg=cfg,
            journal=Journal(nodes=list(journal_snapshot)),
        )
        worker_agent.data_summary = proposal_agent.data_summary
        worker_agent.set_generation_context(
            pending_round_nodes=pending_context,
            round_item_index=item_index,
            round_item_count=item_count,
            scheduled_step=scheduled_step,
            draft_plan=task.draft_plan,
        )
        try:
            return worker_agent.generate_node(task)
        finally:
            worker_agent.clear_generation_context()

    def reattach_generated_node_to_live_parent(
        node: Node, live_parent: Node | None
    ) -> None:
        """Move a worker-generated node from the private prompt graph to the live tree."""
        if node.parent is not None:
            node.parent.children.discard(node)
        node.parent = live_parent
        if live_parent is not None:
            live_parent.children.add(node)

    def handle_finished_generations(
        executor: ThreadPoolExecutor,
        block: bool = False,
    ) -> None:
        futures = list(generation_future_to_meta)
        if not futures:
            return
        done = [future for future in futures if future.done()]
        if block and not done:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
        changed = False
        for future in list(done):
            meta = generation_future_to_meta.pop(future)
            placeholder = meta["placeholder"]
            try:
                node = future.result()
            except Exception as exc:
                if placeholder in generating_nodes:
                    generating_nodes.remove(placeholder)
                append_event(
                    cfg.log_dir,
                    "node_generation_failed",
                    round_index=meta["round_index"],
                    round_item_index=meta["item_index"],
                    round_item_count=meta["item_count"],
                    scheduled_step=meta.get("scheduled_step"),
                    placeholder_id=placeholder.id,
                    stage=placeholder.node_status,
                    parent_id=(
                        placeholder.parent.id
                        if placeholder.parent is not None
                        else None
                    ),
                    exc_type=type(exc).__name__,
                    error=str(exc),
                )
                refresh_live_tree(log_tree=True)
                changed = True
                continue
            task = meta["task"]
            reattach_generated_node_to_live_parent(node, task.parent)
            node.id = placeholder.id
            node.ctime = placeholder.ctime
            node.agent_name = workflow_agent.name
            node.selection_reason = task.selection_reason
            node.assign_schedule(
                round_index=meta["round_index"],
                scheduled_step=meta.get("scheduled_step"),
            )
            node.mark_status("generated")
            if placeholder in generating_nodes:
                generating_nodes.remove(placeholder)
            round_node = RoundNode(
                workflow_agent=workflow_agent,
                node=node,
                round_index=meta["round_index"],
                round_item_index=meta["item_index"],
                round_item_count=meta["item_count"],
            )
            round_nodes_by_round.setdefault(meta["round_index"], []).append(round_node)
            append_event(
                cfg.log_dir,
                "node_generation_finished",
                round_index=meta["round_index"],
                round_item_index=meta["item_index"],
                round_item_count=meta["item_count"],
                scheduled_step=meta.get("scheduled_step"),
                node_id=node.id,
                stage=node.node_status,
                parent_id=node.parent.id if node.parent is not None else None,
            )
            pending_execution_queue.append(round_node)
            append_event(
                cfg.log_dir,
                "node_queued_for_execution",
                round_index=meta["round_index"],
                round_item_index=meta["item_index"],
                round_item_count=meta["item_count"],
                scheduled_step=meta.get("scheduled_step"),
                node_id=node.id,
                stage=node.node_status,
                parent_id=node.parent.id if node.parent is not None else None,
            )
            changed = True
        if changed:
            refresh_live_tree(log_tree=True)
            submit_ready_nodes(executor)

    def handle_scheduler_events(
        executor: ThreadPoolExecutor,
        block: bool = False,
    ) -> None:
        futures = [*generation_future_to_meta, *future_to_round_node]
        done = [future for future in futures if future.done()]
        if block and futures and not done:
            wait(futures, return_when=FIRST_COMPLETED)
        handle_finished_generations(executor, block=False)
        handle_finished_futures(block=False, executor=executor)
        submit_ready_nodes(executor)

    def parse_round_node_if_ready(round_node: RoundNode) -> None:
        if round_node.node.id in parsed_node_ids:
            return
        if round_node.node.id not in raw_exec_results:
            return
        agent.parse_round_result(
            round_node,
            raw_exec_results[round_node.node.id],
            best_node_before=best_node_before_by_round.get(round_node.round_index),
        )
        parsed_node_ids.add(round_node.node.id)
        round_node.node.mark_status("completed")
        append_result_for_round_node(round_node)
        refresh_live_tree(log_tree=True)

    def handle_finished_futures(
        block: bool = False,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        futures = list(future_to_round_node)
        if not futures:
            return
        done = [future for future in futures if future.done()]
        if block and not done:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
        finished_round_nodes = []
        for future in list(done):
            round_node = future_to_round_node.pop(future)
            exec_result = future.result()
            raw_exec_results.setdefault(round_node.node.id, exec_result)
            round_node.node.mark_status("completed")
            finished_round_nodes.append(round_node)
        if executor is not None:
            submit_ready_nodes(executor)
        for round_node in finished_round_nodes:
            parse_round_node_if_ready(round_node)

    def append_result_for_round_node(round_node: RoundNode) -> None:
        nonlocal global_step
        node = round_node.node
        if node.id in result_appended_node_ids:
            return
        solution_workspace = Path(node.execution_workspace or cfg.workspace_dir)
        solution_node = agent.append_round_node(round_node)
        result_appended_node_ids.add(node.id)
        appended_node_ids_by_round.setdefault(round_node.round_index, []).append(
            node.id
        )
        sync_node_working_to_run_workspace(
            cfg,
            solution_workspace,
            target_name="working_latest",
        )
        sync_node_working_to_run_workspace(cfg, solution_workspace)
        best_node = journal.get_best_node(only_good=True)
        if node is best_node:
            sync_node_working_to_run_workspace(
                cfg,
                solution_workspace,
                target_name="working_best",
            )
        private_metrics = None
        private_error = None
        try:
            with compute_pool.lease() as private_slot:
                private_metrics = evaluate_private_outputs(
                    solution_workspace,
                    private_dir,
                    env_overrides=private_slot.env_overrides,
                )
        except Exception as exc:
            logger.warning("Private evaluation failed: %s", exc)
            private_error = str(exc)
        if (
            private_metrics
            or private_error
            or (private_dir / "private_gt.json").is_file()
        ):
            _append_private_metrics_entry(
                cfg,
                {
                    "workflow_step": global_step,
                    "node_step": node.step,
                    "node_id": node.id,
                    "parent_id": node.parent.id if node.parent is not None else None,
                    "stage": node.node_status,
                    "is_buggy": bool(node.is_buggy),
                    "outcome": node.outcome,
                    "metric": _metric_payload(node),
                    "is_current_best": node is best_node,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "private_metrics_available": private_metrics is not None,
                    "private_metrics": private_metrics,
                    "private_error": private_error,
                },
            )
        try:
            tree = journal_to_rich_tree(journal)
        except Exception as exc:
            tree = None
            record_presentation_warning("tree_snapshot", exc)
        save_run(
            cfg,
            journal,
            tree,
            presentation_error_handler=record_presentation_warning,
        )
        append_event(
            cfg.log_dir,
            "step_saved",
            round_index=round_node.round_index,
            round_item_index=round_node.round_item_index,
            round_item_count=round_node.round_item_count,
            step=solution_node.step,
            node_id=solution_node.id,
            best_node_id=best_node.id if best_node is not None else None,
        )
        global_step += 1

    def official_generation_pending(current_round_index: int) -> bool:
        return any(
            meta.get("round_index") == current_round_index
            for meta in generation_future_to_meta.values()
        )

    def round_ready_to_commit(current_round_index: int) -> bool:
        if official_generation_pending(current_round_index):
            return False
        nodes = round_nodes_by_round.get(current_round_index, [])
        return bool(nodes) and all(
            round_node.node.id in parsed_node_ids for round_node in nodes
        )

    def round_has_scheduled_work(current_round_index: int) -> bool:
        if round_nodes_by_round.get(current_round_index):
            return True
        return any(
            meta.get("round_index") == current_round_index
            for meta in generation_future_to_meta.values()
        )

    generation_executor = ThreadPoolExecutor(max_workers=max(1, total_steps))
    executor = ThreadPoolExecutor(max_workers=parallel_workers)
    try:
        while global_step < total_steps and not total_timer.is_timeout():
            handle_scheduler_events(executor, block=False)
            round_index += 1
            logger.info(
                "BioMaker workflow round %d starting at step %d / %d",
                round_index,
                global_step + 1,
                total_steps,
            )
            try:
                tasks = proposal_agent.select_round_tasks()
                budget = max(0, remaining_schedule_budget())
                filtered_tasks = tasks[: min(budget, workflow_agent.steps_left)]
                item_count = len(filtered_tasks)
                round_start_step_by_round[round_index] = global_step
                round_item_count_by_round[round_index] = item_count
                best_node_before_by_round[round_index] = journal.get_best_node(
                    only_good=True
                )
                append_event(
                    cfg.log_dir,
                    "round_started",
                    round_index=round_index,
                    round_start_step=global_step,
                    planned_items=item_count,
                    parallel_workers=parallel_workers,
                )

                round_journal_snapshot = list(journal.nodes)
                for task_offset, task in enumerate(filtered_tasks, start=1):
                    item_index = task_offset
                    scheduled_step = (
                        round_start_step_by_round.get(round_index, global_step)
                        + item_index
                        - 1
                    )
                    submit_generation_task(
                        generation_executor,
                        task,
                        round_index,
                        item_index,
                        item_count,
                        round_journal_snapshot,
                        scheduled_step=scheduled_step,
                    )
                handle_scheduler_events(executor, block=False)

                if not round_has_scheduled_work(round_index):
                    logger.info("No round tasks selected; stopping the run.")
                    break

                while (
                    not round_ready_to_commit(round_index)
                    and not total_timer.is_timeout()
                ):
                    submit_ready_nodes(executor)
                    handle_scheduler_events(
                        executor,
                        block=not round_ready_to_commit(round_index),
                    )

                append_event(
                    cfg.log_dir,
                    "round_finished",
                    round_index=round_index,
                    completed_steps=global_step,
                    node_ids=appended_node_ids_by_round.get(round_index, []),
                    generated_node_ids=[
                        round_node.node.id
                        for round_node in round_nodes_by_round.get(round_index, [])
                    ],
                )
            except Exception as e:
                end_with_exception = True
                logger.error(f"Exception during async generation/execution round: {e}")
                raise e
    finally:
        try:
            with live_tree_lock:
                display_journal = save_live_tree(live_display_nodes())
                if live_tree_display_started:
                    try:
                        try:
                            assert live_tree_display is not None
                            live_tree_display.update(
                                live_tree_view(display_journal),
                                refresh=True,
                            )
                        except Exception as exc:
                            record_presentation_warning("terminal_live_display", exc)
                    finally:
                        try:
                            assert live_tree_display is not None
                            live_tree_display.stop()
                        except Exception as exc:
                            record_presentation_warning("terminal_live_display", exc)
        finally:
            try:
                generation_executor.shutdown(wait=True)
            finally:
                executor.shutdown(wait=True)
    best_node = journal.get_best_node(only_good=True)
    if cfg.final_retrain.enabled and best_node is not None and not end_with_exception:
        retrain_timeout = _final_retrain_timeout_seconds(cfg)
        with compute_pool.lease(final=True) as final_slot:
            final_slot_payload = final_slot.as_dict()
            append_event(
                cfg.log_dir,
                "final_retrain_started",
                node_id=best_node.id,
                node_step=best_node.step,
                timeout_seconds=retrain_timeout,
                compute_backend=final_slot.backend,
                compute_slot=final_slot_payload,
            )
            retrain_result = run_final_retrain(
                cfg,
                best_node,
                private_dir,
                timeout=retrain_timeout,
                env_overrides=final_slot.env_overrides,
            )
            append_event(
                cfg.log_dir,
                "final_retrain_finished",
                ok=retrain_result.ok,
                workspace_dir=str(retrain_result.workspace_dir),
                train_rows_before=retrain_result.train_rows_before,
                score_rows_added=retrain_result.score_rows_added,
                train_rows_after=retrain_result.train_rows_after,
                private_metrics=retrain_result.private_metrics,
                error=retrain_result.error,
                compute_backend=final_slot.backend,
                compute_slot=final_slot_payload,
            )
        _append_private_metrics_entry(
            cfg,
            {
                "workflow_step": global_step,
                "node_step": best_node.step,
                "node_id": best_node.id,
                "parent_id": (
                    best_node.parent.id if best_node.parent is not None else None
                ),
                "stage": "final_retrain",
                "is_buggy": not retrain_result.ok,
                "outcome": (
                    "final_retrain" if retrain_result.ok else "final_retrain_failed"
                ),
                "metric": _metric_payload(best_node),
                "is_current_best": True,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "private_metrics_available": retrain_result.private_metrics is not None,
                "private_metrics": retrain_result.private_metrics,
                "private_error": retrain_result.error,
                "train_rows_before": retrain_result.train_rows_before,
                "score_rows_added": retrain_result.score_rows_added,
                "train_rows_after": retrain_result.train_rows_after,
            },
        )
        if not retrain_result.ok:
            raise RuntimeError(
                "Final retrain did not complete successfully: "
                + (retrain_result.error or "unknown error")
            )
    elif cfg.final_retrain.enabled and best_node is None and not end_with_exception:
        append_event(cfg.log_dir, "final_retrain_skipped", reason="no_best_node")
        raise RuntimeError("Final retrain is enabled but no valid best node exists.")
    agent_loop_exit_handler(end_with_exception=end_with_exception)
    return global_step


def _run_direct_entry() -> None:
    """Run BioMaker and leave one authoritative terminal lifecycle event."""

    cfg = load_cfg()
    previous_sigterm_handler = None
    terminal_event: str | None = None

    def emit_terminal(event: str, **payload) -> None:
        nonlocal terminal_event
        if terminal_event is not None:
            return
        terminal_event = event
        append_event(cfg.log_dir, event, **payload)

    def handle_sigterm(signum, _frame):
        emit_terminal(
            "run_interrupted",
            reason="sigterm",
            signal=signum,
        )
        raise SystemExit(128 + signum)

    try:
        previous_sigterm_handler = signal.signal(signal.SIGTERM, handle_sigterm)
    except (AttributeError, ValueError):
        # Embedded callers may invoke run() outside the main thread.
        previous_sigterm_handler = None
    try:
        try:
            manifest_path = write_run_manifest(cfg, None)
            append_event(cfg.log_dir, "run_manifest_written", path=manifest_path)
            completed_steps = _run(cfg)
        except KeyboardInterrupt:
            emit_terminal("run_interrupted", reason="keyboard_interrupt")
            raise
        except Exception as exc:
            emit_terminal(
                "run_failed",
                exc_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        emit_terminal("run_finished", completed_steps=completed_steps)
    finally:
        if previous_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


def _queue_cli_invocation() -> bool:
    if os.environ.get("BIOMAKER_QUEUE_DIRECT") == "1":
        return False
    if os.environ.get("BIOMAKER_DISABLE_RUN_QUEUE") == "1":
        return False
    entrypoint = Path(sys.argv[0]).stem.lower()
    return entrypoint in {"biomaker", "run"}


def run() -> None:
    """Submit CLI launches to the durable FIFO, or execute a supervised child."""

    if _queue_cli_invocation():
        queue_sidecar = Path(__file__).resolve().parents[1] / "apps" / "webui"
        if not (queue_sidecar / "webui" / "runner.py").is_file():
            raise RuntimeError(
                "The durable run queue is unavailable in this installation. "
                "Run BioMaker from the source checkout, or set "
                "BIOMAKER_DISABLE_RUN_QUEUE=1 to explicitly opt out of queueing."
            )
        from .utils.queue_client import submit_cli_and_wait

        raise SystemExit(submit_cli_and_wait())
    _run_direct_entry()


if __name__ == "__main__":
    run()
