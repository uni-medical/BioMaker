"""configuration and setup utils"""

import io
import logging
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

from . import copytree, preproc_data, serialize, tree_export


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
    handlers=[RichLogHandler(
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True
    )]
)
logger = logging.getLogger("meddle")
logger.setLevel(logging.WARNING)

""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int

@dataclass
class StepConfig:
    proposal: int
    tuning: int
    hpo: int

@dataclass
class AgentConfig:
    steps: int
    step_plan: StepConfig | None
    total_time_limit: int # total time limit for all steps (second)

    k_fold_validation: int
    search: SearchConfig
    
    expose_prediction: bool
    data_preview: bool
    
    force_monai_with_prompt: bool
    enable_monai_knowledge_base: bool
    enable_query2doc: bool
    use_deep_research_knowledge: str | None

    code: StageConfig
    feedback: StageConfig
    knowledge_base: StageConfig


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class Config(Hashable):
    data_dir: Path
    proposed_solution_file: Path | None
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


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

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

    # update the step plan if no detailed plan is provided
    # if the step <= 10, use all steps for proposal
    # if the step > 10, use 5 step for tuning, others for proposal
    if (cfg.agent.step_plan is None):
        tuning_step = 5 if (cfg.agent.steps > 10) else 0
        hpo_step = 2 if (cfg.agent.steps > 25) else 0
        proposal_step = cfg.agent.steps - tuning_step - hpo_step
        cfg.agent.step_plan = StepConfig(
            proposal=proposal_step,
            tuning=tuning_step,
            hpo=hpo_step,
        )

    # if `model_for_all_agents` is set
    # override all model settings with the same model
    if cfg.model_for_all_agents is not None:
        cfg.agent.code.model = cfg.model_for_all_agents
        cfg.agent.knowledge_base.model = cfg.model_for_all_agents
        cfg.report.model = cfg.model_for_all_agents

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

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

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal, solution_tree: Tree=None):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    # create the tree + code visualization
    tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # save the best found solution
    best_node = journal.get_best_node(only_good=False)
    with open(cfg.log_dir / "best_solution.py", "w") as f:
        f.write(best_node.code)

    # Copy submission file if it exists
    submission_path = Path(cfg.workspace_dir) / "working/submission.csv"
    if submission_path.is_file():
        shutil.copy2(submission_path, cfg.log_dir / "submission.csv")
    
    # Copy checkpoint file if it exists
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