import atexit
import json
import logging
import shutil
import time

from omegaconf import OmegaConf
from rich.text import Text
from rich.tree import Tree

from . import backend
from .agent import BioMakerWorkflow
from .interpreter import Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from .utils.artifact_validation import (
    normalize_working_outputs,
    validate_working_outputs,
)
from .utils.config import load_cfg, load_task_desc, prep_agent_workspace, save_run
from .utils.evaluator_setup import (
    clean_candidate_outputs,
    prepare_evaluator,
    score_candidate_with_evaluator,
)
from .utils.private_evaluation import evaluate_private_outputs
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


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()
    new_node = None if len(journal.nodes) < 1 else journal.nodes[-1]

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""
            color = "blue" if node is new_node else "green"
            s = f"[{style}{color}]● {node.metric.value:.3f}"
            if node is best_node:
                s += " (best)"
        if node is new_node:
            s += " (new)"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def run():
    cfg = load_cfg()
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

    task_desc = load_task_desc(cfg)

    logger.info("Preparing agent workspace (copying and extracting files) ...")
    prep_agent_workspace(cfg)
    private_dir = cfg.log_dir / "private_truth"
    split_protocol = prepare_split_protocol(
        cfg.workspace_dir / "input", private_dir=private_dir
    )
    logger.info("Split protocol:\n%s", split_protocol.prompt_text)
    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
    )
    if isinstance(task_desc, dict):
        task_desc["Data split protocol"] = split_protocol.prompt_text
    else:
        task_desc = {
            "Task description": task_desc,
            "Data split protocol": split_protocol.prompt_text,
        }

    logger.info("Preparing frozen evaluator ...")
    prepared_evaluator = prepare_evaluator(
        workspace_dir=cfg.workspace_dir,
        log_dir=cfg.log_dir,
        task_desc=task_desc,
        model=cfg.agent.feedback.model,
        temperature=0.0,
        timeout=min(180, cfg.exec.timeout),
    )
    task_desc["Evaluation protocol"] = (
        "A frozen evaluator has been generated and smoke-tested at `./evaluator.py`. "
        "Strictly follow `evaluator.prediction_spec()` for validation predictions and test submission. "
        "BioMaker computes the final node score after your script exits via no-argument `evaluator.score_val()`. "
        "Key columns in prediction artifacts must exactly match the split CSV values. "
        "If a framework returns key values as tensors or arrays, convert them to ordinary Python scalars or read the key column directly from the source table; never write serialized framework objects such as `tensor(1)`. "
        "Classification probability columns must be complete softmax probabilities for every class and each row must sum to 1; do not fill missing rows with zeros. "
        f"Prediction spec: {prepared_evaluator.prediction_spec}"
    )

    journal = Journal()

    agent = BioMakerWorkflow(
        task_desc=task_desc,
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
        logger.info(Text("Executing code...", style="magenta"))
        node_start_time = time.monotonic()
        clean_candidate_outputs(cfg.workspace_dir)
        res = interpreter.run(*args, **kwargs)
        if res.exc_type is None:
            normalization_messages = normalize_working_outputs(
                cfg.workspace_dir, str(task_desc)
            )
            if normalization_messages:
                res.term_out.append("\n" + "\n".join(normalization_messages))
            artifact_check = validate_working_outputs(cfg.workspace_dir, str(task_desc))
            if artifact_check.messages:
                res.term_out.append("\n" + "\n".join(artifact_check.messages))
            if not artifact_check.ok:
                res.exc_type = "ArtifactValidationError"
                res.exc_info = {"reason": "generated artifacts failed validation"}
        if res.exc_type is None:
            evaluator_result = score_candidate_with_evaluator(
                cfg.workspace_dir, timeout=min(180, cfg.exec.timeout)
            )
            if evaluator_result.output:
                res.term_out.append("\n" + evaluator_result.output)
            if evaluator_result.ok:
                res.term_out.append("\n" + evaluator_result.log_line())
            else:
                res.exc_type = "EvaluatorValidationError"
                res.exc_info = {"reason": evaluator_result.error}
                res.term_out.append(
                    "\nEvaluatorValidationError: "
                    + (evaluator_result.error or "external evaluator failed")
                )
        res.exec_time = time.monotonic() - node_start_time
        logger.info(Text("Execution done.", style="magenta"))
        return res

    def agent_loop_exit_handler(end_with_exception: bool):
        logger.info("Exiting agent loop...")

        if cfg.generate_report and not end_with_exception:
            print("Generating final report from journal...")
            report = journal2report(journal, task_desc, cfg.report)
            print(report)
            report_file_path = cfg.log_dir / "report.md"
            with open(report_file_path, "w") as f:
                f.write(report)
            print("Report written to file:", report_file_path)

    timeout_seconds = OmegaConf.select(cfg, "agent.total_time_limit", default=None)
    total_timer = Timer(timeout_seconds=timeout_seconds)

    logger.info(f"Starting BioMaker workflow")
    logger.info(f"Workflow:\n{agent.workflow_status}")
    total_timer.reset()
    # Continue to the next step only when both the step limit and time limit are not reached
    global_step = 0
    end_with_exception = False
    while global_step < total_steps and not total_timer.is_timeout():
        logger.info(f"BioMaker workflow step: {global_step + 1} / {total_steps}")
        try:
            logger.info(Text("Generating code...", style="green"))
            solution_node = agent.step(exec_callback=exec_callback)
        except Exception as e:
            end_with_exception = True
            logger.error(f"Exception during agent step: {e}")
            raise e
        # If the total time limit is reached, remove the last node from the journal
        if total_timer.is_timeout(tolerance=5):
            logger.error("Total time limit reached, stopping the run.")
            journal.pop()
            break
        best_node = journal.get_best_node(only_good=True)
        if solution_node is not None and solution_node is best_node:
            try:
                private_metrics = evaluate_private_outputs(
                    cfg.workspace_dir, private_dir
                )
            except Exception as exc:
                logger.warning("Private evaluation failed: %s", exc)
                private_metrics = None
            if private_metrics:
                private_path = (
                    cfg.log_dir / f"private_metrics_step_{solution_node.step}.json"
                )
                private_path.write_text(
                    json.dumps(private_metrics, indent=2), encoding="utf-8"
                )
                (cfg.log_dir / "private_metrics_latest.json").write_text(
                    json.dumps(private_metrics, indent=2), encoding="utf-8"
                )

        tree = journal_to_rich_tree(journal)
        save_run(cfg, journal, tree)
        logger.info(tree)
        global_step += 1
    interpreter.cleanup_session()
    agent_loop_exit_handler(end_with_exception=end_with_exception)


if __name__ == "__main__":
    run()
