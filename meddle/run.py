import atexit
import logging
import shutil
import sys

from omegaconf import OmegaConf
from rich.text import Text
from rich.tree import Tree

from . import backend
from .agent import MeddleWorkflow
from .interpreter import Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from .utils.config import (load_cfg, load_task_desc, prep_agent_workspace,
                           save_run)
from .utils.timer import Timer

# Add TRACE level
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)
logging.Logger.trace = trace


logger = logging.getLogger("meddle")
logger.setLevel(logging.INFO)

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

    logger.setLevel(level)
    backend.logger.setLevel(level)
        
    logger.info(f'Starting run "{cfg.exp_name}"')
    logger.info(f'Config: "{cfg}"')

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    logger.info("Preparing agent workspace (copying and extracting files) ...")
    prep_agent_workspace(cfg)


    journal = Journal()
    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
    )

    agent = MeddleWorkflow(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
        interpreter=interpreter,
    )

    # update the validated steps count
    cfg.agent.steps = agent.update_steps_left()

    global_step = len(journal)
    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)
    atexit.register(cleanup)

    def exec_callback(*args, **kwargs):
        logger.info(Text("Executing code...", style="magenta"))
        res = interpreter.run(*args, **kwargs)
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

    timeout_seconds = OmegaConf.select(cfg, 'agent.total_time_limit', default=None)
    total_timer = Timer(timeout_seconds=timeout_seconds)
    
    logger.info(f'Starting MedDLE workflow')
    logger.info(f"Step plan:\n{agent.step_plan}")
    total_timer.reset()
    # Continue to the next step only when both the step limit and time limit are not reached
    global_step = 0
    end_with_exception = False
    while global_step < cfg.agent.steps and not total_timer.is_timeout():
        logger.info(f"MedDLE workflow step: {global_step + 1} / {cfg.agent.steps}")
        try:
            logger.info(Text("Generating code...", style="green"))
            agent.step(exec_callback=exec_callback)
        except Exception as e:
            end_with_exception = True
            logger.error(f"Exception during agent step: {e}")
            raise e
        # If the total time limit is reached, remove the last node from the journal
        if total_timer.is_timeout(tolerance=5):
            logger.error("Total time limit reached, stopping the run.")
            journal.pop()
            break
        tree = journal_to_rich_tree(journal)
        save_run(cfg, journal, tree)
        logger.info(tree)
        global_step += 1
    interpreter.cleanup_session()
    agent_loop_exit_handler(end_with_exception=end_with_exception)


if __name__ == "__main__":
    run()
