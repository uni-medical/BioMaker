import atexit
import json
import logging
import math
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Event, RLock, Thread

from omegaconf import OmegaConf
from rich.console import Console, Group
from rich.live import Live
from rich.tree import Tree

from . import backend
from .agent import BioMakerWorkflow, RoundNode, WorkflowAgent
from .agents.base_agent import RoundTask
from .interpreter import Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from .utils.artifact_validation import (
    normalize_working_outputs,
    validate_working_outputs,
)
from .utils import tree_export
from .utils.config import (
    load_cfg,
    load_task_desc,
    prep_agent_workspace,
    prepare_node_workspace,
    save_run,
    sync_node_working_to_run_workspace,
)
from .utils.events import append_event
from .utils.evaluator_setup import (
    clean_candidate_outputs,
    parse_artifact_report,
    parse_external_validation,
    prepare_evaluator,
    score_candidate_with_evaluator,
)
from .utils.metric import MetricValue, WorstMetricValue
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
    kwargs.pop("parallel_workers", None)
    kwargs.pop("max_generated_backlog", None)
    kwargs.pop("max_draft_generated_backlog", None)
    return kwargs


def journal_to_rich_tree(
    journal: Journal,
    mark_new: bool = True,
    title: str = "[bold blue]Solution tree",
    nodes: list[Node] | None = None,
):
    now = time.monotonic()
    display_nodes = journal.nodes if nodes is None else nodes
    completed_nodes = [node for node in display_nodes if node.step is not None]
    display_good_nodes = [
        node
        for node in display_nodes
        if node.is_buggy is False and node.metric is not None
    ]
    best_node = (
        sorted(display_good_nodes, key=lambda node: node.metric, reverse=True)[0]
        if display_good_nodes
        else None
    )
    new_node = completed_nodes[-1] if mark_new and completed_nodes else None
    children_by_parent: dict[Node | None, list[Node]] = {}
    display_node_set = set(display_nodes)
    for node in display_nodes:
        parent = node.parent if node.parent in display_node_set else None
        children_by_parent.setdefault(parent, []).append(node)
    for children in children_by_parent.values():
        children.sort(key=lambda child: child.step if child.step is not None else 10**9)

    def append_rec(node: Node, tree):
        if node.step is None:
            status = node.tag or "pending"
            elapsed = ""
            if status in {"generating", "running"}:
                status_started_at = getattr(node, "_status_started_at", None)
                if status_started_at is not None:
                    elapsed = f" ({max(0, int(now - status_started_at))}s)"
            color = (
                "cyan"
                if status == "completed"
                else "magenta" if status == "running" else "yellow"
            )
            if status == "completed" and node.is_buggy:
                s = "[red]◍ bug"
            elif status == "completed" and node.metric is not None:
                s = f"[cyan]● {node.metric.value:.3f}"
                if node is best_node:
                    s += " (best)"
            elif status == "completed":
                s = "[yellow]◌ executed"
            else:
                s = f"[{color}]◌ {status}{elapsed}"
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
        entry for entry in entries if better_by_val(entry) is not None
    ]
    test_candidates = [
        entry for entry in entries if better_by_private_test(entry) is not None
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
    }


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
    append_event(
        cfg.log_dir,
        "run_started",
        exp_name=cfg.exp_name,
        data_dir=cfg.data_dir,
        workspace_dir=cfg.workspace_dir,
        steps=cfg.agent.steps,
        per_node_timeout_seconds=cfg.exec.timeout,
        parallel_workers=cfg.exec.parallel_workers,
        total_time_limit_seconds=cfg.agent.total_time_limit,
    )

    task_desc = load_task_desc(cfg)

    logger.info("Preparing agent workspace (copying and extracting files) ...")
    prep_agent_workspace(cfg)
    append_event(cfg.log_dir, "workspace_prepared", workspace_dir=cfg.workspace_dir)
    private_dir = cfg.log_dir / "private_truth"
    split_protocol = prepare_split_protocol(
        cfg.workspace_dir / "input", private_dir=private_dir
    )
    append_event(
        cfg.log_dir,
        "split_protocol_prepared",
        private_dir=private_dir,
        prompt_chars=len(split_protocol.prompt_text),
    )
    logger.info("Split protocol:\n%s", split_protocol.prompt_text)
    interpreter = Interpreter(
        cfg.workspace_dir,
        **_interpreter_kwargs(cfg),
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
    append_event(
        cfg.log_dir,
        "evaluator_prepared",
        prediction_spec_chars=len(prepared_evaluator.prediction_spec),
    )
    task_desc["Evaluation protocol"] = (
        "A frozen evaluator has been generated and smoke-tested at `./evaluator.py`. "
        "Strictly follow `evaluator.prediction_spec()` for validation predictions and test submission. "
        "BioMaker computes the final node score after your script exits using hidden score labels that are not available to generated code. "
        "Generated code may call `evaluator.validate_submission()` after writing outputs, but should not call `evaluator.score_val()` because hidden score labels are only supplied to BioMaker's external scoring process. "
        "Key columns in prediction artifacts must exactly match the split CSV values. "
        "If a framework returns key values as tensors or arrays, convert them to ordinary Python scalars or read the key column directly from the source table; never write serialized framework objects such as `tensor(1)`. "
        "Classification probability columns must be complete softmax probabilities for every class and each row must sum to 1; do not fill missing rows with zeros. "
        f"Prediction spec: {prepared_evaluator.prediction_spec}"
    )
    try:
        prediction_spec = json.loads(prepared_evaluator.prediction_spec)
    except Exception:
        prediction_spec = {"raw": prepared_evaluator.prediction_spec}
    metric_name = "n/a"
    if isinstance(prediction_spec, dict):
        metric_value = prediction_spec.get("primary_metric")
        if metric_value is None and isinstance(
            prediction_spec.get("validation_artifact"), dict
        ):
            metric_value = prediction_spec["validation_artifact"].get(
                "primary_metric"
            )
        if metric_value is None and isinstance(prediction_spec.get("metrics"), list):
            metrics = prediction_spec["metrics"]
            if metrics:
                metric_value = metrics[0]
        if metric_value is not None:
            metric_name = str(metric_value)
    manifest_path = write_run_manifest(
        cfg,
        task_desc,
        split_protocol_text=split_protocol.prompt_text,
        prediction_spec=prediction_spec,
    )
    append_event(cfg.log_dir, "run_manifest_written", path=manifest_path)

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
        node_start_time = time.monotonic()
        node = kwargs.pop("node", None)
        if node is not None:
            node.tag = "running"
            setattr(node, "_status_started_at", node_start_time)
        execution_workspace = (
            prepare_node_workspace(cfg, node) if node is not None else cfg.workspace_dir
        )
        append_event(
            cfg.log_dir,
            "execution_started",
            code_chars=len(str(args[0])) if args else None,
            node_id=node.id if node is not None else None,
            execution_workspace=str(execution_workspace),
        )
        clean_candidate_outputs(execution_workspace)
        node_interpreter = Interpreter(
            execution_workspace,
            **_interpreter_kwargs(cfg),
        )
        res = node_interpreter.run(*args, **kwargs)
        if res.exc_type is None:
            normalization_messages = normalize_working_outputs(
                execution_workspace, str(task_desc)
            )
            if normalization_messages:
                res.term_out.append("\n" + "\n".join(normalization_messages))
            artifact_check = validate_working_outputs(
                execution_workspace, str(task_desc)
            )
            if artifact_check.messages:
                res.term_out.append("\n" + "\n".join(artifact_check.messages))
            if not artifact_check.ok:
                res.exc_type = "ArtifactValidationError"
                res.exc_info = {"reason": "generated artifacts failed validation"}
        if res.exc_type is None:
            evaluator_result = score_candidate_with_evaluator(
                execution_workspace,
                timeout=min(180, cfg.exec.timeout),
                score_truth_path=cfg.log_dir / "private_truth" / "score_val_truth.csv",
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
        append_event(
            cfg.log_dir,
            "execution_finished",
            node_id=node.id if node is not None else None,
            execution_workspace=str(execution_workspace),
            exec_time=res.exec_time,
            exc_type=res.exc_type,
            resource_usage=res.resource_usage,
        )
        return res

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

    def save_live_tree(extra_nodes: list[Node] | None = None) -> Journal:
        """Update tree_plot.html with completed journal plus transient round nodes."""
        display_journal = live_display_journal(extra_nodes)
        tree_export.generate(
            cfg,
            display_journal,
            cfg.log_dir / "tree_plot.html",
            live=display_journal is not journal,
        )
        return display_journal

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
    round_index = 0
    end_with_exception = False
    parallel_workers = max(1, int(cfg.exec.parallel_workers))
    workflow_agent: WorkflowAgent = agent.workflow[0]
    proposal_agent = agent.proposal_agent
    future_to_round_node: dict = {}
    round_nodes_by_round: dict[int, list[RoundNode]] = {}
    raw_exec_results = {}
    parsed_node_ids: set[str] = set()
    result_appended_node_ids: set[str] = set()
    generation_future_to_meta: dict = {}
    scheduled_keys_by_round: dict[int, set[tuple[str, str]]] = {}
    speculative_cache_by_round: dict[int, dict[tuple[str, str], dict]] = {}
    speculative_keys_by_round: dict[int, set[tuple[str, str]]] = {}
    generating_nodes: list[Node] = []
    appended_node_ids_by_round: dict[int, list[str]] = {}
    pending_execution_queue: list[RoundNode] = []
    max_generated_backlog = max(0, int(cfg.exec.max_generated_backlog))
    max_draft_generated_backlog = max(0, int(cfg.exec.max_draft_generated_backlog))
    live_tree_refresh_seconds = 1
    live_tree_lock = RLock()
    live_tree_stop = Event()
    last_live_tree_log_at = 0.0
    live_tree_console = Console(file=sys.stderr, stderr=True)
    live_tree_display = Live(
        console=live_tree_console,
        refresh_per_second=1,
        transient=False,
        vertical_overflow="visible",
    )
    live_tree_display_started = False

    def live_nodes() -> list[Node]:
        nodes = []
        for round_nodes in round_nodes_by_round.values():
            nodes.extend(round_node.node for round_node in round_nodes)
        return [node for node in nodes if node.step is None] + generating_nodes

    def cached_speculative_nodes() -> list[Node]:
        nodes: list[Node] = []
        for cache in speculative_cache_by_round.values():
            for cached in cache.values():
                node = cached.get("node")
                if node is not None and node.step is None:
                    nodes.append(node)
        return nodes

    def live_display_nodes() -> list[Node]:
        return [*live_nodes(), *cached_speculative_nodes()]

    def live_backlog_count(next_task: RoundTask | None = None) -> int:
        nodes = [
            node
            for node in [*generating_nodes, *(rn.node for rn in pending_execution_queue)]
            if node.step is None
        ]
        if next_task is not None and task_stage(next_task) == "draft":
            nodes = [node for node in nodes if node.parent is None]
        return len(nodes)

    def live_progress_line() -> str:
        generated = sum(len(round_nodes) for round_nodes in round_nodes_by_round.values())
        executed = len(raw_exec_results)
        live = live_nodes()
        generating = sum(1 for node in live if node.tag == "generating")
        running = sum(1 for node in live if node.tag == "running")
        queued = sum(
            1
            for round_node in pending_execution_queue
            if round_node.node.step is None and round_node.node.tag == "generated"
        )
        best_candidates = [
            node
            for node in [*journal.nodes, *live_display_nodes()]
            if node.is_buggy is False and node.metric is not None
        ]
        best = (
            sorted(best_candidates, key=lambda node: node.metric, reverse=True)[0]
            if best_candidates
            else None
        )
        best_text = (
            f"{metric_name}={best.metric.value:.3f}"
            if best is not None and best.metric is not None
            else "n/a"
        )
        return (
            f"Progress: generated {generated}/{total_steps} · "
            f"executed {executed}/{total_steps} · "
            f"live generating {generating}, running {running}, queued {queued} · "
            f"best {best_text}"
        )

    def completed_nodes_through_round(current_round_index: int) -> list[Node]:
        nodes: list[Node] = []
        for node_round_index in sorted(round_nodes_by_round):
            if node_round_index > current_round_index:
                continue
            nodes.extend(
                round_node.node
                for round_node in round_nodes_by_round[node_round_index]
                if round_node.node.step is not None
            )
        return nodes

    def round_completed_tree(current_round_index: int) -> Tree:
        return journal_to_rich_tree(
            journal,
            mark_new=False,
            title=f"[bold cyan]Solution tree through round {current_round_index}",
            nodes=completed_nodes_through_round(current_round_index),
        )

    def live_tree_view(display_journal: Journal):
        return Group(
            live_progress_line(),
            journal_to_rich_tree(display_journal),
        )

    def refresh_live_tree(log_tree: bool = True) -> None:
        nonlocal last_live_tree_log_at, live_tree_display_started
        with live_tree_lock:
            display_journal = save_live_tree(live_display_nodes())
            if log_tree:
                if not live_tree_display_started:
                    live_tree_display.start()
                    live_tree_display_started = True
                live_tree_display.update(
                    live_tree_view(display_journal),
                    refresh=True,
                )
                last_live_tree_log_at = time.monotonic()

    def live_tree_ticker() -> None:
        while not live_tree_stop.wait(live_tree_refresh_seconds):
            try:
                if live_nodes() and (
                    time.monotonic() - last_live_tree_log_at
                    >= live_tree_refresh_seconds
                ):
                    refresh_live_tree(log_tree=True)
            except Exception as exc:
                logger.debug("Live tree refresh failed: %s", exc)

    def task_stage(task: RoundTask) -> str:
        if task.parent is None:
            return "draft"
        return "debug" if task.parent.is_buggy else "improve"

    def task_key(task: RoundTask) -> tuple[str, str]:
        parent_id = "root" if task.parent is None else task.parent.id
        return (task_stage(task), parent_id)

    def node_round_index(node: Node) -> int:
        value = getattr(node, "_round_index", 0)
        return int(value) if isinstance(value, int) else 0

    def select_tasks_visible_before_round(task_round_index: int) -> list[RoundTask]:
        """Select parents using only nodes from earlier official rounds."""
        original_journal = proposal_agent.journal
        visible_nodes = [
            node
            for node in original_journal.nodes
            if node_round_index(node) < task_round_index
        ]
        proposal_agent.journal = Journal(nodes=visible_nodes)
        try:
            return proposal_agent.select_round_tasks()
        finally:
            proposal_agent.journal = original_journal

    def active_speculative_keys(round_index: int) -> set[tuple[str, str]]:
        keys = set(speculative_keys_by_round.get(round_index, set()))
        for meta in generation_future_to_meta.values():
            if meta.get("speculative") and meta.get("round_index") == round_index:
                keys.add(task_key(meta["task"]))
        return keys

    def remaining_schedule_budget() -> int:
        live_count = sum(
            1
            for round_nodes in round_nodes_by_round.values()
            for round_node in round_nodes
            if round_node.node.step is None
        )
        return total_steps - global_step - live_count - len(generating_nodes)

    def generation_backlog_full(next_task: RoundTask | None = None) -> bool:
        if next_task is not None and task_stage(next_task) == "draft":
            limit = max_draft_generated_backlog
        else:
            limit = max_generated_backlog
        backlog = live_backlog_count(next_task)
        if limit <= 0:
            return backlog > 0
        return backlog >= limit

    def submit_ready_nodes(executor: ThreadPoolExecutor) -> None:
        changed = False
        while len(future_to_round_node) < parallel_workers and pending_execution_queue:
            round_node = pending_execution_queue.pop(0)
            if round_node.node.step is not None:
                continue
            if round_node.node.tag == "completed":
                continue
            round_node.node.tag = "running"
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
                parent_id=round_node.node.parent.id
                if round_node.node.parent is not None
                else None,
            )
            changed = True
        if changed:
            refresh_live_tree(log_tree=False)

    def submit_generation_task(
        generation_executor: ThreadPoolExecutor,
        task: RoundTask,
        task_round_index: int,
        item_index: int,
        item_count: int,
        speculative: bool = False,
    ) -> bool:
        if remaining_schedule_budget() <= 0:
            return False
        placeholder = Node(
            code="",
            plan="Code generation in progress.",
            parent=task.parent,
            tag="generating",
            selection_reason=task.selection_reason,
        )
        placeholder.agent_name = workflow_agent.name
        setattr(placeholder, "_round_index", task_round_index)
        setattr(placeholder, "_status_started_at", time.monotonic())
        if placeholder.parent is not None:
            placeholder.parent.children.discard(placeholder)
        generating_nodes.append(placeholder)
        append_event(
            cfg.log_dir,
            "node_generation_started",
            round_index=task_round_index,
            round_item_index=item_index,
            round_item_count=item_count,
            speculative=speculative,
            placeholder_id=placeholder.id,
            stage=placeholder.node_status,
            parent_id=placeholder.parent.id if placeholder.parent is not None else None,
            selection_reason=task.selection_reason,
        )
        refresh_live_tree()
        if task.parent is not None:
            key = task_key(task)
            if speculative:
                speculative_keys_by_round.setdefault(task_round_index, set()).add(key)
            else:
                scheduled_keys_by_round.setdefault(task_round_index, set()).add(key)

        pending_context = []
        seen_pending_ids = {placeholder.id}
        for node in live_nodes():
            if node.id not in seen_pending_ids:
                pending_context.append(node)
                seen_pending_ids.add(node.id)
        for round_node in round_nodes_by_round.get(task_round_index, []):
            node = round_node.node
            if node.id in seen_pending_ids or node.step is not None:
                continue
            pending_context.append(node)
            seen_pending_ids.add(node.id)
        future = generation_executor.submit(
            generate_node_in_worker, task, pending_context
        )
        generation_future_to_meta[future] = {
            "task": task,
            "placeholder": placeholder,
            "round_index": task_round_index,
            "item_index": item_index,
            "item_count": item_count,
            "speculative": speculative,
        }
        return True

    def generate_node_in_worker(task: RoundTask, pending_context: list[Node]) -> Node:
        proposal_agent.set_pending_round_nodes(pending_context)
        try:
            return proposal_agent.generate_node(task)
        finally:
            proposal_agent.clear_pending_round_nodes()

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
            except Exception:
                task = meta["task"]
                if meta["speculative"] and task.parent is not None:
                    speculative_keys_by_round.setdefault(
                        meta["round_index"], set()
                    ).discard(task_key(task))
                if placeholder in generating_nodes:
                    generating_nodes.remove(placeholder)
                refresh_live_tree(log_tree=True)
                raise
            if node.parent is not None:
                node.parent.children.discard(node)
            node.id = placeholder.id
            node.ctime = placeholder.ctime
            node.agent_name = workflow_agent.name
            setattr(node, "_round_index", meta["round_index"])
            node.tag = "generated"
            if placeholder in generating_nodes:
                generating_nodes.remove(placeholder)

            task = meta["task"]
            if meta["speculative"]:
                key = task_key(task)
                speculative_keys_by_round.setdefault(meta["round_index"], set()).discard(
                    key
                )
                scheduled_keys_by_round.setdefault(meta["round_index"], set()).add(key)

            round_node = RoundNode(
                workflow_agent=workflow_agent,
                node=node,
                round_index=meta["round_index"],
                round_item_index=meta["item_index"],
                round_item_count=meta["item_count"],
            )
            round_nodes_by_round.setdefault(meta["round_index"], []).append(round_node)
            if task.parent is not None:
                scheduled_keys_by_round.setdefault(meta["round_index"], set()).add(
                    task_key(task)
                )
            append_event(
                cfg.log_dir,
                "node_generation_finished",
                round_index=meta["round_index"],
                round_item_index=meta["item_index"],
                round_item_count=meta["item_count"],
                speculative=meta["speculative"],
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
                speculative=meta["speculative"],
                node_id=node.id,
                stage=node.node_status,
                parent_id=node.parent.id if node.parent is not None else None,
            )
            changed = True
        if changed:
            refresh_live_tree(log_tree=False)
            submit_ready_nodes(executor)

    def invalidate_speculative_cache(
        round_index: int,
        valid_keys: set[tuple[str, str]],
    ) -> None:
        cache = speculative_cache_by_round.setdefault(round_index, {})
        for key in list(cache):
            if key in valid_keys:
                continue
            cached = cache.pop(key)
            node = cached["node"]
            speculative_keys_by_round.setdefault(round_index, set()).discard(key)
            append_event(
                cfg.log_dir,
                "speculative_invalidated",
                round_index=round_index,
                node_id=node.id,
                stage=node.node_status,
                parent_id=node.parent.id if node.parent is not None else None,
                reason="not selected by official round task selection",
            )

    def adopt_speculative_node(
        executor: ThreadPoolExecutor,
        task: RoundTask,
        task_round_index: int,
        item_index: int,
        item_count: int,
    ) -> bool:
        if task.parent is None:
            return False
        key = task_key(task)
        cache = speculative_cache_by_round.setdefault(task_round_index, {})
        cached = cache.pop(key, None)
        if cached is None:
            return False
        speculative_keys_by_round.setdefault(task_round_index, set()).discard(key)
        node = cached["node"]
        original_reason = cached.get("original_selection_reason")
        if original_reason and original_reason != task.selection_reason:
            node.selection_reason = (
                f"{task.selection_reason}\n"
                f"Adopted prefetched candidate originally generated because: "
                f"{original_reason}"
            )
        else:
            node.selection_reason = task.selection_reason
        node.tag = "generated"
        round_node = RoundNode(
            workflow_agent=workflow_agent,
            node=node,
            round_index=task_round_index,
            round_item_index=item_index,
            round_item_count=item_count,
        )
        round_nodes_by_round.setdefault(task_round_index, []).append(round_node)
        scheduled_keys_by_round.setdefault(task_round_index, set()).add(key)
        append_event(
            cfg.log_dir,
            "speculative_adopted",
            round_index=task_round_index,
            round_item_index=item_index,
            round_item_count=item_count,
            node_id=node.id,
            stage=node.node_status,
            parent_id=node.parent.id if node.parent is not None else None,
            official_selection_reason=task.selection_reason,
            original_selection_reason=original_reason,
        )
        pending_execution_queue.append(round_node)
        append_event(
            cfg.log_dir,
            "node_queued_for_execution",
            round_index=task_round_index,
            round_item_index=item_index,
            round_item_count=item_count,
            speculative=False,
            prefetched=True,
            node_id=node.id,
            stage=node.node_status,
            parent_id=node.parent.id if node.parent is not None else None,
        )
        refresh_live_tree(log_tree=False)
        submit_ready_nodes(executor)
        return True

    def handle_scheduler_events(
        executor: ThreadPoolExecutor,
        append_round_index: int | None = None,
        block: bool = False,
    ) -> None:
        futures = [*generation_future_to_meta, *future_to_round_node]
        done = [future for future in futures if future.done()]
        if block and futures and not done:
            wait(futures, return_when=FIRST_COMPLETED)
        handle_finished_generations(executor, block=False)
        handle_finished_futures(block=False, executor=executor)
        submit_ready_nodes(executor)

    def wait_for_scheduler_capacity(
        executor: ThreadPoolExecutor,
        next_task: RoundTask | None = None,
        append_round_index: int | None = None,
    ) -> None:
        while generation_future_to_meta or generation_backlog_full(next_task):
            handle_scheduler_events(
                executor, append_round_index=append_round_index, block=False
            )
            if not generation_future_to_meta and not generation_backlog_full(next_task):
                return
            futures = [*generation_future_to_meta, *future_to_round_node]
            if futures:
                wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
            else:
                time.sleep(0.05)

    def drain_generation(
        executor: ThreadPoolExecutor,
        append_round_index: int | None = None,
    ) -> None:
        while generation_future_to_meta:
            handle_scheduler_events(
                executor, append_round_index=append_round_index, block=True
            )

    def in_flight_child_count(parent: Node) -> int:
        return sum(
            1
            for node in live_nodes()
            if node.parent is parent and node.step is None
        )

    def guaranteed_next_round_tasks(
        current_round_index: int, next_round_index: int
    ) -> list[RoundTask]:
        search_cfg = cfg.agent.search
        current_nodes = round_nodes_by_round.get(current_round_index, [])
        unknown_nodes = [rn.node for rn in current_nodes if rn.node.step is None]
        pending_drafts = sum(1 for node in unknown_nodes if node.parent is None)
        if len(journal.draft_nodes) + pending_drafts < search_cfg.max_children_per_node:
            return []

        unavailable_keys = (
            set(scheduled_keys_by_round.setdefault(next_round_index, set()))
            | active_speculative_keys(next_round_index)
        )
        improve_candidates = [
            node
            for node in proposal_agent.journal.good_nodes
            if (
                node_round_index(node) <= current_round_index
                and node.metric is not None
                and len(node.children) + in_flight_child_count(node)
                < search_cfg.max_children_per_node
            )
        ]
        unknown_candidate_count = len(unknown_nodes)
        guaranteed_improve_count = max(
            0, search_cfg.topk_improve - unknown_candidate_count
        )
        tasks: list[RoundTask] = []
        if guaranteed_improve_count > 0 and improve_candidates:
            ranked = sorted(
                improve_candidates,
                key=lambda node: (
                    node.metric,
                    -(node.step if node.step is not None else 10**9),
                ),
                reverse=True,
            )
            guaranteed_nodes = {node.id for node in ranked[:guaranteed_improve_count]}
            for selected in ranked[: search_cfg.topk_improve]:
                task = RoundTask(
                    parent=selected,
                    selection_reason=(
                        "speculative next-round global frontier improve for node "
                        f"{selected.id[:8]}; cannot be displaced by "
                        f"{unknown_candidate_count} unfinished current-round node(s)"
                    ),
                )
                if (
                    selected.id in guaranteed_nodes
                    and task_key(task) not in unavailable_keys
                ):
                    tasks.append(task)

        debuggable_nodes = [
            node
            for node in proposal_agent.journal.buggy_nodes
            if (
                node_round_index(node) <= current_round_index
                and node.debug_depth < search_cfg.max_debug_depth
                and len(node.children) + in_flight_child_count(node)
                < search_cfg.num_bugs
            )
        ]
        for buggy_node in sorted(
            debuggable_nodes,
            key=lambda node: node.step if node.step is not None else 10**9,
        ):
            task = RoundTask(
                parent=buggy_node,
                selection_reason=(
                    "speculative guaranteed next-round debug for completed buggy node"
                ),
            )
            if task_key(task) not in unavailable_keys:
                tasks.append(task)
        return tasks

    def parse_round_node_if_ready(round_node: RoundNode) -> None:
        if round_node.node.id in parsed_node_ids:
            return
        if round_node.node.id not in raw_exec_results:
            return
        agent.parse_round_result(
            round_node,
            raw_exec_results[round_node.node.id],
            best_node_before=None,
        )
        parsed_node_ids.add(round_node.node.id)
        round_node.node.tag = "completed"
        append_result_for_round_node(round_node)
        refresh_live_tree(log_tree=False)

    def preview_exec_result(round_node: RoundNode, exec_result) -> None:
        node = round_node.node
        if node.id in parsed_node_ids:
            return
        node.absorb_exec_result(exec_result)
        node.artifact_report = parse_artifact_report(exec_result.term_out)
        external_metric = parse_external_validation(exec_result.term_out)
        if node.exc_type is not None or external_metric is None:
            node.is_buggy = True
            node.metric = WorstMetricValue()
            node.outcome = "bug"
        else:
            metric, maximize = external_metric
            node.is_buggy = False
            node.metric = MetricValue(metric, maximize=maximize)
        node.tag = "completed"
        refresh_live_tree(log_tree=False)

    def handle_finished_futures(
        block: bool = False, executor: ThreadPoolExecutor | None = None
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
            round_node.node.tag = "completed"
            preview_exec_result(round_node, exec_result)
            finished_round_nodes.append(round_node)
        if executor is not None:
            submit_ready_nodes(executor)
        for round_node in finished_round_nodes:
            parse_round_node_if_ready(round_node)
            if round_node.node.id not in parsed_node_ids:
                refresh_live_tree(log_tree=False)

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
        sync_node_working_to_run_workspace(cfg, solution_workspace)
        best_node = journal.get_best_node(only_good=True)
        private_metrics = None
        private_error = None
        try:
            private_metrics = evaluate_private_outputs(solution_workspace, private_dir)
        except Exception as exc:
            logger.warning("Private evaluation failed: %s", exc)
            private_error = str(exc)
        if (
            private_metrics
            or private_error
            or (private_dir / "private_split_plan.json").is_file()
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
        tree = journal_to_rich_tree(journal)
        save_run(cfg, journal, tree)
        append_event(
            cfg.log_dir,
            "step_saved",
            round_index=round_node.round_index,
            round_item_index=round_node.round_item_index,
            round_item_count=round_node.round_item_count,
            step=global_step,
            node_id=solution_node.id,
            best_node_id=best_node.id if best_node is not None else None,
        )
        global_step += 1

    def round_ready_to_commit(current_round_index: int) -> bool:
        nodes = round_nodes_by_round.get(current_round_index, [])
        return bool(nodes) and all(
            round_node.node.id in parsed_node_ids for round_node in nodes
        )

    live_tree_thread = Thread(target=live_tree_ticker, daemon=True)
    live_tree_thread.start()
    generation_executor = ThreadPoolExecutor(max_workers=1)
    executor = ThreadPoolExecutor(max_workers=parallel_workers)
    try:
        while global_step < total_steps and not total_timer.is_timeout():
            if generation_future_to_meta:
                drain_generation(executor)
            round_index += 1
            logger.info(
                "BioMaker workflow round %d starting at step %d / %d",
                round_index,
                global_step + 1,
                total_steps,
            )
            try:
                tasks = select_tasks_visible_before_round(round_index)
                existing_keys = scheduled_keys_by_round.setdefault(round_index, set())
                filtered_tasks = []
                for task in tasks:
                    key = task_key(task)
                    if task.parent is not None and key in existing_keys:
                        continue
                    filtered_tasks.append(task)
                budget = max(0, remaining_schedule_budget())
                existing_round_count = len(round_nodes_by_round.get(round_index, []))
                remaining_round_slots = max(0, len(tasks) - existing_round_count)
                filtered_tasks = filtered_tasks[
                    : min(budget, workflow_agent.steps_left, remaining_round_slots)
                ]
                valid_official_keys = {
                    task_key(task) for task in filtered_tasks if task.parent is not None
                }
                invalidate_speculative_cache(round_index, valid_official_keys)
                item_count = len(round_nodes_by_round.get(round_index, [])) + len(
                    filtered_tasks
                )
                append_event(
                    cfg.log_dir,
                    "round_started",
                    round_index=round_index,
                    round_start_step=global_step,
                    planned_items=item_count,
                    parallel_workers=parallel_workers,
                )
                for round_node in list(round_nodes_by_round.get(round_index, [])):
                    parse_round_node_if_ready(round_node)

                for task in filtered_tasks:
                    wait_for_scheduler_capacity(
                        executor, task, append_round_index=round_index
                    )
                    item_index = len(round_nodes_by_round.get(round_index, [])) + 1
                    adopted = adopt_speculative_node(
                        executor,
                        task,
                        round_index,
                        item_index,
                        item_count,
                    )
                    if not adopted:
                        submit_generation_task(
                            generation_executor,
                            task,
                            round_index,
                            item_index,
                            item_count,
                            speculative=False,
                        )
                    handle_scheduler_events(
                        executor, append_round_index=round_index, block=False
                    )
                drain_generation(executor, append_round_index=round_index)

                for round_node in list(round_nodes_by_round.get(round_index, [])):
                    parse_round_node_if_ready(round_node)

                if not round_nodes_by_round.get(round_index):
                    logger.info("No round tasks selected; stopping the run.")
                    break

                while not round_ready_to_commit(
                    round_index
                ) and not total_timer.is_timeout():
                    submit_ready_nodes(executor)
                    next_round_index = round_index + 1
                    guaranteed_tasks = guaranteed_next_round_tasks(
                        round_index, next_round_index
                    )
                    for task in guaranteed_tasks:
                        if (
                            round_ready_to_commit(round_index)
                            or total_timer.is_timeout()
                        ):
                            break
                        wait_for_scheduler_capacity(
                            executor, task, append_round_index=round_index
                        )
                        if (
                            round_ready_to_commit(round_index)
                            or total_timer.is_timeout()
                        ):
                            break
                        item_index = len(active_speculative_keys(next_round_index)) + 1
                        item_count = max(
                            item_index,
                            cfg.agent.search.topk_improve + len(guaranteed_tasks),
                        )
                        submitted = submit_generation_task(
                            generation_executor,
                            task,
                            next_round_index,
                            item_index,
                            item_count,
                            speculative=True,
                        )
                        if not submitted:
                            break
                    handle_scheduler_events(
                        executor, append_round_index=round_index, block=True
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
        live_tree_stop.set()
        live_tree_thread.join(timeout=2)
        with live_tree_lock:
            if live_tree_display_started:
                live_tree_display.stop()
        generation_executor.shutdown(wait=True)
        executor.shutdown(wait=True)
    interpreter.cleanup_session()
    append_event(cfg.log_dir, "run_finished", completed_steps=global_step)
    agent_loop_exit_handler(end_with_exception=end_with_exception)


if __name__ == "__main__":
    run()
