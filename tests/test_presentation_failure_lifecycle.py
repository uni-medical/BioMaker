from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest

import biomaker.run as run_module
from biomaker.interpreter import ExecutionResult
from biomaker.journal import Node
from biomaker.utils import config as config_module
from biomaker.utils import final_retrain as final_retrain_module
from biomaker.utils import tree_export
from biomaker.utils.compute import ComputePool, ComputeSlot
from biomaker.utils.metric import MetricValue


class _ImmediateFuture:
    def __init__(self, function, args, kwargs):
        try:
            self._result = function(*args, **kwargs)
            self._error = None
        except BaseException as exc:
            self._result = None
            self._error = exc

    def done(self):
        return True

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


class _ImmediateExecutor:
    def __init__(self, label, order):
        self.label = label
        self.order = order

    def submit(self, function, *args, **kwargs):
        return _ImmediateFuture(function, args, kwargs)

    def shutdown(self, wait=True):
        assert wait is True
        self.order.append(f"{self.label}_shutdown")


class _LiveThatBreaksOnStop:
    def __init__(self, break_on_stop):
        self.break_on_stop = break_on_stop
        self.start_calls = 0
        self.update_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def update(self, *_args, **_kwargs):
        self.update_calls += 1

    def stop(self):
        self.stop_calls += 1
        if self.break_on_stop:
            raise BrokenPipeError("display reader disappeared")


class _SelectingProposalAgent:
    data_summary = None

    def select_round_tasks(self):
        return [
            run_module.RoundTask(
                parent=None,
                selection_reason="single regression candidate",
            )
        ]


class _GenerationProposalAgent:
    def __init__(self, *args, **kwargs):
        self.data_summary = None

    def set_generation_context(self, **_kwargs):
        return None

    def clear_generation_context(self):
        return None

    def generate_node(self, task):
        return Node(code="print('candidate')", plan=task.selection_reason)


class _MinimalWorkflow:
    def __init__(self, *, journal, **_kwargs):
        self.journal = journal
        self.proposal_agent = _SelectingProposalAgent()
        self.workflow = [
            SimpleNamespace(
                name="proposal",
                steps_left=1,
                agent=self.proposal_agent,
            )
        ]

    @property
    def workflow_status(self):
        return "proposal-steps:1"

    def update_steps_left(self):
        return 1

    def parse_round_result(self, round_node, exec_result, **_kwargs):
        round_node.node.absorb_exec_result(exec_result)
        round_node.node.is_buggy = False
        round_node.node.metric = MetricValue(0.9, maximize=True)
        round_node.node.outcome = "first_success"

    def append_round_node(self, round_node):
        self.workflow[0].steps_left -= 1
        self.journal.append(round_node.node)
        return round_node.node


class _SuccessfulInterpreter:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, *_args, **_kwargs):
        return ExecutionResult(
            term_out=["candidate complete"],
            exec_time=0.1,
            exc_type=None,
        )


class _NoopHandler(logging.Handler):
    def emit(self, _record):
        return None


def _minimal_cfg(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return SimpleNamespace(
        exp_name="broken-pipe-regression",
        log_level="INFO",
        log_dir=log_dir,
        workspace_dir=workspace,
        data_dir=tmp_path / "data",
        resource_dir=tmp_path / "resources",
        generate_report=True,
        report=SimpleNamespace(model="report-model"),
        evaluator_cache=SimpleNamespace(enabled=False),
        split_builder=SimpleNamespace(attempts=1),
        final_retrain=SimpleNamespace(enabled=True),
        agent=SimpleNamespace(
            steps=1,
            total_time_limit=60,
            feedback=SimpleNamespace(model="feedback-model"),
        ),
        exec=SimpleNamespace(
            timeout=30,
            parallel_workers="auto",
            requested_parallel_workers=None,
            effective_parallel_workers=None,
            compute_slots=None,
        ),
    )


def _install_minimal_run(
    monkeypatch,
    tmp_path,
    *,
    retrain_ok: bool,
    break_on_stop: bool,
):
    cfg = _minimal_cfg(tmp_path)
    events = []
    order = []
    live = _LiveThatBreaksOnStop(break_on_stop)
    executor_count = iter(["generation", "execution"])
    compute_pool = ComputePool((ComputeSlot(index=0, backend="cpu"),))

    class Executor(_ImmediateExecutor):
        def __init__(self, **_kwargs):
            super().__init__(next(executor_count), order)

    monkeypatch.setattr(run_module, "load_cfg", lambda: cfg)
    monkeypatch.setattr(
        run_module,
        "append_event",
        lambda _log_dir, event, **payload: events.append((event, payload)),
    )
    monkeypatch.setattr(
        run_module,
        "write_run_manifest",
        lambda *_args, **_kwargs: cfg.log_dir / "run_manifest.json",
    )
    monkeypatch.setattr(run_module, "load_task_desc", lambda _cfg: "minimal task")
    monkeypatch.setattr(run_module, "prep_agent_workspace", lambda _cfg: None)
    monkeypatch.setattr(
        run_module,
        "prepare_split_protocol",
        lambda *_args, **_kwargs: SimpleNamespace(prompt_text="split protocol"),
    )
    monkeypatch.setattr(
        run_module,
        "_run_overview_text",
        lambda *_args, **_kwargs: ("overview", {"summary": "overview"}),
    )
    monkeypatch.setattr(run_module, "_interpreter_kwargs", lambda _cfg: {})
    monkeypatch.setattr(run_module, "Interpreter", _SuccessfulInterpreter)
    monkeypatch.setattr(
        run_module,
        "write_exp_guide",
        lambda *_args, **_kwargs: cfg.workspace_dir / "exp_guide.json",
    )
    monkeypatch.setattr(run_module.shutil, "copy2", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_module,
        "prepare_evaluator",
        lambda **_kwargs: SimpleNamespace(
            prediction_spec='{"primary_metric": "accuracy"}'
        ),
    )
    monkeypatch.setattr(run_module, "BioMakerWorkflow", _MinimalWorkflow)
    monkeypatch.setattr(run_module, "ProposalAgent", _GenerationProposalAgent)
    monkeypatch.setattr(
        run_module,
        "prepare_node_workspace",
        lambda _cfg, node: _set_execution_workspace(node, cfg.workspace_dir),
    )
    monkeypatch.setattr(run_module, "clean_candidate_outputs", lambda *_args: None)
    monkeypatch.setattr(run_module, "normalize_working_outputs", lambda *_args: [])
    monkeypatch.setattr(
        run_module,
        "validate_working_outputs",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        run_module,
        "attach_artifact_validation_result",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        run_module,
        "score_candidate_with_evaluator",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True),
    )
    monkeypatch.setattr(
        run_module,
        "attach_evaluator_run_result",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        run_module, "sync_node_working_to_run_workspace", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        run_module, "evaluate_private_outputs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(run_module, "save_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_module, "journal_to_rich_tree", lambda *_args, **_kwargs: "tree"
    )
    monkeypatch.setattr(
        run_module.tree_export, "generate", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(run_module, "ThreadPoolExecutor", Executor)
    monkeypatch.setattr(run_module, "Live", lambda **_kwargs: live)
    monkeypatch.setattr(run_module, "SHARED_CONSOLE", SimpleNamespace(is_terminal=True))
    monkeypatch.setattr(
        run_module.ComputePool,
        "detect",
        staticmethod(lambda _requested: compute_pool),
    )
    monkeypatch.setattr(
        run_module.OmegaConf,
        "select",
        staticmethod(lambda *_args, **_kwargs: 60),
    )
    monkeypatch.setattr(run_module.atexit, "register", lambda *_args: None)
    monkeypatch.setattr(
        run_module.logging, "FileHandler", lambda *_a, **_k: _NoopHandler()
    )
    monkeypatch.setattr(run_module.logger, "addHandler", lambda *_args: None)
    monkeypatch.setattr(run_module.backend.logger, "addHandler", lambda *_args: None)
    monkeypatch.setattr(run_module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        run_module, "_append_private_metrics_entry", lambda *_args, **_kwargs: None
    )

    def final_retrain(*_args, **_kwargs):
        order.append("final_retrain")
        (cfg.workspace_dir / "final_retrain").mkdir(exist_ok=True)
        return SimpleNamespace(
            ok=retrain_ok,
            workspace_dir=cfg.workspace_dir / "final_retrain",
            train_rows_before=10,
            score_rows_added=2,
            train_rows_after=12,
            private_metrics={"accuracy": 0.9} if retrain_ok else None,
            error=None if retrain_ok else "forced retrain failure",
        )

    monkeypatch.setattr(run_module, "run_final_retrain", final_retrain)

    def report(*_args, **_kwargs):
        order.append("report")
        return "final report"

    monkeypatch.setattr(run_module, "journal2report", report)
    return cfg, events, order, live


def _set_execution_workspace(node, workspace):
    node.execution_workspace = str(workspace)
    return workspace


@pytest.mark.parametrize("retrain_enabled", [False, True])
@pytest.mark.parametrize("invalid_kind", ["exception", "nan", "missing_metric", "fallback", "empty"])
def test_no_valid_candidate_saves_failure_report_and_emits_failed(
    tmp_path, monkeypatch, retrain_enabled, invalid_kind
):
    cfg, events, order, _live = _install_minimal_run(
        monkeypatch, tmp_path, retrain_ok=True, break_on_stop=False,
    )
    cfg.final_retrain.enabled = retrain_enabled

    class InvalidWorkflow(_MinimalWorkflow):
        def parse_round_result(self, round_node, exec_result, **kwargs):
            super().parse_round_result(round_node, exec_result, **kwargs)
            node = round_node.node
            if invalid_kind == "exception":
                node.is_buggy = True
                node.exc_type = "DependencyPolicyError"
            elif invalid_kind == "nan":
                node.metric = MetricValue(float("nan"), maximize=True)
            elif invalid_kind == "missing_metric":
                node.metric = MetricValue(None, maximize=True)
            elif invalid_kind == "fallback":
                node.evaluator_details = {"fallback_used": True}

    monkeypatch.setattr(run_module, "BioMakerWorkflow", InvalidWorkflow)
    if invalid_kind == "empty":
        monkeypatch.setattr(_SelectingProposalAgent, "select_round_tasks", lambda self: [])
    # Use the actual deterministic report renderer and actual terminal wrapper.
    from biomaker.journal2report import journal2report
    monkeypatch.setattr(run_module, "journal2report", journal2report)
    monkeypatch.setattr(
        run_module, "_evaluate_selected_private_outputs",
        lambda *a, **k: pytest.fail("invalid candidates must not reach private scoring"),
    )
    with pytest.raises(RuntimeError, match="No valid scored candidate"):
        run_module.run()
    report = (cfg.log_dir / "report.md").read_text()
    assert "No valid scored candidate is recorded" in report
    assert "Valid scored attempts: **0**" in report
    assert "final_retrain" not in order
    assert order == ["generation_shutdown", "execution_shutdown"]
    names = [event for event, _payload in events]
    assert names.index("report_finished") < names.index("run_failed")
    assert "selection_failed" in names
    assert "selection_finalized" not in names
    assert [n for n in names if n in {"run_finished", "run_failed", "run_interrupted"}] == ["run_failed"]


def test_broken_pipe_during_final_live_stop_does_not_skip_terminal_work(
    tmp_path, monkeypatch
):
    _cfg, events, order, live = _install_minimal_run(
        monkeypatch,
        tmp_path,
        retrain_ok=True,
        break_on_stop=False,
    )
    read_fd, write_fd = os.pipe()
    os.close(read_fd)

    def stop_with_closed_reader():
        live.stop_calls += 1
        os.write(write_fd, b"final display refresh")

    live.stop = stop_with_closed_reader

    try:
        run_module.run()
    finally:
        os.close(write_fd)

    assert live.start_calls == 1
    assert live.update_calls >= 2
    assert live.stop_calls == 1
    assert order == [
        "generation_shutdown",
        "execution_shutdown",
        "final_retrain",
        "report",
    ]
    assert any(
        event == "run_warning"
        and payload.get("component") == "terminal_live_display"
        and payload.get("exc_type") == "BrokenPipeError"
        for event, payload in events
    )
    event_names = [event for event, _payload in events]
    warning_index = next(
        index
        for index, (event, payload) in enumerate(events)
        if event == "run_warning"
        and payload.get("component") == "terminal_live_display"
    )
    assert event_names.index("round_finished") < warning_index
    assert warning_index < event_names.index("final_retrain_started")
    assert any(event == "final_retrain_finished" for event, _payload in events)
    assert any(event == "report_finished" for event, _payload in events)
    terminal_events = [
        event
        for event, _payload in events
        if event in {"run_finished", "run_failed", "run_interrupted"}
    ]
    assert terminal_events == ["run_finished"]
    assert events[-1] == ("run_finished", {"completed_steps": 1})


def test_arbitrary_live_stop_failure_is_only_a_presentation_warning(
    tmp_path, monkeypatch
):
    _cfg, events, order, live = _install_minimal_run(
        monkeypatch,
        tmp_path,
        retrain_ok=True,
        break_on_stop=False,
    )

    def fail_stop():
        live.stop_calls += 1
        raise RuntimeError("renderer cleanup failed")

    live.stop = fail_stop
    run_module.run()

    assert order == [
        "generation_shutdown",
        "execution_shutdown",
        "final_retrain",
        "report",
    ]
    assert any(
        event == "run_warning"
        and payload.get("component") == "terminal_live_display"
        and payload.get("exc_type") == "RuntimeError"
        for event, payload in events
    )
    terminal_events = [
        event
        for event, _payload in events
        if event in {"run_finished", "run_failed", "run_interrupted"}
    ]
    assert terminal_events == ["run_finished"]


def test_live_constructor_failure_does_not_start_or_skip_terminal_work(
    tmp_path, monkeypatch
):
    _cfg, events, order, _live = _install_minimal_run(
        monkeypatch,
        tmp_path,
        retrain_ok=True,
        break_on_stop=False,
    )

    def fail_live(**_kwargs):
        raise BrokenPipeError("display unavailable during construction")

    monkeypatch.setattr(run_module, "Live", fail_live)
    run_module.run()

    assert order == [
        "generation_shutdown",
        "execution_shutdown",
        "final_retrain",
        "report",
    ]
    assert any(
        event == "run_warning"
        and payload.get("component") == "terminal_live_display"
        and payload.get("exc_type") == "BrokenPipeError"
        for event, payload in events
    )
    assert events[-1] == ("run_finished", {"completed_steps": 1})


def test_failed_presentation_warning_sink_does_not_fail_the_run(tmp_path, monkeypatch):
    _cfg, events, order, live = _install_minimal_run(
        monkeypatch,
        tmp_path,
        retrain_ok=True,
        break_on_stop=True,
    )

    def append_event(_log_dir, event, **payload):
        if event == "run_warning":
            raise OSError("warning log unavailable")
        events.append((event, payload))

    monkeypatch.setattr(run_module, "append_event", append_event)
    run_module.run()

    assert live.stop_calls == 1
    assert order == [
        "generation_shutdown",
        "execution_shutdown",
        "final_retrain",
        "report",
    ]
    assert events[-1] == ("run_finished", {"completed_steps": 1})


def test_failed_logger_warning_sink_does_not_fail_the_run(tmp_path, monkeypatch):
    _cfg, events, order, _live = _install_minimal_run(
        monkeypatch,
        tmp_path,
        retrain_ok=True,
        break_on_stop=True,
    )

    monkeypatch.setattr(
        run_module.logger,
        "warning",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("logger handler failed")
        ),
    )

    run_module.run()

    assert order == [
        "generation_shutdown",
        "execution_shutdown",
        "final_retrain",
        "report",
    ]
    assert events[-1] == ("run_finished", {"completed_steps": 1})


def test_final_live_update_failure_still_stops_display_and_finishes(
    tmp_path, monkeypatch
):
    _cfg, events, order, live = _install_minimal_run(
        monkeypatch,
        tmp_path,
        retrain_ok=True,
        break_on_stop=False,
    )

    def fail_final_update(*_args, **_kwargs):
        live.update_calls += 1
        if any(event == "round_finished" for event, _payload in events):
            raise RuntimeError("final display update failed")

    live.update = fail_final_update
    run_module.run()

    assert live.stop_calls == 1
    assert order == [
        "generation_shutdown",
        "execution_shutdown",
        "final_retrain",
        "report",
    ]
    assert events[-1] == ("run_finished", {"completed_steps": 1})


def test_final_retrain_result_sink_does_not_replace_keyboard_interrupt(
    tmp_path, monkeypatch
):
    cfg = SimpleNamespace(
        workspace_dir=tmp_path / "workspace",
        log_dir=tmp_path / "logs",
    )
    cfg.log_dir.mkdir()
    private_dir = tmp_path / "private"
    private_dir.mkdir()

    monkeypatch.setattr(
        final_retrain_module,
        "prepare_final_retrain_workspace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        final_retrain_module,
        "_write_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("result sink unavailable")
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        final_retrain_module.run_final_retrain(
            cfg,
            SimpleNamespace(code=""),
            private_dir,
            timeout=1,
        )


def test_failed_final_retrain_emits_run_failed_without_run_finished(
    tmp_path, monkeypatch
):
    _cfg, events, order, _live = _install_minimal_run(
        monkeypatch,
        tmp_path,
        retrain_ok=False,
        break_on_stop=False,
    )

    with pytest.raises(RuntimeError, match="forced retrain failure"):
        run_module.run()

    assert order == [
        "generation_shutdown",
        "execution_shutdown",
        "final_retrain",
    ]
    retrain_events = [
        payload for event, payload in events if event == "final_retrain_finished"
    ]
    assert len(retrain_events) == 1
    assert retrain_events[0]["ok"] is False
    terminal_events = [
        event
        for event, _payload in events
        if event in {"run_finished", "run_failed", "run_interrupted"}
    ]
    assert terminal_events == ["run_failed"]
    assert not any(event == "report_started" for event, _payload in events)


def test_step_tree_render_failure_is_only_a_presentation_warning(tmp_path, monkeypatch):
    _cfg, events, order, _live = _install_minimal_run(
        monkeypatch,
        tmp_path,
        retrain_ok=True,
        break_on_stop=False,
    )

    monkeypatch.setattr(
        run_module,
        "journal_to_rich_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Rich tree rendering failed")
        ),
    )

    run_module.run()

    assert order == [
        "generation_shutdown",
        "execution_shutdown",
        "final_retrain",
        "report",
    ]
    assert any(
        event == "run_warning"
        and payload.get("component") == "tree_snapshot"
        and payload.get("exc_type") == "RuntimeError"
        for event, payload in events
    )
    assert events[-1] == ("run_finished", {"completed_steps": 1})


def test_save_run_downgrades_tree_html_failure_but_keeps_facts(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    saved = []
    warnings = []

    class EmptyJournal:
        nodes = []

        @staticmethod
        def get_best_node(*, only_good):
            assert only_good is True
            return None

    monkeypatch.setattr(
        config_module.serialize,
        "dump_json",
        lambda _journal, path: saved.append(path.name),
    )
    monkeypatch.setattr(
        config_module,
        "_write_results_table",
        lambda path, _journal: saved.append((path / "results.tsv").name),
    )
    monkeypatch.setattr(
        config_module.OmegaConf,
        "save",
        lambda *, config, f: saved.append(f.name),
    )
    monkeypatch.setattr(
        tree_export,
        "generate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("tree destination unavailable")
        ),
    )
    monkeypatch.setattr(
        config_module,
        "_copy_run_working_dir",
        lambda *_args, **_kwargs: None,
    )

    config_module.save_run(
        SimpleNamespace(log_dir=log_dir, workspace_dir=workspace_dir),
        EmptyJournal(),
        presentation_error_handler=lambda component, exc: warnings.append(
            (component, type(exc).__name__, str(exc))
        ),
    )

    assert saved == ["journal.json", "results.tsv", "config.yaml"]
    assert warnings == [("tree_export", "OSError", "tree destination unavailable")]


@pytest.mark.parametrize("retrain_enabled,retrain_ok", [(False, True), (True, True), (True, False)])
def test_private_evaluation_occurs_once_after_selection_and_optional_retrain(
    tmp_path, monkeypatch, retrain_enabled, retrain_ok
):
    cfg, events, order, _ = _install_minimal_run(
        monkeypatch, tmp_path, retrain_ok=retrain_ok, break_on_stop=False
    )
    cfg.final_retrain.enabled = retrain_enabled
    cfg.agent.steps = 2
    retrain_workspace = cfg.workspace_dir / "final_retrain"
    retrain_workspace.mkdir()
    executed = []
    journals = []
    private_calls = []
    receipts = []

    class TwoCandidateWorkflow(_MinimalWorkflow):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.workflow[0].steps_left = 2
            journals.append(self.journal)

        def update_steps_left(self):
            return 2

        def parse_round_result(self, round_node, exec_result, **kwargs):
            super().parse_round_result(round_node, exec_result, **kwargs)
            # First candidate wins validation; the later workspace must never be scored.
            round_node.node.metric = MetricValue(0.9 if not self.journal.nodes else 0.1, maximize=True)

    def prepare_workspace(_cfg, node):
        workspace = cfg.workspace_dir / "candidates" / node.id
        workspace.mkdir(parents=True)
        executed.append((node.id, workspace))
        return _set_execution_workspace(node, workspace)

    def evaluate(workspace, private_dir, *, env_overrides):
        names = [event for event, _payload in events]
        assert names.count("step_saved") == 2
        assert "selection_finalized" in names
        assert "generation_shutdown" in order
        assert "execution_shutdown" in order
        if retrain_enabled:
            assert "final_retrain_finished" in names
        private_calls.append((workspace, env_overrides))
        return {"results": [{"metric_name": "private_auc", "value": 0.314159265}], "visible_to_agent": False}

    monkeypatch.setattr(run_module, "BioMakerWorkflow", TwoCandidateWorkflow)
    monkeypatch.setattr(run_module, "prepare_node_workspace", prepare_workspace)
    monkeypatch.setattr(run_module, "evaluate_private_outputs", evaluate)
    monkeypatch.setattr(run_module, "_append_private_metrics_entry", lambda _cfg, entry: receipts.append(entry))
    if retrain_enabled and not retrain_ok:
        with pytest.raises(RuntimeError, match="Final retrain did not complete"):
            run_module.run()
        assert private_calls == []
        assert receipts == []
        return

    run_module.run()
    assert len(executed) == 2
    assert len(private_calls) == len(receipts) == 1
    assert private_calls[0][0] == (retrain_workspace if retrain_enabled else executed[0][1])
    assert private_calls[0][0] != executed[-1][1]
    receipt = receipts[0]
    assert receipt["node_id"] == executed[0][0]
    assert receipt["selection_source"] == "validation"
    assert receipt["protocol"] == "post_selection_once"
    assert receipt["stage"] == ("final_retrain" if retrain_enabled else "selected_candidate")
    names = [event for event, _payload in events]
    assert names.index("selection_finalized") < names.index("private_evaluation_started")
    assert names.count("private_evaluation_started") == names.count("private_evaluation_finished") == 1
    assert "0.314159265" not in repr(events)
    assert "0.314159265" not in repr(journals[0])


def test_private_finalizer_uses_final_slot_and_keeps_errors_private(tmp_path, monkeypatch):
    cfg, events, _order, _ = _install_minimal_run(monkeypatch, tmp_path, retrain_ok=True, break_on_stop=False)
    node = Node(code="print('selected')")
    node.metric = MetricValue(0.9, maximize=True)
    node.step = 0
    pool = ComputePool((ComputeSlot(index=0, backend="cuda", selector="GPU-final"),))
    calls = []
    receipts = []

    def evaluate(workspace, private_dir, *, env_overrides):
        calls.append(env_overrides)
        raise ValueError("private score sentinel 0.271828")

    monkeypatch.setattr(run_module, "evaluate_private_outputs", evaluate)
    monkeypatch.setattr(run_module, "_append_private_metrics_entry", lambda _cfg, entry: receipts.append(entry))
    run_module._evaluate_selected_private_outputs(cfg, node, tmp_path / "private_gt", cfg.workspace_dir, pool)
    assert calls == [{"CUDA_VISIBLE_DEVICES": "GPU-final"}]
    assert receipts[0]["status"] == "unavailable"
    assert receipts[0]["private_metrics"] is None
    assert receipts[0]["private_error"] == "private score sentinel 0.271828"
    assert "0.271828" not in repr(events)
    assert events[-1][1]["error_type"] == "ValueError"


def test_private_finalizer_does_not_fall_back_to_shared_workspace(tmp_path, monkeypatch):
    cfg, events, _order, _ = _install_minimal_run(monkeypatch, tmp_path, retrain_ok=True, break_on_stop=False)
    node = Node(code="print('selected')")
    node.metric = MetricValue(0.9, maximize=True)
    node.step = 0
    receipts = []
    monkeypatch.setattr(run_module, "evaluate_private_outputs", lambda *_a, **_k: pytest.fail("No selected workspace must not score shared artifacts"))
    monkeypatch.setattr(run_module, "_append_private_metrics_entry", lambda _cfg, entry: receipts.append(entry))
    pool = ComputePool((ComputeSlot(index=0, backend="cpu"),))
    run_module._evaluate_selected_private_outputs(cfg, node, tmp_path / "private_gt", None, pool)
    assert receipts[0]["status"] == "unavailable"
    assert receipts[0]["artifact_workspace"] is None


@pytest.mark.parametrize("submission_override", [None, "./working/predictions/final.csv"])
def test_private_receipt_hashes_the_same_contract_artifact_it_scores(
    tmp_path, monkeypatch, submission_override
):
    import hashlib
    import json
    from pathlib import Path
    from biomaker.utils.private_evaluation import evaluate_private_outputs

    cfg, _events, _order, _ = _install_minimal_run(
        monkeypatch, tmp_path, retrain_ok=True, break_on_stop=False
    )
    node = Node(code="print('selected')", metric=MetricValue(0.9, maximize=True), step=0)
    spec = {
        "key_columns": ["id"], "prediction_columns": ["prob_0", "prob_1"],
        "class_labels": [0, 1], "primary_metric": "accuracy",
    }
    if submission_override is not None:
        spec["test_submission"] = submission_override
    (cfg.workspace_dir / "evaluator_prediction_spec.json").write_text(json.dumps(spec))
    default_submission = cfg.workspace_dir / "working" / "submission.csv"
    default_submission.parent.mkdir()
    default_submission.write_text("stale file that must not be fingerprinted\n")
    submission = cfg.workspace_dir / Path(submission_override or "working/submission.csv")
    submission.parent.mkdir(parents=True, exist_ok=True)
    submitted_bytes = b"id,prob_0,prob_1\n1,0.9,0.1\n2,0.1,0.9\n"
    submission.write_bytes(submitted_bytes)
    private_dir = tmp_path / "private_gt"
    private_dir.mkdir()
    (private_dir / "private_gt.json").write_text(json.dumps({
        "test": {"file": "test.csv", "answer_columns": ["label"]}
    }))
    (private_dir / "test.csv").write_text("id,label\n1,0\n2,1\n")
    receipts = []
    monkeypatch.setattr(run_module, "evaluate_private_outputs", evaluate_private_outputs)
    monkeypatch.setattr(run_module, "_append_private_metrics_entry", lambda _cfg, entry: receipts.append(entry))
    pool = ComputePool((ComputeSlot(index=0, backend="cpu"),))

    run_module._evaluate_selected_private_outputs(cfg, node, private_dir, cfg.workspace_dir, pool)

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["status"] == "evaluated"
    assert receipt["private_metrics"]["results"][0]["value"] == 1.0
    assert receipt["submission_sha256"] == hashlib.sha256(submitted_bytes).hexdigest()
    if submission_override is not None:
        assert receipt["submission_sha256"] != hashlib.sha256(default_submission.read_bytes()).hexdigest()
