"""Scoring resources and Harness faults are not evidence of broken models."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

import pytest

from biomaker.agents.base_agent import BaseAgent, parse_factual_exec_result
from biomaker.interpreter import ExecutionResult
from biomaker.journal import Journal, Node, _format_execution_summary
from biomaker.utils import evaluator_setup
from biomaker.utils.execution_record import attach_evaluator_run_result


SPEC = {
    "evaluator_api_version": 3,
    "metrics": [{"name": "mae", "maximize": False}],
    "task_type": "regression",
    "key_columns": ["case"],
    "prediction_columns": ["estimate"],
    "answer_columns": ["answer"],
    "validation_artifact": "./working/validation_predictions.csv",
    "test_submission": "./working/submission.csv",
    "test_split": None,
}


@pytest.fixture
def scoring_inputs(tmp_path):
    workspace = tmp_path / "workspace"
    working = workspace / "working"
    working.mkdir(parents=True)
    pred = working / "validation_predictions.csv"
    pred.write_text("case,estimate\na,1\n")
    truth = tmp_path / "truth.csv"
    truth.write_text("case,answer\na,2\n")
    expected = tmp_path / "expected.csv"
    expected.write_text("case\na\n")
    (workspace / "evaluator_prediction_spec.json").write_text(json.dumps(SPEC))
    (working / "model.bin").write_bytes(b"trained model")
    return workspace, pred, truth, expected


def write_evaluator(workspace, body, validation="return {'valid': True}"):
    (workspace / "evaluator.py").write_text(
        "def validate_submission():\n    " + validation + "\n"
        + "def score(*args):\n    " + body + "\n"
    )


def test_actual_scoring_timeout_preserves_model_and_predictions(scoring_inputs):
    workspace, pred, truth, expected = scoring_inputs
    write_evaluator(workspace, "import time; time.sleep(10)")
    result = evaluator_setup.score_artifacts_with_evaluator(
        workspace, pred, truth, expected, timeout=0.1,
    )
    assert not result.ok and result.failure_kind == "timeout"
    assert (workspace / "working/model.bin").read_bytes() == b"trained model"
    assert pred.read_text() == "case,estimate\na,1\n"


@pytest.mark.parametrize("body,kind", [
    ("raise MemoryError('synthetic resource exhaustion')", "resource"),
    ("raise ValueError('unknown internal failure')", "execution"),
    ("return {'metrics': {}}", "protocol"),
])
def test_actual_evaluator_failure_has_no_unproven_model_blame(scoring_inputs, body, kind):
    workspace, pred, truth, expected = scoring_inputs
    write_evaluator(workspace, body)
    result = evaluator_setup.score_artifacts_with_evaluator(workspace, pred, truth, expected, timeout=10)
    assert not result.ok and result.failure_kind == kind


@pytest.mark.parametrize("prediction", [
    None, "case,estimate\na,1\na,2\n", "case,wrong_column\na,1\n",
    "case,estimate\nwrong-case,1\n",
])
def test_proven_prediction_contract_failures_are_debuggable(scoring_inputs, prediction):
    workspace, pred, truth, expected = scoring_inputs
    if prediction is None:
        pred.unlink()
    else:
        pred.write_text(prediction)
    result = evaluator_setup.score_artifacts_with_evaluator(workspace, pred, truth, expected)
    assert not result.ok and result.failure_kind == "prediction_contract"


@pytest.mark.parametrize("answers", [
    "case,answer\na,1\na,2\n", "case,answer\nwrong-case,2\n",
])
def test_bad_private_answers_do_not_implicate_candidate(scoring_inputs, answers):
    workspace, pred, truth, expected = scoring_inputs
    truth.write_text(answers)
    result = evaluator_setup.score_artifacts_with_evaluator(workspace, pred, truth, expected)
    assert not result.ok and result.failure_kind == "protocol"


def test_wrong_prediction_keys_remain_candidate_bugs(scoring_inputs):
    workspace, pred, truth, expected = scoring_inputs
    pred.write_text("case,estimate\nwrong-case,1\n")
    result = evaluator_setup.score_artifacts_with_evaluator(workspace, pred, truth, expected)
    execution = ExecutionResult(["predictions written"], 1, None, candidate_returncode=0)
    attach_evaluator_run_result(
        execution, result, failure_phase="evaluator_scoring",
        failure_type="EvaluatorValidationError", failure_message_prefix="EvaluatorValidationError",
    )
    node = Node(code="pass")
    parse_factual_exec_result(node, execution)
    assert node.is_buggy is True and node.outcome == "bug"


def test_submission_rejection_is_separate_from_explicit_artifact_scoring(scoring_inputs):
    workspace, pred, truth, expected = scoring_inputs
    write_evaluator(workspace, "return {'metrics': {'mae': 1}}", validation="return {'valid': False}")
    result = evaluator_setup.score_artifacts_with_evaluator(workspace, pred, truth, expected, timeout=10)
    assert result.ok and result.metrics == {"mae": 1}
    validation = evaluator_setup.validate_submission_with_evaluator(workspace, timeout=10)
    assert not validation.ok and validation.failure_kind == "prediction_contract"


@pytest.mark.parametrize("policy", ["average_rank", "legacy_primary"])
@pytest.mark.parametrize("kind", ["timeout", "resource", "interrupted", "execution", "protocol"])
def test_unscored_nodes_are_neither_ranked_nor_debugged(policy, kind):
    execution = ExecutionResult(
        term_out=["training and predictions complete\n"], exec_time=1, exc_type=None,
        candidate_returncode=0,
    )
    attach_evaluator_run_result(
        execution,
        evaluator_setup.EvaluatorRunResult(
            False, None, True, {"score": {"metrics": {"mae": 123}, "metric_specs": SPEC["metrics"]}},
            "scoring interrupted", "no complete score", failure_kind=kind,
        ),
        failure_phase="evaluator_scoring", failure_type="EvaluatorValidationError",
        failure_message_prefix="EvaluatorValidationError",
    )
    assert execution.evaluator_result["failure_kind"] == kind
    assert execution.exc_type == "EvaluatorInfrastructureError"
    node = Node(code="pass")
    parse_factual_exec_result(node, execution, record_context=True)
    node.record_experiment_context(best_node_before=None)
    journal = Journal(selection_policy=policy, metric_specs=SPEC["metrics"])
    journal.append(node)
    assert node.outcome == "unscored"
    assert node.is_buggy is None
    assert node.metrics == {} and node.metric is None
    assert journal.buggy_nodes == journal.good_nodes == journal.ranked_nodes() == []
    assert "execution_status: unscored" in _format_execution_summary(node, not_executed=False)
    # Bypass unrelated language-model initialization: exercise the real policy.
    agent = object.__new__(BaseAgent)
    agent.journal = journal
    agent.acfg = SimpleNamespace(search=SimpleNamespace(
        max_children_per_node=1, max_debug_depth=2, topk_improve=1,
    ))
    assert agent.select_round_tasks() == []
    restored = Node.from_json(node.to_json())
    assert restored.outcome == "unscored" and restored.is_buggy is None


def test_real_candidate_failure_still_wins_over_scoring_failure():
    execution = ExecutionResult(
        term_out=["training failed\n"], exec_time=1, exc_type="RuntimeError",
        candidate_returncode=1, candidate_exc_type="RuntimeError",
        evaluator_result={"ok": False, "failure_kind": "timeout"},
    )
    node = Node(code="pass")
    parse_factual_exec_result(node, execution)
    assert node.is_buggy is True and node.outcome == "bug"


def _write_prepared_inputs(workspace):
    split_dir = workspace / "input/biomaker_splits"
    split_dir.mkdir(parents=True)
    (split_dir / "agent_train.csv").write_text("case,answer\ntrain,1\n")
    (split_dir / "score_val.csv").write_text("case\na\n")
    (split_dir / "split_manifest.json").write_text(json.dumps({
        "manifest_version": 2,
        "splits": {"train": {"file": "biomaker_splits/agent_train.csv"},
                   "score_val": {"file": "biomaker_splits/score_val.csv"}},
        "key_columns": ["case"],
        "hidden_answers": {"score_val": {"answer_columns": ["answer"]}},
    }))
    private = workspace / "logs/private_gt"
    private.mkdir(parents=True)
    (private / "score_val.csv").write_text("case,answer\na,2\n")


def test_fixed_preparation_timeout_does_not_regenerate_with_llm(tmp_path, monkeypatch):
    _write_prepared_inputs(tmp_path)
    monkeypatch.setattr(evaluator_setup, "_validate_declared_capabilities", lambda *_a: None)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", 17)
    monkeypatch.setattr(evaluator_setup, "_write_fixed_regression_evaluator", timeout)
    monkeypatch.setattr(evaluator_setup.backend, "query", lambda **kwargs: pytest.fail("must not regenerate"))
    with pytest.raises(evaluator_setup.EvaluatorInfrastructureError) as raised:
        evaluator_setup.prepare_evaluator(tmp_path, tmp_path / "logs", "task", "unused", timeout=17)
    assert raised.value.failure_kind == "timeout"


def test_cached_preparation_timeout_keeps_cache_and_does_not_fall_back(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    evaluator = cache / "evaluator.py"
    evaluator.write_text("# trusted evaluator\n")
    spec = cache / "evaluator_prediction_spec.json"
    spec.write_text(json.dumps(SPEC))
    (cache / "manifest.json").write_text(json.dumps({
        "cache_version": evaluator_setup._EVALUATOR_CACHE_VERSION,
        "evaluator_sha256": evaluator_setup._sha256_file(evaluator),
        "prediction_spec_sha256": evaluator_setup._sha256_file(spec),
    }))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", 17)
    monkeypatch.setattr(evaluator_setup, "_run_self_test", timeout)
    with pytest.raises(evaluator_setup.EvaluatorInfrastructureError):
        evaluator_setup._try_reuse_cached_evaluator(cache, workspace, tmp_path / "logs", 17, tmp_path / "truth.csv")
    assert evaluator.read_text() == "# trusted evaluator\n"
    assert (workspace / "evaluator.py").is_file()


def test_generated_preparation_timeout_does_not_spend_regeneration_attempts(tmp_path, monkeypatch):
    _write_prepared_inputs(tmp_path)
    monkeypatch.setattr(evaluator_setup, "_validate_declared_capabilities", lambda *_a: None)
    for name in ("regression", "classification", "segmentation", "detection", "restoration"):
        monkeypatch.setattr(evaluator_setup, "_write_fixed_" + name + "_evaluator", lambda *_a, **_kw: None)
    monkeypatch.setattr(evaluator_setup, "_standard_adapter_required", lambda *_a: True)
    monkeypatch.setattr(evaluator_setup, "_adapter_prompt", lambda *_a: {})
    queries = []
    def query(**kwargs):
        queries.append(kwargs)
        return "```python\ndef load_sample(*args):\n    return {}\n```"
    monkeypatch.setattr(evaluator_setup.backend, "query", query)
    monkeypatch.setattr(evaluator_setup, "_freeze_data_adapter",
        lambda workspace, code, *_a: (workspace / "evaluator.py").write_text(code))
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", 17)
    monkeypatch.setattr(evaluator_setup, "_run_self_test", timeout)
    with pytest.raises(evaluator_setup.EvaluatorInfrastructureError):
        evaluator_setup.prepare_evaluator(tmp_path, tmp_path / "logs", "task", "unused", timeout=17, attempts=5)
    assert len(queries) == 1
    assert (tmp_path / "evaluator.py").is_file()


@pytest.mark.parametrize("deadline_before_failure", [False, True])
def test_runner_stops_new_work_but_saves_inflight_results_on_scoring_failure(tmp_path, monkeypatch, deadline_before_failure):
    """A scoring fault stops admission and drains both execution and scoring."""
    import biomaker.run as run_module
    from biomaker.utils.compute import ComputePool, ComputeSlot
    from test_presentation_failure_lifecycle import _install_minimal_run, _MinimalWorkflow

    cfg, events, order, _live = _install_minimal_run(
        monkeypatch, tmp_path, retrain_ok=True, break_on_stop=False,
    )
    cfg.agent.steps = 4
    second_started, failure_observed = Event(), Event()
    deadline_reached, release_scoring = Event(), Event()
    score_lock = Lock()
    scoring_workspaces, private_workspaces, saved_nodes = [], [], []

    class Workflow(_MinimalWorkflow):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.workflow[0].steps_left = 4
            self.proposal_agent.select_round_tasks = lambda: [
                run_module.RoundTask(parent=None, selection_reason=f"candidate {i}") for i in range(4)
            ]

        def update_steps_left(self):
            return 4

        def parse_round_result(self, round_node, execution, **_kwargs):
            parse_factual_exec_result(round_node.node, execution, record_context=True)

    class Interpreter:
        def __init__(self, workspace, **kwargs):
            self.workspace = workspace

        def run(self, *_a, **_kw):
            working = self.workspace / "working"
            working.mkdir(parents=True, exist_ok=True)
            (working / "model.bin").write_bytes(b"trained weights")
            (working / "validation_predictions.csv").write_text("case,estimate\na,1\n")
            return ExecutionResult(["training complete"], 0.1, None, candidate_returncode=0)

    def prepare_workspace(_cfg, node):
        workspace = cfg.workspace_dir / "nodes" / node.id
        workspace.mkdir(parents=True)
        node.execution_workspace = str(workspace)
        return workspace

    def score(workspace, **_kwargs):
        with score_lock:
            scoring_workspaces.append(workspace)
            index = len(scoring_workspaces)
        if index == 1:
            assert second_started.wait(5), "second candidate must already be running"
            if deadline_before_failure:
                assert release_scoring.wait(5), "score finishes only during deadline drain"
            return evaluator_setup.EvaluatorRunResult(
                False, None, True, {}, "", "scoring budget exhausted", failure_kind="timeout",
            )
        second_started.set()
        assert failure_observed.wait(5), "running candidate must survive the first scoring failure"
        return evaluator_setup.EvaluatorRunResult(
            True, 0.9, True, {"score": {
                "metrics": {"accuracy": 0.9},
                "metric_specs": [{"name": "accuracy", "maximize": True}],
            }}, "",
        )

    original_event = run_module.append_event
    def append_event(*args, **kwargs):
        original_event(*args, **kwargs)
        if args[1] == "scoring_unavailable":
            failure_observed.set()

    if deadline_before_failure:
        class DeadlineTimer:
            def __init__(self, **kwargs):
                pass

            def reset(self):
                pass

            def is_timeout(self):
                if (second_started.is_set()
                        and len([event for event, _ in events if event == "node_queued_for_execution"]) == 4):
                    deadline_reached.set()
                return deadline_reached.is_set()

        original_wait = run_module.wait
        def wait_after_deadline(*args, **kwargs):
            if deadline_reached.is_set():
                release_scoring.set()
            return original_wait(*args, **kwargs)

        class Executor(ThreadPoolExecutor):
            def shutdown(self, wait=True, **kwargs):
                # Keep a regressed scheduler from deadlocking the test: its
                # shutdown would finish the score but discard the result.
                release_scoring.set()
                return super().shutdown(wait=wait, **kwargs)

        monkeypatch.setattr(run_module, "Timer", DeadlineTimer)
        monkeypatch.setattr(run_module, "wait", wait_after_deadline)

    monkeypatch.setattr(run_module, "append_event", append_event)
    monkeypatch.setattr(run_module, "ThreadPoolExecutor", Executor if deadline_before_failure else ThreadPoolExecutor)
    monkeypatch.setattr(run_module, "BioMakerWorkflow", Workflow)
    monkeypatch.setattr(run_module, "Interpreter", Interpreter)
    monkeypatch.setattr(run_module, "prepare_node_workspace", prepare_workspace)
    monkeypatch.setattr(run_module, "score_candidate_with_evaluator", score)
    monkeypatch.setattr(run_module, "attach_evaluator_run_result", attach_evaluator_run_result)
    monkeypatch.setattr(run_module, "SHARED_CONSOLE", SimpleNamespace(is_terminal=False))
    monkeypatch.setattr(run_module.ComputePool, "detect", staticmethod(lambda _requested: ComputePool((
        ComputeSlot(index=0, backend="cpu"), ComputeSlot(index=1, backend="cpu"),
    ))))
    monkeypatch.setattr(run_module, "evaluate_private_outputs", lambda workspace, *_a, **_kw: private_workspaces.append(workspace))
    def save_run(_cfg, journal, *_a, **_kw):
        saved_nodes[:] = list(journal.nodes)
    monkeypatch.setattr(run_module, "save_run", save_run)

    with pytest.raises(evaluator_setup.EvaluatorInfrastructureError, match="retained for re-scoring"):
        run_module._run_direct_entry()

    assert 2 <= len(scoring_workspaces) <= 4
    assert len(saved_nodes) == len(scoring_workspaces)
    assert sum(node.outcome == "unscored" for node in saved_nodes) == 1
    assert private_workspaces == []  # Search never evaluates private test labels.
    for workspace in scoring_workspaces:
        assert (workspace / "working/model.bin").read_bytes() == b"trained weights"
        assert (workspace / "working/validation_predictions.csv").is_file()
    assert len([event for event, _payload in events if event == "step_saved"]) == len(saved_nodes)
    blocked = next(payload for event, payload in events if event == "run_scoring_blocked")
    assert blocked["completed_steps"] == len(saved_nodes)
    assert len(blocked["unscored_nodes"]) == 1 and len(blocked["unstarted_node_ids"]) == 4 - len(saved_nodes)
    assert [event for event, _payload in events if event.startswith("run_")][-1] == "run_failed"
    assert "run_finished" not in [event for event, _payload in events]
    failure_index = next(i for i, (event, _) in enumerate(events) if event == "scoring_unavailable")
    assert not any(event == "node_submitted" for event, _ in events[failure_index:])
    assert "final_retrain" not in order and "report" not in order
    if deadline_before_failure:
        assert deadline_reached.is_set()
        round_finished = next(i for i, (event, _) in enumerate(events) if event == "round_finished")
        unavailable = next(i for i, (event, _) in enumerate(events) if event == "scoring_unavailable")
        assert round_finished < unavailable, "late timeout must be discovered after the deadline exit"


def test_selected_private_timeout_preserves_validation_and_reports_run_failure(tmp_path, monkeypatch):
    import biomaker.run as run_module
    from biomaker.utils import final_selection
    from test_presentation_failure_lifecycle import _install_minimal_run
    cfg, events, _order, _live = _install_minimal_run(
        monkeypatch, tmp_path, retrain_ok=True, break_on_stop=False)
    cfg.final_retrain.enabled = False
    hidden = tmp_path / "truth.csv"
    hidden.write_text("case,label\na,1\n")
    contract = SimpleNamespace(prediction=SimpleNamespace(raw={}),
        hidden_answers={"test": SimpleNamespace(path=hidden)}, prediction_jobs={"test": object()})
    monkeypatch.setattr(final_selection, "load_experiment_contract", lambda *_a, **_k: contract)
    monkeypatch.setattr(run_module, "freeze_selection", final_selection.freeze_selection)
    monkeypatch.setattr(run_module, "evaluate_selected_delivery", final_selection.evaluate_selected_delivery)
    calls = []
    def fail(*_a, **_k):
        calls.append(True)
        return {"error": "private evaluator timed out", "failure_kind": "timeout", "results": []}
    monkeypatch.setattr(final_selection, "evaluate_private_outputs", fail)
    with pytest.raises(evaluator_setup.EvaluatorInfrastructureError, match="private evaluator timed out"):
        run_module._run_direct_entry()
    assert len(calls) == 1
    receipt = json.loads((cfg.log_dir / "private_metrics.json").read_text())
    assert len(receipt["history"]) == 1
    assert receipt["latest"]["private_failure_kind"] == "timeout"
    assert receipt["latest"]["private_metrics_available"] is False
    saved = [payload for name, payload in events if name == "step_saved"]
    assert saved[0]["best_node_id"] == saved[0]["node_id"]
    assert receipt["validation_selection"]["best_node_id"] == saved[0]["node_id"]
    assert events[-1][0] == "run_failed"


def test_failed_final_retrain_does_not_attempt_private_evaluation(tmp_path, monkeypatch):
    import biomaker.run as run_module
    from test_presentation_failure_lifecycle import _install_minimal_run
    cfg, events, _order, _live = _install_minimal_run(
        monkeypatch, tmp_path, retrain_ok=False, break_on_stop=False)
    monkeypatch.setattr(run_module, "evaluate_selected_delivery", lambda *_a, **_k:
        pytest.fail("Failed retraining cannot enter private evaluation"))
    with pytest.raises(RuntimeError, match="Final retrain did not complete successfully"):
        run_module._run_direct_entry()
    assert not (cfg.log_dir / "private_metrics.json").exists()
    assert events[-1][0] == "run_failed"
