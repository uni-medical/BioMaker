"""Downstream stages consume declared jobs, including absent and broken tests."""

import json
import builtins
import io
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from biomaker.agents.base_agent import BaseAgent
from biomaker.interpreter import ExecutionResult
from biomaker.journal import Node
from biomaker.utils import config, final_retrain, private_evaluation
from biomaker.utils.artifact_validation import normalize_working_outputs, validate_working_outputs
from biomaker.utils.data_summary import data_and_evaluation_summary
from biomaker.utils.exp_guide import build_exp_guide, write_exp_guide
from biomaker.utils.split_contract import SplitContractError


def make_run(tmp_path, *, test=False):
    workspace = tmp_path / "workspace"
    split = workspace / "input/biomaker_splits"
    private = tmp_path / "private"
    split.mkdir(parents=True)
    private.mkdir()
    (workspace / "working").mkdir()
    pd.DataFrame([{"id": "train", "answer": 1.0}]).to_csv(split / "agent_train.csv", index=False)
    pd.DataFrame([{"id": "val"}]).to_csv(split / "score_val.csv", index=False)
    pd.DataFrame([{"id": "val", "answer": 2.0}]).to_csv(private / "score_val.csv", index=False)
    pd.DataFrame([{"id": "val", "estimate": 2.0}]).to_csv(
        workspace / "working/validation_predictions.csv", index=False,
    )
    roles = {"train": "agent_train.csv", "score_val": "score_val.csv"}
    hidden = {"score_val": {"answer_columns": ["answer"]}}
    if test:
        roles["test"] = "agent_test.csv"
        hidden["test"] = {"answer_columns": ["answer"]}
        pd.DataFrame([{"id": "test"}]).to_csv(split / "agent_test.csv", index=False)
        pd.DataFrame([{"id": "test", "answer": 3.0}]).to_csv(private / "test.csv", index=False)
        pd.DataFrame([{"id": "test", "estimate": 3.0}]).to_csv(
            workspace / "working/submission.csv", index=False,
        )
    manifest = {
        "manifest_version": 2,
        "splits": {role: {"file": "biomaker_splits/" + name} for role, name in roles.items()},
        "key_columns": ["id"], "task_heads": ["regression"],
        "hidden_answers": hidden,
        "prediction_contract": {"prediction_columns": ["estimate"]},
        "metric_contract": {"metrics": [{"name": "mae", "maximize": False}]},
    }
    (split / "split_manifest.json").write_text(json.dumps(manifest))
    return workspace, private


def test_no_test_consumers_ignore_raw_test_inputs_and_stale_submission(tmp_path, monkeypatch):
    workspace, private = make_run(tmp_path)
    (workspace / "input/test").mkdir()
    for path in (workspace / "input/test.csv", workspace / "input/test/metadata.csv"):
        path.write_text("id\nunrelated\n")
    stale = workspace / "working/submission.csv"
    stale.write_text("do not parse or normalize this stale file")
    original = stale.read_bytes()
    real_read = pd.read_csv

    def only_declared(path, *args, **kwargs):
        assert Path(path).name not in {"test.csv", "metadata.csv", "submission.csv", "agent_test.csv"}
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", only_declared)
    assert normalize_working_outputs(workspace, "task") == []
    assert validate_working_outputs(workspace, "task").ok
    assert private_evaluation.evaluate_private_outputs(workspace, private) is None
    guide = build_exp_guide(workspace)
    assert set(guide["prediction_jobs"]) == {"score_val"}
    assert guide["prediction"]["test_submission"] is None
    assert "test_submission" not in guide["evaluation_protocol"]
    assert set(guide["splits"]) == {"train", "score_val"}
    write_exp_guide(workspace)
    summary = data_and_evaluation_summary(workspace)
    assert "prediction_jobs" in summary
    assert "submission.csv" not in summary
    assert stale.read_bytes() == original


@pytest.mark.parametrize("missing", ["expected", "artifact", "truth"])
def test_declared_private_test_missing_inputs_are_errors(tmp_path, missing):
    workspace, private = make_run(tmp_path, test=True)
    paths = {
        "expected": workspace / "input/biomaker_splits/agent_test.csv",
        "artifact": workspace / "working/submission.csv",
        "truth": private / "test.csv",
    }
    paths[missing].unlink()
    result = private_evaluation.evaluate_private_outputs(workspace, private)
    assert result["failure_kind"] == "protocol"
    assert result["results"] == []
    if missing != "truth":
        assert not validate_working_outputs(workspace, "task").ok
    if missing == "expected":
        with pytest.raises(SplitContractError, match="Declared test split file is missing"):
            build_exp_guide(workspace)


def test_validation_prediction_is_required_without_test(tmp_path):
    workspace, _ = make_run(tmp_path)
    (workspace / "working/validation_predictions.csv").unlink()
    result = validate_working_outputs(workspace, "task")
    assert not result.ok
    assert "score_val" in result.messages[0]


def test_old_guide_outputs_are_rebuilt_from_contract_without_overwriting_history(tmp_path):
    workspace, _ = make_run(tmp_path)
    guide_path = workspace / "exp_guide.json"
    original = json.dumps({"version": 1, "prediction": {"test_submission": "./working/submission.csv"}})
    guide_path.write_text(original)
    summary = data_and_evaluation_summary(workspace)
    assert "prediction_jobs" in summary
    assert "validation_predictions.csv" in summary
    assert "submission.csv" not in summary
    assert guide_path.read_text() == original


def test_prompt_requires_only_declared_jobs():
    agent = BaseAgent.__new__(BaseAgent)
    agent.cfg = SimpleNamespace(exec=SimpleNamespace(timeout=60))
    agent.acfg = SimpleNamespace(expose_prediction=False)
    text = json.dumps(agent._prompt_impl_guideline)
    assert "prediction_jobs" in text
    assert "./working/submission.csv" not in text


def test_no_test_final_retrain_keeps_model_and_merges_all_rows(tmp_path, monkeypatch):
    workspace, private = make_run(tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    original_train = (workspace / "input/biomaker_splits/agent_train.csv").read_bytes()
    cfg = SimpleNamespace(
        workspace_dir=workspace, log_dir=log_dir,
        final_retrain=SimpleNamespace(enabled=True),
        model_artifacts=SimpleNamespace(required=False),
        exec=SimpleNamespace(format_tb_ipython=False, agent_file_name="runfile.py",
                             enforce_environment=False, include_medical_packages=False),
    )

    class RetrainInterpreter:
        def __init__(self, root, **kwargs):
            self.root = root

        def run(self, code, reset_session):
            train = pd.read_csv(self.root / "input/biomaker_splits/agent_train.csv")
            assert train["id"].tolist() == ["train", "val"]
            assert train["answer"].tolist() == [1.0, 2.0]
            (self.root / "working/model.bin").write_bytes(b"trained model")
            pd.DataFrame([{"id": "val", "estimate": 2.0}]).to_csv(
                self.root / "working/validation_predictions.csv", index=False,
            )
            return ExecutionResult(term_out=["trained"], exec_time=0.1, exc_type=None)

    monkeypatch.setattr(final_retrain, "Interpreter", RetrainInterpreter)
    monkeypatch.setattr(final_retrain, "validate_submission_with_evaluator",
                        lambda *_a, **_kw: pytest.fail("No declared test prediction job"))
    result = final_retrain.run_final_retrain(
        cfg, SimpleNamespace(code="retrain selected model"), private, timeout=60,
    )
    assert result.ok, result.error
    assert (result.train_rows_before, result.score_rows_added, result.train_rows_after) == (1, 1, 2)
    assert result.private_metrics is None
    assert (log_dir / "final_retrain/working/model.bin").read_bytes() == b"trained model"
    assert not (log_dir / "final_submission.csv").exists()
    assert not (result.workspace_dir / "working/submission.csv").exists()
    assert (workspace / "input/biomaker_splits/agent_train.csv").read_bytes() == original_train


@pytest.mark.parametrize("corruption, message", [
    ("missing_key", "Declared sample key columns are missing"),
    ("unordered_rows", "requires declared sample key columns"),
    ("empty_answers", "requires declared score_val answer columns"),
])
def test_canonical_final_retrain_rejects_schema_corruption_without_changing_train(tmp_path, corruption, message):
    workspace, private = make_run(tmp_path)
    manifest_path = workspace / "input/biomaker_splits/split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if corruption == "missing_key":
        manifest["key_columns"] = ["declared_key"]
    elif corruption == "unordered_rows":
        manifest["key_columns"] = []
    else:
        manifest["hidden_answers"]["score_val"]["answer_columns"] = []
    manifest_path.write_text(json.dumps(manifest))
    train = workspace / "input/biomaker_splits/agent_train.csv"
    original = train.read_bytes()
    with pytest.raises(ValueError, match=message):
        final_retrain._write_retrain_training_table(workspace, private)
    assert train.read_bytes() == original


@pytest.mark.parametrize("corruption", [None, "row_count", "public_answer"])
def test_canonical_row_order_final_retrain_uses_positions_and_preserves_train_on_error(tmp_path, corruption):
    workspace, private = make_run(tmp_path)
    split = workspace / "input/biomaker_splits"
    manifest_path = split / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(key_columns=[], row_order_contract=True)
    manifest_path.write_text(json.dumps(manifest))
    train = split / "agent_train.csv"
    pd.DataFrame([{"signal": 10, "answer": 100}]).to_csv(train, index=False)
    public_rows = [{"signal": 20}, {"signal": 30}]
    if corruption == "public_answer":
        public_rows[0]["answer"] = 999
        public_rows[1]["answer"] = 999
    pd.DataFrame(public_rows).to_csv(split / "score_val.csv", index=False)
    truth_rows = [{"answer": 200}, {"answer": 300}]
    if corruption == "row_count":
        truth_rows.pop()
    pd.DataFrame(truth_rows).to_csv(private / "score_val.csv", index=False)
    original = train.read_bytes()
    if corruption is not None:
        message = "equal public and private row counts" if corruption == "row_count" else "hidden answer columns"
        with pytest.raises(ValueError, match=message):
            final_retrain._write_retrain_training_table(workspace, private)
        assert train.read_bytes() == original
    else:
        assert final_retrain._write_retrain_training_table(workspace, private)[:3] == (1, 2, 3)
        merged = pd.read_csv(train)
        assert list(merged) == ["signal", "answer"]
        assert merged.to_dict("records") == [
            {"signal": 10, "answer": 100},
            {"signal": 20, "answer": 200},
            {"signal": 30, "answer": 300},
        ]
        assert train.with_suffix(".search.csv").read_bytes() == original


def test_no_test_export_never_opens_stale_submission_and_keeps_other_outputs(tmp_path, monkeypatch):
    workspace, _ = make_run(tmp_path)
    working = workspace / "working"
    stale = working / "submission.csv"
    stale.write_bytes(b"stale test artifact must not be opened")
    original = stale.read_bytes()
    (working / "model.bin").write_bytes(b"model")
    (working / "progress.csv").write_text("file_path\n/unrelated/file\n")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    cfg = SimpleNamespace(workspace_dir=workspace, log_dir=log_dir,
                          exec=SimpleNamespace(agent_file_name="runfile.py"))
    node = Node(code="print('selected model')")
    node.execution_workspace = str(workspace)
    parsed = []
    monkeypatch.setattr(config, "_copy_csv_referenced_files",
                        lambda path, *_a, **_kw: parsed.append(path))
    monkeypatch.setattr(config, "_copy_submission_artifacts",
                        lambda *_a, **_kw: pytest.fail("No test job may promote submission"))

    def guarded_open(real_open):
        def open_file(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)):
                assert Path(file).name != "submission.csv", "Undeclared submission I/O"
            return real_open(file, *args, **kwargs)
        return open_file

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", guarded_open(builtins.open))
        guard.setattr(io, "open", guarded_open(io.open))
        config.sync_node_working_to_run_workspace(cfg, workspace, target_name="working_best")
        config.sync_node_working_to_run_workspace(cfg, workspace, target_name="working_latest")
        config._copy_node_artifacts(cfg, node)
        config._copy_best_working_outputs(cfg)
        config._copy_run_working_dir(cfg, "working_latest", "working_latest")
        final_retrain._copy_retrain_outputs_to_log(cfg, workspace)

    assert parsed == [working / "validation_predictions.csv"]
    assert stale.read_bytes() == original
    assert not list(log_dir.rglob("submission.csv"))
    for archived in (log_dir / "working_best", log_dir / "working_latest",
                     log_dir / "final_retrain/working"):
        assert (archived / "model.bin").read_bytes() == b"model"
        assert (archived / "progress.csv").is_file()
        assert (archived / "validation_predictions.csv").is_file()


@pytest.mark.parametrize("artifact", ["heldout.csv", "predictions/submission.csv"])
def test_test_export_promotes_the_declared_artifact_path(tmp_path, artifact):
    workspace, _ = make_run(tmp_path, test=True)
    manifest_path = workspace / "input/biomaker_splits/split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prediction_contract"]["test_submission"] = "./working/" + artifact
    manifest_path.write_text(json.dumps(manifest))
    working = workspace / "working"
    (working / artifact).parent.mkdir(parents=True, exist_ok=True)
    (working / artifact).write_text("id,estimate\ntest,3.0\n")
    (working / "submission.csv").write_text("stale unrelated submission")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    cfg = SimpleNamespace(workspace_dir=workspace, log_dir=log_dir)
    node = Node(code="print('selected model')")
    node.execution_workspace = str(workspace)
    config._copy_node_artifacts(cfg, node)
    config.sync_node_working_to_run_workspace(cfg, workspace, target_name="working_best")
    config._copy_best_working_outputs(cfg)
    assert (log_dir / "submission.csv").read_text() == (working / artifact).read_text()
    assert (config._node_artifact_path(log_dir, node) / artifact).read_text() == (working / artifact).read_text()
    assert not (log_dir / "working_best/submission.csv").exists()
