"""Real, tiny CPU reload probes; no provider calls or external datasets."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from biomaker.interpreter import Interpreter
from biomaker.utils.artifact_validation import validate_working_outputs
from biomaker.utils.config import sync_node_working_to_run_workspace
from biomaker.utils.execution_record import attach_artifact_validation_result
from biomaker.utils.model_artifacts import training_calls, validate_model_artifacts


def _manifest(
    workspace,
    *,
    inference=None,
    format_name="joblib",
    model_name="model.joblib",
    inputs=None,
    expected=None,
):
    working = workspace / "working"
    working.mkdir(exist_ok=True)
    (working / "inference.py").write_text(
        inference
        or (
            "import joblib\n"
            "def predict(inputs, artifact_dir):\n"
            "    model = joblib.load(artifact_dir / 'model.joblib')\n"
            "    return model.predict_proba(inputs)\n"
        )
    )
    (working / "smoke_input.json").write_text(
        json.dumps(inputs if inputs is not None else [[-1.0], [1.0]])
    )
    (working / "smoke_expected.json").write_text(
        json.dumps(expected if expected is not None else [[1.0, 0.0], [0.0, 1.0]])
    )
    files = []
    for name in [model_name, "inference.py", "smoke_input.json", "smoke_expected.json"]:
        files.append(
            {
                "path": name,
                "role": "checkpoint" if name == model_name else "asset",
                "format": format_name if name == model_name else None,
                "sha256": hashlib.sha256((working / name).read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "framework": format_name,
        "files": files,
        "inference": {
            "entrypoint": "inference.py",
            "input": "smoke_input.json",
            "expected_output": "smoke_expected.json",
        },
        "metadata": {
            "input_schema": {"features": ["x"]},
            "output_schema": {"shape": ["N", 2]},
            "preprocessing": "No preprocessing",
            "class_labels": [0, 1],
            "dependencies": {"numpy": np.__version__},
        },
    }
    (working / "model_manifest.json").write_text(json.dumps(manifest))
    return manifest


def _sklearn_workspace(workspace):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    (workspace / "working").mkdir(parents=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0)).fit(
        [[-2.0], [-1.0], [1.0], [2.0]], [0, 0, 1, 1]
    )
    joblib.dump(model, workspace / "working/model.joblib")
    _manifest(workspace, expected=model.predict_proba([[-1.0], [1.0]]).tolist())
    return model


def _check(workspace, *, timeout=30):
    interpreter = Interpreter(workspace, timeout=timeout, enforce_environment=False)
    return validate_model_artifacts(
        workspace, candidate_code="model.fit(X, y)", model_probe=interpreter.run
    )


def test_supported_training_detection_and_stateless_compatibility(tmp_path):
    assert (
        training_calls("# clf.fit(X, y)\ntext = 'loss.backward()'\nmodel.eval()") == []
    )
    assert training_calls(
        "import lightgbm as lgb\nlgb.train(data)\nloss.backward()"
    ) == ["backward", "lightgbm.train"]
    assert training_calls(
        "from xgboost import train as fit_booster\nfit_booster(data)"
    ) == ["xgboost.train"]
    assert validate_working_outputs(
        tmp_path, "rule-based processing", candidate_code="print('done')"
    ).ok
    missing = validate_working_outputs(
        tmp_path, "training", candidate_code="model.fit(X,y)"
    )
    assert not missing.ok and "model_manifest.json" in missing.messages[0]
    assert not validate_model_artifacts(tmp_path, required=True).ok


@pytest.mark.parametrize(
    "damage", ["empty", "hash", "symlink", "traversal", "no_checkpoint", "metadata"]
)
def test_rejects_invalid_contract_without_executing_probe(tmp_path, damage):
    _sklearn_workspace(tmp_path)
    working = tmp_path / "working"
    manifest = json.loads((working / "model_manifest.json").read_text())
    if damage == "empty":
        (working / "model.joblib").write_bytes(b"")
    elif damage == "hash":
        (working / "model.joblib").write_bytes(b"placeholder")
    elif damage == "symlink":
        (working / "outside").write_bytes((working / "model.joblib").read_bytes())
        (working / "model.joblib").unlink()
        (working / "model.joblib").symlink_to("outside")
    elif damage == "traversal":
        manifest["files"][0]["path"] = "../outside"
    elif damage == "no_checkpoint":
        manifest["files"][0]["role"] = "asset"
    else:
        del manifest["metadata"]["class_labels"]
    (working / "model_manifest.json").write_text(json.dumps(manifest))

    def forbidden(_code):
        pytest.fail("structurally invalid model must not reach a probe")

    assert not validate_model_artifacts(tmp_path, model_probe=forbidden).ok


def test_fitted_sklearn_pipeline_reloads_and_receipt_attaches(tmp_path):
    _sklearn_workspace(tmp_path)
    result = validate_working_outputs(
        tmp_path,
        "train",
        candidate_code="model.fit(X,y)",
        model_probe=Interpreter(tmp_path, timeout=30, enforce_environment=True).run,
    )
    assert result.ok, result.messages
    assert (
        result.model_artifact["relocated"]
        and result.model_artifact["probe_examples"] == 2
    )
    assert result.model_artifact["detected_training_calls"] == ["fit"]
    from biomaker.interpreter import ExecutionResult

    execution = ExecutionResult([], 1.0, None, None)
    attach_artifact_validation_result(execution, result, failure_reason="invalid")
    assert execution.artifact_validation["model_artifact"]["checkpoint_count"] == 1


def test_synthetic_candidate_uses_allowed_installed_serialization(tmp_path):
    import ast
    import os
    from biomaker.utils.environment import validate_code_environment

    # Extract fixture literals without starting its HTTP server or a workflow.
    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "apps/desktop/tests/engine-workflow-smoke.py"
    )
    tree = ast.parse(fixture_path.read_text())
    code = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CANDIDATE"
            for target in node.targets
        )
    )
    assert validate_code_environment(code).ok
    split = tmp_path / "input/biomaker_splits"
    split.mkdir(parents=True)
    import pandas as pd

    frame = pd.DataFrame(
        {
            "Id": list(range(12)),
            "x0": [-2.0, -1.0, 1.0, 2.0] * 3,
            "x1": [0.0] * 12,
            "label": [0, 0, 1, 1] * 3,
        }
    )
    frame.to_csv(split / "agent_train.csv", index=False)
    for name in ("score_val.csv", "agent_test.csv"):
        frame.drop(columns="label").iloc[:4].to_csv(split / name, index=False)
    # The real workflow installs the inherited network guard. This unit probe
    # only replaces that one fixture guard assertion and makes no network calls.
    code = code.replace(
        "assert os.environ.get('BIOMAKER_QA_GUARD_ACTIVE') == str(os.getpid())",
        "assert True",
    )
    interpreter = Interpreter(tmp_path, timeout=30, enforce_environment=True)
    candidate = interpreter.run(code)
    assert candidate.exc_type is None, candidate.term_out
    result = validate_model_artifacts(
        tmp_path, candidate_code=code, model_probe=interpreter.run
    )
    assert result.ok, result.messages


@pytest.mark.parametrize("kind", ["unfitted", "placeholder"])
def test_nonempty_but_unusable_checkpoint_is_rejected(tmp_path, kind):
    import joblib
    from sklearn.linear_model import LogisticRegression

    (tmp_path / "working").mkdir()
    if kind == "unfitted":
        joblib.dump(LogisticRegression(), tmp_path / "working/model.joblib")
    else:
        (tmp_path / "working/model.joblib").write_bytes(b"this is not a checkpoint")
    _manifest(tmp_path)
    result = _check(tmp_path)
    assert not result.ok and "reload/inference failed" in result.messages[0]


@pytest.mark.parametrize(
    "bad_inference, expected_error",
    [
        (
            "def predict(inputs, artifact_dir):\n    return [[0,0],[0,0]]\n",
            "predictions differ",
        ),
        (
            "import socket\ndef predict(inputs, artifact_dir):\n    socket.create_connection(('example.invalid',443))\n",
            "forbids network",
        ),
        (
            "from sklearn.linear_model import LogisticRegression\ndef predict(inputs, artifact_dir):\n    LogisticRegression().fit([[0],[1]], [0,1])\n",
            "without training",
        ),
        (
            "import subprocess\ndef predict(inputs, artifact_dir):\n    subprocess.run(['echo','not allowed'])\n",
            "child processes",
        ),
    ],
)
def test_probe_rejects_mismatch_network_retraining_and_children(
    tmp_path, bad_inference, expected_error
):
    model = _sklearn_workspace(tmp_path)
    _manifest(
        tmp_path,
        inference=bad_inference,
        expected=model.predict_proba([[-1.0], [1.0]]).tolist(),
    )
    result = _check(tmp_path)
    assert not result.ok and expected_error in result.messages[0], result.messages


def test_probe_cannot_depend_on_original_training_workspace(tmp_path):
    model = _sklearn_workspace(tmp_path)
    (tmp_path / "training_only.txt").write_text("not an inference artifact")
    code = f"def predict(inputs, artifact_dir):\n    open({str(tmp_path / 'training_only.txt')!r}).read()\n"
    _manifest(
        tmp_path, inference=code, expected=model.predict_proba([[-1.0], [1.0]]).tolist()
    )
    result = _check(tmp_path)
    assert not result.ok and "cannot read outside" in result.messages[0]


def test_inference_library_temporary_files_stay_in_the_disposable_copy(tmp_path):
    model = _sklearn_workspace(tmp_path)
    code = """import tempfile
from pathlib import Path
import joblib
def predict(inputs, artifact_dir):
    with tempfile.TemporaryDirectory() as directory:
        assert Path(directory).is_relative_to(artifact_dir)
        temporary = Path(directory) / 'library-capability-probe'
        temporary.write_text('temporary import check')
        assert temporary.read_text() == 'temporary import check'
    model = joblib.load(artifact_dir / 'model.joblib')
    return model.predict_proba(inputs)
"""
    _manifest(
        tmp_path, inference=code, expected=model.predict_proba([[-1.0], [1.0]]).tolist()
    )
    result = _check(tmp_path)
    assert result.ok, result.messages
    assert not (tmp_path / "working/.probe_scratch").exists()


def test_probe_inherits_interpreter_timeout(tmp_path):
    model = _sklearn_workspace(tmp_path)
    _manifest(
        tmp_path,
        inference="def predict(inputs, artifact_dir):\n    while True: pass\n",
        expected=model.predict_proba([[-1.0], [1.0]]).tolist(),
    )
    result = _check(tmp_path, timeout=2)
    assert not result.ok and "TimeoutError" in result.messages[0]


def test_probe_has_separate_script_and_keeps_candidate_result(tmp_path):
    _sklearn_workspace(tmp_path)
    interpreter = Interpreter(
        tmp_path, timeout=120, enforce_environment=False, output_label="original"
    )
    candidate = interpreter.run("print('ORIGINAL CANDIDATE')")
    original = (
        candidate.stdout,
        candidate.candidate_exec_time,
        candidate.resource_usage,
    )
    # A caller-owned file of the configured candidate name is never overwritten
    # by the auxiliary probe, even when an interrupted caller left it behind.
    (tmp_path / "runfile.py").write_text("original source sentinel")
    result = validate_model_artifacts(tmp_path, model_probe=interpreter.run)
    assert result.ok, result.messages
    assert (tmp_path / "runfile.py").read_text() == "original source sentinel"
    assert (
        candidate.stdout,
        candidate.candidate_exec_time,
        candidate.resource_usage,
    ) == original
    assert interpreter.timeout == 120 and interpreter.output_label == "original"
    assert not list(tmp_path.glob(".biomaker-model-probe-*.py"))
    assert result.evidence["probe_exec_time"] > 0
    assert result.evidence["guard_scope"] == "python_audit_and_profile_not_os_sandbox"


@pytest.mark.parametrize(
    "metric, details",
    [
        (float("nan"), {}),
        (float("inf"), {}),
        (0.5, {"fallback_used": True}),
        (0.5, {"valid": False}),
    ],
)
def test_structured_invalid_metrics_cannot_enter_best_selection(metric, details):
    from biomaker.agents.base_agent import parse_factual_exec_result
    from biomaker.interpreter import ExecutionResult
    from biomaker.journal import Node

    node = Node(code="print('candidate')", plan="test")
    result = ExecutionResult(
        [],
        1.0,
        None,
        evaluator_result={"ok": True, "metric": metric, "details": details},
    )
    parse_factual_exec_result(node, result)
    assert node.is_buggy


def test_trained_tiny_cnn_state_dict_reloads_on_cpu(tmp_path):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, 1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
    )
    inputs = torch.randn(2, 1, 3, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1])).backward()
    optimizer.step()
    model.eval()
    expected = model(inputs).softmax(dim=1).detach().tolist()
    (tmp_path / "working").mkdir()
    torch.save(model.state_dict(), tmp_path / "working/model.pt")
    code = """import torch
def predict(inputs, artifact_dir):
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    model.load_state_dict(torch.load(artifact_dir / 'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(inputs)).softmax(dim=1).tolist()
"""
    _manifest(
        tmp_path,
        inference=code,
        format_name="torch_state_dict",
        model_name="model.pt",
        inputs=inputs.tolist(),
        expected=expected,
    )
    result = _check(tmp_path)
    assert result.ok, result.messages


def test_native_lightgbm_reload_and_best_directory_portability(tmp_path):
    import lightgbm as lgb

    x = np.arange(24, dtype=float).reshape(-1, 1)
    y = (x[:, 0] > 12).astype(int)
    model = lgb.train(
        {
            "objective": "binary",
            "verbose": -1,
            "num_threads": 1,
            "min_data_in_leaf": 2,
            "seed": 1,
        },
        lgb.Dataset(x, label=y),
        num_boost_round=3,
    )
    node = tmp_path / "node"
    (node / "working").mkdir(parents=True)
    model.save_model(str(node / "working/model.txt"))
    inputs = [[1.0], [20.0]]
    code = """import lightgbm as lgb
import numpy as np
def predict(inputs, artifact_dir):
    return lgb.Booster(model_file=str(artifact_dir / 'model.txt')).predict(np.asarray(inputs), num_threads=1)
"""
    _manifest(
        node,
        inference=code,
        format_name="lightgbm",
        model_name="model.txt",
        inputs=inputs,
        expected=model.predict(np.asarray(inputs), num_threads=1).tolist(),
    )
    result = _check(node)
    assert result.ok, result.messages
    copied = tmp_path / "run"
    sync_node_working_to_run_workspace(
        SimpleNamespace(workspace_dir=copied), node, target_name="working_best"
    )
    (copied / "working_best").rename(copied / "working")
    import shutil

    shutil.rmtree(node)
    result = _check(copied)
    assert result.ok, result.messages
