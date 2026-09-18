"""Explicit train/validation-only lifecycle for trusted standard evaluators."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biomaker.utils import evaluator_setup
from biomaker.utils.evaluator_adapter import compile_adapter


SPEC = {
    "evaluator_api_version": 3,
    "metrics": [{"name": "mae", "maximize": False}],
    "task_type": "regression",
    "key_columns": ["case"],
    "prediction_columns": ["estimate"],
    "answer_columns": ["answer"],
    "validation_artifact": "./working/validation_predictions.csv",
    "test_submission": "./working/submission.csv",
}
TASK = '```metrics\n[{"name":"mae","maximize":false}]\n```'


def source(spec):
    return 'import pandas as pd\nfrom pathlib import Path\nADAPTER_SPEC = ' + repr(spec) + '''
def prediction_spec():
    return ADAPTER_SPEC

def load_sample(prediction_row, target_row, expected_row):
    sample = {'prediction': float(prediction_row['estimate'])}
    if target_row is not None:
        sample['target'] = float(target_row['answer'])
    return sample

def write_dummy_predictions(expected_rows, artifact_path):
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'case': row['case'], 'estimate': 0.0} for row in expected_rows]).to_csv(artifact_path, index=False)
'''


def workspace_inputs(tmp_path, *, declaration=None, test=False):
    workspace = tmp_path / "workspace"
    split_dir = workspace / "input/biomaker_splits"
    split_dir.mkdir(parents=True)
    private = tmp_path / "private"
    private.mkdir()
    pd.DataFrame([{"case": "train", "answer": 1.0}]).to_csv(split_dir / "agent_train.csv", index=False)
    pd.DataFrame([{"case": "validation"}]).to_csv(split_dir / "score_val.csv", index=False)
    pd.DataFrame([{"case": "validation", "answer": 2.0}]).to_csv(private / "score_val.csv", index=False)
    if test:
        pd.DataFrame([{"case": "unseen"}]).to_csv(split_dir / "agent_test.csv", index=False)
    manifest = {
        "key_columns": ["case"],
        "task_heads": [{"type": "regression"}],
        "prediction_contract": {"prediction_columns": ["estimate"], "prediction_type": "numeric_columns"},
        "metric_contract": {"metrics": SPEC["metrics"]},
        "hidden_answers": {"score_val": {"answer_columns": ["answer"]}},
    }
    manifest.update(declaration or {})
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest))
    (split_dir / "split_plan.json").write_text(json.dumps({
        "hidden_answers": {"score_val": {"answer_columns": ["answer"]}},
    }))
    return workspace, private / "score_val.csv"


@pytest.mark.parametrize("field", ["public_splits", "required_splits"])
def test_explicit_no_test_adapter_scores_without_test_files(tmp_path, monkeypatch, field):
    workspace, truth = workspace_inputs(tmp_path, declaration={field: ["train", "score_val"]})
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("BIOMAKER_SCORE_VAL_PATH", str(truth))
    frozen = evaluator_setup._bind_public_split_lifecycle(SPEC, workspace)
    namespace = {}
    exec(compile_adapter(source(SPEC), frozen), namespace)
    real_read = pd.read_csv

    def forbid_test_read(path, *args, **kwargs):
        assert Path(path).name not in {"agent_test.csv", "test.csv", "submission.csv"}
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", forbid_test_read)
    try:
        result = namespace["self_test"]()
        assert result["score"]["metrics"] == {"mae": 2.0}
        assert namespace["validate_submission"]() == {
            "valid": True, "skipped": True, "reason": "no_test_split", "n": 0,
        }
        assert namespace["prediction_spec"]()["test_split"] is None
        assert not (workspace / "working/.adapter_probe/submission").exists()
        assert not (workspace / "working/submission.csv").exists()
        assert not (workspace / "input/biomaker_splits/agent_test.csv").exists()
    finally:
        namespace["cleanup_self_test"]()


def test_missing_test_without_explicit_contract_is_not_silently_skipped(tmp_path, monkeypatch):
    workspace, truth = workspace_inputs(tmp_path)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("BIOMAKER_SCORE_VAL_PATH", str(truth))
    assert evaluator_setup._load_fixed_split_context(workspace, truth) is None
    bound = evaluator_setup._bind_public_split_lifecycle({**SPEC, "test_split": None,
                                                       "public_splits": ["train", "score_val"]}, workspace)
    assert bound["test_split"] == "./input/biomaker_splits/agent_test.csv"
    assert "public_splits" not in bound
    namespace = {}
    exec(compile_adapter(source(SPEC), bound), namespace)
    try:
        with pytest.raises(FileNotFoundError, match="agent_test.csv"):
            namespace["self_test"]()
    finally:
        namespace["cleanup_self_test"]()


def test_null_test_path_requires_frozen_explicit_split_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="explicit train/validation-only"):
        exec(compile_adapter(source(SPEC), {**SPEC, "test_split": None}), {})


@pytest.mark.parametrize("declaration", [
    {"required_splits": ["train", "score_val"], "public_splits": ["train", "score_val", "test"]},
    {"required_splits": ["train", "score_val"], "hidden_answers": {"test": {}}},
    {"required_splits": ["train", "score_val"], "splits": {"test": {}}},
    {"required_splits": "train,score_val"},
    {"public_splits": ["score_val"]},
])
def test_inconsistent_no_test_declarations_are_rejected(tmp_path, declaration):
    workspace, _ = workspace_inputs(tmp_path, declaration=declaration)
    with pytest.raises(evaluator_setup.EvaluatorSetupError):
        evaluator_setup._bind_public_split_lifecycle(SPEC, workspace)


def test_existing_test_lifecycle_is_preserved(tmp_path, monkeypatch):
    workspace, truth = workspace_inputs(tmp_path, test=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("BIOMAKER_SCORE_VAL_PATH", str(truth))
    frozen = evaluator_setup._bind_public_split_lifecycle(SPEC, workspace)
    namespace = {}
    exec(compile_adapter(source(SPEC), frozen), namespace)
    try:
        assert namespace["self_test"]()["score"]["metrics"] == {"mae": 2.0}
        assert (workspace / "working/.adapter_probe/submission/predictions.csv").is_file()
        assert namespace["prediction_spec"]()["test_split"].endswith("agent_test.csv")
    finally:
        namespace["cleanup_self_test"]()


def test_generated_adapter_freezes_no_test_from_manifest(tmp_path):
    workspace, truth = workspace_inputs(tmp_path, declaration={"required_splits": ["train", "score_val"]})
    evaluator_setup._freeze_data_adapter(workspace, source(SPEC), TASK, truth)
    result = evaluator_setup._run_python(
        workspace,
        "import evaluator, json\nprint(json.dumps(evaluator.prediction_spec()))\nprint(json.dumps(evaluator.validate_submission()))\n",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    frozen, validation = map(json.loads, result.stdout.splitlines())
    assert frozen["test_split"] is None
    assert validation["skipped"] is True


def test_builtin_standard_reader_prepares_without_llm_or_test(tmp_path, monkeypatch):
    workspace, truth = workspace_inputs(tmp_path, declaration={"required_splits": ["train", "score_val"]})

    def unexpected_llm(**kwargs):
        pytest.fail("Built-in no-test regression reader should not need an LLM")

    monkeypatch.setattr(evaluator_setup.backend, "query", unexpected_llm)
    context = evaluator_setup._load_fixed_split_context(workspace, truth)
    assert context is not None and context["test_path"] is None and context["test_df"].empty
    prepared = evaluator_setup.prepare_evaluator(
        workspace, tmp_path / "logs", TASK, model="unused", attempts=1,
        hidden_score_path=truth, timeout=30,
    )
    frozen = json.loads(prepared.prediction_spec)
    assert frozen["adapter_api_version"] == 1
    assert frozen["test_split"] is None
    assert frozen["public_splits"] == ["train", "score_val"]
    assert not (workspace / "input/biomaker_splits/agent_test.csv").exists()
    assert not (workspace / "working/submission.csv").exists()


def test_builtin_physical_multi_metric_segmentation_with_private_tree_references_no_test(tmp_path, monkeypatch):
    """Exercise fixed inference, four private paths, frozen geometry, and no test."""
    import SimpleITK as sitk

    workspace, private = tmp_path / "workspace", tmp_path / "private"
    split_dir = workspace / "input/biomaker_splits"
    split_dir.mkdir(parents=True)
    private.mkdir()
    names = ["branch_accuracy", "branch_f1", "subtree_consistency", "topological_distance",
             "multiclass_dice", "multiclass_cldice"]
    definitions = [{"name": name, "maximize": name != "topological_distance"} for name in names]
    parameters = {
        "task_family": "segmentation", "dice_smooth": 1e-5, "topology_smooth": 1e-5,
        "connectivity": 3, "topology_component_policy": "largest", "branch_coverage_threshold": 0.8,
        "geometry_policy": "physical", "reference_protocol": "provided",
        "image_column": "image_path", "target_column": "label_path",
        "auxiliary_columns": {"skeleton": "skeleton_path", "parsing": "parsing_path", "graph": "graph_path"},
        "label_ids": list(range(21)), "subtree_ignore_labels": [19], "missing_label_penalty": 50,
        "class_aggregation": "gt_present",
    }
    contract = {"metrics": definitions, **parameters}
    task = "```metrics\n" + json.dumps(contract) + "\n```"
    target = np.zeros((5, 5, 10), dtype=np.uint8)
    target[2, 2, 1:4] = 19
    target[2, 2, 4:9] = 20
    parsing = np.zeros_like(target)
    parsing[2, 2, 1:4] = 1
    parsing[2, 2, 4:9] = 2

    def write_volume(path, array):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.7, 0.8, 1.3))
        image.SetOrigin((11.0, 12.0, 13.0))
        image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(image, str(path))

    public_rows, truth_rows = [], []
    for case in ("train", "validation-a", "validation-b"):
        image_path = f"input/images/{case}.mha"
        write_volume(workspace / image_path, np.zeros(target.shape, dtype=np.int16))
        references = workspace / "input/train_references" if case == "train" else private
        row = {"case": case}
        for field, array in (("label", target), ("skeleton", (target > 0).astype(np.uint8)), ("parsing", parsing)):
            path = references / f"{case}_{field}.mha"
            write_volume(path, array)
            row[field + "_path"] = str(path)
        graph = references / f"{case}_graph.npy"
        np.save(graph, np.array([[0], [1]], dtype=np.int64))
        row["graph_path"] = str(graph)
        public = {"case": case, "image_path": image_path}
        if case == "train":
            pd.DataFrame([{**public, **row}]).to_csv(split_dir / "agent_train.csv", index=False)
        else:
            public_rows.append(public)
            truth_rows.append(row)
    pd.DataFrame(public_rows).to_csv(split_dir / "score_val.csv", index=False)
    truth = private / "score_val.csv"
    pd.DataFrame(truth_rows[::-1]).to_csv(truth, index=False)
    answers = ["label_path", "skeleton_path", "parsing_path", "graph_path"]
    hidden = {"score_val": {"answer_columns": answers, "file_reference_columns": answers}}
    manifest = {
        "manifest_version": 1, "required_splits": ["train", "score_val"],
        "key_columns": ["case"], "task_heads": [{"type": "segmentation_3d"}],
        "input_file_reference_columns": ["image_path"], "hidden_answers": hidden,
        "prediction_contract": {"prediction_columns": ["prediction_mask"],
                                "file_reference_columns": ["prediction_mask"],
                                "prediction_type": "mask_file_path"},
        "metric_contract": contract,
    }
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest))
    (split_dir / "split_plan.json").write_text(json.dumps({"hidden_answers": hidden}))

    def unexpected_llm(**kwargs):
        pytest.fail("Physical multi-metric tree segmentation should use the fixed reader without an LLM")

    monkeypatch.setattr(evaluator_setup.backend, "query", unexpected_llm)
    run_python = evaluator_setup._run_python
    # Reject reads AND writes of test fixtures in real evaluator subprocesses.
    # This catches a create-then-clean-up dummy test that existence checks miss.
    guard = '''
import os as _no_test_os, sys as _no_test_sys
from pathlib import Path as _NoTestPath
def _no_test_audit(event, arguments):
    if event == 'open' and isinstance(arguments[0], (str, bytes, _no_test_os.PathLike)):
        path = _NoTestPath(_no_test_os.fsdecode(arguments[0]))
        if path.name in {'agent_test.csv', 'test.csv', 'submission.csv'} or ('.adapter_probe' in path.parts and 'submission' in path.parts):
            raise AssertionError('No-test evaluator attempted to open a test artifact: ' + str(path))
_no_test_sys.addaudithook(_no_test_audit)
'''

    def guarded_python(workspace_dir, code, timeout=120, env=None):
        return run_python(workspace_dir, guard + "\n" + code, timeout=timeout, env=env)

    monkeypatch.setattr(evaluator_setup, "_run_python", guarded_python)
    prepared = evaluator_setup.prepare_evaluator(
        workspace, tmp_path / "logs", task, model="unused", attempts=1,
        hidden_score_path=truth, timeout=30,
    )
    frozen = json.loads(prepared.prediction_spec)
    assert frozen["adapter_api_version"] == 1 and frozen["scoring_backend"] == "trusted_metrics_v1"
    assert frozen["metrics"] == definitions
    for name, expected in parameters.items():
        assert frozen[name] == expected, name
    assert frozen["test_split"] is None and frozen["public_splits"] == ["train", "score_val"]
    assert "answer_columns" not in frozen

    predictions = []
    for row in public_rows:
        prediction = f"working/{row['case']}.mha"
        write_volume(workspace / prediction, target)
        predictions.append({"case": row["case"], "prediction_mask": prediction})
    pd.DataFrame(predictions[::-1]).to_csv(workspace / "working/validation_predictions.csv", index=False)
    result = evaluator_setup.score_candidate_with_evaluator(workspace, hidden_score_path=truth, timeout=30)
    assert result.ok, result.output + str(result.error)
    assert set(result.metrics) == set(names)
    assert all(np.isfinite(value) for value in result.metrics.values())
    assert result.metrics == {"branch_accuracy": 1.0, "branch_f1": 1.0, "subtree_consistency": 0.0,
                              "topological_distance": 0.0, "multiclass_dice": 1.0, "multiclass_cldice": 1.0}
    assert result.details["score"]["n"] == len(public_rows) == 2
    assert not (split_dir / "agent_test.csv").exists()
    assert not (workspace / "working/submission.csv").exists()
    assert not (workspace / "working/.adapter_probe").exists()
