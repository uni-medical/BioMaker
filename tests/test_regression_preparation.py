"""Continuous regression prepares and scores through the shared frozen Harness."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from biomaker.utils import evaluator_setup
from biomaker.utils.private_evaluation import evaluate_private_outputs


RSNA_TASK = """# RSNA Bone Age Estimation

Predict pediatric bone age in months from hand X-rays. Provided sex metadata may be used as an additional input.

Use the official training set (`train/`, `train.csv`) for training and internal validation. Reserve the complete official validation set (`validation/`, `validation.csv`) for final held-out evaluation; this is the author's validation set, not the official 200-image test set.

Primary evaluation metric: MAE, in months (lower is better). Compute the mean absolute difference between predicted and reference bone ages, without rounding predictions.
"""
VECTOR_TASK = "Primary evaluation metric: MAE.\nSecondary evaluation metrics: MSE, RMSE."


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def inputs(tmp_path, *, encoding="continuous_scalar", vector=False):
    workspace, private = tmp_path / "workspace", tmp_path / "private_gt"
    split_dir = workspace / "input/biomaker_splits"
    if encoding == "json_vector":
        answers, predictions = ["target_json"], ["estimate_json"]
        validation_values, test_values = [["[0, 0]"]] * 2, [["[0, 0]"]] * 3
    elif vector:
        answers, predictions = ["length_mm", "offset_mm"], ["first_estimate", "second_estimate"]
        validation_values, test_values = [[0, 0]] * 2, [[0, 0]] * 3
    else:
        answers, predictions = ["boneage"], ["predicted_boneage"]
        validation_values, test_values = [[10], [20]], [[30], [40], [50]]
    for name, keys, values in (
        ("score_val", ["0001", "0002"], validation_values),
        ("test", ["test-A", "test-B", "test-C"], test_values),
    ):
        public_name = "agent_test" if name == "test" else name
        write_csv(split_dir / (public_name + ".csv"), [
            {"image_id": key, "hand_image": f"input/images/{key}.png", "sex": "female"}
            for key in keys
        ])
        write_csv(private / (name + ".csv"), [
            {"image_id": key, **dict(zip(answers, value))}
            for key, value in reversed(list(zip(keys, values)))
        ])
    write_csv(split_dir / "agent_train.csv", [
        {"image_id": "train-A", "hand_image": "input/images/train-A.png",
         "sex": "female", **dict(zip(answers, validation_values[0]))},
    ])
    answer_schema = {name: {"answer_columns": answers} for name in ("score_val", "test")}
    write_json(private / "private_gt.json", answer_schema)
    write_json(split_dir / "split_plan.json", {
        "task_type": "regression", "primary_metric": "mae", "private_gt": answer_schema,
    })
    contract = {
        "primary_metric": "mae", "secondary_metrics": ["mse", "rmse"] if vector else [],
        "maximize": False, "units": "millimeters" if vector else "months",
        "rounding": "none", "target_shape": [2] if vector else [],
        "notes": "Preserve continuous predictions and compute absolute error in the declared units.",
    }
    write_json(split_dir / "split_manifest.json", {
        "manifest_version": 1, "key_columns": ["image_id"],
        "task_heads": [{"type": "regression"}],
        "prediction_contract": {
            "prediction_type": encoding, "prediction_columns": predictions,
            "file_reference_columns": [],
        },
        "metric_contract": contract,
    })
    return workspace, private, predictions


def forbid_generation(monkeypatch):
    def unexpected_query(**kwargs):
        pytest.fail("Declared numeric regression must not request generated evaluator/self_test code.")

    monkeypatch.setattr(evaluator_setup.backend, "query", unexpected_query)


def candidate_predictions(workspace, columns, validation_values, test_values):
    for artifact, keys, values in (
        ("validation_predictions", ["0001", "0002"], validation_values),
        ("submission", ["test-A", "test-B", "test-C"], test_values),
    ):
        write_csv(workspace / "working" / (artifact + ".csv"), [
            {"image_id": key, **dict(zip(columns, value)), "ignored_candidate_answer": -999}
            for key, value in reversed(list(zip(keys, values)))
        ])


def score_both(workspace, private):
    validation = evaluator_setup.score_candidate_with_evaluator(
        workspace, hidden_score_path=private / "score_val.csv", timeout=30,
    )
    assert validation.ok, str(validation.error) + validation.output
    test = evaluate_private_outputs(workspace, private)
    assert test is not None and "error" not in test, test
    assert test["visible_to_agent"] is False
    assert validation.details["score"]["n"] == 2
    assert test["results"][0]["n"] == 3
    return validation, test["results"][0]


def test_rsna_months_prepares_without_generation_and_keeps_fractional_predictions(tmp_path, monkeypatch):
    workspace, private, columns = inputs(tmp_path)
    forbid_generation(monkeypatch)
    prepared = evaluator_setup.prepare_evaluator(
        workspace, tmp_path / "logs", RSNA_TASK, model="must-not-be-used",
        attempts=1, timeout=30, hidden_score_path=private / "score_val.csv",
    )
    frozen = prepared.path.read_bytes()
    declaration = json.loads(prepared.prediction_spec)
    assert declaration["task_family"] == declaration["task_type"] == "regression"
    assert declaration["primary_metric"] == "mae"
    assert declaration["maximize"] is False
    assert declaration["units"] == "months"
    assert declaration["rounding"] == "none"
    assert declaration["target_shape"] == []
    assert declaration["prediction_columns"] == ["predicted_boneage"]
    assert declaration["prediction_type"] == "continuous_scalar"
    assert declaration["scoring_backend"] == "trusted_metrics_v1"
    assert "answer_columns" not in declaration
    assert "self_test" not in declaration
    assert not (workspace / "working/.adapter_probe").exists()
    candidate_predictions(workspace, columns, [[10.25], [19.5]], [[30.5], [39.25], [51.5]])
    validation, test = score_both(workspace, private)
    assert validation.metric == pytest.approx(0.375)
    assert validation.maximize is False
    assert validation.details["score"]["metrics"] == {"mae": 0.375}
    assert test["value"] == pytest.approx(2.75 / 3)
    assert test["metric_name"] == "private_mae"
    assert prepared.path.read_bytes() == frozen


def test_fixed_vector_regression_freezes_uniform_reduction_and_all_requested_metrics(tmp_path, monkeypatch):
    workspace, private, columns = inputs(tmp_path, encoding="continuous_vector", vector=True)
    forbid_generation(monkeypatch)
    prepared = evaluator_setup._write_fixed_regression_evaluator(
        workspace, tmp_path / "logs", private / "score_val.csv", 30, VECTOR_TASK,
    )
    assert prepared is not None
    declaration = json.loads(prepared.prediction_spec)
    assert declaration["target_shape"] == [2]
    assert declaration["multioutput"] == "uniform_average"
    assert declaration["secondary_metrics"] == ["mse", "rmse"]
    candidate_predictions(workspace, columns, [[1, 9], [3, 3]], [[1, 1], [1, 1], [1, 1]])
    validation, test = score_both(workspace, private)
    assert validation.details["score"]["metrics"] == {"mae": 4, "mse": 25, "rmse": 5}
    assert test["metrics"] == {"mae": 1, "mse": 1, "rmse": 1}


def test_json_regression_generates_only_decoder_and_reuses_frozen_metrics(tmp_path, monkeypatch):
    workspace, private, columns = inputs(tmp_path, encoding="json_vector", vector=True)
    declaration = {
        "evaluator_api_version": 2, "task_type": "regression", "primary_metric": "mae",
        "secondary_metrics": ["mse", "rmse"], "maximize": False, "units": "millimeters",
        "rounding": "none", "target_shape": [2], "key_columns": ["image_id"],
        "prediction_columns": ["estimate_json"], "answer_columns": ["target_json"],
        "prediction_type": "json_vector", "file_reference_columns": [],
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
    }
    source = "import json\nimport pandas as pd\nfrom pathlib import Path\nADAPTER_SPEC = " + repr(declaration) + '''
def prediction_spec():
    return ADAPTER_SPEC
def load_sample(prediction_row, target_row, expected_row):
    sample = {'prediction': json.loads(prediction_row['estimate_json'])}
    if target_row is not None:
        sample['target'] = json.loads(target_row['target_json'])
    return sample
def write_dummy_predictions(expected_rows, artifact_path):
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'image_id': row['image_id'], 'estimate_json': '[0.0, 0.0]'}
                  for row in expected_rows]).to_csv(artifact_path, index=False)
'''
    prompts = []

    def query(**kwargs):
        prompts.append(copy.deepcopy(kwargs["system_message"]))
        assert len(prompts) == 1, "A correct thin JSON decoder should pass in one attempt."
        return "```python\n" + source + "\n```"

    monkeypatch.setattr(evaluator_setup.backend, "query", query)
    prepared = evaluator_setup.prepare_evaluator(
        workspace, tmp_path / "logs", VECTOR_TASK, model="mock-regression-adapter",
        attempts=1, timeout=30, hidden_score_path=private / "score_val.csv",
    )
    assert len(prompts) == 1
    assert "Data adapter API" in prompts[0]
    assert "self_test" not in source and "def score" not in source
    public_spec = json.loads(prepared.prediction_spec)
    assert public_spec["prediction_type"] == "json_vector"
    assert public_spec["scoring_backend"] == "trusted_metrics_v1"
    assert "answer_columns" not in public_spec
    candidate_predictions(workspace, columns, [["[1, 9]"], ["[3, 3]"]], [["[1, 1]"]] * 3)
    validation, test = score_both(workspace, private)
    assert validation.details["score"]["metrics"] == {"mae": 4, "mse": 25, "rmse": 5}
    assert test["metrics"] == {"mae": 1, "mse": 1, "rmse": 1}
