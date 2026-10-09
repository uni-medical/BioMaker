from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from biomaker.utils import evaluator_setup as evaluator
from biomaker.utils import private_evaluation as private
from biomaker.utils.evaluator_templates import _FIXED_CLASSIFICATION_EVALUATOR
from biomaker.utils.metric_registry import (
    is_classification_metric_supported,
    is_segmentation_metric_supported,
    metric_info,
    metric_maximize,
)


def _classification_workspace(tmp_path, *, metric="auc"):
    workspace = tmp_path / "workspace"
    split_dir = workspace / "input" / "biomaker_splits"
    split_dir.mkdir(parents=True)
    working = workspace / "working"
    working.mkdir()
    hidden = tmp_path / "private_gt" / "score_val.csv"
    hidden.parent.mkdir()
    spec = {
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
        "score_split": "./input/biomaker_splits/score_val.csv",
        "key_columns": ["id", "split"],
        "prediction_columns": ["prob_0", "prob_1"],
        "class_labels": [0, 1], "answer_columns": ["label"],
        "task_type": "binary", "positive_class": 1,
        "primary_metric": metric, "maximize": True,
    }
    for name, split in [("score_val", "validation"), ("agent_test", "test")]:
        frame = pd.DataFrame({"id": [1, 2, 3, 4], "split": [split] * 4})
        frame.to_csv(split_dir / f"{name}.csv", index=False)
        frame.assign(prob_0=[0.9, 0.1, 0.8, 0.2], prob_1=[0.1, 0.9, 0.2, 0.8]).to_csv(
            working / ("validation_predictions.csv" if name == "score_val" else "submission.csv"), index=False
        )
    pd.DataFrame({"id": [1, 2, 3, 4], "split": ["validation"] * 4, "label": [0, 1, 0, 1]}).to_csv(hidden, index=False)
    (workspace / "evaluator_prediction_spec.json").write_text(json.dumps(spec))
    (workspace / "evaluator.py").write_text(_FIXED_CLASSIFICATION_EVALUATOR.replace("__BIOMAKER_SPEC__", repr(spec)))
    return workspace, hidden, spec


def _contract_check(workspace, hidden, spec):
    evaluator._run_public_artifact_contract_test(workspace, json.dumps(spec), 30, hidden)


def test_trusted_contract_accepts_strict_evaluator_and_restores_artifacts(tmp_path):
    workspace, hidden, spec = _classification_workspace(tmp_path)
    originals = {path.name: path.read_bytes() for path in (workspace / "working").iterdir()}
    truth = hidden.read_bytes()
    _contract_check(workspace, hidden, spec)
    assert originals == {path.name: path.read_bytes() for path in (workspace / "working").iterdir()}
    assert hidden.read_bytes() == truth
    assert not list(hidden.parent.glob(".metric-contract-*"))


@pytest.mark.parametrize("repair_on_disk", [False, True])
def test_trusted_contract_rejects_evaluator_reconstructing_split(tmp_path, repair_on_disk):
    workspace, hidden, spec = _classification_workspace(tmp_path)
    evaluator_path = workspace / "evaluator.py"
    source = evaluator_path.read_text()
    repair = "    if 'split' not in pred.columns:\n        pred['split'] = 'test' if 'submission' in path else 'validation'\n"
    if repair_on_disk:
        repair += "        pred.to_csv(_path(path), index=False)\n"
    source = source.replace("    pred = _read_csv(path)\n", "    pred = _read_csv(path)\n" + repair)
    evaluator_path.write_text(source)
    before = (workspace / "working" / "submission.csv").read_bytes()
    with pytest.raises(evaluator.EvaluatorSetupError, match="missing_column_split"):
        _contract_check(workspace, hidden, spec)
    assert (workspace / "working" / "submission.csv").read_bytes() == before


def test_trusted_contract_rejects_unmarked_chance_auc_fallback(tmp_path):
    workspace, hidden, spec = _classification_workspace(tmp_path)
    evaluator_path = workspace / "evaluator.py"
    evaluator_path.write_text(evaluator_path.read_text() + "\n_original_score = score_val\ndef score_val():\n    try:\n        return _original_score()\n    except RuntimeError as error:\n        if 'undefined' in str(error):\n            return {'metric': 0.5, 'maximize': True}\n        raise\n")
    with pytest.raises(evaluator.EvaluatorSetupError, match="undefined_primary_auc"):
        _contract_check(workspace, hidden, spec)


@pytest.mark.parametrize("score", [
    {"metric": float("nan")}, {"metric": float("inf")}, {"metric": -float("inf")},
    {"metric": "0.8"}, {"metric": True}, {"metric": 0.5, "fallback": True},
    {"metric": 0.5, "details": {"used_fallback": True}},
    {"metric": 0.5, "valid": False}, {"metric": 0.5, "error": "undefined"},
])
def test_runtime_rejects_nonfinite_invalid_or_fallback_results(tmp_path, monkeypatch, score):
    payload = {"score": score, "submission_validation": {"valid": True}}
    monkeypatch.setattr(evaluator, "_run_python", lambda *_a, **_k: SimpleNamespace(
        returncode=0, stdout=evaluator._RESULT_PREFIX + json.dumps(payload), stderr=""
    ))
    result = evaluator.score_candidate_with_evaluator(tmp_path)
    assert not result.ok
    assert result.metric is None
    assert evaluator.parse_external_validation(evaluator.EXTERNAL_VALIDATION_PREFIX + json.dumps(score)) is None


@pytest.mark.parametrize("validation", [None, False, {}, {"valid": False}, {"valid": False, "ok": True}, {"valid": True, "fallback_used": True}])
def test_frozen_runtime_requires_explicit_submission_success(tmp_path, monkeypatch, validation):
    (tmp_path / "evaluator_prediction_spec.json").write_text("{}")
    payload = {"score": {"metric": 0.8}, "submission_validation": validation}
    monkeypatch.setattr(evaluator, "_run_python", lambda *_a, **_k: SimpleNamespace(
        returncode=0, stdout=evaluator._RESULT_PREFIX + json.dumps(payload), stderr=""
    ))
    assert not evaluator.score_candidate_with_evaluator(tmp_path).ok
    assert not evaluator.validate_submission_with_evaluator(tmp_path).ok


@pytest.mark.parametrize("same_bytes", [False, True])
def test_runtime_rejects_validator_modifying_submission(tmp_path, same_bytes):
    workspace, hidden, _ = _classification_workspace(tmp_path)
    path = workspace / "evaluator.py"
    path.write_text(path.read_text() + "\n_original_validate = validate_submission\ndef validate_submission():\n    from pathlib import Path\n    path = Path('working/submission.csv')\n    data = path.read_bytes()\n    path.write_bytes(data" + ("" if same_bytes else " + b'\\n'") + ")\n    return _original_validate()\n")
    result = evaluator.validate_submission_with_evaluator(workspace)
    assert not result.ok
    assert "modified" in result.error
    result = evaluator.score_candidate_with_evaluator(workspace, hidden_score_path=hidden)
    assert not result.ok
    assert "modified" in result.error


def _private_workspace(tmp_path, *, labels=(0, 1, 0, 1), metric="auc"):
    workspace, hidden, spec = _classification_workspace(tmp_path, metric=metric)
    (workspace / "evaluator.py").unlink()
    private_dir = hidden.parent
    (private_dir / "private_gt.json").write_text(json.dumps({"test": {"file": "test.csv", "answer_columns": ["label"]}}))
    pd.DataFrame({"id": [1, 2, 3, 4], "split": ["test"] * 4, "label": labels}).to_csv(private_dir / "test.csv", index=False)
    return workspace, private_dir


def test_private_auc_does_not_substitute_accuracy_for_missing_class(tmp_path):
    workspace, private_dir = _private_workspace(tmp_path, labels=(0, 0, 0, 0))
    with pytest.raises(ValueError, match="undefined"):
        private.evaluate_private_outputs(workspace, private_dir)
    assert np.isnan(private._macro_ovr_auc(np.array([0, 1]), np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]), 3))


@pytest.mark.parametrize("corruption", ["missing_split", "wrong_key", "duplicate_key", "missing_row", "nan", "inf"])
def test_private_rejects_corrupt_final_artifacts(tmp_path, corruption):
    workspace, private_dir = _private_workspace(tmp_path)
    path = workspace / "working" / "submission.csv"
    frame = pd.read_csv(path)
    if corruption == "missing_split":
        frame = frame.drop(columns="split")
    elif corruption == "wrong_key":
        frame.loc[0, "id"] = 99
    elif corruption == "duplicate_key":
        frame.loc[0, "id"] = frame.loc[1, "id"]
    elif corruption == "missing_row":
        frame = frame.iloc[:-1]
    else:
        frame.loc[0, "prob_0"] = float(corruption)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        private.evaluate_private_outputs(workspace, private_dir)


@pytest.mark.parametrize("labels", [[[0, 0], [1, 0]], [[0.5, 0], [1, 1]], [[float("nan"), 0], [1, 1]]])
def test_private_multilabel_rejects_undefined_or_invalid_truth(labels):
    truth = pd.DataFrame(labels, columns=["a", "b"])
    submission = pd.DataFrame({"prob_a": [0.1, 0.9], "prob_b": [0.2, 0.8]})
    with pytest.raises(ValueError):
        private._multilabel_metrics_from_submission(truth, submission, ["a", "b"], None, ["prob_a", "prob_b"], [], "auc")


@pytest.mark.parametrize("metric", ["auc", "auroc", "macro_auroc", "macro_auroc_ovr", "macro_roc_auc", "mean_roc_auc"])
def test_private_auc_aliases_keep_requested_auc_as_primary(tmp_path, metric):
    workspace, private_dir = _private_workspace(tmp_path, metric=metric)
    result = private.evaluate_private_outputs(workspace, private_dir)["results"][0]
    assert result["metric_name"] == "private_auc"
    assert result["value"] == 1.0


@pytest.mark.parametrize("metric", ["macro_auroc", "  MACRO_AUROC  "])
def test_macro_auroc_registry_preserves_auc_classification_semantics(metric):
    info = metric_info(metric)
    assert info is not None
    assert info.name == "auc"
    assert info.task_family == "classification"
    assert info.maximize is True
    assert metric_maximize(metric, default=False) is True
    assert is_classification_metric_supported(metric)
    assert not is_segmentation_metric_supported(metric)


@pytest.mark.parametrize("missing_class", [False, True])
def test_private_macro_auroc_scores_seven_class_probabilities(tmp_path, missing_class):
    from sklearn.metrics import accuracy_score, roc_auc_score

    workspace = tmp_path / "workspace"
    working = workspace / "working"
    working.mkdir(parents=True)
    private_dir = tmp_path / "synthetic_private_gt"
    private_dir.mkdir()
    labels = np.repeat(np.arange(7), np.arange(2, 9))
    if missing_class:
        labels = labels[labels != 6]
    scores = np.random.default_rng(29).uniform(0.01, 1.0, (len(labels), 7))
    scores[np.arange(len(labels)), labels] += np.linspace(0.0, 0.8, len(labels))
    scores /= scores.sum(axis=1, keepdims=True)
    prediction_columns = [f"prob_{index}" for index in range(7)]
    keys = pd.DataFrame({"id": np.arange(len(labels)), "split": "test"})
    keys.assign(label=labels).to_csv(private_dir / "test.csv", index=False)
    predictions = pd.concat([keys, pd.DataFrame(scores, columns=prediction_columns)], axis=1)
    # Private scoring must align declared keys, not rely on submission row order.
    predictions.iloc[::-1].to_csv(working / "submission.csv", index=False)
    (private_dir / "private_gt.json").write_text(json.dumps({
        "test": {"file": "test.csv", "answer_columns": ["label"]},
    }))
    (workspace / "evaluator_prediction_spec.json").write_text(json.dumps({
        "test_submission": "./working/submission.csv",
        "key_columns": ["id", "split"],
        "prediction_columns": prediction_columns,
        "class_labels": list(range(7)),
        "answer_columns": ["label"],
        "task_type": "multiclass",
        "primary_metric": "macro_auroc",
        "maximize": True,
    }))
    if missing_class:
        with pytest.raises(ValueError, match="every declared class needs positive and negative examples"):
            private.evaluate_private_outputs(workspace, private_dir)
        return

    result = private.evaluate_private_outputs(workspace, private_dir)
    assert result["visible_to_agent"] is False
    assert len(result["results"]) == 1
    score = result["results"][0]
    expected = roc_auc_score(labels, scores, labels=list(range(7)), multi_class="ovr", average="macro")
    assert score["metric_name"] == "private_auc"
    assert score["value"] == pytest.approx(expected)
    assert score["metrics"]["auc"] == pytest.approx(expected)
    assert score["metrics"]["accuracy"] == pytest.approx(accuracy_score(labels, scores.argmax(axis=1)))
    assert score["value"] != pytest.approx(score["metrics"]["accuracy"])
    assert score["maximize"] is True
    assert score["n"] == len(labels)


@pytest.mark.parametrize("prediction", [float("nan"), "unknown"])
def test_private_accuracy_rejects_missing_or_unknown_predicted_labels(prediction):
    truth = pd.DataFrame({"label": [0, 1]})
    submission = pd.DataFrame({"label": [prediction, 1]})
    with pytest.raises(ValueError, match="predicted labels"):
        private._classification_metrics_from_submission(truth, submission, "label", None, ["label"], [], "accuracy", {"class_labels": [0, 1]})


def test_trusted_contract_rejects_string_metric_direction(tmp_path):
    workspace, hidden, spec = _classification_workspace(tmp_path)
    evaluator_path = workspace / "evaluator.py"
    evaluator_path.write_text(evaluator_path.read_text() + "\n_original_score = score_val\ndef score_val():\n    result = _original_score()\n    result['maximize'] = 'false'\n    return result\n")
    with pytest.raises(evaluator.EvaluatorSetupError, match="maximize must be a boolean"):
        _contract_check(workspace, hidden, spec)
