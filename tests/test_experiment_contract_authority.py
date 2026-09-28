"""Cross-artifact authority checks for the prepared experiment contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biomaker.utils.experiment_contract import load_experiment_contract
from biomaker.utils.split_contract import SplitContractError


def prepared_workspace(tmp_path: Path, *, test: bool = False) -> tuple[Path, dict]:
    workspace = tmp_path / "workspace"
    split_dir = workspace / "input" / "biomaker_splits"
    split_dir.mkdir(parents=True)
    splits = {"train": "agent_train.csv", "score_val": "score_val.csv"}
    if test:
        splits["test"] = "agent_test.csv"
    for role, name in splits.items():
        (split_dir / name).write_text(f"case_id\n{role}_1\n", encoding="utf-8")
    manifest = {
        "manifest_version": 2,
        "splits": {role: {"file": f"biomaker_splits/{name}"} for role, name in splits.items()},
        "key_columns": ["case_id"],
        "hidden_answers": {"score_val": {"answer_columns": ["label"]}},
        "prediction_contract": {"prediction_columns": ["prediction"]},
        "metric_contract": {"metrics": [{"name": "accuracy", "maximize": True}]},
    }
    (split_dir / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return workspace, manifest


def test_prepared_manifest_controls_roles_keys_metrics_and_artifacts(tmp_path):
    workspace, _ = prepared_workspace(tmp_path)
    # Both an obsolete evaluator declaration and a stale disk file disagree.
    (workspace / "input/biomaker_splits/agent_test.csv").write_text("case_id\nstale\n")
    (workspace / "evaluator_prediction_spec.json").write_text(json.dumps({
        "key_columns": ["wrong"],
        "prediction_columns": ["wrong"],
        "metrics": [{"name": "mae", "maximize": False}],
        "primary_metric": "mae", "secondary_metrics": ["mse"], "maximize": False,
        "test_split": "./input/biomaker_splits/agent_test.csv",
        "test_submission": "./working/submission.csv",
        "validation_artifact": "./working/obsolete.csv",
    }))
    contract = load_experiment_contract(workspace)
    assert list(contract.splits) == ["train", "score_val"]
    assert list(contract.prediction_jobs) == ["score_val"]
    assert contract.prediction.test_submission is None
    assert contract.prediction.key_columns == ["case_id"]
    assert contract.prediction.prediction_columns == ["prediction"]
    assert contract.metric.primary_metric is None
    assert contract.metric.secondary_metrics == []
    assert contract.metric.metrics == [{"name": "accuracy", "maximize": True}]
    assert contract.prediction.validation_artifact == workspace / "working/validation_predictions.csv"
    assert contract.prediction.raw["test_split"] is None
    assert contract.prediction.raw["prediction_jobs"] == contract.public_prediction_jobs()


@pytest.mark.parametrize("role", ["train", "score_val", "test"])
def test_missing_declared_table_is_corruption_not_role_removal(tmp_path, role):
    workspace, manifest = prepared_workspace(tmp_path, test=True)
    (workspace / "input" / manifest["splits"][role]["file"]).unlink()
    with pytest.raises(SplitContractError, match=f"Declared {role} split file is missing"):
        load_experiment_contract(workspace)


def test_public_test_does_not_require_private_test_answers(tmp_path):
    workspace, _ = prepared_workspace(tmp_path, test=True)
    contract = load_experiment_contract(workspace)
    assert set(contract.prediction_jobs) == {"score_val", "test"}
    assert "test" not in contract.hidden_answers
    assert contract.evaluator_lifecycle_spec()["test_submission"] == "./working/submission.csv"


def test_public_view_does_not_contain_private_bindings(tmp_path):
    workspace, _ = prepared_workspace(tmp_path)
    private_dir = tmp_path / "secret_answers"
    contract = load_experiment_contract(workspace, hidden_answers_dir=private_dir)
    assert contract.hidden_answers["score_val"].path == private_dir / "score_val.csv"
    assert str(private_dir) not in json.dumps(contract.evaluator_lifecycle_spec())


def test_malformed_prepared_manifest_does_not_fall_back_to_legacy(tmp_path):
    workspace, _ = prepared_workspace(tmp_path)
    (workspace / "input/biomaker_splits/split_manifest.json").write_text("{broken")
    with pytest.raises(SplitContractError, match="Cannot read contract JSON"):
        load_experiment_contract(workspace)


def test_prepared_answer_schema_does_not_fall_back_to_stale_plan(tmp_path):
    workspace, manifest = prepared_workspace(tmp_path)
    manifest['hidden_answers'] = {}
    split_dir = workspace / 'input/biomaker_splits'
    (split_dir/'split_manifest.json').write_text(json.dumps(manifest))
    (split_dir/'split_plan.json').write_text(json.dumps({
        'hidden_answers': {'score_val': {'answer_columns': ['obsolete_label']}}
    }))
    contract = load_experiment_contract(workspace)
    assert contract.hidden_answers == {}


def test_prepared_metric_direction_is_not_changed_by_legacy_evaluator(tmp_path):
    workspace, manifest = prepared_workspace(tmp_path)
    manifest['metric_contract'] = {'metrics': [{'name': 'mae', 'maximize': False}]}
    (workspace/'input/biomaker_splits/split_manifest.json').write_text(json.dumps(manifest))
    (workspace/'evaluator_prediction_spec.json').write_text(json.dumps({
        'primary_metric': 'accuracy', 'maximize': True,
    }))
    contract = load_experiment_contract(workspace)
    assert contract.prediction.maximize is False
    assert contract.metric.maximize is False
    assert contract.metric.primary_metric is None
