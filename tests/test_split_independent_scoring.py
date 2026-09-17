import json
from pathlib import Path

import pandas as pd
import pytest

from biomaker.utils.evaluator_setup import (
    _normalize_prediction_spec_json,
    _validate_scoring_contract,
    score_candidate_with_evaluator,
)
from biomaker.utils.evaluator_protocol import requested_metrics, validate_score
from biomaker.utils.private_evaluation import evaluate_private_outputs


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def scoring_run(tmp_path):
    workspace, private = tmp_path / "run", tmp_path / "private"
    split = workspace / "input/biomaker_splits"
    spec = {
        "evaluator_api_version": 2,
        "maximize": True,
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
        "key_columns": ["id"],
        "prediction_columns": ["boxes"],
        "primary_metric": "exact_match",
        "secondary_metrics": ["mean_count"],
    }
    write_json(workspace / "evaluator_prediction_spec.json", spec)
    write_json(
        private / "private_gt.json", {"test": {"answer_columns": ["label_path"]}}
    )
    value = json.dumps([[1, 2, 3, 4, 0.9]] * 40)
    for name, ids, artifact in [
        ("score_val", ["a", "b"], "validation_predictions"),
        ("test", ["c", "d", "e"], "submission"),
    ]:
        public = "agent_test" if name == "test" else name
        write_csv(split / f"{public}.csv", [{"id": i} for i in ids])
        write_csv(
            private / f"{name}.csv", [{"id": i, "label_path": value} for i in ids]
        )
        write_csv(
            workspace / f"working/{artifact}.csv",
            [{"id": i, "boxes": value} for i in reversed(ids)],
        )
    # Unknown metric and opaque JSON payload: the harness must not guess either
    # a metric implementation or a file reference from label_path/boxes names.
    (workspace / "evaluator.py").write_text("""
import json
import os
import pandas as pd
def validate_submission():
    return {'valid': True}
def score_val():
    raise AssertionError('Harness must use the split-independent score API')
def score(prediction_path, truth_path, expected_path):
    assert os.environ.get('BIOMAKER_SCORE_VAL_PATH', '') == ''
    pred = pd.read_csv(prediction_path).set_index('id')
    truth = pd.read_csv(truth_path).set_index('id')
    keys = pd.read_csv(expected_path)['id'].tolist()
    hits = [json.loads(pred.loc[k, 'boxes']) == json.loads(truth.loc[k, 'label_path']) for k in keys]
    counts = [len(json.loads(pred.loc[k, 'boxes'])) for k in keys]
    return {'metrics': {'exact_match': sum(hits)/len(keys), 'mean_count': sum(counts)/len(keys)}}
""")
    return workspace, private, spec


@pytest.mark.parametrize(
    "heads", [["detection"], [{"type": "detection"}], ["unseen_task"], None]
)
def test_one_scorer_for_validation_and_test_without_task_routing(scoring_run, heads):
    workspace, private, _ = scoring_run
    write_json(
        workspace / "input/biomaker_splits/split_manifest.json", {"task_heads": heads}
    )
    before = {p: p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    val = score_candidate_with_evaluator(
        workspace, hidden_score_path=private / "score_val.csv"
    )
    assert val.ok, val.error
    test = evaluate_private_outputs(workspace, private)
    assert "error" not in test, test
    assert test["results"][0]["metrics"] == val.details["score"]["metrics"]
    assert test["results"][0]["n"] == 3
    assert val.details["score"]["n"] == 2
    assert test["visible_to_agent"] is False
    assert all(p.read_bytes() == content for p, content in before.items())
    # Test consumes test truth, never validation answers or a cached val score.
    write_csv(
        private / "test.csv", [{"id": i, "label_path": "[]"} for i in ["c", "d", "e"]]
    )
    changed = evaluate_private_outputs(workspace, private)
    assert changed["results"][0]["value"] == 0


@pytest.mark.parametrize(
    "problem", ["missing_secondary", "missing_primary", "nonfinite", "wrong_primary"]
)
def test_score_contract_rejects_substitutions_and_missing_metrics(problem):
    spec = {"primary_metric": "ap50", "secondary_metrics": ["recall"], "maximize": True}
    score = {"metrics": {"ap50": 0.4, "recall": 0.6}}
    if problem == "missing_secondary":
        del score["metrics"]["recall"]
    if problem == "missing_primary":
        del score["metrics"]["ap50"]
    if problem == "nonfinite":
        score["metrics"]["recall"] = float("nan")
    if problem == "wrong_primary":
        score["metric"] = 0.9
    with pytest.raises(ValueError):
        validate_score(score, spec)


def test_secondary_metrics_and_format_survive_normalization(scoring_run):
    _, _, spec = scoring_run
    spec.update(detection_format={"coordinates": "xyxy"}, iou_threshold=0.5)
    normalized = json.loads(_normalize_prediction_spec_json(json.dumps(spec)))
    assert normalized["secondary_metrics"] == spec["secondary_metrics"]
    assert normalized["detection_format"] == spec["detection_format"]
    assert normalized["iou_threshold"] == 0.5


@pytest.mark.parametrize("word", ["evaluation", "validation"])
def test_explicit_requirements_checked(word):
    task = f"Primary {word} metric: mAP@0.5 (higher is better).\nSecondary {word} metric: Recall at IoU=0.50."
    assert requested_metrics(task) == ("ap50", ["recall"])
    spec = {
        "evaluator_api_version": 2,
        "primary_metric": "ap50",
        "secondary_metrics": [],
        "maximize": True,
    }
    with pytest.raises(RuntimeError, match="secondary"):
        _validate_scoring_contract(spec, task)


def test_legacy_evaluator_is_not_silently_rescored(scoring_run):
    workspace, private, spec = scoring_run
    del spec["evaluator_api_version"]
    write_json(workspace / "evaluator_prediction_spec.json", spec)
    result = evaluate_private_outputs(workspace, private)
    assert not result["results"]
    assert "predates" in result["error"]


@pytest.mark.parametrize("problem", ["duplicate", "missing", "wrong_keys"])
def test_private_sample_contract_checked(scoring_run, problem):
    workspace, private, _ = scoring_run
    ids = {
        "duplicate": ["c", "c", "e"],
        "missing": ["c", "d"],
        "wrong_keys": ["c", "d", "x"],
    }[problem]
    write_csv(
        workspace / "working/submission.csv", [{"id": i, "boxes": "[]"} for i in ids]
    )
    result = evaluate_private_outputs(workspace, private)
    assert result["error"]
    assert result["results"] == []


@pytest.mark.parametrize("family", ["classification", "segmentation"])
def test_fixed_templates_use_identical_validation_and_test_math(tmp_path, family):
    from PIL import Image
    from biomaker.utils.evaluator_templates import (
        _FIXED_CLASSIFICATION_EVALUATOR,
        _FIXED_SEGMENTATION_EVALUATOR,
    )

    workspace, private = tmp_path / "run", tmp_path / "private"
    spec = {
        "evaluator_api_version": 2,
        "maximize": True,
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
        "key_columns": ["id"],
        "score_split": "./input/biomaker_splits/score_val.csv",
    }
    if family == "classification":
        spec.update(
            task_type="binary",
            primary_metric="accuracy",
            secondary_metrics=["auc"],
            class_labels=[0, 1],
            positive_class=1,
            prediction_columns=["prob_0", "prob_1"],
            answer_columns=["label"],
        )
        answers = [{"id": "a", "label": 0}, {"id": "b", "label": 1}]
        predictions = [
            {"id": "b", "prob_0": 0.1, "prob_1": 0.9},
            {"id": "a", "prob_0": 0.9, "prob_1": 0.1},
        ]
        template = _FIXED_CLASSIFICATION_EVALUATOR
    else:
        spec.update(
            task_type="segmentation",
            primary_metric="aji",
            secondary_metrics=["binary_dice"],
            prediction_columns=["mask"],
            answer_columns=["mask_path"],
        )
        workspace.mkdir()
        mask = workspace / "mask.png"
        Image.new("L", (4, 4), color=1).save(mask)
        answers = [
            {"id": "a", "mask_path": str(mask)},
            {"id": "b", "mask_path": str(mask)},
        ]
        predictions = [{"id": "b", "mask": str(mask)}, {"id": "a", "mask": str(mask)}]
        template = _FIXED_SEGMENTATION_EVALUATOR
    write_json(workspace / "evaluator_prediction_spec.json", spec)
    write_json(
        private / "private_gt.json",
        {"test": {"answer_columns": spec["answer_columns"]}},
    )
    (workspace / "evaluator.py").write_text(
        template.replace("__BIOMAKER_SPEC__", repr(spec))
    )
    for name in ["score_val", "agent_test"]:
        write_csv(
            workspace / f"input/biomaker_splits/{name}.csv", [{"id": "a"}, {"id": "b"}]
        )
    for name in ["validation_predictions", "submission"]:
        write_csv(workspace / f"working/{name}.csv", predictions)
    for name in ["score_val", "test"]:
        write_csv(private / f"{name}.csv", answers)
    val = score_candidate_with_evaluator(
        workspace, hidden_score_path=private / "score_val.csv"
    )
    test = evaluate_private_outputs(workspace, private)
    assert val.ok, val.error
    assert "error" not in test, test
    assert val.metric == test["results"][0]["value"] == 1.0
    assert val.details["score"]["metrics"] == test["results"][0]["metrics"]
    if family == "classification":
        # Undefined AUC must not silently become accuracy on either split.
        spec.update(primary_metric="auc", secondary_metrics=[])
        write_json(workspace / "evaluator_prediction_spec.json", spec)
        (workspace / "evaluator.py").write_text(
            template.replace("__BIOMAKER_SPEC__", repr(spec))
        )
        for name in ["score_val", "test"]:
            write_csv(
                private / f"{name}.csv",
                [{"id": "a", "label": 0}, {"id": "b", "label": 0}],
            )
        assert not score_candidate_with_evaluator(
            workspace, hidden_score_path=private / "score_val.csv"
        ).ok
        assert evaluate_private_outputs(workspace, private)["results"] == []
