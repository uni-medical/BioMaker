import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from biomaker.utils.detection_contract import fixed_detection_spec
from biomaker.utils.detection_evaluator_template import _FIXED_DETECTION_EVALUATOR

SCHEMA = dict(coordinate_keys=["bbox"], class_key="category_id", box_format="xyxy")
SPEC = dict(
    key_columns=["id"],
    prediction_columns=["boxes"],
    answer_columns=["boxes"],
    prediction_schema=SCHEMA,
    truth_schema=SCHEMA,
    primary_metric="ap50",
    iou_threshold=0.5,
    recall_confidence_threshold=0.0,
)


def module(spec=None):
    result = {}
    exec(
        _FIXED_DETECTION_EVALUATOR.replace("__BIOMAKER_SPEC__", repr(spec or SPEC)),
        result,
    )
    return result


def box(cls=0, score=1.0, coords=(0, 0, 2, 2)):
    return cls, coords, score


def test_perfect_and_missing():
    evaluate = module()["_evaluate"]
    assert evaluate([[box()]], [[box()]])[0] == {"ap50": 1.0, "recall": 1.0}
    assert evaluate([[box()]], [[]])[0] == {"ap50": 0.0, "recall": 0.0}


def test_duplicate_false_positive_and_global_ranking():
    # TP, duplicate FP, TP -> all-point AP = (1 + 2/3)/2.
    result, _ = module()["_evaluate"](
        [[box()], [box()]], [[box(score=0.9), box(score=0.8)], [box(score=0.7)]]
    )
    assert result["ap50"] == pytest.approx(5 / 6)
    assert result["recall"] == 1.0


def test_negative_image_false_positive_and_class_mismatch():
    metrics, _ = module()["_evaluate"](
        [[box()], []], [[box(score=0.8)], [box(score=0.9)]]
    )
    assert metrics["ap50"] == 0.5
    assert module()["_evaluate"]([[box()]], [[box(cls=1)]])[0] == {
        "ap50": 0.0,
        "recall": 0.0,
    }


def test_macro_ap_and_micro_recall():
    gt = [[box(0)], [box(0)], [box(1)]]
    pred = [[box(0)], [], []]
    metrics, _ = module()["_evaluate"](gt, pred)
    assert metrics == {"ap50": 0.25, "recall": 1 / 3}


def test_recall_threshold_does_not_filter_ap():
    spec = {**SPEC, "recall_confidence_threshold": 0.5}
    assert module(spec)["_evaluate"]([[box()]], [[box(score=0.4)]])[0] == {
        "ap50": 1.0,
        "recall": 0.0,
    }


def test_iou_half_inclusive():
    assert (
        module()["_evaluate"]([[box()]], [[box(coords=(0, 0, 1, 2))]])[0]["ap50"] == 1.0
    )


def test_duplicate_does_not_match_second_best_gt():
    truth = [[box(), box(coords=(0, 0, 3, 2))]]
    pred = [[box(score=0.9), box(score=0.8)]]
    assert module()["_evaluate"](truth, pred)[0]["recall"] == 0.5


@pytest.mark.parametrize(
    "payload",
    [
        "NaN",
        "{}",
        "[null]",
        "[[1,2,3,4]]",
        '[{"category_id":0,"bbox":[0,0,1,1]}]',
        '[{"category_id":0,"bbox":[0,0,0,1],"score":0.5}]',
        '[{"category_id":0,"bbox":[0,0,1,1],"score":NaN}]',
        '[{"category_id":0,"bbox":[0,0,1,1],"score":0.5,"iscrowd":1}]',
    ],
)
def test_invalid_predictions_rejected(payload):
    with pytest.raises((ValueError, KeyError)):
        module()["_parse_boxes"](payload, True)


def test_no_ground_truth_is_explicit_error():
    with pytest.raises(ValueError, match="undefined"):
        module()["_evaluate"]([[]], [[]])


def context():
    schema_text = "normalized objects x_center y_center width height class score"
    gt = json.dumps(
        [dict(x_center=0.5, y_center=0.5, width=0.2, height=0.2, **{"class": 0})]
    )
    return dict(
        split_manifest=dict(
            task_heads=["detection"],
            key_columns=["id"],
            prediction_contract={"box_format": schema_text},
        ),
        score_df=pd.DataFrame({"id": ["a"]}),
        test_df=pd.DataFrame({"id": ["b"]}),
        hidden_df=pd.DataFrame({"id": ["a"], "boxes": [gt]}),
        answer_columns=["boxes"],
    )


def test_admission_and_custom_protocol_fallback():
    task = "Primary evaluation metric: mAP@0.5\nSecondary evaluation metric: Recall at IoU=0.50"
    spec = fixed_detection_spec(context(), task)
    assert spec["metrics"][0] == {"name": "ap50", "maximize": True}
    assert spec["prediction_schema"]["box_format"] == "cxcywh"
    assert spec["metrics"][1:] == [{"name": "recall", "maximize": True}]
    for extra in [
        "\nUse COCO interpolation.",
        "\nRecall confidence threshold: 0.2",
        "\nRecall at IoU=0.75",
    ]:
        assert fixed_detection_spec(context(), task + extra) is None
    c = context()
    c["hidden_df"]["boxes"] = "[[1,2,3,4]]"
    assert fixed_detection_spec(c, task) is None


def test_score_alignment_and_no_path_guessing(tmp_path):
    m = module()
    gt = json.dumps([{"category_id": 0, "bbox": [0, 0, 2, 2]}])
    pred = json.dumps([{"category_id": 0, "bbox": [0, 0, 2, 2], "score": 0.8}] * 50)
    pd.DataFrame({"id": ["a", "b"]}).to_csv(tmp_path / "expected.csv", index=False)
    pd.DataFrame({"id": ["b", "a"], "boxes": [gt, gt]}).to_csv(
        tmp_path / "truth.csv", index=False
    )
    pd.DataFrame({"id": ["b", "a"], "boxes": [pred, pred]}).to_csv(
        tmp_path / "pred.csv", index=False
    )
    result = m["score"](
        *(str(tmp_path / n) for n in ["pred.csv", "truth.csv", "expected.csv"])
    )
    assert result["metrics"]["recall"] == 1.0
    pd.DataFrame({"id": ["a", "a"], "boxes": [pred, pred]}).to_csv(
        tmp_path / "pred.csv", index=False
    )
    with pytest.raises(ValueError):
        m["score"](
            *(str(tmp_path / n) for n in ["pred.csv", "truth.csv", "expected.csv"])
        )
