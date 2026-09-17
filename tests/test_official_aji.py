"""Regression checks against MoNuSeg's per-GT maximum-IoU definition."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from biomaker.utils.evaluator_templates import _FIXED_SEGMENTATION_EVALUATOR


@pytest.fixture(scope="module")
def evaluator():
    namespace = {}
    exec(_FIXED_SEGMENTATION_EVALUATOR.replace("__BIOMAKER_SPEC__", "{}"), namespace)
    return namespace


def reference_aji(truth, prediction):
    """Direct mask-wise translation of official matching, independent of pair counts."""
    total_intersection = total_union = 0
    used = set()
    for label in sorted(set(truth.ravel()) - {0}):
        gt = truth == label
        best_iou = 0
        match = None
        for candidate in sorted(set(prediction[gt]) - {0}):
            pred = prediction == candidate
            intersection = np.count_nonzero(gt & pred)
            union = np.count_nonzero(gt | pred)
            if intersection / union > best_iou:
                best_iou = intersection / union
                match = candidate, intersection, union
        if match is None:
            total_union += np.count_nonzero(gt)
        else:
            candidate, intersection, union = match
            total_intersection += intersection
            total_union += union
            used.add(candidate)
    for candidate in set(prediction.ravel()) - {0} - used:
        total_union += np.count_nonzero(prediction == candidate)
    return total_intersection / total_union if total_union else 1.0


@pytest.mark.parametrize(
    "truth,prediction,expected",
    [
        ([0, 1, 1, 2, 2], [0, 1, 1, 1, 1], 0.5),  # merged nuclei, prediction reused
        ([0, 1, 1, 1, 1], [0, 2, 2, 3, 3], 1 / 3),  # split nucleus + unused prediction
        ([0, 1, 1, 2, 2], [0, 5, 5, 0, 0], 0.5),  # missed nucleus
        ([0, 1, 1, 0, 0], [0, 5, 5, 6, 6], 0.5),  # false positive
        ([0, 7, 0, 7, 0], [0, 2, 0, 3, 0], 1 / 3),  # do not split disconnected ID
        ([0, 10**9, 10**9, 8], [0, 9, 9, 10**10], 1.0),  # sparse IDs
        ([1, 1, 2, 2], [5, 5, 6, 6], 1.0),  # background need not occur
        ([0, 0], [0, 0], 1.0),  # documented empty-mask extension
        ([0, 1], [0, 0], 0.0),
        ([0, 0], [0, 1], 0.0),
        # Tied IoUs: lowest predicted ID must win; affects later unmatched area.
        ([0, 1, 1, 1, 2, 2, 2], [0, 5, 9, 9, 9, 9, 9], 0.5),
    ],
)
def test_official_cases(evaluator, truth, prediction, expected):
    truth = np.array([truth], dtype=np.int64)
    prediction = np.array([prediction], dtype=np.int64)
    assert reference_aji(truth, prediction) == pytest.approx(expected)
    assert evaluator["_aji"](truth, prediction) == pytest.approx(expected)


def test_random_masks_match_independent_reference(evaluator):
    rng = np.random.default_rng(42)
    for _ in range(100):
        gt = rng.integers(0, 8, (9, 13))
        pred = rng.integers(0, 6, gt.shape)
        assert evaluator["_aji"](gt, pred) == pytest.approx(reference_aji(gt, pred))


@pytest.mark.parametrize("bad", [np.array([[-1, 0]]), np.array([[0.0, 0.5]])])
def test_reject_invalid_instance_labels(evaluator, bad):
    with pytest.raises(ValueError, match="integer instance labels"):
        evaluator["_aji"](np.zeros_like(bad, dtype=int), bad)


def test_shape_mismatch(evaluator):
    with pytest.raises(ValueError, match="identical shapes"):
        evaluator["_aji"](np.zeros((2, 2), dtype=int), np.zeros((1, 4), dtype=int))


def test_score_averages_images_not_pooled_areas(tmp_path):
    spec = dict(
        primary_metric="aji",
        key_columns=["id"],
        prediction_columns=["mask"],
        answer_columns=["mask"],
        secondary_metrics=["binary_dice"],
    )
    module = {}
    exec(_FIXED_SEGMENTATION_EVALUATOR.replace("__BIOMAKER_SPEC__", repr(spec)), module)
    answers, predictions = [], []
    for i, (gt, pred) in enumerate(
        [
            ([0, 1, 1, 2, 2], [0, 1, 1, 1, 1]),
            ([0, 1, 0, 0, 0], [0, 3, 0, 0, 0]),
        ]
    ):
        for values, dest, tag in [(gt, answers, "gt"), (pred, predictions, "pred")]:
            path = tmp_path / f"{tag}{i}.png"
            Image.fromarray(np.array([values], dtype=np.uint16)).save(path)
            dest.append(dict(id=str(i), mask=str(path)))
    for name, rows in [
        ("truth", answers),
        ("predictions", predictions),
        ("expected", [dict(id="0"), dict(id="1")]),
    ]:
        pd.DataFrame(rows).to_csv(tmp_path / f"{name}.csv", index=False)
    result = module["score"](
        *(
            str(tmp_path / f"{name}.csv")
            for name in ["predictions", "truth", "expected"]
        )
    )
    assert result["metric"] == pytest.approx(0.75)
    assert result["metrics"]["binary_dice"] == 1.0
