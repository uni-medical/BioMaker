"""Foreground IoU follows the frozen per-image aggregation protocol."""

import numpy as np
import pytest

from biomaker.utils.trusted_metrics import metric_capabilities, score_samples


@pytest.mark.parametrize("metric", ["binary_iou", "iou", "jaccard"])
def test_binary_iou_uses_nonzero_foreground_and_registered_aliases(metric):
    result = score_samples([{
        "target": np.array([[4, 4, 0], [0, 9, 0]]),
        "prediction": np.array([[7, 0, 3], [0, 2, 0]]),
    }], {"primary_metric": metric})
    # Two pixels intersect; four pixels are in the foreground union.
    assert result["metrics"] == {"binary_iou": 0.5}
    assert result["metric"] == 0.5
    assert result["maximize"] is True
    assert metric_capabilities()["aliases"][metric] == "binary_iou"


def test_iou_averages_image_scores_instead_of_pooling_pixels():
    result = score_samples([
        {"target": np.array([[1, 0]]), "prediction": np.array([[1, 0]])},
        {"target": np.ones((3, 3), dtype=int), "prediction": np.zeros((3, 3), dtype=int)},
    ], {"primary_metric": "binary_iou", "secondary_metrics": ["binary_dice"]})
    assert result["metrics"] == {"binary_iou": 0.5, "binary_dice": 0.5}
    assert result["n"] == 2
    assert result["metric"] != 0.1  # A pooled-pixel IoU would be 1 / 10.


@pytest.mark.parametrize("true_foreground,pred_foreground,expected", [
    (False, False, 1.0),
    (False, True, 0.0),
    (True, False, 0.0),
    (True, True, 1.0),
])
def test_iou_empty_masks_and_3d_arrays(true_foreground, pred_foreground, expected):
    result = score_samples([{
        "target": np.full((2, 3, 4), int(true_foreground), dtype=int),
        "prediction": np.full((2, 3, 4), int(pred_foreground), dtype=int),
    }], {"primary_metric": "binary_iou"})
    assert result["metric"] == expected


def test_iou_and_dice_have_distinct_values_and_only_requested_metrics_are_returned():
    sample = {
        "target": np.array([[1, 1, 0]]),
        "prediction": np.array([[0, 1, 1]]),
    }
    result = score_samples([sample], {"primary_metric": "binary_dice", "secondary_metrics": ["iou"]})
    assert result["metrics"] == pytest.approx({"binary_dice": 0.5, "binary_iou": 1 / 3})
    assert score_samples([sample], {"primary_metric": "iou"})["metrics"] == pytest.approx({"binary_iou": 1 / 3})


def test_iou_does_not_admit_invalid_mask_shapes():
    with pytest.raises(ValueError, match="shapes must match"):
        score_samples([{
            "target": np.zeros((2, 3), dtype=int),
            "prediction": np.zeros((3, 2), dtype=int),
        }], {"primary_metric": "binary_iou"})
