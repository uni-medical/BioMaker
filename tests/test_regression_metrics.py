"""Regression kernels preserve continuous values and the declared reduction."""

import math

import numpy as np
import pytest

from biomaker.utils.trusted_metrics import (
    UndefinedMetricError,
    metric_capabilities,
    normalize_metric_spec,
    score_samples,
    validate_prediction,
)


def regression_spec(**overrides):
    return {"primary_metric": "mae", "secondary_metrics": [], **overrides}


def test_scalar_regression_preserves_fractional_units_without_rounding():
    spec = regression_spec(secondary_metrics=["mse", "rmse"], target_shape=[], units="months")
    result = score_samples(iter([
        {"prediction": 10.25, "target": 10},
        {"prediction": 19.25, "target": 20},
    ]), spec)
    assert result == {
        "metric": 0.5,
        "maximize": False,
        "metrics": {"mae": 0.5, "mse": 0.3125, "rmse": math.sqrt(0.3125)},
        "n": 2,
    }
    normalized = normalize_metric_spec(spec)
    assert normalized["task_family"] == normalized["task_type"] == "regression"
    assert normalized["units"] == "months"
    assert normalized["rounding"] == "none"
    assert normalized["multioutput"] == "uniform_average"
    assert normalized["aggregation"] == "mean_samples"


def test_vector_regression_uses_root_after_uniform_global_squared_error():
    result = score_samples([
        {"prediction": [1, 9], "target": [0, 0]},
        {"prediction": [3, 3], "target": [0, 0]},
    ], regression_spec(secondary_metrics=["mse", "rmse"], target_shape=[2]))
    assert result["metrics"] == {"mae": 4.0, "mse": 25.0, "rmse": 5.0}
    # Mean per-output roots is a different definition, deliberately unsupported.
    assert result["metrics"]["rmse"] != pytest.approx((math.sqrt(5) + math.sqrt(45)) / 2)


@pytest.mark.parametrize("name,canonical,expected", [
    ("mean_absolute_error", "mae", 3.0),
    ("mean_squared_error", "mse", 9.0),
    ("root_mean_squared_error", "rmse", 3.0),
])
def test_regression_metric_aliases_and_direction_share_the_catalog(name, canonical, expected):
    capabilities = metric_capabilities()
    assert capabilities["aliases"][name] == canonical
    assert capabilities["metrics"][canonical]["maximize"] is False
    assert capabilities["metrics"][canonical]["family"] == "regression"
    result = score_samples([{"prediction": -1.0, "target": 2.0}], regression_spec(primary_metric=name))
    assert result["metrics"] == {canonical: expected}
    assert result["metric"] == expected
    assert result["maximize"] is False


def test_mae_only_does_not_compute_unrequested_overflowing_squared_errors():
    samples = [{"prediction": 1e200, "target": 0.0}]
    result = score_samples(samples, regression_spec())
    assert result["metrics"] == {"mae": 1e200}
    with pytest.raises(ValueError, match="Non-finite requested regression"):
        score_samples(samples, regression_spec(primary_metric="mse"))


@pytest.mark.parametrize("prediction,target", [
    (float("nan"), 0),
    (float("inf"), 0),
    (0, float("inf")),
    (True, 0),
    (0, False),
    ("3.5", 0),
    (0, "3.5"),
    (1 + 2j, 0),
    ([], []),
    ([[1, 2]], [[1, 2]]),
    (1.0, [1.0]),
    ([1.0], 1.0),
    ([1.0, 2.0], [1.0]),
])
def test_invalid_regression_values_and_broadcasting_are_rejected(prediction, target):
    with pytest.raises(ValueError) as error:
        score_samples([{"prediction": prediction, "target": target}], regression_spec())
    assert not isinstance(error.value, UndefinedMetricError)


def test_regression_shape_is_fixed_across_samples_even_without_explicit_shape():
    with pytest.raises(ValueError, match="fixed across all samples"):
        score_samples([
            {"prediction": [1], "target": [1]},
            {"prediction": [1, 2], "target": [1, 2]},
        ], regression_spec())


def test_declared_regression_shape_is_checked_without_private_targets():
    assert validate_prediction({"prediction": 2.4}, regression_spec(target_shape=[])) == {"valid": True}
    assert validate_prediction({"prediction": [2.4, -1]}, regression_spec(target_shape=[2])) == {"valid": True}
    with pytest.raises(ValueError, match="target_shape"):
        validate_prediction({"prediction": [2.4]}, regression_spec(target_shape=[]))
    with pytest.raises(ValueError, match="target_shape"):
        validate_prediction({"prediction": [2.4]}, regression_spec(target_shape=[2]))


@pytest.mark.parametrize("overrides", [
    {"maximize": True},
    {"task_type": "binary"},
    {"multioutput": "raw_values"},
    {"aggregation": "weighted_mean"},
    {"rounding": "nearest_integer"},
    {"units": ""},
    {"units": 12},
    {"target_shape": None},
    {"target_shape": [0]},
    {"target_shape": [-1]},
    {"target_shape": [True]},
    {"target_shape": [2.0]},
    {"target_shape": [2, 2]},
])
def test_conflicting_regression_protocol_is_rejected(overrides):
    with pytest.raises(ValueError):
        normalize_metric_spec(regression_spec(**overrides))


def test_empty_regression_split_fails_explicitly():
    with pytest.raises(ValueError, match="at least one sample"):
        score_samples([], regression_spec())


@pytest.mark.parametrize("labels,positive", [(["no", "yes"], "yes"), (["yes", "no"], "yes")])
@pytest.mark.parametrize("probability", [0.25, 0.5, 0.75])
def test_binary_probability_representation_preserves_threshold_ties(labels, positive, probability):
    spec = {
        "primary_metric": "accuracy", "task_type": "binary", "class_labels": labels,
        "positive_class": positive, "threshold": 0.5,
    }
    negative = next(label for label in labels if label != positive)
    expected = positive if probability >= 0.5 else negative
    one = score_samples([{"prediction": [probability], "target": expected}], spec)
    two = score_samples([{
        "prediction": [probability if label == positive else 1 - probability for label in labels],
        "target": expected,
    }], spec)
    assert one["metric"] == two["metric"] == 1.0


def test_undefined_hd95_is_typed_but_malformed_masks_are_not():
    spec = {"primary_metric": "hd95"}
    with pytest.raises(UndefinedMetricError) as error:
        score_samples([{"prediction": np.zeros((3, 3), dtype=int), "target": np.ones((3, 3), dtype=int)}], spec)
    assert error.value.metric == "hd95"
    assert "exactly one mask is empty" in error.value.reason
    with pytest.raises(ValueError) as error:
        score_samples([{"prediction": np.zeros((3, 3), dtype=int), "target": np.ones((4, 3), dtype=int)}], spec)
    assert not isinstance(error.value, UndefinedMetricError)


def test_undefined_psnr_preserves_strict_real_scoring():
    spec = {
        "primary_metric": "psnr",
        "restoration_evaluation": {
            "data_range": 1, "crop_size": None, "groups": None,
            "aggregation": "mean_images",
        },
    }
    with pytest.raises(UndefinedMetricError) as error:
        score_samples([{"prediction": np.zeros((8, 8)), "target": np.zeros((8, 8))}], spec)
    assert error.value.metric == "psnr"
    assert "zero MSE" in error.value.reason


def test_undefined_auc_distribution_is_distinct_from_invalid_labels():
    spec = {"primary_metric": "auc", "task_type": "binary", "class_labels": [0, 1]}
    with pytest.raises(UndefinedMetricError) as error:
        score_samples([{"prediction": [0.5], "target": 0}], spec)
    assert error.value.metric == "auc"
    with pytest.raises(ValueError) as error:
        score_samples([{"prediction": [0.5], "target": 2}], spec)
    assert not isinstance(error.value, UndefinedMetricError)
