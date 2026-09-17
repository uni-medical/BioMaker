"""Capability preflight checks supplied choices without freezing placeholder data."""

import json

import pytest

from biomaker.utils.evaluator_setup import (
    EvaluatorSetupError,
    _resolve_metric_contract,
    _validate_declared_capabilities,
)


def manifest(tmp_path, contract):
    path = tmp_path / "input/biomaker_splits/split_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"metric_contract": contract}))
    return path


@pytest.mark.parametrize("partial,resolved_protocol", [
    ({"aggregation": "mean_groups", "group_column": "condition"},
     {"data_range": 255, "crop_size": None, "group_column": "condition", "groups": ["low", "high"], "aggregation": "mean_groups"}),
    ({"crop_mode": "four_corners"},
     {"data_range": 255, "crop_size": 16, "groups": None, "aggregation": "mean_images", "crop_mode": "four_corners"}),
    ({"prediction_clipping": [0, 255]},
     {"data_range": 255, "crop_size": None, "groups": None, "aggregation": "mean_images"}),
])
def test_supported_partial_restoration_contract_can_be_completed_by_adapter(tmp_path, partial, resolved_protocol):
    path = manifest(tmp_path, {"primary_metric": "psnr", **partial})
    original = path.read_bytes()
    _validate_declared_capabilities("Primary evaluation metric: PSNR.", tmp_path)
    assert path.read_bytes() == original
    actual = _resolve_metric_contract({
        "primary_metric": "psnr", "secondary_metrics": [], "maximize": True,
        "task_type": "image_restoration", "restoration_evaluation": resolved_protocol,
        "prediction_clipping": [0, 255],
    }, tmp_path, "Primary evaluation metric: PSNR.")
    assert actual["restoration_evaluation"]["data_range"] == 255
    assert actual["restoration_evaluation"]["crop_size"] == resolved_protocol["crop_size"]
    assert actual["restoration_evaluation"]["groups"] == resolved_protocol["groups"]
    assert "__protocol" not in json.dumps(actual)
    assert path.read_bytes() == original


@pytest.mark.parametrize("partial", [
    {"aggregation": "mean_groups", "groups": None},
    {"crop_mode": "four_corners", "crop_size": None},
    {"prediction_clipping": [0, 255], "data_range": None},
])
def test_explicit_conflicts_are_not_treated_as_missing_adapter_information(tmp_path, partial):
    manifest(tmp_path, {"primary_metric": "psnr", **partial})
    with pytest.raises(EvaluatorSetupError, match="Unsupported"):
        _validate_declared_capabilities("Primary evaluation metric: PSNR.", tmp_path)


@pytest.mark.parametrize("partial,actual", [
    ({"task_type": "multiclass"}, {"task_type": "multiclass", "class_labels": ["a", "c", "b"]}),
    ({"aggregation": "mean_labels"}, {"task_type": "multilabel", "aggregation": "mean_labels"}),
])
def test_partial_classification_semantics_preserve_adapter_owned_class_order(tmp_path, partial, actual):
    path = manifest(tmp_path, {"primary_metric": "accuracy", **partial})
    original = path.read_bytes()
    _validate_declared_capabilities("Primary evaluation metric: Accuracy.", tmp_path)
    resolved = _resolve_metric_contract({"primary_metric": "accuracy", **actual}, tmp_path,
                                        "Primary evaluation metric: Accuracy.")
    if "class_labels" in actual:
        assert resolved["class_labels"] == ["a", "c", "b"]
    assert resolved["task_type"] == actual["task_type"]
    assert "__protocol" not in json.dumps(resolved)
    assert path.read_bytes() == original
