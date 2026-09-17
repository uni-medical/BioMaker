"""Prepared experiment semantics apply to standard and custom metric routes."""

import copy
import json

import pytest

from biomaker.utils import evaluator_setup


def manifest_at(workspace, metric_contract, **fields):
    manifest = {
        "manifest_version": 1,
        "sample_unit": "sample",
        "key_columns": ["case_uid"],
        "task_heads": ["image_restoration"],
        "prediction_contract": {"prediction_columns": ["estimate_json"]},
        "metric_contract": metric_contract,
        **fields,
    }
    path = workspace / "input/biomaker_splits/split_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    return manifest


@pytest.fixture
def custom_protocol(tmp_path):
    contract = {
        "primary_metric": "inverse_mae",
        "secondary_metrics": ["ssim"],
        "maximize": True,
        "restoration_evaluation": {
            "data_range": 255,
            "crop_size": 8,
            "crop_mode": "four_corners",
            "groups": ["scanner-a", "scanner-b"],
            "aggregation": "mean_groups",
        },
        "ssim_parameters": {"window_size": 7},
    }
    manifest_at(tmp_path, contract)
    spec = {
        **copy.deepcopy(contract),
        "evaluator_api_version": 2,
        "key_columns": ["case_uid"],
        "prediction_columns": ["estimate_json"],
    }
    return tmp_path, spec


@pytest.mark.parametrize("mutation,field", [
    ("drop_secondary", "secondary"),
    ("reverse_direction", "maximize"),
    ("drop_aggregation", "restoration_evaluation"),
    ("change_aggregation", "restoration_evaluation"),
    ("change_keys", "sample keys"),
    ("change_prediction_columns", "prediction_columns"),
])
def test_custom_extension_must_preserve_prepared_contract(custom_protocol, mutation, field):
    workspace, spec = custom_protocol
    if mutation == "drop_secondary":
        spec["secondary_metrics"] = []
    elif mutation == "reverse_direction":
        spec["maximize"] = False
    elif mutation == "drop_aggregation":
        del spec["restoration_evaluation"]["aggregation"]
    elif mutation == "change_aggregation":
        spec["restoration_evaluation"]["aggregation"] = "mean_images"
    elif mutation == "change_keys":
        spec["key_columns"] = ["new_id"]
    elif mutation == "change_prediction_columns":
        spec["prediction_columns"] = ["other_output"]
    with pytest.raises(evaluator_setup.EvaluatorSetupError, match=field):
        evaluator_setup._validate_custom_metric_contract(spec, workspace, "Primary evaluation metric: inverse_mae.")


def test_custom_extension_accepts_equivalent_supported_parameter_aliases(custom_protocol):
    workspace, spec = custom_protocol
    spec["ssim_parameters"] = {"win_size": 7}
    spec["restoration_evaluation"]["crop_mode"] = "four_corner_crops_no_center"
    evaluator_setup._validate_custom_metric_contract(spec, workspace, "Primary metric: inverse_mae.")


def test_custom_extension_accepts_nested_ssim_declaration(custom_protocol):
    workspace, spec = custom_protocol
    spec["restoration_evaluation"]["ssim_parameters"] = {"win_size": 7}
    del spec["ssim_parameters"]
    evaluator_setup._validate_custom_metric_contract(spec, workspace, "Primary metric: inverse_mae.")


def test_custom_extension_preserves_task_owned_settings_block(custom_protocol):
    workspace, spec = custom_protocol
    task = 'Primary metric: inverse_mae.\n```restoration-evaluation\n{"data_range": 1}\n```'
    with pytest.raises(evaluator_setup.EvaluatorSetupError, match="restoration_evaluation"):
        evaluator_setup._validate_custom_metric_contract(spec, workspace, task)


def test_custom_extension_checks_manifest_when_task_omits_explicit_metric_line(custom_protocol):
    workspace, spec = custom_protocol
    spec["secondary_metrics"] = []
    with pytest.raises(evaluator_setup.EvaluatorSetupError, match="secondary"):
        evaluator_setup._validate_custom_metric_contract(spec, workspace, "Restore these images.")


@pytest.mark.parametrize("contract", [
    {"primary_metric": "auc", "auc_average": "micro"},
    {"primary_metric": "auc", "task_type": "binary", "class_labels": [0, 1], "threshold": 1.2},
    {"primary_metric": "binary_dice", "aggregation": "macro_classes"},
    {"primary_metric": "ap50", "iou_threshold": 0.75},
    {"primary_metric": "ssim", "ssim_parameters": {"window_size": 4}},
    {"primary_metric": "ssim", "ssim_parameters": {"unsupported_setting": True}},
])
def test_unsupported_standard_settings_fail_before_backend_query(tmp_path, monkeypatch, contract):
    manifest_at(tmp_path, contract)
    calls = []

    def forbidden_query(**kwargs):
        calls.append(kwargs)
        raise AssertionError("Unsupported metric semantics must not retry data adapters.")

    monkeypatch.setattr(evaluator_setup.backend, "query", forbidden_query)
    with pytest.raises(evaluator_setup.EvaluatorSetupError, match="Unsupported"):
        evaluator_setup.prepare_evaluator(
            tmp_path, tmp_path / "logs", "", model="must-not-call", attempts=1,
        )
    assert not calls


def test_supported_aliases_pass_capability_preflight(tmp_path):
    manifest_at(tmp_path, {
        "primary_metric": "ssim", "maximize": True,
        "data_range": 255, "crop_size": 8, "groups": None, "aggregation": "mean_images",
        "ssim_parameters": {"window_size": 7},
    })
    evaluator_setup._validate_declared_capabilities("Primary metric: SSIM.", tmp_path)


def test_preparation_context_preserves_data_semantics(tmp_path):
    fields = {
        "data_semantics": {"intensity_units": "raw_u16", "array_axes": ["z", "y", "x"],
                           "label_mapping": {"0": "background", "1": "organ"}},
        "input_file_reference_columns": ["scan_archive"],
        "semantic_assumptions": ["Each row is one patient; channels are stored in the archive."],
        "row_representation": "one_row_per_patient",
        "sample_universe_summary": {"patients": 5},
    }
    manifest_at(tmp_path, {"primary_metric": "custom_volume_metric", "maximize": True}, **fields)
    guide, declared, _ = evaluator_setup._preparation_context(tmp_path, tmp_path / "private/score_val.csv")
    for key, value in fields.items():
        assert declared[key] == value
    assert guide["data_semantics"]["data_semantics"] == fields["data_semantics"]
    assert guide["data_semantics"]["input_file_reference_columns"] == fields["input_file_reference_columns"]
