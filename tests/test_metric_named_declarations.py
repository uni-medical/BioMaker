"""Named metric objects preserve the same protocol as flat declarations."""

import copy
import json

import pytest

from biomaker.utils import evaluator_setup
from biomaker.utils.metric_declarations import (
    canonical_metric_contract,
    declared_metric_names,
    validate_declared_capabilities,
)


def manifest_at(workspace, contract):
    path = workspace / "input/biomaker_splits/split_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metric_contract": contract}))


@pytest.mark.parametrize("identifier", ["name", "metric", "id"])
def test_named_declaration_lifts_settings_without_mutating_source(identifier):
    contract = {"primary_metric": {identifier: "mean_absolute_error", "units": "months", "target_shape": []}}
    before = copy.deepcopy(contract)
    assert canonical_metric_contract(contract) == {"primary_metric": "mae", "units": "months", "target_shape": []}
    validate_declared_capabilities({"metric_contract": contract}, "Primary metric: MAE, in months.")
    assert contract == before


def test_metric_identifier_aliases_must_agree():
    good = {"primary_metric": {"name": "mean_aji", "metric": "aji", "id": "AJI"}}
    assert declared_metric_names({"metric_contract": good}, "") == ("aji", [])
    with pytest.raises(ValueError, match="Conflicting primary_metric"):
        canonical_metric_contract({"primary_metric": {"name": "aji", "metric": "dice"}})


@pytest.mark.parametrize("nested,outer", [
    ({"name": "auc", "threshold": 0.3}, {"decision_threshold": 0.4}),
    ({"name": "auc", "classification_threshold": 0.3}, {"threshold": 0.4}),
    ({"name": "ssim", "ssim_parameters": {"window_size": 7}}, {"ssim_parameters": {"win_size": 9}}),
    ({"name": "ssim", "crop_size": 16}, {"restoration_evaluation": {"crop_size": 32}}),
])
def test_conflicting_nested_and_outer_settings_are_not_overwritten(nested, outer):
    with pytest.raises(ValueError, match="Conflicting metric protocol field"):
        canonical_metric_contract({"primary_metric": nested, **outer})


def test_equivalent_aliases_and_complementary_nested_settings_merge():
    contract = {
        "primary_metric": {"name": "ssim", "crop_mode": "four_corners", "ssim_parameters": {"window_size": 7}},
        "restoration_evaluation": {"crop_mode": "four_corner_crops_no_center", "ssim_parameters": {"win_size": 7, "K1": 0.02}},
    }
    assert canonical_metric_contract(contract) == {
        "primary_metric": "ssim", "restoration_evaluation": {"crop_mode": "four_corner_crops_no_center"},
        "ssim_parameters": {"win_size": 7, "K1": 0.02},
    }


@pytest.mark.parametrize("settings,error", [
    ({"aji_matching": "one_to_one"}, "aji_matching"),
    ({"aggregation": "pooled_pixels"}, "aggregation"),
    ({"unknown_semantic_rule": True}, "unknown_semantic_rule"),
])
def test_nested_unsupported_settings_fail_both_admission_gates(tmp_path, settings, error):
    contract = {"primary_metric": {"name": "aji", **settings}}
    manifest_at(tmp_path, contract)
    with pytest.raises(ValueError, match=error):
        validate_declared_capabilities({"metric_contract": contract}, "")
    with pytest.raises(evaluator_setup.EvaluatorSetupError, match=error):
        evaluator_setup._validate_declared_capabilities("", tmp_path)


def test_evaluator_compilation_cannot_discard_nested_aji_matching(tmp_path):
    manifest_at(tmp_path, {"primary_metric": {"name": "aji", "aji_matching": "one_to_one"}})
    with pytest.raises(evaluator_setup.EvaluatorSetupError, match="aji_matching"):
        evaluator_setup._resolve_metric_contract({"primary_metric": "aji"}, tmp_path, "")


def test_evaluator_applies_nested_units_to_fixed_reader(tmp_path):
    manifest_at(tmp_path, {"primary_metric": {"name": "mae", "units": "months"}})
    actual = evaluator_setup._resolve_metric_contract(
        {"primary_metric": "mae", "evaluator_api_version": 2, "prediction_type": "continuous_scalar", "prediction_columns": ["prediction"],
         "answer_columns": ["target"], "file_reference_columns": []},
        tmp_path, "", require_match=False,
    )
    assert actual["primary_metric"] == "mae" and actual["units"] == "months"


def test_custom_named_metric_preserves_unknown_protocol_and_checks_evaluator(tmp_path):
    contract = {"primary_metric": {"name": "research_metric", "research_protocol": {"matching": "one_to_one", "weights": [1, 2]}}}
    manifest_at(tmp_path, contract)
    canonical = canonical_metric_contract(contract)
    assert canonical["research_protocol"] == contract["primary_metric"]["research_protocol"]
    validate_declared_capabilities({"metric_contract": contract}, "")
    evaluator_setup._validate_custom_metric_contract(canonical, tmp_path, "")
    canonical["research_protocol"]["matching"] = "per_gt_best_iou"
    with pytest.raises(evaluator_setup.EvaluatorSetupError, match="research_protocol"):
        evaluator_setup._validate_custom_metric_contract(canonical, tmp_path, "")


def test_custom_protocol_names_are_not_reinterpreted_as_registered_parameters():
    settings = {"decision_threshold": 0.3, "threshold": 0.4, "aggregation": "custom",
                "restoration_evaluation": {"aggregation": "other", "ssim_parameters": {"window_size": 11}},
                "ssim_parameters": {"window_size": 13}}
    assert canonical_metric_contract({"primary_metric": {"name": "research_metric", **settings}}) == {
        "primary_metric": "research_metric", **settings,
    }


def test_task_metric_can_supply_alias_context_without_becoming_a_new_declaration():
    assert canonical_metric_contract({"decision_threshold": 0.3}, fallback_primary="auc") == {"threshold": 0.3}
    validate_declared_capabilities({"metric_contract": {"decision_threshold": 0.3}}, "Primary metric: AUC.")


def test_custom_extension_keeps_task_metric_context_after_protocol_block(tmp_path):
    manifest_at(tmp_path, {"secondary_metrics": ["research_metric"]})
    task = 'Primary metric: SSIM.\n```restoration-evaluation\n{"data_range": 1, "ssim_parameters": {"window_size": 7}}\n```'
    evaluator_setup._validate_custom_metric_contract(
        {"primary_metric": "ssim", "secondary_metrics": ["research_metric"], "ssim_parameters": {"win_size": 7},
         "restoration_evaluation": {"data_range": 1, "crop_size": None, "groups": None, "aggregation": "mean_images"}},
        tmp_path, task,
    )
