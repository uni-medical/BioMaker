"""Invalid prepared metric declarations must reach the builder's repair loop."""

import copy
import json

import pandas as pd
import pytest

from biomaker.utils import agentic_data_planner as planner, evaluator_setup
from biomaker.utils.metric_declarations import validate_declared_capabilities
from biomaker.utils.split_protocol import _copy_split_package_from_cache


EMPTY_POLICY = "both empty: Dice/IoU/AJI 1; exactly one empty: 0"
SEGMENTATION = {
    "primary_metric": "aji", "secondary_metrics": ["binary_dice", "binary_iou"],
    "maximize": True, "aggregation": "mean_images", "foreground": "nonzero",
    "aji_matching": "per_gt_best_iou",
}


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "template"
    for directory, filename, rows in [
        ("biomaker_splits", "agent_train.csv", [{"sample": "train", "answer": 1}]),
        ("biomaker_splits", "score_val.csv", [{"sample": "val"}]),
        ("private_gt", "score_val.csv", [{"sample": "val", "answer": 1}]),
    ]:
        path = root / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    hidden = {"score_val": {"answer_columns": ["answer"]}}
    (root / "private_gt/private_gt.json").write_text(json.dumps(hidden))
    write_contract(root, SEGMENTATION)
    return root


def write_contract(root, contract):
    manifest = {
        "manifest_version": 1, "sample_unit": "image", "key_columns": ["sample"],
        "task_heads": ["segmentation_2d"],
        "prediction_contract": {"prediction_columns": ["predicted_mask"]},
        "metric_contract": contract,
    }
    (root / "split_manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("contract,expected", [
    ({**SEGMENTATION, "empty_policy": EMPTY_POLICY}, "Unsupported declared metric protocol fields: empty_policy"),
    ({**SEGMENTATION, "aggregation": "pooled_pixels"}, "Unsupported declared metric setting aggregation"),
    ({"primary_metric": "ap50", "iou_threshold": 0.75}, "Unsupported declared metric setting iou_threshold"),
    ({"primary_metric": "auc", "threshold": 1.2}, "threshold must be in [0,1]"),
    ({"primary_metric": "auc", "threshold": 0.4, "decision_threshold": 0.5}, "Conflicting classification threshold aliases"),
    ({"primary_metric": "mae", "maximize": True}, "maximize conflicts"),
    ({"primary_metric": "aji", "secondary_metrics": ["mae"]}, "different canonical sample families"),
    ({"primary_metric": "ssim", "ssim_parameters": {"win_size": 4}}, "win_size"),
    ({"primary_metric": "ssim", "ssim_parameters": {"unsupported": True}}, "Unsupported declared metric settings"),
    ({"primary_metric": "ssim", "ssim_parameters": []}, "must be an object"),
    ({"primary_metric": "ssim", "restoration_evaluation": []}, "must be an object"),
    ({"primary_metric": "aji", "secondary_metrics": "dice"}, "must be a list"),
    ({"primary_metric": "aji", "secondary_metrics": [None]}, "must be a list"),
    ({"primary_metric": {"unrecognized": "aji"}}, "nonempty metric name"),
    ({"primary_metric": 12}, "metric name or named object"),
    ({"primary_metric": None}, "metric name or named object"),
    ({"primary_metric": "   "}, "nonempty metric name"),
    ([], "metric_contract must be an object"),
])
def test_invalid_declarations_are_verifier_failures_and_remain_strict_for_evaluator(
    package, tmp_path, contract, expected,
):
    write_contract(package, contract)
    result = planner.verify_split_package(package)
    assert not result["ok"]
    assert any(expected in error for error in result["errors"]), result["errors"]
    # Neither gate rewrites unsupported semantics to force acceptance.
    assert json.loads((package / "split_manifest.json").read_text())["metric_contract"] == contract
    workspace = tmp_path / "workspace"
    split = workspace / "input/biomaker_splits"
    split.mkdir(parents=True)
    (split / "split_manifest.json").write_bytes((package / "split_manifest.json").read_bytes())
    with pytest.raises(evaluator_setup.EvaluatorSetupError) as failure:
        evaluator_setup._validate_declared_capabilities("", workspace)
    assert expected in str(failure.value)


def test_task_metric_conflict_reaches_preparation_feedback(package):
    result = planner.verify_split_package(package, task_text="Primary metric: binary_dice.")
    assert not result["ok"]
    assert any("disagree on the primary metric" in error for error in result["errors"])


@pytest.mark.parametrize("contract", [
    {**SEGMENTATION, "notes": EMPTY_POLICY},
    {"primary_metric": {"name": "mean_aji"}, "secondary_metrics": ["dice", "iou"]},
    {"primary_metric": "auc", "decision_threshold": 0.5},
    {"primary_metric": "ssim", "ssim_parameters": {"window_size": 7}},
    {"primary_metric": "custom_research_metric", "custom_protocol": {"setting": 1}},
])
def test_supported_partial_declarations_and_custom_extension_still_pass(package, contract):
    write_contract(package, contract)
    before = copy.deepcopy(contract)
    validate_declared_capabilities({"metric_contract": contract}, "")
    assert contract == before
    result = planner.verify_split_package(package)
    assert result["ok"], result["errors"]


@pytest.mark.parametrize("repair", [True, False])
def test_real_empty_policy_error_is_sent_to_next_builder_attempt(package, tmp_path, monkeypatch, repair):
    write_contract(package, {**SEGMENTATION, "empty_policy": EMPTY_POLICY})
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    task_file = tmp_path / "task.md"
    task_file.write_text("Primary metric: AJI.")
    failures = []

    def query(*args, previous_failure=None, **kwargs):
        failures.append(copy.deepcopy(previous_failure))
        source = (
            "import argparse, json, shutil\nfrom pathlib import Path\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--input-dir')\nparser.add_argument('--output-dir')\n"
            "out = Path(parser.parse_args().output_dir)\n"
            f"shutil.copytree({str(package)!r}, out, dirs_exist_ok=True)\n"
        )
        if previous_failure and repair:
            source += (
                "path = out / 'split_manifest.json'\n"
                "manifest = json.loads(path.read_text())\n"
                "contract = manifest['metric_contract']\n"
                "contract['notes'] = contract.pop('empty_policy')\n"
                "path.write_text(json.dumps(manifest))\n"
            )
        return source, "```python\n" + source + "\n```", {}

    monkeypatch.setattr(planner, "_query_builder", query)
    monkeypatch.setattr(planner, "split_builder_cache_key", lambda **kwargs: ("test", {}))
    output = tmp_path / "generated"
    if repair:
        result = planner.run_agentic_split_builder_probe(input_dir, task_file, output, max_attempts=2)
        assert result.verifier["ok"]
        assert [attempt["ok"] for attempt in result.plan["attempts"]] == [False, True]
        final = json.loads((output / "split_manifest.json").read_text())["metric_contract"]
        assert final == {**SEGMENTATION, "notes": EMPTY_POLICY}
    else:
        with pytest.raises(RuntimeError, match="Unsupported declared metric protocol fields: empty_policy"):
            planner.run_agentic_split_builder_probe(input_dir, task_file, output, max_attempts=2)
    assert len(failures) == 2 and failures[0] is None
    assert "empty_policy" in " ".join(failures[1]["errors"])


def test_cache_cannot_admit_unsupported_metric_declaration(package, tmp_path):
    write_contract(package, {**SEGMENTATION, "empty_policy": EMPTY_POLICY})
    (package / "verifier.json").write_text(json.dumps({"ok": True}))
    destination = tmp_path / "admitted"
    assert _copy_split_package_from_cache(
        package, destination / "splits", destination / "private", tmp_path,
        task_text="Primary metric: AJI.",
    ) is None
    assert not destination.exists()


def test_builder_prompt_separates_catalog_documentation_from_protocol_keys(monkeypatch):
    monkeypatch.setattr("biomaker.utils.environment.environment_manifest_text", lambda **kwargs: "test")
    fields = " ".join(planner._builder_user_prompt("", {})["Required split_manifest.json Fields"])
    assert "allowed_contract_fields" in fields
    assert "empty_policy" in fields and "documentation, not protocol fields" in fields
    assert "never remove or replace task-required settings" in fields
