"""Invalid prepared metric declarations must reach the builder's repair loop."""

import copy
import json
import hashlib

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
    ({"primary_metric": "auc", "threshold": 0.4, "decision_threshold": 0.5}, "Conflicting metric protocol field: threshold"),
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


def builder_source(package):
    return (
        "import argparse, shutil\nfrom pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--input-dir')\nparser.add_argument('--output-dir')\n"
        "out = Path(parser.parse_args().output_dir)\n"
        f"shutil.copytree({str(package)!r}, out, dirs_exist_ok=True)\n"
        "with (out / 'builder_executions.txt').open('a') as f: f.write('run\\n')\n"
    )


def data_fingerprints(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("biomaker_splits", "private_gt", "public")
        for path in (root / folder).rglob("*")
        if path.is_file() and path.name != "split_manifest.json"
    }


@pytest.mark.parametrize("repair", [True, False])
@pytest.mark.parametrize("initial", [{**SEGMENTATION, "empty_policy": EMPTY_POLICY}, []])
@pytest.mark.parametrize("first_response", ["not JSON", json.dumps({**SEGMENTATION, "aji_matching": "one_to_one"})])
def test_declaration_repair_preserves_data_and_retries_without_builder(
    package, tmp_path, monkeypatch, repair, initial, first_response,
):
    write_contract(package, initial)
    # Exercise both supported manifest locations and non-CSV prepared artifacts.
    (package / "biomaker_splits/split_manifest.json").write_bytes((package / "split_manifest.json").read_bytes())
    (package / "public").mkdir()
    (package / "public/converted.bin").write_bytes(b"prepared image")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    task_file = tmp_path / "task.md"
    task_file.write_text("Primary metric: AJI.")
    output = tmp_path / "generated"
    builder_calls, failures = [], []
    originals = {}

    def query(*args, previous_failure=None, **kwargs):
        builder_calls.append(previous_failure)
        return builder_source(package), "builder response", {}

    def repair_query(task, manifest, errors, previous_response=None, **kwargs):
        assert data_fingerprints(output) == data_fingerprints(package)
        assert (output / "builder_executions.txt").read_text() == "run\n"
        for relative in ("split_manifest.json", "biomaker_splits/split_manifest.json"):
            current = (output / relative).read_bytes()
            originals.setdefault(relative, current)
            assert current == originals[relative]  # rejected proposals never reach disk
        failures.append(errors)
        if len(failures) == 1:
            assert previous_response is None
            return first_response, {}
        assert previous_response == first_response
        assert "Invalid metric_contract" in errors[0]
        result = {**SEGMENTATION, "notes": EMPTY_POLICY} if repair else initial
        return json.dumps(result), {}

    monkeypatch.setattr(planner, "_query_builder", query)
    monkeypatch.setattr(planner, "_query_metric_contract", repair_query)
    monkeypatch.setattr(planner, "split_builder_cache_key", lambda **kwargs: ("test", {}))
    if repair:
        result = planner.run_agentic_split_builder_probe(input_dir, task_file, output, max_attempts=3)
        assert result.verifier["ok"]
        assert [attempt["ok"] for attempt in result.plan["attempts"]] == [False, False, True]
        assert [attempt["stage"] for attempt in result.plan["attempts"]] == ["data_builder", "metric_contract", "metric_contract"]
        final = json.loads((output / "split_manifest.json").read_text())
        assert final["metric_contract"] == {**SEGMENTATION, "notes": EMPTY_POLICY}
        original = json.loads(originals["split_manifest.json"])
        assert {k: v for k, v in final.items() if k != "metric_contract"} == {k: v for k, v in original.items() if k != "metric_contract"}
        assert (output / "split_manifest.json").read_bytes() == (output / "biomaker_splits/split_manifest.json").read_bytes()
    else:
        with pytest.raises(RuntimeError, match="Invalid metric_contract"):
            planner.run_agentic_split_builder_probe(input_dir, task_file, output, max_attempts=3)
        for relative, content in originals.items():
            assert (output / relative).read_bytes() == content
    assert builder_calls == [None]
    assert len(failures) == 2
    assert data_fingerprints(output) == data_fingerprints(package)


@pytest.mark.parametrize("failure", ["data", "exit"])
def test_data_or_execution_failure_still_reaches_builder(package, tmp_path, monkeypatch, failure):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    task = tmp_path / "task.md"
    task.write_text("Primary metric: AJI.")
    calls = []

    def query(*args, previous_failure=None, **kwargs):
        calls.append(previous_failure)
        code = builder_source(package)
        if len(calls) == 1:
            code += ("(out / 'private_gt/score_val.csv').unlink()\n" if failure == "data" else "raise SystemExit(1)\n")
        return code, "builder response", {}

    monkeypatch.setattr(planner, "_query_builder", query)
    monkeypatch.setattr(planner, "_query_metric_contract", lambda *a, **kw: pytest.fail("Data failures need the builder"))
    monkeypatch.setattr(planner, "split_builder_cache_key", lambda **kwargs: ("test", {}))
    result = planner.run_agentic_split_builder_probe(input_dir, task, tmp_path / "out", max_attempts=2)
    assert result.verifier["ok"]
    assert len(calls) == 2 and not calls[1]["ok"]
    assert [a["stage"] for a in result.plan["attempts"]] == ["data_builder", "data_builder"]


@pytest.mark.parametrize("response", [json.dumps(SEGMENTATION), "Here is the correction:\n```json\n" + json.dumps(SEGMENTATION) + "\n```"])
def test_contract_response_accepts_json_with_optional_fence(response):
    assert planner._parse_metric_contract_response(response) == SEGMENTATION


def test_repair_prompt_contains_task_data_semantics_and_diagnostics(monkeypatch):
    captured = {}
    def query(**kwargs):
        captured.update(kwargs)
        return json.dumps(SEGMENTATION), {}
    monkeypatch.setattr(planner.backend, "query", query)
    manifest = {"metric_contract": SEGMENTATION, "data_semantics": {"axes": "YX"}, "prediction_contract": {"prediction_columns": ["my_mask"]}}
    planner._query_metric_contract("Task-owned definition", manifest, ["specific error"], model="test", temperature=0, previous_response="rejected proposal")
    prompt = captured["user_message"]
    assert prompt["Task Description"] == "Task-owned definition"
    assert prompt["Prepared Data Contract"] == manifest
    assert prompt["Declaration Errors"] == ["specific error"]
    assert prompt["Previous Repair Response"] == "rejected proposal"
    scope = " ".join(prompt["Repair Scope"])
    assert "do not silently drop" in scope and "custom definition" in scope


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
    assert "documentation section" in fields and "metric_contract_example" in fields
    assert "Never remove or replace task-required settings" in fields
