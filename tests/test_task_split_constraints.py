"""The Harness enforces user-assigned file roles without choosing split ratios."""

import json
from pathlib import Path

import pandas as pd
import pytest

from biomaker.utils.agentic_data_planner import (
    _builder_user_prompt,
    parse_split_constraints,
    split_builder_cache_key,
    verify_split_package,
)
from biomaker.utils.split_protocol import _copy_split_package_from_cache


CONSTRAINTS = {
    "required_splits": ["train", "score_val", "test"],
    "required_answer_splits": ["score_val", "test"],
    "source_rules": [
        {"root": "train", "allowed_splits": ["train", "score_val"], "require_complete": True},
        {"root": "validation", "allowed_splits": ["test"], "require_complete": True},
    ],
}
TASK = "The official validation source is final private test.\n```split-constraints\n" + json.dumps(CONSTRAINTS) + "\n```"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def split_package(tmp_path):
    source, output = tmp_path / "source", tmp_path / "package"
    memberships = {"train": ["train/a.bin", "train/b.bin"],
                   "score_val": ["train/c.bin"],
                   "test": ["validation/d.bin", "validation/e.bin"]}
    for paths in memberships.values():
        for relative in paths:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
    filenames = {"train": "agent_train", "score_val": "score_val", "test": "agent_test"}
    for role, paths in memberships.items():
        rows = [{"case_uid": relative, "arbitrary_source": "input/" + relative} for relative in paths]
        truth = [dict(row, months=30) for row in rows]
        write_csv(output / "biomaker_splits" / (filenames[role] + ".csv"), truth if role == "train" else rows)
        if role != "train":
            write_csv(output / "private_gt" / (role + ".csv"), truth)
    hidden = {role: {"answer_columns": ["months"]} for role in ("score_val", "test")}
    write_json(output / "private_gt/private_gt.json", hidden)
    write_json(output / "split_manifest.json", {
        "manifest_version": 1, "sample_unit": "image", "key_columns": ["case_uid"],
        "task_heads": ["regression"], "hidden_answers": hidden,
        "prediction_contract": {"prediction_columns": ["age_prediction"], "file_reference_columns": []},
        "metric_contract": {"primary_metric": "mae", "maximize": False},
        "input_file_reference_columns": ["arbitrary_source"],
    })
    return source, output


def test_roles_and_complete_coverage_allow_arbitrary_keys_and_columns(split_package):
    source, output = split_package
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert result["ok"], result["errors"]
    assert result["split_constraints"]["constraints"] == CONSTRAINTS
    assert [rule["referenced_files"] for rule in result["split_constraints"]["source_rules"]] == [3, 2]


def test_official_validation_cannot_become_internal_score_and_omit_test(split_package):
    source, output = split_package
    for base in ["biomaker_splits", "private_gt"]:
        old = output / base / ("agent_test.csv" if base == "biomaker_splits" else "test.csv")
        old.replace(output / base / "score_val.csv")
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert not result["ok"]
    errors = "\n".join(result["errors"])
    assert "requires a nonempty public test" in errors
    assert "validation' allows only ['test']" in errors
    assert "complete original-file coverage" in errors  # internal train/c omitted


def test_coverage_checks_every_original_file_not_observation_sample(split_package):
    source, output = split_package
    (source / "validation/new_unobserved.bin").write_text("new source")
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert not result["ok"]
    assert any("missing 1 of 3" in error for error in result["errors"])


def test_original_source_cannot_cross_roles_by_changing_sample_ids(split_package):
    source, output = split_package
    row = {"case_uid": "new-id", "arbitrary_source": str((source / "train/a.bin").resolve())}
    for base in ["biomaker_splits", "private_gt"]:
        path = output / base / "score_val.csv"
        old = pd.read_csv(path).to_dict("records")
        write_csv(path, old + [dict(row, months=30) if base == "private_gt" else row])
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert not result["ok"]
    assert any("original files reused across public split roles" in error for error in result["errors"])


def test_required_private_test_answers_cannot_be_omitted(split_package):
    source, output = split_package
    (output / "private_gt/test.csv").unlink()
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert not result["ok"]
    assert any("requires public rows and private answers for test" in error for error in result["errors"])


def test_unconstrained_task_still_allows_unlabeled_test(split_package):
    source, output = split_package
    (output / "private_gt/test.csv").unlink()
    result = verify_split_package(output, source_input_dir=source)
    assert result["ok"], result["errors"]


def test_prediction_file_columns_cannot_refer_to_input_paths(split_package):
    source, output = split_package
    path = output / "split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["prediction_contract"]["file_reference_columns"] = ["arbitrary_source"]
    write_json(path, manifest)
    result = verify_split_package(output, source_input_dir=source)
    assert not result["ok"]
    assert any("must be a subset of prediction_columns" in error for error in result["errors"])


@pytest.mark.parametrize("change", [
    {"required_splits": ["validation"]}, {"required_answer_splits": ["train"]},
    {"source_rules": [{"root": "../outside", "allowed_splits": ["test"]}]},
    {"source_rules": [{"root": "train", "allowed_splits": []}]},
    {"source_rules": [{"root": "train", "allowed_splits": ["test"], "require_complete": "yes"}]},
    {"unknown": True},
])
def test_invalid_explicit_constraints_fail_before_generation(change):
    with pytest.raises(ValueError):
        parse_split_constraints("```split-constraints\n" + json.dumps(change) + "\n```")


def test_cache_rechecks_actual_files_and_task_instead_of_saved_ok(split_package, tmp_path):
    source, cached = split_package
    write_json(cached / "verifier.json", {"ok": True, "stale_marker": True})
    destination = tmp_path / "admitted"
    result = _copy_split_package_from_cache(cached, destination / "splits", destination / "private", source, task_text=TASK)
    assert result and result["ok"]
    assert "stale_marker" not in result
    assert result["split_constraints"]["source_rules"][1]["referenced_files"] == 2
    (source / "validation/new.bin").write_text("uncovered newly added file")
    rejected = tmp_path / "rejected"
    assert _copy_split_package_from_cache(cached, rejected / "splits", rejected / "private", source, task_text=TASK) is None
    assert not rejected.exists()


def test_builder_receives_constraints_and_actual_reader_environment(monkeypatch):
    monkeypatch.setattr("biomaker.utils.environment.environment_manifest_text", lambda **kwargs: "reader environment fixture")
    prompt = _builder_user_prompt(TASK, {})
    assert prompt["Task-owned Split Constraints"] == CONSTRAINTS
    assert prompt["Installed Reader Environment"] == "reader environment fixture"
    text = json.dumps(prompt)
    assert "data_semantics" in text and "input_file_reference_columns" in text
    assert "subset of prediction_contract.prediction_columns" in text
    assert "directory named validation may be reserved for final private test" in text


def test_rsna_task_preserves_explicit_source_roles():
    task = Path(__file__).resolve().parents[2] / "experiments/rsna_bone_age/task.md"
    if not task.is_file():
        pytest.skip("Workspace experiment declarations are separate from the method repository.")
    assert parse_split_constraints(task.read_text()) == CONSTRAINTS


def compact_private_answers(output):
    for role in ("score_val", "test"):
        path = output / "private_gt" / (role + ".csv")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame[["case_uid", "months"]].iloc[::-1].to_csv(path, index=False)


def test_private_truth_needs_keys_and_answers_only_in_any_keyed_order(split_package):
    source, output = split_package
    compact_private_answers(output)
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert result["ok"], result["errors"]
    for role in ("score_val", "test"):
        assert list(pd.read_csv(output / "private_gt" / (role + ".csv")).columns) == ["case_uid", "months"]


@pytest.mark.parametrize("problem,expected", [
    ("missing_key", "missing declared sample key"),
    ("different_keys", "sample keys do not match exactly"),
    ("duplicate_keys", "duplicate sample keys"),
    ("empty_keys", "empty sample keys"),
    ("different_row_count", "row mismatch"),
    ("unlisted_answer", "not listed in answer_columns"),
])
def test_compact_private_truth_preserves_alignment_and_hidden_schema(split_package, problem, expected):
    source, output = split_package
    compact_private_answers(output)
    path = output / "private_gt/test.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if problem == "missing_key":
        frame = frame.drop(columns="case_uid")
    elif problem == "different_keys":
        frame.loc[0, "case_uid"] = "not-the-public-sample"
    elif problem == "duplicate_keys":
        frame.loc[0, "case_uid"] = frame.loc[1, "case_uid"]
    elif problem == "empty_keys":
        frame.loc[0, "case_uid"] = ""
    elif problem == "different_row_count":
        frame = frame.iloc[:1]
    elif problem == "unlisted_answer":
        frame["hidden_extra"] = "answer"
    frame.to_csv(path, index=False)
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert not result["ok"]
    assert any(expected in error for error in result["errors"])


def test_key_alignment_preserves_leading_zeros_and_literal_na(split_package):
    source, output = split_package
    public_path, private_path = output / "biomaker_splits/agent_test.csv", output / "private_gt/test.csv"
    for path in (public_path, private_path):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame["case_uid"] = ["001", "NA"]
        frame.to_csv(path, index=False)
    assert verify_split_package(output, source_input_dir=source, task_text=TASK)["ok"]
    frame = pd.read_csv(private_path, dtype=str, keep_default_na=False)
    frame.loc[0, "case_uid"] = "1"
    frame.to_csv(private_path, index=False)
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert not result["ok"]
    assert any("sample keys do not match exactly" in error for error in result["errors"])


@pytest.mark.parametrize("explicit_order", [False, True])
def test_keyless_compact_answers_require_explicit_row_order(split_package, explicit_order):
    source, output = split_package
    path = output / "split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["key_columns"] = []
    manifest["row_order_contract"] = explicit_order
    write_json(path, manifest)
    for role in ("score_val", "test"):
        private_path = output / "private_gt" / (role + ".csv")
        pd.read_csv(private_path)[["months"]].to_csv(private_path, index=False)
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert result["ok"] is explicit_order
    if not explicit_order:
        assert any("explicit row_order_contract" in error for error in result["errors"])


def test_first_rsna_builder_shape_is_normalized_without_regenerating(split_package):
    # Mirrors the real first response: private files contain id+boneage only,
    # while prediction_columns unnecessarily repeats its declared ID column.
    source, output = split_package
    compact_private_answers(output)
    path = output / "split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["prediction_contract"]["prediction_columns"] = ["case_uid", "age_prediction"]
    write_json(path, manifest)
    write_json(output / "biomaker_splits/split_manifest.json", manifest)
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert result["ok"], result["errors"]
    normalized = json.loads(path.read_text())
    assert normalized["prediction_contract"]["prediction_columns"] == ["age_prediction"]
    assert normalized["normalization_warnings"] == result["warnings"]
    assert json.loads((output / "biomaker_splits/split_manifest.json").read_text()) == normalized
    repeated = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert repeated["ok"] and repeated["warnings"] == result["warnings"]


def test_identity_only_prediction_columns_still_fail(split_package):
    source, output = split_package
    path = output / "split_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["prediction_contract"]["prediction_columns"] = ["case_uid"]
    write_json(path, manifest)
    result = verify_split_package(output, source_input_dir=source, task_text=TASK)
    assert not result["ok"]
    assert any("at least one predicted value" in error for error in result["errors"])


def test_source_coverage_counts_hidden_answer_files_without_public_exposure(split_package):
    source, output = split_package
    compact_private_answers(output)
    hidden = json.loads((output / "private_gt/private_gt.json").read_text())
    for role in ("score_val", "test"):
        path = output / "private_gt" / (role + ".csv")
        frame = pd.read_csv(path, dtype=str)
        paths = []
        for index in range(len(frame)):
            mask = source / "answer_masks" / f"{role}_{index}.bin"
            mask.parent.mkdir(parents=True, exist_ok=True)
            mask.write_text("private target")
            paths.append("input/answer_masks/" + mask.name)
        frame["target_mask"] = paths
        frame.to_csv(path, index=False)
        hidden[role]["answer_columns"].append("target_mask")
    write_json(output / "private_gt/private_gt.json", hidden)
    contract = {**CONSTRAINTS, "source_rules": [*CONSTRAINTS["source_rules"], {
        "root": "answer_masks", "allowed_splits": ["score_val", "test"], "require_complete": True,
    }]}
    task = "```split-constraints\n" + json.dumps(contract) + "\n```"
    result = verify_split_package(output, source_input_dir=source, task_text=task)
    assert result["ok"], result["errors"]
    assert result["split_constraints"]["source_rules"][-1]["referenced_files"] == 3
    assert "target_mask" not in pd.read_csv(output / "biomaker_splits/agent_test.csv")


def test_builder_cache_fingerprint_includes_metric_capabilities(tmp_path, monkeypatch):
    monkeypatch.setattr("biomaker.utils.agentic_data_planner.build_dataset_facts", lambda *_args: {"unchanged": True})
    monkeypatch.setattr("biomaker.utils.trusted_metrics.metric_capabilities", lambda: {"metrics": ["original"]})
    original, _ = split_builder_cache_key(input_dir=tmp_path, task_text=TASK, model="same-model")
    monkeypatch.setattr("biomaker.utils.trusted_metrics.metric_capabilities", lambda: {"metrics": ["original", "new_metric"]})
    updated, _ = split_builder_cache_key(input_dir=tmp_path, task_text=TASK, model="same-model")
    assert updated != original
