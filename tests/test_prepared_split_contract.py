"""Preparation freezes a declared role set; missing artifacts cannot change it."""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from biomaker.utils.agentic_data_planner import parse_split_constraints, verify_split_package
from biomaker.utils.split_contract import (
    SPLIT_TO_FILE, SplitContractError, normalize_split_manifest, resolve_public_splits,
)
from biomaker.utils import agentic_data_planner, split_protocol


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    for role, relative in SPLIT_TO_FILE.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"id": role, **({"answer": 1} if role == "train" else {})}]).to_csv(path, index=False)
    private = root / "private_gt"
    private.mkdir()
    pd.DataFrame([{"id": "score_val", "answer": 2}]).to_csv(private / "score_val.csv", index=False)
    write_json(private / "private_gt.json", {"score_val": {"answer_columns": ["answer"]}})
    write_json(root / "split_manifest.json", {
        "manifest_version": 1, "sample_unit": "row", "key_columns": ["id"],
        "task_heads": ["regression"], "prediction_contract": {"prediction_columns": ["estimate"]},
        "metric_contract": {"metrics": [{"name": "mae", "maximize": False}]},
    })
    return root


def update_manifest(package, **changes):
    path = package / "split_manifest.json"
    value = json.loads(path.read_text())
    value.update(changes)
    write_json(path, value)
    return value


def exact_task(roles):
    return "```split-constraints\n" + json.dumps({"splits": roles}) + "\n```"


@pytest.mark.parametrize("declaration", [
    {"public_splits": ["train", "score_val"]},
    {"public_splits": {role: SPLIT_TO_FILE[role] for role in ("train", "score_val")}},
    {"required_splits": ["train", "score_val"]},
])
def test_legacy_exact_forms_import_to_one_schema_without_changing_meaning(declaration):
    manifest = {
        **declaration,
        "splits": {role: {"public_rows": SPLIT_TO_FILE[role], "row_count": 7, "sha256": "original",
                          "private_answers": "/private/answers.csv"}
                   for role in ("train", "score_val")},
    }
    frozen = normalize_split_manifest(manifest)
    assert frozen["manifest_version"] == 2
    assert "public_splits" not in frozen and "required_splits" not in frozen
    assert resolve_public_splits(frozen) == {role: SPLIT_TO_FILE[role] for role in ("train", "score_val")}
    assert all(entry["row_count"] == 7 and entry["sha256"] == "original"
               and "public_rows" not in entry and "private_answers" not in entry
               for entry in frozen["splits"].values())
    assert normalize_split_manifest(frozen) == frozen


@pytest.mark.parametrize("manifest", [
    {"public_splits": ["train", "score_val"], "required_splits": ["train", "score_val", "test"]},
    {"public_splits": {"train": "other_train.csv", "score_val": SPLIT_TO_FILE["score_val"]}},
    {"public_splits": ["train", "score_val", "score_val"]},
    {"manifest_version": 2},
    {"manifest_version": 2, "splits": {"train": {}, "score_val": {}}},
    {"public_splits": ["train", "score_val"], "hidden_answers": {"test": {}}},
])
def test_ambiguous_or_inconsistent_declarations_fail_at_import(manifest):
    with pytest.raises(SplitContractError):
        resolve_public_splits(manifest)


def test_legacy_absence_uses_three_split_default_and_never_inspects_files():
    assert resolve_public_splits({}) == SPLIT_TO_FILE
    with pytest.raises(SplitContractError, match="complete public split set"):
        resolve_public_splits({}, require_explicit=True)


def test_missing_test_without_declaration_cannot_become_no_test_on_retry(package):
    (package / SPLIT_TO_FILE["test"]).unlink()
    original = (package / "split_manifest.json").read_bytes()
    for _ in range(2):
        result = verify_split_package(package)
        assert not result["ok"]
        assert any("Declared public split test is missing" in error for error in result["errors"])
        assert (package / "split_manifest.json").read_bytes() == original


@pytest.mark.parametrize("invalid", ["{broken", "[]"])
def test_corrupt_manifest_cannot_fall_back_to_a_task_declaration(package, invalid):
    path = package / "split_manifest.json"
    path.write_text(invalid)
    result = verify_split_package(package, task_text=exact_task(list(SPLIT_TO_FILE)))
    assert not result["ok"]
    assert any("Invalid split manifest" in error for error in result["errors"])
    assert path.read_text() == invalid


def test_conflicting_manifest_copies_are_rejected(package):
    write_json(package / "biomaker_splits/split_manifest.json", {"public_splits": ["train", "score_val"]})
    result = verify_split_package(package)
    assert not result["ok"]
    assert any("Conflicting root" in error for error in result["errors"])


def test_task_exact_roles_freeze_without_user_copying_fields_to_manifest(package):
    (package / SPLIT_TO_FILE["test"]).unlink()
    task = exact_task(["train", "score_val"])
    result = verify_split_package(package, task_text=task)
    assert result["ok"], result["errors"]
    frozen = json.loads((package / "split_manifest.json").read_text())
    assert frozen["manifest_version"] == 2 and set(frozen["splits"]) == {"train", "score_val"}
    for entry in frozen["splits"].values():
        assert entry["row_count"] == 1
        assert entry["sha256"] == hashlib.sha256((package / entry["file"]).read_bytes()).hexdigest()
    assert verify_split_package(package, task_text=task)["ok"]
    assert json.loads((package / "split_manifest.json").read_text()) == frozen


def test_minimum_task_requirement_allows_unlabeled_test(package):
    task = '```split-constraints\n{"required_splits":["train","score_val"]}\n```'
    result = verify_split_package(package, task_text=task)
    assert result["ok"], result["errors"]
    frozen = json.loads((package / "split_manifest.json").read_text())
    assert "test" in frozen["splits"] and "test" not in frozen["hidden_answers"]


def test_declared_test_file_loss_is_fatal_after_freezing(package):
    assert verify_split_package(package)["ok"]
    (package / SPLIT_TO_FILE["test"]).unlink()
    result = verify_split_package(package)
    assert not result["ok"]
    assert any("Declared public split test is missing" in error for error in result["errors"])


def test_exact_task_rejects_extra_files_and_conflicting_minimum(package):
    result = verify_split_package(package, task_text=exact_task(["train", "score_val"]))
    assert not result["ok"]
    assert any("test exists outside" in error for error in result["errors"])
    with pytest.raises(ValueError, match="conflicts"):
        parse_split_constraints('```split-constraints\n{"splits":["train","score_val"],"required_splits":["test"]}\n```')


def test_publication_and_cache_preserve_two_split_protocol(package, tmp_path, monkeypatch):
    (package / SPLIT_TO_FILE["test"]).unlink()
    update_manifest(package, public_splits={role: SPLIT_TO_FILE[role] for role in ("train", "score_val")})
    verified = verify_split_package(package)
    assert verified["ok"], verified["errors"]
    source = tmp_path / "run/input"
    source.mkdir(parents=True)
    monkeypatch.setattr(split_protocol, "run_agentic_split_builder_probe", lambda *args, **kwargs: SimpleNamespace(
        output_dir=package, split_dir=package / "biomaker_splits", private_gt_dir=package / "private_gt", verifier=verified,
    ))
    protocol = split_protocol.prepare_split_protocol(source, task_desc="Only train and validation.")
    assert set(protocol.split_plan["agent_visible"]) == {"train", "score_val"}
    assert "submission.csv" not in protocol.prompt_text
    write_json(package / "verifier.json", {"ok": True})
    copied = tmp_path / "copied"
    accepted = split_protocol._copy_split_package_from_cache(
        package, copied / "splits", copied / "private", source,
    )
    assert accepted and accepted["ok"]
    assert set(json.loads((copied / "splits/split_manifest.json").read_text())["splits"]) == {"train", "score_val"}


@pytest.mark.parametrize("changed_protocol", ["schema", "compiler"])
def test_changed_protocol_compiler_cannot_reuse_a_previously_admitted_cache(
    package, tmp_path, monkeypatch, changed_protocol,
):
    """An intact cache is reused only under the schema/compiler that admitted it."""
    source = tmp_path / "run/input"
    source.mkdir(parents=True)
    cache = tmp_path / "cache"
    builds = []

    def build(*args, **kwargs):
        builds.append(True)
        return SimpleNamespace(
            output_dir=package, split_dir=package / "biomaker_splits",
            private_gt_dir=package / "private_gt", verifier=verify_split_package(package),
        )

    monkeypatch.setattr(split_protocol, "run_agentic_split_builder_probe", build)
    initial = split_protocol.prepare_split_protocol(source, task_desc="Predict the answer.", cache_root=cache)
    assert not initial.split_plan["cache_reused"] and len(builds) == 1
    reused = split_protocol.prepare_split_protocol(source, task_desc="Predict the answer.", cache_root=cache)
    assert reused.split_plan["cache_reused"] and len(builds) == 1
    if changed_protocol == "schema":
        monkeypatch.setattr(agentic_data_planner, "MANIFEST_VERSION", agentic_data_planner.MANIFEST_VERSION + 1)
    else:
        digest = agentic_data_planner._file_digest
        monkeypatch.setattr(agentic_data_planner, "_file_digest", lambda path: (
            hashlib.sha256(b"different protocol compiler").hexdigest()
            if path.name == "split_contract.py" else digest(path)
        ))
    rebuilt = split_protocol.prepare_split_protocol(source, task_desc="Predict the answer.", cache_root=cache)
    assert not rebuilt.split_plan["cache_reused"] and len(builds) == 2
    assert len([entry for entry in cache.iterdir() if entry.is_dir()]) == 2
