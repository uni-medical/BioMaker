"""Relocation follows declared file roles and retains copied private answers."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from test_task_split_constraints import TASK, split_package, write_json
from biomaker.utils.split_protocol import prepare_split_protocol
from biomaker.utils.verified_split_package import (
    seal_split_package,
    verify_package_identity,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("linked", [False, True], ids=["copied-input", "linked-input"])
@pytest.mark.parametrize(
    "keys", ["numeric", "path"], ids=["leading-zero-ids", "path-like-ids"]
)
def test_verified_package_keeps_identifiers_inline_values_and_private_answer_bytes(
    split_package, tmp_path, monkeypatch, linked, keys
):
    source, package = split_package
    (source / "masks").mkdir()
    (source / "masks/inline.png").write_bytes(
        b"not a file reference: inline class name"
    )
    original_rows = {}
    for index, (role, filename) in enumerate(
        (("train", "agent_train"), ("score_val", "score_val"), ("test", "agent_test"))
    ):
        public_path = package / f"biomaker_splits/{filename}.csv"
        frame = pd.read_csv(public_path, dtype=str, keep_default_na=False)
        if keys == "numeric":
            frame["case_uid"] = [f"00{index}{row}" for row in range(len(frame))]
        truth = frame.copy()
        truth["months"] = "030"
        truth["inline_label"] = "masks/inline.png"
        masks = []
        for row in range(len(frame)):
            mask = source / f"masks/{role}_{row}.png"
            mask.write_bytes(f"answer bytes: {role}-{row}".encode())
            masks.append("input/" + mask.relative_to(source).as_posix())
        truth["mask"] = masks
        (truth if role == "train" else frame).to_csv(public_path, index=False)
        if role != "train":
            truth.to_csv(package / f"private_gt/{role}.csv", index=False)
        original_rows[role] = truth
    hidden = {
        role: {
            "answer_columns": ["months", "inline_label", "mask"],
            "file_reference_columns": ["mask"],
        }
        for role in ("score_val", "test")
    }
    write_json(package / "private_gt/private_gt.json", hidden)
    manifest = json.loads((package / "split_manifest.json").read_text())
    manifest["hidden_answers"] = hidden
    write_json(package / "split_manifest.json", manifest)
    seal_split_package(package, source, task_text=TASK)
    source_before = {
        path.relative_to(source).as_posix(): _digest(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    package_before = {
        path.relative_to(package).as_posix(): _digest(path)
        for path in package.rglob("*")
        if path.is_file()
    }
    target = tmp_path / "another-machine/workspace/input"
    if linked:
        target.mkdir(parents=True)
        for child in source.iterdir():
            (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
    else:
        shutil.copytree(source, target)
    private = tmp_path / "another-machine/private_gt"
    monkeypatch.setattr(
        "biomaker.utils.split_protocol.run_agentic_split_builder_probe",
        lambda *_a, **_k: pytest.fail("Explicit packages must not invoke a provider"),
    )
    prepare_split_protocol(target, private, task_desc=TASK, package_dir=package)
    for role, filename in (
        ("train", "agent_train"),
        ("score_val", "score_val"),
        ("test", "agent_test"),
    ):
        public = pd.read_csv(
            target / f"biomaker_splits/{filename}.csv", dtype=str, keep_default_na=False
        )
        assert public["case_uid"].tolist() == original_rows[role]["case_uid"].tolist()
        for relative in public["arbitrary_source"]:
            assert (target / Path(relative).relative_to("input")).is_file()
        if role == "train":
            assert (
                public["inline_label"].tolist()
                == original_rows[role]["inline_label"].tolist()
            )
            assert public["months"].tolist() == ["030"] * len(public)
            for relative in public["mask"]:
                assert (target / Path(relative).relative_to("input")).is_file()
        else:
            assert "mask" not in public and "inline_label" not in public
            truth = pd.read_csv(
                private / f"{role}.csv", dtype=str, keep_default_na=False
            )
            assert (
                truth["case_uid"].tolist() == original_rows[role]["case_uid"].tolist()
            )
            assert (
                truth["inline_label"].tolist()
                == original_rows[role]["inline_label"].tolist()
            )
            assert truth["months"].tolist() == ["030"] * len(truth)
            for retained, relative in zip(truth["mask"], original_rows[role]["mask"]):
                assert Path(retained).is_relative_to(private / "answer_files")
                original = source / Path(relative).relative_to("input")
                assert _digest(Path(retained)) == _digest(original)
                assert not (target / Path(relative).relative_to("input")).exists()
    assert not (target / "masks/inline.png").exists()
    assert {
        path.relative_to(source).as_posix(): _digest(path)
        for path in source.rglob("*")
        if path.is_file()
    } == source_before
    assert {
        path.relative_to(package).as_posix(): _digest(path)
        for path in package.rglob("*")
        if path.is_file()
    } == package_before
    assert verify_package_identity(package, source, task_text=TASK)["version"] == 1


def test_private_copy_mismatch_does_not_rewrite_table_or_remove_source(
    tmp_path, monkeypatch
):
    from biomaker.utils import split_protocol as protocol

    source, private = tmp_path / "input", tmp_path / "private"
    source.mkdir()
    private.mkdir()
    answer = source / "answer.png"
    answer.write_bytes(b"original private bytes")
    table = private / "score_val.csv"
    table.write_text("id,mask\n0001,input/answer.png\n")
    write_json(
        private / "private_gt.json",
        {"score_val": {"answer_columns": ["mask"], "file_reference_columns": ["mask"]}},
    )
    before = table.read_bytes()
    monkeypatch.setattr(
        protocol.shutil,
        "copy2",
        lambda src, dst: Path(dst).write_bytes(b"changed during copy"),
    )
    with pytest.raises(ValueError, match="changed while being retained"):
        protocol._make_private_paths_evaluator_visible(private, source)
    assert table.read_bytes() == before
    assert answer.read_bytes() == b"original private bytes"
    assert not list(private.rglob(".answer-*"))


def test_declared_extensionless_input_is_kept_without_rewriting_opaque_key(tmp_path):
    from biomaker.utils import split_protocol as protocol

    source = tmp_path / "input"
    split = source / "biomaker_splits"
    split.mkdir(parents=True)
    (source / "blob").write_bytes(b"image input")
    (source / "not-a-reference.png").write_bytes(b"not public")
    write_json(
        split / "split_manifest.json",
        {"input_file_reference_columns": ["scan"], "hidden_answers": {}},
    )
    (split / "agent_train.csv").write_text(
        "id,scan,label\nnot-a-reference.png,blob,NA\n"
    )
    protocol._normalize_split_paths_in_dir(split, source)
    protocol.sanitize_agent_visible_input(source, split)
    row = pd.read_csv(split / "agent_train.csv", dtype=str, keep_default_na=False).iloc[
        0
    ]
    assert row.to_dict() == {
        "id": "not-a-reference.png",
        "scan": "input/blob",
        "label": "NA",
    }
    assert (source / "blob").read_bytes() == b"image input"
    assert not (source / "not-a-reference.png").exists()


def test_private_output_cannot_overlap_agent_input(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "preserved").write_text("source")
    with pytest.raises(ValueError, match="outside and separate"):
        prepare_split_protocol(source, source / "private_gt", task_desc="synthetic")
    assert (source / "preserved").read_text() == "source"
