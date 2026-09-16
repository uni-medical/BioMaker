from __future__ import annotations

import json
import shutil

import pytest

from biomaker.utils.split_protocol import _make_private_paths_evaluator_visible
from scripts import run_fig2_medmnist_queue as queue


def _private_fixture(tmp_path):
    source = tmp_path / "cache" / "private_gt"
    target = tmp_path / "run" / "private_gt"
    (source / "masks" / "test").mkdir(parents=True)
    (source / "masks" / "test" / "case.png").write_bytes(b"original mask")
    (source / "test.csv").write_text(
        "Id,mask_path,class\ncase,private_gt/masks/test/case.png,1\n",
        encoding="utf-8",
    )
    (source / "private_gt.json").write_text(
        json.dumps({"test": {"file": "test.csv", "answer_columns": ["mask_path"]}}),
        encoding="utf-8",
    )
    _cache, frozen_artifacts = queue._split_cache_artifacts(
        {"cache_entry": str(source.parent)}
    )
    source_record = frozen_artifacts["private_gt/test.csv"]
    shutil.copytree(source, target)
    input_dir = tmp_path / "workspace" / "input"
    input_dir.mkdir(parents=True)
    # Exercise the exact runtime transformation that used to break raw CSV hashes.
    _make_private_paths_evaluator_visible(target, input_dir)
    return source, target, source_record, frozen_artifacts


def _audit(target, source_record, frozen_artifacts):
    errors = []
    queue._verify_private_csv_record(
        source_record,
        frozen_artifacts,
        "private test",
        errors,
        actual_path=target / "test.csv",
    )
    return errors


def test_private_mask_paths_can_relocate_without_changing_answers(tmp_path):
    source, target, record, inventory = _private_fixture(tmp_path)
    assert not (source.parent / "artifact_integrity.json").exists()
    assert (source / "test.csv").read_bytes() != (target / "test.csv").read_bytes()
    assert _audit(target, record, inventory) == []


@pytest.mark.parametrize("change", ["label", "mask", "both_masks", "target", "escape"])
def test_private_path_normalization_does_not_accept_changed_answers(tmp_path, change):
    source, target, record, inventory = _private_fixture(tmp_path)
    csv_path = target / "test.csv"
    text = csv_path.read_text(encoding="utf-8")
    if change == "label":
        csv_path.write_text(text.replace(",1\n", ",0\n"), encoding="utf-8")
    elif change in {"mask", "both_masks"}:
        (target / "masks" / "test" / "case.png").write_bytes(b"changed mask")
        if change == "both_masks":
            (source / "masks" / "test" / "case.png").write_bytes(b"changed mask")
    elif change == "target":
        alternate = target / "masks" / "test" / "alternate.png"
        alternate.write_bytes(b"original mask")
        csv_path.write_text(text.replace("case.png", "alternate.png"), encoding="utf-8")
    else:
        source_file = source / "masks" / "test" / "case.png"
        target_file = target / "masks" / "test" / "case.png"
        csv_path.write_text(
            text.replace(str(target_file), str(source_file)), encoding="utf-8"
        )
    assert _audit(target, record, inventory)


def test_scalar_private_labels_still_require_exact_values(tmp_path):
    source, target, _record, inventory = _private_fixture(tmp_path)
    content = "Id,class\ncase,1\n"
    (source / "test.csv").write_text(content, encoding="utf-8")
    (target / "test.csv").write_text(content, encoding="utf-8")
    record = queue.file_record(source / "test.csv")
    assert _audit(target, record, {}) == []
    (target / "test.csv").write_text("Id,class\ncase,0\n", encoding="utf-8")
    assert _audit(target, record, {})


def test_private_masks_require_frozen_source_hashes(tmp_path):
    _source, target, record, inventory = _private_fixture(tmp_path)
    inventory.pop("private_gt/masks/test/case.png")
    assert _audit(target, record, inventory)
