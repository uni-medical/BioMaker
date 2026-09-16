from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_fig2_medmnist import prepare_task
from scripts.taskpack_integrity import INTEGRITY_FILE, verify_package_integrity


def _prepared_task(tmp_path):
    source = tmp_path / "breastmnist.npz"
    arrays = {}
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        arrays[f"{split}_images"] = np.zeros((count, 28, 28), dtype=np.uint8)
        arrays[f"{split}_labels"] = (np.arange(count) % 2).reshape(-1, 1)
    np.savez(source, **arrays)
    task = tmp_path / "task.md"
    task.write_text("Classify images. Primary metric: accuracy.\n", encoding="utf-8")
    options = {
        "task_desc_path": task,
        "output_root": tmp_path / "prepared",
        "cache_root": tmp_path / "cache",
        "model": "test-model",
        "strict_official": False,
    }
    return source, options, prepare_task(source, **options)


@pytest.mark.parametrize("target", ["image", "cache_label", "package_label", "legacy"])
def test_ready_reuse_rejects_changed_or_unverifiable_artifacts(tmp_path, target):
    source, options, first = _prepared_task(tmp_path)
    data = Path(first["data_dir"])
    cache = Path(first["cache_entry"])
    package = Path(first["package_dir"])
    suite = options["output_root"] / "suite_manifest.json"
    original_suite = suite.read_bytes()
    if target == "image":
        image = np.load(data / "arrays/train_images.npy", mmap_mode="r+")
        image[0, 0, 0] = 255
        image.flush()
        del image
    elif target == "legacy":
        (cache / INTEGRITY_FILE).unlink()
    else:
        root = cache if target == "cache_label" else package
        labels = root / "private_gt/score_val.csv"
        original = labels.read_bytes()
        changed = original.replace(b",0\r\n", b",1\r\n", 1)
        assert changed != original and len(changed) == len(original)
        labels.write_bytes(changed)

    with pytest.raises(RuntimeError, match="integrity"):
        prepare_task(source, **options)
    assert suite.read_bytes() == original_suite

    rebuilt = prepare_task(source, **options, replace=True)
    assert rebuilt["cache_key"] == first["cache_key"]
    assert prepare_task(source, **options)["reused"] is True


def test_integrity_inventory_is_portable_and_outside_agent_visible_data(tmp_path):
    source, options, first = _prepared_task(tmp_path)
    data = Path(first["data_dir"])
    package = Path(first["package_dir"])
    assert not (data / INTEGRITY_FILE).exists()
    inventory = json.loads((package / INTEGRITY_FILE).read_text(encoding="utf-8"))
    assert all(
        not Path(path).is_absolute()
        for kind in ("data", "package")
        for path in inventory[kind]
    )
    relocated = tmp_path / "another-machine"
    shutil.copytree(data, relocated / "data")
    shutil.copytree(package, relocated / "package")
    verify_package_integrity(relocated / "package", relocated / "data")
    reused = prepare_task(source, **options)
    assert reused["reused"] is True
    assert reused["cache_key"] == first["cache_key"]
