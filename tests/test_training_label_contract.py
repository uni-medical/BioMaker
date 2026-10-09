from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from biomaker.utils import agentic_data_planner as planner
from biomaker.utils import split_protocol as protocol


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _package(root: Path, targets=("class",), *, drop=()) -> None:
    for split, name in (("train", "agent_train"), ("val", "score_val"), ("test", "agent_test")):
        rows = [{"split": split, "Id": index, "feature": index / 10} for index in range(2)]
        answers = [dict(row, **{target: index % 2 for target in targets}) for index, row in enumerate(rows)]
        public = answers if split == "train" else rows
        public = [{k: v for k, v in row.items() if k not in drop} for row in public]
        _write_csv(root / "biomaker_splits" / f"{name}.csv", public)
        if split != "train":
            _write_csv(root / "private_gt" / ("score_val.csv" if split == "val" else "test.csv"), answers)
    (root / "private_gt/private_gt.json").write_text(json.dumps({
        "score_val": {"answer_columns": list(targets)}, "test": {"answer_columns": list(targets)},
    }))


@pytest.mark.parametrize("targets,drop", [
    (("class",), ("class",)),
    (("disease_a", "disease_b"), ("disease_b",)),
    (("mask",), ("mask",)),
    (("time", "event"), ("event",)),
])
def test_training_targets_cannot_be_dropped_while_held_out_answers_remain_private(tmp_path, targets, drop):
    _package(tmp_path, targets, drop=drop)
    path = tmp_path / "biomaker_splits/agent_train.csv"
    before = path.read_bytes()
    result = planner.verify_split_package(tmp_path)
    assert result["ok"] is False
    assert any("missing required training answer columns" in error for error in result["errors"])
    assert path.read_bytes() == before  # Admission must not invent or backfill labels.
    for name in ("score_val", "agent_test"):
        assert not set(targets) & set(pd.read_csv(tmp_path / f"biomaker_splits/{name}.csv").columns)


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_partial_missing_training_targets_fail_before_training(tmp_path, missing):
    _package(tmp_path)
    path = tmp_path / "biomaker_splits/agent_train.csv"
    frame = pd.read_csv(path)
    frame["class"] = pd.Series([0, missing], dtype=object)
    frame.to_csv(path, index=False)
    result = planner.verify_split_package(tmp_path)
    assert result["ok"] is False
    assert any("1 missing/blank rows" in error for error in result["errors"])


@pytest.mark.parametrize("targets", [("class",), ("disease_a", "disease_b"), ("time", "event")])
def test_complete_training_supervision_passes_with_zero_valued_answers(tmp_path, targets):
    _package(tmp_path, targets)
    result = planner.verify_split_package(tmp_path)
    assert result["ok"] is True, result["errors"]
    assert result["training_answer_columns"] == sorted(targets)


def test_cache_revalidates_training_columns_instead_of_trusting_old_ok(tmp_path):
    cache = tmp_path / "cache"
    _package(cache, drop=("class",))
    (cache / "verifier.json").write_text(json.dumps({"ok": True}))
    source = tmp_path / "input"
    source.mkdir()
    result = protocol._copy_split_package_from_cache(cache, source / "biomaker_splits", tmp_path / "private", source)
    assert result is None
    assert not (source / "biomaker_splits").exists()
    assert not (tmp_path / "private").exists()
    assert not (cache / "split_manifest.json").exists()  # Read-only cache validation.


def test_valid_cache_is_readmitted_with_live_training_evidence(tmp_path):
    cache = tmp_path / "cache"
    _package(cache)
    (cache / "verifier.json").write_text(json.dumps({"ok": True, "training_answer_columns": ["stale"]}))
    source = tmp_path / "input"
    source.mkdir()
    result = protocol._copy_split_package_from_cache(cache, source / "biomaker_splits", tmp_path / "private", source)
    assert result["ok"] is True
    assert result["training_answer_columns"] == ["class"]
    assert pd.read_csv(source / "biomaker_splits/agent_train.csv")["class"].tolist() == [0, 1]


def test_cached_segmentation_training_masks_remain_visible_and_private_masks_stay_hidden(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset/masks"
    dataset.mkdir(parents=True)
    source = tmp_path / "workspace/input"
    source.mkdir(parents=True)
    (source / "masks").symlink_to(dataset, target_is_directory=True)
    cache_root = tmp_path / "cache"
    cache = cache_root / "valid-cache"
    _package(cache, ("mask",))
    for split, relative in (("train", "biomaker_splits/agent_train.csv"), ("val", "private_gt/score_val.csv"), ("test", "private_gt/test.csv")):
        path = cache / relative
        frame = pd.read_csv(path)
        frame["mask"] = [f"input/masks/{split}_{index}.png" for index in range(2)]
        frame.to_csv(path, index=False)
        for index in range(2):
            Image.new("L", (3, 3), color=index).save(dataset / f"{split}_{index}.png")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in dataset.iterdir()}
    (cache / "verifier.json").write_text(json.dumps({"ok": True}))
    monkeypatch.setattr(protocol, "split_builder_cache_key", lambda **kw: ("valid-cache", {}))
    def no_builder(*args, **kwargs):
        raise AssertionError("A valid current cache should be reused")
    monkeypatch.setattr(protocol, "run_agentic_split_builder_probe", no_builder)
    prepared = protocol.prepare_split_protocol(source, tmp_path / "run/private_gt", task_desc="supervised segmentation", cache_root=cache_root)
    assert prepared.split_plan["cache_reused"] is True
    assert pd.read_csv(source / "biomaker_splits/agent_train.csv")["mask"].tolist() == ["input/masks/train_0.png", "input/masks/train_1.png"]
    for index in range(2):
        assert (source / f"masks/train_{index}.png").is_file()
        assert not (source / f"masks/val_{index}.png").exists()
        assert not (source / f"masks/test_{index}.png").exists()
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in dataset.iterdir()} == before
    assert not (cache / "split_manifest.json").exists()


def test_split_builder_retries_missing_training_labels_using_source_answers(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _write_csv(source / "train.csv", [{"Id": 0, "feature": 0.0, "class": 0}, {"Id": 1, "feature": 0.1, "class": 1}])
    original_sha = hashlib.sha256((source / "train.csv").read_bytes()).hexdigest()
    task = tmp_path / "task.txt"
    task.write_text("Fit a supervised classifier using all training labels.")
    code = '''import argparse, json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--input-dir'); p.add_argument('--output-dir'); a=p.parse_args()
out=Path(a.output_dir); (out/'biomaker_splits').mkdir(exist_ok=True); (out/'private_gt').mkdir(exist_ok=True)
train=pd.read_csv(Path(a.input_dir)/'train.csv'); train.insert(0,'split','train')
train__DROP__.to_csv(out/'biomaker_splits/agent_train.csv',index=False)
for split,name in [('val','score_val'),('test','test')]:
 frame=train.copy(); frame['split']=split
 frame.to_csv(out/'private_gt'/f'{name}.csv',index=False)
 frame.drop(columns=['class']).to_csv(out/'biomaker_splits'/('score_val.csv' if split=='val' else 'agent_test.csv'),index=False)
(out/'private_gt/private_gt.json').write_text(json.dumps({name:{'answer_columns':['class']} for name in ['score_val','test']}))
'''
    calls = []
    def query(*args, **kwargs):
        calls.append(kwargs)
        drop = ".drop(columns=['class'])" if len(calls) == 1 else ""
        return code.replace("__DROP__", drop), "synthetic local builder", {}
    monkeypatch.setattr(planner, "_query_builder", query)
    result = planner.run_agentic_split_builder_probe(source, task, tmp_path / "package", max_attempts=2)
    assert [attempt["ok"] for attempt in result.plan["attempts"]] == [False, True]
    assert "missing required training answer columns" in "\n".join(calls[1]["previous_failure"]["errors"])
    assert pd.read_csv(result.split_dir / "agent_train.csv")["class"].tolist() == [0, 1]
    assert hashlib.sha256((source / "train.csv").read_bytes()).hexdigest() == original_sha


@pytest.mark.parametrize("corrupt_after_sanitize", [False, True])
def test_materialized_training_labels_survive_sanitization_or_preparation_fails(tmp_path, monkeypatch, corrupt_after_sanitize):
    package = tmp_path / "package"
    _package(package)
    verifier = planner.verify_split_package(package)
    source = tmp_path / "workspace/input"
    source.mkdir(parents=True)
    _write_csv(source / "original_training_labels.csv", [{"class": 0}, {"class": 1}])
    result = SimpleNamespace(output_dir=package, split_dir=package / "biomaker_splits", private_gt_dir=package / "private_gt", verifier=verifier)
    monkeypatch.setattr(protocol, "run_agentic_split_builder_probe", lambda *a, **kw: result)
    original_sanitize = protocol.sanitize_agent_visible_input
    def sanitize(*args):
        summary = original_sanitize(*args)
        if corrupt_after_sanitize:
            path = source / "biomaker_splits/agent_train.csv"
            pd.read_csv(path).drop(columns=["class"]).to_csv(path, index=False)
        return summary
    monkeypatch.setattr(protocol, "sanitize_agent_visible_input", sanitize)
    if corrupt_after_sanitize:
        with pytest.raises(RuntimeError, match="missing required training answer columns"):
            protocol.prepare_split_protocol(source, tmp_path / "run/private_gt", task_desc="supervised classification")
    else:
        prepared = protocol.prepare_split_protocol(source, tmp_path / "run/private_gt", task_desc="supervised classification")
        assert pd.read_csv(source / "biomaker_splits/agent_train.csv")["class"].tolist() == [0, 1]
        assert not (source / "original_training_labels.csv").exists()
        assert "class" not in pd.read_csv(source / "biomaker_splits/score_val.csv").columns
        assert prepared.split_plan["hidden_answers"]["score_val"]["answer_columns"] == ["class"]
