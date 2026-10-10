from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from biomaker.utils.final_retrain import (
    _merge_visible_score_with_hidden_labels,
    prepare_final_retrain_workspace,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FinalRetrainTests(unittest.TestCase):
    def test_score_label_merge_allows_nullable_answer_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write_csv(
                workspace / "input" / "biomaker_splits" / "score_val.csv",
                [{"image_id": "val1", "image_path": "input/images/val1.jpg"}],
            )
            _write_json(
                workspace / "input" / "biomaker_splits" / "split_manifest.json",
                {
                    "key_columns": ["image_id"],
                    "hidden_answers": {
                        "score_val": {"answer_columns": ["mask_path", "bbox"]}
                    },
                },
            )
            train_df = pd.DataFrame(
                [
                    {
                        "image_id": "train1",
                        "image_path": "input/images/train1.jpg",
                        "mask_path": "input/masks/train1.png",
                        "bbox": None,
                    }
                ]
            )
            score_gt_df = pd.DataFrame(
                [
                    {
                        "image_id": "val1",
                        "image_path": "input/images/val1.jpg",
                        "mask_path": "input/masks/val1.png",
                        "bbox": None,
                    }
                ]
            )

            merged = _merge_visible_score_with_hidden_labels(
                workspace, train_df, score_gt_df
            )

            self.assertEqual(list(merged.columns), list(train_df.columns))
            self.assertEqual(merged.loc[0, "mask_path"], "input/masks/val1.png")
            self.assertTrue(pd.isna(merged.loc[0, "bbox"]))

    def test_score_label_merge_rejects_unmatched_score_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _write_csv(
                workspace / "input" / "biomaker_splits" / "score_val.csv",
                [{"image_id": "val1", "image_path": "input/images/val1.jpg"}],
            )
            _write_json(
                workspace / "input" / "biomaker_splits" / "split_manifest.json",
                {
                    "key_columns": ["image_id"],
                    "hidden_answers": {"score_val": {"answer_columns": ["label"]}},
                },
            )
            train_df = pd.DataFrame(
                [{"image_id": "train1", "image_path": "train.jpg", "label": "a"}]
            )
            score_gt_df = pd.DataFrame(
                [{"image_id": "other", "image_path": "other.jpg", "label": "b"}]
            )

            with self.assertRaisesRegex(ValueError, "unmatched score rows: 1"):
                _merge_visible_score_with_hidden_labels(
                    workspace, train_df, score_gt_df
                )

@pytest.mark.parametrize("file_column,declared", [("reference", ["reference"]), ("mask_path", None)])
def test_final_retrain_materializes_only_score_file_answers(tmp_path, file_column, declared):
    workspace = tmp_path / "run"
    private = tmp_path / "private_gt"
    (private / "targets").mkdir(parents=True)
    (private / "targets/val.bin").write_bytes(b"score target")
    (private / "targets/test.bin").write_bytes(b"test target")
    split = workspace / "input/biomaker_splits"
    _write_csv(split / "agent_train.csv", [{"id": "train", file_column: "input/train.bin"}])
    (workspace / "input/train.bin").write_bytes(b"train target")
    _write_csv(split / "score_val.csv", [{"id": "val"}])
    _write_csv(private / "score_val.csv", [{"id": "val", file_column: "private_gt/targets/val.bin"}])
    entry = {"answer_columns": [file_column]}
    if declared is not None:
        entry["file_reference_columns"] = declared
    _write_json(split / "split_manifest.json", {"key_columns": ["id"], "hidden_answers": {"score_val": entry}})

    retrain, receipt = prepare_final_retrain_workspace(SimpleNamespace(workspace_dir=workspace), private)
    rows = pd.read_csv(retrain / "input/biomaker_splits/agent_train.csv")
    copied = retrain / rows.loc[1, file_column]
    assert copied.is_file() and not copied.is_symlink()
    assert copied.read_bytes() == b"score target"
    assert receipt["score_file_answers"]["unique_files_copied"] == 1
    assert receipt["score_file_answers"]["columns"] == [file_column]
    assert not any(path.read_bytes() == b"test target" for path in (retrain / "input/final_retrain_score_answers").rglob("*") if path.is_file())
    assert pd.read_csv(split / "agent_train.csv")["id"].tolist() == ["train"]


def test_explicit_inline_answer_role_does_not_infer_a_file_from_column_name(tmp_path):
    workspace = tmp_path / "run"
    private = tmp_path / "private_gt"
    split = workspace / "input/biomaker_splits"
    _write_csv(split / "agent_train.csv", [{"id": "train", "diagnostic_path": "healthy"}])
    _write_csv(split / "score_val.csv", [{"id": "val"}])
    _write_csv(private / "score_val.csv", [{"id": "val", "diagnostic_path": "abnormal"}])
    _write_json(split / "split_manifest.json", {
        "key_columns": ["id"],
        "hidden_answers": {"score_val": {"answer_columns": ["diagnostic_path"], "file_reference_columns": []}},
    })
    retrain, receipt = prepare_final_retrain_workspace(SimpleNamespace(workspace_dir=workspace), private)
    assert pd.read_csv(retrain / "input/biomaker_splits/agent_train.csv")["diagnostic_path"].tolist() == ["healthy", "abnormal"]
    assert receipt["score_file_answers"]["unique_files_copied"] == 0


if __name__ == "__main__":
    unittest.main()
