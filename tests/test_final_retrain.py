from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from biomaker.utils.final_retrain import _merge_visible_score_with_hidden_labels


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

if __name__ == "__main__":
    unittest.main()
