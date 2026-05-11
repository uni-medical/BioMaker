from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from biomaker.utils.evaluator_setup import (
    _evaluator_prompt,
    _infer_fixed_classification_spec,
    _run_self_test,
    _sanitize_private_paths,
    _write_fixed_classification_evaluator,
    _write_fixed_segmentation_evaluator,
    score_candidate_with_evaluator,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_split_plan(path: Path, **updates: object) -> None:
    payload = {
        "private_gt": {"score_val": {"answer_columns": ["class"]}},
        "primary_metric": "accuracy",
        "task_type": "multiclass",
    }
    payload.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class DeterministicEvaluatorCompilerTests(unittest.TestCase):
    def test_classification_scores_and_rejects_bad_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            log_dir = root / "logs" / "run"
            split_dir = workspace / "input" / "biomaker_splits"
            hidden = log_dir / "private_gt" / "score_val.csv"

            _write_csv(
                split_dir / "agent_train.csv",
                [
                    {"Id": 1, "class": 0, "path": "input/train/1.png"},
                    {"Id": 2, "class": 1, "path": "input/train/2.png"},
                    {"Id": 3, "class": 2, "path": "input/train/3.png"},
                ],
            )
            _write_csv(
                split_dir / "score_val.csv",
                [
                    {"Id": 10, "path": "input/val/10.png"},
                    {"Id": 11, "path": "input/val/11.png"},
                    {"Id": 12, "path": "input/val/12.png"},
                ],
            )
            _write_csv(
                split_dir / "agent_test.csv",
                [
                    {"Id": 20, "path": "input/test/20.png"},
                    {"Id": 21, "path": "input/test/21.png"},
                ],
            )
            _write_csv(
                hidden,
                [
                    {"Id": 10, "class": 0},
                    {"Id": 11, "class": 1},
                    {"Id": 12, "class": 2},
                ],
            )
            _write_split_plan(split_dir / "split_plan.json")

            prepared = _write_fixed_classification_evaluator(
                workspace,
                log_dir,
                hidden,
                timeout=30,
                task_desc="Primary validation metric: accuracy",
            )
            self.assertIsNotNone(prepared)
            public_spec = json.loads(prepared.prediction_spec)
            self.assertNotIn("answer_columns", public_spec)
            self.assertNotIn("target_columns", public_spec)

            working = workspace / "working"
            _write_csv(
                working / "validation_predictions.csv",
                [
                    {"Id": 12, "prob_0": 0.0, "prob_1": 0.0, "prob_2": 1.0},
                    {"Id": 10, "prob_0": 1.0, "prob_1": 0.0, "prob_2": 0.0},
                    {"Id": 11, "prob_0": 0.0, "prob_1": 1.0, "prob_2": 0.0},
                ],
            )
            _write_csv(
                working / "submission.csv",
                [
                    {"Id": 21, "prob_0": 0.2, "prob_1": 0.3, "prob_2": 0.5},
                    {"Id": 20, "prob_0": 0.7, "prob_1": 0.2, "prob_2": 0.1},
                ],
            )
            result = score_candidate_with_evaluator(
                workspace, timeout=30, hidden_score_path=hidden
            )
            self.assertTrue(result.ok, result.output + (result.error or ""))
            self.assertEqual(result.metric, 1.0)

            _write_csv(
                working / "validation_predictions.csv",
                [
                    {"Id": 10, "prob_0": 1.0, "prob_1": 0.0, "prob_2": 0.0},
                    {"Id": 10, "prob_0": 1.0, "prob_1": 0.0, "prob_2": 0.0},
                    {"Id": 11, "prob_0": 0.0, "prob_1": 1.0, "prob_2": 0.0},
                ],
            )
            bad = score_candidate_with_evaluator(
                workspace, timeout=30, hidden_score_path=hidden
            )
            self.assertFalse(bad.ok)

            _write_csv(
                working / "validation_predictions.csv",
                [
                    {"Id": 10, "prob_0": 1.0, "prob_1": 1.0, "prob_2": 0.0},
                    {"Id": 11, "prob_0": 0.0, "prob_1": 1.0, "prob_2": 0.0},
                    {"Id": 12, "prob_0": 0.0, "prob_1": 0.0, "prob_2": 1.0},
                ],
            )
            bad_probs = score_candidate_with_evaluator(
                workspace, timeout=30, hidden_score_path=hidden
            )
            self.assertFalse(bad_probs.ok)

            prompt = _evaluator_prompt("task", workspace, hidden)
            prompt_text = json.dumps(prompt, default=str)
            self.assertNotIn(str(hidden), prompt_text)
            self.assertIn("BIOMAKER_SCORE_VAL_PATH", prompt_text)
            self.assertNotIn(str(hidden), _sanitize_private_paths(str(hidden), hidden))

    def test_evaluator_failure_output_sanitizes_hidden_score_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            hidden = root / "logs" / "run" / "private_gt" / "score_val.csv"
            workspace.mkdir(parents=True)
            hidden.parent.mkdir(parents=True)
            hidden.write_text("Id,class\n1,0\n", encoding="utf-8")
            (workspace / "evaluator.py").write_text(
                "\n".join(
                    [
                        "import os",
                        "def score_val():",
                        "    path = os.environ['BIOMAKER_SCORE_VAL_PATH']",
                        "    print('BIOMAKER_SCORE_VAL_PATH=' + path)",
                        "    raise RuntimeError('hidden path: ' + path)",
                    ]
                ),
                encoding="utf-8",
            )

            result = score_candidate_with_evaluator(
                workspace, timeout=30, hidden_score_path=hidden
            )

            self.assertFalse(result.ok)
            self.assertNotIn(str(hidden), result.output)
            self.assertNotIn(str(hidden.parent), result.output)
            self.assertIn("BIOMAKER_SCORE_VAL_PATH=<hidden_score_path>", result.output)
            self.assertIn("hidden path: <hidden_score_path>", result.output)

    def test_classification_spec_uses_split_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            split_dir = workspace / "input" / "biomaker_splits"
            hidden = root / "logs" / "private_gt" / "score_val.csv"
            _write_csv(
                split_dir / "agent_train.csv",
                [
                    {"case_id": "a", "site": "x", "class": 0},
                    {"case_id": "b", "site": "x", "class": 1},
                ],
            )
            _write_csv(split_dir / "score_val.csv", [{"case_id": "c", "site": "y"}])
            _write_csv(split_dir / "agent_test.csv", [{"case_id": "d", "site": "y"}])
            _write_csv(hidden, [{"case_id": "c", "site": "y", "class": 1}])
            _write_split_plan(
                split_dir / "split_plan.json",
                private_gt={"score_val": {"answer_columns": ["class"]}},
            )
            (split_dir / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "sample_unit": "image",
                        "key_columns": ["case_id"],
                        "task_heads": [{"type": "classification"}],
                        "metric_contract": {
                            "primary_metric": "balanced_accuracy",
                            "maximize": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            spec = _infer_fixed_classification_spec(workspace, hidden)
            self.assertIsNotNone(spec)
            self.assertEqual(spec["key_columns"], ["case_id"])
            self.assertEqual(spec["primary_metric"], "balanced_accuracy")

    def test_segmentation_scores_and_rejects_bad_mask_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            log_dir = root / "logs" / "run"
            split_dir = workspace / "input" / "biomaker_splits"
            hidden_dir = log_dir / "private_gt"
            hidden = hidden_dir / "score_val.csv"

            for rel in [
                "input/images/train.png",
                "input/images/val_a.png",
                "input/images/val_b.png",
                "input/images/test.png",
            ]:
                path = workspace / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (8, 8), color=0).save(path)

            mask_a = hidden_dir / "masks" / "val_a.png"
            mask_b = hidden_dir / "masks" / "val_b.png"
            mask_a.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 8), color=0).save(mask_a)
            Image.new("L", (8, 8), color=255).save(mask_b)

            _write_csv(
                split_dir / "agent_train.csv",
                [
                    {
                        "patient": "train",
                        "image_path": "input/images/train.png",
                        "mask_path": "input/masks/train.png",
                    }
                ],
            )
            _write_csv(
                split_dir / "score_val.csv",
                [
                    {"patient": "a", "image_path": "input/images/val_a.png"},
                    {"patient": "b", "image_path": "input/images/val_b.png"},
                ],
            )
            _write_csv(
                split_dir / "agent_test.csv",
                [{"patient": "t", "image_path": "input/images/test.png"}],
            )
            _write_csv(
                hidden,
                [
                    {"patient": "a", "mask_path": str(mask_a)},
                    {"patient": "b", "mask_path": str(mask_b)},
                ],
            )
            _write_split_plan(
                split_dir / "split_plan.json",
                private_gt={"score_val": {"answer_columns": ["mask_path"]}},
                primary_metric="binary_dice",
                task_type="segmentation",
                columns={"path": "image_path"},
            )

            prepared = _write_fixed_segmentation_evaluator(
                workspace,
                log_dir,
                hidden,
                timeout=30,
                task_desc="Primary validation metric: binary Dice",
            )
            self.assertIsNotNone(prepared)

            working = workspace / "working"
            pred_a = working / "pred_masks" / "val_a.png"
            pred_b = working / "pred_masks" / "val_b.png"
            test_mask = working / "pred_masks" / "test.png"
            pred_a.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (8, 8), color=0).save(pred_a)
            Image.new("L", (8, 8), color=255).save(pred_b)
            Image.new("L", (8, 8), color=0).save(test_mask)
            _write_csv(
                working / "validation_predictions.csv",
                [
                    {"patient": "a", "mask_path": str(pred_a)},
                    {"patient": "b", "mask_path": str(pred_b)},
                ],
            )
            _write_csv(
                working / "submission.csv",
                [{"patient": "t", "mask_path": str(test_mask)}],
            )
            result = score_candidate_with_evaluator(
                workspace, timeout=30, hidden_score_path=hidden
            )
            self.assertTrue(result.ok, result.output + (result.error or ""))
            self.assertEqual(result.metric, 1.0)

            bad_test_mask = working / "pred_masks" / "bad_test.png"
            Image.new("L", (7, 8), color=0).save(bad_test_mask)
            _write_csv(
                working / "submission.csv",
                [{"patient": "t", "mask_path": str(bad_test_mask)}],
            )
            bad = score_candidate_with_evaluator(
                workspace, timeout=30, hidden_score_path=hidden
            )
            self.assertFalse(bad.ok)

    def test_3d_segmentation_scores_with_fixed_evaluator(self) -> None:
        if importlib.util.find_spec("SimpleITK") is None:
            self.skipTest("SimpleITK is not installed")
        import SimpleITK as sitk

        def write_volume(path: Path, array: np.ndarray) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((1.0, 1.2, 1.5))
            sitk.WriteImage(image, str(path))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            log_dir = root / "logs" / "run"
            split_dir = workspace / "input" / "biomaker_splits"
            hidden_dir = log_dir / "private_gt"
            hidden = hidden_dir / "score_val.csv"

            image_array = np.zeros((4, 5, 6), dtype=np.int16)
            mask_array = np.zeros((4, 5, 6), dtype=np.uint8)
            mask_array[1:3, 2:4, 2:5] = 1
            for rel in [
                "input/images/train.nii.gz",
                "input/images/val.nii.gz",
                "input/images/test.nii.gz",
            ]:
                write_volume(workspace / rel, image_array)
            val_truth = hidden_dir / "masks" / "val.nii.gz"
            write_volume(val_truth, mask_array)

            _write_csv(
                split_dir / "agent_train.csv",
                [
                    {
                        "case_id": "train",
                        "image_path": "input/images/train.nii.gz",
                        "mask_path": "input/masks/train.nii.gz",
                    }
                ],
            )
            _write_csv(
                split_dir / "score_val.csv",
                [{"case_id": "val", "image_path": "input/images/val.nii.gz"}],
            )
            _write_csv(
                split_dir / "agent_test.csv",
                [{"case_id": "test", "image_path": "input/images/test.nii.gz"}],
            )
            _write_csv(hidden, [{"case_id": "val", "mask_path": str(val_truth)}])
            _write_split_plan(
                split_dir / "split_plan.json",
                private_gt={"score_val": {"answer_columns": ["mask_path"]}},
                primary_metric="binary_dice",
                task_type="segmentation_3d",
                columns={"path": "image_path", "mask": "mask_path"},
            )
            (split_dir / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "manifest_version": 1,
                        "sample_unit": "volume",
                        "key_columns": ["case_id"],
                        "task_heads": [{"type": "segmentation_3d"}],
                        "metric_contract": {
                            "primary_metric": "binary_dice",
                            "maximize": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            prepared = _write_fixed_segmentation_evaluator(
                workspace,
                log_dir,
                hidden,
                timeout=30,
                task_desc="Primary validation metric: binary Dice",
            )
            self.assertIsNotNone(prepared)
            spec = json.loads(prepared.prediction_spec)
            self.assertEqual(spec["task_type"], "segmentation_3d")
            mask_column = spec["prediction_columns"][0]

            working = workspace / "working"
            pred_path = working / "pred_masks" / "val.nii.gz"
            test_pred = working / "pred_masks" / "test.nii.gz"
            write_volume(pred_path, mask_array)
            write_volume(test_pred, mask_array)
            _write_csv(
                working / "validation_predictions.csv",
                [{"case_id": "val", mask_column: str(pred_path)}],
            )
            _write_csv(
                working / "submission.csv",
                [{"case_id": "test", mask_column: str(test_pred)}],
            )
            result = score_candidate_with_evaluator(
                workspace, timeout=30, hidden_score_path=hidden
            )
            self.assertTrue(result.ok, result.output + (result.error or ""))
            self.assertAlmostEqual(result.metric, 1.0)

    def test_multilabel_column_names_match_numeric_suffix_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            split_dir = workspace / "input" / "biomaker_splits"
            hidden = root / "logs" / "private_gt" / "score_val.csv"
            _write_csv(
                split_dir / "agent_train.csv",
                [
                    {"Id": 1, "class_1": 1, "class_2": 0},
                    {"Id": 2, "class_1": 0, "class_2": 1},
                ],
            )
            _write_csv(split_dir / "score_val.csv", [{"Id": 3}, {"Id": 4}])
            _write_csv(split_dir / "agent_test.csv", [{"Id": 5}])
            _write_csv(
                hidden,
                [
                    {"Id": 3, "class_1": 1, "class_2": 0},
                    {"Id": 4, "class_1": 0, "class_2": 1},
                ],
            )
            _write_split_plan(
                split_dir / "split_plan.json",
                private_gt={"score_val": {"answer_columns": ["class_1", "class_2"]}},
                primary_metric="mean_binary_accuracy",
                task_type="multilabel",
            )
            spec = _infer_fixed_classification_spec(workspace, hidden)
            self.assertIsNotNone(spec)
            self.assertEqual(spec["prediction_columns"], ["prob_1", "prob_2"])

    def test_run_self_test_calls_evaluator_self_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "working").mkdir(parents=True)
            (workspace / "evaluator.py").write_text(
                "\n".join(
                    [
                        "from pathlib import Path",
                        "def prediction_spec():",
                        "    return {'validation_artifact': './working/validation_predictions.csv', 'test_submission': './working/submission.csv', 'key_columns': [], 'prediction_columns': ['prob_0'], 'primary_metric': 'accuracy'}",
                        "def score_val():",
                        "    return {'metric': 1.0, 'maximize': True}",
                        "def validate_submission():",
                        "    return {'valid': True}",
                        "def self_test():",
                        "    Path('working/self_test_marker.txt').write_text('called')",
                        "    print('EVALUATOR_OK')",
                    ]
                ),
                encoding="utf-8",
            )
            _run_self_test(workspace, timeout=30)
            self.assertTrue((workspace / "working" / "self_test_marker.txt").is_file())


if __name__ == "__main__":
    unittest.main()
