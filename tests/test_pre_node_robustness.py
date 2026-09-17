from __future__ import annotations

import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from biomaker.utils.data_summary import data_and_evaluation_summary
from biomaker.utils.evaluator_setup import (
    _infer_fixed_classification_spec,
    _infer_fixed_segmentation_spec,
    _write_fixed_classification_evaluator,
    _write_fixed_segmentation_evaluator,
    score_candidate_with_evaluator,
)
from biomaker.utils.exp_guide import build_exp_guide
from biomaker.utils.agentic_data_planner import (
    build_dataset_facts,
    verify_split_package,
)
from biomaker.utils.split_protocol import (
    _make_private_paths_evaluator_visible,
    _normalize_split_paths_in_dir,
    sanitize_agent_visible_input,
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


def _write_png(path: Path, size: tuple[int, int] = (8, 8), value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0]), value, dtype=np.uint8)).save(path)


def _materialize_source_dataset_links(
    workspace_root: Path, split_dir: Path, workspace: Path
) -> None:
    plan_path = split_dir / "split_plan.json"
    if not plan_path.is_file():
        return
    try:
        split_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return
    source_files = split_plan.get("source_files")
    if not isinstance(source_files, dict):
        return
    input_dir = workspace / "input"
    for value in source_files.values():
        if not isinstance(value, str):
            continue
        path = workspace_root / value
        dataset_dirs = []
        parts = path.parts
        if "datasets" in parts:
            index = parts.index("datasets")
            if index + 1 < len(parts):
                dataset_dirs.append(Path(*parts[: index + 2]))
        dataset_dirs.append(path.parent)
        for dataset_dir in dataset_dirs:
            if not dataset_dir.is_dir():
                continue
            for child in dataset_dir.iterdir():
                link = input_dir / child.name
                if link.exists() or link.is_symlink():
                    continue
                os.symlink(child.resolve(), link)
            break


def _assert_first_path_values_resolve_and_decode(
    workspace: Path, split_csv: Path
) -> int:
    df = None
    try:
        df = pd.read_csv(split_csv, nrows=2)
    except Exception:
        return 0
    checked = 0
    for _, row in df.iterrows():
        for column in df.columns:
            name = str(column).lower()
            if "path" not in name and "file" not in name:
                continue
            value = row[column]
            if not isinstance(value, str) or not value.strip():
                continue
            path = Path(value)
            if not path.is_absolute():
                path = (
                    workspace / path
                    if path.parts and path.parts[0] == "input"
                    else workspace / "input" / path
                )
            if not path.is_file():
                continue
            lower = str(path).lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                with Image.open(path) as image:
                    image.verify()
            elif lower.endswith((".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")):
                if importlib.util.find_spec("SimpleITK") is None:
                    continue
                import SimpleITK as sitk

                sitk.ReadImage(str(path))
            checked += 1
    return checked


def _make_classification_workspace(
    root: Path,
    *,
    image_suffix: str = ".png",
    multilabel: bool = False,
) -> tuple[Path, Path]:
    workspace = root / "workspace"
    log_dir = root / "logs" / "run"
    split_dir = workspace / "input" / "biomaker_splits"
    hidden = log_dir / "private_gt" / "score_val.csv"
    image_rel = f"input/images/case_1{image_suffix}"
    if image_suffix == ".png":
        _write_png(workspace / image_rel)

    if multilabel:
        train_rows = [
            {"Id": 1, "class_1": 1, "class_2": 0, "path": image_rel},
            {"Id": 2, "class_1": 0, "class_2": 1, "path": image_rel},
        ]
        hidden_rows = [
            {"Id": 3, "class_1": 1, "class_2": 0},
            {"Id": 4, "class_1": 0, "class_2": 1},
        ]
        private_gt = {"score_val": {"answer_columns": ["class_1", "class_2"]}}
        task_type = "multilabel"
        metric = "mean_binary_accuracy"
    else:
        train_rows = [
            {"Id": 1, "class": 0, "path": image_rel},
            {"Id": 2, "class": 1, "path": image_rel},
        ]
        hidden_rows = [{"Id": 3, "class": 0}, {"Id": 4, "class": 1}]
        private_gt = {"score_val": {"answer_columns": ["class"]}}
        task_type = "binary"
        metric = "accuracy"

    _write_csv(split_dir / "agent_train.csv", train_rows)
    _write_csv(
        split_dir / "score_val.csv",
        [{"Id": 3, "path": image_rel}, {"Id": 4, "path": image_rel}],
    )
    _write_csv(split_dir / "agent_test.csv", [{"Id": 5, "path": image_rel}])
    _write_csv(hidden, hidden_rows)
    _write_json(
        split_dir / "split_plan.json",
        {
            "private_gt": private_gt,
            "primary_metric": metric,
            "task_type": task_type,
        },
    )
    return workspace, hidden


class PreNodeRobustnessTests(unittest.TestCase):
    def test_3d_classification_guide_and_fixed_evaluator(self) -> None:
        if importlib.util.find_spec("SimpleITK") is None:
            self.skipTest("SimpleITK is not installed")
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace, hidden = _make_classification_workspace(
                root, image_suffix=".nii.gz"
            )
            image_path = workspace / "input" / "images" / "case_1.nii.gz"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = sitk.GetImageFromArray(np.zeros((5, 6, 7), dtype=np.uint8))
            image.SetSpacing((1.1, 1.2, 1.3))
            sitk.WriteImage(image, str(image_path))

            guide = build_exp_guide(workspace)
            observation = guide["samples"]["train"]["path_observations"]["path"]
            self.assertEqual(observation["kind"], "medical_image")
            self.assertEqual(observation["array_shape"], [5, 6, 7])
            self.assertNotIn("private_gt", json.dumps(guide, default=str))

            spec = _infer_fixed_classification_spec(workspace, hidden)
            self.assertIsNotNone(spec)
            self.assertEqual(spec["task_type"], "binary")

            prepared = _write_fixed_classification_evaluator(
                workspace,
                root / "logs" / "run",
                hidden,
                timeout=30,
                task_desc="Primary validation metric: accuracy",
            )
            self.assertIsNotNone(prepared)
            working = workspace / "working"
            _write_csv(
                working / "validation_predictions.csv",
                [
                    {"Id": 3, "prob_0": 1.0, "prob_1": 0.0},
                    {"Id": 4, "prob_0": 0.0, "prob_1": 1.0},
                ],
            )
            _write_csv(
                working / "submission.csv",
                [{"Id": 5, "prob_0": 0.4, "prob_1": 0.6}],
            )
            result = score_candidate_with_evaluator(
                workspace, timeout=30, hidden_score_path=hidden
            )
            self.assertTrue(result.ok, result.output + (result.error or ""))
            self.assertEqual(result.metric, 1.0)

            text = data_and_evaluation_summary(workspace)
            self.assertIn("kind=medical_image", text)

    def test_3d_segmentation_does_not_use_2d_pil_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            split_dir = workspace / "input" / "biomaker_splits"
            hidden = root / "logs" / "run" / "private_gt" / "score_val.csv"
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
            _write_csv(hidden, [{"case_id": "val", "mask_path": "masks/val.nii.gz"}])
            _write_json(
                split_dir / "split_plan.json",
                {
                    "private_gt": {"score_val": {"answer_columns": ["mask_path"]}},
                    "primary_metric": "dice",
                    "task_type": "segmentation",
                    "columns": {"path": "image_path", "mask": "mask_path"},
                },
            )
            spec = _infer_fixed_segmentation_spec(workspace, hidden)
            self.assertIsNotNone(spec)
            self.assertEqual(spec["task_type"], "segmentation_3d")

    def test_fixed_segmentation_accepts_structured_metric_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            split_dir = workspace / "input" / "biomaker_splits"
            hidden = root / "logs" / "run" / "private_gt" / "score_val.csv"
            _write_csv(
                split_dir / "agent_train.csv",
                [{"image_id": "train", "image_path": "input/public/train.png"}],
            )
            _write_csv(
                split_dir / "score_val.csv",
                [{"image_id": "val", "image_path": "input/public/val.png"}],
            )
            _write_csv(
                split_dir / "agent_test.csv",
                [{"image_id": "test", "image_path": "input/public/test.png"}],
            )
            _write_csv(
                hidden,
                [{"image_id": "val", "mask_path": "private_gt/val_mask.png"}],
            )
            _write_json(
                split_dir / "split_manifest.json",
                {
                    "manifest_version": 1,
                    "sample_unit": "image",
                    "key_columns": ["image_id"],
                    "task_heads": [{"type": "segmentation_2d"}],
                    "metric_contract": {
                        "primary_metric": {"name": "AJI", "maximize": True}
                    },
                    "hidden_answers": {"score_val": {"answer_columns": ["mask_path"]}},
                },
            )
            _write_json(
                split_dir / "split_plan.json",
                {
                    "hidden_answers": {"score_val": {"answer_columns": ["mask_path"]}},
                    "columns": {"path": "image_path", "mask": "mask_path"},
                },
            )

            spec = _infer_fixed_segmentation_spec(workspace, hidden)

            self.assertIsNotNone(spec)
            self.assertEqual(spec["primary_metric"], "aji")
            self.assertEqual(spec["key_columns"], ["image_id"])
            self.assertEqual(spec["image_column"], "image_path")

    def test_malformed_split_is_reported_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            split_dir = workspace / "input" / "biomaker_splits"
            split_dir.mkdir(parents=True)
            (split_dir / "agent_train.csv").write_text(
                'Id,class,path\n1,0,"unterminated\n', encoding="utf-8"
            )
            _write_csv(split_dir / "score_val.csv", [{"Id": 2, "path": "missing.png"}])
            _write_csv(split_dir / "agent_test.csv", [{"Id": 3, "path": "missing.png"}])
            _write_json(
                split_dir / "split_plan.json",
                {
                    "private_gt": {"score_val": {"answer_columns": ["class"]}},
                    "primary_metric": "accuracy",
                    "task_type": "binary",
                },
            )

            guide = build_exp_guide(workspace)
            self.assertIn("read_error", guide["splits"]["train"])
            self.assertNotIn("cache", guide)
            text = data_and_evaluation_summary(workspace)
            self.assertIn("could not be read", text)

    def test_agentic_split_verifier_catches_source_undercoverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "package"
            for index in range(10):
                _write_png(source / "PNGImages" / f"case_{index}.png")
            train_rows = [
                {"image": f"PNGImages/case_{index}.png", "label": "a"}
                for index in range(2)
            ]
            score_rows = [{"image": "PNGImages/case_2.png"}]
            _write_csv(output / "biomaker_splits" / "agent_train.csv", train_rows)
            _write_csv(output / "biomaker_splits" / "score_val.csv", score_rows)
            _write_csv(
                output / "private_gt" / "score_val.csv",
                [{"image": "PNGImages/case_2.png", "label": "a"}],
            )
            _write_json(
                output / "private_gt" / "private_gt.json",
                {"score_val": {"answer_columns": ["label"]}},
            )

            verifier = verify_split_package(
                output, source_input_dir=source, facts=build_dataset_facts(source)
            )
            self.assertFalse(verifier["ok"])
            self.assertIn("cover only part", "\n".join(verifier["errors"]))

    def test_agentic_split_verifier_rejects_public_copied_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "package"
            _write_png(source / "images" / "case.png", value=7)
            _write_png(output / "public" / "images" / "case.png", value=7)
            _write_csv(
                output / "biomaker_splits" / "agent_train.csv",
                [{"case_id": "train", "image": "public/images/case.png", "label": 0}],
            )
            _write_csv(
                output / "biomaker_splits" / "score_val.csv",
                [{"case_id": "val", "image": "public/images/case.png"}],
            )
            _write_csv(
                output / "private_gt" / "score_val.csv",
                [{"case_id": "val", "image": "public/images/case.png", "label": 1}],
            )
            _write_json(
                output / "private_gt" / "private_gt.json",
                {"score_val": {"answer_columns": ["label"]}},
            )

            verifier = verify_split_package(output, source_input_dir=source)

            self.assertFalse(verifier["ok"])
            self.assertIn(
                "do not copy existing --input-dir files into public/",
                "\n".join(verifier["errors"]),
            )

    def test_split_manifest_sample_universe_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "package"
            for name in ["a.png", "b.png"]:
                _write_png(source / "images" / name)
            _write_csv(
                output / "biomaker_splits" / "agent_train.csv",
                [{"case_id": "a", "image": "images/a.png", "label": 0}],
            )
            _write_csv(
                output / "biomaker_splits" / "score_val.csv",
                [{"case_id": "b", "image": "images/b.png"}],
            )
            _write_csv(
                output / "private_gt" / "score_val.csv",
                [{"case_id": "b", "image": "images/b.png", "label": 1}],
            )
            _write_json(
                output / "private_gt" / "private_gt.json",
                {"score_val": {"answer_columns": ["label"]}},
            )
            _write_json(
                output / "split_manifest.json",
                {
                    "manifest_version": 1,
                    "sample_unit": "image",
                    "key_columns": ["case_id"],
                    "task_heads": [{"type": "classification"}],
                    "sample_universe": [
                        {
                            "sample_id": "a",
                            "split": "train",
                            "source_paths": ["images/a.png"],
                        },
                        {
                            "sample_id": "b",
                            "split": "score_val",
                            "source_paths": ["images/b.png"],
                        },
                        {
                            "sample_id": "extra",
                            "split": "score_val",
                            "source_paths": ["images/b.png"],
                        },
                    ],
                    "visibility": {
                        "agent_visible": ["biomaker_splits/*.csv"],
                        "evaluator_only": ["private_gt/**"],
                    },
                    "prediction_contract": {
                        "validation_artifact": "validation_predictions.csv"
                    },
                    "metric_contract": {"primary_metric": "accuracy", "maximize": True},
                    "private_gt": {"score_val": {"answer_columns": ["label"]}},
                },
            )

            verifier = verify_split_package(output, source_input_dir=source)
            self.assertFalse(verifier["ok"])
            self.assertIn(
                "sample_universe score_val count 2 does not match split rows 1",
                "\n".join(verifier["errors"]),
            )

    def test_split_disjoint_uses_manifest_keys_not_shared_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "package"
            _write_csv(
                output / "biomaker_splits" / "agent_train.csv",
                [{"case_id": "train_1", "site": "same", "label": 0}],
            )
            _write_csv(
                output / "biomaker_splits" / "score_val.csv",
                [{"case_id": "score_1", "site": "same"}],
            )
            _write_csv(
                output / "private_gt" / "score_val.csv",
                [{"case_id": "score_1", "site": "same", "label": 1}],
            )
            _write_json(
                output / "private_gt" / "private_gt.json",
                {"score_val": {"answer_columns": ["label"]}},
            )
            _write_json(
                output / "split_manifest.json",
                {
                    "manifest_version": 1,
                    "sample_unit": "row",
                    "key_columns": ["case_id"],
                    "task_heads": ["classification"],
                    "hidden_answers": {"score_val": {"answer_columns": ["label"]}},
                    "prediction_contract": {},
                    "metric_contract": {"primary_metric": "accuracy", "maximize": True},
                },
            )

            verifier = verify_split_package(output)

            self.assertTrue(verifier["ok"], verifier["errors"])

    def test_split_manifest_accepts_zero_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "package"
            _write_csv(
                output / "biomaker_splits" / "agent_train.csv",
                [{"case_id": "0", "label": "B"}],
            )
            _write_csv(
                output / "biomaker_splits" / "score_val.csv",
                [{"case_id": "1"}],
            )
            _write_csv(
                output / "private_gt" / "score_val.csv",
                [{"case_id": "1", "label": "M"}],
            )
            _write_json(
                output / "private_gt" / "private_gt.json",
                {"score_val": {"answer_columns": ["label"]}},
            )
            _write_json(
                output / "split_manifest.json",
                {
                    "manifest_version": 1,
                    "sample_unit": "volume",
                    "key_columns": ["case_id"],
                    "task_heads": ["classification"],
                    "sample_universe": [
                        {"sample_id": 0, "split": "train", "source_paths": []},
                        {"sample_id": 1, "split": "score_val", "source_paths": []},
                    ],
                    "hidden_answers": {"score_val": {"answer_columns": ["label"]}},
                    "prediction_contract": {},
                    "metric_contract": {"primary_metric": "accuracy", "maximize": True},
                },
            )

            verifier = verify_split_package(output)

            self.assertTrue(verifier["ok"], verifier["errors"])

    def test_split_protocol_normalizes_builder_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "workspace" / "input"
            package = root / "run" / "_split_builder" / "package"
            split_dir = package / "biomaker_splits"
            private_dir = root / "run" / "private_gt"
            _write_png(input_dir / "PNGImages" / "case.png")
            _write_png(input_dir / "PedMasks" / "case_mask.png")
            _write_png(package / "public" / "images" / "anon.png")

            source_rel = os.path.relpath(input_dir / "PNGImages" / "case.png", package)
            _write_csv(
                split_dir / "agent_train.csv",
                [
                    {
                        "image": source_rel,
                        "anon": "public/images/anon.png",
                        "label": "x",
                    }
                ],
            )
            _normalize_split_paths_in_dir(
                split_dir,
                input_dir,
                base_dir=package,
                generated_output_dir=package,
            )
            with (split_dir / "agent_train.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["image"], "input/PNGImages/case.png")
            self.assertEqual(row["anon"], "input/public/images/anon.png")

            _write_csv(
                private_dir / "score_val.csv",
                [
                    {
                        "image": "input/PNGImages/case.png",
                        "mask": "input/PedMasks/case_mask.png",
                    }
                ],
            )
            _write_json(
                private_dir / "private_gt.json",
                {"score_val": {"answer_columns": ["mask"]}},
            )
            _make_private_paths_evaluator_visible(private_dir, input_dir)
            with (private_dir / "score_val.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                private_row = next(csv.DictReader(handle))
            self.assertEqual(private_row["image"], "input/PNGImages/case.png")
            self.assertTrue(Path(private_row["mask"]).is_absolute())
            self.assertTrue(Path(private_row["mask"]).exists())

    def test_agent_visible_input_sanitizer_removes_source_label_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            workspace = root / "workspace"
            input_dir = workspace / "input"
            split_dir = input_dir / "biomaker_splits"
            input_dir.mkdir(parents=True)
            _write_png(dataset / "train_png" / "1.png", value=1)
            _write_png(dataset / "val_png" / "1.png", value=2)
            _write_png(dataset / "test_png" / "1.png", value=3)
            _write_csv(dataset / "train.csv", [{"Id": "train_1", "class": 0}])
            _write_csv(dataset / "val.csv", [{"Id": "val_1", "class": 1}])
            _write_csv(dataset / "test.csv", [{"Id": "test_1", "class": 1}])
            (dataset / "data_description.txt").write_text(
                "classification task\n", encoding="utf-8"
            )
            for child in dataset.iterdir():
                os.symlink(child.resolve(), input_dir / child.name)

            _write_csv(
                split_dir / "agent_train.csv",
                [{"Id": "train_1", "class": 0, "path": "input/train_png/1.png"}],
            )
            _write_csv(
                split_dir / "score_val.csv",
                [{"Id": "val_1", "path": "input/val_png/1.png"}],
            )
            _write_csv(
                split_dir / "agent_test.csv",
                [{"Id": "test_1", "path": "input/test_png/1.png"}],
            )

            summary = sanitize_agent_visible_input(input_dir, split_dir)

            self.assertEqual(summary["referenced_files"], 3)
            self.assertTrue(
                (input_dir / "biomaker_splits" / "agent_train.csv").is_file()
            )
            self.assertTrue((input_dir / "train_png" / "1.png").is_file())
            self.assertTrue((input_dir / "val_png" / "1.png").is_file())
            self.assertTrue((input_dir / "test_png" / "1.png").is_file())
            self.assertTrue((input_dir / "train_png").is_symlink())
            self.assertTrue((input_dir / "data_description.txt").is_file())
            self.assertFalse((input_dir / "train.csv").exists())
            self.assertFalse((input_dir / "val.csv").exists())
            self.assertFalse((input_dir / "test.csv").exists())

    def test_agent_visible_input_sanitizer_keeps_only_referenced_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            workspace = root / "workspace"
            input_dir = workspace / "input"
            split_dir = input_dir / "biomaker_splits"
            input_dir.mkdir(parents=True)
            _write_png(dataset / "DATA" / "train.png", value=1)
            _write_png(dataset / "DATA" / "val.png", value=2)
            _write_png(dataset / "DATA" / "test.png", value=3)
            _write_png(dataset / "MASK" / "train.png", value=4)
            _write_png(dataset / "MASK" / "val.png", value=5)
            _write_png(dataset / "MASK" / "test.png", value=6)
            _write_csv(
                dataset / "metadata.csv",
                [
                    {"Id": "train", "mask_path": "MASK/train.png"},
                    {"Id": "val", "mask_path": "MASK/val.png"},
                    {"Id": "test", "mask_path": "MASK/test.png"},
                ],
            )
            for child in dataset.iterdir():
                os.symlink(child.resolve(), input_dir / child.name)

            _write_csv(
                split_dir / "agent_train.csv",
                [
                    {
                        "Id": "train",
                        "image_path": "input/DATA/train.png",
                        "mask_path": "input/MASK/train.png",
                    }
                ],
            )
            _write_csv(
                split_dir / "score_val.csv",
                [{"Id": "val", "image_path": "input/DATA/val.png"}],
            )
            _write_csv(
                split_dir / "agent_test.csv",
                [{"Id": "test", "image_path": "input/DATA/test.png"}],
            )

            sanitize_agent_visible_input(input_dir, split_dir)

            self.assertTrue((input_dir / "MASK" / "train.png").is_file())
            self.assertFalse((input_dir / "MASK").is_symlink())
            self.assertFalse((input_dir / "MASK" / "val.png").exists())
            self.assertFalse((input_dir / "MASK" / "test.png").exists())
            self.assertFalse((input_dir / "metadata.csv").exists())

    def test_local_current_runs_and_experiments_pre_node_contracts(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        workspace_root = repo.parent
        logs_dir = workspace_root / "logs"
        if not logs_dir.is_dir():
            self.skipTest("workspace logs are not available")

        for run in sorted(logs_dir.glob("*-35step")):
            split_dir = run / "biomaker_splits"
            hidden = run / "private_gt" / "score_val.csv"
            if not split_dir.is_dir() or not hidden.is_file():
                continue
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp) / "workspace"
                (workspace / "input").mkdir(parents=True)
                os.symlink(split_dir.resolve(), workspace / "input" / "biomaker_splits")
                spec = _infer_fixed_classification_spec(workspace, hidden)
                self.assertIsNotNone(spec, run.name)
                if "chestmnist" in run.name:
                    self.assertEqual(spec["prediction_columns"][0], "prob_1")
                    self.assertEqual(spec["prediction_columns"][-1], "prob_14")

        monuseg_splits = workspace_root / "experiments" / "monuseg" / "biomaker_splits"
        monuseg_data = workspace_root / "datasets" / "monuseg"
        if monuseg_splits.is_dir() and monuseg_data.is_dir():
            with tempfile.TemporaryDirectory() as tmp:
                workspace = Path(tmp) / "workspace"
                (workspace / "input").mkdir(parents=True)
                os.symlink(
                    monuseg_splits.resolve(), workspace / "input" / "biomaker_splits"
                )
                os.symlink(
                    (monuseg_data / "train").resolve(), workspace / "input" / "train"
                )
                os.symlink(
                    (monuseg_data / "test").resolve(), workspace / "input" / "test"
                )
                hidden = monuseg_splits / "agent_val.csv"
                spec = _infer_fixed_segmentation_spec(workspace, hidden)
                self.assertIsNotNone(spec)
                self.assertEqual(
                    spec["score_split"], "./input/biomaker_splits/agent_val.csv"
                )
                self.assertEqual(spec["key_columns"], ["patient", "split", "tissue"])
                prepared = _write_fixed_segmentation_evaluator(
                    workspace,
                    Path(tmp) / "log",
                    hidden,
                    timeout=60,
                    task_desc="Primary validation metric: AJI",
                )
                self.assertIsNotNone(prepared)

        experiments_dir = workspace_root / "experiments"
        if experiments_dir.is_dir():
            split_plans = sorted(
                experiments_dir.glob("*/biomaker_splits/split_plan.json")
            )
            if not split_plans:
                self.skipTest("workspace shared split fixtures are not available")
            checked = []
            for split_plan in split_plans:
                split_dir = split_plan.parent
                hidden = next(
                    (
                        path
                        for path in [
                            split_dir / "score_val.csv",
                            split_dir / "agent_val.csv",
                        ]
                        if path.is_file()
                    ),
                    None,
                )
                if hidden is None:
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    workspace = Path(tmp) / "workspace"
                    (workspace / "input").mkdir(parents=True)
                    os.symlink(
                        split_dir.resolve(), workspace / "input" / "biomaker_splits"
                    )
                    _materialize_source_dataset_links(
                        workspace_root, split_dir, workspace
                    )
                    classification = _infer_fixed_classification_spec(workspace, hidden)
                    segmentation = _infer_fixed_segmentation_spec(workspace, hidden)
                    self.assertTrue(
                        classification is not None or segmentation is not None,
                        split_plan.parents[1].name,
                    )
                    decoded_paths = 0
                    for filename in [
                        "agent_train.csv",
                        "score_val.csv",
                        "agent_val.csv",
                        "agent_test.csv",
                    ]:
                        decoded_paths += _assert_first_path_values_resolve_and_decode(
                            workspace,
                            workspace / "input" / "biomaker_splits" / filename,
                        )
                    self.assertGreater(
                        decoded_paths,
                        0,
                        f"{split_plan.parents[1].name} has no resolvable sample path",
                    )
                    checked.append(split_plan.parents[1].name)
            self.assertGreater(len(checked), 0)


if __name__ == "__main__":
    unittest.main()
