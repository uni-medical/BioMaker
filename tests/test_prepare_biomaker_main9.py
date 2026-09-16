from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_biomaker_main9 import _parser, prepare_task


def _xml() -> str:
    return """<Annotations><Annotation><Regions>
    <Region><Vertices><Vertex X="10" Y="10"/><Vertex X="30" Y="10"/>
    <Vertex X="30" Y="30"/><Vertex X="10" Y="30"/></Vertices></Region>
    <Region><Vertices><Vertex X="50" Y="50"/><Vertex X="70" Y="50"/>
    <Vertex X="70" Y="70"/><Vertex X="50" Y="70"/></Vertices></Region>
    </Regions></Annotation></Annotations>"""


def _tiff_bytes() -> bytes:
    handle = io.BytesIO()
    Image.fromarray(np.zeros((1000, 1000, 3), dtype=np.uint8)).save(
        handle, format="TIFF"
    )
    return handle.getvalue()


def _image_bytes(array: np.ndarray, image_format: str) -> bytes:
    handle = io.BytesIO()
    Image.fromarray(array).save(handle, format=image_format)
    return handle.getvalue()


def _write_monuseg_zip(path: Path, sample_ids: list[str], *, train: bool) -> None:
    image_bytes = _tiff_bytes()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for sample_id in sample_ids:
            if train:
                image_member = (
                    "MoNuSeg 2018 Training Data/Tissue Images/" f"{sample_id}.tif"
                )
                xml_member = (
                    "MoNuSeg 2018 Training Data/Annotations/" f"{sample_id}.xml"
                )
            else:
                image_member = f"MoNuSegTestData/{sample_id}.tif"
                xml_member = f"MoNuSegTestData/{sample_id}.xml"
            archive.writestr(image_member, image_bytes)
            archive.writestr(xml_member, _xml())


def _write_kvasir_zip(path: Path, count: int = 20) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Kvasir-SEG/kavsir_bboxes.json", "{}")
        for index in range(1, count + 1):
            sample_id = f"source-{index:04d}"
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[..., 0] = index
            image[0, index % 32, 1] = 255
            mask = np.zeros((32, 32), dtype=np.uint8)
            side = (4, 10, 20)[(index - 1) % 3]
            mask[:side, :side] = 255
            archive.writestr(
                f"Kvasir-SEG/images/{sample_id}.jpg",
                _image_bytes(image, "JPEG"),
            )
            archive.writestr(
                f"Kvasir-SEG/masks/{sample_id}.jpg",
                _image_bytes(mask, "JPEG"),
            )


def _write_busi_zip(path: Path) -> None:
    category_counts = {"benign": 10, "malignant": 6, "normal": 4}
    duplicate_payload: bytes | None = None
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        serial = 0
        for category, count in category_counts.items():
            for index in range(1, count + 1):
                serial += 1
                image = np.zeros((24, 24, 3), dtype=np.uint8)
                image[..., 0] = serial * 7
                image[index % 24, serial % 24, 1] = 255
                payload = _image_bytes(image, "PNG")
                if category == "benign" and index == 1:
                    duplicate_payload = payload
                elif category == "malignant" and index == 1:
                    assert duplicate_payload is not None
                    payload = duplicate_payload
                stem = f"{category} ({index})"
                prefix = f"Dataset_BUSI_with_GT/{category}/{stem}"
                archive.writestr(prefix + ".png", payload)
                mask = np.zeros((24, 24), dtype=np.uint8)
                if category != "normal":
                    mask[2:7, 2:7] = 255
                archive.writestr(prefix + "_mask.png", _image_bytes(mask, "PNG"))
                if category == "benign" and index == 2:
                    second = np.zeros((24, 24), dtype=np.uint8)
                    second[15:20, 15:20] = 255
                    archive.writestr(
                        prefix + "_mask_1.png", _image_bytes(second, "PNG")
                    )


def _write_aeropath_tree(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "README.md").write_text(
        "The dataset contains 27 computed tomography scans.\n", encoding="utf-8"
    )
    (root / "AeroPath.py").write_text("name=datasets.Split.TEST\n", encoding="utf-8")
    for patient_id in range(1, 28):
        patient = root / "data" / str(patient_id)
        patient.mkdir(parents=True)
        for suffix in (
            "_CT_HR.nii.gz",
            "_CT_HR_label_airways.nii.gz",
            "_CT_HR_label_lungs.nii.gz",
        ):
            (patient / f"{patient_id}{suffix}").write_bytes(b"synthetic")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _all_public_rows(package: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "train": _rows(package / "biomaker_splits" / "agent_train.csv"),
        "score_val": _rows(package / "biomaker_splits" / "score_val.csv"),
        "test": _rows(package / "biomaker_splits" / "agent_test.csv"),
    }


def _target_mask(data: Path, package: Path, sample_id: str) -> np.ndarray:
    for csv_path in (
        package / "biomaker_splits" / "agent_train.csv",
        package / "private_gt" / "score_val.csv",
        package / "private_gt" / "test.csv",
    ):
        for row in _rows(csv_path):
            if row["Id"] != sample_id:
                continue
            raw = Path(row["mask_path"])
            mask_path = (
                data / Path(*raw.parts[1:])
                if raw.parts[0] == "input"
                else package / raw
            )
            with Image.open(mask_path) as image:
                return np.asarray(image)
    raise AssertionError(f"No target mask for {sample_id}")


class PrepareMain9Test(unittest.TestCase):
    def test_monuseg_is_deterministic_and_hides_validation_and_test_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw" / "MoNuSeg"
            raw.mkdir(parents=True)
            train_ids = [
                "TCGA-AA-0001-01Z-00-DX1",
                "TCGA-BB-0002-01Z-00-DX1",
                "TCGA-CC-0003-01Z-00-DX1",
                "TCGA-DD-0004-01Z-00-DX1",
                "TCGA-EE-0005-01Z-00-DX1",
            ]
            test_ids = [
                "TCGA-FF-0006-01Z-00-DX1",
                "TCGA-GG-0007-01Z-00-DX1",
            ]
            _write_monuseg_zip(
                raw / "MoNuSeg_2018_Training_Data.zip", train_ids, train=True
            )
            _write_monuseg_zip(raw / "MoNuSegTestData.zip", test_ids, train=False)
            task = root / "monuseg.md"
            task.write_text("Primary validation metric: AJI\n", encoding="utf-8")
            output = root / "prepared"
            cache = root / "cache"

            result = prepare_task(
                "monuseg",
                raw_root=root / "raw",
                task_desc_path=task,
                output_root=output,
                cache_root=cache,
                strict_official=False,
            )

            self.assertEqual(result["status"], "ready")
            self.assertFalse(result["reused"])
            self.assertEqual(result["counts"], {"train": 4, "score_val": 1, "test": 2})
            data = Path(result["data_dir"])
            package = Path(result["package_dir"])
            train = _rows(package / "biomaker_splits" / "agent_train.csv")
            score = _rows(package / "biomaker_splits" / "score_val.csv")
            test = _rows(package / "biomaker_splits" / "agent_test.csv")
            hidden_score = _rows(package / "private_gt" / "score_val.csv")
            hidden_test = _rows(package / "private_gt" / "test.csv")

            self.assertIn("mask_path", train[0])
            self.assertNotIn("mask_path", score[0])
            self.assertNotIn("mask_path", test[0])
            self.assertEqual(len(hidden_score), 1)
            self.assertEqual(len(hidden_test), 2)
            self.assertEqual(len(list((data / "train_masks").glob("*.png"))), 4)
            self.assertFalse((data / "private_gt").exists())
            self.assertFalse((data / "test_masks").exists())
            self.assertFalse((data / "score_val_masks").exists())

            mask_ref = hidden_score[0]["mask_path"]
            mask_path = package / mask_ref
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image)
            self.assertEqual(mask.shape, (1000, 1000))
            self.assertEqual(set(np.unique(mask)), {0, 1, 2})

            manifest = json.loads((package / "split_manifest.json").read_text())
            self.assertEqual(manifest["task_heads"][0]["type"], "instance_segmentation")
            self.assertEqual(manifest["metric_contract"]["primary_metric"], "aji")
            self.assertTrue(manifest["split_contract"]["official_test_preserved"])
            train_groups = {"-".join(row["Id"].split("-")[:3]) for row in train}
            score_groups = {"-".join(row["Id"].split("-")[:3]) for row in score}
            self.assertFalse(train_groups & score_groups)

            verifier = json.loads((package / "verifier.json").read_text())
            self.assertTrue(verifier["ok"], verifier)
            self.assertTrue((cache / result["cache_key"] / "manifest.json").is_file())
            ready = json.loads((data / "_READY.json").read_text())
            self.assertEqual(ready["task_family"], "segmentation")
            self.assertEqual(ready["task_text_sha256"], result["task_text_sha256"])
            suite = json.loads((output / "suite_manifest.json").read_text())
            suite_entry = suite["tasks"][0]
            self.assertEqual(suite_entry["task_family"], "segmentation")
            self.assertEqual(
                suite_entry["task_text_sha256"], result["task_text_sha256"]
            )
            self.assertEqual(suite_entry["primary_metric"], "aji")
            self.assertEqual(
                suite_entry["counts"], {"train": 4, "score_val": 1, "test": 2}
            )

            reused = prepare_task(
                "monuseg",
                raw_root=root / "raw",
                task_desc_path=task,
                output_root=output,
                cache_root=cache,
                strict_official=False,
            )
            self.assertTrue(reused["reused"])
            self.assertEqual(reused["cache_key"], result["cache_key"])

            # A same-size TIFF pixel edit used to pass the structural verifier.
            image_path = next((data / "images").glob("*.tif"))
            original_image = image_path.read_bytes()
            image_path.write_bytes(
                original_image[:-1] + bytes([original_image[-1] ^ 1])
            )
            with self.assertRaisesRegex(RuntimeError, "integrity mismatch"):
                prepare_task(
                    "monuseg",
                    raw_root=root / "raw",
                    task_desc_path=task,
                    output_root=output,
                    cache_root=cache,
                    strict_official=False,
                )

    def test_kvasir_uses_all_pairs_and_seals_binary_holdout_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw" / "Kvasir-SEG"
            raw.mkdir(parents=True)
            _write_kvasir_zip(raw / "kvasir-seg.zip")
            task = root / "kvasir_seg.md"
            task.write_text("Primary metric: binary Dice\n", encoding="utf-8")
            output = root / "prepared"
            cache = root / "cache"
            output.mkdir()
            (output / "suite_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "biomaker_main9_suite",
                        "schema_version": 1,
                        "task_count": 1,
                        "tasks": [{"task_id": "bccd", "status": "waiting_data"}],
                    }
                ),
                encoding="utf-8",
            )

            result = prepare_task(
                "kvasir_seg",
                raw_root=root / "raw",
                task_desc_path=task,
                output_root=output,
                cache_root=cache,
                strict_official=False,
            )

            self.assertEqual(result["counts"], {"train": 14, "score_val": 3, "test": 3})
            data = Path(result["data_dir"])
            package = Path(result["package_dir"])
            rows = _all_public_rows(package)
            self.assertEqual(sum(map(len, rows.values())), 20)
            self.assertEqual(
                len({row["Id"] for values in rows.values() for row in values}), 20
            )
            self.assertTrue(all("mask_path" in row for row in rows["train"]))
            self.assertTrue(
                all("mask_path" not in row for row in rows["score_val"] + rows["test"])
            )
            self.assertFalse((data / "private_gt").exists())
            hidden_ids = {
                row["Id"] for row in _rows(package / "private_gt" / "score_val.csv")
            } | {row["Id"] for row in _rows(package / "private_gt" / "test.csv")}
            self.assertEqual(
                hidden_ids,
                {row["Id"] for row in rows["score_val"] + rows["test"]},
            )
            mask = _target_mask(data, package, rows["score_val"][0]["Id"])
            self.assertEqual(set(np.unique(mask)), {0, 255})

            manifest = json.loads((package / "split_manifest.json").read_text())
            self.assertEqual(
                manifest["metric_contract"]["primary_metric"], "binary_dice"
            )
            self.assertEqual(
                manifest["split_contract"]["group_key"], "exact source-image SHA-256"
            )
            self.assertEqual(
                manifest["mask_contract"]["source_threshold"],
                ">=128 after grayscale conversion",
            )
            self.assertTrue(json.loads((package / "verifier.json").read_text())["ok"])
            suite = json.loads((output / "suite_manifest.json").read_text())
            self.assertEqual(
                [entry["task_id"] for entry in suite["tasks"]],
                ["kvasir_seg", "bccd"],
            )

            reused = prepare_task(
                "kvasir_seg",
                raw_root=root / "raw",
                task_desc_path=task,
                output_root=output,
                cache_root=cache,
                strict_official=False,
            )
            self.assertTrue(reused["reused"])
            task.write_text("Changed protocol\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_task(
                    "kvasir_seg",
                    raw_root=root / "raw",
                    task_desc_path=task,
                    output_root=output,
                    cache_root=cache,
                    strict_official=False,
                )

    def test_busi_unions_masks_uses_neutral_ids_and_groups_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw" / "BUSI"
            raw.mkdir(parents=True)
            _write_busi_zip(raw / "Dataset_BUSI.zip")
            task = root / "busi.md"
            task.write_text("Primary metric: binary Dice\n", encoding="utf-8")
            output = root / "prepared"

            result = prepare_task(
                "busi",
                raw_root=root / "raw",
                task_desc_path=task,
                output_root=output,
                cache_root=root / "cache",
                strict_official=False,
            )

            self.assertEqual(result["counts"], {"train": 14, "score_val": 3, "test": 3})
            data = Path(result["data_dir"])
            package = Path(result["package_dir"])
            rows = _all_public_rows(package)
            flattened = [
                (split, row) for split, values in rows.items() for row in values
            ]
            self.assertEqual(len(flattened), 20)
            self.assertTrue(
                all(re.fullmatch(r"BUSI-\d{4}", row["Id"]) for _, row in flattened)
            )
            self.assertTrue(
                all(
                    not any(
                        category in row["image_path"].lower()
                        for category in ("benign", "malignant", "normal")
                    )
                    for _, row in flattened
                )
            )

            split_by_hash: dict[str, list[str]] = {}
            for split, row in flattened:
                image_path = data / Path(*Path(row["image_path"]).parts[1:])
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                split_by_hash.setdefault(digest, []).append(split)
            self.assertTrue(
                all(len(set(splits)) == 1 for splits in split_by_hash.values())
            )
            self.assertEqual(
                sum(len(values) > 1 for values in split_by_hash.values()), 1
            )

            union_mask = _target_mask(data, package, "BUSI-0002")
            self.assertTrue(np.all(union_mask[2:7, 2:7] == 255))
            self.assertTrue(np.all(union_mask[15:20, 15:20] == 255))
            normal_mask = _target_mask(data, package, "BUSI-0017")
            self.assertEqual(int(np.count_nonzero(normal_mask)), 0)
            manifest = json.loads((package / "split_manifest.json").read_text())
            self.assertFalse(
                manifest["split_contract"]["diagnosis_mapping_agent_visible"]
            )
            self.assertIn(
                "logical union", manifest["mask_contract"]["multiple_source_masks"]
            )
            self.assertTrue(json.loads((package / "verifier.json").read_text())["ok"])

    def test_aeropath_is_test_only_protocol_blocked_without_ready_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_aeropath_tree(root / "raw" / "AeroPath")
            task = root / "aeropath.md"
            task.write_text("Airway segmentation protocol gate\n", encoding="utf-8")
            output = root / "prepared"

            result = prepare_task(
                "aeropath",
                raw_root=root / "raw",
                task_desc_path=task,
                output_root=output,
                cache_root=root / "cache",
                strict_official=False,
            )

            self.assertEqual(result["status"], "protocol_blocked")
            self.assertEqual(result["source_count"], 27)
            self.assertEqual(result["official_protocol"], "TEST-only")
            self.assertEqual(result["primary_target"], "airways")
            self.assertNotIn("counts", result)
            self.assertNotIn("ready_marker", result)
            self.assertFalse((output / "data" / "aeropath").exists())
            self.assertFalse((output / "packages" / "aeropath").exists())
            self.assertFalse((root / "cache").exists())
            suite = json.loads((output / "suite_manifest.json").read_text())
            entry = suite["tasks"][0]
            self.assertEqual(entry["status"], "protocol_blocked")
            self.assertFalse(entry["artifacts_generated"])
            self.assertIn("all 27", entry["reason"])

    def test_cli_parser_has_one_task_description_directory_option(self) -> None:
        parser = _parser()
        actions = [
            action
            for action in parser._actions
            if "--task-desc-dir" in action.option_strings
        ]
        self.assertEqual(len(actions), 1)


if __name__ == "__main__":
    unittest.main()
