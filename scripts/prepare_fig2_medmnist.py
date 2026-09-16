#!/usr/bin/env python3
"""Build deterministic BioMaker packages for the 13 Fig. 2 MedMNIST tasks.

The source NPZ is deliberately kept outside the generated task ``data_dir``.
Each task data directory contains three label-free NPY arrays (one per official
split), while labels live only in the training CSV or evaluator-only package.
The package is also installed under BioMaker's split-builder cache key so
``prepare_split_protocol`` can reuse it without an LLM split-builder call.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from biomaker.utils.agentic_data_planner import (  # noqa: E402
    build_dataset_facts,
    split_builder_cache_key,
    verify_split_package,
)
from scripts.taskpack_integrity import (  # noqa: E402
    verify_package_integrity,
    write_package_integrity,
)

SCHEMA_VERSION = 1
MEDMNIST_RELEASE = "3.0.2"
ZENODO_RECORD = "10519652"
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TaskSpec:
    flag: str
    display_name: str
    dimension: str
    task_type: str
    labels: tuple[str, ...]
    channels: int
    counts: tuple[int, int, int]
    license: str
    md5_by_resolution: dict[int, str]

    @property
    def count_by_split(self) -> dict[str, int]:
        return dict(zip(SPLITS, self.counts))

    @property
    def answer_columns(self) -> list[str]:
        if self.task_type == "multilabel":
            return [f"label_{index}" for index in range(len(self.labels))]
        return ["class"]

    @property
    def prediction_columns(self) -> list[str]:
        return [f"prob_{index}" for index in range(len(self.labels))]

    @property
    def primary_metric(self) -> str:
        if self.task_type == "multilabel":
            return "mean_roc_auc"
        if self.task_type == "ordinal_multiclass":
            return "ordinal_accuracy"
        return "accuracy"


TASK_SPECS: dict[str, TaskSpec] = {
    "pathmnist": TaskSpec(
        "pathmnist",
        "PathMNIST",
        "2d",
        "multiclass",
        (
            "adipose",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth muscle",
            "normal colon mucosa",
            "cancer-associated stroma",
            "colorectal adenocarcinoma epithelium",
        ),
        3,
        (89996, 10004, 7180),
        "CC BY 4.0",
        {
            28: "a8b06965200029087d5bd730944a56c1",
            64: "55aa9c1e0525abe5a6b9d8343a507616",
            128: "ac42d08fb904d92c244187169d1fd1d9",
            224: "2c51a510bcdc9cf8ddb2af93af1eadec",
        },
    ),
    "chestmnist": TaskSpec(
        "chestmnist",
        "ChestMNIST",
        "2d",
        "multilabel",
        (
            "atelectasis",
            "cardiomegaly",
            "effusion",
            "infiltration",
            "mass",
            "nodule",
            "pneumonia",
            "pneumothorax",
            "consolidation",
            "edema",
            "emphysema",
            "fibrosis",
            "pleural",
            "hernia",
        ),
        1,
        (78468, 11219, 22433),
        "CC BY 4.0",
        {
            28: "02c8a6516a18b556561a56cbdd36c4a8",
            64: "9de6cd0b934ebb5b7426cfba5efbae16",
            128: "db107e5590b27930b62dbaf558aebee3",
            224: "45bd33e6f06c3e8cdb481c74a89152aa",
        },
    ),
    "dermamnist": TaskSpec(
        "dermamnist",
        "DermaMNIST",
        "2d",
        "multiclass",
        (
            "actinic keratoses and intraepithelial carcinoma",
            "basal cell carcinoma",
            "benign keratosis-like lesions",
            "dermatofibroma",
            "melanoma",
            "melanocytic nevi",
            "vascular lesions",
        ),
        3,
        (7007, 1003, 2005),
        "CC BY-NC 4.0",
        {
            28: "0744692d530f8e62ec473284d019b0c7",
            64: "b70a2f5635c6199aeaa28c31d7202e1f",
            128: "2defd784463fa5243564e855ed717de1",
            224: "8974907d8e169bef5f5b96bc506ae45d",
        },
    ),
    "pneumoniamnist": TaskSpec(
        "pneumoniamnist",
        "PneumoniaMNIST",
        "2d",
        "binary",
        ("normal", "pneumonia"),
        1,
        (4708, 524, 624),
        "CC BY 4.0",
        {
            28: "28209eda62fecd6e6a2d98b1501bb15f",
            64: "8f4eceb4ccffa70c672198ea285246c6",
            128: "05b46931834c231683c68f40c47b2971",
            224: "d6a3c71de1b945ea11211b03746c1fe1",
        },
    ),
    "retinamnist": TaskSpec(
        "retinamnist",
        "RetinaMNIST",
        "2d",
        "ordinal_multiclass",
        ("0", "1", "2", "3", "4"),
        3,
        (1080, 120, 400),
        "CC BY 4.0",
        {
            28: "bd4c0672f1bba3e3a89f0e4e876791e4",
            64: "afda852cc34dcda56f86ad2b2457dbcc",
            128: "e48e916a24454daf90583d4e6efb1a18",
            224: "eae7e3b6f3fcbda4ae613ebdcbe35348",
        },
    ),
    "breastmnist": TaskSpec(
        "breastmnist",
        "BreastMNIST",
        "2d",
        "binary",
        ("malignant", "normal or benign"),
        1,
        (546, 78, 156),
        "CC BY 4.0",
        {
            28: "750601b1f35ba3300ea97c75c52ff8f6",
            64: "742edef2a1fd1524b2efff4bd7ba9364",
            128: "363e4b3f8d712e9b5de15470a2aaadf1",
            224: "b56378a6eefa9fed602bb16d192d4c8b",
        },
    ),
    "bloodmnist": TaskSpec(
        "bloodmnist",
        "BloodMNIST",
        "2d",
        "multiclass",
        (
            "basophil",
            "eosinophil",
            "erythroblast",
            "immature granulocytes",
            "lymphocyte",
            "monocyte",
            "neutrophil",
            "platelet",
        ),
        3,
        (11959, 1712, 3421),
        "CC BY 4.0",
        {
            28: "7053d0359d879ad8a5505303e11de1dc",
            64: "2b94928a2ae4916078ca51e05b6b800b",
            128: "adace1e0ed228fccda1f39692059dd4c",
            224: "b718ff6835fcbdb22ba9eacccd7b2601",
        },
    ),
    "tissuemnist": TaskSpec(
        "tissuemnist",
        "TissueMNIST",
        "2d",
        "multiclass",
        (
            "collecting duct or connecting tubule",
            "distal convoluted tubule",
            "glomerular endothelial cells",
            "interstitial endothelial cells",
            "leukocytes",
            "podocytes",
            "proximal tubule segments",
            "thick ascending limb",
        ),
        1,
        (165466, 23640, 47280),
        "CC BY 4.0",
        {
            28: "ebe78ee8b05294063de985d821c1c34b",
            64: "123ece2eba09d0aa5d698fda57103344",
            128: "61b955355d7425a89687b06cca3ce0c2",
            224: "b077128c4a949f0a4eb01517f9037b9c",
        },
    ),
    "nodulemnist3d": TaskSpec(
        "nodulemnist3d",
        "NoduleMNIST3D",
        "3d",
        "binary",
        ("benign", "malignant"),
        1,
        (1158, 165, 310),
        "CC BY 4.0",
        {
            28: "8755a7e9e05a4d9ce80a24c3e7a256f3",
            64: "c47c5b7d457bf6332200d2ea6d64ecd8",
        },
    ),
    "adrenalmnist3d": TaskSpec(
        "adrenalmnist3d",
        "AdrenalMNIST3D",
        "3d",
        "binary",
        ("normal", "hyperplasia"),
        1,
        (1188, 98, 298),
        "CC BY 4.0",
        {
            28: "bbd3c5a5576322bc4cdfea780653b1ce",
            64: "17721accfe9fb005146a47d33bc54b2f",
        },
    ),
    "fracturemnist3d": TaskSpec(
        "fracturemnist3d",
        "FractureMNIST3D",
        "3d",
        "multiclass",
        (
            "buckle rib fracture",
            "nondisplaced rib fracture",
            "displaced rib fracture",
        ),
        1,
        (1027, 103, 240),
        "CC BY 4.0",
        {
            28: "6aa7b0143a6b42da40027a9dda61302f",
            64: "f01d7e6316aedf4210da0da5b7437b42",
        },
    ),
    "vesselmnist3d": TaskSpec(
        "vesselmnist3d",
        "VesselMNIST3D",
        "3d",
        "binary",
        ("vessel", "aneurysm"),
        1,
        (1335, 191, 382),
        "CC BY 4.0",
        {
            28: "b41fd4f7e7e2feedddb201585ecafa1b",
            64: "6bb274a8846e1097066dcd64e2c4520f",
        },
    ),
    "synapsemnist3d": TaskSpec(
        "synapsemnist3d",
        "SynapseMNIST3D",
        "3d",
        "binary",
        ("inhibitory synapse", "excitatory synapse"),
        1,
        (1230, 177, 352),
        "CC BY 4.0",
        {
            28: "1235b78a3cd6280881dd7850a78eadb6",
            64: "43bd14ebf3af9d3dd072446fedc14d5e",
        },
    ),
}

TASK_ORDER = tuple(TASK_SPECS)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _workspace_input_view(data_dir: Path) -> Iterator[Path]:
    """Mirror ``prep_agent_workspace`` for an exact split-cache fingerprint."""
    with tempfile.TemporaryDirectory(prefix="biomaker-medmnist-input-") as tmp:
        input_dir = Path(tmp) / "input"
        input_dir.mkdir()
        for source in sorted(data_dir.iterdir(), key=lambda path: path.name):
            (input_dir / source.name).symlink_to(source.resolve())
        yield input_dir


def _split_cache_key_for_data_dir(
    data_dir: Path,
    *,
    task_text: str,
    model: str,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    with _workspace_input_view(data_dir) as input_dir:
        return split_builder_cache_key(
            input_dir=input_dir,
            task_text=task_text,
            model=model,
            temperature=temperature,
        )


def _parse_source_name(path: Path) -> tuple[TaskSpec, int]:
    stem = path.stem.lower()
    for flag in sorted(TASK_SPECS, key=len, reverse=True):
        if stem == flag:
            return TASK_SPECS[flag], 28
        prefix = f"{flag}_"
        if stem.startswith(prefix) and stem[len(prefix) :].isdigit():
            resolution = int(stem[len(prefix) :])
            spec = TASK_SPECS[flag]
            if resolution not in spec.md5_by_resolution:
                supported = ", ".join(map(str, sorted(spec.md5_by_resolution)))
                raise ValueError(
                    f"Unsupported resolution {resolution} for {flag}; expected {supported}."
                )
            return spec, resolution
    raise ValueError(f"{path.name} is not one of the 13 frozen Fig. 2 MedMNIST tasks.")


def _npy_header(
    archive: zipfile.ZipFile, member: str
) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    with archive.open(member, "r") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(handle)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(handle)
        if version == (3, 0):
            return np.lib.format.read_array_header_2_0(handle)
        raise ValueError(f"Unsupported NPY format version {version} in {member}.")


def _load_label_array(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with archive.open(member, "r") as handle:
        payload = handle.read()
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{member} is not a NumPy array.")
    return array


def _copy_image_member(
    archive: zipfile.ZipFile, member: str, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member, "r") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _validate_image_shape(
    spec: TaskSpec,
    resolution: int,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    strict_official: bool,
) -> None:
    if not shape or shape[0] <= 0:
        raise ValueError(f"Invalid empty image array shape: {shape}.")
    if (
        dtype.hasobject
        or dtype.fields is not None
        or not np.issubdtype(dtype, np.number)
    ):
        raise ValueError(f"Image arrays must have a plain numeric dtype, got {dtype}.")
    sample_shape = shape[1:]
    if spec.dimension == "2d":
        if len(sample_shape) not in (2, 3):
            raise ValueError(f"{spec.flag} expects 2D samples, got {sample_shape}.")
        if len(sample_shape) == 3 and sample_shape[-1] != spec.channels:
            raise ValueError(
                f"{spec.flag} expects {spec.channels} channels, got {sample_shape}."
            )
        if len(sample_shape) == 2 and spec.channels != 1:
            raise ValueError(
                f"{spec.flag} expects an explicit {spec.channels}-channel axis."
            )
        spatial_shape = sample_shape[:2]
    else:
        if len(sample_shape) != 3:
            raise ValueError(f"{spec.flag} expects 3D volumes, got {sample_shape}.")
        spatial_shape = sample_shape
    if strict_official and any(axis != resolution for axis in spatial_shape):
        raise ValueError(
            f"{spec.flag} filename declares {resolution}, but sample shape is {sample_shape}."
        )


def _normalize_labels(spec: TaskSpec, labels: np.ndarray, split: str) -> np.ndarray:
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    if labels.ndim != 2:
        raise ValueError(f"{split}_labels must be 1D/2D, got {labels.shape}.")
    expected_width = len(spec.labels) if spec.task_type == "multilabel" else 1
    if labels.shape[1] != expected_width:
        raise ValueError(
            f"{spec.flag} {split}_labels width is {labels.shape[1]}; "
            f"expected {expected_width}."
        )
    if not np.issubdtype(labels.dtype, np.number):
        raise ValueError(f"{split}_labels must be numeric, got {labels.dtype}.")
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError(f"{split}_labels must contain finite integer values.")
    labels = labels.astype(np.int64, copy=False)
    if spec.task_type == "multilabel":
        if not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{spec.flag} multilabel answers must be binary.")
    elif labels.min() < 0 or labels.max() >= len(spec.labels):
        raise ValueError(
            f"{spec.flag} labels must be within [0, {len(spec.labels) - 1}]."
        )
    return labels


def _write_split_csvs(
    package_dir: Path,
    spec: TaskSpec,
    labels_by_split: dict[str, np.ndarray],
) -> None:
    split_dir = package_dir / "biomaker_splits"
    private_dir = package_dir / "private_gt"
    split_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    public_names = {
        "train": "agent_train.csv",
        "val": "score_val.csv",
        "test": "agent_test.csv",
    }
    private_names = {"val": "score_val.csv", "test": "test.csv"}
    answers = spec.answer_columns
    public_fields = ["Id", "array_path", "row_index"]

    for split in SPLITS:
        labels = labels_by_split[split]
        fieldnames = public_fields + (answers if split == "train" else [])
        public_path = split_dir / public_names[split]
        private_path = private_dir / private_names[split] if split != "train" else None
        with public_path.open("w", newline="", encoding="utf-8") as public_handle:
            public_writer = csv.DictWriter(public_handle, fieldnames=fieldnames)
            public_writer.writeheader()
            if private_path is None:
                private_handle = None
                private_writer = None
            else:
                private_handle = private_path.open("w", newline="", encoding="utf-8")
                private_writer = csv.DictWriter(
                    private_handle, fieldnames=public_fields + answers
                )
                private_writer.writeheader()
            try:
                array_path = f"input/arrays/{split}_images.npy"
                for index in range(labels.shape[0]):
                    base: dict[str, Any] = {
                        "Id": f"{spec.flag}_{split}_{index:06d}",
                        "array_path": array_path,
                        "row_index": index,
                    }
                    answer_values = {
                        column: int(labels[index, answer_index])
                        for answer_index, column in enumerate(answers)
                    }
                    public_writer.writerow(
                        {**base, **answer_values} if split == "train" else base
                    )
                    if private_writer is not None:
                        private_writer.writerow({**base, **answer_values})
            finally:
                if private_handle is not None:
                    private_handle.close()

    hidden = {
        "score_val": {"file": "score_val.csv", "answer_columns": answers},
        "test": {"file": "test.csv", "answer_columns": answers},
    }
    _write_json(private_dir / "private_gt.json", hidden)


def _description_text(spec: TaskSpec, resolution: int) -> str:
    answer_text = (
        "14 independent binary label columns"
        if spec.task_type == "multilabel"
        else "the integer `class` column"
    )
    metric_text = {
        "mean_roc_auc": "mean one-vs-rest ROC AUC across all 14 labels",
        "ordinal_accuracy": (
            "ordinal accuracy: argmax prediction is correct when its absolute "
            "grade error is at most one"
        ),
        "accuracy": "classification accuracy",
    }[spec.primary_metric]
    labels = "\n".join(f"- {index}: {label}" for index, label in enumerate(spec.labels))
    return (
        f"Dataset: {spec.display_name} (MedMNIST {MEDMNIST_RELEASE})\n"
        f"Task family: {spec.dimension.upper()} {spec.task_type} classification\n"
        f"Resolution: {resolution}\n"
        f"Primary metric: {metric_text}\n\n"
        "The official train/val/test split is preserved exactly. Read "
        "`input/biomaker_splits/*.csv`. Each row identifies one sample using "
        "`array_path` plus zero-based `row_index`; load the referenced NPY once "
        "with `numpy.load(path, mmap_mode='r')` and index it by `row_index`. "
        f"Only agent_train.csv contains answers ({answer_text}). Validation and "
        "test answers are evaluator-only.\n\n"
        f"Labels:\n{labels}\n"
    )


def _task_metadata(
    spec: TaskSpec,
    source_name: str,
    source_md5: str,
    source_sha256: str,
    resolution: int,
    task_text_sha256: str,
    image_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "biomaker_medmnist_task",
        "schema_version": SCHEMA_VERSION,
        "medmnist_release": MEDMNIST_RELEASE,
        "zenodo_record": ZENODO_RECORD,
        "flag": spec.flag,
        "display_name": spec.display_name,
        "dimension": spec.dimension,
        "resolution": resolution,
        "channels": spec.channels,
        "task_type": spec.task_type,
        "primary_metric": spec.primary_metric,
        "class_labels": list(range(len(spec.labels))),
        "label_names": {str(index): label for index, label in enumerate(spec.labels)},
        "answer_columns": spec.answer_columns,
        "prediction_columns": spec.prediction_columns,
        "official_splits_preserved": True,
        "split_counts": image_meta,
        "license": spec.license,
        "source": {
            "archive_name": source_name,
            "url": (
                f"https://zenodo.org/records/{ZENODO_RECORD}/files/"
                f"{source_name}?download=1"
            ),
            "md5": source_md5,
            "sha256": source_sha256,
            "copied_into_data_dir": False,
        },
        "task_text_sha256": task_text_sha256,
        "storage": {
            "format": "npy",
            "one_array_per_split": True,
            "label_free_image_arrays": True,
            "row_locator_columns": ["array_path", "row_index"],
        },
    }


def _split_manifest(
    spec: TaskSpec,
    resolution: int,
    source_name: str,
    source_md5: str,
    source_sha256: str,
    task_text_sha256: str,
    package_dir: Path,
    image_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    split_files = {
        "train": package_dir / "biomaker_splits" / "agent_train.csv",
        "score_val": package_dir / "biomaker_splits" / "score_val.csv",
        "test": package_dir / "biomaker_splits" / "agent_test.csv",
    }
    task_head: dict[str, Any] = {
        "type": "classification",
        "subtype": spec.task_type,
        "answer_columns": spec.answer_columns,
        "class_count": len(spec.labels),
    }
    if spec.task_type == "multilabel":
        task_head["label_count"] = len(spec.labels)
    metric_contract: dict[str, Any] = {
        "primary_metric": spec.primary_metric,
        "maximize": True,
    }
    if spec.primary_metric == "ordinal_accuracy":
        metric_contract["definition"] = (
            "abs(argmax_class_index - true_class_index) <= 1"
        )
    elif spec.primary_metric == "mean_roc_auc":
        metric_contract["definition"] = "macro mean of per-label binary ROC AUC"
    hidden = {
        "score_val": {"answer_columns": spec.answer_columns},
        "test": {"answer_columns": spec.answer_columns},
    }
    prediction_contract: dict[str, Any] = {
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
        "score_split": "./input/biomaker_splits/score_val.csv",
        "key_columns": ["Id"],
        "prediction_columns": spec.prediction_columns,
        "file_reference_columns": [],
        "prediction_type": spec.task_type,
        "class_labels": list(range(len(spec.labels))),
        "primary_metric": spec.primary_metric,
        "maximize": True,
    }
    if spec.task_type == "binary":
        prediction_contract["positive_class"] = 1
    return {
        "schema": "biomaker_split_manifest",
        "manifest_version": 1,
        "package_schema_version": SCHEMA_VERSION,
        "sample_unit": "image" if spec.dimension == "2d" else "volume",
        "row_representation": "one_row_per_sample",
        "key_columns": ["Id"],
        "task_heads": [task_head],
        "splits": {
            split: {
                "file": str(path.relative_to(package_dir)),
                "row_count": image_meta["val" if split == "score_val" else split][
                    "row_count"
                ],
                "sha256": _digest(path, "sha256"),
            }
            for split, path in split_files.items()
        },
        "hidden_answers": hidden,
        "visibility": {
            "agent_visible": [
                "biomaker_splits/*.csv",
                "arrays/*_images.npy",
                "data_description.txt",
            ],
            "evaluator_only": ["private_gt/**"],
            "forbidden": ["*.npz", "**/*_labels.npy"],
        },
        "prediction_contract": prediction_contract,
        "metric_contract": metric_contract,
        "array_contract": {
            "path_column": "array_path",
            "row_index_column": "row_index",
            "index_origin": 0,
            "format": "npy",
            "mmap_compatible": True,
            "split_arrays": {
                split: {
                    "path": f"input/arrays/{split}_images.npy",
                    **image_meta[split],
                }
                for split in SPLITS
            },
        },
        "source_contract": {
            "dataset": spec.flag,
            "medmnist_release": MEDMNIST_RELEASE,
            "resolution": resolution,
            "source_archive_name": source_name,
            "source_md5": source_md5,
            "source_sha256": source_sha256,
            "task_text_sha256": task_text_sha256,
            "official_splits_preserved": True,
            "source_archive_agent_visible": False,
        },
    }


def _inspect_and_extract(
    source: Path,
    destination_data_dir: Path,
    spec: TaskSpec,
    resolution: int,
    *,
    strict_official: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    required = {
        f"{split}_{kind}.npy" for split in SPLITS for kind in ("images", "labels")
    }
    labels_by_split: dict[str, np.ndarray] = {}
    image_meta: dict[str, dict[str, Any]] = {}
    sample_shape: tuple[int, ...] | None = None
    sample_dtype: str | None = None
    with zipfile.ZipFile(source, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"{source.name} contains duplicate ZIP member names.")
        missing = sorted(required - set(names))
        if missing:
            raise ValueError(f"{source.name} is missing NPZ members: {missing}.")
        for split in SPLITS:
            image_member = f"{split}_images.npy"
            label_member = f"{split}_labels.npy"
            shape, fortran_order, dtype = _npy_header(archive, image_member)
            shape = tuple(map(int, shape))
            _validate_image_shape(
                spec, resolution, shape, dtype, strict_official=strict_official
            )
            labels = _normalize_labels(
                spec, _load_label_array(archive, label_member), split
            )
            if shape[0] != labels.shape[0]:
                raise ValueError(
                    f"{split} image/label rows differ: {shape[0]} vs {labels.shape[0]}."
                )
            if strict_official and shape[0] != spec.count_by_split[split]:
                raise ValueError(
                    f"{spec.flag} official {split} count should be "
                    f"{spec.count_by_split[split]}, got {shape[0]}."
                )
            if sample_shape is None:
                sample_shape = shape[1:]
                sample_dtype = str(dtype)
            elif shape[1:] != sample_shape or str(dtype) != sample_dtype:
                raise ValueError(
                    f"Image sample shape/dtype changed across splits: "
                    f"{sample_shape}/{sample_dtype} vs {shape[1:]}/{dtype}."
                )
            destination = destination_data_dir / "arrays" / image_member
            _copy_image_member(archive, image_member, destination)
            loaded = np.load(destination, mmap_mode="r", allow_pickle=False)
            if tuple(loaded.shape) != shape or loaded.dtype != dtype:
                raise ValueError(f"Copied NPY verification failed for {destination}.")
            del loaded
            labels_by_split[split] = labels
            image_meta[split] = {
                "row_count": shape[0],
                "sample_shape": list(shape[1:]),
                "dtype": str(dtype),
                "fortran_order": bool(fortran_order),
                "npy_sha256": _digest(destination, "sha256"),
                "size_bytes": destination.stat().st_size,
            }
    return labels_by_split, image_meta


def _cache_payload(
    package_dir: Path,
    *,
    verifier: dict[str, Any],
    spec: TaskSpec,
    source_md5: str,
    source_sha256: str,
    task_text_sha256: str,
) -> None:
    _write_json(package_dir / "verifier.json", verifier)
    _write_json(
        package_dir / "manifest.json",
        {
            "cache": "split_builder",
            "package_schema": "biomaker_fig2_medmnist",
            "package_schema_version": SCHEMA_VERSION,
            "dataset": spec.flag,
            "source_md5": source_md5,
            "source_sha256": source_sha256,
            "task_text_sha256": task_text_sha256,
            "verifier_ok": bool(verifier.get("ok")),
        },
    )


def _install_directory(source: Path, destination: Path, *, replace: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not replace:
            raise FileExistsError(destination)
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    source.rename(destination)


def _copy_directory_atomically(
    source: Path, destination: Path, *, replace: bool
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        shutil.rmtree(temp)
        shutil.copytree(source, temp)
        if destination.exists() or destination.is_symlink():
            if not replace:
                shutil.rmtree(temp)
                return
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        temp.rename(destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def _ready_matches(
    ready: dict[str, Any],
    *,
    spec: TaskSpec,
    source_md5: str,
    source_sha256: str,
    task_text_sha256: str,
) -> bool:
    return all(
        (
            ready.get("status") == "ready",
            ready.get("flag") == spec.flag,
            ready.get("source_md5") == source_md5,
            ready.get("source_sha256") == source_sha256,
            ready.get("task_text_sha256") == task_text_sha256,
        )
    )


def _suite_entry(
    *,
    spec: TaskSpec,
    resolution: int,
    source: Path,
    source_md5: str,
    source_sha256: str,
    task_text_sha256: str,
    task_desc_path: Path,
    data_dir: Path,
    package_dir: Path,
    cache_root: Path,
    cache_key: str,
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "task_id": spec.flag,
        "flag": spec.flag,
        "medmnist_flag": spec.flag,
        "dimension": spec.dimension,
        "resolution": resolution,
        "task_type": spec.task_type,
        "primary_metric": spec.primary_metric,
        "status": "ready",
        "source_archive": source.name,
        "source_md5": source_md5,
        "source_sha256": source_sha256,
        "task_text_sha256": task_text_sha256,
        "desc_file": str(task_desc_path.resolve()),
        "data_dir_name": data_dir.name,
        "data_dir": str(data_dir.resolve()),
        "task_data_dir": str(data_dir.resolve()),
        "package_dir": str(package_dir.resolve()),
        "ready_marker": str((data_dir / "_READY.json").resolve()),
        "cache_key": cache_key,
        "cache_entry": str((cache_root / cache_key).resolve()),
        "counts": counts,
    }


def _update_suite_manifest(output_root: Path, entry: dict[str, Any]) -> Path:
    path = output_root / "suite_manifest.json"
    with _exclusive_lock(output_root / ".suite_manifest.lock"):
        if path.is_file():
            current = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                raise ValueError(f"Invalid existing suite manifest: {path}")
        else:
            current = {}
        raw_tasks = current.get("tasks", [])
        tasks = {
            str(item.get("task_id") or item.get("flag")): item
            for item in raw_tasks
            if isinstance(item, dict) and (item.get("task_id") or item.get("flag"))
        }
        tasks[str(entry["task_id"])] = entry
        ordered = [tasks[flag] for flag in TASK_ORDER if flag in tasks]
        extras = [tasks[key] for key in sorted(set(tasks) - set(TASK_ORDER))]
        payload = {
            "schema": "biomaker_fig2_medmnist_suite",
            "schema_version": SCHEMA_VERSION,
            "medmnist_release": MEDMNIST_RELEASE,
            "task_count": len(ordered) + len(extras),
            "tasks": ordered + extras,
        }
        _atomic_write_json(path, payload)
    return path


def _verify_installed_cache(
    cache_entry: Path, data_dir: Path
) -> tuple[bool, dict[str, Any]]:
    if not (cache_entry / "manifest.json").is_file():
        return False, {"ok": False, "errors": ["cache manifest missing"]}
    with _workspace_input_view(data_dir) as input_dir:
        facts = build_dataset_facts(input_dir)
        verifier = verify_split_package(
            cache_entry, source_input_dir=input_dir, facts=facts
        )
    return bool(verifier.get("ok")), verifier


def prepare_task(
    source: Path,
    *,
    task_desc_path: Path,
    output_root: Path,
    cache_root: Path,
    model: str,
    temperature: float = 0.0,
    strict_official: bool = True,
    replace: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    task_desc_path = task_desc_path.resolve()
    output_root = output_root.resolve()
    cache_root = cache_root.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not task_desc_path.is_file():
        raise FileNotFoundError(task_desc_path)
    spec, resolution = _parse_source_name(source)
    task_text = task_desc_path.read_text(encoding="utf-8")
    task_text_sha256 = _sha256_text(task_text)
    source_md5 = _digest(source, "md5")
    source_sha256 = _digest(source, "sha256")
    expected_md5 = spec.md5_by_resolution[resolution]
    if strict_official and source_md5 != expected_md5:
        raise ValueError(
            f"{source.name} MD5 mismatch: expected {expected_md5}, got {source_md5}."
        )

    data_dir = output_root / "data" / source.stem
    package_dir = output_root / "packages" / source.stem
    ready_path = data_dir / "_READY.json"
    if ready_path.is_file() and not replace:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if not _ready_matches(
            ready,
            spec=spec,
            source_md5=source_md5,
            source_sha256=source_sha256,
            task_text_sha256=task_text_sha256,
        ):
            raise FileExistsError(
                f"Existing task differs from requested source/task text: {data_dir}"
            )
        cache_key, _facts = _split_cache_key_for_data_dir(
            data_dir,
            task_text=task_text,
            model=model,
            temperature=temperature,
        )
        cache_entry = cache_root / cache_key
        data_hashes = verify_package_integrity(cache_entry, data_dir)
        ok, verifier = _verify_installed_cache(cache_entry, data_dir)
        if not ok:
            raise RuntimeError(
                "Existing ready task has no valid matching cache: "
                + json.dumps(verifier, ensure_ascii=False)
            )
        if not package_dir.is_dir():
            _copy_directory_atomically(cache_entry, package_dir, replace=False)
        elif package_dir.resolve() != cache_entry.resolve():
            # The large arrays were already hashed while verifying the cache.
            verify_package_integrity(
                package_dir, data_dir, validated_data_hashes=data_hashes
            )
        counts = {key: int(value) for key, value in ready["counts"].items()}
        entry = _suite_entry(
            spec=spec,
            resolution=resolution,
            source=source,
            source_md5=source_md5,
            source_sha256=source_sha256,
            task_text_sha256=task_text_sha256,
            task_desc_path=task_desc_path,
            data_dir=data_dir,
            package_dir=package_dir,
            cache_root=cache_root,
            cache_key=cache_key,
            counts=counts,
        )
        _update_suite_manifest(output_root, entry)
        return {**entry, "reused": True}

    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".medmnist-prepare-", dir=output_root))
    stage_data = staging_root / source.stem
    stage_package = staging_root / "package"
    stage_data.mkdir(parents=True)
    stage_package.mkdir(parents=True)
    try:
        labels_by_split, image_meta = _inspect_and_extract(
            source,
            stage_data,
            spec,
            resolution,
            strict_official=strict_official,
        )
        counts = {split: int(labels_by_split[split].shape[0]) for split in SPLITS}
        _write_split_csvs(stage_package, spec, labels_by_split)
        metadata = _task_metadata(
            spec,
            source.name,
            source_md5,
            source_sha256,
            resolution,
            task_text_sha256,
            image_meta,
        )
        _write_json(stage_data / "task_metadata.json", metadata)
        (stage_data / "data_description.txt").write_text(
            _description_text(spec, resolution), encoding="utf-8"
        )
        ready = {
            "schema": "biomaker_medmnist_ready",
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "flag": spec.flag,
            "source_archive": source.name,
            "source_md5": source_md5,
            "source_sha256": source_sha256,
            "task_text_sha256": task_text_sha256,
            "counts": counts,
            "dimension": spec.dimension,
            "resolution": resolution,
            "label_free_arrays": True,
            "source_archive_agent_visible": False,
        }
        _write_json(stage_data / "_READY.json", ready)

        cache_key, facts = _split_cache_key_for_data_dir(
            stage_data,
            task_text=task_text,
            model=model,
            temperature=temperature,
        )
        manifest = _split_manifest(
            spec,
            resolution,
            source.name,
            source_md5,
            source_sha256,
            task_text_sha256,
            stage_package,
            image_meta,
        )
        _write_json(stage_package / "split_manifest.json", manifest)
        with _workspace_input_view(stage_data) as input_dir:
            verifier = verify_split_package(
                stage_package, source_input_dir=input_dir, facts=facts
            )
        if not verifier.get("ok"):
            raise RuntimeError(
                "Generated split package failed verification: "
                + json.dumps(verifier, ensure_ascii=False)
            )
        _cache_payload(
            stage_package,
            verifier=verifier,
            spec=spec,
            source_md5=source_md5,
            source_sha256=source_sha256,
            task_text_sha256=task_text_sha256,
        )
        write_package_integrity(stage_package, stage_data)

        cache_entry = cache_root / cache_key
        with _exclusive_lock(cache_root / ".locks" / f"{cache_key}.lock"):
            if cache_entry.exists() and replace:
                shutil.rmtree(cache_entry)
            if not cache_entry.exists():
                _copy_directory_atomically(stage_package, cache_entry, replace=False)
            verify_package_integrity(cache_entry, stage_data)
            cache_ok, cache_verifier = _verify_installed_cache(cache_entry, stage_data)
            if not cache_ok:
                raise RuntimeError(
                    "Installed cache failed verification: "
                    + json.dumps(cache_verifier, ensure_ascii=False)
                )

        package_copy = staging_root / "package-install"
        shutil.copytree(stage_package, package_copy)
        _install_directory(package_copy, package_dir, replace=replace)
        _install_directory(stage_data, data_dir, replace=replace)

        actual_key, _ = _split_cache_key_for_data_dir(
            data_dir,
            task_text=task_text,
            model=model,
            temperature=temperature,
        )
        if actual_key != cache_key:
            raise RuntimeError(
                f"Cache key changed after atomic install: {cache_key} -> {actual_key}."
            )
        entry = _suite_entry(
            spec=spec,
            resolution=resolution,
            source=source,
            source_md5=source_md5,
            source_sha256=source_sha256,
            task_text_sha256=task_text_sha256,
            task_desc_path=task_desc_path,
            data_dir=data_dir,
            package_dir=package_dir,
            cache_root=cache_root,
            cache_key=cache_key,
            counts=counts,
        )
        _update_suite_manifest(output_root, entry)
        return {**entry, "reused": False}
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def _parse_desc_overrides(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--task-desc must use FLAG=PATH syntax.")
        flag, raw_path = value.split("=", 1)
        flag = flag.strip().lower()
        if flag not in TASK_SPECS:
            raise ValueError(f"Unsupported --task-desc flag: {flag}")
        result[flag] = Path(raw_path).expanduser()
    return result


def _source_paths(explicit: Sequence[str], npz_dir: Path | None) -> list[Path]:
    paths = [Path(value).expanduser() for value in explicit]
    if npz_dir is not None:
        if not npz_dir.is_dir():
            raise FileNotFoundError(npz_dir)
        paths.extend(sorted(npz_dir.glob("*.npz")))
    unique = {str(path.resolve()): path.resolve() for path in paths}
    return list(unique.values())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz", action="append", default=[], help="Source NPZ; repeatable."
    )
    parser.add_argument(
        "--npz-dir", type=Path, help="Directory containing downloaded NPZs."
    )
    parser.add_argument("--task", action="append", choices=TASK_ORDER, default=[])
    parser.add_argument(
        "--task-desc-dir",
        type=Path,
        default=WORKSPACE_ROOT / "experiments" / "fig2_medmnist" / "tasks",
        help="Directory containing exact <flag>.md task descriptions.",
    )
    parser.add_argument(
        "--task-desc",
        action="append",
        default=[],
        metavar="FLAG=PATH",
        help="Override one exact task-description file.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="Exact resource_dir/split_builder directory used by BioMaker.",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--allow-nonofficial-source",
        action="store_true",
        help="Skip official checksum/count/spatial-size checks (tests only).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the exact existing task/package/cache targets.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require all 13 frozen task NPZs instead of packaging available files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sources = _source_paths(args.npz, args.npz_dir)
    selected_flags = set(args.task or TASK_ORDER)
    parsed: dict[str, Path] = {}
    for source in sources:
        try:
            spec, _resolution = _parse_source_name(source)
        except ValueError:
            if args.npz:
                raise
            continue
        if spec.flag not in selected_flags:
            continue
        if spec.flag in parsed:
            raise ValueError(
                f"Multiple source resolutions supplied for {spec.flag}: "
                f"{parsed[spec.flag]} and {source}."
            )
        parsed[spec.flag] = source
    missing = selected_flags - set(parsed)
    if (args.require_all or args.task) and missing:
        raise FileNotFoundError(
            "Missing selected MedMNIST NPZs: " + ", ".join(sorted(missing))
        )
    if not parsed:
        raise FileNotFoundError("No supported Fig. 2 MedMNIST NPZs were found.")

    desc_overrides = _parse_desc_overrides(args.task_desc)
    results = []
    for flag in TASK_ORDER:
        source = parsed.get(flag)
        if source is None:
            continue
        desc_path = desc_overrides.get(flag, args.task_desc_dir / f"{flag}.md")
        result = prepare_task(
            source,
            task_desc_path=desc_path,
            output_root=args.output_root,
            cache_root=args.cache_root,
            model=args.model,
            temperature=args.temperature,
            strict_official=not args.allow_nonofficial_source,
            replace=args.replace,
        )
        results.append(result)
        state = "reused" if result["reused"] else "prepared"
        print(
            f"{flag}: {state}; data_dir={result['data_dir']}; "
            f"cache_key={result['cache_key']}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "prepared": len(results),
                "suite_manifest": str(
                    (args.output_root.resolve() / "suite_manifest.json")
                ),
                "tasks": [result["task_id"] for result in results],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
