#!/usr/bin/env python3
"""Build deterministic, sealed BioMaker TaskPacks for Main-9 segmentation tasks.

Raw archives remain outside the agent-visible task directory.  Training masks
are visible, while score-validation and test masks are installed only in the
split-builder cache's ``private_gt`` tree.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageDraw

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
DEFAULT_SEED = 42
TASK_ORDER = ("monuseg", "kvasir_seg", "busi", "aeropath")
MANIFEST_TASK_ORDER = (
    "monuseg",
    "kvasir_seg",
    "busi",
    "bccd",
    "fracatlas",
    "dentex",
    "fmd",
    "aeropath",
    "mp_idb",
)
SPLIT_NAMES = ("train", "score_val", "test")
SPLIT_RATIOS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    group_id: str
    source_partition: str
    image_member: str
    mask_members: tuple[str, ...]
    stratum: str = "all"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    display_name: str
    task_head: str
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    sample_unit: str
    license: str


SPECS = {
    "monuseg": TaskSpec(
        "monuseg",
        "MoNuSeg 2018",
        "instance_segmentation",
        "aji",
        ("binary_dice", "hd95"),
        "whole-slide image patch grouped by TCGA patient",
        "CC BY-NC-SA 4.0",
    ),
    "kvasir_seg": TaskSpec(
        "kvasir_seg",
        "Kvasir-SEG",
        "binary_segmentation",
        "binary_dice",
        ("hd95",),
        "endoscopy image",
        "CC BY 4.0",
    ),
    "busi": TaskSpec(
        "busi",
        "Dataset of Breast Ultrasound Images (BUSI)",
        "binary_segmentation",
        "binary_dice",
        ("hd95",),
        "breast ultrasound image",
        "CC BY 4.0",
    ),
    "aeropath": TaskSpec(
        "aeropath",
        "AeroPath",
        "segmentation_3d",
        "binary_dice",
        ("hd95",),
        "patient CT volume",
        "CC BY 4.0",
    ),
}

FROZEN_ARCHIVES = {
    "MoNuSeg/MoNuSeg_2018_Training_Data.zip": (
        125_778_974,
        "25d3d3185bb2970b397cafa72eb664c9b4d24294aee382e7e3df9885affce742",
    ),
    "MoNuSeg/MoNuSegTestData.zip": (
        41_620_178,
        "13e522387ae8b1bcc0530e13ff9c7b4d91ec74959ef6f6e57747368d7ee6f88a",
    ),
    "Kvasir-SEG/kvasir-seg.zip": (
        46_227_172,
        "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7",
    ),
    "BUSI/Dataset_BUSI.zip": (
        205_873_341,
        "b2ce09f6063a31a73f628b6a6ee1245187cbaec225e93e563735691d68654de7",
    ),
}


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    with tempfile.TemporaryDirectory(prefix="biomaker-main9-input-") as temp:
        input_dir = Path(temp) / "input"
        input_dir.mkdir()
        for source in sorted(data_dir.iterdir(), key=lambda path: path.name):
            (input_dir / source.name).symlink_to(source.resolve())
        yield input_dir


def _split_cache_key(
    data_dir: Path, *, task_text: str, model: str, temperature: float
) -> tuple[str, dict[str, Any]]:
    with _workspace_input_view(data_dir) as input_dir:
        return split_builder_cache_key(
            input_dir=input_dir,
            task_text=task_text,
            model=model,
            temperature=temperature,
        )


def _validate_zip(
    path: Path, relative_path: str, *, strict_official: bool
) -> SourceFile:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    sha256 = _sha256_file(path)
    if strict_official:
        expected_size, expected_sha256 = FROZEN_ARCHIVES[relative_path]
        if size != expected_size:
            raise ValueError(
                f"{relative_path} size mismatch: expected {expected_size}, got {size}."
            )
        if sha256 != expected_sha256:
            raise ValueError(
                f"{relative_path} SHA-256 mismatch: expected {expected_sha256}, got {sha256}."
            )
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"{relative_path} failed ZIP CRC at {bad_member}.")
    return SourceFile(relative_path, size, sha256)


def _patient_id(sample_id: str) -> str:
    parts = sample_id.split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3]).upper()
    return sample_id


def _monuseg_members(archive: zipfile.ZipFile, *, partition: str) -> list[Sample]:
    images: dict[str, str] = {}
    masks: dict[str, str] = {}
    for member in archive.namelist():
        if member.startswith("__MACOSX/") or member.endswith("/"):
            continue
        path = PurePosixPath(member)
        suffix = path.suffix.lower()
        stem = path.stem
        if suffix in {".tif", ".tiff"}:
            images[stem] = member
        elif suffix == ".xml":
            masks[stem] = member
    if set(images) != set(masks):
        raise ValueError(
            f"MoNuSeg {partition} image/XML IDs differ: "
            f"images_only={sorted(set(images) - set(masks))[:5]}, "
            f"xml_only={sorted(set(masks) - set(images))[:5]}."
        )
    return [
        Sample(
            sample_id=sample_id,
            group_id=_patient_id(sample_id),
            source_partition=partition,
            image_member=images[sample_id],
            mask_members=(masks[sample_id],),
        )
        for sample_id in sorted(images)
    ]


def _grouped_holdout(
    samples: Sequence[Sample], *, seed: int, fraction: float
) -> tuple[list[Sample], list[Sample]]:
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_id, []).append(sample)
    if len(groups) < 2:
        raise ValueError("At least two independent groups are required for validation.")
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    target = max(1, math.ceil(len(samples) * fraction))
    selected: set[str] = set()
    selected_count = 0
    for group_id in group_ids:
        if len(selected) >= len(groups) - 1:
            break
        selected.add(group_id)
        selected_count += len(groups[group_id])
        if selected_count >= target:
            break
    train = sorted(
        (sample for sample in samples if sample.group_id not in selected),
        key=lambda sample: sample.sample_id,
    )
    holdout = sorted(
        (sample for sample in samples if sample.group_id in selected),
        key=lambda sample: sample.sample_id,
    )
    if not train or not holdout:
        raise ValueError("Deterministic grouped split produced an empty partition.")
    return train, holdout


def _read_image(archive: zipfile.ZipFile, member: str) -> Image.Image:
    payload = archive.read(member)
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return image.copy()


def _xml_instance_mask(xml_bytes: bytes, size: tuple[int, int]) -> np.ndarray:
    root = ET.fromstring(xml_bytes)
    labels = np.zeros((size[1], size[0]), dtype=np.uint16)
    next_label = 1
    for region in root.iter("Region"):
        vertices = []
        for vertex in region.iter("Vertex"):
            try:
                x = int(round(float(vertex.attrib["X"])))
                y = int(round(float(vertex.attrib["Y"])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("MoNuSeg XML contains an invalid vertex.") from exc
            vertices.append((x, y))
        if len(vertices) < 3:
            continue
        if next_label > np.iinfo(np.uint16).max:
            raise ValueError("MoNuSeg XML has more instances than uint16 can encode.")
        polygon = Image.new("1", size, 0)
        ImageDraw.Draw(polygon).polygon(vertices, fill=1)
        region_mask = np.asarray(polygon, dtype=bool)
        labels[np.logical_and(region_mask, labels == 0)] = next_label
        next_label += 1
    return labels


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", optimize=False)


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _apportion(total: int, ratios: Sequence[float]) -> list[int]:
    if total < 0 or not ratios or any(ratio < 0 for ratio in ratios):
        raise ValueError(
            "Split apportionment requires a non-negative total and ratios."
        )
    ratio_sum = sum(ratios)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to one; got {ratio_sum}.")
    exact = [total * ratio for ratio in ratios]
    result = [math.floor(value) for value in exact]
    remainder = total - sum(result)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(exact[index] - result[index]), index),
    )
    for index in order[:remainder]:
        result[index] += 1
    return result


def _stratified_group_split(
    samples: Sequence[Sample], *, seed: int
) -> dict[str, list[Sample]]:
    """Make an exact-size three-way split without separating any source group."""

    if len(samples) < len(SPLIT_NAMES):
        raise ValueError("At least three samples are required for a three-way split.")
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_id, []).append(sample)
    if len(groups) < len(SPLIT_NAMES):
        raise ValueError("At least three independent groups are required.")

    target_sizes = dict(zip(SPLIT_NAMES, _apportion(len(samples), SPLIT_RATIOS)))
    if any(value == 0 for value in target_sizes.values()):
        raise ValueError(
            f"Three-way split would contain an empty partition: {target_sizes}."
        )
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    # Place the largest and mixed-stratum groups first. All official Kvasir groups
    # are singletons; BUSI has one exact-image duplicate pair that must stay intact.
    group_ids.sort(
        key=lambda group_id: (
            -len(groups[group_id]),
            -len({sample.stratum for sample in groups[group_id]}),
        )
    )

    strata = sorted({sample.stratum for sample in samples})
    stratum_totals = Counter(sample.stratum for sample in samples)
    assigned_sizes = {name: 0 for name in SPLIT_NAMES}
    assigned_strata = {name: Counter() for name in SPLIT_NAMES}
    assignments: dict[str, str] = {}

    def assignment_error(split: str, group: Sequence[Sample]) -> float:
        trial_sizes = dict(assigned_sizes)
        trial_sizes[split] += len(group)
        trial_strata = {
            name: Counter(counts) for name, counts in assigned_strata.items()
        }
        trial_strata[split].update(sample.stratum for sample in group)
        total_error = sum(
            ((trial_sizes[name] - target_sizes[name]) ** 2) / max(1, target_sizes[name])
            for name in SPLIT_NAMES
        )
        stratum_error = 0.0
        for name, ratio in zip(SPLIT_NAMES, SPLIT_RATIOS):
            for stratum in strata:
                target = stratum_totals[stratum] * ratio
                stratum_error += (trial_strata[name][stratum] - target) ** 2 / max(
                    1.0, target
                )
        return total_error + stratum_error

    for group_id in group_ids:
        group = groups[group_id]
        candidates = [
            name
            for name in SPLIT_NAMES
            if assigned_sizes[name] + len(group) <= target_sizes[name]
        ]
        if not candidates:
            raise ValueError(
                "Grouped split cannot meet the exact partition sizes; "
                f"unplaced group={group_id!r}, size={len(group)}, "
                f"assigned={assigned_sizes}, targets={target_sizes}."
            )
        chosen = min(
            candidates,
            key=lambda name: (assignment_error(name, group), SPLIT_NAMES.index(name)),
        )
        assignments[group_id] = chosen
        assigned_sizes[chosen] += len(group)
        assigned_strata[chosen].update(sample.stratum for sample in group)

    if assigned_sizes != target_sizes:
        raise ValueError(
            f"Grouped split size mismatch: expected {target_sizes}, got {assigned_sizes}."
        )
    partitions = {name: [] for name in SPLIT_NAMES}
    for sample in samples:
        partitions[assignments[sample.group_id]].append(sample)
    for name in SPLIT_NAMES:
        partitions[name].sort(key=lambda sample: sample.sample_id)
    return partitions


def _binary_union_mask(
    archive: zipfile.ZipFile,
    mask_members: Sequence[str],
    expected_size: tuple[int, int],
) -> np.ndarray:
    if not mask_members:
        raise ValueError("A segmentation sample has no source mask.")
    union = np.zeros((expected_size[1], expected_size[0]), dtype=bool)
    for member in mask_members:
        mask = _read_image(archive, member).convert("L")
        if mask.size != expected_size:
            raise ValueError(
                f"Mask/image size mismatch for {member}: {mask.size} vs {expected_size}."
            )
        # Kvasir masks are JPEGs and contain small compression values around zero.
        union |= np.asarray(mask, dtype=np.uint8) >= 128
    return np.where(union, 255, 0).astype(np.uint8)


def _unique_member_map(
    archive: zipfile.ZipFile,
    predicate: Callable[[PurePosixPath], bool],
    *,
    label: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for member in archive.namelist():
        if member.startswith("__MACOSX/") or member.endswith("/"):
            continue
        path = PurePosixPath(member)
        if not predicate(path):
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate {label} stem in archive: {path.stem}.")
        result[path.stem] = member
    return result


def _kvasir_samples(
    archive: zipfile.ZipFile, *, strict_official: bool
) -> tuple[list[Sample], dict[str, Any]]:
    image_suffixes = {".jpg", ".jpeg", ".png"}
    images = _unique_member_map(
        archive,
        lambda path: path.parent.name.lower() == "images"
        and path.suffix.lower() in image_suffixes,
        label="Kvasir image",
    )
    masks = _unique_member_map(
        archive,
        lambda path: path.parent.name.lower() == "masks"
        and path.suffix.lower() in image_suffixes,
        label="Kvasir mask",
    )
    if set(images) != set(masks):
        raise ValueError(
            "Kvasir image/mask IDs differ: "
            f"images_only={sorted(set(images) - set(masks))[:5]}, "
            f"masks_only={sorted(set(masks) - set(images))[:5]}."
        )
    if strict_official and len(images) != 1000:
        raise ValueError(
            f"Frozen Kvasir-SEG source must contain 1000 pairs; got {len(images)}."
        )

    samples: list[Sample] = []
    group_sizes: Counter[str] = Counter()
    area_counts: Counter[str] = Counter()
    for index, source_id in enumerate(sorted(images), start=1):
        image_payload = archive.read(images[source_id])
        image = _read_image(archive, images[source_id])
        binary_mask = _binary_union_mask(archive, (masks[source_id],), image.size)
        foreground_fraction = float(np.count_nonzero(binary_mask)) / binary_mask.size
        if not np.any(binary_mask):
            raise ValueError(f"Kvasir mask is empty for source ID {source_id}.")
        if foreground_fraction < 0.05:
            stratum = "foreground_lt_5pct"
        elif foreground_fraction < 0.20:
            stratum = "foreground_5_to_20pct"
        else:
            stratum = "foreground_ge_20pct"
        group_id = hashlib.sha256(image_payload).hexdigest()
        group_sizes[group_id] += 1
        area_counts[stratum] += 1
        samples.append(
            Sample(
                sample_id=f"KVS-{index:04d}",
                group_id=group_id,
                source_partition="complete_public_set",
                image_member=images[source_id],
                mask_members=(masks[source_id],),
                stratum=stratum,
            )
        )
    return samples, {
        "source_pair_count": len(samples),
        "foreground_area_strata": dict(sorted(area_counts.items())),
        "exact_duplicate_image_groups": sum(
            value > 1 for value in group_sizes.values()
        ),
        "patient_or_video_ids_available": False,
    }


def _busi_mask_order(member: str) -> tuple[int, str]:
    match = re.search(r"_mask(?:_(\d+))?$", PurePosixPath(member).stem, re.IGNORECASE)
    return (int(match.group(1)) if match and match.group(1) else 0, member)


def _busi_samples(
    archive: zipfile.ZipFile, *, strict_official: bool
) -> tuple[list[Sample], dict[str, Any]]:
    categories = ("benign", "malignant", "normal")
    images: dict[tuple[str, str], str] = {}
    masks: dict[tuple[str, str], list[str]] = {}
    for member in archive.namelist():
        if member.startswith("__MACOSX/") or member.endswith("/"):
            continue
        path = PurePosixPath(member)
        category = path.parent.name.lower()
        if category not in categories or path.suffix.lower() != ".png":
            continue
        match = re.search(r"_mask(?:_\d+)?$", path.stem, re.IGNORECASE)
        if match:
            key = (category, path.stem[: match.start()])
            masks.setdefault(key, []).append(member)
        else:
            key = (category, path.stem)
            if key in images:
                raise ValueError(f"Duplicate BUSI image record: {key}.")
            images[key] = member
    if set(images) != set(masks):
        raise ValueError(
            "BUSI image/mask records differ: "
            f"images_only={sorted(set(images) - set(masks))[:5]}, "
            f"masks_only={sorted(set(masks) - set(images))[:5]}."
        )
    category_counts = Counter(category for category, _ in images)
    mask_count = sum(len(value) for value in masks.values())
    if strict_official:
        expected_categories = {"benign": 437, "malignant": 210, "normal": 133}
        if len(images) != 780 or dict(category_counts) != expected_categories:
            raise ValueError(
                "Frozen BUSI source must contain 780 images with category counts "
                f"{expected_categories}; got {len(images)} and {dict(category_counts)}."
            )
        if mask_count != 798:
            raise ValueError(
                f"Frozen BUSI source must contain 798 mask files; got {mask_count}."
            )

    def source_order(key: tuple[str, str]) -> tuple[int, int, str]:
        category, stem = key
        number = re.search(r"\((\d+)\)$", stem)
        return (
            categories.index(category),
            int(number.group(1)) if number else sys.maxsize,
            stem,
        )

    samples: list[Sample] = []
    group_sizes: Counter[str] = Counter()
    multiplicity: Counter[int] = Counter()
    for index, key in enumerate(sorted(images, key=source_order), start=1):
        category, source_id = key
        image_payload = archive.read(images[key])
        image = _read_image(archive, images[key])
        mask_members = tuple(sorted(masks[key], key=_busi_mask_order))
        binary_mask = _binary_union_mask(archive, mask_members, image.size)
        is_empty = not np.any(binary_mask)
        if category == "normal" and not is_empty:
            raise ValueError(f"BUSI normal source mask is not empty: {source_id}.")
        if category != "normal" and is_empty:
            raise ValueError(f"BUSI lesion source mask is empty: {source_id}.")
        group_id = hashlib.sha256(image_payload).hexdigest()
        group_sizes[group_id] += 1
        multiplicity[len(mask_members)] += 1
        samples.append(
            Sample(
                sample_id=f"BUSI-{index:04d}",
                group_id=group_id,
                source_partition="complete_public_set",
                image_member=images[key],
                mask_members=mask_members,
                stratum=category,
            )
        )
    return samples, {
        "source_image_count": len(samples),
        "source_mask_file_count": mask_count,
        "source_category_counts": dict(sorted(category_counts.items())),
        "mask_multiplicity": {
            str(count): samples_with_count
            for count, samples_with_count in sorted(multiplicity.items())
        },
        "exact_duplicate_image_groups": sum(
            value > 1 for value in group_sizes.values()
        ),
        "patient_ids_available": False,
    }


def _copy_binary_sample(
    archive: zipfile.ZipFile,
    sample: Sample,
    split: str,
    data_dir: Path,
    package_dir: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    image_payload = archive.read(sample.image_member)
    image = _read_image(archive, sample.image_member)
    suffix = PurePosixPath(sample.image_member).suffix.lower()
    image_name = f"{sample.sample_id}{suffix}"
    image_path = data_dir / "images" / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_payload)
    binary_mask = _binary_union_mask(archive, sample.mask_members, image.size)
    public = {"Id": sample.sample_id, "image_path": f"input/images/{image_name}"}
    mask_name = f"{sample.sample_id}.png"
    if split == "train":
        mask_path = data_dir / "train_masks" / mask_name
        _write_png(mask_path, binary_mask)
        public["mask_path"] = f"input/train_masks/{mask_name}"
        private = {}
    else:
        mask_path = package_dir / "private_gt" / "masks" / split / mask_name
        _write_png(mask_path, binary_mask)
        private = {**public, "mask_path": f"private_gt/masks/{split}/{mask_name}"}
    return (
        public,
        private,
        {
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "source_partition": sample.source_partition,
            "split": split,
            "stratum": sample.stratum,
            "source_record_sha256": _sha256_text(
                "\n".join((sample.image_member, *sample.mask_members))
            ),
            "image_sha256": _sha256_file(image_path),
            "mask_sha256": _sha256_file(mask_path),
            "shape": [image.size[1], image.size[0]],
            "foreground_pixels": int(np.count_nonzero(binary_mask)),
            "source_mask_count": len(sample.mask_members),
        },
    )


def _copy_monuseg_sample(
    archive: zipfile.ZipFile,
    sample: Sample,
    split: str,
    data_dir: Path,
    package_dir: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    image = _read_image(archive, sample.image_member)
    if image.size != (1000, 1000):
        raise ValueError(
            f"MoNuSeg {sample.sample_id} has unexpected size {image.size}."
        )
    image_suffix = PurePosixPath(sample.image_member).suffix.lower()
    image_name = f"{sample.sample_id}{image_suffix}"
    image_path = data_dir / "images" / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    instance_mask = _xml_instance_mask(archive.read(sample.mask_members[0]), image.size)
    public = {"Id": sample.sample_id, "image_path": f"input/images/{image_name}"}
    mask_name = f"{sample.sample_id}.png"
    if split == "train":
        mask_path = data_dir / "train_masks" / mask_name
        _write_png(mask_path, instance_mask)
        public["mask_path"] = f"input/train_masks/{mask_name}"
        private = {}
    else:
        mask_path = package_dir / "private_gt" / "masks" / split / mask_name
        _write_png(mask_path, instance_mask)
        private = {
            **public,
            "mask_path": f"private_gt/masks/{split}/{mask_name}",
        }
    meta = {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "source_partition": sample.source_partition,
        "split": split,
        "image_sha256": _sha256_file(image_path),
        "mask_sha256": _sha256_file(mask_path),
        "shape": [image.size[1], image.size[0]],
        "instance_count": int(instance_mask.max()),
    }
    return public, private, meta


def _split_manifest(
    spec: TaskSpec,
    package_dir: Path,
    *,
    counts: dict[str, int],
    seed: int,
    source_files: Sequence[SourceFile],
    task_text_sha256: str,
    sample_records: Sequence[dict[str, Any]],
    split_policy: dict[str, Any],
) -> dict[str, Any]:
    split_paths = {
        "train": package_dir / "biomaker_splits" / "agent_train.csv",
        "score_val": package_dir / "biomaker_splits" / "score_val.csv",
        "test": package_dir / "biomaker_splits" / "agent_test.csv",
    }
    return {
        "schema": "biomaker_split_manifest",
        "manifest_version": 1,
        "package_schema_version": SCHEMA_VERSION,
        "sample_unit": spec.sample_unit,
        "row_representation": "one_row_per_image",
        "key_columns": ["Id"],
        "task_heads": [
            {
                "type": spec.task_head,
                "answer_columns": ["mask_path"],
                "mask_encoding": (
                    "uint16 instance IDs; 0 is background"
                    if spec.task_head == "instance_segmentation"
                    else "uint8 binary; nonzero is foreground"
                ),
            }
        ],
        "splits": {
            name: {
                "file": str(path.relative_to(package_dir)),
                "row_count": counts[name],
                "sha256": _sha256_file(path),
            }
            for name, path in split_paths.items()
        },
        "hidden_answers": {
            "score_val": {"answer_columns": ["mask_path"]},
            "test": {"answer_columns": ["mask_path"]},
        },
        "visibility": {
            "agent_visible": [
                "biomaker_splits/*.csv",
                "images/**",
                "train_masks/**",
                "data_description.txt",
            ],
            "evaluator_only": ["private_gt/**"],
            "forbidden": ["raw archives", "validation/test ground-truth masks"],
        },
        "prediction_contract": {
            "validation_artifact": "./working/validation_predictions.csv",
            "test_submission": "./working/submission.csv",
            "score_split": "./input/biomaker_splits/score_val.csv",
            "key_columns": ["Id"],
            "prediction_columns": ["pred_mask_path"],
            "file_reference_columns": ["pred_mask_path"],
            "prediction_type": "mask_file_path",
            "primary_metric": spec.primary_metric,
            "maximize": True,
        },
        "metric_contract": {
            "primary_metric": spec.primary_metric,
            "secondary_metrics": list(spec.secondary_metrics),
            "maximize": True,
        },
        "split_contract": {
            "seed": seed,
            **split_policy,
            "assignment_sha256": _sha256_text(
                "\n".join(
                    f"{item['sample_id']}\t{item['group_id']}\t{item['split']}"
                    for item in sorted(
                        sample_records, key=lambda item: item["sample_id"]
                    )
                )
            ),
            "group_overlap_checked": True,
        },
        "sample_universe_summary": {
            "count": len(sample_records),
            "counts": counts,
            "group_count": len({item["group_id"] for item in sample_records}),
        },
        "source_contract": {
            "dataset": spec.task_id,
            "source_files": [source.__dict__ for source in source_files],
            "task_text_sha256": task_text_sha256,
            "raw_sources_agent_visible": False,
        },
    }


def _copytree_atomic(source: Path, destination: Path, *, replace: bool) -> None:
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


def _install(source: Path, destination: Path, *, replace: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not replace:
            raise FileExistsError(destination)
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    source.rename(destination)


def _verify_package(package_dir: Path, data_dir: Path) -> dict[str, Any]:
    with _workspace_input_view(data_dir) as input_dir:
        facts = build_dataset_facts(input_dir)
        return verify_split_package(
            package_dir, source_input_dir=input_dir, facts=facts
        )


def _suite_entry(
    spec: TaskSpec,
    *,
    status: str,
    task_desc_path: Path,
    task_text_sha256: str,
    reason: str | None = None,
    data_dir: Path | None = None,
    package_dir: Path | None = None,
    cache_key: str | None = None,
    cache_root: Path | None = None,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "task_id": spec.task_id,
        "display_name": spec.display_name,
        "family": "segmentation",
        "task_family": "segmentation",
        "task_type": spec.task_head,
        "primary_metric": spec.primary_metric,
        "task_text_sha256": task_text_sha256,
        "status": status,
        "desc_file": str(task_desc_path.resolve()),
    }
    if reason:
        entry["reason"] = reason
    if data_dir is not None:
        entry.update(
            {
                "data_dir": str(data_dir.resolve()),
                "task_data_dir": str(data_dir.resolve()),
                "ready_marker": str((data_dir / "_READY.json").resolve()),
            }
        )
    if package_dir is not None:
        entry["package_dir"] = str(package_dir.resolve())
    if cache_key is not None and cache_root is not None:
        entry.update(
            {
                "cache_key": cache_key,
                "cache_entry": str((cache_root / cache_key).resolve()),
            }
        )
    if counts is not None:
        entry["counts"] = counts
    return entry


def _update_suite_manifest(output_root: Path, entry: dict[str, Any]) -> Path:
    path = output_root / "suite_manifest.json"
    with _exclusive_lock(output_root / ".suite_manifest.lock"):
        if path.is_file():
            current = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                raise ValueError(f"Invalid suite manifest: {path}")
        else:
            current = {}
        entries = {
            str(item["task_id"]): item
            for item in current.get("tasks", [])
            if isinstance(item, dict) and item.get("task_id")
        }
        entries[str(entry["task_id"])] = entry
        tasks = [
            entries[task_id] for task_id in MANIFEST_TASK_ORDER if task_id in entries
        ]
        _atomic_write_json(
            path,
            {
                "schema": "biomaker_main9_suite",
                "schema_version": SCHEMA_VERSION,
                "task_count": len(tasks),
                "tasks": tasks,
            },
        )
    return path


def _prepare_binary_archive_task(
    *,
    spec: TaskSpec,
    raw_root: Path,
    relative_archive: str,
    task_desc_path: Path,
    output_root: Path,
    cache_root: Path,
    model: str,
    temperature: float,
    seed: int,
    strict_official: bool,
    replace: bool,
    sample_reader: Callable[[zipfile.ZipFile], tuple[list[Sample], dict[str, Any]]],
    data_description: str,
    split_policy: dict[str, Any],
    mask_policy: dict[str, Any],
) -> dict[str, Any]:
    archive_path = raw_root / relative_archive
    source = _validate_zip(
        archive_path, relative_archive, strict_official=strict_official
    )
    task_text = task_desc_path.read_text(encoding="utf-8")
    task_text_sha256 = _sha256_text(task_text)
    data_dir = output_root / "data" / spec.task_id
    package_dir = output_root / "packages" / spec.task_id
    ready_path = data_dir / "_READY.json"
    expected_hashes = {source.relative_path: source.sha256}
    if ready_path.is_file() and not replace:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if not all(
            (
                ready.get("status") == "ready",
                ready.get("task_id") == spec.task_id,
                ready.get("task_family") == "segmentation",
                ready.get("task_type") == spec.task_head,
                ready.get("primary_metric") == spec.primary_metric,
                ready.get("seed") == seed,
                ready.get("source_sha256") == expected_hashes,
                ready.get("task_text_sha256") == task_text_sha256,
            )
        ):
            raise FileExistsError(
                f"Existing {spec.display_name} data differs from the frozen request: "
                f"{data_dir}"
            )
        cache_key, _ = _split_cache_key(
            data_dir, task_text=task_text, model=model, temperature=temperature
        )
        cache_entry = cache_root / cache_key
        data_hashes = verify_package_integrity(cache_entry, data_dir)
        verifier = _verify_package(cache_entry, data_dir)
        if not verifier.get("ok"):
            raise RuntimeError(
                f"Existing {spec.display_name} data has no valid exact split cache: "
                + json.dumps(verifier, ensure_ascii=False)
            )
        if not package_dir.is_dir():
            _copytree_atomic(cache_entry, package_dir, replace=False)
        elif package_dir.resolve() != cache_entry.resolve():
            verify_package_integrity(
                package_dir, data_dir, validated_data_hashes=data_hashes
            )
        counts = {key: int(value) for key, value in ready["counts"].items()}
        entry = _suite_entry(
            spec,
            status="ready",
            task_desc_path=task_desc_path,
            task_text_sha256=task_text_sha256,
            data_dir=data_dir,
            package_dir=package_dir,
            cache_key=cache_key,
            cache_root=cache_root,
            counts=counts,
        )
        _update_suite_manifest(output_root, entry)
        return {**entry, "reused": True}

    with zipfile.ZipFile(archive_path) as archive:
        samples, source_summary = sample_reader(archive)
    partitions = _stratified_group_split(samples, seed=seed)
    counts = {name: len(partitions[name]) for name in SPLIT_NAMES}
    expected_counts = dict(zip(SPLIT_NAMES, _apportion(len(samples), SPLIT_RATIOS)))
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected split counts: {counts} vs {expected_counts}.")
    group_sets = {
        name: {sample.group_id for sample in partitions[name]} for name in SPLIT_NAMES
    }
    if any(
        group_sets[left] & group_sets[right]
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    ):
        raise RuntimeError("A source group appears in more than one split.")
    stratum_counts = {
        name: dict(sorted(Counter(sample.stratum for sample in values).items()))
        for name, values in partitions.items()
    }

    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".main9-{spec.task_id}-", dir=output_root)
    )
    stage_data = stage_root / "data"
    stage_package = stage_root / "package"
    stage_data.mkdir()
    stage_package.mkdir()
    try:
        public_rows: dict[str, list[dict[str, str]]] = {
            name: [] for name in SPLIT_NAMES
        }
        private_rows: dict[str, list[dict[str, str]]] = {
            "score_val": [],
            "test": [],
        }
        sample_records: list[dict[str, Any]] = []
        with zipfile.ZipFile(archive_path) as archive:
            for split in SPLIT_NAMES:
                for sample in partitions[split]:
                    public, private, metadata = _copy_binary_sample(
                        archive, sample, split, stage_data, stage_package
                    )
                    public_rows[split].append(public)
                    if private:
                        private_rows[split].append(private)
                    sample_records.append(metadata)

        split_dir = stage_package / "biomaker_splits"
        private_dir = stage_package / "private_gt"
        _write_csv(
            split_dir / "agent_train.csv",
            ("Id", "image_path", "mask_path"),
            public_rows["train"],
        )
        for split, filename in (
            ("score_val", "score_val.csv"),
            ("test", "agent_test.csv"),
        ):
            _write_csv(
                split_dir / filename,
                ("Id", "image_path"),
                public_rows[split],
            )
        _write_csv(
            private_dir / "score_val.csv",
            ("Id", "image_path", "mask_path"),
            private_rows["score_val"],
        )
        _write_csv(
            private_dir / "test.csv",
            ("Id", "image_path", "mask_path"),
            private_rows["test"],
        )
        _write_json(
            private_dir / "private_gt.json",
            {
                "score_val": {"file": "score_val.csv", "answer_columns": ["mask_path"]},
                "test": {"file": "test.csv", "answer_columns": ["mask_path"]},
            },
        )
        manifest = _split_manifest(
            spec,
            stage_package,
            counts=counts,
            seed=seed,
            source_files=[source],
            task_text_sha256=task_text_sha256,
            sample_records=sample_records,
            split_policy={
                **split_policy,
                "ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
                "stratum_counts": stratum_counts,
            },
        )
        manifest["mask_contract"] = mask_policy
        _write_json(stage_package / "split_manifest.json", manifest)
        _write_json(
            stage_data / "task_metadata.json",
            {
                "schema": "biomaker_main9_task",
                "schema_version": SCHEMA_VERSION,
                "task_id": spec.task_id,
                "display_name": spec.display_name,
                "task_family": "segmentation",
                "task_type": spec.task_head,
                "primary_metric": spec.primary_metric,
                "secondary_metrics": list(spec.secondary_metrics),
                "license": spec.license,
                "counts": counts,
                "seed": seed,
                "source_files": [source.__dict__],
                "source_summary": source_summary,
                "split_stratum_counts": stratum_counts,
                "mask_contract": mask_policy,
                "task_text_sha256": task_text_sha256,
            },
        )
        (stage_data / "data_description.txt").write_text(
            data_description.rstrip() + "\n", encoding="utf-8"
        )
        _write_json(
            stage_data / "_READY.json",
            {
                "schema": "biomaker_main9_ready",
                "schema_version": SCHEMA_VERSION,
                "status": "ready",
                "task_id": spec.task_id,
                "task_family": "segmentation",
                "task_type": spec.task_head,
                "primary_metric": spec.primary_metric,
                "counts": counts,
                "seed": seed,
                "source_sha256": expected_hashes,
                "task_text_sha256": task_text_sha256,
                "raw_sources_agent_visible": False,
            },
        )
        cache_key, facts = _split_cache_key(
            stage_data, task_text=task_text, model=model, temperature=temperature
        )
        with _workspace_input_view(stage_data) as input_dir:
            verifier = verify_split_package(
                stage_package, source_input_dir=input_dir, facts=facts
            )
        if not verifier.get("ok"):
            raise RuntimeError(
                f"Generated {spec.display_name} package failed verification: "
                + json.dumps(verifier, ensure_ascii=False)
            )
        _write_json(stage_package / "verifier.json", verifier)
        _write_json(
            stage_package / "manifest.json",
            {
                "cache": "split_builder",
                "package_schema": "biomaker_main9_segmentation",
                "package_schema_version": SCHEMA_VERSION,
                "dataset": spec.task_id,
                "task_text_sha256": task_text_sha256,
                "verifier_ok": True,
            },
        )
        cache_entry = cache_root / cache_key
        write_package_integrity(stage_package, stage_data)
        with _exclusive_lock(cache_root / ".locks" / f"{cache_key}.lock"):
            if cache_entry.exists() and replace:
                shutil.rmtree(cache_entry)
            if not cache_entry.exists():
                _copytree_atomic(stage_package, cache_entry, replace=False)
        verify_package_integrity(cache_entry, stage_data)
        installed_verifier = _verify_package(cache_entry, stage_data)
        if not installed_verifier.get("ok"):
            raise RuntimeError(
                f"Installed {spec.display_name} split cache failed verification: "
                + json.dumps(installed_verifier, ensure_ascii=False)
            )
        package_copy = stage_root / "package-install"
        shutil.copytree(stage_package, package_copy)
        _install(package_copy, package_dir, replace=replace)
        _install(stage_data, data_dir, replace=replace)
        actual_key, _ = _split_cache_key(
            data_dir, task_text=task_text, model=model, temperature=temperature
        )
        if actual_key != cache_key:
            raise RuntimeError(
                f"Cache key changed after install: {cache_key} -> {actual_key}."
            )
        entry = _suite_entry(
            spec,
            status="ready",
            task_desc_path=task_desc_path,
            task_text_sha256=task_text_sha256,
            data_dir=data_dir,
            package_dir=package_dir,
            cache_key=cache_key,
            cache_root=cache_root,
            counts=counts,
        )
        _update_suite_manifest(output_root, entry)
        return {**entry, "reused": False}
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)


def _prepare_kvasir_seg(
    **kwargs: Any,
) -> dict[str, Any]:
    return _prepare_binary_archive_task(
        spec=SPECS["kvasir_seg"],
        relative_archive="Kvasir-SEG/kvasir-seg.zip",
        sample_reader=lambda archive: _kvasir_samples(
            archive, strict_official=bool(kwargs["strict_official"])
        ),
        data_description=(
            "Kvasir-SEG binary polyp segmentation using all 1000 public image/mask "
            "pairs. Masks are normalized to binary PNG. Source images with identical "
            "bytes are grouped before the deterministic seed-42 70/15/15 split; the "
            "split is stratified by foreground-area band. The public release has no "
            "patient or video identifiers, so patient-level independence is not claimed."
        ),
        split_policy={
            "policy": "benchmark-defined deterministic grouped and stratified 70/15/15 split of the complete public set",
            "group_key": "exact source-image SHA-256",
            "stratification": "mask foreground area: <5%, 5-20%, >=20%",
            "official_test_preserved": False,
            "patient_level_independence": "not claimed; public patient/video identifiers are unavailable",
        },
        mask_policy={
            "encoding": "uint8 PNG; 0 background and 255 foreground",
            "source_threshold": ">=128 after grayscale conversion",
            "multiple_source_masks": "not applicable; exactly one source mask per image",
            "empty_mask": "forbidden",
        },
        **kwargs,
    )


def _prepare_busi(
    **kwargs: Any,
) -> dict[str, Any]:
    return _prepare_binary_archive_task(
        spec=SPECS["busi"],
        relative_archive="BUSI/Dataset_BUSI.zip",
        sample_reader=lambda archive: _busi_samples(
            archive, strict_official=bool(kwargs["strict_official"])
        ),
        data_description=(
            "BUSI binary lesion segmentation using all 780 public ultrasound images. "
            "Every `_mask.png`, `_mask_1.png`, and later mask belonging to one image "
            "is unioned into one binary target. Normal images retain an explicit empty "
            "mask. Public image names and paths use neutral IDs; source diagnosis is "
            "used only for split stratification. Exact duplicate images are grouped "
            "before the seed-42 70/15/15 split. The release has no patient IDs, so "
            "patient-level independence is not claimed."
        ),
        split_policy={
            "policy": "benchmark-defined deterministic grouped and diagnosis-stratified 70/15/15 split of the complete public set",
            "group_key": "exact source-image SHA-256; keeps the known cross-category duplicate in one split",
            "stratification": "source diagnosis category: benign, malignant, normal",
            "official_test_preserved": False,
            "patient_level_independence": "not claimed; public patient identifiers are unavailable",
            "diagnosis_mapping_agent_visible": False,
        },
        mask_policy={
            "encoding": "uint8 PNG; 0 background and 255 foreground",
            "source_threshold": ">=128 after grayscale conversion",
            "multiple_source_masks": "pixelwise logical union of all masks for the same source image",
            "normal_class": "all-zero mask is required and retained as a valid negative target",
            "benign_or_malignant": "non-empty union mask is required",
        },
        **kwargs,
    )


def _prepare_aeropath_blocked(
    *,
    raw_root: Path,
    task_desc_path: Path,
    output_root: Path,
    **_: Any,
) -> dict[str, Any]:
    spec = SPECS["aeropath"]
    source_root = raw_root / "AeroPath"
    data_root = source_root / "data"
    missing = []
    for patient_id in range(1, 28):
        patient_dir = data_root / str(patient_id)
        for suffix in (
            "_CT_HR.nii.gz",
            "_CT_HR_label_airways.nii.gz",
            "_CT_HR_label_lungs.nii.gz",
        ):
            path = patient_dir / f"{patient_id}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "AeroPath public 27-case inventory is incomplete: " + ", ".join(missing[:5])
        )
    readme = source_root / "README.md"
    loader = source_root / "AeroPath.py"
    if not readme.is_file() or not loader.is_file():
        raise FileNotFoundError("AeroPath README.md and official loader are required.")
    readme_text = readme.read_text(encoding="utf-8")
    loader_text = loader.read_text(encoding="utf-8")
    if (
        "27 computed tomography" not in readme_text
        or "datasets.Split.TEST" not in loader_text
    ):
        raise ValueError(
            "AeroPath source does not prove the frozen 27-case TEST-only protocol."
        )
    task_text_sha256 = _sha256_text(task_desc_path.read_text(encoding="utf-8"))
    reason = (
        "The official public AeroPath loader exposes all 27 annotated cases as one "
        "TEST split and provides no independent training set. BioMaker will not invent "
        "a formal train/score-validation/test protocol from that same test-only set."
    )
    entry = _suite_entry(
        spec,
        status="protocol_blocked",
        task_desc_path=task_desc_path,
        task_text_sha256=task_text_sha256,
        reason=reason,
    )
    entry.update(
        {
            "source_count": 27,
            "official_protocol": "TEST-only",
            "primary_target": "airways",
            "source_root": str(source_root.resolve()),
            "source_evidence": {
                "readme_sha256": _sha256_file(readme),
                "official_loader_sha256": _sha256_file(loader),
            },
            "artifacts_generated": False,
        }
    )
    _update_suite_manifest(output_root, entry)
    return {**entry, "reused": False}


def _prepare_monuseg(
    *,
    raw_root: Path,
    task_desc_path: Path,
    output_root: Path,
    cache_root: Path,
    model: str,
    temperature: float,
    seed: int,
    strict_official: bool,
    replace: bool,
) -> dict[str, Any]:
    spec = SPECS["monuseg"]
    train_rel = "MoNuSeg/MoNuSeg_2018_Training_Data.zip"
    test_rel = "MoNuSeg/MoNuSegTestData.zip"
    train_zip = raw_root / train_rel
    test_zip = raw_root / test_rel
    sources = [
        _validate_zip(train_zip, train_rel, strict_official=strict_official),
        _validate_zip(test_zip, test_rel, strict_official=strict_official),
    ]
    task_text = task_desc_path.read_text(encoding="utf-8")
    task_text_sha256 = _sha256_text(task_text)
    data_dir = output_root / "data" / spec.task_id
    package_dir = output_root / "packages" / spec.task_id
    ready_path = data_dir / "_READY.json"
    if ready_path.is_file() and not replace:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        expected_hashes = {source.relative_path: source.sha256 for source in sources}
        if not all(
            (
                ready.get("status") == "ready",
                ready.get("task_id") == spec.task_id,
                ready.get("seed") == seed,
                ready.get("source_sha256") == expected_hashes,
                ready.get("task_text_sha256") == task_text_sha256,
            )
        ):
            raise FileExistsError(
                f"Existing MoNuSeg data differs from the frozen request: {data_dir}"
            )
        cache_key, _ = _split_cache_key(
            data_dir, task_text=task_text, model=model, temperature=temperature
        )
        cache_entry = cache_root / cache_key
        data_hashes = verify_package_integrity(cache_entry, data_dir)
        verifier = _verify_package(cache_entry, data_dir)
        if not verifier.get("ok"):
            raise RuntimeError(
                "Existing MoNuSeg data has no valid exact split cache: "
                + json.dumps(verifier, ensure_ascii=False)
            )
        if not package_dir.is_dir():
            _copytree_atomic(cache_entry, package_dir, replace=False)
        elif package_dir.resolve() != cache_entry.resolve():
            verify_package_integrity(
                package_dir, data_dir, validated_data_hashes=data_hashes
            )
        counts = {key: int(value) for key, value in ready["counts"].items()}
        entry = _suite_entry(
            spec,
            status="ready",
            task_desc_path=task_desc_path,
            task_text_sha256=task_text_sha256,
            data_dir=data_dir,
            package_dir=package_dir,
            cache_key=cache_key,
            cache_root=cache_root,
            counts=counts,
        )
        _update_suite_manifest(output_root, entry)
        return {**entry, "reused": True}
    output_root.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".main9-monuseg-", dir=output_root))
    stage_data = stage_root / "data"
    stage_package = stage_root / "package"
    stage_data.mkdir()
    stage_package.mkdir()
    try:
        with (
            zipfile.ZipFile(train_zip) as train_archive,
            zipfile.ZipFile(test_zip) as test_archive,
        ):
            official_train = _monuseg_members(train_archive, partition="official_train")
            official_test = _monuseg_members(test_archive, partition="official_test")
            if strict_official and (len(official_train), len(official_test)) != (
                37,
                14,
            ):
                raise ValueError(
                    "Frozen MoNuSeg source must contain 37 train and 14 test pairs; "
                    f"got {len(official_train)} and {len(official_test)}."
                )
            train_samples, score_samples = _grouped_holdout(
                official_train, seed=seed, fraction=0.20
            )
            partitions = {
                "train": (train_archive, train_samples),
                "score_val": (train_archive, score_samples),
                "test": (test_archive, official_test),
            }
            public_rows: dict[str, list[dict[str, str]]] = {
                name: [] for name in partitions
            }
            private_rows: dict[str, list[dict[str, str]]] = {
                "score_val": [],
                "test": [],
            }
            sample_records: list[dict[str, Any]] = []
            for split, (archive, samples) in partitions.items():
                for sample in samples:
                    public, private, metadata = _copy_monuseg_sample(
                        archive, sample, split, stage_data, stage_package
                    )
                    public_rows[split].append(public)
                    if private:
                        private_rows[split].append(private)
                    sample_records.append(metadata)

        split_dir = stage_package / "biomaker_splits"
        private_dir = stage_package / "private_gt"
        _write_csv(
            split_dir / "agent_train.csv",
            ("Id", "image_path", "mask_path"),
            public_rows["train"],
        )
        for split, filename in (
            ("score_val", "score_val.csv"),
            ("test", "agent_test.csv"),
        ):
            _write_csv(
                split_dir / filename,
                ("Id", "image_path"),
                public_rows[split],
            )
        _write_csv(
            private_dir / "score_val.csv",
            ("Id", "image_path", "mask_path"),
            private_rows["score_val"],
        )
        _write_csv(
            private_dir / "test.csv",
            ("Id", "image_path", "mask_path"),
            private_rows["test"],
        )
        _write_json(
            private_dir / "private_gt.json",
            {
                "score_val": {"file": "score_val.csv", "answer_columns": ["mask_path"]},
                "test": {"file": "test.csv", "answer_columns": ["mask_path"]},
            },
        )
        counts = {name: len(rows) for name, rows in public_rows.items()}
        manifest = _split_manifest(
            spec,
            stage_package,
            counts=counts,
            seed=seed,
            source_files=sources,
            task_text_sha256=task_text_sha256,
            sample_records=sample_records,
            split_policy={
                "policy": "preserve official test; patient-grouped 80/20 holdout inside official train",
                "group_key": "TCGA patient ID (first three hyphen-separated fields)",
                "official_test_preserved": True,
            },
        )
        _write_json(stage_package / "split_manifest.json", manifest)
        _write_json(
            stage_data / "task_metadata.json",
            {
                "schema": "biomaker_main9_task",
                "schema_version": SCHEMA_VERSION,
                "task_id": spec.task_id,
                "display_name": spec.display_name,
                "task_family": "segmentation",
                "task_type": spec.task_head,
                "primary_metric": spec.primary_metric,
                "secondary_metrics": list(spec.secondary_metrics),
                "license": spec.license,
                "counts": counts,
                "seed": seed,
                "source_files": [source.__dict__ for source in sources],
                "task_text_sha256": task_text_sha256,
                "xml_rasterization": "Pillow polygon fill after nearest-integer vertex rounding; first region wins overlaps; uint16 instance IDs",
            },
        )
        (stage_data / "data_description.txt").write_text(
            "MoNuSeg 2018 instance segmentation. Images are public. Only training "
            "masks are agent-visible; score-validation and official-test masks are "
            "evaluator-only. The official 14-image test split is unchanged.\n",
            encoding="utf-8",
        )
        _write_json(
            stage_data / "_READY.json",
            {
                "schema": "biomaker_main9_ready",
                "schema_version": SCHEMA_VERSION,
                "status": "ready",
                "task_id": spec.task_id,
                "task_family": "segmentation",
                "task_type": spec.task_head,
                "primary_metric": spec.primary_metric,
                "counts": counts,
                "seed": seed,
                "source_sha256": {
                    source.relative_path: source.sha256 for source in sources
                },
                "task_text_sha256": task_text_sha256,
                "raw_sources_agent_visible": False,
            },
        )
        cache_key, facts = _split_cache_key(
            stage_data, task_text=task_text, model=model, temperature=temperature
        )
        with _workspace_input_view(stage_data) as input_dir:
            verifier = verify_split_package(
                stage_package, source_input_dir=input_dir, facts=facts
            )
        if not verifier.get("ok"):
            raise RuntimeError(
                "Generated MoNuSeg package failed verification: "
                + json.dumps(verifier, ensure_ascii=False)
            )
        _write_json(stage_package / "verifier.json", verifier)
        _write_json(
            stage_package / "manifest.json",
            {
                "cache": "split_builder",
                "package_schema": "biomaker_main9_segmentation",
                "package_schema_version": SCHEMA_VERSION,
                "dataset": spec.task_id,
                "task_text_sha256": task_text_sha256,
                "verifier_ok": True,
            },
        )
        cache_entry = cache_root / cache_key
        write_package_integrity(stage_package, stage_data)
        with _exclusive_lock(cache_root / ".locks" / f"{cache_key}.lock"):
            if cache_entry.exists() and replace:
                shutil.rmtree(cache_entry)
            if not cache_entry.exists():
                _copytree_atomic(stage_package, cache_entry, replace=False)
        verify_package_integrity(cache_entry, stage_data)
        installed_verifier = _verify_package(cache_entry, stage_data)
        if not installed_verifier.get("ok"):
            raise RuntimeError(
                "Installed MoNuSeg split cache failed verification: "
                + json.dumps(installed_verifier, ensure_ascii=False)
            )
        package_copy = stage_root / "package-install"
        shutil.copytree(stage_package, package_copy)
        _install(package_copy, package_dir, replace=replace)
        _install(stage_data, data_dir, replace=replace)
        actual_key, _ = _split_cache_key(
            data_dir, task_text=task_text, model=model, temperature=temperature
        )
        if actual_key != cache_key:
            raise RuntimeError(
                f"Cache key changed after install: {cache_key} -> {actual_key}."
            )
        entry = _suite_entry(
            spec,
            status="ready",
            task_desc_path=task_desc_path,
            task_text_sha256=task_text_sha256,
            data_dir=data_dir,
            package_dir=package_dir,
            cache_key=cache_key,
            cache_root=cache_root,
            counts=counts,
        )
        _update_suite_manifest(output_root, entry)
        return {**entry, "reused": False}
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)


def prepare_task(
    task_id: str,
    *,
    raw_root: Path,
    task_desc_path: Path,
    output_root: Path,
    cache_root: Path,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.0,
    seed: int = DEFAULT_SEED,
    strict_official: bool = True,
    replace: bool = False,
) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    task_desc_path = task_desc_path.resolve()
    output_root = output_root.resolve()
    cache_root = cache_root.resolve()
    if task_id not in SPECS:
        raise ValueError(f"Unknown Main-9 task: {task_id}")
    if not task_desc_path.is_file():
        raise FileNotFoundError(task_desc_path)
    if seed != DEFAULT_SEED:
        raise ValueError(f"Main-9 split seed is frozen at {DEFAULT_SEED}.")
    if task_id == "monuseg":
        return _prepare_monuseg(
            raw_root=raw_root,
            task_desc_path=task_desc_path,
            output_root=output_root,
            cache_root=cache_root,
            model=model,
            temperature=temperature,
            seed=seed,
            strict_official=strict_official,
            replace=replace,
        )
    common = {
        "raw_root": raw_root,
        "task_desc_path": task_desc_path,
        "output_root": output_root,
        "cache_root": cache_root,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "strict_official": strict_official,
        "replace": replace,
    }
    if task_id == "kvasir_seg":
        return _prepare_kvasir_seg(**common)
    if task_id == "busi":
        return _prepare_busi(**common)
    if task_id == "aeropath":
        return _prepare_aeropath_blocked(**common)
    raise AssertionError(f"Unhandled Main-9 task: {task_id}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=TASK_ORDER, default=[])
    parser.add_argument(
        "--task-desc-dir",
        type=Path,
        default=WORKSPACE_ROOT / "experiments" / "main9" / "tasks",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-unverified-source", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected = args.task or list(TASK_ORDER)
    results = []
    for task_id in TASK_ORDER:
        if task_id not in selected:
            continue
        result = prepare_task(
            task_id,
            raw_root=args.raw_root,
            task_desc_path=args.task_desc_dir / f"{task_id}.md",
            output_root=args.output_root,
            cache_root=args.cache_root,
            model=args.model,
            temperature=args.temperature,
            seed=args.seed,
            strict_official=not args.allow_unverified_source,
            replace=args.replace,
        )
        results.append(result)
        print(
            f"{task_id}: {result['status']}; data_dir={result.get('data_dir')}; "
            f"cache_key={result.get('cache_key')}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "ok": True,
                "prepared": len(results),
                "tasks": [result["task_id"] for result in results],
                "suite_manifest": str(
                    (args.output_root / "suite_manifest.json").resolve()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
