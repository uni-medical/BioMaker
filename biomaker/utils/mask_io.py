"""Shared 2D/3D mask IO and deterministic segmentation metrics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
    import SimpleITK as sitk
except Exception:
    sitk = None


def is_volume_path(path: Path | str) -> bool:
    lower = str(path).lower()
    return lower.endswith((".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".npy", ".npz"))


def load_mask(path: Path | str) -> tuple[np.ndarray, tuple[float, ...] | None]:
    path = Path(path)
    lower = str(path).lower()
    if lower.endswith(".npy"):
        return np.load(path), None
    if lower.endswith(".npz"):
        data = np.load(path)
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        return data[key], None
    if is_volume_path(path):
        if sitk is None:
            raise RuntimeError(
                "SimpleITK is required to read medical volume masks such as NIfTI/NRRD/MHA."
            )
        image = sitk.ReadImage(str(path))
        spacing = tuple(reversed(tuple(float(value) for value in image.GetSpacing())))
        return sitk.GetArrayFromImage(image), spacing
    with Image.open(path) as image:
        return np.asarray(image), None


def spatial_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 3 and shape[-1] in (3, 4):
        return tuple(shape[:2])
    return tuple(shape)


def binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[-1] in (3, 4):
        mask = mask[..., 0]
    return mask > 0


def dice(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    true_bin = binary_mask(mask_true)
    pred_bin = binary_mask(mask_pred)
    denom = true_bin.sum() + pred_bin.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(true_bin, pred_bin).sum() / denom)


def connected_components(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[-1] in (3, 4):
        mask = mask[..., 0]
    unique = np.unique(mask)
    if mask.max() > 1 and len(unique) > 2:
        positive = unique[unique > 0]
        remapped = np.zeros(mask.shape, dtype=np.int32)
        for index, label in enumerate(positive, start=1):
            remapped[mask == label] = index
        return remapped
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    labels, _ = ndi.label(mask > 0, structure=structure)
    return labels.astype(np.int32, copy=False)


def surface_distances(
    source: np.ndarray, target: np.ndarray, spacing: tuple[float, ...]
) -> np.ndarray:
    if not source.any():
        return np.asarray([], dtype=float)
    if not target.any():
        return np.asarray([math.inf], dtype=float)
    structure = ndi.generate_binary_structure(source.ndim, 1)
    source_border = np.logical_xor(
        source, ndi.binary_erosion(source, structure=structure, border_value=0)
    )
    target_border = np.logical_xor(
        target, ndi.binary_erosion(target, structure=structure, border_value=0)
    )
    distance = ndi.distance_transform_edt(~target_border, sampling=spacing)
    return distance[source_border]


def hd95(
    mask_true: np.ndarray, mask_pred: np.ndarray, spacing: tuple[float, ...] | None
) -> float:
    true_bin = binary_mask(mask_true)
    pred_bin = binary_mask(mask_pred)
    if not true_bin.any() and not pred_bin.any():
        return 0.0
    if not true_bin.any() or not pred_bin.any():
        return math.inf
    if spacing is None:
        spacing = (1.0,) * true_bin.ndim
    distances = np.concatenate(
        [
            surface_distances(true_bin, pred_bin, spacing),
            surface_distances(pred_bin, true_bin, spacing),
        ]
    )
    return float(np.percentile(distances, 95))


def aji(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    true_instances = connected_components(mask_true)
    pred_instances = connected_components(mask_pred)
    true_ids = np.setdiff1d(np.unique(true_instances), [0])
    pred_ids = np.setdiff1d(np.unique(pred_instances), [0])
    if len(true_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    true_area = np.bincount(true_instances.ravel())
    pred_area = np.bincount(pred_instances.ravel())
    stride = len(pred_area) + 1
    pair_key = true_instances.ravel().astype(np.int64) * stride + pred_instances.ravel()
    pair_count = np.bincount(pair_key)
    candidates = []
    for key, intersection in enumerate(pair_count):
        if intersection == 0:
            continue
        true_id = key // stride
        pred_id = key % stride
        if true_id == 0 or pred_id == 0:
            continue
        union = int(true_area[true_id] + pred_area[pred_id] - intersection)
        candidates.append(
            (intersection / union, int(true_id), int(pred_id), int(intersection), union)
        )

    candidates.sort(reverse=True)
    used_true: set[int] = set()
    used_pred: set[int] = set()
    intersection_sum = 0
    union_sum = 0
    for _, true_id, pred_id, intersection, union in candidates:
        if true_id in used_true or pred_id in used_pred:
            continue
        used_true.add(true_id)
        used_pred.add(pred_id)
        intersection_sum += intersection
        union_sum += union

    for true_id in true_ids:
        if int(true_id) not in used_true:
            union_sum += int(true_area[int(true_id)])
    for pred_id in pred_ids:
        if int(pred_id) not in used_pred:
            union_sum += int(pred_area[int(pred_id)])
    return float(intersection_sum / union_sum) if union_sum else 0.0
