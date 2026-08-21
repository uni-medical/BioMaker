"""User-only evaluation that is never fed back to the agent."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import rankdata

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

from .evaluator_setup import validate_submission_with_evaluator
from .experiment_contract import load_experiment_contract
from .prediction_contract import key_columns_from_spec, resolve_artifact_path


def _load_private_test_plan(private_dir: Path) -> dict[str, Any] | None:
    private_path = private_dir / "private_gt.json"
    if not private_path.is_file():
        return None
    hidden_answers = json.loads(private_path.read_text(encoding="utf-8"))
    test_entry = (
        hidden_answers.get("test") if isinstance(hidden_answers, dict) else None
    )
    if not isinstance(test_entry, dict):
        return None
    answer_columns = [str(c) for c in test_entry.get("answer_columns", [])]
    mask_column = next(
        (
            column
            for column in answer_columns
            if "path" in column.lower()
            or "mask" in column.lower()
            or "annotation" in column.lower()
            or "bbox" in column.lower()
        ),
        None,
    )
    label_columns = [] if mask_column else answer_columns
    label_column = label_columns[0] if len(label_columns) == 1 else None
    return {
        "source": str(test_entry.get("file") or "test.csv"),
        "label_column": label_column,
        "label_columns": label_columns,
        "mask_column": mask_column,
        "id_column": None,
    }


def _test_submission_spec(spec: dict[str, Any]) -> dict[str, Any]:
    value = spec.get("test_submission")
    return value if isinstance(value, dict) else {}


def _spec_list(spec: dict[str, Any], key: str) -> list[str]:
    if key == "key_columns":
        return key_columns_from_spec(spec)
    value = spec.get(key)
    if not isinstance(value, list):
        value = _test_submission_spec(spec).get(key)
    return [str(item) for item in value] if isinstance(value, list) else []


def _probability_class_values(
    columns: list[str], spec: dict[str, Any] | None = None
) -> list[str]:
    if spec:
        labels = spec.get("class_labels")
        if isinstance(labels, list) and len(labels) == len(columns):
            return [str(label) for label in labels]

    values = []
    for index, column in enumerate(columns):
        name = str(column)
        if name.startswith("class_prob_"):
            values.append(name.removeprefix("class_prob_"))
        elif name.startswith("class_") and name.endswith("_prob"):
            values.append(name.removeprefix("class_").removesuffix("_prob"))
        elif name.startswith("class_"):
            values.append(name.removeprefix("class_"))
        elif name.startswith("prob_"):
            values.append(name.removeprefix("prob_"))
        elif name.startswith("p_"):
            values.append(name.removeprefix("p_"))
        else:
            values.append(str(index))
    return values


def _positive_class_from_spec(
    y_true: pd.Series, score_column: str, spec: dict[str, Any] | None = None
) -> str | None:
    if spec:
        value = spec.get("positive_class")
        if value is not None:
            return str(value)

    name = str(score_column)
    for prefix in ["class_prob_", "prob_", "p_", "class_"]:
        if name.startswith(prefix):
            return name.removeprefix(prefix)

    unique = sorted(str(value) for value in y_true.dropna().unique())
    if "1" in unique:
        return "1"
    if len(unique) == 2:
        return unique[-1]
    return None


def _negative_class_for_positive(y_true: pd.Series, positive_class: str) -> str:
    unique = sorted(str(value) for value in y_true.dropna().unique())
    for value in unique:
        if value != str(positive_class):
            return value
    return f"not_{positive_class}"


def _binary_auc(y_true: pd.Series, y_score: np.ndarray, positive_class: str) -> float:
    true_positive = y_true.astype(str).to_numpy() == str(positive_class)
    if true_positive.all() or (~true_positive).all():
        return float("nan")
    ranks = rankdata(y_score, method="average")
    n_pos = int(true_positive.sum())
    n_neg = int((~true_positive).sum())
    rank_sum = float(ranks[true_positive].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _macro_ovr_auc(y_true: np.ndarray, y_score: np.ndarray, n_classes: int) -> float:
    scores = []
    for class_index in range(n_classes):
        positives = y_true == class_index
        negatives = ~positives
        if not positives.any() or not negatives.any():
            continue
        ranks = rankdata(y_score[:, class_index], method="average")
        n_pos = int(positives.sum())
        n_neg = int(negatives.sum())
        rank_sum = float(ranks[positives].sum())
        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        scores.append(float(auc))
    return float(np.mean(scores)) if scores else float("nan")


def _classification_metrics_from_submission(
    truth: pd.DataFrame,
    submission: pd.DataFrame,
    label_column: str,
    id_column: str | None,
    prediction_columns: list[str],
    key_columns: list[str],
    primary_metric: str | None,
    prediction_spec: dict[str, Any],
) -> dict[str, Any] | None:
    if label_column not in truth.columns:
        return None

    has_pred_label = (
        label_column in prediction_columns and label_column in submission.columns
    )
    prob_columns = [
        column
        for column in prediction_columns
        if column in submission.columns
        and not (has_pred_label and column == label_column)
    ]
    if not has_pred_label and not prob_columns:
        return None

    submission_columns = [*prob_columns]
    if has_pred_label:
        submission_columns.insert(0, label_column)

    merge_keys = [
        column
        for column in key_columns
        if column in truth.columns and column in submission.columns
    ]
    if (
        not merge_keys
        and id_column
        and id_column in truth.columns
        and id_column in submission.columns
    ):
        merge_keys = [id_column]

    if merge_keys:
        truth_view = truth[[*merge_keys, label_column]].copy()
        submission_view = submission[[*merge_keys, *submission_columns]].copy()
        for column in merge_keys:
            truth_view[column] = truth_view[column].astype(str)
            submission_view[column] = submission_view[column].astype(str)
        if submission_view.duplicated(merge_keys).any():
            return None
        if len(submission_view) != len(truth_view):
            return None
        merged = truth_view.merge(
            submission_view,
            on=merge_keys,
            suffixes=("_true", "_pred"),
        )
        if len(merged) != len(truth_view):
            return None
        y_true_column = (
            f"{label_column}_true"
            if f"{label_column}_true" in merged.columns
            else label_column
        )
        y_true = merged[y_true_column]
        y_pred = merged[f"{label_column}_pred"] if has_pred_label else None
        scores_df = merged[prob_columns] if prob_columns else None
    else:
        if len(truth) != len(submission):
            return None
        y_true = truth[label_column].reset_index(drop=True)
        y_pred = (
            submission[label_column].reset_index(drop=True) if has_pred_label else None
        )
        scores_df = submission[prob_columns] if prob_columns else None

    class_values = _probability_class_values(prob_columns, prediction_spec)
    if y_pred is None and scores_df is not None and not scores_df.empty:
        if scores_df.shape[1] == 1:
            positive_class = _positive_class_from_spec(
                y_true, prob_columns[0], prediction_spec
            )
            if positive_class is not None:
                negative_class = _negative_class_for_positive(y_true, positive_class)
                score_values = scores_df.iloc[:, 0].to_numpy(dtype=float)
                y_pred = pd.Series(
                    [
                        positive_class if score >= 0.5 else negative_class
                        for score in score_values
                    ]
                )
        else:
            pred_index = scores_df.to_numpy(dtype=float).argmax(axis=1)
            y_pred = pd.Series([class_values[index] for index in pred_index])
    if y_pred is None:
        return None

    accuracy = float((y_true.astype(str) == y_pred.astype(str)).mean())
    metrics: dict[str, float] = {"accuracy": accuracy}
    primary_name = "private_accuracy"
    primary_value = accuracy
    requested_auc = primary_metric in {"auc", "roc_auc", "macro_auc", "ovr_auc"}
    requested_accuracy = primary_metric in {"accuracy", "acc"}
    if scores_df is not None and not scores_df.empty:
        y_score = scores_df.to_numpy(dtype=float)
        if scores_df.shape[1] == 1:
            positive_class = _positive_class_from_spec(
                y_true, prob_columns[0], prediction_spec
            )
            auc = (
                _binary_auc(y_true, y_score[:, 0], positive_class)
                if positive_class is not None
                else float("nan")
            )
        else:
            class_to_index = {
                str(value): index for index, value in enumerate(class_values)
            }
            if not y_true.astype(str).isin(class_to_index).all():
                truth_values = set(y_true.astype(str).unique())
                if truth_values.issubset(set(prob_columns)):
                    class_to_index = {
                        str(value): index for index, value in enumerate(prob_columns)
                    }
                else:
                    class_to_index = {}
            if class_to_index:
                y_true_index = (
                    y_true.astype(str).map(class_to_index).to_numpy(dtype=int)
                )
                auc = _macro_ovr_auc(y_true_index, y_score, len(class_to_index))
            else:
                auc = float("nan")
        if np.isfinite(auc):
            metrics["auc"] = float(auc)
            if requested_auc or not requested_accuracy:
                primary_name = "private_auc"
                primary_value = float(auc)
    return {
        "metric_name": primary_name,
        "value": primary_value,
        "metrics": metrics,
        "n": int(len(y_true)),
        "maximize": True,
    }


def _multilabel_metrics_from_submission(
    truth: pd.DataFrame,
    submission: pd.DataFrame,
    label_columns: list[str],
    id_column: str | None,
    prediction_columns: list[str],
    key_columns: list[str],
    primary_metric: str | None,
) -> dict[str, Any] | None:
    if not label_columns or any(
        column not in truth.columns for column in label_columns
    ):
        return None
    if len(prediction_columns) < len(label_columns):
        return None
    score_columns = prediction_columns[: len(label_columns)]
    if any(column not in submission.columns for column in score_columns):
        return None

    merge_keys = [
        column
        for column in key_columns
        if column in truth.columns and column in submission.columns
    ]
    if (
        not merge_keys
        and id_column
        and id_column in truth.columns
        and id_column in submission.columns
    ):
        merge_keys = [id_column]

    if merge_keys:
        truth_view = truth[[*merge_keys, *label_columns]].copy()
        submission_view = submission[[*merge_keys, *score_columns]].copy()
        for column in merge_keys:
            truth_view[column] = truth_view[column].astype(str)
            submission_view[column] = submission_view[column].astype(str)
        if submission_view.duplicated(merge_keys).any():
            return None
        if len(submission_view) != len(truth_view):
            return None
        merged = truth_view.merge(submission_view, on=merge_keys, how="inner")
        if len(merged) != len(truth_view):
            return None
        y_true = merged[label_columns].to_numpy(dtype=int)
        scores = merged[score_columns].to_numpy(dtype=float)
    else:
        if len(truth) != len(submission):
            return None
        y_true = truth[label_columns].to_numpy(dtype=int)
        scores = submission[score_columns].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        return None
    y_pred = (scores >= 0.5).astype(int)
    per_label_accuracy = (y_true == y_pred).mean(axis=0)
    mean_binary_accuracy = float(per_label_accuracy.mean())
    metrics: dict[str, Any] = {
        "mean_binary_accuracy": mean_binary_accuracy,
        "per_label_accuracy": {
            column: float(value)
            for column, value in zip(label_columns, per_label_accuracy)
        },
    }
    primary_name = "private_mean_binary_accuracy"
    primary_value = mean_binary_accuracy
    if primary_metric in {"auc", "roc_auc", "mean_roc_auc", "macro_auc"}:
        aucs = []
        for index, column in enumerate(label_columns):
            y_column = pd.Series(y_true[:, index])
            if len(set(y_column.astype(str))) < 2:
                continue
            aucs.append(_binary_auc(y_column, scores[:, index], "1"))
        if aucs:
            auc = float(np.mean(aucs))
            metrics["mean_roc_auc"] = auc
            primary_name = "private_mean_roc_auc"
            primary_value = auc
    return {
        "metric_name": primary_name,
        "value": primary_value,
        "metrics": metrics,
        "n": int(len(y_true)),
        "maximize": True,
    }


def _dice(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    if mask_true.ndim == 3 and mask_true.shape[-1] in (3, 4):
        mask_true = mask_true[..., 0]
    if mask_pred.ndim == 3 and mask_pred.shape[-1] in (3, 4):
        mask_pred = mask_pred[..., 0]
    true_bin = mask_true > 0
    pred_bin = mask_pred > 0
    denom = true_bin.sum() + pred_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(true_bin, pred_bin).sum() / denom)


def _connected_components(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[-1] in (3, 4):
        mask = mask[..., 0]
    if mask.max() > 1 and len(np.unique(mask)) > 2:
        return mask.astype(np.int32, copy=False)
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    labels, _ = ndi.label(mask > 0, structure=structure)
    return labels.astype(np.int32, copy=False)


def _is_volume_path(path: Path) -> bool:
    lower = str(path).lower()
    return lower.endswith((".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".npy", ".npz"))


def _load_mask(path: Path) -> tuple[np.ndarray, tuple[float, ...] | None]:
    lower = str(path).lower()
    if lower.endswith(".npy"):
        return np.load(path), None
    if lower.endswith(".npz"):
        data = np.load(path)
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        return data[key], None
    if _is_volume_path(path):
        if sitk is None:
            raise RuntimeError(
                "SimpleITK is required to read private medical volume masks."
            )
        image = sitk.ReadImage(str(path))
        spacing = tuple(reversed(tuple(float(value) for value in image.GetSpacing())))
        return sitk.GetArrayFromImage(image), spacing
    with Image.open(path) as image:
        return np.asarray(image), None


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[-1] in (3, 4):
        mask = mask[..., 0]
    return mask > 0


def _surface_distances(
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


def _hd95(
    mask_true: np.ndarray, mask_pred: np.ndarray, spacing: tuple[float, ...] | None
) -> float:
    true_bin = _binary_mask(mask_true)
    pred_bin = _binary_mask(mask_pred)
    if not true_bin.any() and not pred_bin.any():
        return 0.0
    if not true_bin.any() or not pred_bin.any():
        return math.inf
    if spacing is None:
        spacing = (1.0,) * true_bin.ndim
    distances = np.concatenate(
        [
            _surface_distances(true_bin, pred_bin, spacing),
            _surface_distances(pred_bin, true_bin, spacing),
        ]
    )
    return float(np.percentile(distances, 95))


def _aji(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    true_instances = _connected_components(mask_true)
    pred_instances = _connected_components(mask_pred)
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


def _private_mask_metrics(
    workspace_dir: Path,
    private_dir: Path,
    truth: pd.DataFrame,
    submission: pd.DataFrame,
    mask_column: str,
    prediction_columns: list[str],
    key_columns: list[str],
    primary_metric: str | None,
) -> dict[str, Any] | None:
    path_columns = [
        column for column in prediction_columns if column in submission.columns
    ]
    if mask_column not in truth.columns or not path_columns:
        return None

    id_columns = [
        column
        for column in key_columns
        if column in truth.columns and column in submission.columns
    ]
    pred_column = path_columns[0]
    pred_artifact_column = "__biomaker_pred_artifact_path"

    if id_columns:
        truth_view = truth[[*id_columns, mask_column]].copy()
        submission_view = submission[[*id_columns, pred_column]].copy()
        submission_view = submission_view.rename(
            columns={pred_column: pred_artifact_column}
        )
        for column in id_columns:
            truth_view[column] = truth_view[column].astype(str)
            submission_view[column] = submission_view[column].astype(str)
        if submission_view.duplicated(id_columns).any():
            return None
        if len(submission_view) != len(truth_view):
            return None
        rows = truth_view.merge(
            submission_view,
            on=id_columns,
            how="inner",
        )
        if len(rows) != len(truth):
            return None
    else:
        if len(truth) != len(submission):
            return None
        rows = truth.copy()
        rows[pred_artifact_column] = submission[pred_column].values

    dice_scores = []
    aji_scores = []
    hd95_scores = []
    for _, row in rows.iterrows():
        true_path = _private_gt_path(private_dir, row[mask_column])
        if not true_path.is_file():
            return None
        pred_path = resolve_artifact_path(workspace_dir, str(row[pred_artifact_column]))
        if not pred_path.is_file():
            return None
        try:
            true_mask, true_spacing = _load_mask(true_path)
            pred_mask, pred_spacing = _load_mask(pred_path)
        except Exception:
            return None
        if true_mask.shape != pred_mask.shape:
            return None
        if true_spacing is not None and pred_spacing is not None:
            if len(true_spacing) != len(pred_spacing) or any(
                abs(a - b) > 1e-4 for a, b in zip(true_spacing, pred_spacing)
            ):
                return None
        dice_scores.append(_dice(true_mask, pred_mask))
        aji_scores.append(_aji(true_mask, pred_mask))
        hd95_scores.append(_hd95(true_mask, pred_mask, true_spacing))

    if len(dice_scores) != len(rows):
        return None
    mean_aji = float(np.mean(aji_scores))
    mean_dice = float(np.mean(dice_scores))
    finite_hd95 = [value for value in hd95_scores if math.isfinite(float(value))]
    mean_hd95 = (
        float(np.mean(finite_hd95))
        if len(finite_hd95) == len(hd95_scores) and finite_hd95
        else math.inf
    )
    if primary_metric in {"hd95", "hausdorff95", "hausdorff_95"}:
        metric_name = "private_hd95"
        value = mean_hd95
        maximize = False
    elif primary_metric in {"binary_dice", "dice", "mean_dice"}:
        metric_name = "private_binary_dice"
        value = mean_dice
        maximize = True
    elif primary_metric in {"aji", "mean_aji"}:
        metric_name = "private_mean_aji"
        value = mean_aji
        maximize = True
    else:
        return None
    if not math.isfinite(float(value)):
        return None
    return {
        "metric_name": metric_name,
        "value": value,
        "metrics": {
            "aji": mean_aji,
            "binary_dice": mean_dice,
            "hd95": mean_hd95,
        },
        "n": int(len(dice_scores)),
        "maximize": maximize,
    }


def _private_gt_path(private_dir: Path, value: Any) -> Path:
    path = Path(str(value).removeprefix("./"))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "input":
        path = Path(*path.parts[1:])
    return private_dir / path


def evaluate_private_outputs(
    workspace_dir: Path | str,
    private_dir: Path | str,
    *,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Evaluate private test metrics for user inspection only."""
    workspace_dir = Path(workspace_dir)
    private_dir = Path(private_dir)
    private_test = _load_private_test_plan(private_dir)
    if not private_test:
        return None

    contract = load_experiment_contract(workspace_dir)
    prediction_spec = contract.prediction.raw
    key_columns = contract.prediction.key_columns or _spec_list(
        prediction_spec, "key_columns"
    )
    prediction_columns = contract.prediction.prediction_columns
    primary_metric = contract.metric.primary_metric
    submission_path = contract.prediction.test_submission
    if not submission_path.is_file():
        return None

    if (workspace_dir / "evaluator.py").is_file():
        validation = validate_submission_with_evaluator(
            workspace_dir,
            timeout=120,
            env_overrides=env_overrides,
        )
        if not validation.ok:
            return {
                "visible_to_agent": False,
                "note": "Private metrics are for user inspection only and are not passed to the agent.",
                "results": [],
                "error": validation.error
                or "Frozen evaluator submission validation failed.",
            }

    submission = pd.read_csv(submission_path)
    results = []
    source = private_test.get("source")
    truth_path = private_dir / str(source)
    if not truth_path.is_file():
        return None

    truth = pd.read_csv(truth_path)
    label_column = private_test.get("label_column")
    mask_column = private_test.get("mask_column")
    label_columns = [
        str(column) for column in private_test.get("label_columns", []) if column
    ]
    id_column = private_test.get("id_column")
    result = None
    if mask_column:
        result = _private_mask_metrics(
            workspace_dir,
            private_dir,
            truth,
            submission,
            mask_column,
            prediction_columns,
            key_columns,
            primary_metric,
        )
    elif len(label_columns) > 1:
        result = _multilabel_metrics_from_submission(
            truth,
            submission,
            label_columns,
            id_column,
            prediction_columns,
            key_columns,
            primary_metric,
        )
    elif label_column:
        result = _classification_metrics_from_submission(
            truth,
            submission,
            label_column,
            id_column,
            prediction_columns,
            key_columns,
            primary_metric,
            prediction_spec,
        )
    if result:
        result["name"] = "private_test"
        result["source"] = source
        result["visible_to_agent"] = False
        results.append(result)

    if not results:
        return None
    return {
        "visible_to_agent": False,
        "note": "Private metrics are for user inspection only and are not passed to the agent.",
        "results": results,
    }
