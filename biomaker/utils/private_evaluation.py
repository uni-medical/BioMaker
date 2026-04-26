"""User-only evaluation that is never fed back to the agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import rankdata

from .artifact_validation import _resolve_artifact_path


def _load_split_plan(private_dir: Path) -> dict[str, Any] | None:
    private_path = private_dir / "private_split_plan.json"
    if private_path.is_file():
        return json.loads(private_path.read_text(encoding="utf-8"))
    return None


def _load_prediction_spec(workspace_dir: Path) -> dict[str, Any]:
    spec_path = workspace_dir / "evaluator_prediction_spec.json"
    if not spec_path.is_file():
        return {}
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return spec if isinstance(spec, dict) else {}


def _submission_path_from_spec(workspace_dir: Path, spec: dict[str, Any]) -> Path:
    value = spec.get("test_submission")
    if isinstance(value, dict):
        value = value.get("path") or value.get("output_path")
    raw_path = str(value or "./working/submission.csv").removeprefix("./")
    path = Path(raw_path)
    return path if path.is_absolute() else workspace_dir / path


def _test_submission_spec(spec: dict[str, Any]) -> dict[str, Any]:
    value = spec.get("test_submission")
    return value if isinstance(value, dict) else {}


def _spec_list(spec: dict[str, Any], key: str) -> list[str]:
    value = spec.get(key)
    if not isinstance(value, list):
        value = _test_submission_spec(spec).get(key)
    return [str(item) for item in value] if isinstance(value, list) else []


def _prediction_columns_from_spec(spec: dict[str, Any]) -> list[str]:
    key_columns = set(_spec_list(spec, "key_columns"))
    for container in [spec, _test_submission_spec(spec)]:
        for key in ["prediction_columns", "file_reference_columns", "columns"]:
            value = container.get(key)
            if isinstance(value, list):
                columns = [str(item) for item in value]
                columns = [column for column in columns if column not in key_columns]
                if columns:
                    return columns
    format_value = spec.get("format")
    if isinstance(format_value, dict):
        raw_columns = format_value.get("columns")
        if isinstance(raw_columns, list):
            return [
                str(column) for column in raw_columns if str(column) not in key_columns
            ]
        if isinstance(raw_columns, dict):
            return [
                str(column) for column in raw_columns if str(column) not in key_columns
            ]
    return []


def _primary_metric_from_spec(spec: dict[str, Any]) -> str | None:
    for container in [spec, _test_submission_spec(spec)]:
        value = container.get("primary_metric")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    metrics = spec.get("metrics")
    if isinstance(metrics, list) and metrics:
        return str(metrics[0]).lower()
    return None


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


def _dice(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    if mask_true.ndim == 3:
        mask_true = mask_true[..., 0]
    if mask_pred.ndim == 3:
        mask_pred = mask_pred[..., 0]
    true_bin = mask_true > 0
    pred_bin = mask_pred > 0
    denom = true_bin.sum() + pred_bin.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(true_bin, pred_bin).sum() / denom)


def _connected_components(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.max() > 1 and len(np.unique(mask)) > 2:
        return mask.astype(np.int32, copy=False)
    labels, _ = ndi.label(mask > 0, structure=np.ones((3, 3), dtype=np.uint8))
    return labels.astype(np.int32, copy=False)


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
    for _, row in rows.iterrows():
        true_path = _private_truth_path(private_dir, row[mask_column])
        if not true_path.is_file():
            return None
        with Image.open(true_path) as true_image:
            true_mask = np.array(true_image)
            pred_path = _resolve_artifact_path(
                workspace_dir, str(row[pred_artifact_column])
            )
            if not pred_path.is_file():
                return None
            with Image.open(pred_path) as pred_image:
                if true_image.size != pred_image.size:
                    return None
                pred_mask = np.array(pred_image)
            dice_scores.append(_dice(true_mask, pred_mask))
            aji_scores.append(_aji(true_mask, pred_mask))

    if len(dice_scores) != len(rows):
        return None
    mean_aji = float(np.mean(aji_scores))
    mean_dice = float(np.mean(dice_scores))
    if primary_metric in {"binary_dice", "dice", "mean_dice"}:
        metric_name = "private_binary_dice"
        value = mean_dice
    else:
        metric_name = "private_mean_aji"
        value = mean_aji
    return {
        "metric_name": metric_name,
        "value": value,
        "metrics": {
            "aji": mean_aji,
            "binary_dice": mean_dice,
        },
        "n": int(len(dice_scores)),
        "maximize": True,
    }


def _private_truth_path(private_dir: Path, value: Any) -> Path:
    path = Path(str(value).removeprefix("./"))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "input":
        path = Path(*path.parts[1:])
    return private_dir / path


def evaluate_private_outputs(
    workspace_dir: Path | str, private_dir: Path | str
) -> dict[str, Any] | None:
    """Evaluate private test metrics for user inspection only."""
    workspace_dir = Path(workspace_dir)
    private_dir = Path(private_dir)
    split_plan = _load_split_plan(private_dir)
    if not split_plan:
        return None

    prediction_spec = _load_prediction_spec(workspace_dir)
    key_columns = _spec_list(prediction_spec, "key_columns")
    prediction_columns = _prediction_columns_from_spec(prediction_spec)
    primary_metric = _primary_metric_from_spec(prediction_spec)
    submission_path = _submission_path_from_spec(workspace_dir, prediction_spec)
    if not submission_path.is_file():
        return None

    submission = pd.read_csv(submission_path)
    results = []
    for private_eval in split_plan.get("private_evaluation", []):
        source = private_eval.get("source")
        if not source:
            continue
        truth_path = private_dir / source
        if not truth_path.is_file():
            continue

        truth = pd.read_csv(truth_path)
        label_column = private_eval.get("label_column")
        mask_column = private_eval.get("mask_column")
        id_column = private_eval.get("id_column")
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
            result["name"] = private_eval.get("name", "private_eval")
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
