"""Split-independent scoring contract shared by validation and private testing."""

from __future__ import annotations

import math
import re
from typing import Any
from pathlib import Path

import pandas as pd

EVALUATOR_API_VERSION = 2


def validate_scoring_rows(
    prediction_path: Path, truth_path: Path, expected_path: Path, spec: dict[str, Any]
) -> int:
    """Align the sample universe, without interpreting task-specific cells."""
    keys = spec.get("key_columns", [])
    frames = [
        pd.read_csv(path, dtype={key: str for key in keys})
        for path in (prediction_path, truth_path, expected_path)
    ]
    pred, truth, expected = frames
    if not len(expected):
        raise ValueError("Cannot score an empty split.")
    for label, frame in zip(("predictions", "answers", "expected rows"), frames):
        if len(frame) != len(expected):
            raise ValueError(f"{label} row count does not match the scoring split.")
        if any(key not in frame.columns for key in keys):
            raise ValueError(f"{label} missing sample key columns.")
        if keys and (frame[keys].isna().any().any() or frame.duplicated(keys).any()):
            raise ValueError(f"{label} has missing or duplicate sample keys.")
    if keys:
        expected_keys = set(expected[keys].itertuples(index=False, name=None))
        if any(
            set(frame[keys].itertuples(index=False, name=None)) != expected_keys
            for frame in (pred, truth)
        ):
            raise ValueError(
                "Predictions or answers do not match the scoring sample keys."
            )
    if any(column not in pred.columns for column in spec["prediction_columns"]):
        raise ValueError("Predictions missing declared output columns.")
    return len(expected)


def metric_name(value: str) -> str:
    """Canonicalize metric names, not metric parameters or task types."""
    from .metric_registry import metric_info

    name = value.strip().lower().strip("`*_ .")
    registered = metric_info(name)
    if registered is not None:
        name = registered.name
    aliases = {
        "acc": "accuracy",
        "roc_auc": "auc",
        "macro_auc": "auc",
        "macro_roc_auc": "auc",
        "ovr_auc": "auc",
        "mean_roc_auc": "auc",
        "mean_binary_accuracy": "accuracy",
        "dice": "binary_dice",
        "mean_dice": "binary_dice",
        "mean_aji": "aji",
        "hausdorff95": "hd95",
        "hausdorff_95": "hd95",
        "map50": "ap50",
        "map@0.5": "ap50",
        "map@0.50": "ap50",
        "coco_map_50_95": "map50_95",
        "map@0.5:0.95": "map50_95",
    }
    return aliases.get(name, name)


# Exact prose spellings describe identities, not substring matches. In particular,
# PR-AUC, partial ROC AUC and class-wise Dice must never become ROC AUC or binary
# foreground Dice merely because their names contain "auc" or "dice".
_DECLARED_METRIC_NAMES = {
    "roc auc": "auc",
    "macro roc auc": "auc",
    "macro one-vs-rest roc auc": "auc",
    "mean one-vs-rest roc auc": "auc",
    "mean binary accuracy": "accuracy",
    "mean binary accuracy across labels": "accuracy",
    "binary dice": "binary_dice",
    "aggregated jaccard index": "aji",
    "aggregated jaccard index (aji)": "aji",
    "hausdorff distance 95": "hd95",
    "95th percentile hausdorff distance": "hd95",
    "balanced accuracy": "balanced_accuracy",
    "mean absolute error": "mae",
    "mean squared error": "mse",
    "root mean squared error": "rmse",
    "intersection over union": "iou",
    "intersection-over-union": "iou",
    "pr-auc": "pr_auc",
    "pr auc": "pr_auc",
    "precision-recall auc": "pr_auc",
    "average precision": "average_precision",
}
_EXPLANATION_PREFIX = re.compile(
    r"^(?:in|measured|reported|computed|calculated|evaluated|treating|where|"
    r"without|with units)\b", re.IGNORECASE
)


def _declared_metric_name(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip().strip("`*_ .").lower()
    # Optimization direction is descriptive; metric parameters are not. Keep
    # unrecognized parenthesized settings in the identity so unsupported variants
    # are reported instead of silently receiving the standard implementation.
    text = re.sub(r"\s*\((?:higher|lower) is better\)", "", text).strip()
    if re.fullmatch(r"recall at iou\s*=\s*0\.50*", text):
        return "recall"
    if text in _DECLARED_METRIC_NAMES:
        return _DECLARED_METRIC_NAMES[text]
    expanded = re.fullmatch(r"(.+)\s+\(([^()]+)\)", text)
    if expanded and expanded[1] in _DECLARED_METRIC_NAMES:
        canonical = _DECLARED_METRIC_NAMES[expanded[1]]
        if metric_name(expanded[2]) == canonical:
            return canonical
    canonical = metric_name(text)
    if canonical != text:
        return canonical
    return re.sub(r"[\s-]+", "_", text)


def _is_metric_list_item(value: str) -> bool:
    text = value.strip().strip("`*_ .").lower()
    return text in _DECLARED_METRIC_NAMES or bool(
        re.fullmatch(r"[a-z][a-z0-9_@.:-]*", text)
    )


def _metric_items(line: str) -> list[str]:
    # A period followed by another sentence ends the declaration; decimals in
    # mAP@0.5 and numeric settings do not. Semicolons explicitly delimit metrics.
    line = re.split(r"\.(?=\s+[A-Za-z]|$)", line, maxsplit=1)[0]
    values = []
    for item in line.split(";"):
        item = item.strip().strip("`*_ .")
        if not item:
            continue
        comma_parts = [part.strip() for part in item.split(",")]
        if len(comma_parts) > 1 and _EXPLANATION_PREFIX.match(comma_parts[1]):
            comma_parts = comma_parts[:1]
        elif not all(_is_metric_list_item(part) for part in comma_parts):
            comma_parts = [item]
        for part in comma_parts:
            # A conjunction is a list only when each side is a recognizable name
            # or identifier. Do not split prose qualifiers within unknown metrics.
            conjunction = re.split(r"\s+(?:and|&)\s+", part, flags=re.IGNORECASE)
            if len(conjunction) > 1 and all(
                _is_metric_list_item(value)
                for value in conjunction
            ):
                values.extend(_declared_metric_name(value) for value in conjunction)
            else:
                values.append(_declared_metric_name(part))
    return list(dict.fromkeys(values))


def requested_metrics(task: str) -> tuple[str | None, list[str]]:
    """Read explicit declarations without replacing unsupported metric variants.

    Metric names use exact aliases. Explanatory text such as ``MAE, in months``
    is excluded, while explicit secondary lists may use semicolons, commas or
    unambiguous conjunctions. Unknown names remain available to extension routing.
    """
    def parse(role: str) -> list[str]:
        matches = re.finditer(
            rf"{role}\s+(?:(?:validation|evaluation)\s+)?metrics?\s*:\s*([^\n\r]+)",
            task,
            re.IGNORECASE,
        )
        return list(dict.fromkeys(name for match in matches for name in _metric_items(match.group(1))))

    primary = parse("primary")
    secondary = parse("secondary")
    if not primary:
        generic = re.search(
            r"^\s*(?:[-*]\s+)?(?:evaluation\s+|validation\s+)?metrics?\s*:\s*([^\n\r]+)",
            task, re.IGNORECASE | re.MULTILINE,
        )
        if generic:
            primary = _metric_items(generic[1])
            secondary = list(dict.fromkeys(primary[1:] + secondary))
    return (metric_name(primary[0]) if primary else None), list(dict.fromkeys(metric_name(name) for name in secondary))


def validate_score(score: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Reject missing metrics and silent substitutions identically for all splits."""
    if not isinstance(score, dict) or not isinstance(score.get("metrics"), dict):
        raise ValueError("score() must return a dict with a named metrics mapping.")
    primary = metric_name(str(spec["primary_metric"]))
    metrics = {}
    for name, value in score["metrics"].items():
        key = metric_name(str(name))
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"Metric {name!r} must be a finite number; no fallback is allowed."
            )
        if key in metrics and metrics[key] != float(value):
            raise ValueError(f"Conflicting values for metric {key!r}.")
        metrics[key] = float(value)
    required = {
        primary,
        *(metric_name(str(m)) for m in spec.get("secondary_metrics", [])),
    }
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"Missing requested metrics: {sorted(missing)}")
    declared = score.get("metric")
    if declared is not None and (
        isinstance(declared, bool) or declared != metrics[primary]
    ):
        raise ValueError("Returned metric does not match the declared primary metric.")
    maximize = spec.get("maximize")
    if not isinstance(maximize, bool):
        raise ValueError("prediction_spec must declare boolean maximize.")
    if "maximize" in score and score["maximize"] != maximize:
        raise ValueError("Score direction disagrees with prediction_spec.")
    return {
        **score,
        "metric": metrics[primary],
        "maximize": maximize,
        "metrics": metrics,
    }
