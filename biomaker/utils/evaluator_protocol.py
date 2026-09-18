"""Split-independent scoring contract shared by validation and private testing."""

from __future__ import annotations

import math
import copy
import json
import re
from typing import Any
from pathlib import Path

import pandas as pd

EVALUATOR_API_VERSION = 3


def metric_definitions(spec: dict[str, Any], *, allow_legacy_unknown: bool = False) -> list[dict[str, Any]]:
    """Return canonical, distinct objectives with explicit independent directions.

    Historical v2 scoring may leave an unregistered secondary direction unknown:
    those values remain diagnostics, never new selection objectives.
    """
    from .trusted_metrics import normalize_metric_definitions

    if not isinstance(spec, dict):
        raise ValueError('Metric specification must be a mapping.')
    spec = copy.deepcopy(spec)
    def canonical(value):
        if isinstance(value, str):
            return metric_name(value)
        if isinstance(value, dict):
            return {key: metric_name(item) if key in {'name', 'metric', 'id'} and isinstance(item, str) else item for key, item in value.items()}
        return value
    if 'metrics' in spec and isinstance(spec['metrics'], list):
        spec['metrics'] = [canonical(item) for item in spec['metrics']]
    if 'primary_metric' in spec:
        spec['primary_metric'] = canonical(spec['primary_metric'])
    if isinstance(spec.get('secondary_metrics'), list):
        spec['secondary_metrics'] = [canonical(item) for item in spec['secondary_metrics']]
    return normalize_metric_definitions(spec, allow_legacy_unknown=allow_legacy_unknown)


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
    name = _DECLARED_METRIC_NAMES.get(name, name)
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
    structured = requested_metric_definitions(task)
    if structured:
        return structured[0]['name'], [item['name'] for item in structured[1:]]

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


def requested_metric_contract(task: str) -> dict[str, Any] | None:
    """Read and preserve the complete explicit metrics JSON/YAML contract.

    Prose-only legacy declarations are handled by requested_metrics; directions
    then come from the prepared contract or registered metric catalog.
    """
    import yaml
    from .metric_declarations import canonical_metric_contract

    declarations = []
    for language, block in re.findall(r'```(metrics|yaml|yml|json)\s*\n(.*?)```', task, re.S | re.I):
        try:
            # JSON scientific notation (e.g. 1e-05 Dice smoothing) is numeric.
            # YAML 1.1 interprets that spelling as a string; preserve JSON
            # types before trying the more permissive YAML representation.
            try:
                value = json.loads(block)
            except json.JSONDecodeError:
                value = yaml.safe_load(block)
        except (ValueError, yaml.YAMLError):
            if language.lower() == 'metrics':
                raise ValueError('The metrics block must contain valid JSON or YAML.')
            continue
        if language.lower() == 'metrics' or isinstance(value, dict) and 'metrics' in value:
            declarations.append(value if isinstance(value, dict) else {'metrics': value})
    if len(declarations) > 1:
        raise ValueError('Declare exactly one metrics block.')
    if declarations:
        if 'metrics' not in declarations[0]:
            raise ValueError('The metrics block must declare a nonempty metrics list.')
        return canonical_metric_contract(declarations[0])
    match = re.search(r'^metrics:\s*\n((?:[ \t]+[^\n]*\n?|\s*-\s*[^\n]*\n?)+)', task, re.M)
    if match:
        return canonical_metric_contract(yaml.safe_load(match[0]))
    return None


def requested_metric_definitions(task: str) -> list[dict[str, Any]] | None:
    contract = requested_metric_contract(task)
    return contract['metrics'] if contract is not None else None


def validate_task_metric_contract(actual: dict[str, Any], task: str) -> None:
    """Reject changed or dropped settings from an explicit task metrics block."""
    from .metric_declarations import canonical_metric_contract

    expected = requested_metric_contract(task)
    if expected is None:
        return
    actual = canonical_metric_contract(actual)
    if {item['name']: item for item in metric_definitions(actual)} != {item['name']: item for item in expected['metrics']}:
        raise ValueError('Task and prepared metric_contract disagree on metrics, directions or parameters.')

    def preserves(value, requested):
        if isinstance(requested, dict):
            return isinstance(value, dict) and all(key in value and preserves(value[key], item) for key, item in requested.items())
        return value == requested

    for key, value in expected.items():
        if key not in {'metrics', 'notes', 'description'} and (key not in actual or not preserves(actual[key], value)):
            raise ValueError('Task and prepared metric_contract disagree on metric protocol field: ' + key)


def validate_score(score: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Reject missing metrics and silent substitutions identically for all splits."""
    if not isinstance(score, dict) or not isinstance(score.get("metrics"), dict):
        raise ValueError("score() must return a dict with a named metrics mapping.")
    legacy = spec.get('evaluator_api_version', 2) == 2 and 'primary_metric' in spec
    definitions = metric_definitions(spec, allow_legacy_unknown=legacy)
    primary = metric_name(str(spec['primary_metric'])) if legacy else None
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
    required = {item['name'] for item in definitions}
    if legacy:
        required.update(metric_name(str(m)) for m in spec.get('secondary_metrics', []))
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"Missing requested metrics: {sorted(missing)}")
    if not legacy:
        if 'rank' in score or 'average_rank' in score:
            raise ValueError('Evaluator returns raw metrics; ranks are computed by the Journal.')
        result = {**score, 'metrics': metrics, 'metric_specs': definitions}
        result.pop('metric', None)
        result.pop('maximize', None)
        return result
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
        "metric_specs": definitions,
    }
