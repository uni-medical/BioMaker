"""Shared prediction artifact contract helpers.

This module is intentionally small: it owns the normalized public contract that
nodes, artifact validation, final retrain, and private evaluation all consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _nested_dict(spec: dict[str, Any], key: str) -> dict[str, Any]:
    value = spec.get(key)
    return value if isinstance(value, dict) else {}


def _first_non_empty_list(*values: Any) -> list[str]:
    for value in values:
        items = list_of_strings(value)
        if items:
            return items
    return []


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def path_from_spec_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ["path", "output_path", "file"]:
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_split_manifest_for_workspace(workspace_dir: Path | str) -> dict[str, Any]:
    from .experiment_contract import _load_split_manifest

    return _load_split_manifest(Path(workspace_dir) / "input" / "biomaker_splits")


def primary_metric_from_spec(spec: dict[str, Any]) -> str | None:
    submission_spec = _nested_dict(spec, "test_submission")
    for container in [spec, submission_spec]:
        value = container.get("primary_metric")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def key_columns_from_spec(spec: dict[str, Any]) -> list[str]:
    key_columns = list_of_strings(spec.get("key_columns"))
    if key_columns:
        return key_columns
    return list_of_strings(_nested_dict(spec, "test_submission").get("key_columns"))


def prediction_columns_from_spec(spec: dict[str, Any]) -> list[str]:
    key_columns = set(key_columns_from_spec(spec))
    submission_spec = _nested_dict(spec, "test_submission")
    for container in [spec, submission_spec]:
        for key in ["prediction_columns", "file_reference_columns", "columns"]:
            columns = list_of_strings(container.get(key))
            columns = [column for column in columns if column not in key_columns]
            if columns:
                return columns
    format_value = _nested_dict(spec, "format")
    raw_columns = format_value.get("columns")
    if isinstance(raw_columns, list):
        return [str(column) for column in raw_columns if str(column) not in key_columns]
    if isinstance(raw_columns, dict):
        return [str(column) for column in raw_columns if str(column) not in key_columns]
    return []


def file_reference_columns_from_spec(spec: dict[str, Any]) -> list[str]:
    columns = list_of_strings(spec.get("file_reference_columns"))
    if columns:
        return columns
    nested = list_of_strings(
        _nested_dict(spec, "test_submission").get("file_reference_columns")
    )
    return nested


def artifact_path_from_spec(workspace_dir: Path, value: Any, fallback: Path) -> Path:
    raw = path_from_spec_value(value)
    if not raw:
        return fallback
    path = Path(raw.removeprefix("./"))
    return path if path.is_absolute() else workspace_dir / path


def resolve_artifact_path(workspace_dir: Path, value: str) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    normalized = Path(str(path).removeprefix("./"))
    candidates = [
        workspace_dir / normalized,
        workspace_dir / "working" / normalized,
        workspace_dir / "input" / normalized,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return workspace_dir / "working" / normalized


@dataclass(frozen=True)
class PredictionContract:
    raw: dict[str, Any]
    validation_artifact: Path
    test_submission: Path | None
    key_columns: list[str]
    prediction_columns: list[str]
    file_reference_columns: list[str]
    primary_metric: str | None
    maximize: bool
    prediction_type: str | None
    task_type: str | None
    class_labels: list[Any] | None
    positive_class: Any | None
    metrics: list[dict[str, Any]] = field(default_factory=list)


def load_prediction_spec(workspace_dir: Path | str) -> dict[str, Any]:
    workspace_dir = Path(workspace_dir)
    spec_path = workspace_dir / "evaluator_prediction_spec.json"
    if not spec_path.is_file():
        return {}
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_prediction_contract(workspace_dir: Path | str) -> PredictionContract:
    # Compatibility entry point. The run-level ExperimentContract is the single
    # owner of split/evaluator/metric facts; existing callers that only need the
    # prediction view still import this helper.
    from .experiment_contract import load_experiment_contract

    return load_experiment_contract(workspace_dir).prediction
