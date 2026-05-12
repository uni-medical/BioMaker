"""Single run-level contract for split, prediction, metric, and hidden answers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metric_registry import metric_maximize
from .prediction_contract import (
    PredictionContract,
    artifact_path_from_spec,
    file_reference_columns_from_spec,
    key_columns_from_spec,
    list_of_strings,
    prediction_columns_from_spec,
    primary_metric_from_spec,
)


@dataclass(frozen=True)
class SplitContract:
    name: str
    path: Path
    public_ref: str
    role: str
    legacy_alias: str | None = None


@dataclass(frozen=True)
class HiddenAnswerContract:
    split: str
    path: Path
    answer_columns: list[str]


@dataclass(frozen=True)
class MetricContract:
    primary_metric: str | None
    maximize: bool
    secondary_metrics: list[str]


@dataclass(frozen=True)
class ExperimentContract:
    workspace_dir: Path
    split_dir: Path
    split_plan: dict[str, Any]
    split_manifest: dict[str, Any]
    splits: dict[str, SplitContract]
    hidden_answers: dict[str, HiddenAnswerContract]
    prediction: PredictionContract
    metric: MetricContract

    def split_path(self, name: str) -> Path | None:
        split = self.splits.get(name)
        return split.path if split is not None else None

    def hidden_answer_columns(self, split: str) -> list[str]:
        hidden = self.hidden_answers.get(split)
        return list(hidden.answer_columns) if hidden is not None else []


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _nested_dict(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    child = value.get(key)
    return child if isinstance(child, dict) else {}


def _public_ref(workspace_dir: Path, path: Path) -> str:
    try:
        return "./" + str(path.relative_to(workspace_dir))
    except ValueError:
        return str(path)


def _load_split_manifest(split_dir: Path) -> dict[str, Any]:
    manifest = _read_json_dict(split_dir / "split_manifest.json")
    if manifest:
        return manifest
    embedded = _nested_dict(
        _read_json_dict(split_dir / "split_plan.json"), "split_manifest"
    )
    return embedded


def _split_path_from_plan(
    workspace_dir: Path,
    split_dir: Path,
    split_plan: dict[str, Any],
    name: str,
    default_file: str,
    *,
    legacy_file: str | None = None,
) -> tuple[Path, str | None]:
    agent_visible = _nested_dict(split_plan, "agent_visible")
    raw = agent_visible.get(name)
    if isinstance(raw, str) and raw.strip():
        candidate = Path(raw.removeprefix("./"))
        if candidate.is_absolute():
            return candidate, None
        candidates = [
            workspace_dir / candidate,
            workspace_dir / "input" / candidate,
            split_dir / candidate,
        ]
        for path in candidates:
            if path.is_file():
                return path, None
        return candidates[0], None
    path = split_dir / default_file
    if path.is_file() or legacy_file is None:
        return path, None
    legacy = split_dir / legacy_file
    return legacy, legacy_file if legacy.is_file() else None


def _load_splits(
    workspace_dir: Path,
    split_dir: Path,
    split_plan: dict[str, Any],
) -> dict[str, SplitContract]:
    specs = [
        ("train", "training", "agent_train.csv", None),
        ("score_val", "score validation", "score_val.csv", "agent_val.csv"),
        ("test", "test prediction", "agent_test.csv", None),
    ]
    splits: dict[str, SplitContract] = {}
    for name, role, default_file, legacy_file in specs:
        path, legacy_alias = _split_path_from_plan(
            workspace_dir,
            split_dir,
            split_plan,
            name,
            default_file,
            legacy_file=legacy_file,
        )
        if path.is_file():
            splits[name] = SplitContract(
                name=name,
                path=path,
                public_ref=_public_ref(workspace_dir, path),
                role=role,
                legacy_alias=legacy_alias,
            )
    return splits


def _hidden_answer_source(
    split_plan: dict[str, Any], split_manifest: dict[str, Any]
) -> dict[str, Any]:
    for source in (
        split_manifest.get("hidden_answers"),
        split_manifest.get("private_gt"),
        split_plan.get("hidden_answers"),
        split_plan.get("private_gt"),
    ):
        if isinstance(source, dict):
            return source
    return {}


def _load_hidden_answers(
    workspace_dir: Path,
    split_plan: dict[str, Any],
    split_manifest: dict[str, Any],
    hidden_answers_dir: Path | None,
) -> dict[str, HiddenAnswerContract]:
    source = _hidden_answer_source(split_plan, split_manifest)
    private_dir = hidden_answers_dir or workspace_dir.parent / "private_gt"
    result: dict[str, HiddenAnswerContract] = {}
    for split in ("score_val", "test"):
        entry = source.get(split)
        if not isinstance(entry, dict):
            continue
        default_file = "score_val.csv" if split == "score_val" else "test.csv"
        file_name = str(entry.get("file") or default_file)
        result[split] = HiddenAnswerContract(
            split=split,
            path=private_dir / file_name,
            answer_columns=list_of_strings(entry.get("answer_columns")),
        )
    return result


def _load_prediction_spec(workspace_dir: Path) -> dict[str, Any]:
    return _read_json_dict(workspace_dir / "evaluator_prediction_spec.json")


def _manifest_primary_metric(split_manifest: dict[str, Any]) -> str | None:
    metric_contract = _nested_dict(split_manifest, "metric_contract")
    value = metric_contract.get("primary_metric")
    return str(value).strip().lower() if value else None


def _manifest_maximize(
    split_manifest: dict[str, Any], primary_metric: str | None
) -> bool:
    metric_contract = _nested_dict(split_manifest, "metric_contract")
    if isinstance(metric_contract.get("maximize"), bool):
        return bool(metric_contract["maximize"])
    return metric_maximize(primary_metric)


def _build_prediction_contract(
    workspace_dir: Path,
    split_plan: dict[str, Any],
    split_manifest: dict[str, Any],
    spec: dict[str, Any],
) -> PredictionContract:
    manifest_contract = _nested_dict(split_manifest, "prediction_contract")
    manifest_metric = _nested_dict(split_manifest, "metric_contract")
    validation_artifact = spec.get("validation_artifact") or manifest_contract.get(
        "validation_artifact"
    )
    test_submission = spec.get("test_submission") or manifest_contract.get(
        "test_submission"
    )
    key_columns = (
        key_columns_from_spec(spec)
        or list_of_strings(split_manifest.get("key_columns"))
        or list_of_strings(manifest_contract.get("key_columns"))
    )
    file_reference_columns = file_reference_columns_from_spec(spec) or list_of_strings(
        manifest_contract.get("file_reference_columns")
    )
    prediction_columns = prediction_columns_from_spec(spec) or (
        list_of_strings(manifest_contract.get("prediction_columns"))
        or list_of_strings(manifest_contract.get("file_reference_columns"))
        or list_of_strings(manifest_contract.get("columns"))
    )
    primary_metric = (
        primary_metric_from_spec(spec)
        or (
            str(manifest_metric.get("primary_metric")).strip().lower()
            if manifest_metric.get("primary_metric")
            else None
        )
        or (
            str(split_plan.get("primary_metric")).strip().lower()
            if split_plan.get("primary_metric")
            else None
        )
    )
    return PredictionContract(
        raw=spec,
        validation_artifact=artifact_path_from_spec(
            workspace_dir,
            validation_artifact,
            workspace_dir / "working" / "validation_predictions.csv",
        ),
        test_submission=artifact_path_from_spec(
            workspace_dir,
            test_submission,
            workspace_dir / "working" / "submission.csv",
        ),
        key_columns=key_columns,
        prediction_columns=prediction_columns,
        file_reference_columns=file_reference_columns,
        primary_metric=primary_metric,
        maximize=bool(
            spec.get(
                "maximize",
                manifest_metric.get("maximize", metric_maximize(primary_metric)),
            )
        ),
        prediction_type=(
            str(spec.get("prediction_type"))
            if spec.get("prediction_type")
            else (
                str(manifest_contract.get("prediction_type"))
                if manifest_contract.get("prediction_type")
                else (
                    str(spec.get("task_type"))
                    if spec.get("task_type")
                    else (
                        str(split_plan.get("task_type"))
                        if split_plan.get("task_type")
                        else None
                    )
                )
            )
        ),
        task_type=(
            str(spec.get("task_type"))
            if spec.get("task_type")
            else (
                str(split_manifest.get("task_type"))
                if split_manifest.get("task_type")
                else (
                    str(split_plan.get("task_type"))
                    if split_plan.get("task_type")
                    else None
                )
            )
        ),
        class_labels=(
            spec.get("class_labels")
            if isinstance(spec.get("class_labels"), list)
            else None
        ),
        positive_class=spec.get("positive_class"),
    )


def load_experiment_contract(
    workspace_dir: Path | str,
    *,
    hidden_answers_dir: Path | str | None = None,
) -> ExperimentContract:
    workspace_dir = Path(workspace_dir)
    hidden_answers_path = Path(hidden_answers_dir) if hidden_answers_dir else None
    split_dir = workspace_dir / "input" / "biomaker_splits"
    split_plan = _read_json_dict(split_dir / "split_plan.json")
    split_manifest = _load_split_manifest(split_dir)
    prediction_spec = _load_prediction_spec(workspace_dir)
    prediction = _build_prediction_contract(
        workspace_dir, split_plan, split_manifest, prediction_spec
    )
    primary_metric = prediction.primary_metric or _manifest_primary_metric(
        split_manifest
    )
    metric = MetricContract(
        primary_metric=primary_metric,
        maximize=(
            prediction.maximize
            if prediction.primary_metric
            else _manifest_maximize(split_manifest, primary_metric)
        ),
        secondary_metrics=list_of_strings(
            prediction_spec.get("secondary_metrics") or prediction_spec.get("metrics")
        ),
    )
    return ExperimentContract(
        workspace_dir=workspace_dir,
        split_dir=split_dir,
        split_plan=split_plan,
        split_manifest=split_manifest,
        splits=_load_splits(workspace_dir, split_dir, split_plan),
        hidden_answers=_load_hidden_answers(
            workspace_dir, split_plan, split_manifest, hidden_answers_path
        ),
        prediction=prediction,
        metric=metric,
    )
