"""Run-level experiment guide used as the structured pre-node fact source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .experiment_contract import load_experiment_contract

EXP_GUIDE_VERSION = 1
EXP_GUIDE_FILENAME = "exp_guide.json"
MEDICAL_IMAGE_SUFFIXES = {".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"}


def _compound_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _safe_rel(path: Path, base: Path) -> str:
    try:
        return "./" + str(path.relative_to(base))
    except ValueError:
        return str(path)


def _csv_profile(
    path: Path, base: Path
) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    if not path.is_file():
        return None, None
    try:
        df = pd.read_csv(path, nrows=1)
    except Exception as exc:
        return {
            "path": _safe_rel(path, base),
            "read_error": str(exc),
        }, None
    rows = 0
    columns = [str(column) for column in df.columns.tolist()]
    if columns:
        try:
            for chunk in pd.read_csv(path, usecols=[columns[0]], chunksize=100_000):
                rows += int(len(chunk))
        except Exception:
            rows = int(len(pd.read_csv(path)))
    return {
        "path": _safe_rel(path, base),
        "rows": rows,
        "columns": columns,
    }, df


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _visible_label_profile(
    df: pd.DataFrame | None,
    *,
    answer_columns: set[str],
    key_columns: set[str],
) -> dict[str, Any] | None:
    """Summarize labels that are already visible in the training split."""
    if df is None or df.empty:
        return None
    profiles: dict[str, Any] = {}
    for column in df.columns:
        column_name = str(column)
        lower = column_name.lower()
        if column_name in key_columns:
            continue
        if "path" in lower or "file" in lower:
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        should_include = (
            column_name in answer_columns
            or lower in {"label", "class", "target", "y"}
            or lower.startswith("label")
            or lower.startswith("class")
        )
        if not should_include:
            continue
        counts = series.astype(str).value_counts(dropna=False)
        profiles[column_name] = {
            "rows": int(len(series)),
            "unique_count": int(counts.shape[0]),
            "counts": {str(index): int(value) for index, value in counts.items()},
        }
    return profiles or None


def _path_observation(path_value: Any, input_dir: Path) -> dict[str, Any] | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (
            input_dir.parent / path
            if path.parts and path.parts[0] == "input"
            else input_dir / path
        )
    result: dict[str, Any] = {
        "value": path_value,
        "exists": path.is_file(),
    }
    if not path.is_file():
        return result
    result["size_bytes"] = int(path.stat().st_size)
    if path.suffix.lower() in {".npy", ".npz"}:
        try:
            if path.suffix.lower() == ".npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                result.update(kind="numpy_array", array_shape=list(array.shape), array_dtype=str(array.dtype))
            else:
                with np.load(path, allow_pickle=False) as archive:
                    result.update(kind="numpy_archive", arrays={
                        key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
                        for key in archive.files[:8]
                    })
            return result
        except Exception as exc:
            result["read_error"] = str(exc)
            return result
    try:
        with Image.open(path) as image:
            arr = np.asarray(image)
            result.update(
                {
                    "kind": "image",
                    "mode": image.mode,
                    "pil_size": list(image.size),
                    "array_shape": list(arr.shape),
                    "array_dtype": str(arr.dtype),
                    "frame_count": getattr(image, "n_frames", 1),
                }
            )
            return result
    except Exception as exc:
        result["read_error"] = str(exc)
    if _compound_suffix(path) in MEDICAL_IMAGE_SUFFIXES:
        try:
            import SimpleITK as sitk

            image = sitk.ReadImage(str(path))
            result.update(
                {
                    "kind": "medical_image",
                    "reader": "SimpleITK",
                    "array_shape": list(reversed(image.GetSize())),
                    "spacing": list(image.GetSpacing()),
                    "pixel_type": image.GetPixelIDTypeAsString(),
                }
            )
            result.pop("read_error", None)
        except Exception as exc:
            result["read_error"] = str(exc)
    return result


def _sample_profile(df: pd.DataFrame | None, input_dir: Path, *, declared_columns=(), file_columns=()) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    values: dict[str, Any] = {}
    columns = list(dict.fromkeys([c for c in declared_columns if c in df] + list(df.columns[:8])))
    for column in columns:
        value = row[column]
        if isinstance(value, np.generic):
            value = value.item()
        values[str(column)] = (
            value if isinstance(value, (str, int, float, bool)) else str(value)
        )

    path_observations = {}
    for column in dict.fromkeys([c for c in file_columns if c in df] + list(df.columns)):
        column_name = str(column)
        value = row[column]
        declared_file = column in file_columns
        looks_like_path = isinstance(value, str) and ("/" in value or "." in value)
        if not declared_file and not looks_like_path and "path" not in column_name.lower() and "file" not in column_name.lower():
            continue
        observation = _path_observation(value, input_dir)
        if observation is not None and (declared_file or observation["exists"] or "path" in column_name.lower() or "file" in column_name.lower()):
            path_observations[column_name] = observation
        if len(path_observations) >= 4:
            break
    return {
        "first_row": values,
        "path_observations": path_observations,
    }


def build_exp_guide(
    workspace_dir: Path | str,
    *,
    prediction_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an experiment guide from the current workspace artifacts."""
    workspace_dir = Path(workspace_dir)
    input_dir = workspace_dir / "input"
    split_dir = input_dir / "biomaker_splits"
    contract = load_experiment_contract(workspace_dir)
    prediction_contract = contract.prediction
    raw_prediction_spec = prediction_spec or prediction_contract.raw

    split_guide: dict[str, Any] = {}
    samples: dict[str, Any] = {}
    split_profiles: dict[str, Any] = {}
    answer_columns = {
        column
        for hidden in contract.hidden_answers.values()
        for column in hidden.answer_columns
    }
    key_columns = set(prediction_contract.key_columns)
    manifest = contract.split_manifest
    declared_prediction = manifest.get("prediction_contract", {})
    declared_metric = manifest.get("metric_contract", {})
    restoration = declared_metric.get("restoration_evaluation", declared_metric)
    file_columns = list(dict.fromkeys([
        *manifest.get("input_file_reference_columns", []),
        *declared_prediction.get("input_file_reference_columns", []),
        *declared_prediction.get("file_reference_columns", []),
        *[c for c in [declared_prediction.get("image_column"), restoration.get("input_column"), restoration.get("target_column")] if isinstance(c, str)],
    ]))
    declared_columns = list(dict.fromkeys([
        *manifest.get("key_columns", []), *declared_prediction.get("input_columns", []),
        *declared_prediction.get("prediction_columns", []), *sorted(answer_columns), *file_columns,
    ]))
    for split_name in ["train", "score_val", "test"]:
        split = contract.splits.get(split_name)
        if split is None:
            continue
        profile, df = _csv_profile(split.path, workspace_dir)
        if profile is None:
            continue
        profile["role"] = (
            "legacy public validation" if split.legacy_alias is not None else split.role
        )
        split_guide[split_name] = profile
        sample = _sample_profile(df, input_dir, declared_columns=declared_columns, file_columns=file_columns)
        if sample is not None:
            samples[split_name] = sample
        if split_name == "train":
            full_df = _safe_read_csv(split.path)
            label_profile = _visible_label_profile(
                full_df,
                answer_columns=answer_columns,
                key_columns=key_columns,
            )
            if label_profile:
                split_profiles[split_name] = {"visible_labels": label_profile}

    hidden_answers = {
        split: hidden.answer_columns
        for split, hidden in contract.hidden_answers.items()
        if hidden.answer_columns
    }

    score_split = raw_prediction_spec.get("score_split")
    if not score_split and "score_val" in contract.splits:
        score_split = contract.splits["score_val"].public_ref

    prediction = {
        "validation_artifact": _safe_rel(
            prediction_contract.validation_artifact, workspace_dir
        ),
        "test_submission": _safe_rel(
            prediction_contract.test_submission, workspace_dir
        ),
        "key_columns": prediction_contract.key_columns,
        "prediction_columns": prediction_contract.prediction_columns,
        "file_reference_columns": prediction_contract.file_reference_columns,
        "prediction_type": prediction_contract.prediction_type or "columns",
        "class_labels": prediction_contract.class_labels,
        "positive_class": prediction_contract.positive_class,
        "score_split": score_split,
    }
    metric = {
        "primary_metric": contract.metric.primary_metric,
        "maximize": contract.metric.maximize,
        "secondary_metrics": contract.metric.secondary_metrics,
    }

    dataset_report = _read_json_dict(split_dir / "dataset_report.json")
    return {
        "version": EXP_GUIDE_VERSION,
        "source": "workspace_artifacts",
        "dataset": {
            "input_root": "./input",
            "dataset_report": dataset_report,
        },
        "splits": split_guide,
        "target": {
            "hidden_answers": hidden_answers,
            "answer_columns": sorted(
                {column for columns in hidden_answers.values() for column in columns}
            ),
        },
        "prediction": prediction,
        "metric": metric,
        "evaluation_protocol": {
            key: value for key, value in raw_prediction_spec.items()
            if key not in {"answer_columns", "truth_schema"}
        },
        "data_semantics": {key: manifest[key] for key in ("data_semantics", "row_representation", "semantic_assumptions", "sample_unit", "task_heads", "input_file_reference_columns") if key in manifest},
        "visibility": {
            "agent_input_root": "./input",
            "agent_output_root": "./working",
            "hidden_answers_outside_agent_workspace": True,
        },
        "samples": samples,
        "split_profiles": split_profiles,
    }


def write_exp_guide(
    workspace_dir: Path | str,
    *,
    prediction_spec: dict[str, Any] | None = None,
) -> Path:
    workspace_dir = Path(workspace_dir)
    guide = build_exp_guide(
        workspace_dir,
        prediction_spec=prediction_spec,
    )
    path = workspace_dir / EXP_GUIDE_FILENAME
    path.write_text(json.dumps(guide, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _guide_view(guide: dict[str, Any], *, include_samples: bool) -> dict[str, Any]:
    view = {
        "version": guide.get("version"),
        "source": guide.get("source"),
        "splits": guide.get("splits", {}),
        "target": guide.get("target", {}),
        "prediction": guide.get("prediction", {}),
        "metric": guide.get("metric", {}),
        "data_semantics": guide.get("data_semantics", {}),
        "visibility": guide.get("visibility", {}),
    }
    if include_samples:
        # Candidate prompts include the complete frozen public interface, not
        # only a metric name and a shortened output-column summary.
        view["evaluation_protocol"] = guide.get("evaluation_protocol", {})
        view["samples"] = guide.get("samples", {})
        view["split_profiles"] = guide.get("split_profiles", {})
    return view


def node_exp_guide_view(guide: dict[str, Any]) -> dict[str, Any]:
    """Return the agent-facing guide view rendered into node prompts."""
    return _guide_view(guide, include_samples=True)


def evaluator_exp_guide_view(guide: dict[str, Any]) -> dict[str, Any]:
    """Return the evaluator-only view used for compiler and prompt decisions."""
    return _guide_view(guide, include_samples=False)


# Backward-compatible name for existing callers; node view is the safe default.
compact_exp_guide_for_prompt = node_exp_guide_view


def load_exp_guide(workspace_dir: Path | str) -> dict[str, Any] | None:
    workspace_dir = Path(workspace_dir)
    data = _read_json_dict(workspace_dir / EXP_GUIDE_FILENAME)
    return data or None
