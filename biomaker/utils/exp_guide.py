"""Run-level experiment guide used as the structured pre-node fact source."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .experiment_contract import load_experiment_contract

EXP_GUIDE_VERSION = 1
EXP_GUIDE_FILENAME = "exp_guide.json"
MEDICAL_IMAGE_SUFFIXES = {".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"}
MAX_CACHE_PROFILE_ROWS_PER_GROUP = 200


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


def _sample_profile(df: pd.DataFrame | None, input_dir: Path) -> dict[str, Any] | None:
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    values: dict[str, Any] = {}
    for column in df.columns[:8]:
        value = row[column]
        if isinstance(value, np.generic):
            value = value.item()
        values[str(column)] = (
            value if isinstance(value, (str, int, float, bool)) else str(value)
        )

    path_observations = {}
    for column in df.columns:
        column_name = str(column)
        if "path" not in column_name.lower() and "file" not in column_name.lower():
            continue
        observation = _path_observation(row[column], input_dir)
        if observation is not None:
            path_observations[column_name] = observation
        if len(path_observations) >= 4:
            break
    return {
        "first_row": values,
        "path_observations": path_observations,
    }


def _cache_guide(input_dir: Path) -> dict[str, Any] | None:
    cache_dir = input_dir / "biomaker_cache"
    manifest = _read_json_dict(cache_dir / "manifest.json")
    if not manifest:
        return None
    result: dict[str, Any] = {
        "path": "./input/biomaker_cache/files.csv",
        "files_total": manifest.get("files_total"),
        "arrays_decoded": manifest.get("arrays_decoded"),
        "errors": manifest.get("errors"),
        "decode_arrays": manifest.get("decode_arrays"),
    }
    files_csv = cache_dir / "files.csv"
    if files_csv.is_file():
        try:
            index_df = pd.read_csv(files_csv, nrows=1)
            columns = [str(column) for column in index_df.columns.tolist()]
            result["index_columns"] = columns
            if "original_path" in columns and "cache_array_path" in columns:
                result["path_mapping"] = {
                    "source_path_column": "original_path",
                    "cache_array_column": "cache_array_path",
                    "usable_status": "decoded" if "status" in columns else None,
                }
        except Exception as exc:
            result["index_read_error"] = str(exc)
        profiles = _cache_array_profiles(files_csv, input_dir)
        if profiles:
            result["array_profiles"] = profiles
    return result


def _resolve_cache_path(value: Any, input_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    path = Path(text)
    if path.is_absolute():
        return path
    if text.startswith("./input/"):
        return input_dir.parent / text[2:]
    if path.parts and path.parts[0] == "input":
        return input_dir.parent / path
    return input_dir / path


def _shape_summary(shapes: list[list[int]]) -> dict[str, Any]:
    ranks = sorted({len(shape) for shape in shapes})
    max_rank = max(ranks) if ranks else 0
    dim_ranges = []
    for dim_index in range(max_rank):
        values = [shape[dim_index] for shape in shapes if len(shape) > dim_index]
        if not values:
            continue
        dim_ranges.append(
            {
                "min": int(min(values)),
                "median": int(statistics.median(values)),
                "max": int(max(values)),
            }
        )
    examples: list[list[int]] = []
    for shape in shapes:
        if shape not in examples:
            examples.append(shape)
        if len(examples) >= 3:
            break
    return {"ranks": ranks, "dim_ranges": dim_ranges, "examples": examples}


def _cache_array_profiles(files_csv: Path, input_dir: Path) -> dict[str, Any] | None:
    try:
        rows = pd.read_csv(files_csv)
    except Exception:
        return None
    required = {"split", "column", "cache_metadata_path", "status"}
    if not required.issubset(set(map(str, rows.columns))):
        return None

    profiles: dict[str, Any] = {}
    for (split_name, column), group in rows.groupby(["split", "column"], dropna=False):
        shapes: list[list[int]] = []
        dtypes: dict[str, int] = {}
        readers: dict[str, int] = {}
        spacings: list[list[float]] = []
        size_mb: list[float] = []
        decoded_group = group[group["status"].astype(str) == "decoded"]
        decoded_total = int(len(decoded_group))
        profiled_rows = 0
        for _, row in decoded_group.head(MAX_CACHE_PROFILE_ROWS_PER_GROUP).iterrows():
            metadata_path = _resolve_cache_path(row.get("cache_metadata_path"), input_dir)
            if metadata_path is None or not metadata_path.is_file():
                continue
            metadata = _read_json_dict(metadata_path)
            if not metadata:
                continue
            profiled_rows += 1
            shape = metadata.get("array_shape")
            if isinstance(shape, list) and all(
                isinstance(item, (int, float)) for item in shape
            ):
                shapes.append([int(item) for item in shape])
            dtype = metadata.get("array_dtype")
            if dtype:
                dtypes[str(dtype)] = dtypes.get(str(dtype), 0) + 1
            reader = metadata.get("reader")
            if reader:
                readers[str(reader)] = readers.get(str(reader), 0) + 1
            spacing = metadata.get("spacing")
            if isinstance(spacing, list) and spacing not in spacings:
                spacings.append(spacing)
            try:
                size = float(metadata.get("size_bytes", 0)) / (1024 * 1024)
            except Exception:
                size = 0.0
            if size > 0:
                size_mb.append(size)
        if profiled_rows == 0:
            continue
        entry: dict[str, Any] = {
            "decoded_rows": decoded_total,
            "profiled_rows": profiled_rows,
            "dtypes": dtypes,
        }
        if shapes:
            entry["shape_summary"] = _shape_summary(shapes)
        if readers:
            entry["readers"] = readers
        if spacings:
            entry["spacing_examples"] = spacings[:3]
        if size_mb:
            entry["size_mb"] = {
                "min": round(float(min(size_mb)), 1),
                "median": round(float(statistics.median(size_mb)), 1),
                "max": round(float(max(size_mb)), 1),
            }
        profiles[f"{split_name}.{column}"] = entry
    return profiles or None


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
        sample = _sample_profile(df, input_dir)
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
        "visibility": {
            "agent_input_root": "./input",
            "agent_output_root": "./working",
            "hidden_answers_outside_agent_workspace": True,
        },
        "samples": samples,
        "split_profiles": split_profiles,
        "cache": _cache_guide(input_dir),
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
        "visibility": guide.get("visibility", {}),
        "cache": guide.get("cache"),
    }
    if include_samples:
        view["samples"] = guide.get("samples", {})
        view["split_profiles"] = guide.get("split_profiles", {})
    return view


def node_exp_guide_view(guide: dict[str, Any]) -> dict[str, Any]:
    """Return the agent-facing guide view rendered into node prompts."""
    return _guide_view(guide, include_samples=True)


def evaluator_exp_guide_view(guide: dict[str, Any]) -> dict[str, Any]:
    """Return the evaluator-only view used for compiler/prompt/cache decisions."""
    return _guide_view(guide, include_samples=False)


# Backward-compatible name for existing callers; node view is the safe default.
compact_exp_guide_for_prompt = node_exp_guide_view


def load_exp_guide(workspace_dir: Path | str) -> dict[str, Any] | None:
    workspace_dir = Path(workspace_dir)
    data = _read_json_dict(workspace_dir / EXP_GUIDE_FILENAME)
    return data or None
