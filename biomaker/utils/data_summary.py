"""Build the factual data/evaluator summary shown to generated code."""

import json
from pathlib import Path

import humanize
import numpy as np
import pandas as pd
from PIL import Image

from .response import truncate_text

# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv", ".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# these files are treated as image
image_files = {".png", ".jpeg", ".jpg", ".tiff"}
excluded_dirs = {
    "__pycache__",
    "private",
    "private_gt",
    "working",
    "node_workspaces",
    "biomaker_splits",
    "biomaker_cache",
    "_split_builder",
}
excluded_files = {"evaluator_prediction_spec.json"}


def _load_json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _image_observation(path_value: object, base_dir: Path | None = None) -> dict | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if not path.is_absolute() and base_dir is not None:
        if path.parts and path.parts[0] == "input":
            path = base_dir.parent / path
        else:
            path = base_dir / path
    if not path.is_file():
        return {"path": path_value, "exists": False}
    try:
        with Image.open(path) as img:
            arr = np.asarray(img)
            info = {
                "path": path_value,
                "exists": True,
                "pil_mode": img.mode,
                "pil_size": list(img.size),
                "array_shape": list(arr.shape),
                "array_dtype": str(arr.dtype),
            }
            if arr.size:
                info["min"] = int(np.min(arr))
                info["max"] = int(np.max(arr))
                unique = np.unique(arr)
                info["unique_count"] = int(len(unique))
                if len(unique) <= 12:
                    info["unique_values"] = [int(v) for v in unique.tolist()]
                else:
                    info["unique_sample"] = [int(v) for v in unique[:12].tolist()]
            return info
    except Exception as exc:
        return {"path": path_value, "exists": True, "error": str(exc)}


def _read_json_dict(path: Path) -> dict | None:
    data = _load_json_file(path)
    return data if isinstance(data, dict) else None


def _csv_fact(path: Path, label: str, base_path: Path) -> tuple[str | None, pd.DataFrame | None]:
    if not path.is_file():
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        rel = path.relative_to(base_path)
        return f"- {label}: `./{rel}` could not be read: {exc}", None
    rel = path.relative_to(base_path)
    columns = ", ".join(map(str, df.columns.tolist()))
    return f"- {label}: `./{rel}`; rows={len(df)}; columns=[{columns}]", df


def _sample_row_facts(
    df: pd.DataFrame | None,
    *,
    label: str,
    input_dir: Path,
    max_columns: int = 8,
) -> list[str]:
    if df is None or df.empty:
        return []
    row = df.iloc[0]
    lines = []
    values = []
    for column in df.columns[:max_columns]:
        value = row[column]
        text = str(value)
        if len(text) > 80:
            text = text[:77] + "..."
        values.append(f"{column}={text}")
    lines.append(f"- {label} first row: " + "; ".join(values))

    path_like = [
        column
        for column in df.columns
        if "path" in str(column).lower() or "file" in str(column).lower()
    ][:4]
    for column in path_like:
        obs = _image_observation(row[column], input_dir)
        if obs is None:
            continue
        status = "exists" if obs.get("exists") else "missing"
        shape = obs.get("array_shape") or obs.get("pil_size")
        shape_text = f"; shape={shape}" if shape else ""
        mode = obs.get("pil_mode")
        mode_text = f"; mode={mode}" if mode else ""
        lines.append(
            f"- {label} sample `{column}`: {row[column]} -> {status}{shape_text}{mode_text}"
        )
    return lines


def _prediction_spec_summary(base_path: Path) -> list[str]:
    spec = _read_json_dict(base_path / "evaluator_prediction_spec.json")
    if not spec:
        return ["- `./evaluator.py` is available; call `evaluator.prediction_spec()` for the exact artifact contract."]
    lines = []
    validation_artifact = spec.get("validation_artifact")
    test_submission = spec.get("test_submission")
    if validation_artifact:
        lines.append(f"- validation predictions: `{validation_artifact}`")
    if test_submission:
        lines.append(f"- test submission: `{test_submission}`")
    key_columns = spec.get("key_columns")
    if key_columns is not None:
        lines.append(f"- key columns: {key_columns}")
    prediction_columns = spec.get("prediction_columns") or spec.get("file_reference_columns")
    if prediction_columns:
        lines.append(f"- prediction columns: {prediction_columns}")
    primary_metric = spec.get("primary_metric")
    if primary_metric:
        lines.append(f"- validation metric used for node selection: {primary_metric}")
    prediction_type = spec.get("prediction_type") or spec.get("task_type")
    if prediction_type:
        lines.append(f"- prediction type: {prediction_type}")
    return lines or ["- `./evaluator.py` is available; call `evaluator.prediction_spec()` for the exact artifact contract."]


def data_and_evaluation_summary(base_path: Path | str, max_chars: int | None = None) -> str:
    """Return the single factual data/evaluator summary shown to generated nodes."""
    base_path = Path(base_path)
    input_dir = base_path / "input"
    split_dir = input_dir / "biomaker_splits"
    lines = ["Data and Evaluation Summary"]
    lines.append("")
    lines.append("Generated code should read only agent-visible inputs under `./input` and write outputs under `./working`.")

    tree_text = file_tree(input_dir) if input_dir.is_dir() else ""
    if tree_text:
        lines.append("")
        lines.append("Input file structure:")
        lines.append("```")
        lines.append(tree_text)
        lines.append("```")

    split_frames: dict[str, pd.DataFrame | None] = {}
    if split_dir.is_dir():
        lines.append("")
        lines.append("Agent-visible split files:")
        for label, filename in [
            ("training", "agent_train.csv"),
            ("score validation", "score_val.csv"),
            ("test prediction", "agent_test.csv"),
        ]:
            fact, df = _csv_fact(split_dir / filename, label, base_path)
            split_frames[label] = df
            if fact:
                lines.append(fact)

    split_plan = _read_json_dict(split_dir / "split_plan.json")
    private_gt = split_plan.get("private_gt") if isinstance(split_plan, dict) else None
    if isinstance(private_gt, dict):
        hidden = []
        for split_name in ["score_val", "test"]:
            item = private_gt.get(split_name)
            if isinstance(item, dict):
                answers = item.get("answer_columns") or []
                if answers:
                    hidden.append(f"{split_name}: {answers}")
        if hidden:
            lines.append("")
            lines.append("Hidden answers:")
            lines.append("- " + "; ".join(hidden))
            lines.append("- Hidden answers are outside the agent workspace and must not be used by generated code.")

    lines.append("")
    lines.append("Required evaluator artifacts:")
    lines.extend(_prediction_spec_summary(base_path))
    lines.append("- Use `./evaluator.py` as the source of truth for exact output format.")
    lines.append("- BioMaker scores validation predictions after the script exits; generated code should not call `evaluator.score_val()`.")

    sample_lines = []
    for label, df in split_frames.items():
        sample_lines.extend(_sample_row_facts(df, label=label, input_dir=input_dir))
    if sample_lines:
        lines.append("")
        lines.append("Sample data facts:")
        lines.extend(sample_lines)

    cache_manifest = _read_json_dict(input_dir / "biomaker_cache" / "manifest.json")
    if cache_manifest:
        lines.append("")
        lines.append("Input cache:")
        lines.append(
            "- `./input/biomaker_cache/files.csv`: "
            f"files={cache_manifest.get('files_total', 'unknown')}, "
            f"decoded_arrays={cache_manifest.get('arrays_decoded', 'unknown')}, "
            f"errors={cache_manifest.get('errors', 'unknown')}."
        )
        lines.append("- Use `cache_array_path` when present for faster loading; original files remain available.")

    text = "\n".join(lines)
    if max_chars is not None:
        return truncate_text(text, max_chars=max_chars, marker="data and evaluation summary trimmed")
    return text


def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.
    """
    if not f.exists():
        return 0, "missing"
    if f.suffix in plaintext_files:
        try:
            num_lines = sum(1 for _ in open(f))
        except OSError:
            return 0, "unreadable"
        return num_lines, f"{num_lines} lines"
    else:
        try:
            s = f.stat().st_size
        except OSError:
            return 0, "unreadable"
        img_size_info = ""
        if f.suffix in image_files:
            try:
                with Image.open(f) as img:
                    width, height = img.size
                    img_size_info = f"{width}x{height} pixels"
            except:
                img_size_info = ""
        return s, str(humanize.naturalsize(s)) + f", {img_size_info}"


def file_tree(path: Path, depth=0) -> str:
    """Generate a tree structure of files in a directory"""
    result = []
    try:
        children = list(Path(path).iterdir())
    except OSError:
        return f"{' '*depth*4}<unreadable>"
    files = [
        p
        for p in children
        if not p.is_dir() and p.name not in excluded_files and p.name not in excluded_dirs
    ]
    dirs = [
        p for p in children if p.is_dir() and p.name not in excluded_dirs
    ]
    max_n = 4 if len(files) > 30 else 8
    for p in sorted(files)[:max_n]:
        if p.name == "BioMaker.md":
            continue
        result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' '*depth*4}... and {len(files)-max_n} other files")

    for p in sorted(dirs):
        result.append(f"{' '*depth*4}{p.name}/")
        result.append(file_tree(p, depth + 1))

    return "\n".join(result)
