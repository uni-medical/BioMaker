"""Render the experiment guide as the data/evaluator summary shown to nodes."""

import json
from pathlib import Path
from typing import Any

import humanize
from PIL import Image

from .exp_guide import build_exp_guide, load_exp_guide
from .response import truncate_text

plaintext_files = {
    ".txt",
    ".csv",
    ".json",
    ".tsv",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".md",
    ".html",
    ".xml",
    ".log",
    ".rst",
}
image_files = {".png", ".jpeg", ".jpg", ".tiff"}
excluded_dirs = {
    "__pycache__",
    "private",
    "private_gt",
    "working",
    "node_workspaces",
    "biomaker_splits",
    "_split_builder",
}
excluded_files = {"evaluator_prediction_spec.json", "exp_guide.json"}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_text(value: Any) -> str:
    return str(value if isinstance(value, list) else [])


def _append_input_tree(lines: list[str], input_dir: Path) -> None:
    tree_text = file_tree(input_dir) if input_dir.is_dir() else ""
    if not tree_text:
        return
    lines.extend(["", "Input file structure:", "```", tree_text, "```"])


def _append_splits(lines: list[str], guide: dict[str, Any]) -> None:
    splits = _as_dict(guide.get("splits"))
    if not splits:
        return
    lines.extend(["", "Agent-visible split files:"])
    split_columns: dict[str, set[str]] = {}
    for split_name in ["train", "score_val", "test"]:
        split = _as_dict(splits.get(split_name))
        if not split:
            continue
        column_values = [str(column) for column in split.get("columns") or []]
        split_columns[split_name] = set(column_values)
        columns = ", ".join(column_values)
        rows = split.get("rows", "unknown")
        path = split.get("path", "unknown")
        label = split.get("role") or split_name
        read_error = split.get("read_error")
        if read_error:
            lines.append(f"- {label}: `{path}` could not be read: {read_error}")
        else:
            lines.append(f"- {label}: `{path}`; rows={rows}; columns=[{columns}]")
    train_columns = split_columns.get("train", set())
    for split_name in ["score_val", "test"]:
        columns = split_columns.get(split_name)
        if not columns:
            continue
        missing_from_split = sorted(train_columns - columns)
        path_like = [
            column
            for column in missing_from_split
            if "path" in column.lower() or "file" in column.lower()
        ]
        if path_like:
            label = _as_dict(splits.get(split_name)).get("role") or split_name
            lines.append(
                f"- {label} does not include training-only file column(s): "
                f"{path_like}; generated code must handle this split separately."
            )


def _append_split_profiles(lines: list[str], guide: dict[str, Any]) -> None:
    profiles = _as_dict(guide.get("split_profiles"))
    train_profile = _as_dict(profiles.get("train"))
    visible_labels = _as_dict(train_profile.get("visible_labels"))
    if not visible_labels:
        return
    lines.extend(["", "Visible training-label profile:"])
    for column, profile in visible_labels.items():
        profile = _as_dict(profile)
        counts = _as_dict(profile.get("counts"))
        counts_text = ", ".join(f"{key}: {value}" for key, value in counts.items())
        lines.append(
            f"- `{column}`: rows={profile.get('rows', 'unknown')}; "
            f"unique={profile.get('unique_count', 'unknown')}; counts={{ {counts_text} }}"
        )


def _append_hidden_answers(lines: list[str], guide: dict[str, Any]) -> None:
    hidden_answers = _as_dict(_as_dict(guide.get("target")).get("hidden_answers"))
    hidden = [split_name for split_name, columns in hidden_answers.items() if columns]
    if not hidden:
        return
    lines.extend(
        [
            "",
            "Hidden answers:",
            "- score/private answer values and answer column names are withheld from generated code.",
            f"- withheld splits: {', '.join(hidden)}",
            "- Hidden answers are outside the agent workspace and must not be used by generated code.",
        ]
    )


def _append_artifacts(lines: list[str], guide: dict[str, Any]) -> None:
    prediction = _as_dict(guide.get("prediction"))
    metric = _as_dict(guide.get("metric"))
    contract = {
        "validation_artifact": prediction.get("validation_artifact"),
        "test_submission": prediction.get("test_submission"),
        "key_columns": prediction.get("key_columns") or [],
        "prediction_columns": prediction.get("prediction_columns") or [],
        "prediction_type": prediction.get("prediction_type") or "columns",
        "primary_metric": metric.get("primary_metric"),
        "maximize": metric.get("maximize"),
    }
    for optional_key in ["class_labels", "positive_class", "score_split"]:
        value = prediction.get(optional_key) or metric.get(optional_key)
        if value not in (None, [], {}):
            contract[optional_key] = value
    if metric.get("secondary_metrics"):
        contract["secondary_metrics"] = metric.get("secondary_metrics")
    lines.extend(["", "Evaluator contract:"])
    lines.append("```json")
    lines.append(json.dumps(contract, ensure_ascii=False, sort_keys=True))
    lines.append("```")
    output_encoding = _as_dict(guide.get("output_encoding"))
    if output_encoding:
        lines.extend([
            "", "Public prediction encoding:", "```json",
            json.dumps(output_encoding, ensure_ascii=False, sort_keys=True), "```",
            "- Follow the declared field order, coordinate units and confidence "
            "representation; sample key columns are separate from predictions. "
            "Frozen output details take precedence over prepared declarations. "
            "Consult `evaluator.prediction_spec()` and its decoder if needed; "
            "do not infer a positional format from the prediction column name.",
        ])
    lines.extend(["", "Required evaluator artifacts:"])
    if prediction.get("validation_artifact"):
        lines.append(f"- validation predictions: `{prediction['validation_artifact']}`")
    if prediction.get("test_submission"):
        lines.append(f"- test submission: `{prediction['test_submission']}`")
    if "key_columns" in prediction:
        lines.append(f"- key columns: {_list_text(prediction.get('key_columns'))}")
    prediction_columns = prediction.get("prediction_columns") or prediction.get(
        "file_reference_columns"
    )
    if prediction_columns:
        lines.append(f"- prediction columns: {prediction_columns}")
    if metric.get("primary_metric"):
        lines.append(
            f"- validation metric used for node selection: {metric['primary_metric']}"
        )
    if (
        prediction.get("prediction_type")
        and prediction.get("prediction_type") != "columns"
    ):
        lines.append(f"- prediction type: {prediction['prediction_type']}")
    lines.append(
        "- Use `./evaluator.py` as the source of truth for exact output format."
    )
    lines.append(
        "- BioMaker scores validation predictions after the script exits; "
        "generated code should not call `evaluator.score_val()`."
    )


def _append_samples(lines: list[str], guide: dict[str, Any]) -> None:
    samples = _as_dict(guide.get("samples"))
    splits = _as_dict(guide.get("splits"))
    if not samples:
        return
    lines.extend(["", "Sample data facts:"])
    for split_name in ["train", "score_val", "test"]:
        sample = _as_dict(samples.get(split_name))
        if not sample:
            continue
        label = _as_dict(splits.get(split_name)).get("role") or split_name
        first_row = _as_dict(sample.get("first_row"))
        if first_row:
            values = []
            for column, value in first_row.items():
                text = str(value)
                if len(text) > 80:
                    text = text[:77] + "..."
                values.append(f"{column}={text}")
            lines.append(f"- {label} first row: " + "; ".join(values))

        for column, observation in _as_dict(sample.get("path_observations")).items():
            observation = _as_dict(observation)
            status = "exists" if observation.get("exists") else "missing"
            shape = observation.get("array_shape") or observation.get("pil_size")
            shape_text = f"; shape={shape}" if shape else ""
            mode = observation.get("mode")
            mode_text = f"; mode={mode}" if mode else ""
            kind = observation.get("kind")
            kind_text = f"; kind={kind}" if kind else ""
            reader = observation.get("reader")
            reader_text = f"; reader={reader}" if reader else ""
            pixel_type = observation.get("pixel_type")
            pixel_text = f"; pixel_type={pixel_type}" if pixel_type else ""
            value = observation.get("value", "")
            lines.append(
                f"- {label} sample `{column}`: {value} -> "
                f"{status}{shape_text}{mode_text}{kind_text}{reader_text}{pixel_text}"
            )



def _render_exp_guide_summary(guide: dict[str, Any], base_path: Path) -> str:
    lines = [
        "Data and Evaluation Summary",
        "",
        "Generated code should read only agent-visible inputs under `./input` "
        "and write outputs under `./working`.",
        "- This summary is rendered from BioMaker's internal experiment guide.",
    ]
    _append_artifacts(lines, guide)
    _append_splits(lines, guide)
    _append_split_profiles(lines, guide)
    _append_hidden_answers(lines, guide)
    _append_samples(lines, guide)
    _append_input_tree(lines, base_path / "input")
    return "\n".join(lines)


def data_and_evaluation_summary(
    base_path: Path | str, max_chars: int | None = None
) -> str:
    """Return the single factual data/evaluator summary shown to generated nodes."""
    base_path = Path(base_path)
    guide = load_exp_guide(base_path) or build_exp_guide(base_path)
    text = _render_exp_guide_summary(guide, base_path)
    if max_chars is not None:
        return truncate_text(
            text,
            max_chars=max_chars,
            marker="data and evaluation summary trimmed",
        )
    return text


def get_file_len_size(f: Path) -> tuple[int, str]:
    """Return line counts for text files and human-readable sizes for other files."""
    if not f.exists():
        return 0, "missing"
    if f.suffix in plaintext_files:
        try:
            with f.open(encoding="utf-8", errors="replace") as handle:
                num_lines = sum(1 for _ in handle)
        except OSError:
            return 0, "unreadable"
        return num_lines, f"{num_lines} lines"

    try:
        size = f.stat().st_size
    except OSError:
        return 0, "unreadable"
    image_size_info = ""
    if f.suffix in image_files:
        try:
            with Image.open(f) as img:
                width, height = img.size
                image_size_info = f"{width}x{height} pixels"
        except Exception:
            image_size_info = ""
    return size, str(humanize.naturalsize(size)) + f", {image_size_info}"


def file_tree(path: Path, depth=0, max_depth=3) -> str:
    """Generate a compact tree structure of files in a directory."""
    result = []
    if depth > max_depth:
        return f"{' '*depth*4}..."
    try:
        children = list(Path(path).iterdir())
    except OSError:
        return f"{' '*depth*4}<unreadable>"
    files = [
        p
        for p in children
        if not p.is_dir()
        and p.name not in excluded_files
        and p.name not in excluded_dirs
    ]
    dirs = [p for p in children if p.is_dir() and p.name not in excluded_dirs]
    max_n = 4 if len(files) > 30 else 8
    for p in sorted(files)[:max_n]:
        if p.name == "BioMaker.md":
            continue
        result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' '*depth*4}... and {len(files)-max_n} other files")

    for p in sorted(dirs)[:20]:
        result.append(f"{' '*depth*4}{p.name}/")
        result.append(file_tree(p, depth + 1, max_depth=max_depth))
    if len(dirs) > 20:
        result.append(f"{' '*depth*4}... and {len(dirs)-20} other directories")

    return "\n".join(result)
