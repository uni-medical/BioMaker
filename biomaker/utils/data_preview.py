"""
Contains functions to manually generate a textual preview of some common file types (.csv, .json,..) for the agent.
"""

import json
from collections import defaultdict
from pathlib import Path

import humanize
import numpy as np
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype
from PIL import Image

from .response import truncate_text

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files
# these files are treated as image
image_files = {".png", ".jpeg", ".jpg", ".tiff"}
excluded_dirs = {
    "__pycache__",
    "private",
    "private_truth",
    "_private_truth",
    "_score_truth",
    "working",
    "node_workspaces",
    "biomaker_cache",
}
excluded_files = {"evaluator_prediction_spec.json"}
internal_generated_preview_files = {
    "input/biomaker_splits/dataset_report.json",
    "input/biomaker_splits/split_plan.json",
}
task_description_protocol_files = {
    "input/biomaker_splits/split_protocol.txt",
}


def _compact_json(value, max_chars=1600) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    return truncate_text(text, max_chars)


def _load_json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _image_observation(path_value: object) -> dict | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
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


def _segmentation_sample_facts(split_dir: Path) -> str:
    train_path = split_dir / "agent_train.csv"
    if not train_path.is_file():
        return ""
    try:
        train_df = pd.read_csv(train_path)
    except Exception:
        return ""
    if train_df.empty or "mask_path" not in train_df.columns:
        return ""

    image_column = next(
        (column for column in ["image_path", "path", "file_path"] if column in train_df.columns),
        None,
    )
    if image_column is None:
        return ""

    row = train_df.iloc[0]
    facts = {
        "source": "./input/biomaker_splits/agent_train.csv first row",
        "image_column": image_column,
        "mask_column": "mask_path",
        "image": _image_observation(row[image_column]),
        "mask": _image_observation(row["mask_path"]),
    }
    return (
        "Observed segmentation sample facts:\n"
        + "```json\n"
        + _compact_json(facts, max_chars=1800)
        + "\n```"
    )


def _csv_schema_line(path: Path, base_path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        rel = path.relative_to(base_path)
        return f"- `./{rel}`: could not read CSV schema: {exc}"

    columns = ", ".join(f"{name} ({dtype})" for name, dtype in df.dtypes.items())
    rel = path.relative_to(base_path)
    return f"- `./{rel}`: {len(df)} rows; columns: {columns}"


def _csv_target_summary(path: Path, base_path: Path) -> str | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    target_columns = [col for col in ["class", "label", "target", "mask"] if col in df]
    if not target_columns:
        return None

    rel = path.relative_to(base_path)
    parts = []
    for col in target_columns[:2]:
        counts = df[col].value_counts(dropna=False).sort_index()
        if len(counts) > 12:
            continue
        values = ", ".join(f"{key}: {value}" for key, value in counts.items())
        parts.append(f"{col} distribution: {values}")
    if not parts:
        return None
    return f"- `./{rel}`: " + "; ".join(parts)


def biomaker_essentials(base_path: Path) -> str:
    """Short deterministic run contract for generated solutions."""
    base_path = Path(base_path)
    split_dir = base_path / "input" / "biomaker_splits"
    if not split_dir.is_dir():
        return ""

    lines = ["BioMaker essentials:"]
    split_lines = []
    for filename in [
        "agent_train.csv",
        "score_val.csv",
        "agent_val.csv",
        "agent_test.csv",
    ]:
        line = _csv_schema_line(split_dir / filename, base_path)
        if line:
            split_lines.append(line)
    if split_lines:
        lines.append("Input splits:")
        lines.extend(split_lines)

    target_lines = []
    for filename in ["agent_train.csv", "agent_val.csv"]:
        line = _csv_target_summary(split_dir / filename, base_path)
        if line:
            target_lines.append(line)
    if target_lines:
        lines.append("Visible target summary:")
        lines.extend(target_lines)

    cache_manifest_path = base_path / "input" / "biomaker_cache" / "manifest.json"
    cache_files_path = base_path / "input" / "biomaker_cache" / "files.csv"
    cache_manifest = _load_json_file(cache_manifest_path)
    if cache_manifest is not None:
        lines.append("Input cache:")
        lines.append(
            "- `./input/biomaker_cache/files.csv`: deterministic agent-visible file cache "
            f"with {cache_manifest.get('files_total', 'unknown')} referenced files, "
            f"{cache_manifest.get('arrays_decoded', 'unknown')} decoded arrays, "
            f"{cache_manifest.get('errors', 'unknown')} errors."
        )
        if cache_files_path.is_file():
            lines.append(
                "  Use `cache_array_path` when present for faster loading; original files remain available."
            )

    segmentation_facts = _segmentation_sample_facts(split_dir)
    if segmentation_facts:
        lines.append(segmentation_facts)

    return "\n".join(lines)


def _private_preview_paths(base_path: Path) -> set[str]:
    split_plan_path = base_path / "input" / "biomaker_splits" / "split_plan.json"
    if not split_plan_path.is_file():
        split_plan_path = base_path / "biomaker_splits" / "split_plan.json"
    if not split_plan_path.is_file():
        return set()

    try:
        split_plan = json.loads(split_plan_path.read_text(encoding="utf-8"))
    except Exception:
        return set()

    private_paths = set()
    for private_eval in split_plan.get("private_evaluation", []):
        source = private_eval.get("source")
        if source:
            private_paths.add(f"input/{source}")
            private_paths.add(source)
    return private_paths


def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.
    """
    if f.suffix in plaintext_files:
        num_lines = sum(1 for _ in open(f))
        return num_lines, f"{num_lines} lines"
    else:
        s = f.stat().st_size
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
    files = [
        p
        for p in Path(path).iterdir()
        if not p.is_dir() and p.name not in excluded_files
    ]
    dirs = [
        p for p in Path(path).iterdir() if p.is_dir() and p.name not in excluded_dirs
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


def _walk(path: Path):
    """Recursively walk a directory (analogous to os.walk but for pathlib.Path)"""
    for p in sorted(Path(path).iterdir()):
        if p.name in excluded_dirs or p.name in excluded_files:
            continue
        if p.is_dir():
            yield from _walk(p)
            continue
        yield p


def preview_csv(p: Path, file_name: str, simple=True) -> str:
    """Generate a textual preview of a csv file

    Args:
        p (Path): the path to the csv file
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.

    Returns:
        str: the textual preview
    """
    df = pd.read_csv(p)

    out = []

    out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

    if simple:
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols)-sel_cols} more columns"
        out.append(res)
    else:
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    return "\n".join(out)


def preview_json(p: Path, file_name: str):
    """Generate a textual preview of a json file using a generated json schema"""
    builder = SchemaBuilder()
    with open(p) as f:
        builder.add_object(json.load(f))
    return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(
        indent=2
    )


def preview_internal_generated_json(p: Path, file_name: str) -> str:
    """Preview BioMaker-generated metadata as task facts instead of JSON schema."""
    data = _load_json_file(p)
    if not isinstance(data, dict):
        return (
            f"-> {file_name} is an internal BioMaker generated metadata file, "
            "but it could not be summarized."
        )
    if file_name.endswith("split_plan.json"):
        return _preview_split_plan(data, file_name)
    if file_name.endswith("dataset_report.json"):
        return _preview_dataset_report(data, file_name)
    return (
        f"-> {file_name} is an internal BioMaker generated metadata file. "
        "Its relevant facts are summarized elsewhere in Data Overview."
    )


def _preview_split_plan(plan: dict, file_name: str) -> str:
    lines = [f"-> {file_name} split plan summary:"]
    strategy = plan.get("strategy")
    if strategy:
        lines.append(f"- strategy: {strategy}")

    sources = plan.get("sources")
    if isinstance(sources, dict):
        for key in ["train", "val", "test"]:
            value = sources.get(key)
            if value is not None:
                lines.append(f"- source {key}: {value}")

    agent_visible = plan.get("agent_visible")
    if isinstance(agent_visible, dict) and agent_visible:
        visible = ", ".join(f"{key}={value}" for key, value in agent_visible.items())
        lines.append(f"- agent-visible split files: {visible}")

    score_validation = plan.get("score_validation")
    if isinstance(score_validation, dict):
        public = score_validation.get("public")
        labels_visible = score_validation.get("labels_visible_to_agent")
        if public:
            lines.append(
                f"- score validation: {public}; labels_visible_to_agent={labels_visible}"
            )
    metric_split = plan.get("metric_split")
    if metric_split:
        lines.append(f"- metric split used by evaluator: {metric_split}")

    private_sources = []
    for item in plan.get("private_evaluation", []) or []:
        if isinstance(item, dict):
            source = item.get("source")
            visible = item.get("visible_to_agent")
            if source:
                private_sources.append(f"{source} (visible_to_agent={visible})")
    if private_sources:
        lines.append("- private evaluation sources: " + ", ".join(private_sources))

    columns = plan.get("columns")
    if isinstance(columns, dict):
        column_parts = [
            f"{key}={value}"
            for key, value in columns.items()
            if value is not None
        ]
        if column_parts:
            lines.append("- detected columns: " + ", ".join(column_parts))

    diagnostics = plan.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics:
        lines.append("- split diagnostics: " + _compact_json(diagnostics, max_chars=500))
    warnings = plan.get("warnings")
    if warnings:
        lines.append("- warnings: " + "; ".join(str(w) for w in warnings[:3]))
    return "\n".join(lines)


def _preview_dataset_report(report: dict, file_name: str) -> str:
    lines = [f"-> {file_name} dataset report summary:"]
    tables = report.get("tables")
    if not isinstance(tables, dict) or not tables:
        lines.append("- no table summaries available")
        return "\n".join(lines)

    for rel, summary in sorted(tables.items()):
        if not isinstance(summary, dict):
            continue
        if summary.get("error"):
            lines.append(f"- {rel}: error={summary.get('error')}")
            continue
        parts = [
            f"rows={summary.get('rows')}",
            "columns=[" + ", ".join(map(str, summary.get("columns", []))) + "]",
        ]
        for key in ["id_column", "label_column", "path_column", "mask_column", "group_column"]:
            value = summary.get(key)
            if value:
                parts.append(f"{key}={value}")
        label_distribution = summary.get("label_distribution")
        if isinstance(label_distribution, dict) and label_distribution:
            parts.append(
                "label_distribution="
                + _compact_json(label_distribution, max_chars=300).replace("\n", " ")
            )
        lines.append(f"- {rel}: " + "; ".join(parts))
    return "\n".join(lines)


def preview_xlsx(p: Path, file_name: str, simple=True) -> str:
    """Generate a textual preview of an Excel file

    Args:
        p (Path): the path to the Excel file
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.

    Returns:
        str: the textual preview
    """
    # Read the Excel file
    df = pd.read_excel(p)

    out = []

    # Basic information about the file
    out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

    if simple:
        # Display a subset of columns
        cols = df.columns.tolist()
        sel_cols = 15
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols)-sel_cols} more columns"
        out.append(res)
    else:
        # Detailed information about each column
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(
                    f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}"
                )
            elif is_numeric_dtype(df[col]):
                out.append(
                    f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values"
                )
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )

    return "\n".join(out)


def generate(base_path, include_file_details=True, simple=False, max_chars=None):
    """
    Generate a directory preview with aggregated image size information.
    Summarizes images by type, count, and size ranges instead of individual entries.
    """
    tree = f"```\n{file_tree(base_path)}```"
    out = [tree]
    essentials = biomaker_essentials(Path(base_path))
    if essentials:
        out.append(essentials)
    private_paths = _private_preview_paths(Path(base_path))

    if include_file_details:
        image_data = defaultdict(list)  # {suffix: [(width, height)]}
        image_errors = 0

        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.name == "BioMaker.md":
                # 特殊处理 BioMaker.md 文件，读取全部内容
                with open(fn) as f:
                    content = f.read()
                    out.append(f"Memory of Dataset Details:\n\n```\n{content}\n```")
            elif file_name in private_paths:
                out.append(
                    f"-> {file_name} is reserved for private user evaluation. "
                    "Detailed label/artifact values are hidden from the agent; use input/biomaker_splits/agent_test.csv for test-time prediction inputs."
                )
            elif file_name in internal_generated_preview_files:
                out.append(preview_internal_generated_json(fn, file_name))
            elif file_name in task_description_protocol_files:
                out.append(
                    f"-> {file_name} is the BioMaker split protocol. "
                    "Its full text is provided in Task Description under Data split protocol."
                )
            elif fn.name.endswith("submission.csv"):
                out.append(preview_csv(fn, file_name, simple=False))
            elif fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix == ".xlsx":
                out.append(preview_xlsx(fn, file_name, simple=simple))
            elif fn.suffix in image_files:
                try:
                    with Image.open(fn) as img:
                        image_data[fn.suffix].append(img.size)
                except Exception:
                    image_errors += 1
            elif fn.suffix in plaintext_files:
                if get_file_len_size(fn)[0] < 30:
                    with open(fn) as f:
                        content = f.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        out.append(f"-> {file_name} has content:\n\n{content}")

        # Build image summary
        if image_data or image_errors:
            summary = ["Image Summary:"]

            for ext, sizes in image_data.items():
                count = len(sizes)
                ext_name = ext.lstrip(".").upper()
                areas = [w * h for w, h in sizes]
                min_size = sizes[areas.index(min(areas))]
                max_size = sizes[areas.index(max(areas))]

                size_range = (
                    f"{min_size[0]}x{min_size[1]} - " f"{max_size[0]}x{max_size[1]}"
                )
                summary.append(f"  • {count} {ext_name}: {size_range}")

            if image_errors:
                summary.append(f"  • {image_errors} files could not be read")

            out.append("\n".join(summary))

    result = "\n\n".join(out)

    if max_chars is not None and len(result) > 6000 and not simple:
        return generate(
            base_path,
            include_file_details=include_file_details,
            simple=True,
            max_chars=max_chars,
        )

    if max_chars is None:
        return result
    return truncate_text(result, max_chars=max_chars, marker="data preview trimmed")


if __name__ == "__main__":
    # prev = generate(Path("../med_dl_tasks/odir5k_2d_mlc"))
    # prev = generate(Path("../med_dl_tasks/mco_wsi_conch"))
    prev = generate(Path("../med_dl_tasks/hancock_wsi_uni"))
    print(prev)
