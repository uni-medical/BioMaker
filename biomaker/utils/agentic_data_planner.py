"""Agentic data-split builder used by the BioMaker run path.

It follows a small loop:

1. Observe the dataset directory as facts.
2. Ask an LLM to write a dataset-specific builder script.
3. Run that script and verify only universal output invariants.

The verifier does not decide whether a task is classification, segmentation,
detection, report generation, or something else. That semantic decision belongs
to the builder script generated from the observed data and task description.
"""

from __future__ import annotations

import json
import hashlib
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .. import backend
from .split_manifest import (
    legacy_manifest,
    load_split_manifest,
    validate_split_manifest,
    write_split_manifest,
)
from .response import extract_code
from .metric_registry import METRIC_CONTRACT_REQUIREMENTS

TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
GENERATED_DIR_NAMES = {
    "biomaker_splits",
    "private_gt",
    "_split_builder",
}
SPLIT_BUILDER_CACHE_VERSION = 5


@dataclass(frozen=True)
class PlannerResult:
    plan: dict[str, Any]
    output_dir: Path
    split_dir: Path
    private_gt_dir: Path
    inspection_history: list[dict[str, Any]]
    verifier: dict[str, Any]
    comparison: dict[str, Any] | None = None


def _compound_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _inside_generated_dir(path: Path) -> bool:
    return any(part in GENERATED_DIR_NAMES for part in path.parts)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = _compound_suffix(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table suffix: {suffix}")


def _is_private_example_column(column: str, series: pd.Series) -> bool:
    """Hide small scalar examples in the prompt; this is not task inference."""
    lowered = column.lower()
    if any(
        token in lowered
        for token in (
            "label",
            "class",
            "target",
            "answer",
            "truth",
            "gt",
            "diagnosis",
            "grade",
        )
    ):
        return True
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    unique_count = int(non_null.nunique(dropna=True))
    return unique_count <= max(20, min(100, math.ceil(len(non_null) * 0.05)))


def _path_candidates(value: Any, input_dir: Path, base_dir: Path) -> list[Path]:
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    raw = Path(text)
    candidates = [raw] if raw.is_absolute() else [input_dir / raw, base_dir / raw]
    if raw.parts and raw.parts[0] == "input":
        candidates.append(input_dir.parent / raw)
        if len(raw.parts) > 1:
            candidates.append(input_dir / Path(*raw.parts[1:]))
    return candidates


def _resolve_path(value: Any, input_dir: Path, base_dir: Path) -> Path | None:
    for candidate in _path_candidates(value, input_dir, base_dir):
        if candidate.exists():
            return candidate
    return None


def _table_facts(path: Path, input_dir: Path) -> dict[str, Any]:
    df = _read_table(path)
    columns = []
    for column in df.columns:
        series = df[column]
        item: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if series.dtype == object:
            sample = series.dropna().head(80)
            resolved = []
            for value in sample:
                resolved_path = _resolve_path(value, input_dir, path.parent)
                if resolved_path is not None and len(resolved) < 5:
                    resolved.append(
                        {
                            "value": str(value)[:160],
                            "resolved": _relative(resolved_path, input_dir),
                            "suffix": _compound_suffix(resolved_path),
                        }
                    )
            if resolved:
                item["path_examples"] = resolved
        if len(series.dropna()) and not _is_private_example_column(str(column), series):
            item["examples"] = [str(v)[:120] for v in series.dropna().head(3).tolist()]
        elif len(series.dropna()):
            item["examples_hidden"] = (
                "small scalar examples hidden to avoid leaking possible answers"
            )
        columns.append(item)
    return {
        "path": _relative(path, input_dir),
        "rows": int(len(df)),
        "columns": columns,
    }


def _directory_facts(input_dir: Path, max_dirs: int = 160) -> list[dict[str, Any]]:
    facts = []
    directories = [input_dir, *[p for p in sorted(input_dir.rglob("*")) if p.is_dir()]]
    for directory in directories[:max_dirs]:
        if _inside_generated_dir(directory):
            continue
        direct_dirs = [
            p
            for p in sorted(directory.iterdir())
            if p.is_dir() and p.name not in GENERATED_DIR_NAMES
        ]
        direct_files = [p for p in sorted(directory.iterdir()) if p.is_file()]
        recursive_files = [
            p
            for p in sorted(directory.rglob("*"))
            if p.is_file() and not _inside_generated_dir(p)
        ]
        suffix_counts: dict[str, int] = {}
        for path in recursive_files[:2000]:
            suffix = _compound_suffix(path) or "<none>"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        facts.append(
            {
                "path": _relative(directory, input_dir),
                "direct_dirs": [p.name for p in direct_dirs[:40]],
                "direct_file_count": len(direct_files),
                "recursive_file_count": len(recursive_files),
                "suffix_counts_sample": dict(sorted(suffix_counts.items())),
                "sample_files": [_relative(p, input_dir) for p in recursive_files[:12]],
            }
        )
    return facts


def _image_group_facts(
    files: list[Path], input_dir: Path, max_files: int = 3000
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    errors = 0
    for path in files[:max_files]:
        if _compound_suffix(path) not in IMAGE_SUFFIXES:
            continue
        group_key = _relative(path.parent, input_dir)
        group = groups.setdefault(group_key, {"count": 0, "sizes": [], "suffixes": {}})
        suffix = _compound_suffix(path)
        group["count"] += 1
        group["suffixes"][suffix] = group["suffixes"].get(suffix, 0) + 1
        if len(group["sizes"]) < 8:
            try:
                with Image.open(path) as img:
                    group["sizes"].append(
                        {"file": path.name, "size": list(img.size), "mode": img.mode}
                    )
            except Exception:
                errors += 1
    return {"groups": groups, "errors": errors}


def build_dataset_facts(
    input_dir: Path, *, max_files: int = 5000, max_tables: int = 30
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    files = [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and not _inside_generated_dir(p)
    ]
    suffix_counts: dict[str, int] = {}
    sample_files = []
    for path in files[:max_files]:
        suffix = _compound_suffix(path) or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        if len(sample_files) < 160:
            sample_files.append(
                {
                    "path": _relative(path, input_dir),
                    "suffix": suffix,
                    "size_bytes": int(path.stat().st_size),
                }
            )

    tables = []
    for path in files:
        if _compound_suffix(path) not in TABLE_SUFFIXES:
            continue
        try:
            tables.append(_table_facts(path, input_dir))
        except Exception as exc:
            tables.append({"path": _relative(path, input_dir), "error": str(exc)})
        if len(tables) >= max_tables:
            break

    return {
        "input_dir_name": input_dir.name,
        "file_count": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "sample_files": sample_files,
        "directories": _directory_facts(input_dir),
        "tables": tables,
        "image_groups": _image_group_facts(files, input_dir),
    }


def split_builder_cache_key(
    *,
    input_dir: Path,
    task_text: str,
    model: str,
    temperature: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    facts = build_dataset_facts(input_dir)
    payload = {
        "cache_version": SPLIT_BUILDER_CACHE_VERSION,
        "model": model,
        "temperature": temperature,
        "task_text": task_text,
        "facts": facts,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24], facts


def _builder_system_prompt() -> str:
    return (
        "You are a data-preparation coding agent. You receive an unfamiliar dataset, "
        "a task description, and observed filesystem/table facts. Write the simplest "
        "Python builder that converts the dataset into BioMaker's public split files "
        "and evaluator-only hidden-answer files. Infer semantics from evidence; do not rely on "
        "a fixed task type or one naming convention."
    )


def _compact_error_list(errors: Any, *, limit: int = 40) -> dict[str, Any]:
    if not isinstance(errors, list):
        return {"items": []}
    items = [str(error) for error in errors]
    duplicate_sample_id_count = sum(
        1 for error in items if error.startswith("sample_universe has duplicate sample_id:")
    )
    compact = {
        "count": len(items),
        "items": items[:limit],
    }
    if len(items) > limit:
        compact["omitted_count"] = len(items) - limit
    if duplicate_sample_id_count:
        compact["duplicate_sample_id_errors"] = duplicate_sample_id_count
    return compact


def _compact_split_manifest_for_prompt(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        "manifest_version",
        "sample_unit",
        "key_columns",
        "task_heads",
        "visibility",
        "prediction_contract",
        "metric_contract",
        "hidden_answers",
    ):
        if key in manifest:
            compact[key] = manifest[key]
    sample_universe = manifest.get("sample_universe")
    if isinstance(sample_universe, list):
        compact["sample_universe_summary"] = {
            "count": len(sample_universe),
            "first_items": sample_universe[:3],
            "last_items": sample_universe[-3:] if len(sample_universe) > 3 else [],
        }
    elif sample_universe is not None:
        compact["sample_universe_summary"] = str(type(sample_universe).__name__)
    return compact


def _manifest_summary_for_verifier(manifest: Any) -> dict[str, Any]:
    summary = _compact_split_manifest_for_prompt(manifest)
    if isinstance(manifest, dict):
        summary["present_fields"] = sorted(map(str, manifest.keys()))
    return summary


def _compact_builder_failure_for_prompt(failure: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "ok": failure.get("ok"),
        "builder_returncode": failure.get("builder_returncode"),
        "errors": _compact_error_list(failure.get("errors")),
        "warnings": _compact_error_list(failure.get("warnings"), limit=20),
    }
    for key in ("row_counts", "hidden_answers", "private_gt"):
        if key in failure:
            compact[key] = failure[key]
    if "split_manifest" in failure:
        compact["split_manifest"] = _compact_split_manifest_for_prompt(
            failure.get("split_manifest")
        )
    return compact


def _builder_user_prompt(
    task_text: str,
    facts: dict[str, Any],
    previous_script: str | None = None,
    previous_failure: dict[str, Any] | None = None,
    previous_stdout: str | None = None,
) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "Task Description": task_text,
        "Observed Data Facts": json.dumps(facts, indent=2, ensure_ascii=False),
        "Required Output Files": [
            "biomaker_splits/agent_train.csv: public training rows; may include training answers.",
            "biomaker_splits/score_val.csv: public validation-input rows; must not include validation answers.",
            "biomaker_splits/agent_test.csv: public test-input rows when a test source exists; must not include private test answers.",
            "private_gt/score_val.csv: validation rows with the hidden answers retained.",
            "private_gt/test.csv: private test rows with hidden answers retained when private answers exist.",
            'private_gt/private_gt.json: hidden-answer index with keys score_val and test. Each entry must contain answer_columns, for example {"score_val":{"answer_columns":["label"]},"test":{"answer_columns":["label"]}}.',
            "split_manifest.json: structured data contract for the prepared task.",
            "public/: optional generated agent-visible files only when a derived/converted/anonymized public artifact is truly needed; CSV paths may reference this directory.",
        ],
        "Required split_manifest.json Fields": [
            "manifest_version: integer 1.",
            "sample_unit: what one prediction row represents, for example image, volume, patient, slide, study, encounter, window, or row.",
            "key_columns: public columns that uniquely align score/test predictions with private truth; use [] only when row order is the explicit contract.",
            "task_heads: list of task heads, each either a string or object with type such as classification, multilabel, segmentation_2d, detection, regression, survival, or segmentation_3d.",
            "visibility: object listing agent_visible, evaluator_only, and forbidden columns/files.",
            "prediction_contract: object describing required candidate output files/columns and probability/file-path semantics.",
            "metric_contract: object with primary_metric and maximize.",
            METRIC_CONTRACT_REQUIREMENTS,
            "hidden_answers: object mirroring private_gt/private_gt.json answer_columns.",
        ],
        "Optional split_manifest.json Fields": [
            "row_representation: short explanation when one CSV row is not one prediction sample.",
            "dropped_samples: compact list of excluded samples with reasons, only when samples are intentionally dropped.",
            "sample_universe_summary: compact counts or source-file summary when useful; do not enumerate every sample.",
        ],
        "Builder Interface": [
            "Return one Python script.",
            "The script must accept --input-dir and --output-dir.",
            "Use deterministic splitting with seed 42 when no explicit validation source exists.",
            "Use only Python standard library plus pandas and PIL if needed.",
        ],
        "Principles": [
            "Treat tables and directories as equally valid sample sources.",
            "If a folder level, table column, or referenced file is evidence of an answer, keep it in agent_train.csv but hide it from score_val.csv and agent_test.csv.",
            "If an answer is stored as a file path, hide that path from public score/test files and retain it in hidden answers.",
            "If a column is computed from the answer, hide it from public score/test files too.",
            "Make your semantic assumptions explicit in split_manifest.json; do not leave task type, key columns, sample unit, or class/mask/box schema implicit.",
            "The public and private CSV files are the source of truth for row-level membership; do not create a per-sample sample_universe list for ordinary row-based datasets.",
            "Prefer references to existing files under --input-dir in split CSVs; do not copy existing input files or directories into public/ just to make paths convenient.",
            "Only write public/ for genuinely derived, converted, or anonymized files that do not already exist under --input-dir.",
            "Do not create modeling code; only prepare split files.",
            "Keep the implementation short and direct.",
        ],
        "Required Response": "Brief explanation followed by exactly one Python code block.",
    }
    if previous_script is not None:
        prompt["Previous Builder Script"] = previous_script
    if previous_failure is not None:
        prompt["Previous Verification Failure"] = json.dumps(
            _compact_builder_failure_for_prompt(previous_failure),
            indent=2,
            ensure_ascii=False,
        )
    if previous_stdout:
        prompt["Previous Builder Output"] = previous_stdout[-4000:]
    return prompt


def _query_builder(
    task_text: str,
    facts: dict[str, Any],
    *,
    model: str,
    temperature: float,
    previous_script: str | None = None,
    previous_failure: dict[str, Any] | None = None,
    previous_stdout: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    response, metadata = backend.query(
        system_message=_builder_system_prompt(),
        user_message=_builder_user_prompt(
            task_text,
            facts,
            previous_script=previous_script,
            previous_failure=previous_failure,
            previous_stdout=previous_stdout,
        ),
        model=model,
        temperature=temperature,
        return_metadata=True,
    )
    raw_response = str(response)
    code = extract_code(raw_response)
    return code, raw_response, metadata


def _read_output_csv(
    output_dir: Path, rel: str, errors: list[str], required: bool
) -> pd.DataFrame | None:
    path = output_dir / rel
    if not path.is_file():
        if required:
            errors.append(f"Missing required file: {rel}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        errors.append(f"Could not read {rel}: {exc}")
        return None


def _validate_manifest(manifest: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        errors.append(
            "private_gt/private_gt.json hidden-answer index must contain a JSON object."
        )
        return {}
    for key in ("score_val", "test"):
        if key not in manifest:
            continue
        item = manifest[key]
        if not isinstance(item, dict):
            errors.append(f"hidden_answers entry `{key}` must be an object.")
            continue
        if "answer_columns" not in item:
            present = ", ".join(sorted(map(str, item.keys())))
            errors.append(
                f"hidden_answers entry `{key}` must use the exact key answer_columns; present keys: {present}"
            )
            continue
        if not isinstance(item.get("answer_columns"), list):
            errors.append(
                f"hidden_answers entry `{key}` answer_columns must be a list."
            )
    return manifest


def _looks_like_path_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    return (
        "/" in text
        or "\\" in text
        or _compound_suffix(Path(lowered))
        in (
            TABLE_SUFFIXES
            | IMAGE_SUFFIXES
            | {".txt", ".xml", ".nii", ".nii.gz", ".npy"}
        )
    )


def _check_referenced_paths(
    df: pd.DataFrame | None,
    name: str,
    output_dir: Path,
    source_input_dir: Path | None,
    errors: list[str],
) -> None:
    if df is None:
        return
    missing: list[str] = []
    for column in df.columns:
        if df[column].dtype != object:
            continue
        for value in df[column].dropna().astype(str).head(200).tolist():
            if not _looks_like_path_text(value):
                continue
            if _resolve_path(value, output_dir, output_dir) is not None:
                continue
            if source_input_dir is not None and _resolve_path(
                value, source_input_dir, source_input_dir
            ):
                continue
            missing.append(f"{column}={value}")
            if len(missing) >= 5:
                break
        if len(missing) >= 5:
            break
    if missing:
        errors.append(f"{name} references missing files: {missing}")


def _check_source_coverage(
    facts: dict[str, Any], row_counts: dict[str, Any], errors: list[str]
) -> None:
    groups = (facts.get("image_groups") or {}).get("groups") or {}
    source_counts = []
    for group_path, group in groups.items():
        lowered = str(group_path).lower()
        if any(
            token in lowered for token in ("mask", "label", "annotation", "groundtruth")
        ):
            continue
        count = int(group.get("count") or 0)
        if count:
            source_counts.append((str(group_path), count))
    if not source_counts:
        return
    largest_path, largest_count = max(source_counts, key=lambda item: item[1])
    public_rows = sum(
        int(row_counts.get(name) or 0)
        for name in ("agent_train", "score_val", "agent_test")
    )
    if largest_count >= 10 and public_rows < int(largest_count * 0.8):
        errors.append(
            "Public split rows appear to cover only part of the source images: "
            f"{public_rows} rows across train/score/test vs {largest_count} images in `{largest_path}`. "
            "Unless the task explicitly asks for a subset, include all primary samples."
        )


def _file_digest(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _check_generated_public_does_not_copy_inputs(
    output_dir: Path, source_input_dir: Path | None, errors: list[str]
) -> None:
    if source_input_dir is None:
        return
    public_dir = output_dir / "public"
    if not public_dir.is_dir():
        return
    source_by_size: dict[int, list[Path]] = {}
    for source_path in source_input_dir.rglob("*"):
        if not source_path.is_file() or _inside_generated_dir(source_path):
            continue
        try:
            size = source_path.stat().st_size
        except OSError:
            continue
        source_by_size.setdefault(size, []).append(source_path)

    copied: list[str] = []
    source_digest_cache: dict[Path, str | None] = {}
    for public_path in public_dir.rglob("*"):
        if not public_path.is_file():
            continue
        try:
            candidates = source_by_size.get(public_path.stat().st_size, [])
        except OSError:
            continue
        if not candidates:
            continue
        public_digest = _file_digest(public_path)
        if public_digest is None:
            continue
        for source_path in candidates:
            if source_path not in source_digest_cache:
                source_digest_cache[source_path] = _file_digest(source_path)
            if source_digest_cache[source_path] != public_digest:
                continue
            copied.append(
                f"public/{public_path.relative_to(public_dir)} duplicates source input `{_relative(source_path, source_input_dir)}`"
            )
            break
        if len(copied) >= 5:
            break
    if copied:
        errors.append(
            "public/ must contain only genuinely derived, converted, or anonymized files; "
            "do not copy existing --input-dir files into public/. Examples: "
            + "; ".join(copied)
        )


def verify_split_package(
    output_dir: Path,
    source_input_dir: Path | None = None,
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    train_df = _read_output_csv(
        output_dir, "biomaker_splits/agent_train.csv", errors, True
    )
    score_df = _read_output_csv(
        output_dir, "biomaker_splits/score_val.csv", errors, True
    )
    test_df = _read_output_csv(
        output_dir, "biomaker_splits/agent_test.csv", errors, False
    )
    score_gt_df = _read_output_csv(output_dir, "private_gt/score_val.csv", errors, True)
    test_gt_df = _read_output_csv(output_dir, "private_gt/test.csv", errors, False)

    private_manifest_path = output_dir / "private_gt" / "private_gt.json"
    if private_manifest_path.is_file():
        try:
            manifest = _validate_manifest(
                json.loads(private_manifest_path.read_text(encoding="utf-8")), errors
            )
        except Exception as exc:
            errors.append(f"Could not read private_gt/private_gt.json: {exc}")
            manifest = {}
    else:
        errors.append("Missing required file: private_gt/private_gt.json")
        manifest = {}

    split_manifest = load_split_manifest(output_dir)

    if train_df is not None and len(train_df) == 0:
        errors.append("agent_train.csv has zero rows.")
    if score_df is not None and len(score_df) == 0:
        errors.append("score_val.csv has zero rows.")
    key_columns = (
        [str(column) for column in split_manifest.get("key_columns", [])]
        if isinstance(split_manifest.get("key_columns"), list)
        else []
    )
    _check_public_disjoint(
        train_df,
        "agent_train.csv",
        score_df,
        "score_val.csv",
        errors,
        key_columns,
    )
    _check_public_disjoint(
        train_df,
        "agent_train.csv",
        test_df,
        "agent_test.csv",
        errors,
        key_columns,
    )
    _check_public_disjoint(
        score_df,
        "score_val.csv",
        test_df,
        "agent_test.csv",
        errors,
        key_columns,
    )

    if score_df is not None and score_gt_df is not None:
        if len(score_df) != len(score_gt_df):
            errors.append(
                f"score_val/private_gt row mismatch: {len(score_df)} vs {len(score_gt_df)}"
            )
        public_cols = set(map(str, score_df.columns))
        private_cols = set(map(str, score_gt_df.columns))
        if not public_cols <= private_cols:
            errors.append(
                "private_gt/score_val.csv must retain all public score_val columns; "
                f"missing {sorted(public_cols - private_cols)}"
            )
        answers = set(
            map(str, manifest.get("score_val", {}).get("answer_columns") or [])
        )
        if not answers:
            errors.append(
                "score_val answer_columns is empty in private_gt/private_gt.json."
            )
        hidden_columns = private_cols - public_cols
        unlisted_hidden = sorted(hidden_columns - answers)
        if unlisted_hidden:
            errors.append(
                "private_gt/score_val.csv contains hidden columns not listed in answer_columns: "
                f"{unlisted_hidden}"
            )
        missing_answers = sorted(answers - private_cols)
        if missing_answers:
            errors.append(
                f"score_val answer columns missing from private_gt: {missing_answers}"
            )
        leaked = sorted(answers & public_cols)
        if leaked:
            errors.append(f"score_val public file leaks answer columns: {leaked}")

    if test_df is not None and test_gt_df is not None:
        if len(test_df) != len(test_gt_df):
            errors.append(
                f"test/private_gt row mismatch: {len(test_df)} vs {len(test_gt_df)}"
            )
        public_cols = set(map(str, test_df.columns))
        private_cols = set(map(str, test_gt_df.columns))
        if not public_cols <= private_cols:
            errors.append(
                "private_gt/test.csv must retain all public agent_test columns; "
                f"missing {sorted(public_cols - private_cols)}"
            )
        answers = set(map(str, manifest.get("test", {}).get("answer_columns") or []))
        hidden_columns = private_cols - public_cols
        unlisted_hidden = sorted(hidden_columns - answers)
        if unlisted_hidden:
            errors.append(
                "private_gt/test.csv contains hidden columns not listed in answer_columns: "
                f"{unlisted_hidden}"
            )
        missing_answers = sorted(answers - private_cols)
        if missing_answers:
            errors.append(
                f"test answer columns missing from private_gt: {missing_answers}"
            )
        leaked = sorted(answers & public_cols)
        if leaked:
            errors.append(f"agent_test public file leaks answer columns: {leaked}")

    row_counts = {
        "agent_train": None if train_df is None else int(len(train_df)),
        "score_val": None if score_df is None else int(len(score_df)),
        "agent_test": None if test_df is None else int(len(test_df)),
        "score_val_gt": None if score_gt_df is None else int(len(score_gt_df)),
        "test_gt": None if test_gt_df is None else int(len(test_gt_df)),
    }
    for name, df in (
        ("agent_train.csv", train_df),
        ("score_val.csv", score_df),
        ("agent_test.csv", test_df),
        ("private_gt/score_val.csv", score_gt_df),
        ("private_gt/test.csv", test_gt_df),
    ):
        _check_referenced_paths(df, name, output_dir, source_input_dir, errors)
    if facts is not None:
        _check_source_coverage(facts, row_counts, errors)
    _check_generated_public_does_not_copy_inputs(output_dir, source_input_dir, errors)
    if not split_manifest:
        split_manifest = legacy_manifest(
            output_dir, private_gt=manifest, row_counts=row_counts
        )
        write_split_manifest(output_dir, split_manifest)
    validate_split_manifest(
        split_manifest,
        output_dir=output_dir,
        source_input_dir=source_input_dir,
        frames={"train": train_df, "score_val": score_df, "test": test_df},
        errors=errors,
        warnings=warnings,
    )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "row_counts": row_counts,
        "hidden_answers": manifest,
        "private_gt": manifest,
        "split_manifest": _manifest_summary_for_verifier(split_manifest),
    }


def _check_public_disjoint(
    left: pd.DataFrame | None,
    left_name: str,
    right: pd.DataFrame | None,
    right_name: str,
    errors: list[str],
    key_columns: list[str],
) -> None:
    if left is None or right is None:
        return
    shared_cols = [
        column for column in key_columns if column in left and column in right
    ]
    if not shared_cols:
        return
    left_keys = set(
        map(tuple, left[shared_cols].astype(str).itertuples(index=False, name=None))
    )
    right_keys = set(
        map(tuple, right[shared_cols].astype(str).itertuples(index=False, name=None))
    )
    overlap = left_keys & right_keys
    if overlap:
        errors.append(
            f"{left_name} and {right_name} overlap on shared public columns {shared_cols}; "
            f"overlap_rows_at_least={len(overlap)}"
        )


def compare_split_dirs(
    candidate_dir: Path, reference_dir: Path | None
) -> dict[str, Any] | None:
    if reference_dir is None or not reference_dir.is_dir():
        return None

    def read_info(name: str, base: Path) -> dict[str, Any]:
        path = base / name
        if not path.is_file():
            return {"exists": False}
        df = pd.read_csv(path)
        return {
            "exists": True,
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
        }

    result = {"reference_dir": str(reference_dir), "files": {}}
    for name in ["agent_train.csv", "score_val.csv", "agent_test.csv"]:
        candidate = read_info(name, candidate_dir)
        reference = read_info(name, reference_dir)
        item = {"candidate": candidate, "reference": reference}
        if candidate.get("exists") and reference.get("exists"):
            item["same_rows"] = candidate["rows"] == reference["rows"]
            item["same_columns"] = candidate["columns"] == reference["columns"]
        result["files"][name] = item
    return result


def run_agentic_split_builder_probe(
    input_dir: Path,
    task_file: Path,
    output_dir: Path,
    *,
    model: str = "gpt-5.4",
    temperature: float = 0.0,
    timeout: int = 300,
    max_attempts: int = 2,
    reference_split_dir: Path | None = None,
    debug_artifacts: bool = False,
) -> PlannerResult:
    input_dir = input_dir.resolve()
    task_file = task_file.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_text = task_file.read_text(encoding="utf-8", errors="replace")
    _cache_key, facts = split_builder_cache_key(
        input_dir=input_dir,
        task_text=task_text,
        model=model,
        temperature=temperature,
    )
    attempts: list[dict[str, Any]] = []
    previous_script: str | None = None
    previous_failure: dict[str, Any] | None = None
    previous_stdout: str | None = None

    final_verifier: dict[str, Any] | None = None
    for attempt_index in range(1, max_attempts + 1):
        code, raw_response, metadata = _query_builder(
            task_text,
            facts,
            model=model,
            temperature=temperature,
            previous_script=previous_script,
            previous_failure=previous_failure,
            previous_stdout=previous_stdout,
        )
        if not code.strip():
            verifier = {
                "ok": False,
                "errors": [
                    "Builder response did not contain a valid Python code block.",
                    "Return a brief explanation followed by exactly one fenced Python code block, for example ```python ... ```.",
                ],
                "builder_returncode": None,
            }
            attempts.append(
                {
                    "attempt": attempt_index,
                    "ok": False,
                    "verifier": verifier,
                    "stdout_tail": raw_response[-4000:],
                    "prompt_tokens": metadata.get("prompt_tokens"),
                    "completion_tokens": metadata.get("completion_tokens"),
                }
            )
            if debug_artifacts:
                debug_dir = output_dir / "_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"response_{attempt_index}.txt").write_text(
                    raw_response, encoding="utf-8"
                )
                (debug_dir / f"stdout_{attempt_index}.txt").write_text(
                    raw_response, encoding="utf-8"
                )
                (debug_dir / f"verifier_{attempt_index}.json").write_text(
                    json.dumps(verifier, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            final_verifier = verifier
            previous_script = None
            previous_failure = verifier
            previous_stdout = raw_response
            continue

        builder_path = output_dir / f"_builder_attempt_{attempt_index}.py"
        builder_path.write_text(code, encoding="utf-8")

        for rel in ("biomaker_splits", "private_gt"):
            path = output_dir / rel
            if path.exists():
                shutil.rmtree(path)

        completed = subprocess.run(
            [
                sys.executable,
                str(builder_path),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
            cwd=str(output_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        verifier = verify_split_package(
            output_dir, source_input_dir=input_dir, facts=facts
        )
        verifier["builder_returncode"] = completed.returncode
        if completed.returncode != 0:
            verifier["ok"] = False
            verifier.setdefault("errors", []).append("builder script exited nonzero")
        attempts.append(
            {
                "attempt": attempt_index,
                "ok": bool(verifier.get("ok")),
                "verifier": verifier,
                "stdout_tail": completed.stdout[-4000:],
                "prompt_tokens": metadata.get("prompt_tokens"),
                "completion_tokens": metadata.get("completion_tokens"),
            }
        )
        if debug_artifacts:
            debug_dir = output_dir / "_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / f"response_{attempt_index}.txt").write_text(
                raw_response, encoding="utf-8"
            )
            (debug_dir / f"stdout_{attempt_index}.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            (debug_dir / f"verifier_{attempt_index}.json").write_text(
                json.dumps(verifier, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        final_verifier = verifier
        if verifier.get("ok"):
            break
        previous_script = code
        previous_failure = verifier
        previous_stdout = completed.stdout

    if final_verifier is None:
        raise RuntimeError("Dataset planning verifier did not run")
    if debug_artifacts:
        debug_dir = output_dir / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "dataset_facts.json").write_text(
            json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (debug_dir / "attempts.json").write_text(
            json.dumps(attempts, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if not final_verifier.get("ok"):
        raise RuntimeError(
            "Agentic split builder failed: "
            + json.dumps(
                _compact_builder_failure_for_prompt(final_verifier),
                ensure_ascii=False,
            )
            + "\nLast builder output:\n"
            + str(previous_stdout or "")[-4000:]
        )

    comparison = compare_split_dirs(output_dir / "biomaker_splits", reference_split_dir)
    return PlannerResult(
        plan={"mode": "agentic_builder", "attempts": attempts},
        output_dir=output_dir,
        split_dir=output_dir / "biomaker_splits",
        private_gt_dir=output_dir / "private_gt",
        inspection_history=[],
        verifier=final_verifier,
        comparison=comparison,
    )
