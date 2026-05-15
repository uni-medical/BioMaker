"""Structured split manifest helpers for agentic data preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

MANIFEST_VERSION = 1
MANIFEST_NAME = "split_manifest.json"
SPLIT_TO_FILE = {
    "train": "biomaker_splits/agent_train.csv",
    "score_val": "biomaker_splits/score_val.csv",
    "test": "biomaker_splits/agent_test.csv",
}
FILE_TO_SPLIT = {value: key for key, value in SPLIT_TO_FILE.items()}


def load_split_manifest(base_dir: Path) -> dict[str, Any]:
    for path in (
        base_dir / MANIFEST_NAME,
        base_dir / "biomaker_splits" / MANIFEST_NAME,
    ):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}
    return {}


def write_split_manifest(base_dir: Path, manifest: dict[str, Any]) -> None:
    (base_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_row_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def _answer_columns(hidden_answers: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name in ("score_val", "test"):
        entry = hidden_answers.get(split_name)
        if isinstance(entry, dict):
            result[split_name] = {
                "answer_columns": list(map(str, entry.get("answer_columns") or []))
            }
    return result


def legacy_manifest(
    output_dir: Path,
    *,
    private_gt: dict[str, Any],
    row_counts: dict[str, Any],
) -> dict[str, Any]:
    """Build a compatibility manifest from legacy hidden-answer outputs."""
    splits = {}
    for split_name, rel in SPLIT_TO_FILE.items():
        path = output_dir / rel
        if not path.is_file():
            continue
        splits[split_name] = {
            "file": rel,
            "row_count": row_counts.get(Path(rel).stem) or _csv_row_count(path),
            "sha256": _file_sha256(path),
        }
    return {
        "manifest_version": MANIFEST_VERSION,
        "schema": "biomaker_split_manifest",
        "compatibility": "legacy_hidden_answers_json",
        "sample_unit": "row",
        "key_columns": [],
        "task_heads": [],
        "splits": splits,
        "hidden_answers": _answer_columns(private_gt),
        "visibility": {
            "agent_visible": ["biomaker_splits/*.csv", "public/**"],
            "evaluator_only": ["private_gt/**"],
        },
        "prediction_contract": {},
        "metric_contract": {},
    }


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _validate_task_heads(manifest: dict[str, Any], errors: list[str]) -> None:
    heads = manifest.get("task_heads")
    if isinstance(heads, str):
        return
    if isinstance(heads, list):
        for index, head in enumerate(heads):
            if isinstance(head, str):
                continue
            if isinstance(head, dict) and isinstance(head.get("type"), str):
                continue
            errors.append(f"task_heads[{index}] must be a string or object with type.")
        return
    errors.append("split_manifest.json must declare task_heads as a list or string.")


def _validate_key_columns(
    manifest: dict[str, Any],
    frames: dict[str, pd.DataFrame | None],
    errors: list[str],
) -> None:
    key_columns = manifest.get("key_columns")
    if key_columns is None:
        return
    if not isinstance(key_columns, list) or not all(
        isinstance(column, str) and column for column in key_columns
    ):
        errors.append("split_manifest.json key_columns must be a list of strings.")
        return
    for split_name, df in frames.items():
        if df is None or split_name == "train":
            continue
        missing = [column for column in key_columns if column not in df.columns]
        if missing:
            errors.append(f"{split_name} is missing manifest key_columns: {missing}")
            continue
        if key_columns and df[key_columns].astype(str).duplicated().any():
            errors.append(f"{split_name} has duplicate manifest key_columns.")


def _validate_sample_universe(
    manifest: dict[str, Any],
    output_dir: Path,
    source_input_dir: Path | None,
    frames: dict[str, pd.DataFrame | None],
    errors: list[str],
) -> None:
    samples = manifest.get("sample_universe")
    if samples is None:
        return
    if not isinstance(samples, list):
        errors.append("split_manifest.json sample_universe must be a list.")
        return
    seen: set[str] = set()
    counts = {"train": 0, "score_val": 0, "test": 0}
    duplicate_sample_id_errors = 0
    missing_source_path_errors = 0
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            errors.append(f"sample_universe[{index}] must be an object.")
            continue
        raw_sample_id = item.get("sample_id")
        sample_id = "" if raw_sample_id is None else str(raw_sample_id).strip()
        if not sample_id:
            errors.append(f"sample_universe[{index}] is missing sample_id.")
        elif sample_id in seen:
            duplicate_sample_id_errors += 1
            if duplicate_sample_id_errors <= 20:
                errors.append(
                    f"sample_universe has duplicate sample_id: {sample_id}"
                )
        else:
            seen.add(sample_id)
        split = str(item.get("split") or "").strip()
        if split in counts:
            counts[split] += 1
        elif split == "dropped":
            if not str(item.get("reason") or "").strip():
                errors.append(f"sample_universe[{index}] is dropped without a reason.")
        else:
            errors.append(
                f"sample_universe[{index}] split must be train, score_val, test, or dropped."
            )
        for raw_path in _as_list(item.get("source_paths")):
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            path = Path(raw_path)
            candidates = [path] if path.is_absolute() else [output_dir / path]
            if source_input_dir is not None and not path.is_absolute():
                candidates.append(source_input_dir / path)
                if path.parts and path.parts[0] == "input" and len(path.parts) > 1:
                    candidates.append(source_input_dir / Path(*path.parts[1:]))
            if not any(candidate.exists() for candidate in candidates):
                missing_source_path_errors += 1
                if missing_source_path_errors <= 20:
                    errors.append(
                        f"sample_universe[{index}] source path does not exist: {raw_path}"
                    )
                break
    if duplicate_sample_id_errors > 20:
        errors.append(
            "sample_universe has additional duplicate sample_id errors: "
            f"{duplicate_sample_id_errors - 20}"
        )
    if missing_source_path_errors > 20:
        errors.append(
            "sample_universe has additional missing source path errors: "
            f"{missing_source_path_errors - 20}"
        )
    if manifest.get("row_representation", "one_row_per_sample") == "one_row_per_sample":
        for split_name, expected in counts.items():
            df = frames.get(split_name)
            if df is None:
                if expected:
                    errors.append(
                        f"sample_universe declares {expected} {split_name} samples but split file is missing."
                    )
                continue
            if int(len(df)) != expected:
                errors.append(
                    f"sample_universe {split_name} count {expected} does not match split rows {len(df)}."
                )


def _validate_prediction_contract(manifest: dict[str, Any], errors: list[str]) -> None:
    contract = manifest.get("prediction_contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        errors.append("split_manifest.json prediction_contract must be an object.")
        return
    for key in (
        "key_columns",
        "prediction_columns",
        "file_reference_columns",
    ):
        value = contract.get(key)
        if value is not None and (
            not isinstance(value, list)
            or not all(isinstance(column, str) and column for column in value)
        ):
            errors.append(
                f"split_manifest.json prediction_contract.{key} must be a list of strings."
            )
    for key in ("validation_artifact", "test_submission", "prediction_type"):
        value = contract.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(
                f"split_manifest.json prediction_contract.{key} must be a string."
            )
    if contract.get("file_reference_columns") and not contract.get(
        "prediction_columns"
    ):
        contract["prediction_columns"] = list(contract["file_reference_columns"])


def validate_split_manifest(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    source_input_dir: Path | None,
    frames: dict[str, pd.DataFrame | None],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    if not manifest:
        warnings.append(
            "split_manifest.json missing; using legacy hidden-answers compatibility mode."
        )
        return manifest
    version = manifest.get("manifest_version")
    if version != MANIFEST_VERSION:
        errors.append(
            f"split_manifest.json manifest_version must be {MANIFEST_VERSION}; got {version!r}."
        )
    if not isinstance(manifest.get("sample_unit"), str) or not manifest.get(
        "sample_unit"
    ):
        errors.append("split_manifest.json must declare sample_unit.")
    _validate_task_heads(manifest, errors)
    _validate_key_columns(manifest, frames, errors)
    _validate_sample_universe(manifest, output_dir, source_input_dir, frames, errors)
    _validate_prediction_contract(manifest, errors)
    for key in ("visibility", "prediction_contract", "metric_contract"):
        if key in manifest and not isinstance(manifest[key], dict):
            errors.append(f"split_manifest.json {key} must be an object.")
    return manifest
