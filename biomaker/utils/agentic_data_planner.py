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
from .response import extract_code


TABLE_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
GENERATED_DIR_NAMES = {
    "biomaker_splits",
    "biomaker_cache",
    "private_gt",
    "_split_builder",
}
SPLIT_BUILDER_CACHE_VERSION = 1


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
            item["examples_hidden"] = "small scalar examples hidden to avoid leaking possible answers"
        columns.append(item)
    return {"path": _relative(path, input_dir), "rows": int(len(df)), "columns": columns}


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


def _image_group_facts(files: list[Path], input_dir: Path, max_files: int = 3000) -> dict[str, Any]:
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


def build_dataset_facts(input_dir: Path, *, max_files: int = 5000, max_tables: int = 30) -> dict[str, Any]:
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
        "and private ground-truth files. Infer semantics from evidence; do not rely on "
        "a fixed task type or one naming convention."
    )


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
            "private_gt/private_gt.json: JSON object with keys score_val and test. Each entry must contain answer_columns, for example {\"score_val\":{\"answer_columns\":[\"label\"]},\"test\":{\"answer_columns\":[\"label\"]}}.",
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
            "If an answer is stored as a file path, hide that path from public score/test files and retain it in private_gt.",
            "If a column is computed from the answer, hide it from public score/test files too.",
            "Do not create modeling code; only prepare split files.",
            "Keep the implementation short and direct.",
        ],
        "Required Response": "Brief explanation followed by exactly one Python code block.",
    }
    if previous_script is not None:
        prompt["Previous Builder Script"] = previous_script
    if previous_failure is not None:
        prompt["Previous Verification Failure"] = json.dumps(
            previous_failure, indent=2, ensure_ascii=False
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
    if not code.strip():
        raise RuntimeError("Builder response did not contain a valid Python code block.")
    return code, raw_response, metadata


def _read_output_csv(output_dir: Path, rel: str, errors: list[str], required: bool) -> pd.DataFrame | None:
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
        errors.append("private_gt/private_gt.json must contain a JSON object.")
        return {}
    for key in ("score_val", "test"):
        if key not in manifest:
            continue
        item = manifest[key]
        if not isinstance(item, dict):
            errors.append(f"private_gt entry `{key}` must be an object.")
            continue
        if "answer_columns" not in item:
            present = ", ".join(sorted(map(str, item.keys())))
            errors.append(
                f"private_gt entry `{key}` must use the exact key answer_columns; present keys: {present}"
            )
            continue
        if not isinstance(item.get("answer_columns"), list):
            errors.append(f"private_gt entry `{key}` answer_columns must be a list.")
    return manifest


def verify_split_package(output_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    train_df = _read_output_csv(output_dir, "biomaker_splits/agent_train.csv", errors, True)
    score_df = _read_output_csv(output_dir, "biomaker_splits/score_val.csv", errors, True)
    test_df = _read_output_csv(output_dir, "biomaker_splits/agent_test.csv", errors, False)
    score_gt_df = _read_output_csv(output_dir, "private_gt/score_val.csv", errors, True)
    test_gt_df = _read_output_csv(output_dir, "private_gt/test.csv", errors, False)

    manifest_path = output_dir / "private_gt" / "private_gt.json"
    if manifest_path.is_file():
        try:
            manifest = _validate_manifest(
                json.loads(manifest_path.read_text(encoding="utf-8")), errors
            )
        except Exception as exc:
            errors.append(f"Could not read private_gt/private_gt.json: {exc}")
            manifest = {}
    else:
        errors.append("Missing required file: private_gt/private_gt.json")
        manifest = {}

    if train_df is not None and len(train_df) == 0:
        errors.append("agent_train.csv has zero rows.")
    if score_df is not None and len(score_df) == 0:
        errors.append("score_val.csv has zero rows.")
    _check_public_disjoint(
        train_df,
        "agent_train.csv",
        score_df,
        "score_val.csv",
        errors,
    )
    _check_public_disjoint(
        train_df,
        "agent_train.csv",
        test_df,
        "agent_test.csv",
        errors,
    )
    _check_public_disjoint(
        score_df,
        "score_val.csv",
        test_df,
        "agent_test.csv",
        errors,
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
        answers = set(map(str, manifest.get("score_val", {}).get("answer_columns") or []))
        if not answers:
            errors.append("score_val answer_columns is empty in private_gt/private_gt.json.")
        hidden_columns = private_cols - public_cols
        unlisted_hidden = sorted(hidden_columns - answers)
        if unlisted_hidden:
            errors.append(
                "private_gt/score_val.csv contains hidden columns not listed in answer_columns: "
                f"{unlisted_hidden}"
            )
        missing_answers = sorted(answers - private_cols)
        if missing_answers:
            errors.append(f"score_val answer columns missing from private_gt: {missing_answers}")
        leaked = sorted(answers & public_cols)
        if leaked:
            errors.append(f"score_val public file leaks answer columns: {leaked}")

    if test_df is not None and test_gt_df is not None:
        if len(test_df) != len(test_gt_df):
            errors.append(f"test/private_gt row mismatch: {len(test_df)} vs {len(test_gt_df)}")
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
            errors.append(f"test answer columns missing from private_gt: {missing_answers}")
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
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "row_counts": row_counts,
        "private_gt": manifest,
    }


def _check_public_disjoint(
    left: pd.DataFrame | None,
    left_name: str,
    right: pd.DataFrame | None,
    right_name: str,
    errors: list[str],
) -> None:
    if left is None or right is None:
        return
    shared_cols = [c for c in right.columns if c in left.columns]
    if not shared_cols:
        return
    left_keys = set(map(tuple, left[shared_cols].astype(str).itertuples(index=False, name=None)))
    right_keys = set(map(tuple, right[shared_cols].astype(str).itertuples(index=False, name=None)))
    overlap = left_keys & right_keys
    if overlap:
        errors.append(
            f"{left_name} and {right_name} overlap on shared public columns {shared_cols}; "
            f"overlap_rows_at_least={len(overlap)}"
        )


def compare_split_dirs(candidate_dir: Path, reference_dir: Path | None) -> dict[str, Any] | None:
    if reference_dir is None or not reference_dir.is_dir():
        return None

    def read_info(name: str, base: Path) -> dict[str, Any]:
        path = base / name
        if not path.is_file():
            return {"exists": False}
        df = pd.read_csv(path)
        return {"exists": True, "rows": int(len(df)), "columns": list(map(str, df.columns))}

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
        verifier = verify_split_package(output_dir)
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

    assert final_verifier is not None
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
            + json.dumps(final_verifier, ensure_ascii=False)
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
