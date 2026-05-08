"""Deterministic agent-visible input cache utilities."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    import SimpleITK as sitk
except Exception:  # pragma: no cover - optional runtime dependency
    sitk = None


CACHE_VERSION = 1
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MEDICAL_IMAGE_SUFFIXES = {".nii", ".nii.gz", ".nrrd", ".mha", ".mhd"}


@dataclass(frozen=True)
class InputCacheResult:
    path: Path
    fingerprint: str
    files_total: int
    arrays_decoded: int
    errors: int
    reused: bool
    prompt_text: str


def _compound_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def _json_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "exists": True,
            "resolved_path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "suffix": _compound_suffix(path),
        }
    except FileNotFoundError:
        return {
            "exists": False,
            "resolved_path": str(path),
            "size_bytes": None,
            "mtime_ns": None,
            "suffix": _compound_suffix(path),
        }


def _looks_like_path_column(series: pd.Series, input_dir: Path, base_dir: Path) -> bool:
    values = [str(value).strip() for value in series.dropna().head(20).tolist()]
    values = [value for value in values if value]
    if not values:
        return False
    hits = 0
    for value in values[:8]:
        path = Path(value)
        candidates = [path] if path.is_absolute() else [input_dir / path, base_dir / path]
        if any(candidate.exists() for candidate in candidates):
            hits += 1
    return hits > 0


def _resolve_visible_path(value: Any, input_dir: Path, base_dir: Path) -> Path | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [input_dir / path, base_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _collect_visible_file_records(
    input_dir: Path, split_plan: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    agent_visible = split_plan.get("agent_visible")
    if not isinstance(agent_visible, dict):
        return records

    for split_name, split_rel in sorted(agent_visible.items()):
        split_csv = input_dir / str(split_rel)
        if not split_csv.is_file():
            continue
        try:
            df = pd.read_csv(split_csv)
        except Exception:
            continue

        path_columns = []
        for column in df.columns:
            lower = column.lower()
            if "path" in lower or "file" in lower:
                path_columns.append(column)
            elif df[column].dtype == object and _looks_like_path_column(
                df[column], input_dir, split_csv.parent
            ):
                path_columns.append(column)

        for row_index, row in df.iterrows():
            for column in path_columns:
                resolved = _resolve_visible_path(row.get(column), input_dir, split_csv.parent)
                if resolved is None:
                    continue
                raw_path = str(row.get(column)).strip()
                file_info = _file_fingerprint(resolved)
                record_key = _json_hash(
                    {
                        "file": file_info,
                    },
                    length=20,
                )
                records.append(
                    {
                        "split": split_name,
                        "row_index": int(row_index),
                        "column": column,
                        "original_path": raw_path,
                        "resolved_path": file_info["resolved_path"],
                        "cache_key": record_key,
                        **file_info,
                    }
                )
    return records


def _load_array(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = _compound_suffix(path)
    metadata: dict[str, Any] = {"suffix": suffix}
    if suffix == ".npy":
        arr = np.load(path)
        metadata["reader"] = "numpy"
        return np.asarray(arr), metadata
    if suffix in IMAGE_SUFFIXES:
        with Image.open(path) as img:
            arr = np.asarray(img)
            metadata.update({"reader": "PIL", "mode": img.mode})
            return arr, metadata
    if suffix in MEDICAL_IMAGE_SUFFIXES:
        if sitk is None:
            raise RuntimeError("SimpleITK is not installed")
        image = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(image)
        metadata.update(
            {
                "reader": "SimpleITK",
                "spacing": list(image.GetSpacing()),
                "origin": list(image.GetOrigin()),
                "direction": list(image.GetDirection()),
            }
        )
        return arr, metadata
    raise RuntimeError(f"no lossless array decoder registered for suffix {suffix!r}")


def _write_cache(cache_dir: Path, records: list[dict[str, Any]], decode_arrays: bool) -> None:
    arrays_dir = cache_dir / "arrays"
    metadata_dir = cache_dir / "metadata"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    arrays_decoded = 0
    errors = 0
    for record in records:
        key = record["cache_key"]
        array_rel = f"arrays/{key}.npy"
        metadata_rel = f"metadata/{key}.json"
        metadata: dict[str, Any] = {
            "cache_version": CACHE_VERSION,
            "original_path": record["original_path"],
            "resolved_path": record["resolved_path"],
            "exists": record["exists"],
            "size_bytes": record["size_bytes"],
            "mtime_ns": record["mtime_ns"],
            "suffix": record["suffix"],
            "array_cached": False,
        }
        status = "metadata_only"
        error = ""
        if decode_arrays and record["exists"]:
            try:
                arr, reader_metadata = _load_array(Path(record["resolved_path"]))
                np.save(arrays_dir / f"{key}.npy", arr, allow_pickle=False)
                metadata.update(reader_metadata)
                metadata.update(
                    {
                        "array_cached": True,
                        "array_shape": list(arr.shape),
                        "array_dtype": str(arr.dtype),
                    }
                )
                status = "decoded"
                arrays_decoded += 1
            except Exception as exc:
                error = str(exc)
                status = "error"
                errors += 1
        elif not record["exists"]:
            status = "missing"
            error = "source file does not exist"
            errors += 1

        (metadata_dir / f"{key}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        rows.append(
            {
                "split": record["split"],
                "row_index": record["row_index"],
                "column": record["column"],
                "original_path": record["original_path"],
                "cache_array_path": (
                    f"./input/biomaker_cache/{array_rel}" if status == "decoded" else ""
                ),
                "cache_metadata_path": f"./input/biomaker_cache/{metadata_rel}",
                "exists": record["exists"],
                "size_bytes": record["size_bytes"],
                "suffix": record["suffix"],
                "status": status,
                "error": error,
            }
        )

    columns = [
        "split",
        "row_index",
        "column",
        "original_path",
        "cache_array_path",
        "cache_metadata_path",
        "exists",
        "size_bytes",
        "suffix",
        "status",
        "error",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(cache_dir / "files.csv", index=False)
    manifest = {
        "cache_version": CACHE_VERSION,
        "files_total": len(records),
        "arrays_decoded": arrays_decoded,
        "errors": errors,
        "decode_arrays": decode_arrays,
        "note": (
            "This directory contains deterministic caches derived only from "
            "agent-visible input files. It exists to reduce repeated decoding and "
            "I/O cost. It does not define the modeling representation."
        ),
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (cache_dir / "README.md").write_text(
        "# BioMaker Input Cache\n\n"
        "This directory contains deterministic caches derived only from agent-visible "
        "input files. It exists to reduce repeated decoding and I/O cost.\n\n"
        "These caches are not a prescribed feature set. Candidate solutions may use "
        "these caches or the original files and may choose their own preprocessing.\n",
        encoding="utf-8",
    )


def _write_run_index(cache_dir: Path, records: list[dict[str, Any]], decode_arrays: bool) -> tuple[int, int]:
    rows = []
    errors = 0
    for record in records:
        key = record["cache_key"]
        array_rel = f"arrays/{key}.npy"
        metadata_rel = f"metadata/{key}.json"
        array_path = cache_dir / array_rel
        metadata_path = cache_dir / metadata_rel
        status = "decoded" if array_path.is_file() else "metadata_only"
        error = ""
        if decode_arrays and not array_path.is_file():
            status = "error"
            error = "decoded array is missing from shared cache"
            errors += 1
        if not metadata_path.is_file():
            status = "error"
            error = "metadata is missing from shared cache"
            errors += 1
        rows.append(
            {
                "split": record["split"],
                "row_index": record["row_index"],
                "column": record["column"],
                "original_path": record["original_path"],
                "cache_array_path": (
                    f"./input/biomaker_cache/{array_rel}" if array_path.is_file() else ""
                ),
                "cache_metadata_path": (
                    f"./input/biomaker_cache/{metadata_rel}"
                    if metadata_path.is_file()
                    else ""
                ),
                "exists": record["exists"],
                "size_bytes": record["size_bytes"],
                "suffix": record["suffix"],
                "status": status,
                "error": error,
            }
        )

    columns = [
        "split",
        "row_index",
        "column",
        "original_path",
        "cache_array_path",
        "cache_metadata_path",
        "exists",
        "size_bytes",
        "suffix",
        "status",
        "error",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(cache_dir / "files.csv", index=False)
    arrays_decoded = sum(1 for row in rows if row["cache_array_path"])
    manifest = {
        "cache_version": CACHE_VERSION,
        "files_total": len(records),
        "arrays_decoded": arrays_decoded,
        "errors": errors,
        "decode_arrays": decode_arrays,
        "note": (
            "This directory contains a run-specific index into deterministic caches "
            "derived only from agent-visible input files. It exists to reduce repeated "
            "decoding and I/O cost. It does not define the modeling representation."
        ),
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (cache_dir / "README.md").write_text(
        "# BioMaker Input Cache\n\n"
        "This directory contains deterministic caches derived only from agent-visible "
        "input files. It exists to reduce repeated decoding and I/O cost.\n\n"
        "These caches are not a prescribed feature set. Candidate solutions may use "
        "these caches or the original files and may choose their own preprocessing.\n",
        encoding="utf-8",
    )
    return arrays_decoded, errors


def _replace_with_symlink(link_path: Path, target: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target)


def _load_manifest(cache_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def prepare_input_cache(
    input_dir: Path | str,
    split_plan: dict[str, Any],
    *,
    shared_root: Path | str | None = None,
    decode_arrays: bool = True,
) -> InputCacheResult:
    input_dir = Path(input_dir)
    records = _collect_visible_file_records(input_dir, split_plan)
    fingerprint = _json_hash(
        {
            "cache_version": CACHE_VERSION,
            "decode_arrays": decode_arrays,
            "records": [
                {
                    key: record.get(key)
                    for key in [
                        "split",
                        "row_index",
                        "column",
                        "resolved_path",
                        "exists",
                        "size_bytes",
                        "mtime_ns",
                        "suffix",
                    ]
                }
                for record in records
            ],
        },
        length=24,
    )

    link_path = input_dir / "biomaker_cache"
    reused = False
    if shared_root is not None:
        shared_root = Path(shared_root)
        shared_root.mkdir(parents=True, exist_ok=True)
        cache_dir = shared_root / fingerprint
        if not (cache_dir / "manifest.json").is_file():
            tmp_dir = shared_root / f".tmp-{fingerprint}-{os.getpid()}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            _write_cache(tmp_dir, records, decode_arrays=decode_arrays)
            try:
                tmp_dir.rename(cache_dir)
            except FileExistsError:
                shutil.rmtree(tmp_dir)
        else:
            reused = True
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.mkdir(parents=True, exist_ok=True)
        _replace_with_symlink(link_path / "arrays", cache_dir / "arrays")
        _replace_with_symlink(link_path / "metadata", cache_dir / "metadata")
        arrays_decoded, errors = _write_run_index(
            link_path, records, decode_arrays=decode_arrays
        )
    else:
        cache_dir = link_path
        if cache_dir.exists() or cache_dir.is_symlink():
            if cache_dir.is_dir() and not cache_dir.is_symlink():
                shutil.rmtree(cache_dir)
            else:
                cache_dir.unlink()
        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_cache(cache_dir, records, decode_arrays=decode_arrays)
        manifest = _load_manifest(cache_dir)
        arrays_decoded = int(manifest.get("arrays_decoded") or 0)
        errors = int(manifest.get("errors") or 0)

    files_total = len(records)
    prompt_text = (
        "BioMaker provides deterministic agent-visible input caches under "
        "`./input/biomaker_cache` to reduce repeated decoding and I/O cost. "
        "These caches are equivalent access representations, not prescribed "
        "modeling features. Original files remain available. See "
        "`./input/biomaker_cache/files.csv` for the mapping."
    )
    return InputCacheResult(
        path=link_path,
        fingerprint=fingerprint,
        files_total=files_total,
        arrays_decoded=arrays_decoded,
        errors=errors,
        reused=reused,
        prompt_text=prompt_text,
    )
