"""Canonical path helpers for BioMaker split artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PATH_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".nii",
    ".nii.gz",
    ".dcm",
    ".xml",
    ".txt",
    ".npy",
    ".npz",
}


def compound_suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return path.suffix.lower()


def looks_like_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    return (
        "/" in text or "\\" in text or compound_suffix(Path(lowered)) in PATH_SUFFIXES
    )


def input_relative(candidate: Path, input_dir: Path) -> str | None:
    """Return an agent-visible `input/...` path for a file under input_dir."""
    try:
        resolved = candidate.resolve()
    except Exception:
        resolved = candidate
    try:
        return str(Path("input") / resolved.relative_to(input_dir.resolve()))
    except Exception:
        pass
    for child in input_dir.iterdir():
        if not child.exists():
            continue
        try:
            rel = resolved.relative_to(child.resolve())
        except Exception:
            continue
        return str(Path("input") / child.name / rel)
    return None


def generated_public_relative(
    candidate: Path, generated_output_dir: Path | None
) -> str | None:
    if generated_output_dir is None:
        return None
    try:
        rel = candidate.resolve().relative_to(
            (generated_output_dir / "public").resolve()
        )
    except Exception:
        return None
    return str(Path("input") / "public" / rel)


def workspace_relative_input_path(
    value: Any,
    input_dir: Path,
    *,
    base_dir: Path | None = None,
    generated_output_dir: Path | None = None,
) -> str | None:
    """Resolve arbitrary builder paths to the only agent-visible path form."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not looks_like_path(text):
        return None

    raw = Path(text)
    if raw.parts and raw.parts[0] == "input":
        return str(raw)

    candidates = [raw] if raw.is_absolute() else [input_dir / raw]
    if not raw.is_absolute():
        if base_dir is not None:
            candidates.append(base_dir / raw)
        if generated_output_dir is not None:
            candidates.append(generated_output_dir / raw)

    for candidate in candidates:
        if not candidate.exists():
            continue
        replacement = input_relative(candidate, input_dir)
        if replacement is not None:
            return replacement
        replacement = generated_public_relative(candidate, generated_output_dir)
        if replacement is not None:
            return replacement
    return None


def evaluator_visible_path(value: Any, input_dir: Path, base_dir: Path) -> str | None:
    """Resolve private answer file references for evaluator-only use."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not looks_like_path(text):
        return None
    raw = Path(text)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [input_dir / raw, base_dir / raw, base_dir.parent / raw]
        if raw.parts and raw.parts[0] == "input":
            candidates.append(input_dir.parent / raw)
            if len(raw.parts) > 1:
                candidates.append(input_dir / Path(*raw.parts[1:]))
    for candidate in candidates:
        if candidate.exists():
            try:
                return str(candidate.resolve())
            except Exception:
                return str(candidate)
    return None
