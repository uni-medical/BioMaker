"""Portable, content-verified split packages supplied explicitly by a task.

The integrity inventory is an identity check, not a signature or a replacement
for the current split verifier. Packages and data remain outside the code repo.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from .split_manifest import load_split_manifest, split_file_reference_roles, file_columns_for_frame

INTEGRITY_NAME = "package_integrity.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path, *, package: bool) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"Missing {'package' if package else 'source'} directory: {root}")
    result: dict[str, str] = {}
    visited: set[Path] = set()
    def fail_walk(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=not package, onerror=fail_walk):
        current = Path(directory)
        resolved = current.resolve()
        if resolved in visited:
            raise ValueError(f"Repeated or cyclic directory in inventory: {current}")
        visited.add(resolved)
        for name in dirs + files:
            child = current / name
            if package and child.is_symlink():
                raise ValueError(f"Split packages must contain regular files, not symlinks: {child}")
        for name in sorted(files):
            path = current / name
            relative = path.relative_to(root).as_posix()
            if package and relative == INTEGRITY_NAME:
                continue
            if not path.is_file():
                raise ValueError(f"Non-regular inventory entry: {relative}")
            result[relative] = _sha256(path)
    return dict(sorted(result.items()))


def _check_declared_paths(package: Path) -> None:
    manifest = load_split_manifest(package)
    hidden = json.loads((package / "private_gt/private_gt.json").read_text())
    errors: list[str] = []
    roles = split_file_reference_roles(manifest, hidden, errors)
    if roles["inputs"] is None or any(
        roles["answer_files"].get(role) is None for role in hidden
    ):
        errors.append("Verified packages must declare input_file_reference_columns and each hidden answer's file_reference_columns (use [] for inline values).")
    for sample in manifest.get("sample_universe", []) or []:
        if not isinstance(sample, dict):
            continue  # The structural verifier reports malformed entries.
        for value in sample.get("source_paths", []) or []:
            if not isinstance(value, str):
                errors.append("sample_universe source_paths must be strings.")
                continue
            rel = PurePosixPath(value)
            if (not value or rel.is_absolute() or ".." in rel.parts or "\\" in value or ":" in value):
                errors.append("sample_universe source_paths must be logical relative paths.")
    for private, directory, filenames in (
        (False, "biomaker_splits", {"train": "agent_train.csv", "score_val": "score_val.csv", "test": "agent_test.csv"}),
        (True, "private_gt", {"score_val": "score_val.csv", "test": "test.csv"}),
    ):
        for role, name in filenames.items():
            path = package / directory / name
            if not path.is_file():
                continue
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            required, legacy = file_columns_for_frame(roles, frame.columns, role=role, private=private)
            if legacy:
                errors.append(f"{directory}/{name} has undeclared file encodings: {sorted(legacy)}")
            for column in required & set(frame.columns):
                for value in frame[column]:
                    rel = PurePosixPath(value)
                    if (not value or rel.is_absolute() or ".." in rel.parts
                            or "\\" in value or ":" in value
                            or not rel.parts or rel.parts[0] not in {"input", "public"}):
                        errors.append(f"{directory}/{name}:{column} requires logical input/... or public/... paths.")
                        break
    if errors:
        raise ValueError("Invalid portable split package: " + "; ".join(errors))


def seal_split_package(package_dir: Path | str, source_input_dir: Path | str, *, task_text: str) -> Path:
    """Validate and record a package's files, source dataset and exact task text."""
    from .agentic_data_planner import verify_split_package
    package, source = Path(package_dir).resolve(), Path(source_input_dir).resolve()
    if package == source or source in package.parents or package in source.parents:
        raise ValueError("Split package and source data must be separate directories.")
    _inventory(package, package=True)  # Reject aliases before verifier normalization.
    _check_declared_paths(package)
    verified = verify_split_package(package, source_input_dir=source, task_text=task_text)
    if not verified.get("ok"):
        raise ValueError("Split package failed verification: " + "; ".join(verified.get("errors", [])))
    identity = {
        "schema": "biomaker_verified_split_package", "version": 1,
        "task_sha256": hashlib.sha256(task_text.encode()).hexdigest(),
        "source_files": _inventory(source, package=False),
        "package_files": _inventory(package, package=True),
    }
    destination = package / INTEGRITY_NAME
    destination.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n")
    return destination


def verify_package_identity(package_dir: Path | str, source_input_dir: Path | str, *, task_text: str) -> dict[str, Any]:
    """Reject changed inputs before admitting a package or invoking any builder."""
    package, source = Path(package_dir).resolve(), Path(source_input_dir).resolve()
    identity = json.loads((package / INTEGRITY_NAME).read_text())
    if not isinstance(identity, dict) or identity.get("schema") != "biomaker_verified_split_package" or identity.get("version") != 1:
        raise ValueError("Unsupported verified split package integrity schema.")
    if identity.get("task_sha256") != hashlib.sha256(task_text.encode()).hexdigest():
        raise ValueError("Verified split package task identity mismatch.")
    for key, root, is_package in (("package_files", package, True), ("source_files", source, False)):
        actual = _inventory(root, package=is_package)
        if not isinstance(identity.get(key), dict) or identity[key] != actual:
            raise ValueError(f"Verified split package {key} inventory/hash mismatch.")
    _check_declared_paths(package)
    return identity
