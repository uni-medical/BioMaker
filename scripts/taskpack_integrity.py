"""Content checks for prepared data and sealed split packages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

INTEGRITY_FILE = "artifact_integrity.json"


def _file_hashes(root: Path, *, package: bool = False) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if package and relative == INTEGRITY_FILE:
            continue
        if path.is_symlink():
            raise RuntimeError(f"TaskPack integrity check rejects symlink: {path}")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative] = digest.hexdigest()
    return hashes


def write_package_integrity(package_dir: Path, data_dir: Path) -> None:
    """Record only relative paths, keeping the inventory outside public data."""

    payload = {
        "schema_version": 1,
        "data": _file_hashes(data_dir),
        "package": _file_hashes(package_dir, package=True),
    }
    (package_dir / INTEGRITY_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_package_integrity(
    package_dir: Path,
    data_dir: Path,
    *,
    validated_data_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Verify content before reuse; share data hashes across identical package copies."""

    path = package_dir / INTEGRITY_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"TaskPack integrity manifest is missing or unreadable: {path}. "
            "Rerun preparation with --replace to rebuild the package."
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported TaskPack integrity manifest: {path}")
    data_hashes = (
        _file_hashes(data_dir)
        if validated_data_hashes is None
        else validated_data_hashes
    )
    for kind, actual in (
        ("data", data_hashes),
        ("package", _file_hashes(package_dir, package=True)),
    ):
        expected = payload.get(kind)
        if not isinstance(expected, dict):
            raise RuntimeError(f"Invalid {kind} inventory in {path}")
        mismatches = sorted(
            name
            for name in set(expected) | set(actual)
            if expected.get(name) != actual.get(name)
        )
        if mismatches:
            raise RuntimeError(
                f"TaskPack integrity mismatch in {kind}: {', '.join(mismatches[:5])}. "
                "Rerun preparation with --replace to rebuild the package."
            )
    return data_hashes
