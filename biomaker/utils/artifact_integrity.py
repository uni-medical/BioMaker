"""Fingerprint delivered artifacts around trusted evaluator calls."""

from __future__ import annotations

import hashlib
from pathlib import Path


def artifact_snapshot(workspace: Path, explicit_paths=()) -> dict:
    """Include metadata so even an identical-byte rewrite invalidates a result.

    All candidate outputs are included, including referenced masks and model
    dependencies. Input data are included only when explicitly used by scoring.
    This is a mutation detector, not a security sandbox.
    """
    paths = {workspace / "evaluator.py", workspace / "evaluator_prediction_spec.json"}
    working = workspace / "working"
    if working.exists():
        paths.update(path for path in working.rglob("*") if path.is_file() or path.is_symlink())
    paths.update(Path(path) for path in explicit_paths if path is not None)
    snapshot = {}
    for path in sorted(paths, key=str):
        if not path.exists():
            snapshot[str(path)] = None
            continue
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        snapshot[str(path)] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, digest.hexdigest())
    return snapshot
