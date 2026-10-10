"""Read a bounded text snapshot from a registered experiment's artifact.

The result describes the exact UTF-8 bytes read, not a stat-based revision.
No files are changed, and this module never resolves an arbitrary client path.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any

MAX_PREVIEW_BYTES = 500_000
_MESSAGES = {
    "invalid_request": (422, "The artifact request is invalid."),
    "unavailable": (404, "The artifact is unavailable. Refresh the experiment and try again."),
    "changed": (409, "The artifact changed. Preview it again before saving."),
    "too_large": (413, "The artifact is too large to preview. Save it from the file list."),
    "invalid_text": (422, "The artifact is not valid UTF-8 text. Save it from the file list."),
}


class ArtifactPreviewError(ValueError):
    def __init__(self, code: str):
        self.code = code
        self.status_code, message = _MESSAGES[code]
        super().__init__(f"[artifact:{code}] {message}")


def _identifier(value: Any, limit: int) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= limit
            and not value.startswith(".")
            and re.search(r"[\\/\x00-\x1f\x7f]", value) is None)


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _directory_identity(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode):
        raise ArtifactPreviewError("unavailable")
    return value.st_dev, value.st_ino, value.st_mode


def _authorized_directory(logs_root: Path, raw_path: Any) -> tuple[Path, list[tuple[Path, tuple[int, ...]]]]:
    if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
        raise ArtifactPreviewError("unavailable")
    root = Path(logs_root)
    path = Path(raw_path)
    if not root.is_absolute() or ".." in root.parts or ".." in path.parts:
        raise ArtifactPreviewError("unavailable")
    # Never resolve the authorized root or the run: following even an ancestor
    # link here would turn a replaced workspace into a new authorized location.
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ArtifactPreviewError("unavailable") from None
    if not relative.parts:
        raise ArtifactPreviewError("unavailable")
    # Capture the complete absolute ancestry, not just the part below logs_root.
    # _open_source then pins this same chain using descriptor-relative opens.
    directory = Path(path.anchor)
    identities = [(directory, _directory_identity(directory))]
    for part in path.parts[1:]:
        directory /= part
        identities.append((directory, _directory_identity(directory)))
    return directory, identities


def _check_directories(identities: list[tuple[Path, tuple[int, ...]]]) -> None:
    try:
        if any(_directory_identity(path) != identity for path, identity in identities):
            raise ArtifactPreviewError("changed")
    except (OSError, ValueError):
        raise ArtifactPreviewError("changed") from None


def _open_source(path: Path, identities: list[tuple[Path, tuple[int, ...]]]) -> int:
    # macOS/Linux use pinned directory descriptors for the complete absolute
    # ancestry. Other platforms need an equivalent safe-open adapter;
    # they must not silently fall back to following a changed path.
    if os.open not in os.supports_dir_fd or any(not hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK")):
        raise ArtifactPreviewError("unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptors: list[int] = []
    try:
        for index, (directory, identity) in enumerate(identities):
            descriptor = (os.open(directory, directory_flags) if index == 0
                          else os.open(directory.name, directory_flags, dir_fd=descriptors[-1]))
            descriptors.append(descriptor)
            value = os.fstat(descriptor)
            if (value.st_dev, value.st_ino, value.st_mode) != identity:
                raise ArtifactPreviewError("changed")
        return os.open(path.name, flags, dir_fd=descriptors[-1])
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_bytes(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while size <= MAX_PREVIEW_BYTES:
        chunk = os.read(descriptor, min(64 * 1024, MAX_PREVIEW_BYTES + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


def preview_artifact(logs_root: Path, run_store: Any, run_id: str, name: str) -> dict[str, Any]:
    """Return one stable snapshot, or a fixed error without private paths.

    The store must be the desktop app's registered run store. Unknown runs are
    never discovered by scanning disk. A changing file is not retried silently.
    """
    if not _identifier(run_id, 180) or not _identifier(name, 240):
        raise ArtifactPreviewError("invalid_request")
    try:
        run = run_store.get_record("runs", run_id)
        if not isinstance(run, dict):
            raise ArtifactPreviewError("unavailable")
        directory, identities = _authorized_directory(logs_root, run.get("log_dir"))
        path = directory / name
        source = path.lstat()
        if not stat.S_ISREG(source.st_mode):
            raise ArtifactPreviewError("unavailable")
        if source.st_size > MAX_PREVIEW_BYTES:
            raise ArtifactPreviewError("too_large")
        descriptor = _open_source(path, identities)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or _signature(before) != _signature(source):
                raise ArtifactPreviewError("changed")
            raw = _read_bytes(descriptor)
            if len(raw) > MAX_PREVIEW_BYTES:
                raise ArtifactPreviewError("too_large")
            if len(raw) != before.st_size or _signature(os.fstat(descriptor)) != _signature(before):
                raise ArtifactPreviewError("changed")
            try:
                if _signature(path.lstat()) != _signature(before):
                    raise ArtifactPreviewError("changed")
            except OSError:
                raise ArtifactPreviewError("changed") from None
            _check_directories(identities)
        finally:
            os.close(descriptor)
        try:
            content = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise ArtifactPreviewError("invalid_text") from None
        return {"name": name, "content": content, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    except ArtifactPreviewError:
        raise
    except Exception:
        # Store/SQLite and filesystem failures share a nonleaking public error.
        # Cancellation and process-exit exceptions are not intercepted.
        raise ArtifactPreviewError("unavailable") from None
