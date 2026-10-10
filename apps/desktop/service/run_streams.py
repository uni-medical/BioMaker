"""Bounded, observed run output. These readers never infer run progress.

Only registered paths are opened, with pinned directory descriptors. Cursors
are authenticated, run-specific byte positions, not client filesystem paths.
They retain small continuity anchors; they are not a hash of the entire log.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Iterator
import weakref

from artifact_preview import (
    ArtifactPreviewError, _authorized_directory, _check_directories,
    _identifier, _open_source, _signature,
)

CONSOLE_BYTES = 256 * 1024
CONSOLE_LINES = 400
ACTIVITY_BYTES = 512 * 1024
ACTIVITY_RECORDS = 200
MAX_CONSOLE_DOWNLOAD_BYTES = 256 * 1024 * 1024
_ANCHOR_BYTES = 128
_MESSAGES = {
    "invalid_request": (422, "The output request is invalid."),
    "unavailable": (404, "The output could not be read. Refresh the experiment and try again."),
    "changed": (409, "The output changed. Return to the latest output and try again."),
    "too_large": (413, "The console log is too large to export from the app."),
}


class StreamError(ValueError):
    def __init__(self, code: str):
        self.code = code
        self.status_code, message = _MESSAGES[code]
        super().__init__(f"[stream:{code}] {message}")


def _identity(value: os.stat_result) -> tuple[Any, ...]:
    return (value.st_dev, value.st_ino, value.st_mode,
            getattr(value, "st_birthtime_ns", getattr(value, "st_birthtime", None)))


def _pread(descriptor: int, start: int, length: int) -> bytes:
    """Read a bounded byte range; never follow a growing EOF."""
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = os.pread(descriptor, min(64 * 1024, remaining), start + length - remaining)
        if not chunk:
            raise StreamError("changed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass
class _Source:
    path: Path
    directories: list[tuple[Path, tuple[int, ...]]]
    descriptor: int
    initial: os.stat_result
    source_id: str
    _cleanup: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Also closes an unconsumed response if a client disconnects before the
        # iterator starts and the framework does not reach background cleanup.
        self._cleanup = weakref.finalize(self, os.close, self.descriptor)

    def close(self) -> None:
        if self.descriptor >= 0:
            self.descriptor = -1
            self._cleanup()

    def validate(self) -> None:
        try:
            current = os.fstat(self.descriptor)
            named = self.path.lstat()
            for value in (current, named):
                if _identity(value) != _identity(self.initial) or value.st_size < self.initial.st_size:
                    raise StreamError("changed")
                # An append is expected; a same-size rewrite is not.
                if value.st_size == self.initial.st_size and _signature(value) != _signature(self.initial):
                    raise StreamError("changed")
            _check_directories(self.directories)
        except (OSError, ArtifactPreviewError):
            raise StreamError("changed") from None


@dataclass
class ConsoleDownload:
    source: _Source
    size: int
    sha256: str

    def close(self) -> None:
        self.source.close()

    def chunks(self) -> Iterator[bytes]:
        try:
            position = 0
            while position < self.size:
                self.source.validate()
                chunk = _pread(self.source.descriptor, position, min(64 * 1024, self.size - position))
                position += len(chunk)
                yield chunk
            self.source.validate()
        finally:
            self.close()


def _float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite")
    return result


def _constant(_value: str) -> None:
    raise ValueError("nonfinite")


class RunStreams:
    def __init__(self, logs_root: Path, state_dir: Path, store: Any):
        self.logs_root, self.state_dir, self.store = Path(logs_root), Path(state_dir), store
        self._secret = secrets.token_bytes(32)

    def _digest(self, value: Any) -> str:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        return hmac.new(self._secret, raw, hashlib.sha256).hexdigest()

    def _open(self, run_id: str, kind: str) -> _Source | None:
        record = self.store.get_record("runs", run_id)
        if not isinstance(record, dict):
            raise StreamError("unavailable")
        if kind == "activity":
            raw_directory = record.get("log_dir")
            if not raw_directory:
                return None
            root = self.logs_root
            name = "events.jsonl"
        else:
            raw = record.get("console_log")
            if not raw:
                return None
            if not isinstance(raw, str) or not Path(raw).is_absolute() or ".." in Path(raw).parts:
                raise StreamError("unavailable")
            path = Path(raw)
            name = path.name
            if path.parent == self.state_dir / "console":
                root, raw_directory = self.state_dir, str(path.parent)
            elif isinstance(record.get("log_dir"), str) and path.parent == Path(record["log_dir"]):
                root, raw_directory = self.logs_root, record["log_dir"]
            else:
                raise StreamError("unavailable")
        try:
            directory, identities = _authorized_directory(root, raw_directory)
            path = directory / name
            before = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(before.st_mode):
            raise StreamError("unavailable")
        descriptor = _open_source(path, identities)
        source = None
        try:
            current = os.fstat(descriptor)
            if not stat.S_ISREG(current.st_mode) or _identity(before) != _identity(current):
                raise StreamError("changed")
            source_id = self._digest([run_id, kind, _identity(current),
                                      [(str(path), identity) for path, identity in identities]])
            source = _Source(path, identities, descriptor, current, source_id)
            source.validate()
            return source
        except BaseException:
            if source is not None:
                source.close()
            else:
                os.close(descriptor)
            raise

    def _cursor(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        signature = hmac.new(self._secret, raw, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(signature + raw).decode().rstrip("=")

    def _parse_cursor(self, cursor: str | None) -> dict[str, Any] | None:
        if cursor is None:
            return None
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048:
            raise StreamError("invalid_request")
        try:
            encoded = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
            signature, raw = encoded[:32], encoded[32:]
            if not hmac.compare_digest(signature, hmac.new(self._secret, raw, hashlib.sha256).digest()):
                raise StreamError("changed")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except StreamError:
            raise
        except (ValueError, UnicodeError):
            raise StreamError("invalid_request") from None

    def _anchors(self, source: _Source, offset: int, size: int) -> list[dict[str, Any]]:
        ranges = {(0, min(size, _ANCHOR_BYTES)),
                  (max(0, size - _ANCHOR_BYTES), min(size, _ANCHOR_BYTES)),
                  (max(0, offset - _ANCHOR_BYTES), min(size, offset + _ANCHOR_BYTES) - max(0, offset - _ANCHOR_BYTES))}
        return [{"start": start, "length": length,
                 "sha256": hashlib.sha256(_pread(source.descriptor, start, length)).hexdigest()}
                for start, length in sorted(ranges)]

    def _end(self, source: _Source, run_id: str, kind: str, cursor: dict[str, Any] | None) -> int:
        if cursor is None:
            return source.initial.st_size
        if (cursor.get("run") != run_id or cursor.get("kind") != kind
                or cursor.get("source") != source.source_id
                or source.initial.st_size < cursor["size"]):
            raise StreamError("changed")
        if (source.initial.st_size == cursor["size"]
                and (source.initial.st_mtime_ns != cursor["mtime_ns"]
                     or source.initial.st_ctime_ns != cursor["ctime_ns"])):
            raise StreamError("changed")
        for anchor in cursor["anchors"]:
            raw = _pread(source.descriptor, anchor["start"], anchor["length"])
            if hashlib.sha256(raw).hexdigest() != anchor["sha256"]:
                raise StreamError("changed")
        return cursor["offset"]

    def _missing(self, kind: str) -> dict[str, Any]:
        return {"status": "missing", "source_id": None, "revision": self._digest([kind, "missing"]),
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "start": 0, "end": 0, "source_size": 0, "has_older": False,
                "older_cursor": None, "partial_first_line": False, "invalid_utf8": False,
                **({"text": ""} if kind == "console" else {"events": [], "skipped_records": 0})}

    def window(self, run_id: str, kind: str, before: str | None = None) -> dict[str, Any]:
        if not _identifier(run_id, 180) or kind not in {"console", "activity"}:
            raise StreamError("invalid_request")
        cursor = self._parse_cursor(before)
        source = None
        try:
            source = self._open(run_id, kind)
            if source is None:
                if cursor is not None:
                    raise StreamError("changed")
                return self._missing(kind)
            end = self._end(source, run_id, kind, cursor)
            limit = CONSOLE_BYTES if kind == "console" else ACTIVITY_BYTES
            read_start = max(0, end - limit)
            raw = _pread(source.descriptor, read_start, end - read_start)
            preceding = _pread(source.descriptor, read_start - 1, 1) if read_start else b"\n"
            first_partial = read_start > 0 and preceding != b"\n"
            if kind == "console" and preceding == b"\r" and not raw.startswith(b"\n"):
                # Console splitlines accepts a standalone CR, but starting on
                # the LF half of CRLF still means the first line is partial.
                first_partial = False
            start = read_start
            if kind == "console":
                pieces = raw.splitlines(keepends=True)
                selected = b"".join(pieces[-CONSOLE_LINES:])
                start = end - len(selected)
                content = {"text": selected.decode("utf-8", errors="replace")}
            else:
                # Newline is the engine's record commit marker. An unfinished
                # trailing JSON object stays excluded until its newline arrives.
                pieces = raw.split(b"\n")
                offset = end
                events = []
                skipped = 0
                for index in range(len(pieces) - 1, -1, -1):
                    piece = pieces[index]
                    length = len(piece) + (1 if index < len(pieces) - 1 else 0)
                    offset -= length
                    if not piece.strip():
                        continue
                    if (index == 0 and first_partial) or index == len(pieces) - 1:
                        skipped += 1
                        continue
                    try:
                        data = json.loads(piece.decode("utf-8"), parse_float=_float, parse_constant=_constant)
                        if not isinstance(data, dict):
                            raise ValueError()
                    except (ValueError, UnicodeError, RecursionError):
                        skipped += 1
                        continue
                    events.append({"id": f"{source.source_id}:{offset:x}", "data": data})
                    if len(events) == ACTIVITY_RECORDS:
                        start = offset
                        break
                selected = raw[start - read_start:]
                content = {"events": list(reversed(events)), "skipped_records": skipped}
            try:
                selected.decode("utf-8", errors="strict")
                invalid_utf8 = False
            except UnicodeDecodeError:
                invalid_utf8 = True
            # Re-read only this bounded region. Ordinary appends are allowed,
            # but output cannot be accepted from bytes that changed mid-read.
            if _pread(source.descriptor, read_start, len(raw)) != raw:
                raise StreamError("changed")
            has_older = start > 0
            older = self._cursor({"run": run_id, "kind": kind, "source": source.source_id,
                                  "offset": start, "size": source.initial.st_size,
                                  "mtime_ns": source.initial.st_mtime_ns,
                                  "ctime_ns": source.initial.st_ctime_ns,
                                  "anchors": self._anchors(source, start, source.initial.st_size)}) if has_older else None
            source.validate()
            return {"status": "ready", "source_id": source.source_id,
                    "revision": self._digest([source.source_id, start, end, source.initial.st_size,
                                              source.initial.st_mtime_ns, hashlib.sha256(selected).hexdigest()]),
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "start": start, "end": end, "source_size": source.initial.st_size,
                    "has_older": has_older, "older_cursor": older,
                    "partial_first_line": first_partial and start == read_start,
                    "invalid_utf8": invalid_utf8, **content}
        except StreamError:
            raise
        except ArtifactPreviewError as exc:
            raise StreamError("changed" if exc.code == "changed" else "unavailable") from None
        except Exception:
            raise StreamError("unavailable") from None
        finally:
            if source is not None:
                source.close()

    def download(self, run_id: str) -> ConsoleDownload:
        if not _identifier(run_id, 180):
            raise StreamError("invalid_request")
        source = None
        try:
            source = self._open(run_id, "console")
            if source is None:
                raise StreamError("unavailable")
            size = source.initial.st_size
            if size > MAX_CONSOLE_DOWNLOAD_BYTES:
                raise StreamError("too_large")
            digest = hashlib.sha256()
            position = 0
            while position < size:
                raw = _pread(source.descriptor, position, min(64 * 1024, size - position))
                digest.update(raw)
                position += len(raw)
            source.validate()
            result = ConsoleDownload(source, size, digest.hexdigest())
            source = None  # The response iterator/background cleanup owns it.
            return result
        except StreamError:
            raise
        except ArtifactPreviewError as exc:
            raise StreamError("changed" if exc.code == "changed" else "unavailable") from None
        except Exception:
            raise StreamError("unavailable") from None
        finally:
            if source is not None:
                source.close()
