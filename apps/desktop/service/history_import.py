"""Copy explicitly selected historical records into the desktop's own log root.

This module never opens a source ledger, writes into a source experiment, follows
source symlinks, or launches/controls an experiment. Imports are static records,
not dataset, workspace, checkpoint, or model migrations.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

MAX_RUNS = 100
MAX_FILES = 128
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_BATCH_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 5_000_000
MAX_JOURNAL_VIEW_BYTES = 12 * 1024 * 1024
PREVIEW_SECONDS = 600
MAX_PREVIEWS = 20
SCAN_SECONDS = 15
COPY_CHUNK = 1024 * 1024
DISK_RESERVE = 16 * 1024 * 1024
ALLOWED_FILES = frozenset({
    "run_manifest.json", "journal.json", "results.tsv", "report.md",
    "best_solution.py", "evaluator.py", "events.jsonl", "tree_plot.html",
    "experiment_guide.md", "config.yaml", "run_overview.json",
    "exp_guide.json", "tree_snapshot.html",
})
EVIDENCE_FILES = frozenset({"journal.json", "results.tsv", "report.md", "best_solution.py"})
LIMITS = {
    "max_runs": MAX_RUNS,
    "max_files_per_run": MAX_FILES,
    "max_file_bytes": MAX_FILE_BYTES,
    "max_batch_bytes": MAX_BATCH_BYTES,
}


class HistoryImportError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_id(path: Path) -> str:
    return hashlib.sha256(os.fsencode(path)).hexdigest()


def run_id(identifier: str) -> str:
    return "desktop-import-" + identifier[:32]


def _signature(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _directory_identity(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode) or path.resolve() != path:
        raise HistoryImportError("Source directory changed or is a symbolic link; preview it again.", 409)
    return value.st_dev, value.st_ino, value.st_mtime_ns, value.st_ctime_ns


def _publish_directory(source: Path, destination: Path) -> None:
    """Atomically publish without replacing even an empty concurrent directory."""
    if os.name == "nt":
        # Windows os.rename already fails if the destination exists.
        os.rename(source, destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise OSError("Atomic import publication without replacement is unavailable on this platform.")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _read_regular(path: Path, limit: int, expected: tuple[int, ...] | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise HistoryImportError(f"{path.name} is not a supported regular file or exceeds its size limit.")
        if expected is not None and _signature(before) != expected:
            raise HistoryImportError("Source files changed; preview the experiment again.", 409)
        value = stream.read(limit + 1)
        if len(value) > limit or _signature(os.fstat(stream.fileno())) != _signature(before):
            raise HistoryImportError("Source files changed while being read; preview again.", 409)
        return value


def _json_object(raw: bytes, filename: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HistoryImportError(f"{filename} is not a valid JSON object.") from exc
    if not isinstance(value, dict):
        raise HistoryImportError(f"{filename} must contain a JSON object.")
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 128:
            raise HistoryImportError(f"{filename} exceeds the supported JSON nesting depth.")
        if isinstance(item, float) and not math.isfinite(item):
            # Overflow such as 1e999 is parsed as infinity without invoking
            # json.loads(parse_constant), so inspect all decoded fields too.
            raise HistoryImportError(f"{filename} contains a nonfinite JSON number.")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class Candidate:
    path: Path
    identity: tuple[int, ...]
    files: dict[str, tuple[int, ...]]
    manifest: dict[str, Any]
    public: dict[str, Any]


class HistoryImporter:
    def __init__(self, settings, manager):
        self.settings = settings
        self.manager = manager
        self.jobs_dir = settings.state_dir / "history-imports"
        self.staging_root = settings.logs_root / ".desktop-import-staging"
        self._lock = threading.RLock()
        self._previews: dict[str, tuple[float, dict[str, Candidate]]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def _job_path(self, identifier: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise HistoryImportError("Import job was not found. Preview the source again.", 404)
        return self.jobs_dir / f"{identifier}.json"

    def _save_job(self, job: dict[str, Any]) -> None:
        with self._lock:
            job["updated_at"] = now()
            # Still return an actionable failure when the disk fills after the
            # initial job receipt was created. Restart can settle that receipt.
            self._jobs[job["job_id"]] = copy.deepcopy(job)
            _atomic_json(self._job_path(job["job_id"]), job)

    def get_job(self, identifier: str) -> dict[str, Any]:
        with self._lock:
            path = self._job_path(identifier)
            if identifier in self._jobs:
                return copy.deepcopy(self._jobs[identifier])
            try:
                return _json_object(_read_regular(path, 2_000_000), "Import job")
            except FileNotFoundError as exc:
                raise HistoryImportError("Import job was not found. Preview the source again.", 404) from exc

    def recover(self) -> None:
        """Settle interrupted jobs and recover already-published local copies."""
        for path in self.jobs_dir.glob("*.json"):
            if not re.fullmatch(r"[a-f0-9]{32}", path.stem):
                continue
            try:
                job = self.get_job(path.stem)
                if job.get("status") not in {"queued", "running"}:
                    continue
                job["status"] = "failed"
                job["current_name"] = None
                job.setdefault("errors", []).append({
                    "message": "Import was interrupted when the desktop service stopped. Completed copies are preserved; preview again to continue."
                })
                self._save_job(job)
                temporary = self.staging_root / path.stem
                if temporary.is_dir() and not temporary.is_symlink():
                    shutil.rmtree(temporary)
            except (OSError, ValueError, KeyError, TypeError):
                # A broken job receipt must not prevent normal desktop startup.
                continue
        for path in self.settings.logs_root.glob("desktop-import-*"):
            if not re.fullmatch(r"desktop-import-[a-f0-9]{32}", path.name) or path.is_symlink():
                continue
            try:
                metadata = _json_object(_read_regular(path / "desktop_import.json", 2_000_000), "Import provenance")
                valid = (
                    metadata.get("run_id") == path.name
                    and metadata.get("mode") == "copy"
                    and isinstance(metadata.get("source_id"), str)
                    and re.fullmatch(r"[a-f0-9]{64}", metadata["source_id"])
                    and run_id(metadata["source_id"]) == path.name
                    and all(isinstance(metadata.get(key), str) for key in ("source_path", "name", "imported_at"))
                )
                if valid:
                    self._register(path, metadata)
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def close(self) -> None:
        self._closed.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _existing(self, identifier: str) -> str | None:
        record = self.manager.store.get_record("runs", run_id(identifier))
        if record is not None:
            origin = record.get("history_import") or {}
            if origin.get("source_id") == identifier:
                return str(record["id"])
        return None

    def _scan(self, path: Path, deadline: float) -> Candidate:
        identifier = source_id(path)
        public: dict[str, Any] = {
            "source_id": identifier, "name": path.name, "created_at": None,
            "status": "unknown", "total_bytes": 0, "files": [],
            "capabilities": {key: False for key in ("candidates", "code", "report", "results", "logs")},
            "already_imported": False, "existing_run_id": None,
            "eligible": False, "warnings": [],
        }
        candidate = Candidate(path, (0, 0), {}, {}, public)
        try:
            candidate.identity = _directory_identity(path)
            entries = []
            with os.scandir(path) as listing:
                for entry in listing:
                    if time.monotonic() > deadline:
                        raise HistoryImportError("Preview scanning exceeded 15 seconds. Select one experiment folder.")
                    entries.append(entry)
                    if len(entries) > MAX_FILES:
                        raise HistoryImportError(f"This experiment has more than {MAX_FILES} top-level entries. Select a smaller records folder.")
            skipped = []
            for entry in sorted(entries, key=lambda value: value.name):
                name = entry.name
                value = entry.stat(follow_symlinks=False)
                if name not in ALLOWED_FILES and not (name.endswith(".log") and not name.startswith(".")):
                    skipped.append(name)
                    continue
                if not stat.S_ISREG(value.st_mode):
                    skipped.append(name)
                    if name == "run_manifest.json":
                        raise HistoryImportError("run_manifest.json must be a regular file, not a symbolic link.")
                    continue
                if value.st_size > MAX_FILE_BYTES:
                    raise HistoryImportError(f"{name} exceeds the 64 MiB file limit.")
                candidate.files[name] = _signature(value)
                public["files"].append({"name": name, "size": value.st_size})
                public["total_bytes"] += value.st_size
            if "run_manifest.json" not in candidate.files:
                raise HistoryImportError("No regular run_manifest.json was found in this experiment folder.")
            manifest_raw = _read_regular(path / "run_manifest.json", MAX_MANIFEST_BYTES, candidate.files["run_manifest.json"])
            manifest = _json_object(manifest_raw, "run_manifest.json")
            if not isinstance(manifest.get("data_dir"), str) or not Path(manifest["data_dir"]).name:
                raise HistoryImportError("run_manifest.json requires a nonempty data_dir string.")
            candidate.manifest = manifest
            if not any(candidate.files.get(name, (0, 0, 0, 0))[3] > 0 for name in EVIDENCE_FILES):
                raise HistoryImportError("No recorded candidates, results, report, or solution was found; an empty record cannot be imported.")
            if public["total_bytes"] > MAX_BATCH_BYTES:
                raise HistoryImportError("This experiment exceeds the 256 MiB import limit.")
            public["created_at"] = manifest.get("created_at") if isinstance(manifest.get("created_at"), str) else None
            # Read only a bounded event tail for preview. Full local event parsing
            # is done after a stable copy has been made.
            terminal = None
            if "events.jsonl" in candidate.files:
                descriptor = os.open(path / "events.jsonl", os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
                with os.fdopen(descriptor, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if _signature(info) != candidate.files["events.jsonl"]:
                        raise HistoryImportError("Source files changed during preview; scan again.", 409)
                    stream.seek(max(0, info.st_size - 256_000))
                    lines = stream.read(256_000).splitlines()
                    for line in reversed(lines):
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeError, RecursionError):
                            continue
                        if isinstance(event, dict) and isinstance(event.get("event"), str) and event["event"] in {"run_finished", "run_failed", "run_interrupted"}:
                            terminal = event["event"]
                            break
            public["status"] = {"run_finished": "succeeded", "run_failed": "failed", "run_interrupted": "interrupted"}.get(terminal, "snapshot")
            if terminal is None:
                public["warnings"].append("No completion event was observed. This is a static snapshot; the original experiment may still be running.")
            sizes = {item["name"]: item["size"] for item in public["files"]}
            public["capabilities"] = {
                "candidates": 0 < sizes.get("journal.json", 0) <= MAX_JOURNAL_VIEW_BYTES,
                "code": sizes.get("best_solution.py", 0) > 0 or 0 < sizes.get("journal.json", 0) <= MAX_JOURNAL_VIEW_BYTES,
                "report": 0 < sizes.get("report.md", 0) <= 16 * 1024 * 1024,
                "results": sizes.get("results.tsv", 0) > 0,
                "logs": any(name.endswith(".log") and size > 0 for name, size in sizes.items()),
            }
            if sizes.get("journal.json", 0) > MAX_JOURNAL_VIEW_BYTES:
                public["warnings"].append("journal.json exceeds the 12 MiB candidate viewer limit; it will be available for file export only.")
            if sizes.get("report.md", 0) > 16 * 1024 * 1024:
                public["warnings"].append("report.md exceeds the 16 MiB viewer limit; it will be available for file export only.")
            if skipped:
                public["warnings"].append("Not copied (folders, links, or unsupported files): " + ", ".join(skipped))
            existing = self._existing(identifier)
            public.update(already_imported=bool(existing), existing_run_id=existing, eligible=not bool(existing))
            if (self.settings.logs_root / run_id(identifier)).exists() and not existing:
                raise HistoryImportError("A local import directory already exists but is not registered. Restart BioMaker to recover it; it will not be overwritten.")
        except (OSError, ValueError) as exc:
            public["eligible"] = False
            public["warnings"].append(str(exc))
        return candidate

    def preview(self, raw_path: str) -> dict[str, Any]:
        requested = Path(raw_path).expanduser()
        if not requested.is_absolute() or requested.is_symlink():
            raise HistoryImportError("Choose an absolute, regular experiment or logs folder.")
        try:
            path = requested.resolve(strict=True)
            _directory_identity(path)
        except (OSError, ValueError) as exc:
            raise HistoryImportError("The selected folder is unavailable or is not a directory.") from exc
        # The app's own output is not a new external source.
        if path == self.settings.logs_root.resolve() or self.settings.logs_root.resolve() in path.parents:
            raise HistoryImportError("This folder already belongs to the desktop workspace. Open its existing records instead.")
        deadline = time.monotonic() + SCAN_SECONDS
        candidates: dict[str, Candidate] = {}
        is_run = os.path.lexists(path / "run_manifest.json")
        paths = [path] if is_run else []
        warnings = ["Only listed experiment records are copied. Datasets, checkpoints, workspaces, private ground truth, and subfolders are not migrated."]
        if not is_run:
            with os.scandir(path) as listing:
                entries_seen = 0
                for entry in listing:
                    entries_seen += 1
                    if entries_seen > 1000 or time.monotonic() > deadline:
                        raise HistoryImportError("This folder is too large or slow to scan. Select an individual experiment folder.")
                    if entry.is_dir(follow_symlinks=False) or entry.is_symlink():
                        paths.append(path / entry.name)
                        if len(paths) > MAX_RUNS:
                            raise HistoryImportError("More than 100 child folders were found. Select one experiment folder; no entries were silently omitted.")
        for child in sorted(paths, key=lambda item: item.name):
            if time.monotonic() > deadline:
                raise HistoryImportError("Preview scanning exceeded 15 seconds. Select one experiment folder.")
            candidate = self._scan(child, deadline)
            candidates[candidate.public["source_id"]] = candidate
        if not candidates:
            raise HistoryImportError("No experiment folders were found. Select a folder containing run_manifest.json or its parent logs folder.")
        identifier = uuid.uuid4().hex
        with self._lock:
            current = time.monotonic()
            self._previews = {key: item for key, item in self._previews.items() if item[0] > current}
            while len(self._previews) >= MAX_PREVIEWS:
                self._previews.pop(next(iter(self._previews)))
            self._previews[identifier] = (current + PREVIEW_SECONDS, candidates)
        return {
            "preview_id": identifier, "source_path": str(path), "source_kind": "run" if is_run else "logs",
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=PREVIEW_SECONDS)).isoformat(),
            "limits": dict(LIMITS), "runs": [copy.deepcopy(item.public) for item in candidates.values()], "warnings": warnings,
        }

    def submit(self, preview_id: str, source_ids: list[str]) -> dict[str, Any]:
        with self._lock:
            if self._closed.is_set():
                raise HistoryImportError("The desktop service is stopping. Reopen BioMaker and preview again.", 409)
            if self._thread and self._thread.is_alive():
                raise HistoryImportError("An import is already in progress. Wait for it to finish.", 409)
            preview = self._previews.get(preview_id)
            if preview is None or preview[0] <= time.monotonic():
                raise HistoryImportError("This preview expired. Scan the source folder again.", 409)
            if not source_ids or len(source_ids) > MAX_RUNS or len(source_ids) != len(set(source_ids)):
                raise HistoryImportError("Choose between 1 and 100 distinct preview entries.")
            if any(identifier not in preview[1] for identifier in source_ids):
                raise HistoryImportError("An entry does not belong to this preview. Scan the folder again.", 409)
            candidates = [preview[1][identifier] for identifier in source_ids]
            if any(not item.public["eligible"] and not item.public["already_imported"] for item in candidates):
                raise HistoryImportError("A selected experiment is not eligible. Review the preview warnings.")
            required = sum(item.public["total_bytes"] for item in candidates if not self._existing(item.public["source_id"]))
            if required > MAX_BATCH_BYTES:
                raise HistoryImportError("The selected records exceed 256 MiB. Import a smaller batch.")
            # Keep room for provenance, a preserved source manifest, and SQLite.
            overhead = sum(item.files.get("run_manifest.json", (0, 0, 0, 0))[3] for item in candidates)
            if shutil.disk_usage(self.settings.logs_root).free < required + overhead + DISK_RESERVE:
                raise HistoryImportError("There is not enough free disk space for these records and import metadata.", 409)
            job = {
                "job_id": uuid.uuid4().hex, "status": "queued", "completed_runs": 0,
                "total_runs": len(candidates), "current_name": None, "imported": [], "skipped": [], "errors": [],
                "created_at": now(), "updated_at": now(),
            }
            self._save_job(job)
            result = copy.deepcopy(job)
            self._thread = threading.Thread(target=self._run_job, args=(job, candidates), daemon=True, name="biomaker-history-import")
            self._thread.start()
            return result

    def _check_source(self, item: Candidate) -> None:
        if _directory_identity(item.path) != item.identity:
            raise HistoryImportError("Source directory was replaced; preview again.", 409)
        for name, expected in item.files.items():
            if _signature((item.path / name).lstat()) != expected:
                raise HistoryImportError(f"Source file {name} changed; preview again.", 409)

    def _copy_file(self, item: Candidate, name: str, destination: Path) -> str:
        descriptor = os.open(item.path / name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as source:
            if _signature(os.fstat(source.fileno())) != item.files[name]:
                raise HistoryImportError(f"Source file {name} changed; preview again.", 409)
            copied = 0
            with destination.open("xb") as target:
                while chunk := source.read(COPY_CHUNK):
                    if self._closed.is_set():
                        raise HistoryImportError("Import interrupted because the desktop service is closing.", 409)
                    copied += len(chunk)
                    if copied > item.files[name][3]:
                        raise HistoryImportError(f"Source file {name} grew during copying; preview again.", 409)
                    target.write(chunk)
                    digest.update(chunk)
                target.flush()
                os.fsync(target.fileno())
            if copied != item.files[name][3] or _signature(os.fstat(source.fileno())) != item.files[name]:
                raise HistoryImportError(f"Source file {name} changed during copying; preview again.", 409)
        return digest.hexdigest()

    def _register(self, destination: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        from webui.run_catalog import run_from_log_dir

        record = run_from_log_dir(self.settings, destination)
        if record is None:
            raise HistoryImportError("The copied experiment could not be read by BioMaker; it was not imported.")
        record.update({
            "managed": False, "read_only": True, "source": "desktop_import",
            "pid": None, "pgid": None, "process_started_at": None, "launch_token": None,
            "workspace_dir": None, "workspaces_root": None, "command": [],
            "allowed_actions": [], "controllable": False,
            "history_import": {key: metadata[key] for key in ("mode", "source_id", "source_path", "name", "imported_at")},
        })
        if record.get("terminal_event") is None:
            record["status_reason"] = "Imported static snapshot without a completion event. The original experiment may still be running and is not controlled by this app."
            record["error"] = None
        existing = self.manager.store.get_record("runs", record["id"])
        if existing is not None:
            # The normal catalog may have discovered this completed local copy
            # during restart. Only our own provenance-bearing directory is used.
            if existing.get("managed", True):
                raise HistoryImportError("A managed run already uses this import ID; it will not be overwritten.", 409)
            return self.manager.store.update_record("runs", record["id"], record)
        return self.manager.import_log_record(record)

    def _validate_staging(self, staging: Path) -> None:
        """Prove the copy is safe for the unmodified WebUI startup and APIs.

        The WebUI imports log folders before the desktop recovery hook runs.
        Never publish an event stream that can poison its JSON ledger or HTTP
        responses. Journal NaN values remain untouched: its viewer uses JSON5.
        """
        from webui.run_catalog import run_from_log_dir

        events = staging / "events.jsonl"
        if events.is_file():
            with events.open("rb") as stream:
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            event = _json_object(line, f"events.jsonl line {line_number}")
                            if not isinstance(event.get("event"), str):
                                raise HistoryImportError("Every event requires a string event name.")
                        except HistoryImportError as exc:
                            raise HistoryImportError(
                                f"events.jsonl line {line_number} is not a valid finite JSON event. The records were not published; repair the source or select another experiment."
                            ) from exc
        try:
            # This changes only the reader's boundary; it creates no directories
            # and does not register staging records in the durable ledger.
            staging_settings = replace(self.settings, logs_root=staging.parent)
            record = run_from_log_dir(staging_settings, staging)
            if record is None:
                raise HistoryImportError("The staged experiment is not readable by BioMaker.")
            json.dumps(record, allow_nan=False)
        except (ValueError, TypeError, RecursionError) as exc:
            raise HistoryImportError("The copied experiment contains metadata that BioMaker cannot safely display. No records were published.") from exc

    def _copy_run(self, job_id: str, item: Candidate) -> str:
        identifier = item.public["source_id"]
        destination = self.settings.logs_root / run_id(identifier)
        if destination.exists() or destination.is_symlink():
            raise HistoryImportError("The destination already exists and will not be overwritten. Preview again.", 409)
        self._check_source(item)
        job_root = self.staging_root / job_id
        job_root.mkdir(exist_ok=True)
        staging = job_root / run_id(identifier)
        staging.mkdir()
        published = False
        try:
            hashes = {}
            for name in item.files:
                target_name = "source_run_manifest.json" if name == "run_manifest.json" else name
                hashes[name] = self._copy_file(item, name, staging / target_name)
            self._check_source(item)
            # Preserve the original bytes separately; the viewer manifest cannot
            # retain process ownership or point at the original working files.
            manifest = copy.deepcopy(item.manifest)
            for key in ("pid", "pgid", "process_started_at", "launch_token", "workspace_dir"):
                manifest[key] = None
            _atomic_json(staging / "run_manifest.json", manifest)
            metadata = {
                "version": 1, "mode": "copy", "run_id": destination.name,
                "source_id": identifier, "source_path": str(item.path), "name": item.path.name,
                "imported_at": now(), "files": item.public["files"], "source_sha256": hashes,
                "notice": "Static experiment records only; datasets, workspaces, model checkpoints and private ground truth were not copied.",
            }
            _atomic_json(staging / "desktop_import.json", metadata)
            self._validate_staging(staging)
            if self._closed.is_set():
                raise HistoryImportError("Import interrupted because the desktop service is closing.", 409)
            if destination.exists() or destination.is_symlink():
                raise HistoryImportError("The destination appeared during copying; it will not be overwritten.", 409)
            _publish_directory(staging, destination)
            published = True
            self._register(destination, metadata)
            return destination.name
        except Exception:
            if published:
                # A complete publication is recoverable on next startup; do not
                # delete evidence after a transient ledger write error.
                raise HistoryImportError("A complete local copy was saved but registration failed. Restart BioMaker to recover it; the source was not changed.")
            raise
        finally:
            if not published and staging.is_dir():
                shutil.rmtree(staging)

    def _run_job(self, job: dict[str, Any], candidates: list[Candidate]) -> None:
        try:
            job["status"] = "running"
            with self._lock:
                self._save_job(job)
            for item in candidates:
                identifier = item.public["source_id"]
                name = item.public["name"]
                if self._closed.is_set():
                    job["errors"].append({"message": "Import interrupted because the desktop service is closing. Completed copies are preserved."})
                    break
                job["current_name"] = name
                with self._lock:
                    self._save_job(job)
                try:
                    existing = self._existing(identifier)
                    if existing:
                        job["skipped"].append({"source_id": identifier, "name": name, "reason": "Already imported; the existing copy was preserved.", "run_id": existing})
                    else:
                        imported_id = self._copy_run(job["job_id"], item)
                        job["imported"].append({"source_id": identifier, "run_id": imported_id, "name": name})
                except (OSError, ValueError) as exc:
                    job["errors"].append({"source_id": identifier, "name": name, "message": str(exc)})
                job["completed_runs"] += 1
                with self._lock:
                    self._save_job(job)
            job["status"] = "completed" if job["completed_runs"] == job["total_runs"] and (job["imported"] or job["skipped"]) else "failed"
        except Exception as exc:
            job["status"] = "failed"
            job["errors"].append({"message": str(exc)})
        finally:
            job["current_name"] = None
            with self._lock:
                try:
                    self._save_job(job)
                except OSError:
                    # The last durable 'running' receipt is converted to an
                    # interrupted failure at restart if the disk became full.
                    pass
            job_root = self.staging_root / job["job_id"]
            if job_root.is_dir() and not job_root.is_symlink():
                shutil.rmtree(job_root, ignore_errors=True)
