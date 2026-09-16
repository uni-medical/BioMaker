#!/usr/bin/env python3
"""Run the frozen Figure 2 MedMNIST suite as a durable, single-run FIFO.

The queue owns orchestration only. Dataset preparation may run concurrently in a
separate process and advertise completed task directories through a suite manifest.
At most one ``python -m biomaker.run`` child is started at a time. Every dataset is
gated by a four-step canary before its 35-step run, and every child must pass a
strict lifecycle/artifact audit before the next stage is eligible.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DEFAULT_REGISTRY = WORKSPACE_ROOT / "experiments" / "fig2_medmnist" / "tasks.json"
DEFAULT_TARGET_ROOT = WORKSPACE_ROOT
TERMINAL_EVENTS = {"run_finished", "run_failed", "run_interrupted"}
READY_STATUSES = {"ready", "complete", "completed", "prepared", "ok", "success"}
ACTIVE_FLAGS = {
    "pathmnist",
    "chestmnist",
    "dermamnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "nodulemnist3d",
    "adrenalmnist3d",
    "fracturemnist3d",
    "vesselmnist3d",
    "synapsemnist3d",
}
FROZEN_TASK_ORDER = (
    "pneumoniamnist",
    "breastmnist",
    "adrenalmnist3d",
    "vesselmnist3d",
    "fracturemnist3d",
    "nodulemnist3d",
    "synapsemnist3d",
    "dermamnist",
    "retinamnist",
    "bloodmnist",
    "tissuemnist",
    "chestmnist",
    "pathmnist",
)
GPU_UUID_RE = re.compile(r"GPU-[0-9A-Za-z-]+")
STOP_REQUESTED = threading.Event()
CURRENT_CHILD: subprocess.Popen[Any] | None = None
INCOMPLETE_STAGE_STATUSES = {"starting", "running"}
RECOVERY_BLOCKED_STATUS = "recovery_blocked"
PRIVATE_EVALUATION_BLOCKED_STATUS = "private_evaluation_blocked"
PRIVATE_EVALUATION_ARTIFACT_NAMES = {
    "final_retrain_result.json",
    "final_submission.csv",
    "private_metrics.json",
}
PRIVATE_EVALUATION_AUDIT_MARKERS = (
    "consumed final retrain/private-test evaluation",
    "forbidden private/final artifacts",
    "private-evaluation",
    "private evaluation",
    "final_retrain_started",
    "final_retrain_finished",
    "final_retrain_result.json",
    "final_submission.csv",
    "private_metrics.json",
)


class QueueConfigurationError(RuntimeError):
    """Raised when a frozen queue invariant is not satisfied."""


@dataclass(frozen=True)
class DatasetCheck:
    ready: bool
    reason: str
    path: Path
    row_counts: dict[str, int] | None = None


@dataclass(frozen=True)
class RunAudit:
    ok: bool
    errors: tuple[str, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class QueuePaths:
    runtime_root: Path
    data_root: Path
    logs_root: Path
    workspaces_root: Path
    resource_root: Path
    state_dir: Path
    python: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path.resolve())}
    if not path.is_file():
        return {**record, "present": False}
    return {
        **record,
        "present": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_create_json_once(path: Path, payload: Any) -> None:
    """Publish a complete JSON ledger exactly once without replacing a claim."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard link is an atomic no-replace publication on the same filesystem.
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_source_tree(runtime_root: Path) -> dict[str, Any]:
    """Hash the exact Python/config inputs used by the frozen runtime."""

    candidates: list[Path] = []
    for relative in ("biomaker", "pyproject.toml", "uv.lock"):
        path = runtime_root / relative
        if path.is_dir():
            candidates.extend(item for item in path.rglob("*") if item.is_file() or item.is_symlink())
        elif path.is_file() or path.is_symlink():
            candidates.append(path)
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: str(item.relative_to(runtime_root))):
        relative = path.relative_to(runtime_root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            target = os.readlink(path)
            file_hash = hashlib.sha256(("symlink\0" + target).encode()).hexdigest()
            size = len(target.encode())
            kind = "symlink"
        else:
            file_hash = sha256_file(path)
            size = path.stat().st_size
            kind = "file"
        record = {
            "path": relative.as_posix(),
            "kind": kind,
            "size_bytes": size,
            "sha256": file_hash,
        }
        records.append(record)
        digest.update(
            f"{record['path']}\0{kind}\0{size}\0{file_hash}\n".encode("utf-8")
        )
    return {
        "algorithm": "sha256(path\\0kind\\0size\\0sha256\\n)",
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def _git_output(runtime_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(runtime_root), *args],
        capture_output=True,
        check=False,
        timeout=60,
    )


def _git_source_state(runtime_root: Path, patch_path: Path) -> dict[str, Any]:
    """Capture Git provenance or reuse provenance embedded by the snapshot launcher."""

    embedded_path = runtime_root / ".biomaker_source_provenance.json"
    if embedded_path.is_file():
        embedded = read_json(embedded_path)
        if not isinstance(embedded, dict):
            raise QueueConfigurationError(
                f"embedded source provenance is not an object: {embedded_path}"
            )
        embedded_patch = runtime_root / str(embedded.get("patch_file", "source.patch"))
        if embedded_patch.is_file():
            patch_bytes = embedded_patch.read_bytes()
            _atomic_write_bytes(patch_path, patch_bytes)
            expected = embedded.get("git_binary_diff_sha256")
            actual = hashlib.sha256(patch_bytes).hexdigest()
            if expected not in (None, actual):
                raise QueueConfigurationError(
                    "embedded Git patch digest does not match its provenance"
                )
        else:
            _atomic_write_bytes(patch_path, b"")
            actual = hashlib.sha256(b"").hexdigest()
        return {
            **embedded,
            "source": "embedded_snapshot_provenance",
            "git_binary_diff_sha256": actual,
            "captured_patch": str(patch_path),
        }

    head = _git_output(runtime_root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise QueueConfigurationError(
            "runtime is not a Git checkout and has no .biomaker_source_provenance.json"
        )
    branch = _git_output(runtime_root, "branch", "--show-current")
    status = _git_output(
        runtime_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    patch = _git_output(
        runtime_root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
        ":(exclude).env",
        ":(exclude)**/.env",
    )
    if any(result.returncode != 0 for result in (branch, status, patch)):
        raise QueueConfigurationError("failed to capture complete Git source provenance")
    _atomic_write_bytes(patch_path, patch.stdout)
    status_text = status.stdout.decode("utf-8", errors="replace")
    return {
        "source": "runtime_git_checkout",
        "git_head": head.stdout.decode().strip(),
        "git_branch": branch.stdout.decode().strip() or None,
        "git_dirty": bool(status_text.strip()),
        "git_status_porcelain": status_text.splitlines(),
        "git_binary_diff_sha256": hashlib.sha256(patch.stdout).hexdigest(),
        "git_binary_diff_size_bytes": len(patch.stdout),
        "captured_patch": str(patch_path),
        "excluded_from_patch": [".env", "**/.env"],
    }


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str) + "\n")
        handle.flush()


def emit(state_dir: Path, event: str, **payload: Any) -> None:
    record = {"ts": utc_now(), "event": event, **payload}
    append_jsonl(state_dir / "queue_events.jsonl", record)
    details = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"[{record['ts']}] {event}{(' ' + details) if details else ''}", flush=True)


def load_registry(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise QueueConfigurationError(f"cannot read task registry {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise QueueConfigurationError("task registry must be a schema_version=1 object")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, dict) for task in tasks):
        raise QueueConfigurationError("task registry 'tasks' must be a list of objects")

    ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) or not task_id for task_id in ids):
        raise QueueConfigurationError("every registry task requires a non-empty task_id")
    if len(ids) != len(set(ids)):
        raise QueueConfigurationError("registry task_id values must be unique")

    active = [task for task in tasks if task.get("status") == "active"]
    active_flags = {task.get("medmnist_flag") for task in active}
    if len(active) != 13 or active_flags != ACTIVE_FLAGS:
        raise QueueConfigurationError(
            "frozen Figure 2 registry must contain exactly the approved 13 active MedMNIST flags"
        )
    ordered_ids = tuple(
        str(task["task_id"])
        for task in sorted(active, key=lambda task: (int(task["order"]), str(task["task_id"])))
    )
    if ordered_ids != FROZEN_TASK_ORDER:
        raise QueueConfigurationError(
            "active Figure 2 tasks do not match the frozen FIFO order"
        )

    statuses = {task.get("task_id"): task.get("status") for task in tasks}
    expected_boundaries = {
        "tdsc-abus2023": "hold",
        "aapm-low-dose-ct": "excluded",
        "calgary-campinas": "excluded",
    }
    for task_id, expected_status in expected_boundaries.items():
        if statuses.get(task_id) != expected_status:
            raise QueueConfigurationError(
                f"{task_id} must remain {expected_status!r} in the frozen registry"
            )

    registry_root = path.parent
    for task in active:
        desc_path = (registry_root / str(task.get("desc_file", ""))).resolve()
        if not desc_path.is_file() or desc_path.stat().st_size == 0:
            raise QueueConfigurationError(
                f"missing non-empty task description for {task['task_id']}: {desc_path}"
            )
        expected_rows = task.get("expected_rows")
        if not isinstance(expected_rows, dict) or set(expected_rows) != {
            "train",
            "val",
            "test",
        }:
            raise QueueConfigurationError(
                f"{task['task_id']} requires exact train/val/test expected_rows"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in expected_rows.values()
        ):
            raise QueueConfigurationError(
                f"{task['task_id']} expected_rows must be positive integers"
            )
    payload["_registry_path"] = str(path.resolve())
    return payload


def active_tasks(
    registry: Mapping[str, Any], selected: set[str] | None = None
) -> list[dict[str, Any]]:
    tasks = [
        {**dict(task), "_registry_path": registry.get("_registry_path")}
        for task in registry["tasks"]
        if task.get("status") == "active"
        and (selected is None or task.get("task_id") in selected)
    ]
    tasks.sort(key=lambda task: (int(task["order"]), str(task["task_id"])))
    if selected is not None:
        missing = selected - {str(task["task_id"]) for task in tasks}
        if missing:
            raise QueueConfigurationError(
                "--only contains unknown or non-active task(s): " + ", ".join(sorted(missing))
            )
    return tasks


def _manifest_records(payload: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    container = payload.get("tasks", payload.get("datasets", payload))
    if isinstance(container, list):
        return [record for record in container if isinstance(record, dict)]
    if isinstance(container, dict):
        records: list[dict[str, Any]] = []
        for key, value in container.items():
            if not isinstance(value, dict):
                continue
            record = dict(value)
            record.setdefault("task_id", key)
            records.append(record)
        return records
    return []


def load_prepared_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    mapping: dict[str, dict[str, Any]] = {}
    for record in _manifest_records(read_json(path)):
        key = next(
            (
                record.get(name)
                for name in ("task_id", "medmnist_flag", "flag", "dataset", "name")
                if isinstance(record.get(name), str) and record.get(name)
            ),
            None,
        )
        if key is not None:
            normalized = dict(record)
            normalized["_manifest_root"] = str(path.parent.resolve())
            mapping[str(key)] = normalized
    return mapping


def _prepared_record_for_task(
    task: Mapping[str, Any], prepared: Mapping[str, dict[str, Any]]
) -> dict[str, Any] | None:
    for key in (task["task_id"], task.get("medmnist_flag")):
        if isinstance(key, str) and key in prepared:
            return prepared[key]
    return None


def resolve_dataset_dir(
    task: Mapping[str, Any],
    data_root: Path,
    prepared: Mapping[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any] | None]:
    record = _prepared_record_for_task(task, prepared)
    if record is not None:
        value = next(
            (
                record.get(name)
                for name in ("data_dir", "task_data_dir", "output_dir", "path")
                if isinstance(record.get(name), str) and record.get(name)
            ),
            None,
        )
        if value is not None:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = Path(str(record["_manifest_root"])) / candidate
            return candidate.resolve(), record
    return (data_root / str(task["data_dir_name"])).expanduser().resolve(), record


def _record_is_authoritatively_ready(record: Mapping[str, Any] | None) -> bool:
    if record is None:
        return False
    status_value = record.get("status")
    if isinstance(status_value, str):
        return status_value.strip().lower() in READY_STATUSES
    return record.get("ready") is True or record.get("complete") is True


def _ready_marker_state(data_dir: Path) -> tuple[bool, bool, str | None]:
    """Return (marker_present, marker_ready, reason)."""

    for name in ("_READY.json", ".ready.json", "dataset_manifest.json", "task_manifest.json"):
        path = data_dir / name
        if not path.is_file():
            continue
        try:
            payload = read_json(path)
        except (OSError, ValueError) as exc:
            return True, False, f"invalid readiness marker {path.name}: {exc}"
        if not isinstance(payload, dict):
            return True, False, f"readiness marker {path.name} is not a JSON object"
        status_value = payload.get("status")
        ready = (
            isinstance(status_value, str)
            and status_value.strip().lower() in READY_STATUSES
        ) or payload.get("ready") is True or payload.get("complete") is True
        if status_value is None and not {"ready", "complete"}.intersection(payload):
            # A completed dataset manifest need not redundantly carry a status.
            ready = bool(payload.get("source_md5") or payload.get("expected_rows"))
        return True, ready, None if ready else f"readiness marker {path.name} is not ready"
    return False, False, None


def evaluation_support_error(task: Mapping[str, Any]) -> str | None:
    family = str(task.get("task_family", "classification"))
    metric = str(task.get("primary_metric", ""))
    supported = {
        "classification": {"accuracy", "mean_roc_auc"},
        "segmentation": {"aji", "binary_dice"},
    }
    if metric in supported.get(family, set()):
        return None
    return (
        f"unsupported evaluation: {family}/{metric} has no compatible fixed "
        "and final private evaluator in the current suite runtime"
    )


def require_supported_tasks(tasks: Sequence[Mapping[str, Any]]) -> None:
    unsupported = [
        f"{task['task_id']}: {reason}"
        for task in tasks if (reason := evaluation_support_error(task)) is not None
    ]
    if unsupported:
        raise QueueConfigurationError(
            "; ".join(unsupported)
            + ". Use --only to select supported tasks; the frozen metrics were not changed."
        )


class ReadinessTracker:
    """Validate the packager's atomic suite entry, ready marker, and cache."""

    def check(
        self,
        task: Mapping[str, Any],
        data_dir: Path,
        prepared_record: Mapping[str, Any] | None,
        *,
        expected_cache_root: Path | None = None,
    ) -> DatasetCheck:
        unsupported = evaluation_support_error(task)
        if unsupported is not None:
            return DatasetCheck(False, unsupported, data_dir)
        if prepared_record is None:
            return DatasetCheck(
                False,
                "task is not yet advertised by the prepared suite_manifest.json",
                data_dir,
            )
        if not _record_is_authoritatively_ready(prepared_record):
            return DatasetCheck(False, "suite manifest entry is not ready", data_dir)
        if not data_dir.is_dir():
            return DatasetCheck(False, "dataset directory is absent", data_dir)

        try:
            ready = read_json(data_dir / "_READY.json")
        except (OSError, ValueError) as exc:
            return DatasetCheck(False, f"cannot read _READY.json: {exc}", data_dir)
        if not isinstance(ready, dict) or ready.get("status") != "ready":
            return DatasetCheck(False, "_READY.json does not record status=ready", data_dir)
        if ready.get("flag") != task.get("medmnist_flag"):
            return DatasetCheck(False, "_READY.json MedMNIST flag mismatch", data_dir)
        if ready.get("source_archive") != task.get("source_archive"):
            return DatasetCheck(False, "_READY.json source archive mismatch", data_dir)
        if ready.get("source_md5") != task.get("source_md5"):
            return DatasetCheck(False, "_READY.json source MD5 mismatch", data_dir)
        source_sha256 = ready.get("source_sha256")
        if (
            not isinstance(source_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None
        ):
            return DatasetCheck(False, "_READY.json has no valid source SHA-256", data_dir)
        expected_resolution = 64 if task.get("dimension") == "3D" else 224
        if ready.get("resolution") != expected_resolution:
            return DatasetCheck(
                False,
                f"_READY.json resolution is {ready.get('resolution')!r}, expected {expected_resolution}",
                data_dir,
            )
        expected = {str(key): int(value) for key, value in task["expected_rows"].items()}
        raw_counts = ready.get("counts")
        counts = (
            {str(key): int(value) for key, value in raw_counts.items()}
            if isinstance(raw_counts, dict)
            else None
        )
        if counts != expected:
            return DatasetCheck(
                False,
                f"ready counts {counts} do not match frozen counts {expected}",
                data_dir,
                counts,
            )

        registry_path = Path(str(task.get("_registry_path", DEFAULT_REGISTRY)))
        desc_path = registry_path.parent / str(task["desc_file"])
        if ready.get("task_text_sha256") != sha256_file(desc_path.resolve()):
            return DatasetCheck(False, "_READY.json task text digest mismatch", data_dir, counts)
        for key in (
            "source_md5",
            "source_sha256",
            "task_text_sha256",
            "primary_metric",
        ):
            expected_value = (
                task.get("primary_metric") if key == "primary_metric" else ready.get(key)
            )
            if prepared_record.get(key) != expected_value:
                return DatasetCheck(
                    False,
                    f"suite manifest {key} does not match frozen task/ready marker",
                    data_dir,
                    counts,
                )

        arrays = [data_dir / "arrays" / f"{split}_images.npy" for split in ("train", "val", "test")]
        if any(not _nonempty_file(path) for path in arrays):
            return DatasetCheck(False, "one or more label-free split arrays are absent", data_dir, counts)
        if not _nonempty_file(data_dir / "task_metadata.json"):
            return DatasetCheck(False, "task_metadata.json is absent", data_dir, counts)
        if any(data_dir.rglob("*.npz")) or any(data_dir.rglob("*labels*.npy")):
            return DatasetCheck(False, "agent-visible data directory contains source/label arrays", data_dir, counts)

        cache_value = prepared_record.get("cache_entry")
        if not isinstance(cache_value, str) or not cache_value:
            return DatasetCheck(False, "suite manifest has no cache_entry", data_dir, counts)
        cache_entry = Path(cache_value).expanduser()
        if not cache_entry.is_absolute():
            cache_entry = Path(str(prepared_record["_manifest_root"])) / cache_entry
        cache_entry = cache_entry.resolve()
        cache_key = prepared_record.get("cache_key")
        if not isinstance(cache_key, str) or not cache_key:
            return DatasetCheck(False, "suite manifest has no cache_key", data_dir, counts)
        if expected_cache_root is not None:
            expected_cache_entry = (expected_cache_root / cache_key).resolve()
            if cache_entry != expected_cache_entry:
                return DatasetCheck(
                    False,
                    "split cache is not installed under this run's resource_root: "
                    f"{cache_entry} != {expected_cache_entry}",
                    data_dir,
                    counts,
                )
        try:
            cache_manifest = read_json(cache_entry / "manifest.json")
            verifier = read_json(cache_entry / "verifier.json")
        except (OSError, ValueError) as exc:
            return DatasetCheck(False, f"split cache is incomplete: {exc}", data_dir, counts)
        if not isinstance(cache_manifest, dict) or cache_manifest.get("verifier_ok") is not True:
            return DatasetCheck(False, "split cache manifest is not verified", data_dir, counts)
        if cache_manifest.get("dataset") != task.get("task_id"):
            return DatasetCheck(False, "split cache dataset mismatch", data_dir, counts)
        if cache_manifest.get("source_md5") != task.get("source_md5"):
            return DatasetCheck(False, "split cache source MD5 mismatch", data_dir, counts)
        if cache_manifest.get("source_sha256") != source_sha256:
            return DatasetCheck(False, "split cache source SHA-256 mismatch", data_dir, counts)
        if cache_manifest.get("task_text_sha256") != ready.get("task_text_sha256"):
            return DatasetCheck(False, "split cache task text digest mismatch", data_dir, counts)
        if not isinstance(verifier, dict) or verifier.get("ok") is not True:
            return DatasetCheck(False, "split cache verifier does not record ok=true", data_dir, counts)
        if not _nonempty_file(cache_entry / "split_manifest.json"):
            return DatasetCheck(False, "split cache has no split_manifest.json", data_dir, counts)
        return DatasetCheck(True, "ready", data_dir, counts)


def runtime_env_file(paths: QueuePaths) -> Path:
    path = Path(os.environ.get("BIOMAKER_ENV_FILE") or ".env").expanduser()
    if not path.is_absolute():
        path = paths.runtime_root / path
    return path.resolve()


def _credential_metadata(paths: QueuePaths) -> dict[str, Any]:
    env_path = runtime_env_file(paths)
    endpoint: str | None = None
    api_key_present = False
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if name == "OPENAI_BASE_URL":
                endpoint = value or None
            elif name == "OPENAI_API_KEY":
                api_key_present = bool(value)
    else:
        endpoint = os.environ.get("OPENAI_BASE_URL") or None
        api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "env_file": str(env_path),
        "env_file_present": env_path.is_file(),
        "env_file_mode": (
            format(stat.S_IMODE(env_path.stat().st_mode), "04o")
            if env_path.is_file()
            else None
        ),
        "openai_base_url": endpoint,
        "openai_api_key_present": api_key_present,
        "secret_values_recorded": False,
    }


def _split_cache_artifacts(
    prepared_record: Mapping[str, Any] | None,
) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    if prepared_record is None:
        return None, {}
    raw_cache = prepared_record.get("cache_entry")
    if not isinstance(raw_cache, str) or not raw_cache:
        return None, {}
    cache_entry = Path(raw_cache).expanduser()
    if not cache_entry.is_absolute():
        cache_entry = Path(str(prepared_record.get("_manifest_root", "."))) / cache_entry
    cache_entry = cache_entry.resolve()
    relative_names = (
        "manifest.json",
        "artifact_integrity.json",
        "verifier.json",
        "split_manifest.json",
        "biomaker_splits/agent_train.csv",
        "biomaker_splits/score_val.csv",
        "biomaker_splits/agent_test.csv",
        "private_gt/score_val.csv",
        "private_gt/test.csv",
        "private_gt/private_gt.json",
    )
    artifacts = {
        name: file_record(cache_entry / name) for name in relative_names
    }
    # Freeze answer-file bytes without modifying legacy prepared caches, which
    # may predate artifact_integrity.json. CSV relocation must not hide changes
    # made to both the cached mask and its runtime copy after launch.
    private_root = cache_entry / "private_gt"
    if private_root.is_dir():
        for path in sorted(private_root.rglob("*")):
            if path.is_file():
                artifacts[path.relative_to(cache_entry).as_posix()] = file_record(path)
    return cache_entry, artifacts


def create_source_state(
    *,
    paths: QueuePaths,
    registry_path: Path,
    task: Mapping[str, Any],
    stage: str,
    steps: int,
    search_time_limit_seconds: int,
    hard_wall_clock_seconds: int,
    progress_timeout_seconds: int,
    final_retrain_progress_timeout_seconds: int,
    model: str,
    workers: int,
    gpu_uuids: Sequence[str],
    requested_name: str,
    prepared_record: Mapping[str, Any] | None,
    snapshot_path: Path,
    patch_path: Path,
) -> dict[str, Any]:
    desc_path = (registry_path.parent / str(task["desc_file"])).resolve()
    ready_path = Path(str(prepared_record.get("ready_marker"))) if prepared_record else Path()
    if not ready_path.is_absolute() and prepared_record is not None:
        ready_path = Path(str(prepared_record.get("_manifest_root", "."))) / ready_path
    ready_payload = read_json(ready_path) if ready_path.is_file() else {}
    cache_entry, cache_artifacts = _split_cache_artifacts(prepared_record)
    git_state = _git_source_state(paths.runtime_root, patch_path)
    thread_budget = max(1, available_cpu_count() // workers)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "protocol_id": "fig2-deepseek-flash-v1",
        "requested_name": requested_name,
        "stage": stage,
        "evidence_tier": "formal" if stage == "full" else "engineering_only",
        "task_id": task["task_id"],
        "medmnist_flag": task["medmnist_flag"],
        "repeat_index": int(os.environ.get("BIOMAKER_REPEAT_INDEX", "1")),
        "model": model,
        "steps": steps,
        "search_time_limit_seconds": search_time_limit_seconds,
        "hard_wall_clock_seconds": hard_wall_clock_seconds,
        "progress_timeout_seconds": progress_timeout_seconds,
        "final_retrain_progress_timeout_seconds": final_retrain_progress_timeout_seconds,
        "max_children_per_node": 4,
        "max_debug_depth": 2,
        "topk_improve": 3,
        "private_evaluation_per_node": False,
        "parallel_workers": workers,
        "gpu_uuids": list(gpu_uuids),
        "cpu_count": os.cpu_count(),
        "default_threads_per_worker": thread_budget,
        "seed_state": {
            "python_hash_seed": os.environ.get("BIOMAKER_PYTHONHASHSEED", "42"),
            "declared_experiment_seed": os.environ.get("BIOMAKER_EXPERIMENT_SEED"),
            "central_harness_seed_controlled": False,
            "note": "Current BioMaker has no central candidate-training seed control; candidate scripts own their seeds.",
        },
        "runtime": {
            "root": str(paths.runtime_root),
            "python_executable": str(paths.python),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "source_tree": _runtime_source_tree(paths.runtime_root),
        },
        "git": git_state,
        "dependencies": {
            "pyproject.toml": file_record(paths.runtime_root / "pyproject.toml"),
            "uv.lock": file_record(paths.runtime_root / "uv.lock"),
        },
        "orchestrator": {
            "queue_script": file_record(SCRIPT_PATH),
            "task_registry": file_record(registry_path),
            "task_description": file_record(desc_path),
        },
        "dataset": {
            "data_dir": str(
                Path(str(prepared_record.get("data_dir"))).resolve()
                if prepared_record and prepared_record.get("data_dir")
                else ""
            ),
            "ready_marker": file_record(ready_path),
            "source_archive": task.get("source_archive"),
            "source_md5": (
                prepared_record.get("source_md5") if prepared_record else None
            ),
            "source_sha256": (
                prepared_record.get("source_sha256") if prepared_record else None
            ),
            "ready_contract": {
                key: ready_payload.get(key)
                for key in (
                    "status",
                    "flag",
                    "resolution",
                    "counts",
                    "source_archive",
                    "source_md5",
                    "source_sha256",
                    "task_text_sha256",
                    "primary_metric",
                )
            },
        },
        "split_cache": {
            "root": str(cache_entry) if cache_entry is not None else None,
            "artifacts": cache_artifacts,
        },
        "credentials": _credential_metadata(paths),
    }
    payload["source_state_payload_sha256"] = sha256_json(payload)
    atomic_write_json(snapshot_path, payload)
    os.chmod(snapshot_path, 0o600)
    return payload


def install_source_state(
    run_dir: Path,
    source_state: Mapping[str, Any],
    patch_path: Path,
) -> None:
    destination = run_dir / "source_state.json"
    if destination.is_file():
        existing = read_json(destination)
        if existing != source_state:
            raise QueueConfigurationError(
                f"run directory already contains a different source_state.json: {run_dir}"
            )
    else:
        atomic_write_json(destination, source_state)
        os.chmod(destination, 0o600)
    patch_destination = run_dir / "source.patch"
    patch_bytes = patch_path.read_bytes() if patch_path.is_file() else b""
    if patch_destination.is_file():
        if patch_destination.read_bytes() != patch_bytes:
            raise QueueConfigurationError(
                f"run directory already contains a different source.patch: {run_dir}"
            )
    else:
        _atomic_write_bytes(patch_destination, patch_bytes)


def write_formal_validation(
    *,
    run_dir: Path,
    audit: RunAudit,
    stage: str,
    task: Mapping[str, Any],
    source_state: Mapping[str, Any],
    prepared_record: Mapping[str, Any] | None,
    watchdog_reason: str | None,
    attempt_path: Path,
) -> dict[str, Any]:
    cache_entry, cache_artifacts = _split_cache_artifacts(prepared_record)
    split_manifest: dict[str, Any] | None = None
    if cache_entry is not None and (cache_entry / "split_manifest.json").is_file():
        loaded = read_json(cache_entry / "split_manifest.json")
        split_manifest = loaded if isinstance(loaded, dict) else None
    run_artifact_names = [
        "source_state.json",
        "source.patch",
        "run_manifest.json",
        "events.jsonl",
        "journal.json",
        "results.tsv",
        "evaluator.py",
        "evaluator_prediction_spec.json",
        "best_solution.py",
    ]
    if stage == "full":
        run_artifact_names.extend(
            ["final_retrain_result.json", "final_submission.csv", "private_metrics.json"]
        )
    payload = {
        "schema_version": 1,
        "validated_at": utc_now(),
        "task_id": task["task_id"],
        "stage": stage,
        "evidence_tier": "formal" if stage == "full" else "engineering_only",
        "eligible_for_figure_2": bool(stage == "full" and audit.ok),
        "ok": audit.ok,
        "errors": list(audit.errors),
        "audit_summary": audit.summary,
        "watchdog_reason": watchdog_reason,
        "source_state_payload_sha256": source_state.get(
            "source_state_payload_sha256"
        ),
        "source_state_file": file_record(run_dir / "source_state.json"),
        "split_manifest": split_manifest,
        "split_cache_artifacts": cache_artifacts,
        "run_artifacts": {
            name: file_record(run_dir / name) for name in run_artifact_names
        },
        "private_result": audit.summary.get("private_result"),
    }
    payload["formal_validation_payload_sha256"] = sha256_json(payload)
    atomic_write_json(attempt_path, payload)
    atomic_write_json(run_dir / "formal_validation.json", payload)
    return payload


def parse_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return events, ["missing events.jsonl"]
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError as exc:
                errors.append(f"events.jsonl line {line_number} is invalid JSON: {exc}")
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("event"), str):
                errors.append(f"events.jsonl line {line_number} is not an event object")
                continue
            events.append(payload)
    return events, errors


def last_valid_event_name(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("event"), str):
            return str(payload["event"])
    return None


def progress_timeout_for_event(
    event_name: str | None,
    *,
    default_seconds: int,
    final_retrain_seconds: int,
) -> int:
    return (
        final_retrain_seconds
        if event_name == "final_retrain_started"
        else default_seconds
    )


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _verify_file_record(
    record: Any,
    label: str,
    errors: list[str],
    *,
    actual_path: Path | None = None,
    require_present: bool = False,
) -> None:
    if not isinstance(record, dict):
        errors.append(f"{label} provenance record is not an object")
        return
    path_value = record.get("path")
    path = actual_path or (Path(path_value) if isinstance(path_value, str) else None)
    expected_present = record.get("present") is True
    if path is None:
        errors.append(f"{label} provenance record has no path")
        return
    if require_present and not expected_present:
        errors.append(f"{label} was not present when source state was frozen")
        return
    if not expected_present:
        if actual_path is not None and path.exists():
            errors.append(f"{label} unexpectedly exists at {path}")
        return
    if not path.is_file():
        errors.append(f"{label} is missing at {path}")
        return
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path)
    if record.get("size_bytes") != actual_size:
        errors.append(
            f"{label} size changed: {actual_size} != {record.get('size_bytes')!r}"
        )
    if record.get("sha256") != actual_hash:
        errors.append(f"{label} SHA-256 changed")


def _private_answer_path(value: str, private_root: Path) -> tuple[Path, Path]:
    raw = Path(value)
    if not raw.is_absolute():
        if raw.parts and raw.parts[0] == "private_gt":
            raw = Path(*raw.parts[1:])
        raw = private_root / raw
    resolved = raw.resolve()
    relative = resolved.relative_to(private_root.resolve())
    if not resolved.is_file():
        raise ValueError(f"private answer file is missing: {relative}")
    return resolved, relative


def _verify_private_csv_record(
    record: Any,
    frozen_artifacts: Mapping[str, Any],
    label: str,
    errors: list[str],
    *,
    actual_path: Path,
) -> None:
    """Allow runtime path relocation while preserving every private answer."""
    before = len(errors)
    _verify_file_record(record, f"{label} frozen CSV", errors, require_present=True)
    if len(errors) != before:
        return
    try:
        source_path = Path(record["path"])
        with source_path.open(newline="", encoding="utf-8") as handle:
            frozen = list(csv.reader(handle))
        with actual_path.open(newline="", encoding="utf-8") as handle:
            actual = list(csv.reader(handle))
        if not frozen or not actual or actual[0] != frozen[0]:
            raise ValueError("private CSV columns differ from frozen source")
        if len(actual) != len(frozen):
            raise ValueError("private CSV row count differs from frozen source")
        columns = frozen[0]
        if len(set(columns)) != len(columns):
            raise ValueError("private CSV has duplicate columns")
        for row_index, (source_row, actual_row) in enumerate(
            zip(frozen[1:], actual[1:]), start=1
        ):
            if len(source_row) != len(columns) or len(actual_row) != len(columns):
                raise ValueError(f"private CSV row {row_index} has the wrong width")
            for column, source_value, actual_value in zip(columns, source_row, actual_row):
                # Private file answers use these names in the frozen task packs.
                # Public image_path references remain identical and are not labels.
                is_answer_path = any(
                    token in column.lower() for token in ("mask", "annotation", "bbox")
                ) and ("/" in source_value or "\\" in source_value)
                if not is_answer_path:
                    if actual_value != source_value:
                        raise ValueError(f"private CSV row {row_index} {column} changed")
                    continue
                source_file, source_relative = _private_answer_path(
                    source_value, source_path.parent
                )
                actual_file, actual_relative = _private_answer_path(
                    actual_value, actual_path.parent
                )
                if actual_relative != source_relative:
                    raise ValueError(f"private CSV row {row_index} {column} target changed")
                artifact_record = frozen_artifacts.get(
                    f"private_gt/{source_relative.as_posix()}"
                )
                expected_hash = (
                    artifact_record.get("sha256")
                    if isinstance(artifact_record, dict)
                    else None
                )
                if not isinstance(expected_hash, str) or any(
                    sha256_file(path) != expected_hash
                    for path in (source_file, actual_file)
                ):
                    raise ValueError(f"private CSV row {row_index} {column} bytes changed")
    except (OSError, ValueError, csv.Error, TypeError) as exc:
        errors.append(f"{label}: {exc}")


def _private_result_summary(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    history = payload.get("history")
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
        return None
    private = history[-1].get("private_metrics")
    if not isinstance(private, dict):
        return None
    results = private.get("results")
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return None
    result = results[0]
    return {
        "metric_name": result.get("metric_name"),
        "value": result.get("value"),
        "metrics": result.get("metrics"),
        "n": result.get("n"),
    }


def _step_order_errors(
    nodes: Sequence[Mapping[str, Any]],
    saved_events: Sequence[Mapping[str, Any]],
    expected_steps: int,
) -> list[str]:
    """Audit committed order without assuming parallel jobs finish in plan order."""
    errors: list[str] = []
    expected = list(range(expected_steps))
    if [event.get("step") for event in saved_events] != expected:
        errors.append("step_saved events are not in exact commit order")
    if [node.get("step") for node in nodes] != expected:
        errors.append("journal step values are not in exact commit order")
    scheduled = [node.get("scheduled_step") for node in nodes]
    if (
        any(not isinstance(step, int) or isinstance(step, bool) for step in scheduled)
        or sorted(scheduled) != expected
    ):
        errors.append("journal scheduled_step set is not complete and unique")
    planned_by_id = {node.get("id"): node.get("scheduled_step") for node in nodes}
    # Older runtimes emitted only the committed step. Where present, the
    # planned step must match the same node in the journal, not its finish rank.
    for event in saved_events:
        if "scheduled_step" in event and (
            event["scheduled_step"] != planned_by_id.get(event.get("node_id"))
        ):
            errors.append("step_saved node-id to scheduled_step mapping differs from journal")
    return errors


def validate_run_artifacts(
    run_dir: Path,
    *,
    stage: str,
    expected_steps: int,
    expected_model: str,
    expected_metric: str,
    expected_test_rows: int,
    expected_workers: int,
    expected_gpu_uuids: Sequence[str],
    require_report: bool,
    return_code: int | None = None,
) -> RunAudit:
    if stage not in {"canary", "full"}:
        raise QueueConfigurationError(f"unknown Figure 2 stage: {stage!r}")
    require_final_retrain = stage == "full"
    errors: list[str] = []
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "stage": stage,
        "expected_steps": expected_steps,
        "evidence_tier": "formal" if require_final_retrain else "engineering_only",
    }
    if not run_dir.is_dir():
        return RunAudit(False, (f"run directory does not exist: {run_dir}",), summary)
    if return_code not in (None, 0):
        errors.append(f"BioMaker process exited with code {return_code}")

    events, event_errors = parse_events(run_dir / "events.jsonl")
    errors.extend(event_errors)
    terminal = [event for event in events if event.get("event") in TERMINAL_EVENTS]
    if len(terminal) != 1:
        errors.append(f"expected exactly one terminal event, found {len(terminal)}")
    elif terminal[0].get("event") != "run_finished":
        errors.append(f"terminal event is {terminal[0].get('event')}, not run_finished")
    else:
        summary["terminal_event"] = "run_finished"
        if terminal[0].get("completed_steps") != expected_steps:
            errors.append(
                "run_finished completed_steps does not equal " + str(expected_steps)
            )
    started = [event for event in events if event.get("event") == "run_started"]
    if len(started) != 1:
        errors.append(f"expected exactly one run_started event, found {len(started)}")
    elif started[0].get("steps") != expected_steps:
        errors.append("run_started steps does not match the requested stage")
    retrain_started = [
        event for event in events if event.get("event") == "final_retrain_started"
    ]
    retrain = [event for event in events if event.get("event") == "final_retrain_finished"]
    if require_final_retrain:
        if len(retrain_started) != 1:
            errors.append(
                f"expected exactly one final_retrain_started event, found {len(retrain_started)}"
            )
        if len(retrain) != 1 or retrain[0].get("ok") is not True:
            errors.append("expected exactly one successful final_retrain_finished event")
    elif retrain_started or retrain:
        errors.append("canary consumed final retrain/private-test evaluation")
    fixed_evaluators = [
        event for event in events if event.get("event") == "fixed_evaluator_prepared"
    ]
    if len(fixed_evaluators) != 1:
        errors.append(
            f"expected exactly one fixed_evaluator_prepared event, found {len(fixed_evaluators)}"
        )
    generated_evaluator_attempts = [
        event for event in events if event.get("event") == "evaluator_attempt_started"
    ]
    if generated_evaluator_attempts:
        errors.append("LLM evaluator fallback was used; fixed evaluator gate failed")
    saved_events = [event for event in events if event.get("event") == "step_saved"]
    if len(saved_events) != expected_steps:
        errors.append(
            f"step_saved event count is {len(saved_events)}, expected {expected_steps}"
        )
    saved_ids = {
        str(event.get("node_id"))
        for event in saved_events
        if isinstance(event.get("node_id"), str)
    }
    saved_id_to_step = {
        str(event.get("node_id")): event.get("step")
        for event in saved_events
        if isinstance(event.get("node_id"), str)
    }

    required_files = [
        "run_manifest.json",
        "source_state.json",
        "journal.json",
        "results.tsv",
        "evaluator.py",
        "evaluator_prediction_spec.json",
        "best_solution.py",
    ]
    if require_final_retrain:
        required_files.extend(
            ["final_retrain_result.json", "final_submission.csv", "private_metrics.json"]
        )
    if require_report:
        required_files.append("report.md")
    for name in required_files:
        if not _nonempty_file(run_dir / name):
            errors.append(f"missing or empty required artifact: {name}")

    journal_ids: set[str] = set()
    journal_short_ids: set[str] = set()
    journal_id_to_step: dict[str, int] = {}
    journal_short_id_to_step: dict[str, int] = {}
    try:
        journal = read_json(run_dir / "journal.json")
        nodes = journal.get("nodes") if isinstance(journal, dict) else None
        if not isinstance(nodes, list):
            errors.append("journal.json nodes is not a list")
        else:
            summary["journal_nodes"] = len(nodes)
            if len(nodes) != expected_steps:
                errors.append(
                    f"journal has {len(nodes)} nodes, expected exactly {expected_steps}"
                )
            errors.extend(_step_order_errors(
                [node for node in nodes if isinstance(node, dict)],
                saved_events,
                expected_steps,
            ))
            journal_ids = {
                str(node.get("id"))
                for node in nodes
                if isinstance(node, dict) and isinstance(node.get("id"), str)
            }
            journal_short_ids = {node_id[:8] for node_id in journal_ids}
            if len(journal_ids) != expected_steps:
                errors.append("journal node IDs are missing or duplicated")
            if len(journal_short_ids) != expected_steps:
                errors.append("journal short node IDs collide")
            journal_id_to_step = {
                str(node.get("id")): int(node.get("step"))
                for node in nodes
                if isinstance(node, dict)
                and isinstance(node.get("id"), str)
                and isinstance(node.get("step"), int)
                and not isinstance(node.get("step"), bool)
            }
            journal_short_id_to_step = {
                node_id[:8]: step for node_id, step in journal_id_to_step.items()
            }
            if saved_ids != journal_ids:
                errors.append("step_saved node-id set differs from journal node-id set")
            if saved_id_to_step != journal_id_to_step:
                errors.append("step_saved node-id to step mapping differs from journal")
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"cannot validate journal.json: {exc}")

    manifest: dict[str, Any] = {}
    workspace_dir: Path | None = None
    try:
        manifest = read_json(run_dir / "run_manifest.json")
        config = manifest.get("config", {}) if isinstance(manifest, dict) else {}
        workspace_value = manifest.get("workspace_dir") if isinstance(manifest, dict) else None
        if isinstance(workspace_value, str) and workspace_value:
            workspace_dir = Path(workspace_value)
        if config.get("agent_steps") != expected_steps:
            errors.append("run_manifest agent_steps does not match the requested stage")
        if config.get("main_model") != expected_model:
            errors.append(
                f"run_manifest main_model is {config.get('main_model')!r}, expected {expected_model!r}"
            )
        for model_field in (
            "feedback_model",
            "knowledge_base_model",
            "report_model",
        ):
            if config.get(model_field) != expected_model:
                errors.append(
                    f"run_manifest {model_field} is {config.get(model_field)!r}, "
                    f"expected {expected_model!r}"
                )
        if config.get("requested_parallel_workers") != expected_workers:
            errors.append(
                "run_manifest requested_parallel_workers is "
                f"{config.get('requested_parallel_workers')!r}, expected {expected_workers}"
            )
        if config.get("parallel_workers") != expected_workers:
            errors.append(
                "run_manifest effective parallel_workers is "
                f"{config.get('parallel_workers')!r}, expected {expected_workers}"
            )
        compute_slots = config.get("compute_slots")
        expected_selectors = list(expected_gpu_uuids)
        if not isinstance(compute_slots, list) or len(compute_slots) != expected_workers:
            errors.append("run_manifest compute_slots do not match the worker count")
        else:
            selectors = [
                slot.get("selector") if isinstance(slot, dict) else None
                for slot in compute_slots
            ]
            backends = [
                slot.get("backend") if isinstance(slot, dict) else None
                for slot in compute_slots
            ]
            indices = [
                slot.get("index") if isinstance(slot, dict) else None
                for slot in compute_slots
            ]
            if selectors != expected_selectors:
                errors.append("run_manifest compute-slot selectors differ from GPU UUID mask")
            if backends != ["cuda"] * expected_workers:
                errors.append("run_manifest contains a non-CUDA compute slot")
            if indices != list(range(expected_workers)):
                errors.append("run_manifest compute-slot indices are not canonical")
        expected_retrain = require_final_retrain
        if config.get("private_evaluation_per_node") is not False:
            errors.append("run_manifest must disable per-node private evaluation")
        if config.get("final_retrain_enabled") is not expected_retrain:
            errors.append(
                "run_manifest final_retrain_enabled is "
                f"{config.get('final_retrain_enabled')!r}, expected {expected_retrain!r}"
            )
    except (OSError, ValueError) as exc:
        errors.append(f"cannot validate run_manifest.json: {exc}")

    source_state: dict[str, Any] = {}
    try:
        source_state = read_json(run_dir / "source_state.json")
        if not isinstance(source_state, dict) or not source_state:
            errors.append("source_state.json is not a non-empty object")
        else:
            stored_digest = source_state.get("source_state_payload_sha256")
            digest_payload = dict(source_state)
            digest_payload.pop("source_state_payload_sha256", None)
            if stored_digest != sha256_json(digest_payload):
                errors.append("source_state payload digest does not verify")
            if source_state.get("stage") != stage:
                errors.append("source_state stage does not match the run stage")
            runtime = source_state.get("runtime")
            source_tree = runtime.get("source_tree") if isinstance(runtime, dict) else None
            if not isinstance(source_tree, dict) or not source_tree.get("sha256"):
                errors.append("source_state has no runtime source-tree digest")
            else:
                runtime_root_value = runtime.get("root")
                if not isinstance(runtime_root_value, str) or not runtime_root_value:
                    errors.append("source_state runtime root is absent")
                else:
                    runtime_root = Path(runtime_root_value)
                    actual_tree = _runtime_source_tree(runtime_root)
                    if actual_tree.get("sha256") != source_tree.get("sha256"):
                        errors.append("runtime source-tree digest changed after launch")
                    if actual_tree.get("file_count") != source_tree.get("file_count"):
                        errors.append("runtime source-tree file count changed after launch")
            git_state = source_state.get("git")
            if not isinstance(git_state, dict) or not git_state.get("git_head"):
                errors.append("source_state has no Git HEAD provenance")
            expected_patch_hash = (
                git_state.get("git_binary_diff_sha256")
                if isinstance(git_state, dict)
                else None
            )
            source_patch = run_dir / "source.patch"
            if not source_patch.is_file():
                errors.append("missing source.patch")
            elif expected_patch_hash != sha256_file(source_patch):
                errors.append("source.patch digest differs from source_state")

            dependencies = source_state.get("dependencies")
            if not isinstance(dependencies, dict):
                errors.append("source_state dependencies are absent")
            else:
                for name in ("pyproject.toml", "uv.lock"):
                    _verify_file_record(
                        dependencies.get(name),
                        f"dependency {name}",
                        errors,
                        require_present=True,
                    )
            orchestrator = source_state.get("orchestrator")
            if not isinstance(orchestrator, dict):
                errors.append("source_state orchestrator artifacts are absent")
            else:
                for name in ("queue_script", "task_registry", "task_description"):
                    _verify_file_record(
                        orchestrator.get(name),
                        f"orchestrator {name}",
                        errors,
                        require_present=True,
                    )
            dataset_state = source_state.get("dataset")
            if not isinstance(dataset_state, dict):
                errors.append("source_state dataset artifacts are absent")
            else:
                _verify_file_record(
                    dataset_state.get("ready_marker"),
                    "dataset ready marker",
                    errors,
                    require_present=True,
                )
            split_cache_state = source_state.get("split_cache")
            split_cache_artifacts = (
                split_cache_state.get("artifacts")
                if isinstance(split_cache_state, dict)
                else None
            )
            if not isinstance(split_cache_artifacts, dict):
                errors.append("source_state split-cache artifacts are absent")
            else:
                for name in (
                    "manifest.json",
                    "verifier.json",
                    "split_manifest.json",
                    "biomaker_splits/agent_train.csv",
                    "biomaker_splits/score_val.csv",
                    "biomaker_splits/agent_test.csv",
                    "private_gt/score_val.csv",
                    "private_gt/test.csv",
                    "private_gt/private_gt.json",
                ):
                    _verify_file_record(
                        split_cache_artifacts.get(name),
                        f"split cache {name}",
                        errors,
                        require_present=True,
                    )
    except (OSError, ValueError) as exc:
        errors.append(f"cannot validate source_state.json: {exc}")

    split_cache_state = source_state.get("split_cache") if source_state else None
    frozen_split_artifacts = (
        split_cache_state.get("artifacts")
        if isinstance(split_cache_state, dict)
        else None
    )
    if workspace_dir is None:
        errors.append("run_manifest has no usable workspace_dir")
    elif not isinstance(frozen_split_artifacts, dict):
        errors.append("cannot compare run splits without frozen split-cache records")
    else:
        split_dir = workspace_dir / "input" / "biomaker_splits"
        for name in ("agent_train.csv", "score_val.csv", "agent_test.csv"):
            _verify_file_record(
                frozen_split_artifacts.get(f"biomaker_splits/{name}"),
                f"run split {name}",
                errors,
                actual_path=split_dir / name,
                require_present=True,
            )
        _verify_file_record(
            frozen_split_artifacts.get("split_manifest.json"),
            "run split_manifest.json",
            errors,
            actual_path=split_dir / "split_manifest.json",
            require_present=True,
        )
        for name in ("score_val.csv", "test.csv"):
            _verify_private_csv_record(
                frozen_split_artifacts.get(f"private_gt/{name}"),
                frozen_split_artifacts,
                f"run private_gt/{name}",
                errors,
                actual_path=run_dir / "private_gt" / name,
            )
        _verify_file_record(
            frozen_split_artifacts.get("private_gt/private_gt.json"),
            "run private_gt/private_gt.json",
            errors,
            actual_path=run_dir / "private_gt" / "private_gt.json",
            require_present=True,
        )
        try:
            split_plan = read_json(split_dir / "split_plan.json")
            if not isinstance(split_plan, dict) or split_plan.get("cache_reused") is not True:
                errors.append("run did not reuse the frozen split-builder cache")
        except (OSError, ValueError) as exc:
            errors.append(f"cannot validate run split_plan.json: {exc}")

        for name in ("evaluator.py", "evaluator_prediction_spec.json"):
            run_evaluator = run_dir / name
            workspace_evaluator = workspace_dir / name
            if not _nonempty_file(run_evaluator) or not _nonempty_file(workspace_evaluator):
                errors.append(f"run/workspace evaluator copy is missing: {name}")
            elif sha256_file(run_evaluator) != sha256_file(workspace_evaluator):
                errors.append(f"run/workspace evaluator hash mismatch: {name}")
    summary["provenance_recomputed"] = not any(
        "source" in error.lower()
        or "split" in error.lower()
        or "dependency" in error.lower()
        or "orchestrator" in error.lower()
        or "evaluator hash" in error.lower()
        for error in errors
    )

    try:
        prediction_spec = read_json(run_dir / "evaluator_prediction_spec.json")
        actual_metric = (
            prediction_spec.get("primary_metric")
            if isinstance(prediction_spec, dict)
            else None
        )
        summary["primary_metric"] = actual_metric
        if actual_metric != expected_metric:
            errors.append(
                f"evaluator primary_metric is {actual_metric!r}, expected {expected_metric!r}"
            )
    except (OSError, ValueError) as exc:
        errors.append(f"cannot validate evaluator_prediction_spec.json: {exc}")

    expected_private_metric = {
        "accuracy": "private_accuracy",
        "mean_roc_auc": "private_mean_roc_auc",
        "ordinal_accuracy": "private_ordinal_accuracy",
    }.get(expected_metric)
    final_private_result: dict[str, Any] | None = None
    history_private_result: dict[str, Any] | None = None
    if require_final_retrain:
        try:
            final_retrain = read_json(run_dir / "final_retrain_result.json")
            if not isinstance(final_retrain, dict) or final_retrain.get("ok") is not True:
                errors.append("final_retrain_result.json does not record ok=true")
            elif final_retrain.get("error") not in (None, "") or final_retrain.get("exc_type") not in (None, ""):
                errors.append("final_retrain_result.json contains a final error")
            else:
                private = final_retrain.get("private_metrics")
                results = private.get("results") if isinstance(private, dict) else None
                if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
                    errors.append("final_retrain_result must contain exactly one private_test result")
                else:
                    final_private_result = results[0]
        except (OSError, ValueError) as exc:
            errors.append(f"cannot validate final_retrain_result.json: {exc}")

        try:
            private_metrics = read_json(run_dir / "private_metrics.json")
            history = private_metrics.get("history") if isinstance(private_metrics, dict) else None
            if not isinstance(history, list) or len(history) != 1:
                count = len(history) if isinstance(history, list) else "invalid"
                errors.append(
                    f"private metric history must contain only final retrain, found {count} entries"
                )
            elif not isinstance(history[0], dict) or history[0].get("stage") != "final_retrain":
                errors.append("the only private metric entry is not the final_retrain stage")
            elif history[0].get("private_error") not in (None, ""):
                errors.append("final private metric history entry contains private_error")
            else:
                private = history[0].get("private_metrics")
                results = private.get("results") if isinstance(private, dict) else None
                if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
                    errors.append("private metric history must contain exactly one private_test result")
                else:
                    history_private_result = results[0]
            if isinstance(private_metrics, dict) and private_metrics.get("visible_to_agent") is not False:
                errors.append("private_metrics.json must record visible_to_agent=false")
            summary["private_result"] = (
                _private_result_summary(private_metrics)
                if isinstance(private_metrics, dict)
                else None
            )
        except (OSError, ValueError) as exc:
            errors.append(f"cannot validate private_metrics.json: {exc}")
    else:
        forbidden_private = [
            name
            for name in (
                "final_retrain_result.json",
                "final_submission.csv",
                "private_metrics.json",
            )
            if (run_dir / name).exists()
        ]
        if forbidden_private:
            errors.append(
                "canary produced forbidden private/final artifacts: "
                + ", ".join(forbidden_private)
            )

    for source_name, result in (
        ("final_retrain_result", final_private_result),
        ("private_metrics", history_private_result),
    ):
        if result is None:
            continue
        value = result.get("value")
        if result.get("name") != "private_test":
            errors.append(f"{source_name} result name is not private_test")
        if expected_private_metric is None or result.get("metric_name") != expected_private_metric:
            errors.append(
                f"{source_name} metric_name is {result.get('metric_name')!r}, expected {expected_private_metric!r}"
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{source_name} private metric value is not finite")
        if result.get("n") != expected_test_rows:
            errors.append(
                f"{source_name} private n is {result.get('n')!r}, expected {expected_test_rows}"
            )
        if result.get("error") not in (None, ""):
            errors.append(f"{source_name} private result contains an error")
    if final_private_result is not None and history_private_result is not None:
        if final_private_result != history_private_result:
            errors.append("final_retrain_result and private_metrics private results differ")

    try:
        with (run_dir / "results.tsv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        summary["results_rows"] = len(rows)
        if len(rows) != expected_steps:
            errors.append(f"results.tsv has {len(rows)} rows, expected {expected_steps}")
        result_ids = {
            str(row.get("node_id")) for row in rows if isinstance(row.get("node_id"), str)
        }
        if result_ids != journal_short_ids:
            errors.append("results.tsv node-id set differs from journal node-id set")
        result_steps: list[int | None] = []
        result_id_to_step: dict[str, int] = {}
        for row in rows:
            raw_step = row.get("step")
            try:
                parsed_step = int(str(raw_step))
            except (TypeError, ValueError):
                parsed_step = None
            result_steps.append(parsed_step)
            node_id = row.get("node_id")
            if isinstance(node_id, str) and parsed_step is not None:
                result_id_to_step[node_id] = parsed_step
        if result_steps != list(range(expected_steps)):
            errors.append("results.tsv steps are not in exact commit order")
        if result_id_to_step != journal_short_id_to_step:
            errors.append("results.tsv node-id to step mapping differs from journal")
    except (OSError, csv.Error) as exc:
        errors.append(f"cannot validate results.tsv: {exc}")

    if require_final_retrain:
        try:
            with (run_dir / "final_submission.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                submission_rows = list(csv.reader(handle))
            actual_submission_rows = max(0, len(submission_rows) - 1)
            if actual_submission_rows != expected_test_rows:
                errors.append(
                    f"final_submission.csv has {actual_submission_rows} rows, expected {expected_test_rows}"
                )
            summary["final_submission_rows"] = actual_submission_rows
        except (OSError, csv.Error) as exc:
            errors.append(f"cannot validate final_submission.csv: {exc}")

    artifact_ids: set[str] = set()
    artifact_steps: list[int] = []
    artifact_id_to_step: dict[str, int] = {}
    node_artifact_root = run_dir / "node_artifacts"
    if not node_artifact_root.is_dir():
        errors.append("missing node_artifacts directory")
    else:
        metadata_paths = sorted(node_artifact_root.glob("*/node_metadata.json"))
        if len(metadata_paths) != expected_steps:
            errors.append(
                f"node_artifacts has {len(metadata_paths)} metadata files, expected {expected_steps}"
            )
        for metadata_path in metadata_paths:
            try:
                metadata = read_json(metadata_path)
            except (OSError, ValueError) as exc:
                errors.append(f"invalid {metadata_path}: {exc}")
                continue
            if not isinstance(metadata, dict):
                errors.append(f"{metadata_path} is not an object")
                continue
            node_id = metadata.get("node_id")
            step = metadata.get("step")
            if isinstance(node_id, str):
                artifact_ids.add(node_id)
            if isinstance(step, int) and not isinstance(step, bool):
                artifact_steps.append(step)
                if isinstance(node_id, str):
                    artifact_id_to_step[node_id] = step
        if artifact_ids != journal_ids:
            errors.append("node_artifacts node-id set differs from journal node-id set")
        if sorted(artifact_steps) != list(range(expected_steps)):
            errors.append("node_artifacts step set is not complete and unique")
        if artifact_id_to_step != journal_id_to_step:
            errors.append("node_artifacts node-id to step mapping differs from journal")

    execution_started = [
        event for event in events if event.get("event") == "execution_started"
    ]
    if len(execution_started) != expected_steps:
        errors.append(
            f"execution_started event count is {len(execution_started)}, expected {expected_steps}"
        )
    execution_ids = {
        str(event.get("node_id"))
        for event in execution_started
        if isinstance(event.get("node_id"), str)
    }
    if execution_ids != journal_ids:
        errors.append("execution_started node-id set differs from journal")
    allowed_selectors = set(expected_gpu_uuids)
    for event in execution_started:
        slot = event.get("compute_slot")
        if event.get("compute_backend") != "cuda" or not isinstance(slot, dict):
            errors.append("execution_started lacks a CUDA compute slot")
            continue
        selector = slot.get("selector")
        if (
            not isinstance(selector, str)
            or "," in selector
            or selector not in allowed_selectors
            or slot.get("backend") != "cuda"
        ):
            errors.append("execution_started used an invalid GPU selector")
    if require_final_retrain and retrain_started:
        slot = retrain_started[0].get("compute_slot")
        selector = slot.get("selector") if isinstance(slot, dict) else None
        if (
            retrain_started[0].get("compute_backend") != "cuda"
            or not isinstance(selector, str)
            or "," in selector
            or selector not in allowed_selectors
        ):
            errors.append("final_retrain_started used an invalid GPU selector")

    return RunAudit(not errors, tuple(errors), summary)


def parse_gpu_uuids(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values or any(GPU_UUID_RE.fullmatch(value) is None for value in values):
        raise QueueConfigurationError(
            "BIOMAKER_GPU_UUIDS must be a comma-separated list of exact GPU-* UUIDs"
        )
    if len(values) != len(set(values)):
        raise QueueConfigurationError("BIOMAKER_GPU_UUIDS must not contain duplicates")
    return values


def gpu_inventory() -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QueueConfigurationError(f"cannot query CUDA GPU inventory: {exc}") from exc
    if result.returncode != 0:
        raise QueueConfigurationError("nvidia-smi GPU inventory query failed")
    if not result.stdout.strip():
        raise QueueConfigurationError("no CUDA GPU is available")
    return parse_gpu_uuids(",".join(result.stdout.splitlines()))


def visible_gpu_inventory(
    inventory: Sequence[str], *, python: Path | None = None,
) -> tuple[str, ...]:
    """Restrict automatic assignment to the caller's CUDA device allocation."""
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return tuple(inventory)
    selectors = [selector.strip() for selector in mask.split(",")]
    if any(selector.isdecimal() for selector in selectors):
        # CUDA ordinals can differ from nvidia-smi's order (FASTEST_FIRST,
        # PCI_BUS_ID, containers). Resolve them through CUDA with the caller's
        # exact mask and device order, without initializing CUDA in this queue.
        if any(
            not (selector.startswith("GPU-") or (
                selector.isdecimal() and int(selector) < len(inventory)
            ))
            for selector in selectors
        ) or len(selectors) != len(set(selectors)):
            raise QueueConfigurationError("CUDA_VISIBLE_DEVICES contains invalid selectors")
        probe_code = (
            "import json, torch; "
            "print(json.dumps([str(torch.cuda.get_device_properties(i).uuid) "
            "for i in range(torch.cuda.device_count())]))"
        )
        try:
            result = subprocess.run(
                [str(python or sys.executable), "-c", probe_code],
                capture_output=True, text=True, check=False, timeout=30,
            )
            if result.returncode != 0:
                raise ValueError((result.stderr or result.stdout).strip())
            values = json.loads(result.stdout)
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError("CUDA did not return a UUID list")
            resolved = parse_gpu_uuids(",".join(
                value if value.startswith("GPU-") else f"GPU-{value}" for value in values
            ))
            if len(resolved) != len(selectors) or not set(resolved).issubset(inventory):
                raise ValueError("CUDA device count/UUIDs do not match the allocated mask")
            return resolved
        except (OSError, ValueError, QueueConfigurationError, subprocess.TimeoutExpired) as exc:
            raise QueueConfigurationError(
                f"cannot resolve CUDA_VISIBLE_DEVICES through the runtime Python: {exc}"
            ) from exc
    visible: list[str] = []
    for selector in selectors:
        if selector.startswith("GPU-"):
            matches = [uuid for uuid in inventory if uuid.startswith(selector)]
        else:
            matches = []
        if len(matches) != 1 or matches[0] in visible:
            raise QueueConfigurationError(
                f"CUDA_VISIBLE_DEVICES has an unavailable or ambiguous selector: {selector!r}"
            )
        visible.append(matches[0])
    return tuple(visible)


def resolve_compute_profile(
    inventory_provider: Callable[[], tuple[str, ...]] | None = None,
    *, python: Path | None = None,
) -> tuple[tuple[str, ...], int]:
    """Use one worker per allocated idle GPU, up to the suite's four-worker cap."""
    explicit = os.environ.get("BIOMAKER_GPU_UUIDS")
    selected = parse_gpu_uuids(explicit) if explicit else None
    if selected is None or "CUDA_VISIBLE_DEVICES" in os.environ:
        inventory = visible_gpu_inventory((inventory_provider or gpu_inventory)(), python=python)
        if selected is not None:
            if not set(selected).issubset(inventory):
                raise QueueConfigurationError(
                    "BIOMAKER_GPU_UUIDS must stay within CUDA_VISIBLE_DEVICES"
                )
        else:
            busy = {uuid for _pid, uuid in gpu_compute_processes(inventory)}
            available = tuple(uuid for uuid in inventory if uuid not in busy)
            if not available:
                raise QueueConfigurationError("no idle allocated CUDA GPU is available")
            workers = env_int("BIOMAKER_PARALLEL_WORKERS", min(4, len(available)))
            if workers > len(available):
                raise QueueConfigurationError(
                    f"parallel_workers={workers} exceeds {len(available)} idle GPU(s)"
                )
            selected = available[:workers]
    workers = env_int("BIOMAKER_PARALLEL_WORKERS", len(selected))
    if workers > 4:
        raise QueueConfigurationError(
            "one suite run supports at most four workers; use separate repeats for additional GPUs"
        )
    if workers != len(selected):
        raise QueueConfigurationError(
            "parallel_workers must equal the exact GPU UUID mask size"
        )
    return tuple(selected), workers


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise QueueConfigurationError(f"{name} must be an integer") from exc
    if value < minimum:
        raise QueueConfigurationError(f"{name} must be >= {minimum}")
    return value


def runtime_python(runtime_root: Path, explicit: Path | None = None) -> Path:
    if explicit or os.environ.get("BIOMAKER_PYTHON"):
        return Path(explicit or os.environ["BIOMAKER_PYTHON"]).expanduser().absolute()
    environment = Path(os.environ.get("UV_PROJECT_ENVIRONMENT", ".venv")).expanduser()
    if not environment.is_absolute():
        environment = runtime_root / environment
    # Keep the environment entry point, even when it is a symlink into uv's
    # interpreter cache: resolving the link loses virtualenv site-packages.
    return (environment / "bin" / "python").absolute()


def available_cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def build_run_command(
    *,
    paths: QueuePaths,
    registry_path: Path,
    task: Mapping[str, Any],
    stage: str,
    steps: int,
    total_time_limit: int,
    node_timeout: int,
    workers: int,
    model: str,
    requested_name: str,
    data_dir: Path,
    generate_report: bool,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    desc_path = (registry_path.parent / str(task["desc_file"])).resolve()
    command = [
        str(paths.python),
        "-m",
        "biomaker.run",
        f"data_dir={data_dir}",
        f"desc_file={desc_path}",
        f"log_dir={paths.logs_root}",
        f"workspace_dir={paths.workspaces_root}",
        f"resource_dir={paths.resource_root}",
        "copy_data=false",
        "preprocess_data=false",
        f"exp_name={requested_name}",
        f"model_for_all_agents={model}",
        f"agent.steps={steps}",
        f"agent.total_time_limit={total_time_limit}",
        "agent.search.max_children_per_node=4",
        "agent.search.max_debug_depth=2",
        "agent.search.topk_improve=3",
        "agent.max_node_tool_calls=4",
        "agent.enable_node_tools=true",
        "agent.enable_monai_knowledge_base=true",
        "agent.knowledge_base.retrieval_mode=keyword",
        "agent.enable_web_search=true",
        "agent.web_search_topk=2",
        f"exec.timeout={node_timeout}",
        f"exec.parallel_workers={workers}",
        "evaluator_cache.enabled=true",
        "split_builder.attempts=5",
        "private_evaluation.per_node=false",
        f"final_retrain.enabled={'true' if stage == 'full' else 'false'}",
        f"generate_report={'true' if generate_report else 'false'}",
        *extra_overrides,
    ]
    return command


def child_environment(
    gpu_uuids: Sequence[str],
    workers: int,
    *,
    launch_token: str | None = None,
    cpu_count: int | None = None,
) -> dict[str, str]:
    environ = dict(os.environ)
    available_cpus = cpu_count if cpu_count is not None else available_cpu_count()
    thread_budget = str(max(1, available_cpus // workers))
    environ.update(
        {
            "BIOMAKER_DISABLE_RUN_QUEUE": "1",
            "BIOMAKER_LAUNCH_TOKEN": launch_token or "",
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids),
            "OMP_NUM_THREADS": environ.get("BIOMAKER_OMP_NUM_THREADS", thread_budget),
            "MKL_NUM_THREADS": environ.get("BIOMAKER_MKL_NUM_THREADS", thread_budget),
            "OPENBLAS_NUM_THREADS": environ.get(
                "BIOMAKER_OPENBLAS_NUM_THREADS", thread_budget
            ),
            "NUMEXPR_NUM_THREADS": environ.get(
                "BIOMAKER_NUMEXPR_NUM_THREADS", thread_budget
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": environ.get("BIOMAKER_PYTHONHASHSEED", "42"),
        }
    )
    return environ


def find_run_dir(logs_root: Path, requested_name: str) -> Path | None:
    matches = sorted(
        path for path in logs_root.glob(f"*-{requested_name}") if path.is_dir()
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise QueueConfigurationError(
            f"multiple run directories match requested name {requested_name!r}: {matches}"
        )
    return matches[0].resolve()


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def external_biomaker_processes(exclude: set[int] | None = None) -> list[tuple[int, str]]:
    exclude = {os.getpid(), *(exclude or set())}
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    matches: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in exclude:
            continue
        if re.search(r"(?:^|\s)-m\s+biomaker\.run(?:\s|$)", command):
            matches.append((pid, command))
    return matches


def gpu_compute_processes(gpu_uuids: Sequence[str]) -> list[tuple[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise QueueConfigurationError(
            "nvidia-smi compute-process query failed: " + result.stderr.strip()
        )
    selected = set(gpu_uuids)
    busy: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 2 and fields[1] in selected:
            busy.append((fields[0], fields[1]))
    return busy


def validate_gpu_inventory(gpu_uuids: Sequence[str], workers: int) -> None:
    if workers > len(gpu_uuids):
        raise QueueConfigurationError(
            f"parallel workers ({workers}) exceed supplied GPU UUIDs ({len(gpu_uuids)})"
        )
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        raise QueueConfigurationError("nvidia-smi GPU inventory query failed")
    available = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    missing = set(gpu_uuids) - available
    if missing:
        raise QueueConfigurationError(
            "configured GPU UUIDs are not present: " + ", ".join(sorted(missing))
        )


def validate_runtime(paths: QueuePaths) -> None:
    if not paths.runtime_root.is_dir():
        raise QueueConfigurationError(f"runtime root is absent: {paths.runtime_root}")
    if not paths.python.is_file() or not os.access(paths.python, os.X_OK):
        raise QueueConfigurationError(f"runtime Python is not executable: {paths.python}")
    env_path = runtime_env_file(paths)
    shell_has_credentials = bool(
        os.environ.get("OPENAI_BASE_URL") and os.environ.get("OPENAI_API_KEY")
    )
    if not env_path.is_file() and not shell_has_credentials:
        raise QueueConfigurationError(
            f"no runtime .env and no OPENAI_BASE_URL/OPENAI_API_KEY in the shell: {env_path}"
        )
    if env_path.is_file():
        mode = stat.S_IMODE(env_path.stat().st_mode)
        if mode & 0o077:
            raise QueueConfigurationError(
                f"credential file must not be group/world accessible: {env_path} mode={mode:o}"
            )
    probe = subprocess.run(
        [str(paths.python), "-c", "import biomaker; print(biomaker.__file__)"],
        cwd=paths.runtime_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(paths.runtime_root)},
    )
    if probe.returncode != 0:
        raise QueueConfigurationError(
            "runtime cannot import biomaker: " + (probe.stderr or probe.stdout).strip()
        )


def queue_lock_paths(
    paths: QueuePaths, *, repeat_index: int | None = None
) -> tuple[Path, Path]:
    """Return a cross-state lock followed by the state-local lock."""

    # ``repeat_index`` is accepted for a stable call surface, but admission is
    # deliberately global: the resource gate rejects every other biomaker.run,
    # so splitting this lock by repeat would reintroduce a check-then-launch race.
    del repeat_index
    coordination_root = paths.resource_root / ".fig2_medmnist_queue"
    return (
        coordination_root / "queue.global.lock",
        paths.state_dir / "queue.lock",
    )


def acquire_queue_locks(
    paths: QueuePaths, *, repeat_index: int | None = None
) -> list[tuple[Path, Any]]:
    acquired: list[tuple[Path, Any]] = []
    try:
        for index, path in enumerate(
            queue_lock_paths(paths, repeat_index=repeat_index)
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            os.chmod(path, 0o600)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                scope = "cross-state" if index == 0 else "state-local"
                raise QueueConfigurationError(
                    f"another Figure 2 queue owns the {scope} lock {path}"
                ) from exc
            acquired.append((path, handle))
    except BaseException:
        release_queue_locks(acquired)
        raise
    return acquired


def release_queue_locks(acquired: Sequence[tuple[Path, Any]]) -> None:
    for _path, handle in reversed(acquired):
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def wait_for_exclusive_resources(
    state_dir: Path,
    gpu_uuids: Sequence[str],
    poll_seconds: int,
    *,
    skip_resource_check: bool,
    state_path: Path | None = None,
    state: dict[str, Any] | None = None,
    active_task: str | None = None,
    active_stage: str | None = None,
) -> None:
    if skip_resource_check:
        return
    if (state is None) != (state_path is None):
        raise QueueConfigurationError(
            "resource-wait state and state_path must be supplied together"
        )
    last_reason: str | None = None
    entered_wait = False
    while not STOP_REQUESTED.is_set():
        processes = external_biomaker_processes()
        gpu_busy = gpu_compute_processes(gpu_uuids)
        reason = None
        if processes:
            reason = "other biomaker.run process(es): " + ",".join(
                str(pid) for pid, _command in processes
            )
        elif gpu_busy:
            reason = "selected GPU compute process(es): " + ",".join(
                f"pid={pid}@{uuid}" for pid, uuid in gpu_busy
            )
        if reason is None:
            if entered_wait and state is not None and state_path is not None:
                state["status"] = "running"
                state["active_task"] = active_task
                state["active_stage"] = active_stage
                if active_task is not None and active_stage is not None:
                    task_state = state.get("tasks", {}).get(active_task)
                    if isinstance(task_state, dict):
                        task_state["status"] = f"preparing_{active_stage}"
                save_state(state_path, state)
                write_heartbeat(
                    state_dir / "heartbeat.json",
                    state,
                    active_task=active_task,
                    active_stage=active_stage,
                    resource_wait_reason=None,
                )
            return
        entered_wait = True
        if state is not None and state_path is not None:
            state["status"] = "waiting_resources"
            state["active_task"] = active_task
            state["active_stage"] = active_stage
            if active_task is not None and active_stage is not None:
                task_state = state.get("tasks", {}).get(active_task)
                if isinstance(task_state, dict):
                    task_state["status"] = f"waiting_resources_{active_stage}"
            save_state(state_path, state)
            write_heartbeat(
                state_dir / "heartbeat.json",
                state,
                active_task=active_task,
                active_stage=active_stage,
                resource_wait_reason=reason,
            )
        if reason != last_reason:
            emit(state_dir, "waiting_for_exclusive_resources", reason=reason)
            last_reason = reason
        STOP_REQUESTED.wait(poll_seconds)
    raise KeyboardInterrupt


def _terminate_child(child: subprocess.Popen[Any], grace_seconds: int = 30) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def install_signal_handlers() -> None:
    def request_stop(signum: int, _frame: Any) -> None:
        STOP_REQUESTED.set()
        child = CURRENT_CHILD
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        print(f"received signal {signum}; stopping after child cleanup", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def initial_state(
    state_path: Path,
    registry_path: Path,
    registry_hash: str,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if state_path.is_file():
        state = read_json(state_path)
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise QueueConfigurationError(f"invalid queue state: {state_path}")
        if state.get("registry_sha256") != registry_hash:
            raise QueueConfigurationError(
                "task registry changed since queue state was created; use a new state directory"
            )
    else:
        queue_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + re.sub(r"[^a-zA-Z0-9]+", "-", socket.gethostname()).strip("-")
            + "-"
            + hashlib.sha256(str(state_path.resolve()).encode()).hexdigest()[:8]
        )
        state = {
            "schema_version": 1,
            "queue_id": queue_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "registry": str(registry_path),
            "registry_sha256": registry_hash,
            "tasks": {},
            "status": "running",
        }
    task_states = state.setdefault("tasks", {})
    for task in tasks:
        task_states.setdefault(
            str(task["task_id"]),
            {"status": "pending", "stages": {}, "data_dir": None},
        )
    atomic_write_json(state_path, state)
    return state


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_write_json(state_path, state)


def fail_closed_on_cold_start(
    state_path: Path,
    state: dict[str, Any],
    *,
    state_dir: Path,
) -> None:
    """Persistently block ambiguous in-flight stages instead of replaying them."""

    blocked: list[dict[str, Any]] = []
    for task_id, task_state in state.get("tasks", {}).items():
        if not isinstance(task_state, dict):
            continue
        stages = task_state.get("stages", {})
        if not isinstance(stages, dict):
            continue
        for stage, stage_state in stages.items():
            if not isinstance(stage_state, dict):
                continue
            observed = stage_state.get("status")
            if observed in INCOMPLETE_STAGE_STATUSES:
                error = (
                    f"cold-start recovery refused automatic replay of {task_id}/{stage}: "
                    f"persisted status is {observed!r}; PID/run artifacts and the private-"
                    "evaluation boundary require manual audit"
                )
                stage_state.update(
                    {
                        "status": RECOVERY_BLOCKED_STATUS,
                        "recovery_from_status": observed,
                        "recovery_blocked_at": utc_now(),
                        "recovery_error": error,
                    }
                )
            elif observed == RECOVERY_BLOCKED_STATUS:
                error = str(
                    stage_state.get("recovery_error")
                    or f"{task_id}/{stage} remains blocked after ambiguous cold start"
                )
            else:
                continue
            task_state["status"] = RECOVERY_BLOCKED_STATUS
            blocked.append(
                {
                    "task_id": str(task_id),
                    "stage": str(stage),
                    "error": error,
                }
            )

    if not blocked:
        return
    state["status"] = RECOVERY_BLOCKED_STATUS
    state["recovery_blocked_at"] = utc_now()
    state["recovery_errors"] = blocked
    save_state(state_path, state)
    emit(
        state_dir,
        "queue_recovery_blocked",
        blocked=len(blocked),
        error=" | ".join(item["error"] for item in blocked),
    )
    raise QueueConfigurationError(
        "queue cold-start recovery is blocked: "
        + " | ".join(item["error"] for item in blocked)
    )


def private_evaluation_claim_path(
    resource_root: Path, task_id: str, repeat_index: int
) -> Path:
    safe_task = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task_id).strip("-")
    suffix = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
    return (
        resource_root
        / ".fig2_medmnist_queue"
        / "private_evaluation_claims"
        / f"repeat-{repeat_index}-{safe_task}-{suffix}.json"
    )


def claim_private_evaluation_once(
    *,
    resource_root: Path,
    task_id: str,
    repeat_index: int,
    queue_id: str,
    requested_name: str,
    source_state_sha256: str | None,
) -> Path:
    """Claim a full-stage private-evaluation intent before its child can start."""

    path = private_evaluation_claim_path(resource_root, task_id, repeat_index)
    payload = {
        "schema_version": 1,
        "status": "private_evaluation_intent_claimed",
        "claimed_at": utc_now(),
        "task_id": task_id,
        "repeat_index": repeat_index,
        "queue_id": queue_id,
        "requested_name": requested_name,
        "source_state_payload_sha256": source_state_sha256,
        "policy": "one full-stage launch per task and repeat; never auto-retry",
    }
    try:
        _atomic_create_json_once(path, payload)
    except FileExistsError as exc:
        raise QueueConfigurationError(
            "private-evaluation one-shot claim already exists; refusing another full "
            f"stage for task={task_id!r} repeat={repeat_index}: {path}"
        ) from exc
    return path


def _private_evaluation_artifacts(run_dir: Path) -> list[str]:
    found = [
        name
        for name in sorted(PRIVATE_EVALUATION_ARTIFACT_NAMES)
        if (run_dir / name).exists()
    ]
    for directory in ("final_retrain", "private_evaluation"):
        if (run_dir / directory).exists():
            found.append(directory + "/")
    return found


def _is_private_evaluation_event(event: Mapping[str, Any]) -> bool:
    name = str(event.get("event", "")).lower()
    return (
        name.startswith("final_retrain")
        or "private_evaluation" in name
        or "private_test" in name
    )


def _retry_attempt_safety(
    *,
    task_id: str,
    stage: str,
    attempt: Mapping[str, Any],
    attempt_index: int,
    logs_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    evidence: dict[str, Any] = {
        "task_id": task_id,
        "stage": stage,
        "attempt_index": attempt_index,
        "checked_at": utc_now(),
    }
    prefix = f"{task_id}/{stage} attempt {attempt_index}"

    if stage != "canary":
        reasons.append(
            f"{prefix}: the full stage carries frozen private-evaluation intent"
        )
    if attempt.get("status") not in {"failed", "interrupted"}:
        reasons.append(f"{prefix}: attempt is not terminally failed/interrupted")
    if not attempt.get("finished_at"):
        reasons.append(f"{prefix}: missing finished_at evidence")
    if not isinstance(attempt.get("return_code"), int):
        reasons.append(f"{prefix}: missing integer child return_code evidence")

    command = attempt.get("command")
    final_overrides = (
        [str(item) for item in command if str(item).startswith("final_retrain.enabled=")]
        if isinstance(command, list)
        else []
    )
    evidence["final_retrain_overrides"] = final_overrides
    if final_overrides != ["final_retrain.enabled=false"]:
        reasons.append(
            f"{prefix}: command does not prove exactly one final_retrain.enabled=false"
        )
    for key in (
        "private_evaluation_claim",
        "private_evaluation_claimed_at",
        "private_evaluation_intent_at",
        "private_evaluation_started_at",
        "private_evaluation_finished_at",
    ):
        if attempt.get(key):
            reasons.append(f"{prefix}: recorded private-evaluation intent/state in {key}")

    launcher_log_value = attempt.get("launcher_log")
    launcher_log = Path(str(launcher_log_value)) if launcher_log_value else None
    if launcher_log is None or not launcher_log.is_file():
        reasons.append(f"{prefix}: launcher log is absent")
    else:
        evidence["launcher_log"] = file_record(launcher_log)

    source_value = attempt.get("source_state")
    source_path = Path(str(source_value)) if source_value else None
    if source_path is None or not source_path.is_file():
        reasons.append(f"{prefix}: frozen source_state evidence is absent")
    else:
        try:
            source_state = read_json(source_path)
            if not isinstance(source_state, dict) or source_state.get("stage") != "canary":
                reasons.append(f"{prefix}: source_state does not prove canary stage")
            evidence["source_state"] = file_record(source_path)
        except (OSError, ValueError) as exc:
            reasons.append(f"{prefix}: cannot read source_state evidence: {exc}")

    audit_value = attempt.get("audit")
    audit_path = Path(str(audit_value)) if audit_value else None
    if audit_path is None or not audit_path.is_file():
        reasons.append(f"{prefix}: terminal queue audit is absent")
    else:
        try:
            audit = read_json(audit_path)
            if not isinstance(audit, dict) or audit.get("stage") != "canary":
                reasons.append(f"{prefix}: terminal audit does not prove canary stage")
            elif audit.get("ok") is not False:
                reasons.append(
                    f"{prefix}: terminal audit does not explicitly record ok=false"
                )
            else:
                summary = audit.get("summary")
                if isinstance(summary, dict) and summary.get("private_result") is not None:
                    reasons.append(f"{prefix}: terminal audit records a private result")
                audit_errors = audit.get("errors")
                if isinstance(audit_errors, list) and any(
                    marker in str(error).lower()
                    for error in audit_errors
                    for marker in PRIVATE_EVALUATION_AUDIT_MARKERS
                ):
                    reasons.append(
                        f"{prefix}: terminal audit mentions private/final evaluation"
                    )
            evidence["audit"] = file_record(audit_path)
        except (OSError, ValueError) as exc:
            reasons.append(f"{prefix}: cannot read terminal audit: {exc}")

    requested_name = attempt.get("requested_name")
    if not isinstance(requested_name, str) or not requested_name:
        reasons.append(f"{prefix}: requested_name is absent")
        discovered_run_dir = None
    else:
        try:
            discovered_run_dir = find_run_dir(logs_root, requested_name)
        except QueueConfigurationError as exc:
            reasons.append(f"{prefix}: {exc}")
            discovered_run_dir = None

    recorded_value = attempt.get("run_dir")
    recorded_run_dir = Path(str(recorded_value)).resolve() if recorded_value else None
    if recorded_run_dir is not None and not recorded_run_dir.is_dir():
        reasons.append(f"{prefix}: recorded run directory is missing: {recorded_run_dir}")
        run_dir = None
    elif recorded_run_dir is not None:
        if recorded_run_dir.parent != logs_root.resolve():
            reasons.append(f"{prefix}: recorded run directory is outside logs_root")
        if discovered_run_dir is not None and discovered_run_dir != recorded_run_dir:
            reasons.append(f"{prefix}: recorded and discovered run directories differ")
        run_dir = recorded_run_dir
    else:
        run_dir = discovered_run_dir

    if run_dir is None:
        evidence["run_directory"] = "absent_for_exact_requested_name"
    else:
        evidence["run_directory"] = str(run_dir)
        artifacts = _private_evaluation_artifacts(run_dir)
        evidence["private_evaluation_artifacts"] = artifacts
        if artifacts:
            reasons.append(
                f"{prefix}: private/final artifacts exist: {', '.join(artifacts)}"
            )
        events_path = run_dir / "events.jsonl"
        if not events_path.is_file():
            reasons.append(f"{prefix}: events.jsonl is absent, so private use is unprovable")
        else:
            events, event_errors = parse_events(events_path)
            if event_errors:
                reasons.append(
                    f"{prefix}: events.jsonl is not fully parseable: {' | '.join(event_errors)}"
                )
            private_events = [
                str(event.get("event"))
                for event in events
                if _is_private_evaluation_event(event)
            ]
            evidence["private_evaluation_events"] = private_events
            if private_events:
                reasons.append(
                    f"{prefix}: private-evaluation event(s) exist: "
                    + ", ".join(private_events)
                )
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = read_json(manifest_path)
                config = manifest.get("config") if isinstance(manifest, dict) else None
                if isinstance(config, dict) and config.get("final_retrain_enabled") is not False:
                    reasons.append(
                        f"{prefix}: run_manifest does not prove final retrain disabled"
                    )
            except (OSError, ValueError) as exc:
                reasons.append(f"{prefix}: cannot read run_manifest: {exc}")

    evidence["ok"] = not reasons
    return reasons, evidence


def reset_failed_stages_for_one_retry(
    state: dict[str, Any],
    *,
    logs_root: Path,
    resource_root: Path,
    repeat_index: int,
    task_ids: set[str] | None = None,
) -> None:
    """Reset only failed canaries with positive evidence of zero private use."""

    candidates: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    rejections: list[dict[str, Any]] = []
    for task_id, task_state in state.get("tasks", {}).items():
        if task_ids is not None and str(task_id) not in task_ids:
            continue
        if not isinstance(task_state, dict):
            continue
        stages = task_state.get("stages", {})
        if not isinstance(stages, dict):
            continue
        for stage, stage_state in stages.items():
            if not isinstance(stage_state, dict) or stage_state.get("status") not in {
                "failed",
                "interrupted",
            }:
                continue
            previous = stage_state.get("previous_attempts", [])
            if not isinstance(previous, list) or not all(
                isinstance(item, dict) for item in previous
            ):
                rejections.append(
                    {
                        "task_id": str(task_id),
                        "stage": str(stage),
                        "reasons": ["previous_attempts evidence is malformed"],
                    }
                )
                continue
            current = dict(stage_state)
            current.pop("previous_attempts", None)
            attempts = [*previous, current]
            stage_reasons: list[str] = []
            stage_evidence: list[dict[str, Any]] = []
            if previous:
                stage_reasons.append(
                    f"{task_id}/{stage}: the single retry budget was already consumed"
                )
            for index, attempt in enumerate(attempts, start=1):
                reasons, evidence = _retry_attempt_safety(
                    task_id=str(task_id),
                    stage=str(stage),
                    attempt=attempt,
                    attempt_index=index,
                    logs_root=logs_root,
                )
                stage_reasons.extend(reasons)
                stage_evidence.append(evidence)
            claim_path = private_evaluation_claim_path(
                resource_root, str(task_id), repeat_index
            )
            if claim_path.exists():
                stage_reasons.append(
                    f"{task_id}/{stage}: private-evaluation intent claim exists: {claim_path}"
                )
            full_state = stages.get("full")
            if stage == "canary" and isinstance(full_state, dict) and full_state:
                stage_reasons.append(
                    f"{task_id}/{stage}: task already contains a full-stage record"
                )
            if stage_reasons:
                rejections.append(
                    {
                        "task_id": str(task_id),
                        "stage": str(stage),
                        "reasons": stage_reasons,
                        "evidence": stage_evidence,
                    }
                )
            else:
                candidates.append((task_state, stage_state, stage_evidence))

    if rejections:
        state["status"] = "retry_rejected"
        state["retry_rejected_at"] = utc_now()
        state["retry_rejections"] = rejections
        raise QueueConfigurationError(
            "--retry-failed rejected by the private-evaluation boundary: "
            + " | ".join(
                reason
                for rejection in rejections
                for reason in rejection["reasons"]
            )
        )

    authorized_at = utc_now()
    for task_state, stage_state, evidence in candidates:
        previous_attempts = list(stage_state.get("previous_attempts", []))
        previous = dict(stage_state)
        previous.pop("previous_attempts", None)
        previous["retry_safety_evidence"] = evidence[-1]
        previous_attempts.append(previous)
        stage_state.clear()
        stage_state.update(
            {
                "status": "pending",
                "previous_attempts": previous_attempts,
                "retry_authorized_at": authorized_at,
                "retry_safety_evidence": evidence,
            }
        )
        task_state["status"] = "pending"
    state.pop("retry_rejections", None)
    state.pop("retry_rejected_at", None)
    state["retry_authorized_at"] = authorized_at
    state["retry_authorized_stage_count"] = len(candidates)
    if candidates:
        state["status"] = "running"
        state.pop("failed_tasks", None)


def next_stage(task_state: Mapping[str, Any]) -> str | None:
    stages = task_state.get("stages", {})
    canary = stages.get("canary", {}) if isinstance(stages, dict) else {}
    full = stages.get("full", {}) if isinstance(stages, dict) else {}
    canary_status = canary.get("status") if isinstance(canary, dict) else None
    full_status = full.get("status") if isinstance(full, dict) else None
    blocked_statuses = {
        *INCOMPLETE_STAGE_STATUSES,
        RECOVERY_BLOCKED_STATUS,
        PRIVATE_EVALUATION_BLOCKED_STATUS,
    }
    if canary_status in blocked_statuses:
        raise QueueConfigurationError(
            f"refusing to schedule ambiguous canary state {canary_status!r}"
        )
    if full_status in blocked_statuses:
        raise QueueConfigurationError(
            f"refusing to schedule ambiguous full state {full_status!r}"
        )
    if canary_status in {"failed", "interrupted"}:
        return None
    if canary_status != "succeeded":
        return "canary"
    if full_status in {"succeeded", "failed", "interrupted"}:
        return None
    return "full"


def all_tasks_resolved(state: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]) -> bool:
    for task in tasks:
        task_state = state["tasks"][task["task_id"]]
        stages = task_state.get("stages", {})
        canary_status = stages.get("canary", {}).get("status")
        full_status = stages.get("full", {}).get("status")
        if canary_status in {"failed", "interrupted"}:
            continue
        if canary_status == "succeeded" and full_status in {
            "succeeded",
            "failed",
            "interrupted",
        }:
            continue
        return False
    return True


def write_heartbeat(
    path: Path,
    state: Mapping[str, Any],
    *,
    active_task: str | None = None,
    active_stage: str | None = None,
    child_pid: int | None = None,
    elapsed_seconds: float | None = None,
    seconds_since_event_progress: float | None = None,
    watchdog_reason: str | None = None,
    resource_wait_reason: str | None = None,
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "ts": utc_now(),
            "queue_id": state.get("queue_id"),
            "queue_status": state.get("status"),
            "queue_pid": os.getpid(),
            "active_task": active_task,
            "active_stage": active_stage,
            "child_pid": child_pid,
            "elapsed_seconds": elapsed_seconds,
            "seconds_since_event_progress": seconds_since_event_progress,
            "watchdog_reason": watchdog_reason,
            "resource_wait_reason": resource_wait_reason,
        },
    )


def run_one_stage(
    *,
    paths: QueuePaths,
    registry_path: Path,
    state_path: Path,
    state: dict[str, Any],
    task: Mapping[str, Any],
    prepared_record: Mapping[str, Any] | None,
    stage: str,
    data_dir: Path,
    steps: int,
    total_time_limit: int,
    hard_wall_clock_seconds: int,
    node_timeout: int,
    workers: int,
    gpu_uuids: Sequence[str],
    model: str,
    generate_report: bool,
    heartbeat_seconds: int,
    progress_timeout_seconds: int,
    final_retrain_progress_timeout_seconds: int,
    shutdown_grace_seconds: int,
    extra_overrides: Sequence[str],
    skip_resource_check: bool,
) -> bool:
    global CURRENT_CHILD

    task_id = str(task["task_id"])
    task_state = state["tasks"][task_id]
    stage_state = task_state.setdefault("stages", {}).setdefault(stage, {})
    stage_state_before_launch = dict(stage_state)
    task_status_before_launch = task_state.get("status", "pending")

    def cancel_before_private_claim(reason: str) -> bool:
        stage_state.clear()
        stage_state.update(stage_state_before_launch)
        task_state["status"] = task_status_before_launch
        state["status"] = "interrupted"
        state["active_task"] = None
        state["active_stage"] = None
        save_state(state_path, state)
        emit(
            paths.state_dir,
            "stage_launch_cancelled",
            task_id=task_id,
            stage=stage,
            private_claimed=False,
            reason=reason,
        )
        return False

    attempt = 1 + len(stage_state.get("previous_attempts", []))
    queue_token = re.sub(r"[^a-zA-Z0-9-]+", "-", str(state["queue_id"]))
    requested_name = (
        f"fig2-{task_id}-{model}-{steps}step-{stage}-{queue_token}-a{attempt}"
    )
    launcher_dir = paths.state_dir / "attempts"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = launcher_dir / f"{requested_name}.console.log"
    audit_path = launcher_dir / f"{requested_name}.audit.json"
    source_state_path = launcher_dir / f"{requested_name}.source_state.json"
    source_patch_path = launcher_dir / f"{requested_name}.source.patch"
    formal_validation_path = launcher_dir / f"{requested_name}.formal_validation.json"

    wait_for_exclusive_resources(
        paths.state_dir,
        gpu_uuids,
        heartbeat_seconds,
        skip_resource_check=skip_resource_check,
        state_path=state_path,
        state=state,
        active_task=task_id,
        active_stage=stage,
    )
    if STOP_REQUESTED.is_set():
        return cancel_before_private_claim("stop requested after resource admission")
    state["status"] = "running"
    state["active_task"] = task_id
    state["active_stage"] = stage
    task_state["status"] = f"preparing_{stage}"
    save_state(state_path, state)
    source_state = create_source_state(
        paths=paths,
        registry_path=registry_path,
        task=task,
        stage=stage,
        steps=steps,
        search_time_limit_seconds=total_time_limit,
        hard_wall_clock_seconds=hard_wall_clock_seconds,
        progress_timeout_seconds=progress_timeout_seconds,
        final_retrain_progress_timeout_seconds=final_retrain_progress_timeout_seconds,
        model=model,
        workers=workers,
        gpu_uuids=gpu_uuids,
        requested_name=requested_name,
        prepared_record=prepared_record,
        snapshot_path=source_state_path,
        patch_path=source_patch_path,
    )
    command = build_run_command(
        paths=paths,
        registry_path=registry_path,
        task=task,
        stage=stage,
        steps=steps,
        total_time_limit=total_time_limit,
        node_timeout=node_timeout,
        workers=workers,
        model=model,
        requested_name=requested_name,
        data_dir=data_dir,
        generate_report=generate_report,
        extra_overrides=extra_overrides,
    )
    if STOP_REQUESTED.is_set():
        return cancel_before_private_claim("stop requested during source freeze")
    private_claim_path: Path | None = None
    private_claimed_at: str | None = None
    if stage == "full":
        repeat_index = int(source_state.get("repeat_index", 1))
        try:
            private_claim_path = claim_private_evaluation_once(
                resource_root=paths.resource_root,
                task_id=task_id,
                repeat_index=repeat_index,
                queue_id=str(state["queue_id"]),
                requested_name=requested_name,
                source_state_sha256=source_state.get("source_state_payload_sha256"),
            )
        except QueueConfigurationError as exc:
            stage_state.update(
                {
                    "status": PRIVATE_EVALUATION_BLOCKED_STATUS,
                    "private_evaluation_error": str(exc),
                    "blocked_at": utc_now(),
                }
            )
            task_state["status"] = PRIVATE_EVALUATION_BLOCKED_STATUS
            state["status"] = PRIVATE_EVALUATION_BLOCKED_STATUS
            state["active_task"] = None
            state["active_stage"] = None
            save_state(state_path, state)
            emit(
                paths.state_dir,
                "private_evaluation_claim_blocked",
                task_id=task_id,
                stage=stage,
                error=str(exc),
            )
            raise
        private_claimed_at = utc_now()
    stage_state.update(
        {
            "status": "starting",
            "attempt": attempt,
            "started_at": utc_now(),
            "requested_name": requested_name,
            "data_dir": str(data_dir),
            "launcher_log": str(launcher_log),
            "command": command,
            "gpu_uuids": list(gpu_uuids),
            "parallel_workers": workers,
            "source_state": str(source_state_path),
            "source_patch": str(source_patch_path),
            "progress_timeout_seconds": progress_timeout_seconds,
            "hard_wall_clock_seconds": hard_wall_clock_seconds,
            "private_evaluation_claim": (
                str(private_claim_path) if private_claim_path is not None else None
            ),
            "private_evaluation_claimed_at": private_claimed_at,
        }
    )
    task_state["status"] = f"running_{stage}"
    save_state(state_path, state)
    emit(
        paths.state_dir,
        "stage_starting",
        task_id=task_id,
        stage=stage,
        steps=steps,
        launcher_log=launcher_log,
    )

    if STOP_REQUESTED.is_set():
        if private_claim_path is not None:
            stage_state.update(
                {
                    "status": PRIVATE_EVALUATION_BLOCKED_STATUS,
                    "private_evaluation_error": (
                        "stop requested after the one-shot private intent claim and "
                        "before child launch; automatic replay is forbidden"
                    ),
                    "blocked_at": utc_now(),
                }
            )
            task_state["status"] = PRIVATE_EVALUATION_BLOCKED_STATUS
        else:
            return cancel_before_private_claim(
                "stop requested after stage-state publication"
            )
        state["active_task"] = None
        state["active_stage"] = None
        save_state(state_path, state)
        emit(
            paths.state_dir,
            "stage_launch_cancelled",
            task_id=task_id,
            stage=stage,
            private_claimed=private_claim_path is not None,
        )
        return False

    return_code: int | None = None
    run_dir: Path | None = None
    watchdog_reason: str | None = None
    stage_started_monotonic = time.monotonic()
    last_event_progress_monotonic = stage_started_monotonic
    previous_event_signature: tuple[int, int] | None = None
    latest_event_name: str | None = None
    with launcher_log.open("ab", buffering=0) as console:
        child = subprocess.Popen(
            command,
            cwd=paths.runtime_root,
            env=child_environment(
                gpu_uuids, workers, launch_token=requested_name
            ),
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        CURRENT_CHILD = child
        stage_state.update({"status": "running", "pid": child.pid})
        save_state(state_path, state)
        emit(paths.state_dir, "stage_started", task_id=task_id, stage=stage, pid=child.pid)
        try:
            while child.poll() is None and not STOP_REQUESTED.wait(heartbeat_seconds):
                if run_dir is None:
                    run_dir = find_run_dir(paths.logs_root, requested_name)
                    if run_dir is not None:
                        install_source_state(run_dir, source_state, source_patch_path)
                        stage_state["run_dir"] = str(run_dir)
                        save_state(state_path, state)
                now = time.monotonic()
                if run_dir is not None:
                    events_path = run_dir / "events.jsonl"
                    if events_path.is_file():
                        event_stat = events_path.stat()
                        signature = (event_stat.st_mtime_ns, event_stat.st_size)
                        if signature != previous_event_signature:
                            previous_event_signature = signature
                            last_event_progress_monotonic = now
                            latest_event_name = last_valid_event_name(events_path)
                elapsed = now - stage_started_monotonic
                stalled = now - last_event_progress_monotonic
                active_progress_timeout = progress_timeout_for_event(
                    latest_event_name,
                    default_seconds=progress_timeout_seconds,
                    final_retrain_seconds=final_retrain_progress_timeout_seconds,
                )
                if elapsed >= hard_wall_clock_seconds:
                    watchdog_reason = (
                        f"hard wall-clock exceeded {hard_wall_clock_seconds}s "
                        f"(elapsed={elapsed:.1f}s)"
                    )
                elif stalled >= active_progress_timeout:
                    watchdog_reason = (
                        f"events.jsonl made no progress for {stalled:.1f}s "
                        f"(limit={active_progress_timeout}s, last_event={latest_event_name})"
                    )
                write_heartbeat(
                    paths.state_dir / "heartbeat.json",
                    state,
                    active_task=task_id,
                    active_stage=stage,
                    child_pid=child.pid,
                    elapsed_seconds=round(elapsed, 3),
                    seconds_since_event_progress=round(stalled, 3),
                    watchdog_reason=watchdog_reason,
                )
                if watchdog_reason is not None:
                    emit(
                        paths.state_dir,
                        "stage_watchdog_terminating",
                        task_id=task_id,
                        stage=stage,
                        reason=watchdog_reason,
                    )
                    _terminate_child(child, grace_seconds=shutdown_grace_seconds)
                    break
            if STOP_REQUESTED.is_set() and child.poll() is None:
                _terminate_child(child, grace_seconds=shutdown_grace_seconds)
            return_code = child.wait()
        finally:
            CURRENT_CHILD = None

    if run_dir is None:
        run_dir = find_run_dir(paths.logs_root, requested_name)
    if run_dir is not None:
        install_source_state(run_dir, source_state, source_patch_path)
    audit = (
        validate_run_artifacts(
            run_dir,
            stage=stage,
            expected_steps=steps,
            expected_model=model,
            expected_metric=str(task["primary_metric"]),
            expected_test_rows=int(task["expected_rows"]["test"]),
            expected_workers=workers,
            expected_gpu_uuids=gpu_uuids,
            require_report=generate_report,
            return_code=return_code,
        )
        if run_dir is not None
        else RunAudit(False, ("BioMaker created no matching run directory",), {})
    )
    if watchdog_reason is not None:
        audit = RunAudit(
            False,
            (watchdog_reason, *audit.errors),
            {**audit.summary, "watchdog_reason": watchdog_reason},
        )
    if run_dir is not None:
        try:
            write_formal_validation(
                run_dir=run_dir,
                audit=audit,
                stage=stage,
                task=task,
                source_state=source_state,
                prepared_record=prepared_record,
                watchdog_reason=watchdog_reason,
                attempt_path=formal_validation_path,
            )
        except Exception as exc:
            audit = RunAudit(
                False,
                (*audit.errors, f"cannot write formal_validation.json: {exc}"),
                audit.summary,
            )
    audit_payload = {
        "schema_version": 1,
        "task_id": task_id,
        "stage": stage,
        "attempt": attempt,
        "requested_name": requested_name,
        "return_code": return_code,
        "ok": audit.ok,
        "errors": list(audit.errors),
        "summary": audit.summary,
        "source_state": str(source_state_path),
        "source_patch": str(source_patch_path),
        "formal_validation": (
            str(formal_validation_path) if formal_validation_path.is_file() else None
        ),
        "watchdog_reason": watchdog_reason,
        "audited_at": utc_now(),
    }
    atomic_write_json(audit_path, audit_payload)
    stage_state.update(
        {
            "status": "succeeded" if audit.ok else "interrupted" if STOP_REQUESTED.is_set() else "failed",
            "finished_at": utc_now(),
            "return_code": return_code,
            "run_dir": str(run_dir) if run_dir is not None else None,
            "audit": str(audit_path),
            "audit_errors": list(audit.errors),
            "summary": audit.summary,
            "formal_validation": (
                str(formal_validation_path)
                if formal_validation_path.is_file()
                else None
            ),
            "watchdog_reason": watchdog_reason,
        }
    )
    if audit.ok:
        task_state["status"] = "canary_succeeded" if stage == "canary" else "succeeded"
        emit(paths.state_dir, "stage_succeeded", task_id=task_id, stage=stage, run_dir=run_dir)
    else:
        task_state["status"] = "failed"
        emit(
            paths.state_dir,
            "stage_failed",
            task_id=task_id,
            stage=stage,
            run_dir=run_dir,
            errors=" | ".join(audit.errors),
        )
    state["active_task"] = None
    state["active_stage"] = None
    save_state(state_path, state)
    return audit.ok


def print_status(state_path: Path) -> int:
    if not state_path.is_file():
        print(f"no queue state at {state_path}")
        return 1
    state = read_json(state_path)
    print(f"queue {state.get('queue_id')} status={state.get('status')} updated={state.get('updated_at')}")
    for task_id, task_state in state.get("tasks", {}).items():
        stages = task_state.get("stages", {})
        canary = stages.get("canary", {}).get("status", "pending")
        full = stages.get("full", {}).get("status", "pending")
        reason = task_state.get("data_status_reason")
        print(
            f"{task_id:20} task={task_state.get('status', 'pending'):18} "
            f"canary={canary:11} full={full:11}"
            + (f" data={reason}" if reason else "")
        )
    return 0


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(os.environ.get("BIOMAKER_FIG2_REGISTRY", DEFAULT_REGISTRY)),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(os.environ.get("BIOMAKER_RUNTIME_ROOT", REPOSITORY_ROOT)),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--workspaces-root", type=Path, default=None)
    parser.add_argument("--resource-root", type=Path, default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--prepared-manifest", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=env_int("BIOMAKER_QUEUE_POLL_SECONDS", 30),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=env_int("BIOMAKER_QUEUE_HEARTBEAT_SECONDS", 30),
    )
    parser.add_argument("--no-wait-for-data", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-resource-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> QueuePaths:
    runtime_root = args.runtime_root.expanduser().resolve()
    storage_default = Path(
        os.environ.get("BIOMAKER_STORAGE_ROOT", runtime_root.parent)
    ).expanduser().resolve()
    data_root = Path(
        args.data_root or os.environ.get(
            "BIOMAKER_DATA_ROOT",
            storage_default / "datasets" / "fig2_medmnist" / "data",
        )
    )
    logs_root = Path(
        args.logs_root or os.environ.get("BIOMAKER_LOGS_ROOT", storage_default / "logs")
    )
    workspaces_root = Path(
        args.workspaces_root
        or os.environ.get("BIOMAKER_WORKSPACES_ROOT", storage_default / "workspaces")
    )
    resource_root = Path(
        args.resource_root
        or os.environ.get(
            "BIOMAKER_RESOURCE_ROOT", storage_default / "workspaces" / "_shared"
        )
    )
    state_dir = Path(
        args.state_dir
        or os.environ.get("BIOMAKER_FIG2_STATE_DIR", storage_default / "state" / "fig2_medmnist")
    )
    python = runtime_python(runtime_root, args.python)
    return QueuePaths(
        runtime_root=runtime_root,
        data_root=data_root.expanduser().resolve(),
        logs_root=logs_root.expanduser().resolve(),
        workspaces_root=workspaces_root.expanduser().resolve(),
        resource_root=resource_root.expanduser().resolve(),
        state_dir=state_dir.expanduser().resolve(),
        # Keep the virtualenv entry-point path intact.  Resolving this symlink
        # can turn ``.venv/bin/python`` into uv's base interpreter and bypass
        # the virtualenv (and its installed dependencies) entirely.
        python=python.expanduser().absolute(),
    )


def queue_main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.extra_override:
        raise QueueConfigurationError(
            "--extra-override is disabled for the frozen Figure 2 protocol"
        )
    registry_path = args.registry.expanduser().resolve()
    registry = load_registry(registry_path)
    selected = set(args.only) if args.only else None
    tasks = active_tasks(registry, selected)
    paths = resolve_paths(args)
    state_path = paths.state_dir / "queue_state.json"
    if args.status:
        return print_status(state_path)
    if not args.dry_run:
        require_supported_tasks(tasks)

    defaults = registry["defaults"]
    model = os.environ.get("BIOMAKER_MODEL", str(defaults["model"]))
    if model != "deepseek-v4-flash":
        raise QueueConfigurationError(
            "the frozen Figure 2 queue requires model deepseek-v4-flash"
        )
    gpu_uuids, workers = resolve_compute_profile(python=paths.python)
    repeat_index = env_int("BIOMAKER_REPEAT_INDEX", 1)
    canary_steps = env_int("BIOMAKER_CANARY_STEPS", int(defaults["canary_steps"]))
    full_steps = env_int("BIOMAKER_FULL_STEPS", int(defaults["full_steps"]))
    if canary_steps != 4:
        raise QueueConfigurationError("the frozen canary protocol requires exactly 4 steps")
    if full_steps != 35:
        raise QueueConfigurationError("the frozen full Figure 2 protocol requires exactly 35 steps")
    node_timeout = env_int(
        "BIOMAKER_NODE_TIMEOUT", int(defaults["per_node_timeout_seconds"])
    )
    canary_limit = env_int(
        "BIOMAKER_CANARY_TOTAL_TIMEOUT",
        int(defaults["canary_total_time_limit_seconds"]),
    )
    full_limit = env_int(
        "BIOMAKER_FULL_TOTAL_TIMEOUT", int(defaults["full_total_time_limit_seconds"])
    )
    canary_hard_limit = env_int(
        "BIOMAKER_CANARY_HARD_WALL_CLOCK",
        int(defaults["canary_hard_wall_clock_seconds"]),
    )
    full_hard_limit = env_int(
        "BIOMAKER_FULL_HARD_WALL_CLOCK",
        int(defaults["full_hard_wall_clock_seconds"]),
    )
    progress_timeout = env_int(
        "BIOMAKER_PROGRESS_TIMEOUT", int(defaults["progress_timeout_seconds"])
    )
    final_retrain_progress_timeout = env_int(
        "BIOMAKER_FINAL_RETRAIN_PROGRESS_TIMEOUT",
        int(defaults["final_retrain_progress_timeout_seconds"]),
    )
    shutdown_grace = env_int(
        "BIOMAKER_SHUTDOWN_GRACE", int(defaults["shutdown_grace_seconds"])
    )
    final_retrain_timeout = (node_timeout * 3 + 1) // 2
    if canary_hard_limit <= canary_limit:
        raise QueueConfigurationError(
            "canary outer hard wall-clock must leave setup headroom beyond search time"
        )
    if full_hard_limit <= full_limit + final_retrain_timeout:
        raise QueueConfigurationError(
            "full outer hard wall-clock must leave setup and final-retrain headroom"
        )
    if final_retrain_progress_timeout <= final_retrain_timeout:
        raise QueueConfigurationError(
            "final-retrain progress watchdog must exceed the runtime retrain timeout"
        )
    if workers > 4:
        raise QueueConfigurationError(
            "one Figure 2 run supports at most four workers; use separate repeats for additional GPUs"
        )
    if workers != len(gpu_uuids):
        raise QueueConfigurationError(
            "parallel worker count must equal the exact GPU UUID mask size"
        )

    prepared_path = args.prepared_manifest
    if prepared_path is None and os.environ.get("BIOMAKER_PREPARED_MANIFEST"):
        prepared_path = Path(os.environ["BIOMAKER_PREPARED_MANIFEST"])
    if prepared_path is None:
        candidate = paths.data_root.parent / "suite_manifest.json"
        prepared_path = candidate
    prepared_path = prepared_path.expanduser().resolve() if prepared_path else None
    prepared = load_prepared_manifest(prepared_path)

    tracker = ReadinessTracker()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol_id": registry["protocol_id"],
                    "model": model,
                    "canary_steps": canary_steps,
                    "full_steps": full_steps,
                    "canary_search_time_limit_seconds": canary_limit,
                    "full_search_time_limit_seconds": full_limit,
                    "canary_hard_wall_clock_seconds": canary_hard_limit,
                    "full_hard_wall_clock_seconds": full_hard_limit,
                    "progress_timeout_seconds": progress_timeout,
                    "final_retrain_progress_timeout_seconds": final_retrain_progress_timeout,
                    "shutdown_grace_seconds": shutdown_grace,
                    "repeat_index": repeat_index,
                    "parallel_workers": workers,
                    "gpu_uuids": gpu_uuids,
                    "runtime_root": str(paths.runtime_root),
                    "data_root": str(paths.data_root),
                    "logs_root": str(paths.logs_root),
                    "workspaces_root": str(paths.workspaces_root),
                    "resource_root": str(paths.resource_root),
                    "state_dir": str(paths.state_dir),
                    "prepared_manifest": str(prepared_path) if prepared_path else None,
                    "tasks": [
                        {
                            "order": task["order"],
                            "task_id": task["task_id"],
                            "data_dir": str(resolve_dataset_dir(task, paths.data_root, prepared)[0]),
                            "primary_metric": task["primary_metric"],
                            "evaluation_supported": evaluation_support_error(task) is None,
                            "evaluation_support_reason": evaluation_support_error(task),
                            "stages": ["canary", "full"],
                        }
                        for task in tasks
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_root.mkdir(parents=True, exist_ok=True)
    paths.workspaces_root.mkdir(parents=True, exist_ok=True)
    acquired_locks = acquire_queue_locks(paths, repeat_index=repeat_index)
    try:
        validate_runtime(paths)
        if not args.skip_resource_check:
            validate_gpu_inventory(gpu_uuids, workers)
        install_signal_handlers()
        state_existed = state_path.is_file()
        state = initial_state(
            state_path, registry_path, sha256_file(registry_path), tasks
        )
        if state_existed:
            fail_closed_on_cold_start(
                state_path, state, state_dir=paths.state_dir
            )
        if args.retry_failed:
            try:
                reset_failed_stages_for_one_retry(
                    state,
                    logs_root=paths.logs_root,
                    resource_root=paths.resource_root,
                    repeat_index=repeat_index,
                    task_ids={str(task["task_id"]) for task in tasks},
                )
            except QueueConfigurationError as exc:
                save_state(state_path, state)
                emit(
                    paths.state_dir,
                    "queue_retry_rejected",
                    error=str(exc),
                )
                raise
            save_state(state_path, state)
        emit(
            paths.state_dir,
            "queue_started",
            queue_id=state["queue_id"],
            tasks=len(tasks),
            model=model,
            workers=workers,
            gpu_uuids=",".join(gpu_uuids),
            repeat_index=repeat_index,
        )
    except BaseException:
        release_queue_locks(acquired_locks)
        raise

    try:
        while not STOP_REQUESTED.is_set():
            # The independent packager publishes each task atomically. Refresh on
            # every scheduling decision so strict FIFO sees a just-rebuilt entry.
            prepared = load_prepared_manifest(prepared_path)
            if all_tasks_resolved(state, tasks):
                failed = [
                    task["task_id"]
                    for task in tasks
                    if state["tasks"][task["task_id"]].get("status") == "failed"
                ]
                state["status"] = "completed_with_failures" if failed else "completed"
                state["active_task"] = None
                state["active_stage"] = None
                state["failed_tasks"] = failed
                save_state(state_path, state)
                write_heartbeat(paths.state_dir / "heartbeat.json", state)
                emit(paths.state_dir, "queue_finished", status=state["status"], failed=failed)
                return 1 if failed else 0

            chosen: tuple[
                dict[str, Any], str, DatasetCheck, dict[str, Any] | None
            ] | None = None
            for task in tasks:
                task_id = str(task["task_id"])
                task_state = state["tasks"][task_id]
                stage = next_stage(task_state)
                if stage is None:
                    continue
                data_dir, prepared_record = resolve_dataset_dir(task, paths.data_root, prepared)
                check = tracker.check(
                    task,
                    data_dir,
                    prepared_record,
                    expected_cache_root=paths.resource_root / "split_builder",
                )
                task_state["data_dir"] = str(data_dir)
                task_state["data_status_reason"] = check.reason
                task_state["data_row_counts"] = check.row_counts
                if check.ready:
                    chosen = (task, stage, check, prepared_record)
                    break
                task_state["status"] = "waiting_data"
            save_state(state_path, state)

            if chosen is None:
                if args.no_wait_for_data:
                    state["status"] = "waiting_data"
                    state["active_task"] = None
                    state["active_stage"] = None
                    save_state(state_path, state)
                    emit(paths.state_dir, "queue_idle_exit", reason="no prepared task is ready")
                    return 3
                state["status"] = "waiting_data"
                state["active_task"] = None
                state["active_stage"] = None
                save_state(state_path, state)
                write_heartbeat(paths.state_dir / "heartbeat.json", state)
                emit(paths.state_dir, "waiting_for_data", poll_seconds=args.poll_seconds)
                STOP_REQUESTED.wait(args.poll_seconds)
                continue

            task, stage, check, prepared_record = chosen
            state["status"] = "running"
            save_state(state_path, state)
            steps = canary_steps if stage == "canary" else full_steps
            total_limit = canary_limit if stage == "canary" else full_limit
            hard_wall_limit = (
                canary_hard_limit if stage == "canary" else full_hard_limit
            )
            generate_report = bool(
                defaults[
                    "generate_report_for_canary"
                    if stage == "canary"
                    else "generate_report_for_full"
                ]
            )
            run_one_stage(
                paths=paths,
                registry_path=registry_path,
                state_path=state_path,
                state=state,
                task=task,
                prepared_record=prepared_record,
                stage=stage,
                data_dir=check.path,
                steps=steps,
                total_time_limit=total_limit,
                hard_wall_clock_seconds=hard_wall_limit,
                node_timeout=node_timeout,
                workers=workers,
                gpu_uuids=gpu_uuids,
                model=model,
                generate_report=generate_report,
                heartbeat_seconds=args.heartbeat_seconds,
                progress_timeout_seconds=progress_timeout,
                final_retrain_progress_timeout_seconds=final_retrain_progress_timeout,
                shutdown_grace_seconds=shutdown_grace,
                extra_overrides=args.extra_override,
                skip_resource_check=args.skip_resource_check,
            )
        state["status"] = "interrupted"
        state["active_task"] = None
        state["active_stage"] = None
        save_state(state_path, state)
        write_heartbeat(paths.state_dir / "heartbeat.json", state)
        emit(paths.state_dir, "queue_interrupted")
        return 130
    finally:
        child = CURRENT_CHILD
        if child is not None:
            _terminate_child(child)
        release_queue_locks(acquired_locks)


def main() -> None:
    try:
        raise SystemExit(queue_main())
    except QueueConfigurationError as exc:
        print(f"Figure 2 queue configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
