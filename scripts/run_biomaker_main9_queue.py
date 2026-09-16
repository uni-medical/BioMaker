#!/usr/bin/env python3
"""Run the frozen BioMaker Main-9 suite as a durable single-task FIFO.

The queue deliberately reuses the already-tested lifecycle, process, provenance,
and artifact checks from ``run_fig2_medmnist_queue.py`` without changing that
frozen protocol.  Main-9 supplies its own registry/readiness contract, source
state, private-metric mapping, one-shot private-evaluation namespace, and queue
state.  A dataset is schedulable only after its packager publishes an atomic
``suite_manifest.json`` record and all corresponding cache artifacts.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import socket
import stat
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
sys.path.insert(0, str(SCRIPT_PATH.parent))
import run_fig2_medmnist_queue as _harness  # noqa: E402


PROTOCOL_ID = "biomaker-main9-v1"
DEFAULT_REGISTRY = WORKSPACE_ROOT / "experiments" / "main9" / "tasks.json"
DEFAULT_TARGET_ROOT = WORKSPACE_ROOT
FROZEN_TASK_ORDER = (
    "monuseg",
    "kvasir_seg",
    "busi",
    "bccd",
    "fracatlas",
    "dentex",
    "fmd",
    "aeropath",
    "mp_idb",
)
FROZEN_TASK_CONTRACT = {
    "monuseg": ("segmentation", "aji"),
    "kvasir_seg": ("segmentation", "binary_dice"),
    "busi": ("segmentation", "binary_dice"),
    "bccd": ("detection", "coco_map"),
    "fracatlas": ("detection", "coco_map"),
    "dentex": ("detection", "coco_map"),
    "fmd": ("enhancement", "psnr"),
    "aeropath": ("segmentation", "binary_dice"),
    "mp_idb": ("detection", "coco_map"),
}
PRIVATE_METRIC_BY_PRIMARY = {
    "aji": "private_mean_aji",
    "binary_dice": "private_binary_dice",
    "coco_map": "private_coco_map",
    "psnr": "private_psnr",
}
FORBIDDEN_TASK_TOKENS = (
    "aapm",
    "calgary",
    "tdsc",
    "organ3d",
    "organamnist3d",
    "organcmnist3d",
    "organsmnist3d",
)
READY_STATUSES = {"ready", "complete", "completed", "prepared", "ok", "success"}
BASE_VALIDATE_RUN_ARTIFACTS = _harness.validate_run_artifacts
_AUDIT_TASK_CONTEXT: dict[str, Any] | None = None

QueueConfigurationError = _harness.QueueConfigurationError
RunAudit = _harness.RunAudit
QueuePaths = _harness.QueuePaths


@dataclass(frozen=True)
class DatasetCheck:
    ready: bool
    reason: str
    path: Path | None
    row_counts: dict[str, int] | None = None


def _task_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _is_forbidden_task(value: str) -> bool:
    token = _task_token(value)
    return any(_task_token(blocked) in token for blocked in FORBIDDEN_TASK_TOKENS)


def load_registry(
    path: Path, *, require_descriptions_for: set[str] | None = None
) -> dict[str, Any]:
    try:
        payload = _harness.read_json(path)
    except (OSError, ValueError) as exc:
        raise QueueConfigurationError(f"cannot read task registry {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise QueueConfigurationError("Main-9 registry must be a schema_version=1 object")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise QueueConfigurationError(
            f"Main-9 registry protocol_id must be {PROTOCOL_ID!r}"
        )
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(item, dict) for item in tasks):
        raise QueueConfigurationError("Main-9 registry tasks must be a list of objects")
    if len(tasks) != len(FROZEN_TASK_ORDER):
        raise QueueConfigurationError("Main-9 registry must contain exactly nine tasks")

    ordered = tuple(str(task.get("task_id", "")) for task in sorted(
        tasks, key=lambda item: (int(item.get("order", 10**9)), str(item.get("task_id", "")))
    ))
    if ordered != FROZEN_TASK_ORDER:
        raise QueueConfigurationError("Main-9 tasks do not match the frozen order")
    if any(_is_forbidden_task(str(task.get("task_id", ""))) for task in tasks):
        raise QueueConfigurationError(
            "AAPM, Calgary-Campinas, TDSC-ABUS, and Organ3D must not enter Main-9"
        )

    registry_root = path.parent
    seen: set[str] = set()
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        if not task_id or task_id in seen:
            raise QueueConfigurationError("Main-9 task_id values must be non-empty and unique")
        seen.add(task_id)
        if task.get("status") != "active":
            raise QueueConfigurationError(f"Main-9 task {task_id!r} must remain active")
        family, metric = FROZEN_TASK_CONTRACT[task_id]
        if task.get("task_family") != family or task.get("primary_metric") != metric:
            raise QueueConfigurationError(
                f"Main-9 contract mismatch for {task_id}: expected {family}/{metric}"
            )
        if require_descriptions_for is None or task_id in require_descriptions_for:
            desc_path = (registry_root / str(task.get("desc_file", ""))).resolve()
            if not desc_path.is_file() or desc_path.stat().st_size == 0:
                raise QueueConfigurationError(
                    f"missing non-empty Main-9 task description for {task_id}: {desc_path}"
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
        and (selected is None or str(task.get("task_id")) in selected)
    ]
    tasks.sort(key=lambda item: (int(item["order"]), str(item["task_id"])))
    if selected is not None:
        missing = selected - {str(task["task_id"]) for task in tasks}
        if missing:
            raise QueueConfigurationError(
                "--only contains unknown/non-active Main-9 task(s): "
                + ", ".join(sorted(missing))
            )
    return tasks


def _manifest_records(payload: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    container = payload.get("tasks", payload.get("datasets", payload.get("records", payload)))
    if isinstance(container, list):
        return [dict(item) for item in container if isinstance(item, dict)]
    if isinstance(container, dict):
        records: list[dict[str, Any]] = []
        for key, value in container.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("task_id", key)
                records.append(record)
        return records
    return []


def load_prepared_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    payload = _harness.read_json(path)
    records: dict[str, dict[str, Any]] = {}
    for record in _manifest_records(payload):
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        normalized = dict(record)
        normalized["_manifest_root"] = str(path.parent.resolve())
        normalized["_manifest_path"] = str(path.resolve())
        records[task_id] = normalized
    return records


def resolve_dataset_dir(
    task: Mapping[str, Any], prepared: Mapping[str, dict[str, Any]]
) -> tuple[Path | None, dict[str, Any] | None]:
    record = prepared.get(str(task["task_id"]))
    if record is None:
        return None, None
    raw = record.get("data_dir") or record.get("task_data_dir")
    if not isinstance(raw, str) or not raw:
        return None, record
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path(str(record["_manifest_root"])) / candidate
    return candidate.resolve(), record


def _resolve_record_path(record: Mapping[str, Any], value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(str(record.get("_manifest_root", "."))) / path
    return path.resolve()


def _positive_split_counts(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict) or set(value) not in (
        {"train", "score_val", "test"},
        {"train", "val", "test"},
    ):
        return None
    result: dict[str, int] = {}
    for name, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            return None
        # Generic BioMaker packages call the evaluator-controlled validation
        # partition ``score_val``.  Normalize it to ``val`` only at the queue
        # boundary because the downstream lifecycle audit uses train/val/test.
        result["val" if name == "score_val" else str(name)] = raw
    return result


class ReadinessTracker:
    """Fail closed until a generic Main-9 package and split cache are complete."""

    def check(
        self,
        task: Mapping[str, Any],
        data_dir: Path | None,
        prepared_record: Mapping[str, Any] | None,
        *,
        expected_cache_root: Path,
    ) -> DatasetCheck:
        task_id = str(task["task_id"])
        unsupported = _harness.evaluation_support_error(task)
        if unsupported is not None:
            return DatasetCheck(False, unsupported, data_dir)
        if prepared_record is None:
            return DatasetCheck(False, "task is absent from suite_manifest.json", None)
        status_value = str(prepared_record.get("status", "")).strip().lower()
        if status_value not in READY_STATUSES:
            reason = prepared_record.get("reason") or prepared_record.get("blocked_reason")
            suffix = f": {reason}" if reason else ""
            return DatasetCheck(False, f"manifest status={status_value or 'missing'}{suffix}", data_dir)
        if data_dir is None:
            return DatasetCheck(False, "ready manifest record has no data_dir", None)
        if not data_dir.is_dir():
            return DatasetCheck(False, f"data_dir is absent: {data_dir}", data_dir)
        if (data_dir / "private_gt").exists():
            return DatasetCheck(False, "agent-visible data_dir contains private_gt", data_dir)
        if not (data_dir / "task_metadata.json").is_file():
            return DatasetCheck(False, "task_metadata.json is absent", data_dir)

        ready_path = _resolve_record_path(prepared_record, prepared_record.get("ready_marker"))
        if ready_path is None:
            ready_path = data_dir / "_READY.json"
        try:
            ready = _harness.read_json(ready_path)
        except (OSError, ValueError) as exc:
            return DatasetCheck(False, f"cannot read readiness marker: {exc}", data_dir)
        if not isinstance(ready, dict) or str(ready.get("status", "")).lower() not in READY_STATUSES:
            return DatasetCheck(False, "readiness marker is not ready", data_dir)
        if ready.get("task_id") != task_id:
            return DatasetCheck(False, "readiness marker task_id mismatch", data_dir)

        registry_path = Path(str(task["_registry_path"]))
        desc_path = (registry_path.parent / str(task["desc_file"])).resolve()
        expected_desc_hash = _harness.sha256_file(desc_path)
        expected_family = str(task["task_family"])
        expected_metric = str(task["primary_metric"])
        for source, label in ((prepared_record, "suite manifest"), (ready, "readiness marker")):
            if source.get("task_text_sha256") != expected_desc_hash:
                return DatasetCheck(False, f"{label} task text digest mismatch", data_dir)
            if source.get("task_family") != expected_family:
                return DatasetCheck(False, f"{label} task family mismatch", data_dir)
            if source.get("primary_metric") != expected_metric:
                return DatasetCheck(False, f"{label} primary metric mismatch", data_dir)

        counts = _positive_split_counts(prepared_record.get("counts"))
        ready_counts = _positive_split_counts(ready.get("counts"))
        if counts is None or ready_counts != counts:
            return DatasetCheck(
                False,
                "train/score_val/test counts are absent, invalid, or inconsistent",
                data_dir,
            )

        cache_key = prepared_record.get("cache_key")
        cache_entry = _resolve_record_path(prepared_record, prepared_record.get("cache_entry"))
        if not isinstance(cache_key, str) or not cache_key or cache_entry is None:
            return DatasetCheck(False, "split cache key/entry is absent", data_dir, counts)
        expected_entry = (expected_cache_root / cache_key).resolve()
        if cache_entry != expected_entry:
            return DatasetCheck(
                False,
                f"split cache is outside resource_root: {cache_entry} != {expected_entry}",
                data_dir,
                counts,
            )
        try:
            cache_manifest = _harness.read_json(cache_entry / "manifest.json")
            verifier = _harness.read_json(cache_entry / "verifier.json")
        except (OSError, ValueError) as exc:
            return DatasetCheck(False, f"split cache is incomplete: {exc}", data_dir, counts)
        if not isinstance(cache_manifest, dict) or cache_manifest.get("verifier_ok") is not True:
            return DatasetCheck(False, "split cache manifest is not verified", data_dir, counts)
        if cache_manifest.get("dataset") != task_id:
            return DatasetCheck(False, "split cache dataset mismatch", data_dir, counts)
        if cache_manifest.get("task_text_sha256") != expected_desc_hash:
            return DatasetCheck(False, "split cache task text digest mismatch", data_dir, counts)
        if cache_manifest.get("primary_metric") not in (None, expected_metric):
            return DatasetCheck(False, "split cache primary metric mismatch", data_dir, counts)
        if not isinstance(verifier, dict) or verifier.get("ok") is not True:
            return DatasetCheck(False, "split cache verifier does not record ok=true", data_dir, counts)
        required = (
            "split_manifest.json",
            "biomaker_splits/agent_train.csv",
            "biomaker_splits/score_val.csv",
            "biomaker_splits/agent_test.csv",
            "private_gt/score_val.csv",
            "private_gt/test.csv",
            "private_gt/private_gt.json",
        )
        missing = [name for name in required if not (cache_entry / name).is_file()]
        if missing:
            return DatasetCheck(
                False, "split cache misses required artifact(s): " + ", ".join(missing), data_dir, counts
            )
        return DatasetCheck(True, "ready", data_dir, counts)


def choose_next_ready(
    tasks: Sequence[dict[str, Any]],
    state: dict[str, Any],
    prepared: Mapping[str, dict[str, Any]],
    tracker: ReadinessTracker,
    *,
    expected_cache_root: Path,
) -> tuple[dict[str, Any], str, DatasetCheck, dict[str, Any]] | None:
    """Choose the first ready unresolved task without blocking on earlier data."""

    for task in tasks:
        task_id = str(task["task_id"])
        task_state = state["tasks"][task_id]
        stage = _harness.next_stage(task_state)
        if stage is None:
            continue
        data_dir, record = resolve_dataset_dir(task, prepared)
        check = tracker.check(
            task,
            data_dir,
            record,
            expected_cache_root=expected_cache_root,
        )
        task_state["data_dir"] = str(data_dir) if data_dir is not None else None
        task_state["data_status_reason"] = check.reason
        task_state["data_row_counts"] = check.row_counts
        if check.ready and record is not None:
            return task, stage, check, record
        task_state["status"] = "waiting_data"
    return None


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
    if prepared_record is None:
        raise QueueConfigurationError("cannot freeze source state without a prepared record")
    desc_path = (registry_path.parent / str(task["desc_file"])).resolve()
    data_dir, _ = resolve_dataset_dir(task, {str(task["task_id"]): dict(prepared_record)})
    ready_path = _resolve_record_path(prepared_record, prepared_record.get("ready_marker"))
    if ready_path is None and data_dir is not None:
        ready_path = data_dir / "_READY.json"
    ready_payload = _harness.read_json(ready_path) if ready_path and ready_path.is_file() else {}
    cache_entry, cache_artifacts = _harness._split_cache_artifacts(prepared_record)
    git_state = _harness._git_source_state(paths.runtime_root, patch_path)
    thread_budget = max(1, _harness.available_cpu_count() // workers)
    manifest_path = _resolve_record_path(prepared_record, prepared_record.get("_manifest_path"))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": _harness.utc_now(),
        "protocol_id": PROTOCOL_ID,
        "requested_name": requested_name,
        "stage": stage,
        "evidence_tier": "formal" if stage == "full" else "engineering_only",
        "task_id": task["task_id"],
        "task_family": task["task_family"],
        "primary_metric": task["primary_metric"],
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
        },
        "runtime": {
            "root": str(paths.runtime_root),
            "python_executable": str(paths.python),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "source_tree": _harness._runtime_source_tree(paths.runtime_root),
        },
        "git": git_state,
        "dependencies": {
            "pyproject.toml": _harness.file_record(paths.runtime_root / "pyproject.toml"),
            "uv.lock": _harness.file_record(paths.runtime_root / "uv.lock"),
        },
        "orchestrator": {
            "queue_script": _harness.file_record(SCRIPT_PATH),
            "task_registry": _harness.file_record(registry_path),
            "task_description": _harness.file_record(desc_path),
        },
        "dataset": {
            "data_dir": str(data_dir) if data_dir is not None else None,
            "suite_manifest": (
                _harness.file_record(manifest_path) if manifest_path is not None else {"present": False}
            ),
            "ready_marker": (
                _harness.file_record(ready_path) if ready_path is not None else {"present": False}
            ),
            "source_sha256": prepared_record.get("source_sha256"),
            "counts": prepared_record.get("counts"),
            "ready_contract": {
                key: ready_payload.get(key)
                for key in (
                    "status",
                    "task_id",
                    "task_family",
                    "primary_metric",
                    "counts",
                    "task_text_sha256",
                    "source_sha256",
                )
            },
        },
        "split_cache": {
            "root": str(cache_entry) if cache_entry is not None else None,
            "artifacts": cache_artifacts,
        },
        "credentials": _harness._credential_metadata(paths),
    }
    payload["source_state_payload_sha256"] = _harness.sha256_json(payload)
    _harness.atomic_write_json(snapshot_path, payload)
    os.chmod(snapshot_path, 0o600)
    return payload


def _private_result(path: Path, *, history: bool) -> dict[str, Any] | None:
    try:
        payload = _harness.read_json(path)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if history:
        entries = payload.get("history")
        if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
            return None
        private = entries[0].get("private_metrics")
    else:
        private = payload.get("private_metrics")
    results = private.get("results") if isinstance(private, dict) else None
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        return None
    return results[0]


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
    base_audit = BASE_VALIDATE_RUN_ARTIFACTS(
        run_dir,
        stage=stage,
        expected_steps=expected_steps,
        expected_model=expected_model,
        expected_metric=expected_metric,
        expected_test_rows=expected_test_rows,
        expected_workers=expected_workers,
        expected_gpu_uuids=expected_gpu_uuids,
        require_report=require_report,
        return_code=return_code,
    )
    errors = list(base_audit.errors)
    summary = dict(base_audit.summary)
    expected_private = PRIVATE_METRIC_BY_PRIMARY.get(expected_metric)
    if expected_private is None:
        errors.append(f"Main-9 has no private metric mapping for {expected_metric!r}")
    elif stage == "full":
        for source_name, path, history in (
            ("final_retrain_result", run_dir / "final_retrain_result.json", False),
            ("private_metrics", run_dir / "private_metrics.json", True),
        ):
            # The frozen Fig.2 auditor intentionally knows classification metrics
            # only. Remove solely its exact ``expected None`` diagnostic after
            # independently enforcing Main-9's stricter family-specific mapping.
            expected_none = (
                f"{source_name} metric_name is {expected_private!r}, expected None"
            )
            errors = [error for error in errors if error != expected_none]
            result = _private_result(path, history=history)
            actual = result.get("metric_name") if isinstance(result, dict) else None
            if actual != expected_private:
                errors.append(
                    f"{source_name} metric_name is {actual!r}, expected {expected_private!r}"
                )
        summary["expected_private_metric"] = expected_private

    try:
        source_state = _harness.read_json(run_dir / "source_state.json")
        if not isinstance(source_state, dict) or source_state.get("protocol_id") != PROTOCOL_ID:
            errors.append("source_state does not record the Main-9 protocol")
        context = _AUDIT_TASK_CONTEXT or {}
        if context and source_state.get("task_id") != context.get("task_id"):
            errors.append("source_state task_id differs from the scheduled Main-9 task")
        if context and source_state.get("task_family") != context.get("task_family"):
            errors.append("source_state task_family differs from the scheduled Main-9 task")
    except (OSError, ValueError) as exc:
        errors.append(f"cannot validate Main-9 source state: {exc}")
    return RunAudit(not errors, tuple(errors), summary)


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
    cache_entry, cache_artifacts = _harness._split_cache_artifacts(prepared_record)
    split_manifest: dict[str, Any] | None = None
    if cache_entry is not None and (cache_entry / "split_manifest.json").is_file():
        loaded = _harness.read_json(cache_entry / "split_manifest.json")
        split_manifest = loaded if isinstance(loaded, dict) else None
    names = [
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
        names.extend(("final_retrain_result.json", "final_submission.csv", "private_metrics.json"))
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "validated_at": _harness.utc_now(),
        "task_id": task["task_id"],
        "task_family": task["task_family"],
        "primary_metric": task["primary_metric"],
        "stage": stage,
        "evidence_tier": "formal" if stage == "full" else "engineering_only",
        "eligible_for_main9": bool(stage == "full" and audit.ok),
        "ok": audit.ok,
        "errors": list(audit.errors),
        "audit_summary": audit.summary,
        "watchdog_reason": watchdog_reason,
        "source_state_payload_sha256": source_state.get("source_state_payload_sha256"),
        "source_state_file": _harness.file_record(run_dir / "source_state.json"),
        "split_manifest": split_manifest,
        "split_cache_artifacts": cache_artifacts,
        "run_artifacts": {name: _harness.file_record(run_dir / name) for name in names},
        "private_result": audit.summary.get("private_result"),
    }
    payload["formal_validation_payload_sha256"] = _harness.sha256_json(payload)
    _harness.atomic_write_json(attempt_path, payload)
    _harness.atomic_write_json(run_dir / "formal_validation.json", payload)
    return payload


def private_evaluation_claim_path(
    resource_root: Path, task_id: str, repeat_index: int
) -> Path:
    safe_task = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task_id).strip("-")
    suffix = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8]
    return (
        resource_root
        / ".biomaker_main9_queue"
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
    path = private_evaluation_claim_path(resource_root, task_id, repeat_index)
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "private_evaluation_intent_claimed",
        "claimed_at": _harness.utc_now(),
        "task_id": task_id,
        "repeat_index": repeat_index,
        "queue_id": queue_id,
        "requested_name": requested_name,
        "source_state_payload_sha256": source_state_sha256,
        "policy": "one Main-9 full-stage launch per task and repeat; never auto-retry",
    }
    try:
        _harness._atomic_create_json_once(path, payload)
    except FileExistsError as exc:
        raise QueueConfigurationError(
            "Main-9 private-evaluation one-shot claim already exists: " + str(path)
        ) from exc
    return path


@contextmanager
def _main9_harness_hooks(task: Mapping[str, Any]):
    global _AUDIT_TASK_CONTEXT
    previous_context = _AUDIT_TASK_CONTEXT
    originals = {
        "create_source_state": _harness.create_source_state,
        "validate_run_artifacts": _harness.validate_run_artifacts,
        "write_formal_validation": _harness.write_formal_validation,
        "claim_private_evaluation_once": _harness.claim_private_evaluation_once,
        "private_evaluation_claim_path": _harness.private_evaluation_claim_path,
    }
    _AUDIT_TASK_CONTEXT = dict(task)
    _harness.create_source_state = create_source_state
    _harness.validate_run_artifacts = validate_run_artifacts
    _harness.write_formal_validation = write_formal_validation
    _harness.claim_private_evaluation_once = claim_private_evaluation_once
    _harness.private_evaluation_claim_path = private_evaluation_claim_path
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(_harness, name, value)
        _AUDIT_TASK_CONTEXT = previous_context


def run_one_stage(
    *,
    task: Mapping[str, Any],
    check: DatasetCheck,
    **kwargs: Any,
) -> bool:
    if check.path is None or check.row_counts is None:
        raise QueueConfigurationError("Main-9 stage cannot start without frozen row counts")
    task_with_counts = dict(task)
    task_with_counts["expected_rows"] = dict(check.row_counts)
    with _main9_harness_hooks(task_with_counts):
        return _harness.run_one_stage(
            task=task_with_counts,
            data_dir=check.path,
            **kwargs,
        )


def _gpu_inventory() -> tuple[str, ...]:
    return _harness.gpu_inventory()


def resolve_compute_profile(*, python: Path | None = None) -> tuple[tuple[str, ...], int]:
    return _harness.resolve_compute_profile(_gpu_inventory, python=python)


def initial_state(
    state_path: Path,
    registry_path: Path,
    registry_hash: str,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if state_path.is_file():
        state = _harness.read_json(state_path)
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise QueueConfigurationError(f"invalid Main-9 queue state: {state_path}")
        if state.get("protocol_id") != PROTOCOL_ID:
            raise QueueConfigurationError("queue state belongs to another protocol")
        if state.get("registry_sha256") != registry_hash:
            raise QueueConfigurationError(
                "Main-9 registry changed since queue creation; use a new state directory"
            )
    else:
        token = hashlib.sha256(str(state_path.resolve()).encode()).hexdigest()[:8]
        state = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "queue_id": (
                "main9-"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "-"
                + re.sub(r"[^a-zA-Z0-9]+", "-", socket.gethostname()).strip("-")
                + "-"
                + token
            ),
            "created_at": _harness.utc_now(),
            "updated_at": _harness.utc_now(),
            "registry": str(registry_path),
            "registry_sha256": registry_hash,
            "tasks": {},
            "status": "running",
        }
    for task in tasks:
        state.setdefault("tasks", {}).setdefault(
            str(task["task_id"]), {"status": "pending", "stages": {}, "data_dir": None}
        )
    _harness.atomic_write_json(state_path, state)
    return state


def _acquire_lock(path: Path, *, required: bool) -> tuple[Path, Any] | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    os.chmod(path, 0o600)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        if required:
            raise QueueConfigurationError(f"another Main-9 queue owns {path}")
        return None
    return path, handle


def _release_lock(lock: tuple[Path, Any] | None) -> None:
    if lock is None:
        return
    _path, handle = lock
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def validate_unique_roots(paths: QueuePaths) -> None:
    named = {
        "runtime_root": paths.runtime_root,
        "data_root": paths.data_root,
        "logs_root": paths.logs_root,
        "workspaces_root": paths.workspaces_root,
        "resource_root": paths.resource_root,
        "state_dir": paths.state_dir,
    }
    resolved = {name: path.resolve() for name, path in named.items()}
    for first in resolved:
        for second in resolved:
            if first < second and resolved[first] == resolved[second]:
                raise QueueConfigurationError(
                    f"Main-9 roots must be unique: {first} == {second} == {resolved[first]}"
                )


def print_status(state_path: Path) -> int:
    if not state_path.is_file():
        print(f"no Main-9 queue state at {state_path}")
        return 1
    state = _harness.read_json(state_path)
    print(
        f"queue {state.get('queue_id')} status={state.get('status')} "
        f"updated={state.get('updated_at')}"
    )
    for task_id in FROZEN_TASK_ORDER:
        task_state = state.get("tasks", {}).get(task_id, {})
        stages = task_state.get("stages", {}) if isinstance(task_state, dict) else {}
        canary = stages.get("canary", {}).get("status", "pending")
        full = stages.get("full", {}).get("status", "pending")
        reason = task_state.get("data_status_reason") if isinstance(task_state, dict) else None
        print(
            f"{task_id:14} task={task_state.get('status', 'pending'):22} "
            f"canary={canary:20} full={full:20}"
            + (f" data={reason}" if reason else "")
        )
    return 0


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path(os.environ.get("BIOMAKER_MAIN9_REGISTRY", DEFAULT_REGISTRY)))
    parser.add_argument("--runtime-root", type=Path, default=Path(os.environ.get("BIOMAKER_RUNTIME_ROOT", REPOSITORY_ROOT)))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--logs-root", type=Path, default=None)
    parser.add_argument("--workspaces-root", type=Path, default=None)
    parser.add_argument("--resource-root", type=Path, default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--prepared-manifest", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--extra-override", action="append", default=[])
    parser.add_argument("--poll-seconds", type=int, default=_harness.env_int("BIOMAKER_MAIN9_POLL_SECONDS", 30))
    parser.add_argument("--heartbeat-seconds", type=int, default=_harness.env_int("BIOMAKER_MAIN9_HEARTBEAT_SECONDS", 30))
    parser.add_argument("--no-wait-for-data", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-resource-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> QueuePaths:
    runtime_root = args.runtime_root.expanduser().resolve()
    storage = Path(os.environ.get("BIOMAKER_STORAGE_ROOT", runtime_root.parent)).expanduser().resolve()
    data_root = Path(args.data_root or os.environ.get("BIOMAKER_MAIN9_DATA_ROOT", storage / "datasets" / "main9" / "data"))
    formal_root = storage / "formal" / "biomaker-main9"
    logs_root = Path(args.logs_root or os.environ.get("BIOMAKER_MAIN9_LOGS_ROOT", formal_root / "logs"))
    workspaces_root = Path(args.workspaces_root or os.environ.get("BIOMAKER_MAIN9_WORKSPACES_ROOT", formal_root / "workspaces"))
    resource_root = Path(args.resource_root or os.environ.get("BIOMAKER_RESOURCE_ROOT", storage / "workspaces" / "_shared"))
    state_dir = Path(args.state_dir or os.environ.get("BIOMAKER_MAIN9_STATE_DIR", formal_root / "state"))
    python = _harness.runtime_python(runtime_root, args.python)
    return QueuePaths(
        runtime_root=runtime_root,
        data_root=data_root.expanduser().resolve(),
        logs_root=logs_root.expanduser().resolve(),
        workspaces_root=workspaces_root.expanduser().resolve(),
        resource_root=resource_root.expanduser().resolve(),
        state_dir=state_dir.expanduser().resolve(),
        # Do not resolve the virtualenv's ``python`` symlink to the base
        # interpreter: Python uses argv[0] to activate the venv site-packages.
        python=python.expanduser().absolute(),
    )


def _reset_failed_canaries_once(
    state: dict[str, Any], *, paths: QueuePaths, repeat_index: int
) -> None:
    resets: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    errors: list[str] = []
    for task_id, task_state in state.get("tasks", {}).items():
        stages = task_state.get("stages", {}) if isinstance(task_state, dict) else {}
        for stage, stage_state in stages.items():
            if not isinstance(stage_state, dict) or stage_state.get("status") not in {"failed", "interrupted"}:
                continue
            if stage != "canary":
                errors.append(f"{task_id}/{stage}: full stages are never automatically retried")
                continue
            previous = stage_state.get("previous_attempts", [])
            if previous:
                errors.append(f"{task_id}/canary: the single retry budget was already consumed")
                continue
            reasons, evidence = _harness._retry_attempt_safety(
                task_id=str(task_id),
                stage="canary",
                attempt=stage_state,
                attempt_index=1,
                logs_root=paths.logs_root,
            )
            if private_evaluation_claim_path(paths.resource_root, str(task_id), repeat_index).exists():
                reasons.append(f"{task_id}/canary: Main-9 private intent claim exists")
            if reasons:
                errors.extend(reasons)
            else:
                resets.append((task_state, stage_state, [evidence]))
    if errors:
        raise QueueConfigurationError(
            "--retry-failed rejected by the private-evaluation boundary: " + " | ".join(errors)
        )
    for task_state, stage_state, evidence in resets:
        previous = dict(stage_state)
        previous["retry_safety_evidence"] = evidence[0]
        stage_state.clear()
        stage_state.update({
            "status": "pending",
            "previous_attempts": [previous],
            "retry_authorized_at": _harness.utc_now(),
            "retry_safety_evidence": evidence,
        })
        task_state["status"] = "pending"


def queue_main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    if args.extra_override:
        raise QueueConfigurationError("--extra-override is disabled for the frozen Main-9 protocol")
    if args.skip_resource_check and not args.dry_run:
        raise QueueConfigurationError("--skip-resource-check is forbidden for Main-9 runs")
    paths = resolve_paths(args)
    state_path = paths.state_dir / "queue_state.json"
    if args.status:
        return print_status(state_path)
    validate_unique_roots(paths)

    registry_path = args.registry.expanduser().resolve()
    selected = set(args.only) if args.only else None
    registry = load_registry(
        registry_path,
        require_descriptions_for=selected,
    )
    tasks = active_tasks(registry, selected)
    if not args.dry_run:
        _harness.require_supported_tasks(tasks)
    defaults = registry["defaults"]
    model = os.environ.get("BIOMAKER_MODEL", str(defaults["model"]))
    if model != "deepseek-v4-flash":
        raise QueueConfigurationError("Main-9 requires deepseek-v4-flash for every Agent role")
    gpu_uuids, workers = resolve_compute_profile(python=paths.python)
    repeat_index = _harness.env_int("BIOMAKER_REPEAT_INDEX", 1)
    canary_steps = _harness.env_int("BIOMAKER_CANARY_STEPS", int(defaults["canary_steps"]))
    full_steps = _harness.env_int("BIOMAKER_FULL_STEPS", int(defaults["full_steps"]))
    if canary_steps != 4 or full_steps != 35:
        raise QueueConfigurationError("Main-9 requires exactly 4 canary and 35 full steps")
    node_timeout = _harness.env_int("BIOMAKER_NODE_TIMEOUT", int(defaults["per_node_timeout_seconds"]))
    canary_limit = _harness.env_int("BIOMAKER_CANARY_TOTAL_TIMEOUT", int(defaults["canary_total_time_limit_seconds"]))
    full_limit = _harness.env_int("BIOMAKER_FULL_TOTAL_TIMEOUT", int(defaults["full_total_time_limit_seconds"]))
    canary_hard = _harness.env_int("BIOMAKER_CANARY_HARD_WALL_CLOCK", int(defaults["canary_hard_wall_clock_seconds"]))
    full_hard = _harness.env_int("BIOMAKER_FULL_HARD_WALL_CLOCK", int(defaults["full_hard_wall_clock_seconds"]))
    progress_timeout = _harness.env_int("BIOMAKER_PROGRESS_TIMEOUT", int(defaults["progress_timeout_seconds"]))
    retrain_progress_timeout = _harness.env_int("BIOMAKER_FINAL_RETRAIN_PROGRESS_TIMEOUT", int(defaults["final_retrain_progress_timeout_seconds"]))
    shutdown_grace = _harness.env_int("BIOMAKER_SHUTDOWN_GRACE", int(defaults["shutdown_grace_seconds"]))
    final_retrain_timeout = (node_timeout * 3 + 1) // 2
    if canary_hard <= canary_limit:
        raise QueueConfigurationError("canary hard wall-clock must exceed search time")
    if full_hard <= full_limit + final_retrain_timeout:
        raise QueueConfigurationError("full hard wall-clock must include final-retrain headroom")
    if retrain_progress_timeout <= final_retrain_timeout:
        raise QueueConfigurationError("final-retrain progress watchdog must exceed runtime timeout")

    prepared_path = args.prepared_manifest
    if prepared_path is None and os.environ.get("BIOMAKER_PREPARED_MANIFEST"):
        prepared_path = Path(os.environ["BIOMAKER_PREPARED_MANIFEST"])
    if prepared_path is None:
        prepared_path = paths.data_root.parent / "suite_manifest.json"
    prepared_path = prepared_path.expanduser().resolve()
    prepared = load_prepared_manifest(prepared_path)
    tracker = ReadinessTracker()

    if args.dry_run:
        preview_state = {"tasks": {task["task_id"]: {"status": "pending", "stages": {}} for task in tasks}}
        readiness: list[dict[str, Any]] = []
        for task in tasks:
            data_dir, record = resolve_dataset_dir(task, prepared)
            check = tracker.check(
                task, data_dir, record, expected_cache_root=paths.resource_root / "split_builder"
            )
            readiness.append({
                "order": task["order"],
                "task_id": task["task_id"],
                "task_family": task["task_family"],
                "primary_metric": task["primary_metric"],
                "data_dir": str(data_dir) if data_dir is not None else None,
                "ready": check.ready,
                "reason": check.reason,
                "stages": ["canary", "full"],
            })
        del preview_state
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "model": model,
            "canary_steps": canary_steps,
            "full_steps": full_steps,
            "max_children_per_node": 4,
            "max_debug_depth": 2,
            "topk_improve": 3,
            "private_evaluation_per_node": False,
            "parallel_workers": workers,
            "gpu_uuids": gpu_uuids,
            "repeat_index": repeat_index,
            "runtime_root": str(paths.runtime_root),
            "logs_root": str(paths.logs_root),
            "workspaces_root": str(paths.workspaces_root),
            "resource_root": str(paths.resource_root),
            "state_dir": str(paths.state_dir),
            "prepared_manifest": str(prepared_path),
            "tasks": readiness,
        }, ensure_ascii=False, indent=2))
        return 0

    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_root.mkdir(parents=True, exist_ok=True)
    paths.workspaces_root.mkdir(parents=True, exist_ok=True)
    state_lock = _acquire_lock(paths.state_dir / "queue.lock", required=True)
    try:
        _harness.validate_runtime(paths)
        _harness.validate_gpu_inventory(gpu_uuids, workers)
        _harness.install_signal_handlers()
        existed = state_path.is_file()
        state = initial_state(state_path, registry_path, _harness.sha256_file(registry_path), tasks)
        if existed:
            _harness.fail_closed_on_cold_start(state_path, state, state_dir=paths.state_dir)
        if args.retry_failed:
            _reset_failed_canaries_once(state, paths=paths, repeat_index=repeat_index)
            _harness.save_state(state_path, state)
        _harness.emit(
            paths.state_dir,
            "queue_started",
            protocol_id=PROTOCOL_ID,
            queue_id=state["queue_id"],
            tasks=len(tasks),
            model=model,
            workers=workers,
            gpu_uuids=",".join(gpu_uuids),
            repeat_index=repeat_index,
        )
        while not _harness.STOP_REQUESTED.is_set():
            prepared = load_prepared_manifest(prepared_path)
            if _harness.all_tasks_resolved(state, tasks):
                failed = [
                    task["task_id"] for task in tasks
                    if state["tasks"][task["task_id"]].get("status") == "failed"
                ]
                state["status"] = "completed_with_failures" if failed else "completed"
                state["active_task"] = None
                state["active_stage"] = None
                state["failed_tasks"] = failed
                _harness.save_state(state_path, state)
                _harness.write_heartbeat(paths.state_dir / "heartbeat.json", state)
                _harness.emit(paths.state_dir, "queue_finished", status=state["status"], failed=failed)
                return 1 if failed else 0

            chosen = choose_next_ready(
                tasks,
                state,
                prepared,
                tracker,
                expected_cache_root=paths.resource_root / "split_builder",
            )
            _harness.save_state(state_path, state)
            if chosen is None:
                state["status"] = "waiting_data"
                state["active_task"] = None
                state["active_stage"] = None
                _harness.save_state(state_path, state)
                _harness.write_heartbeat(paths.state_dir / "heartbeat.json", state)
                if args.no_wait_for_data:
                    _harness.emit(paths.state_dir, "queue_idle_exit", reason="no Main-9 task is ready")
                    return 3
                _harness.STOP_REQUESTED.wait(args.poll_seconds)
                continue

            task, stage, check, record = chosen
            global_lock = _acquire_lock(
                paths.resource_root / ".fig2_medmnist_queue" / "queue.global.lock",
                required=False,
            )
            if global_lock is None:
                state["status"] = "waiting_resources"
                state["active_task"] = task["task_id"]
                state["active_stage"] = stage
                _harness.save_state(state_path, state)
                _harness.write_heartbeat(
                    paths.state_dir / "heartbeat.json",
                    state,
                    active_task=task["task_id"],
                    active_stage=stage,
                    resource_wait_reason="another BioMaker formal queue owns the global admission lock",
                )
                _harness.STOP_REQUESTED.wait(args.poll_seconds)
                continue
            try:
                steps = canary_steps if stage == "canary" else full_steps
                total_limit = canary_limit if stage == "canary" else full_limit
                hard_limit = canary_hard if stage == "canary" else full_hard
                generate_report = bool(defaults[
                    "generate_report_for_canary" if stage == "canary" else "generate_report_for_full"
                ])
                run_one_stage(
                    paths=paths,
                    registry_path=registry_path,
                    state_path=state_path,
                    state=state,
                    task=task,
                    check=check,
                    prepared_record=record,
                    stage=stage,
                    steps=steps,
                    total_time_limit=total_limit,
                    hard_wall_clock_seconds=hard_limit,
                    node_timeout=node_timeout,
                    workers=workers,
                    gpu_uuids=gpu_uuids,
                    model=model,
                    generate_report=generate_report,
                    heartbeat_seconds=args.heartbeat_seconds,
                    progress_timeout_seconds=progress_timeout,
                    final_retrain_progress_timeout_seconds=retrain_progress_timeout,
                    shutdown_grace_seconds=shutdown_grace,
                    extra_overrides=(),
                    skip_resource_check=False,
                )
            finally:
                _release_lock(global_lock)
        state["status"] = "interrupted"
        state["active_task"] = None
        state["active_stage"] = None
        _harness.save_state(state_path, state)
        _harness.write_heartbeat(paths.state_dir / "heartbeat.json", state)
        _harness.emit(paths.state_dir, "queue_interrupted")
        return 130
    finally:
        child = _harness.CURRENT_CHILD
        if child is not None:
            _harness._terminate_child(child)
        _release_lock(state_lock)


def main() -> None:
    try:
        raise SystemExit(queue_main())
    except QueueConfigurationError as exc:
        print(f"Main-9 queue configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
