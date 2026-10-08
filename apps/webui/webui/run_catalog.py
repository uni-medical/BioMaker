"""Read BioMaker experiment history directly from the configured log root."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import read_events
from .settings import Settings

MAX_MANIFEST_BYTES = 5_000_000
TERMINAL_EVENT_STATUSES = {
    "run_finished": "complete",
    "run_failed": "incomplete",
    "run_interrupted": "incomplete",
}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_manifest(log_dir: Path, *, strict: bool = False) -> dict[str, Any] | None:
    path = log_dir / "run_manifest.json"
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_MANIFEST_BYTES
        ):
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        if strict:
            raise
        return None
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _task_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n\n".join(
            f"{key}: {item}" for key, item in value.items() if item is not None
        )
    return "" if value is None else str(value)


def _event_value(events: list[dict[str, Any]], name: str, *, reverse: bool = False):
    values = reversed(events) if reverse else events
    return next((event for event in values if event.get("event") == name), None)


def _console_log(log_dir: Path, *, strict: bool = False) -> Path | None:
    exact = log_dir / f"{log_dir.name}.log"
    if exact.is_file() and not exact.is_symlink():
        return exact
    try:
        return next(
            path
            for path in sorted(log_dir.glob("*.log"))
            if path.is_file() and not path.is_symlink() and path.parent == log_dir
        )
    except OSError:
        if strict:
            raise
        return None
    except StopIteration:
        return None


def run_from_log_dir(settings: Settings, log_dir: Path, *, strict: bool = False) -> dict[str, Any] | None:
    root = settings.logs_root.resolve()
    try:
        if log_dir.is_symlink() or not log_dir.is_dir():
            return None
        resolved = log_dir.resolve()
        if not _inside(resolved, root) or resolved.parent != root:
            return None
        manifest = _read_manifest(resolved, strict=strict)
        if manifest is None:
            return None
    except OSError:
        if strict:
            raise
        return None

    data_dir_value = manifest.get("data_dir")
    if not data_dir_value:
        return None
    data_dir = str(Path(str(data_dir_value)).expanduser())
    dataset_name = Path(data_dir).name
    if not dataset_name:
        return None

    run_stub = {"id": resolved.name, "log_dir": str(resolved)}
    event_response = read_events(settings, run_stub, tail=5_000, strict=strict)
    events = event_response["events"]
    started = _event_value(events, "run_started")
    terminal_event = next(
        (
            event
            for event in reversed(events)
            if isinstance(event.get("event"), str)
            and event["event"] in TERMINAL_EVENT_STATUSES
        ),
        None,
    )
    last_event = events[-1] if events else None
    status = (
        TERMINAL_EVENT_STATUSES[str(terminal_event.get("event"))]
        if terminal_event is not None
        else "incomplete"
    )
    terminal_event_name = (
        str(terminal_event.get("event")) if terminal_event is not None else None
    )
    detailed_status = {
        "run_finished": "succeeded",
        "run_failed": "failed",
        "run_interrupted": "interrupted",
    }.get(terminal_event_name, "orphaned")
    if terminal_event is None:
        status_reason = (
            "The process is not known to be alive and no terminal lifecycle "
            "event was recorded."
        )
    else:
        status_reason = (terminal_event or {}).get("error") or (
            (terminal_event or {}).get("reason") if status == "incomplete" else None
        )

    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    total_steps = config.get("agent_steps")
    parallel_workers = config.get("parallel_workers")
    if parallel_workers is None:
        parallel_workers = (started or {}).get("parallel_workers")
    final_retrain_enabled = config.get("final_retrain_enabled")
    if not isinstance(final_retrain_enabled, bool):
        final_retrain_enabled = any(
            str(event.get("event") or "").startswith("final_retrain_")
            for event in events
        )
    completed_steps = (
        terminal_event.get("completed_steps") if terminal_event is not None else None
    )
    if completed_steps is None:
        round_finished = _event_value(events, "round_finished", reverse=True)
        if round_finished is not None:
            completed_steps = round_finished.get("completed_steps")

    try:
        modified_at = resolved.stat().st_mtime
    except OSError:
        if strict:
            raise
        modified_at = 0.0
    created_at = (started or {}).get("ts") or manifest.get("created_at")
    finished_at = (terminal_event or {}).get("ts")
    console = _console_log(resolved, strict=strict)
    workspace_dir = manifest.get("workspace_dir")

    return {
        "id": resolved.name,
        "source": "logs",
        "read_only": True,
        "project_snapshot": {
            "dataset_path": data_dir,
            "task_template_path": None,
            "task_body": _task_text(manifest.get("task_description")),
        },
        "dataset_path": data_dir,
        "dataset_name": dataset_name,
        "status": status,
        "status_source": "events",
        "detailed_status": detailed_status,
        "status_reason": status_reason,
        "terminal_event": terminal_event_name,
        "requested_exp_name": None,
        "actual_exp_name": resolved.name,
        "options": {
            "model_for_all_agents": config.get("model_for_all_agents"),
            "main_model": config.get("main_model"),
            "agent_steps": total_steps,
            "exec_timeout": config.get("per_node_timeout_seconds"),
            "exec_parallel_workers": parallel_workers,
            "agent_total_time_limit": config.get("total_time_limit_seconds"),
            "final_retrain_enabled": final_retrain_enabled,
        },
        "console_log": str(console) if console else None,
        "log_dir": str(resolved),
        "workspace_dir": str(workspace_dir) if workspace_dir else None,
        "return_code": None,
        "pid": None,
        "launch_token": manifest.get("launch_token"),
        "error": status_reason if status == "incomplete" else None,
        "timestamps": {
            "created_at": created_at,
            "queued_at": None,
            "started_at": created_at,
            "finished_at": finished_at,
            "updated_at": (last_event or {}).get("ts") or finished_at or created_at,
        },
        "progress": {
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "last_event": (last_event or {}).get("event"),
        },
        "log_modified_at": modified_at,
        "controllable": False,
        "allowed_actions": [],
    }


def list_log_runs(settings: Settings) -> list[dict[str, Any]]:
    root = settings.logs_root.resolve()
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    try:
        paths = list(root.iterdir())
    except OSError:
        return []
    for path in paths:
        run = run_from_log_dir(settings, path)
        if run is not None:
            runs.append(run)
    return sorted(
        runs,
        key=lambda run: (
            str(run.get("timestamps", {}).get("created_at") or ""),
            float(run.get("log_modified_at") or 0.0),
            str(run.get("actual_exp_name") or ""),
        ),
        reverse=True,
    )
