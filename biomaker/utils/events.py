"""Lightweight structured event logging for BioMaker runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

MAX_EVENT_STRING_CHARS = 1200
_EVENT_WRITE_LOCK = RLock()
TERMINAL_EVENTS = frozenset({"run_finished", "run_failed", "run_interrupted"})


def _json_safe(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) > MAX_EVENT_STRING_CHARS:
            return (
                value[: MAX_EVENT_STRING_CHARS // 2]
                + f"\n... [{len(value) - MAX_EVENT_STRING_CHARS} chars trimmed] ...\n"
                + value[-MAX_EVENT_STRING_CHARS // 2 :]
            )
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _write_event(
    log_dir: Path,
    event: str,
    payload: dict[str, Any],
    *,
    durable: bool,
) -> None:
    with _EVENT_WRITE_LOCK:
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **_json_safe(payload),
        }
        with (log_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            if durable:
                handle.flush()
                os.fsync(handle.fileno())


def append_event(log_dir: Path, event: str, **payload: Any) -> None:
    """Append one non-terminal lifecycle fact on a best-effort basis."""
    try:
        _write_event(log_dir, event, payload, durable=False)
    except Exception:
        return


def append_terminal_event(log_dir: Path, event: str, **payload: Any) -> None:
    """Durably append one terminal lifecycle fact or propagate the write failure."""
    if event not in TERMINAL_EVENTS:
        raise ValueError(f"Unsupported terminal event: {event}")
    _write_event(log_dir, event, payload, durable=True)
