"""Lightweight structured event logging for BioMaker runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


MAX_EVENT_STRING_CHARS = 1200
_EVENT_WRITE_LOCK = RLock()


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


def append_event(log_dir: Path, event: str, **payload: Any) -> None:
    """Append one compact JSON event line. This must never fail the run."""
    try:
        with _EVENT_WRITE_LOCK:
            log_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **_json_safe(payload),
            }
            with (log_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return
