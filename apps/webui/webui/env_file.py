from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

MANAGED_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "SERPER_API_KEY",
    "JINA_API_KEY",
)


def _assignment_pattern(key: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=")


def _encode_value(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def read_managed_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = dotenv_values(path)
    return {
        key: str(values[key])
        for key in MANAGED_ENV_KEYS
        if values.get(key) not in (None, "")
    }


def mask_value(value: str) -> str:
    if not value:
        return ""
    suffix = value[-4:] if len(value) > 4 else ""
    return f"********{suffix}"


def masked_env(path: Path) -> dict[str, dict[str, str | bool]]:
    values = read_managed_values(path)
    return {
        key: {
            "configured": key in values,
            "masked": mask_value(values.get(key, "")),
        }
        for key in MANAGED_ENV_KEYS
    }


def update_env_file(path: Path, updates: Mapping[str, str | None]) -> None:
    unknown = set(updates) - set(MANAGED_ENV_KEYS)
    if unknown:
        raise ValueError(f"Unsupported environment keys: {sorted(unknown)}")
    if not updates:
        return

    original = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = original.splitlines()
    output: list[str] = []
    handled: set[str] = set()

    for line in lines:
        matched_key = next(
            (key for key in updates if _assignment_pattern(key).match(line)), None
        )
        if matched_key is None:
            output.append(line)
            continue
        if matched_key in handled:
            continue
        handled.add(matched_key)
        value = updates[matched_key]
        if value is not None and value.strip():
            output.append(f"{matched_key}={_encode_value(value.strip())}")

    additions = [
        f"{key}={_encode_value(str(value).strip())}"
        for key, value in updates.items()
        if key not in handled and value is not None and str(value).strip()
    ]
    if additions and output and output[-1].strip():
        output.append("")
    output.extend(additions)
    rendered = "\n".join(output)
    if rendered:
        rendered += "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temporary_path, path.stat().st_mode)
        else:
            os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def subprocess_environment(path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for key in MANAGED_ENV_KEYS:
        environment.pop(key, None)
    environment.update(read_managed_values(path))
    return environment
