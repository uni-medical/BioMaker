from __future__ import annotations

import csv
import hashlib
import json
import math
import stat as stat_module
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .event_reader import read_event_window
from .settings import Settings


class ArtifactError(ValueError):
    pass


def opaque_revision(payload: Any) -> str:
    """Return a stable token whose representation is not an API contract."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def file_revision(path: Path) -> str | None:
    """Return a cheap revision for a regular file, or ``None`` if unavailable."""

    try:
        if path.is_symlink():
            return None
        stat = path.stat()
        if not path.is_file():
            return None
    except OSError:
        return None
    return opaque_revision({"dev": stat.st_dev, "ino": stat.st_ino, "mode": stat.st_mode,
                            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                            "ctime_ns": stat.st_ctime_ns})


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_run_log_dir(settings: Settings, run: dict[str, Any]) -> Path:
    raw_path = run.get("log_dir")
    if not raw_path:
        raise ArtifactError("Run log directory is not available yet")
    root = settings.logs_root.resolve()
    path = Path(str(raw_path)).expanduser().resolve()
    if not _inside(path, root) or path == root:
        raise ArtifactError("Run log directory is outside the configured log root")
    if not path.is_dir():
        raise ArtifactError("Run log directory does not exist")
    return path


def resolve_artifact(
    settings: Settings, run: dict[str, Any], artifact_name: str
) -> Path:
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise ArtifactError("Invalid artifact name")
    log_dir = resolve_run_log_dir(settings, run)
    candidate = (log_dir / artifact_name).resolve()
    if not _inside(candidate, log_dir) or candidate.parent != log_dir:
        raise ArtifactError("Artifact path escapes the run log directory")
    if not candidate.is_file():
        raise ArtifactError("Artifact does not exist")
    return candidate


def list_artifacts(settings: Settings, run: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        log_dir = resolve_run_log_dir(settings, run)
    except ArtifactError:
        return []
    artifacts = []
    for path in sorted(log_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.name.startswith(".") or not path.is_file() or path.is_symlink():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        revision = file_revision(path)
        if revision is None:
            continue
        artifacts.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
                "revision": revision,
                "url": (
                    f"/artifacts/runs/{quote(str(run['id']), safe='')}/"
                    f"{quote(path.name, safe='')}"
                ),
            }
        )
    return artifacts


def artifact_collection_revision(artifacts: list[dict[str, Any]]) -> str | None:
    """Summarize the visible artifact set without reading artifact contents."""

    if not artifacts:
        return None
    return opaque_revision(
        [
            {"name": artifact["name"], "revision": artifact["revision"]}
            for artifact in artifacts
        ]
    )


def artifact_revision(
    settings: Settings, run: dict[str, Any], artifact_name: str
) -> str | None:
    try:
        path = resolve_artifact(settings, run, artifact_name)
    except ArtifactError:
        return None
    return file_revision(path)


def _observation_artifact(settings: Settings, run: dict[str, Any], name: str,
                          strict: bool) -> Path | None:
    if not strict:
        try:
            return resolve_artifact(settings, run, name)
        except ArtifactError:
            return None
    directory = resolve_run_log_dir(settings, run)
    candidate = (directory / name).resolve()
    if not _inside(candidate, directory) or candidate.parent != directory:
        raise ArtifactError("Artifact path escapes the run log directory")
    try:
        info = candidate.stat()
    except FileNotFoundError:
        # A confirmed missing file is different from an I/O failure. The cache
        # records missing dependency metadata and invalidates on its creation.
        return None
    if not stat_module.S_ISREG(info.st_mode):
        raise ArtifactError("Artifact is not a regular file")
    return candidate


def _read_event_window(
    settings: Settings, run: dict[str, Any], *, tail: int, count_all: bool, strict: bool
) -> tuple[int | None, list[dict[str, Any]]]:
    path = _observation_artifact(settings, run, "events.jsonl", strict)
    if path is None:
        return (0 if count_all else None), []
    try:
        return read_event_window(path, tail=tail, count_all=count_all)
    except OSError:
        if strict:
            raise
        return (0 if count_all else None), []


def read_events(
    settings: Settings, run: dict[str, Any], *, tail: int = 200, strict: bool = False
) -> dict[str, Any]:
    count, events = _read_event_window(settings, run, tail=tail, count_all=True, strict=strict)
    return {"count": count, "events": events}


def read_event_tail(
    settings: Settings, run: dict[str, Any], *, tail: int = 200, strict: bool = False
) -> list[dict[str, Any]]:
    """Read only the needed valid-object suffix when a total count is unused."""
    return _read_event_window(settings, run, tail=tail, count_all=False, strict=strict)[1]


def read_results_summary(settings: Settings, run: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    path = _observation_artifact(settings, run, "results.tsv", strict)
    if path is None:
        return {"count": 0, "best_metric": None, "latest": None}
    count = 0
    minimum = maximum = None
    maximize = True
    latest_row = None
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                count += 1
                latest_row = row
                try:
                    metric = float(row.get("metric", ""))
                except (TypeError, ValueError):
                    continue
                # Preserve the historical rule: the last finite score's row
                # chooses the direction, even if later rows have no score.
                if math.isfinite(metric):
                    minimum = metric if minimum is None else min(minimum, metric)
                    maximum = metric if maximum is None else max(maximum, metric)
                    maximize = str(row.get("maximize", "true")).lower() != "false"
    except (OSError, csv.Error):
        if strict:
            raise
        return {"count": 0, "best_metric": None, "latest": None}
    latest = (
        {key: latest_row.get(key) for key in ("step", "status", "metric", "outcome", "exc_type", "error")}
        if latest_row is not None else None
    )
    return {"count": count, "best_metric": maximum if maximize else minimum, "latest": latest}


def read_manifest_summary(
    settings: Settings, run: dict[str, Any], *, strict: bool = False
) -> dict[str, Any] | None:
    path = _observation_artifact(settings, run, "run_manifest.json", strict)
    if path is None:
        return None
    try:
        if path.stat().st_size > 5_000_000:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        if strict:
            raise
        return None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {
        key: data.get(key)
        for key in ("created_at", "exp_name", "data_dir", "workspace_dir", "config")
    }


def read_observed_summary(settings: Settings, run: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    events = read_events(settings, run, tail=1, strict=strict)
    return {
        "event_count": events["count"],
        "last_event": events["events"][-1] if events["events"] else None,
        "results": read_results_summary(settings, run, strict=strict),
        "manifest": read_manifest_summary(settings, run, strict=strict),
    }


def detailed_run(settings: Settings, run: dict[str, Any]) -> dict[str, Any]:
    result = dict(run)
    result["observed"] = read_observed_summary(settings, run)
    result["artifacts"] = list_artifacts(settings, run)
    return result


def tail_text(path: Path, lines: int, max_bytes: int = 2_000_000) -> str:
    if lines < 1 or not path.is_file():
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            chunks: list[bytes] = []
            newline_count = 0
            bytes_read = 0
            while position > 0 and newline_count <= lines and bytes_read < max_bytes:
                chunk_size = min(8192, position, max_bytes - bytes_read)
                if chunk_size <= 0:
                    break
                position -= chunk_size
                handle.seek(position)
                chunk = handle.read(chunk_size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
                bytes_read += len(chunk)
        text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
    except OSError:
        return ""
