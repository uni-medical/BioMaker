from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
    return opaque_revision({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})


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


def read_events(
    settings: Settings, run: dict[str, Any], *, tail: int = 200
) -> dict[str, Any]:
    try:
        path = resolve_artifact(settings, run, "events.jsonl")
    except ArtifactError:
        return {"count": 0, "events": []}
    parsed: list[dict[str, Any]] = []
    count = 0
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    parsed.append(event)
                    if len(parsed) > tail:
                        parsed.pop(0)
    except OSError:
        return {"count": 0, "events": []}
    return {"count": count, "events": parsed}


def read_results_summary(settings: Settings, run: dict[str, Any]) -> dict[str, Any]:
    try:
        path = resolve_artifact(settings, run, "results.tsv")
    except ArtifactError:
        return {"count": 0, "best_metric": None, "latest": None}
    rows: list[dict[str, str]] = []
    try:
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, csv.Error):
        return {"count": 0, "best_metric": None, "latest": None}

    def number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    def json_field(row, name, default):
        try:
            parsed = json.loads(row.get(name) or "null")
            return parsed if isinstance(parsed, type(default)) else default
        except (TypeError, ValueError):
            return default

    best_row = None
    if rows and "is_best" in rows[0]:
        # The Journal owns selection, including ties and invalid candidates.
        best_row = next(
            (row for row in rows if str(row.get("is_best", "")).lower() == "true"), None
        )
    else:
        # Historical TSVs predate explicit winner export and remain readable.
        metric_rows = [(number(row.get("metric")), row) for row in rows]
        metric_rows = [(value, row) for value, row in metric_rows if value is not None]
        if metric_rows:
            maximize = (
                str(metric_rows[-1][1].get("maximize", "true")).lower() != "false"
            )
            best_row = (max if maximize else min)(
                metric_rows, key=lambda item: item[0]
            )[1]
    best_metric = number(best_row.get("metric")) if best_row else None
    best = None
    if best_row:
        best = {
            "node_id": best_row.get("node_id"),
            "step": best_row.get("step"),
            "metrics": json_field(best_row, "metrics", {}),
            "metric_specs": json_field(best_row, "metric_specs", []),
            "metric_ranks": json_field(best_row, "metric_ranks", {}),
            "average_rank": number(best_row.get("average_rank")),
            "worst_rank": number(best_row.get("worst_rank")),
            "candidate_count": number(best_row.get("candidate_count")),
            "ranking_version": best_row.get("ranking_version"),
            "selection_policy": best_row.get("selection_policy", "legacy_primary"),
        }
    latest = None
    if rows:
        latest = {
            key: rows[-1].get(key)
            for key in ("step", "status", "metric", "outcome", "exc_type", "error")
        }
        latest["metrics"] = json_field(rows[-1], "metrics", {})
        latest["average_rank"] = number(rows[-1].get("average_rank"))
    return {
        "count": len(rows),
        "best_metric": best_metric,
        "best": best,
        "latest": latest,
    }


def read_manifest_summary(
    settings: Settings, run: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        path = resolve_artifact(settings, run, "run_manifest.json")
    except ArtifactError:
        return None
    try:
        if path.stat().st_size > 5_000_000:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return {
        key: data.get(key)
        for key in ("created_at", "exp_name", "data_dir", "workspace_dir", "config")
    }


def detailed_run(settings: Settings, run: dict[str, Any]) -> dict[str, Any]:
    result = dict(run)
    events = read_events(settings, run, tail=1)
    result["observed"] = {
        "event_count": events["count"],
        "last_event": events["events"][-1] if events["events"] else None,
        "results": read_results_summary(settings, run),
        "manifest": read_manifest_summary(settings, run),
    }
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
