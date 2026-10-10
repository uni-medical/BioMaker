"""Shared read-side projection of ledger state and current log observations.

The durable ledger owns lifecycle and controls. File observations supplement
that state without changing its reason, status, or permitted actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .observation_cache import ObservationCache
from .runner import TERMINAL_STATUSES


CANONICAL_STATUS = {
    "queued": "queued",
    "running": "running",
    "stopping": "stopping",
    "succeeded": "complete",
    "complete": "complete",
    "failed": "incomplete",
    "cancelled": "incomplete",
    "stopped": "incomplete",
    "interrupted": "incomplete",
    "incomplete": "incomplete",
}
CANONICAL_TERMINAL_STATUSES = {"complete", "incomplete"}


def session_run(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result.setdefault("source", "run_ledger")
    result.setdefault("read_only", False)
    result["status_source"] = "ledger"
    snapshot = result.get("project_snapshot") or {}
    dataset_path = str(snapshot.get("dataset_path") or "")
    result["dataset_path"] = dataset_path or None
    result["dataset_name"] = Path(dataset_path).name if dataset_path else None
    detailed_status = str(
        result.get("detailed_status") or result.get("status") or "incomplete"
    ).lower()
    status_value = CANONICAL_STATUS.get(detailed_status, "incomplete")
    actions = []
    if result["read_only"]:
        actions = []
    elif detailed_status == "queued":
        actions.append("cancel")
    elif detailed_status == "running":
        actions.append("stop")
    elif detailed_status in TERMINAL_STATUSES:
        actions.append("requeue")
    result["detailed_status"] = detailed_status
    result["status"] = status_value
    if status_value == "incomplete" and not result.get("status_reason"):
        result["status_reason"] = result.get("error") or {
            "cancelled": "Cancelled before the run started.",
            "stopped": "Stopped by the user.",
            "interrupted": "The run was interrupted before completion.",
            "failed": "The run failed before completion.",
        }.get(detailed_status, "The run did not record successful completion.")
    result["allowed_actions"] = actions
    result["controllable"] = bool(actions)
    return result


class RunQueries:
    """One query path for API, CLI, and desktop run views."""

    def __init__(self, settings, store, *, observations: ObservationCache | None = None):
        self.settings = settings
        self.store = store
        self.observations = observations if observations is not None else ObservationCache(settings)

    def enrich_run(self, raw_session: dict[str, Any]) -> dict[str, Any]:
        current = session_run(raw_session)
        # Observations saved at import are not evidence about the current tail.
        # Drop them even when the log directory can no longer be catalogued.
        current.pop("event_tail", None)
        current["terminal_event"] = None
        raw_log_dir = current.get("log_dir")
        base = self.observations.catalog(Path(str(raw_log_dir))) if raw_log_dir else None
        if base is None:
            return current
        # The ledger owns lifecycle state. Reuse only log observations; apply
        # them to a fresh ledger record on every list and individual request.
        for key in (
            "progress",
            "project_snapshot",
            "dataset_path",
            "dataset_name",
            "actual_exp_name",
            "options",
            "log_modified_at",
        ):
            if base.get(key) is not None:
                current[key] = base[key]
        # Tail evidence is separate from the ledger's lifecycle explanation.
        # None is meaningful: an earlier terminal event may have left the window.
        current["terminal_event"] = base.get("terminal_event")
        current["event_tail"] = base["event_tail"]
        return current

    def list_runs(self) -> list[dict[str, Any]]:
        merged = [self.enrich_run(record) for record in self.store.list_records("runs")]
        return sorted(
            merged,
            key=lambda item: (
                str(item.get("timestamps", {}).get("created_at") or ""),
                float(item.get("log_modified_at") or 0.0),
                str(
                    item.get("actual_exp_name") or item.get("requested_exp_name") or ""
                ),
            ),
            reverse=True,
        )

    def status(self, run_id: str) -> dict[str, Any] | None:
        record = self.store.get_record("runs", run_id)
        return self.enrich_run(record) if record is not None else None

    def detail(self, run_id: str) -> dict[str, Any]:
        run = self.status(run_id)
        if run is None:
            raise KeyError(run_id)
        return self.observations.detail(run)

    def results(self, run_id: str) -> dict[str, Any]:
        return self.detail(run_id)["observed"]["results"]
