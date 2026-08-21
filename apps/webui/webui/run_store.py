from __future__ import annotations

import copy
from collections.abc import Callable
from threading import RLock
from typing import Any


class MemoryRunStore:
    """Session-only queue and process state for WebUI-launched runs."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._runs: dict[str, dict[str, Any]] = {}

    def list_records(self, collection: str) -> list[dict[str, Any]]:
        self._validate_collection(collection)
        with self._lock:
            return copy.deepcopy(list(self._runs.values()))

    def get_record(self, collection: str, record_id: str) -> dict[str, Any] | None:
        self._validate_collection(collection)
        with self._lock:
            record = self._runs.get(record_id)
            return copy.deepcopy(record) if record is not None else None

    def add_record(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        self._validate_collection(collection)
        record_id = str(record["id"])
        with self._lock:
            if record_id in self._runs:
                raise ValueError(f"Duplicate run id: {record_id}")
            self._runs[record_id] = copy.deepcopy(record)
            return copy.deepcopy(self._runs[record_id])

    def update_record(
        self,
        collection: str,
        record_id: str,
        updates: dict[str, Any] | Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        self._validate_collection(collection)
        with self._lock:
            if record_id not in self._runs:
                raise KeyError(record_id)
            record = self._runs[record_id]
            if callable(updates):
                updates(record)
            else:
                record.update(copy.deepcopy(updates))
            return copy.deepcopy(record)

    def mutate(
        self, callback: Callable[[dict[str, Any]], Any]
    ) -> tuple[dict[str, Any], Any]:
        self._validate_collection("runs")
        with self._lock:
            working = {"runs": copy.deepcopy(self._runs)}
            result = callback(working)
            self._runs = working["runs"]
            return copy.deepcopy(working), copy.deepcopy(result)

    @staticmethod
    def _validate_collection(collection: str) -> None:
        if collection != "runs":
            raise ValueError(f"Unknown in-memory collection: {collection}")
