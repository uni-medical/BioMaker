from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Any


class RegistryError(RuntimeError):
    pass


class JsonRegistry:
    """Small thread-safe JSON registry with same-directory atomic replacement."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = RLock()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"version": 1, "projects": {}, "runs": {}}

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"Cannot read state registry: {exc}") from exc
        if not isinstance(data, dict):
            raise RegistryError("State registry root must be a JSON object")
        if data.get("version") != 1:
            raise RegistryError("Unsupported state registry version")
        for key in ("projects", "runs"):
            if not isinstance(data.get(key), dict):
                raise RegistryError(f"State registry field {key!r} must be an object")
        return data

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def read(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._read_unlocked())

    def mutate(
        self, callback: Callable[[dict[str, Any]], Any]
    ) -> tuple[dict[str, Any], Any]:
        with self._lock:
            data = self._read_unlocked()
            result = callback(data)
            self._write_unlocked(data)
            return copy.deepcopy(data), copy.deepcopy(result)

    def list_records(self, collection: str) -> list[dict[str, Any]]:
        data = self.read()
        self._validate_collection(collection)
        return list(data[collection].values())

    def get_record(self, collection: str, record_id: str) -> dict[str, Any] | None:
        data = self.read()
        self._validate_collection(collection)
        record = data[collection].get(record_id)
        return copy.deepcopy(record) if record is not None else None

    def add_record(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        self._validate_collection(collection)
        record_id = str(record["id"])

        def add(data: dict[str, Any]) -> dict[str, Any]:
            if record_id in data[collection]:
                raise RegistryError(f"Duplicate {collection} id: {record_id}")
            data[collection][record_id] = copy.deepcopy(record)
            return data[collection][record_id]

        _, result = self.mutate(add)
        return result

    def replace_record(
        self, collection: str, record_id: str, record: dict[str, Any]
    ) -> dict[str, Any]:
        self._validate_collection(collection)

        def replace(data: dict[str, Any]) -> dict[str, Any]:
            if record_id not in data[collection]:
                raise KeyError(record_id)
            data[collection][record_id] = copy.deepcopy(record)
            return data[collection][record_id]

        _, result = self.mutate(replace)
        return result

    def update_record(
        self,
        collection: str,
        record_id: str,
        updates: dict[str, Any] | Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        self._validate_collection(collection)

        def update(data: dict[str, Any]) -> dict[str, Any]:
            if record_id not in data[collection]:
                raise KeyError(record_id)
            record = data[collection][record_id]
            if callable(updates):
                updates(record)
            else:
                record.update(copy.deepcopy(updates))
            return record

        _, result = self.mutate(update)
        return result

    def delete_record(self, collection: str, record_id: str) -> dict[str, Any]:
        self._validate_collection(collection)

        def delete(data: dict[str, Any]) -> dict[str, Any]:
            if record_id not in data[collection]:
                raise KeyError(record_id)
            return data[collection].pop(record_id)

        _, result = self.mutate(delete)
        return result

    @staticmethod
    def _validate_collection(collection: str) -> None:
        if collection not in {"projects", "runs"}:
            raise ValueError(f"Unknown registry collection: {collection}")
