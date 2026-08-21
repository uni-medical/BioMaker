from __future__ import annotations

import copy
import json
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

DEFAULT_QUEUE_CONTROL: dict[str, Any] = {
    "paused": False,
    "reason": None,
    "pause_generation": 0,
    "runner_pid": None,
    "runner_pgid": None,
    "runner_process_started_at": None,
    "runner_token": None,
    "runner_claimed_at": None,
}

QUEUE_SEQUENCE_KEY = "queue_sequence"


class SQLiteRunStore:
    """Transactional run and queue state backed by SQLite.

    A store instance owns one connection. Calls are safe across threads, and
    separate instances can use the same database file. Run records remain
    JSON-shaped dictionaries so callers do not need to know about the storage
    representation.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        if busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be positive")

        raw_database_path = str(database_path)
        self.database_path = (
            raw_database_path
            if raw_database_path == ":memory:"
            else str(Path(raw_database_path).expanduser().resolve())
        )
        self._lock = RLock()
        self._closed = False
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            self.database_path,
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        if self.database_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._ensure_open()
            self._connection.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    record_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS queue_control (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    record_json TEXT NOT NULL
                );
                """)
            self._connection.execute(
                """
                INSERT OR IGNORE INTO queue_control(singleton, record_json)
                VALUES (1, ?)
                """,
                (self._encode_json(DEFAULT_QUEUE_CONTROL),),
            )

    def list_records(self, collection: str) -> list[dict[str, Any]]:
        self._validate_collection(collection)
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT sequence, record_json FROM runs ORDER BY sequence"
            ).fetchall()
        return [self._project_row(row) for row in rows]

    def get_record(self, collection: str, record_id: str) -> dict[str, Any] | None:
        self._validate_collection(collection)
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT sequence, record_json FROM runs WHERE run_id = ?",
                (str(record_id),),
            ).fetchone()
        return self._project_row(row) if row is not None else None

    def add_record(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        self._validate_collection(collection)
        normalized = self._record_for_storage(record)
        record_id = str(normalized["id"])
        try:
            with self._write_transaction():
                cursor = self._connection.execute(
                    "INSERT INTO runs(run_id, record_json) VALUES (?, ?)",
                    (record_id, self._encode_json(normalized)),
                )
                sequence = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate run id: {record_id}") from exc
        return self._project_record(normalized, sequence)

    def update_record(
        self,
        collection: str,
        record_id: str,
        updates: dict[str, Any] | Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        self._validate_collection(collection)
        record_id = str(record_id)
        with self._write_transaction():
            row = self._connection.execute(
                "SELECT sequence, record_json FROM runs WHERE run_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise KeyError(record_id)

            sequence = int(row["sequence"])
            record = self._project_row(row)
            if callable(updates):
                updates(record)
            else:
                record.update(copy.deepcopy(updates))
            normalized = self._record_for_storage(record, expected_id=record_id)
            self._connection.execute(
                "UPDATE runs SET record_json = ? WHERE run_id = ?",
                (self._encode_json(normalized), record_id),
            )
        return self._project_record(normalized, sequence)

    def mutate(
        self, callback: Callable[[dict[str, Any]], Any]
    ) -> tuple[dict[str, Any], Any]:
        """Atomically mutate the complete run mapping.

        This compatibility method intentionally mirrors the former in-memory
        store. New code should prefer record-level updates so transactions stay
        small.
        """

        state, result = self.mutate_state(callback)
        return {"runs": state["runs"]}, result

    def mutate_state(
        self, callback: Callable[[dict[str, Any]], Any]
    ) -> tuple[dict[str, Any], Any]:
        """Atomically mutate runs and queue control under one SQLite lock."""

        self._validate_collection("runs")
        with self._write_transaction():
            rows = self._connection.execute(
                "SELECT sequence, run_id, record_json FROM runs ORDER BY sequence"
            ).fetchall()
            control_row = self._connection.execute(
                "SELECT record_json FROM queue_control WHERE singleton = 1"
            ).fetchone()
            original_ids = {str(row["run_id"]) for row in rows}
            sequences = {str(row["run_id"]): int(row["sequence"]) for row in rows}
            working_runs = {str(row["run_id"]): self._project_row(row) for row in rows}
            working_control = (
                self._decode_record(control_row["record_json"])
                if control_row is not None
                else copy.deepcopy(DEFAULT_QUEUE_CONTROL)
            )
            working = {
                "runs": working_runs,
                "queue_control": working_control,
            }
            result = callback(working)
            normalized_runs = self._normalize_run_mapping(working.get("runs"))
            stored_runs = {
                record_id: self._record_for_storage(record, expected_id=record_id)
                for record_id, record in normalized_runs.items()
            }
            normalized_control = self._normalize_queue_control(
                working.get("queue_control")
            )

            removed_ids = original_ids - stored_runs.keys()
            if removed_ids:
                self._connection.executemany(
                    "DELETE FROM runs WHERE run_id = ?",
                    ((record_id,) for record_id in removed_ids),
                )
            for record_id, record in stored_runs.items():
                encoded = self._encode_json(record)
                if record_id in original_ids:
                    self._connection.execute(
                        "UPDATE runs SET record_json = ? WHERE run_id = ?",
                        (encoded, record_id),
                    )
                else:
                    cursor = self._connection.execute(
                        "INSERT INTO runs(run_id, record_json) VALUES (?, ?)",
                        (record_id, encoded),
                    )
                    sequences[record_id] = int(cursor.lastrowid)
            self._connection.execute(
                """
                INSERT INTO queue_control(singleton, record_json) VALUES (1, ?)
                ON CONFLICT(singleton) DO UPDATE
                SET record_json = excluded.record_json
                """,
                (self._encode_json(normalized_control),),
            )

        persisted_runs = {
            record_id: self._project_record(
                stored_runs[record_id], sequences[record_id]
            )
            for record_id in sorted(stored_runs, key=sequences.__getitem__)
        }
        persisted = {
            "runs": persisted_runs,
            "queue_control": copy.deepcopy(normalized_control),
        }
        persisted_result = copy.deepcopy(result)
        if isinstance(result, dict) and "id" in result:
            persisted_record = persisted_runs.get(str(result["id"]))
            if persisted_record is not None:
                persisted_result = copy.deepcopy(persisted_record)
        return persisted, persisted_result

    def get_queue_control(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                "SELECT record_json FROM queue_control WHERE singleton = 1"
            ).fetchone()
        if row is None:  # Defensive: the schema always installs this row.
            return copy.deepcopy(DEFAULT_QUEUE_CONTROL)
        return self._normalize_queue_control(self._decode_record(row["record_json"]))

    def update_queue_control(
        self,
        updates: Mapping[str, Any] | Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        """Atomically update pause state and temporary runner identity."""

        with self._write_transaction():
            row = self._connection.execute(
                "SELECT record_json FROM queue_control WHERE singleton = 1"
            ).fetchone()
            control = (
                self._normalize_queue_control(self._decode_record(row["record_json"]))
                if row is not None
                else copy.deepcopy(DEFAULT_QUEUE_CONTROL)
            )
            if callable(updates):
                updates(control)
            else:
                control.update(copy.deepcopy(dict(updates)))
            normalized = self._normalize_queue_control(control)
            self._connection.execute(
                """
                INSERT INTO queue_control(singleton, record_json) VALUES (1, ?)
                ON CONFLICT(singleton) DO UPDATE SET record_json = excluded.record_json
                """,
                (self._encode_json(normalized),),
            )
        return copy.deepcopy(normalized)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def __enter__(self) -> SQLiteRunStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Run store is closed")

    @classmethod
    def _normalize_run_mapping(cls, value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            raise TypeError("runs must remain a dictionary")
        normalized: dict[str, dict[str, Any]] = {}
        for raw_id, record in value.items():
            record_id = str(raw_id)
            normalized[record_id] = cls._normalize_record(record, expected_id=record_id)
        return normalized

    @classmethod
    def _normalize_record(
        cls, record: Any, *, expected_id: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise TypeError("Run record must be a dictionary")
        if "id" not in record:
            raise KeyError("id")
        record_id = str(record["id"])
        if expected_id is not None and record_id != expected_id:
            raise ValueError("Run id cannot be changed")
        return cls._decode_record(cls._encode_json(record))

    @classmethod
    def _record_for_storage(
        cls, record: Any, *, expected_id: str | None = None
    ) -> dict[str, Any]:
        normalized = cls._normalize_record(record, expected_id=expected_id)
        normalized.pop(QUEUE_SEQUENCE_KEY, None)
        return normalized

    @classmethod
    def _project_record(cls, record: dict[str, Any], sequence: int) -> dict[str, Any]:
        projected = copy.deepcopy(record)
        projected[QUEUE_SEQUENCE_KEY] = int(sequence)
        return projected

    @classmethod
    def _project_row(cls, row: sqlite3.Row) -> dict[str, Any]:
        return cls._project_record(
            cls._decode_record(row["record_json"]), int(row["sequence"])
        )

    @classmethod
    def _normalize_queue_control(cls, record: Any) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise TypeError("Queue control must be a dictionary")
        normalized = cls._decode_record(cls._encode_json(record))
        if not isinstance(normalized.get("paused"), bool):
            raise TypeError("queue_control.paused must be a boolean")
        generation = normalized.setdefault("pause_generation", 0)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise TypeError(
                "queue_control.pause_generation must be a non-negative integer"
            )
        return normalized

    @staticmethod
    def _encode_json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def _decode_record(value: str) -> dict[str, Any]:
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise TypeError("Stored record must be a JSON object")
        return decoded

    @staticmethod
    def _validate_collection(collection: str) -> None:
        if collection != "runs":
            raise ValueError(f"Unknown run collection: {collection}")


class MemoryRunStore(SQLiteRunStore):
    """Backward-compatible isolated store for tests and embedded callers.

    Passing a file path opts into the same durable behavior as
    :class:`SQLiteRunStore`; omitting it creates a private in-memory database.
    """

    def __init__(
        self,
        database_path: str | Path = ":memory:",
        *,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        super().__init__(database_path, busy_timeout_ms=busy_timeout_ms)
