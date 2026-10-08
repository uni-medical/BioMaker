"""Desktop display names and archive flags, independent of experiment evidence."""

from __future__ import annotations

import json
import sqlite3
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "stopped", "interrupted"}


class RunMetadataError(ValueError):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


class RunMetadataUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=0, le=2**63 - 2)
    display_name: str | None = None
    archived: bool = False

    @field_validator("display_name")
    @classmethod
    def valid_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if any(unicodedata.category(character) in {"Cc", "Cs", "Zl", "Zp"} for character in value):
            raise ValueError("Display names cannot contain control characters or line breaks.")
        value = value.strip()
        if len(value) > 120:
            raise ValueError("Display names must contain at most 120 characters after trimming.")
        return value or None

    @model_validator(mode="after")
    def changes_are_present(self):
        if not (self.model_fields_set & {"display_name", "archived"}):
            raise ValueError("Provide a display name or archive change.")
        return self


def default_metadata(run_id: str) -> dict[str, Any]:
    return {"run_id": run_id, "display_name": None, "archived": False, "revision": 0, "updated_at": None}


class RunMetadataStore:
    """Versioned SQLite metadata with a short authoritative lifecycle barrier.

    Organization never updates the run ledger. PUT holds a RESERVED lock on its
    existing database only while checking the record and committing metadata,
    preventing another process from changing lifecycle between those two steps.
    Lock order is ledger first, then metadata; no RunStore lock is acquired while
    holding that ledger transaction. Reads do not block ledger writers.
    """

    def __init__(self, path: Path, run_store):
        self.path = path
        self.run_store = run_store
        self._lock = threading.RLock()
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, timeout=5, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=5000")
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            self._connection.close()
            raise RunMetadataError("This organization database belongs to a newer BioMaker version. Update the app to use it.", 409)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    archived INTEGER NOT NULL CHECK (archived IN (0, 1)),
                    revision INTEGER NOT NULL CHECK (revision > 0),
                    updated_at TEXT NOT NULL
                )
            """)
            self._connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            self._connection.close()
            raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    @staticmethod
    def _row(value: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": value["run_id"], "display_name": value["display_name"],
            "archived": bool(value["archived"]), "revision": value["revision"],
            "updated_at": value["updated_at"],
        }

    def list_items(self) -> dict[str, Any]:
        known = {str(record["id"]) for record in self.run_store.list_records("runs")}
        with self._lock:
            rows = self._connection.execute("SELECT * FROM run_metadata ORDER BY run_id").fetchall()
        return {"items": [self._row(row) for row in rows if row["run_id"] in known]}

    def get(self, run_id: str) -> dict[str, Any]:
        if self.run_store.get_record("runs", run_id) is None:
            raise RunMetadataError("Run not found.", 404)
        with self._lock:
            row = self._connection.execute("SELECT * FROM run_metadata WHERE run_id=?", (run_id,)).fetchone()
            return self._row(row) if row else default_metadata(run_id)

    @contextmanager
    def _locked_run(self, run_id: str):
        path = str(self.run_store.database_path)
        if path == ":memory:":
            # Desktop production always uses an existing file-backed ledger.
            raise RunMetadataError("Organization requires a persistent run ledger.", 409)
        uri = Path(path).resolve().as_uri() + "?mode=rw"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5)
        try:
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT record_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise RunMetadataError("Run not found.", 404)
            record = json.loads(row[0])
            yield record
        finally:
            # No ledger INSERT, UPDATE, DELETE or schema command is issued.
            connection.rollback()
            connection.close()

    def update(self, run_id: str, payload: RunMetadataUpdate) -> dict[str, Any]:
        changes = payload.model_dump(include=payload.model_fields_set - {"expected_revision"})
        with self._locked_run(run_id) as run:
            if changes.get("archived") is True:
                if run.get("cleanup_pending"):
                    raise RunMetadataError("This experiment is still cleaning up. Wait for cleanup before archiving it.", 409)
                if run.get("managed", True) and run.get("status") not in TERMINAL_STATUSES:
                    raise RunMetadataError("Only finished experiments can be archived. Queued, running, or stopping experiments remain visible.", 409)
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    row = self._connection.execute("SELECT * FROM run_metadata WHERE run_id=?", (run_id,)).fetchone()
                    current = self._row(row) if row else default_metadata(run_id)
                    if current["revision"] != payload.expected_revision:
                        raise RunMetadataError("These organization settings changed elsewhere. Reload the current values before saving again.", 409)
                    current.update(changes)
                    current["revision"] += 1
                    current["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._connection.execute("""
                        INSERT INTO run_metadata(run_id, display_name, archived, revision, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(run_id) DO UPDATE SET
                            display_name=excluded.display_name, archived=excluded.archived,
                            revision=excluded.revision, updated_at=excluded.updated_at
                    """, (run_id, current["display_name"], int(current["archived"]), current["revision"], current["updated_at"]))
                    self._connection.commit()
                    return current
                except Exception:
                    self._connection.rollback()
                    raise
