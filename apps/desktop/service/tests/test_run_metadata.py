from __future__ import annotations

import json
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError

SERVICE = Path(__file__).resolve().parents[1]
REPOSITORY = SERVICE.parents[2]
sys.path.insert(0, str(SERVICE))
sys.path.insert(0, str(REPOSITORY / "apps/webui"))
sys.path.insert(0, str(REPOSITORY))

from run_metadata import RunMetadataError, RunMetadataStore, RunMetadataUpdate, default_metadata
from webui.run_store import SQLiteRunStore
from test_desktop_service import AUTH, client_for, config, create_desktop_app


@pytest.fixture
def stores(tmp_path):
    ledger = SQLiteRunStore(tmp_path / "runs.sqlite3")
    metadata = RunMetadataStore(tmp_path / "desktop-metadata.sqlite3", ledger)
    yield ledger, metadata
    metadata.close()


def add_run(ledger, identifier="existing", *, status="succeeded", managed=True, cleanup=False):
    record = {
        "id": identifier, "status": status, "managed": managed, "read_only": not managed,
        "cleanup_pending": cleanup, "requested_exp_name": "original-name",
        "actual_exp_name": "0-original-name", "pid": None, "launch_token": "original-token",
        "project_snapshot": {"dataset_path": "/original/data", "task_body": "Original task"},
    }
    return ledger.add_record("runs", record)


def ledger_dump(ledger):
    with sqlite3.connect(ledger.database_path) as connection:
        return "\n".join(connection.iterdump())


def test_default_and_unknown_records_do_not_create_metadata(stores):
    ledger, metadata = stores
    add_run(ledger)
    assert metadata.get("existing") == default_metadata("existing")
    assert metadata.list_items() == {"items": []}
    with pytest.raises(RunMetadataError) as exc:
        metadata.get("unknown")
    assert exc.value.status_code == 404
    with pytest.raises(RunMetadataError) as exc:
        metadata.update("unknown", RunMetadataUpdate(expected_revision=0, display_name="test"))
    assert exc.value.status_code == 404
    assert metadata.list_items() == {"items": []}


def test_names_archive_and_restart_preserve_original_ledger_and_artifacts(stores, tmp_path):
    ledger, metadata = stores
    add_run(ledger, managed=False, status="incomplete")
    source = tmp_path / "research-source"
    source.mkdir()
    artifact = source / "report.md"
    artifact.write_text("# Unchanged evidence\n")
    before_file = (artifact.read_bytes(), artifact.stat().st_mtime_ns)
    before_ledger = ledger_dump(ledger)
    renamed = metadata.update("existing", RunMetadataUpdate(expected_revision=0, display_name="  中文实验 🔬  "))
    assert renamed["display_name"] == "中文实验 🔬" and renamed["revision"] == 1
    archived = metadata.update("existing", RunMetadataUpdate(expected_revision=1, archived=True))
    assert archived["archived"] is True and archived["display_name"] == "中文实验 🔬"
    assert archived["revision"] == 2 and archived["updated_at"]
    metadata.close()
    restored_store = RunMetadataStore(metadata.path, ledger)
    try:
        assert restored_store.get("existing") == archived
        assert restored_store.list_items() == {"items": [archived]}
        restored = restored_store.update("existing", RunMetadataUpdate(expected_revision=2, archived=False, display_name="   "))
        assert restored["display_name"] is None and restored["archived"] is False
        assert restored["revision"] == 3
    finally:
        restored_store.close()
    assert ledger_dump(ledger) == before_ledger
    assert (artifact.read_bytes(), artifact.stat().st_mtime_ns) == before_file


def test_cas_conflict_does_not_lose_saved_values(stores):
    ledger, metadata = stores
    add_run(ledger)
    saved = metadata.update("existing", RunMetadataUpdate(expected_revision=0, display_name="First editor"))
    with pytest.raises(RunMetadataError) as exc:
        metadata.update("existing", RunMetadataUpdate(expected_revision=0, archived=True, display_name="Stale editor"))
    assert exc.value.status_code == 409
    assert metadata.get("existing") == saved


def test_concurrent_connections_allow_only_one_matching_revision(stores):
    ledger, metadata = stores
    add_run(ledger)
    second = RunMetadataStore(metadata.path, ledger)
    gate = threading.Barrier(2)
    def save(store, value):
        gate.wait(timeout=3)
        try:
            return store.update("existing", RunMetadataUpdate(expected_revision=0, display_name=value))
        except RunMetadataError as exc:
            return exc.status_code
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(save, metadata, "One"), pool.submit(save, second, "Two")]
            results = [future.result(timeout=5) for future in futures]
        assert results.count(409) == 1
        assert metadata.get("existing")["revision"] == 1
        assert metadata.get("existing")["display_name"] in {"One", "Two"}
    finally:
        second.close()


@pytest.mark.parametrize("status", ["queued", "running", "stopping", "orphaned", "unknown"])
def test_managed_nonterminal_archive_is_rejected_but_rename_is_allowed(stores, status):
    ledger, metadata = stores
    add_run(ledger, status=status)
    before = ledger_dump(ledger)
    with pytest.raises(RunMetadataError) as exc:
        metadata.update("existing", RunMetadataUpdate(expected_revision=0, archived=True))
    assert exc.value.status_code == 409
    assert metadata.get("existing")["revision"] == 0
    renamed = metadata.update("existing", RunMetadataUpdate(expected_revision=0, display_name="Working display name"))
    assert renamed["display_name"] == "Working display name"
    assert ledger_dump(ledger) == before


@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled", "stopped", "interrupted"])
def test_managed_terminal_archive_is_allowed(stores, status):
    ledger, metadata = stores
    add_run(ledger, status=status)
    assert metadata.update("existing", RunMetadataUpdate(expected_revision=0, archived=True))["archived"] is True


@pytest.mark.parametrize("managed", [True, False])
def test_cleanup_blocks_archive_and_restore_remains_allowed(stores, managed):
    ledger, metadata = stores
    add_run(ledger, managed=managed)
    metadata.update("existing", RunMetadataUpdate(expected_revision=0, archived=True))
    ledger.update_record("runs", "existing", {"status": "running", "cleanup_pending": True})
    stored = metadata.get("existing")
    assert stored["archived"] is True and stored["revision"] == 1
    with pytest.raises(RunMetadataError, match="cleaning up"):
        metadata.update("existing", RunMetadataUpdate(expected_revision=1, archived=True))
    restored = metadata.update("existing", RunMetadataUpdate(expected_revision=1, archived=False))
    assert restored["archived"] is False and restored["revision"] == 2


def test_lifecycle_read_and_metadata_commit_share_ledger_writer_barrier(stores, monkeypatch):
    ledger, metadata = stores
    add_run(ledger)
    second_ledger = SQLiteRunStore(ledger.database_path)
    entered = threading.Event()
    finish_archive = threading.Event()
    writer_finished = threading.Event()
    original = metadata._locked_run
    @contextmanager
    def held(identifier):
        with original(identifier) as run:
            entered.set()
            assert finish_archive.wait(3)
            yield run
    monkeypatch.setattr(metadata, "_locked_run", held)
    def reactivate():
        second_ledger.update_record("runs", "existing", {"status": "running"})
        writer_finished.set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        archive = pool.submit(metadata.update, "existing", RunMetadataUpdate(expected_revision=0, archived=True))
        assert entered.wait(3)
        writer = pool.submit(reactivate)
        assert not writer_finished.wait(0.05)
        finish_archive.set()
        assert archive.result(timeout=3)["archived"] is True
        writer.result(timeout=3)
    # The field remains faithful to its stored revision. UI must keep active
    # runs visible even when another actor reactivates an archived record later.
    assert metadata.get("existing")["archived"] is True
    assert ledger.get_record("runs", "existing")["status"] == "running"
    with pytest.raises(RunMetadataError, match="finished"):
        metadata.update("existing", RunMetadataUpdate(expected_revision=1, archived=True))


@pytest.mark.parametrize("values", [
    {"expected_revision": True, "archived": True},
    {"expected_revision": "0", "archived": True},
    {"expected_revision": -1, "archived": True},
    {"expected_revision": 0, "archived": "true"},
    {"expected_revision": 0, "archived": 1},
    {"expected_revision": 0, "archived": None},
    {"expected_revision": 0, "display_name": 5},
    {"expected_revision": 0, "display_name": "line\nbreak"},
    {"expected_revision": 0, "display_name": "\tname"},
    {"expected_revision": 0, "display_name": "name\u2028split"},
    {"expected_revision": 0, "display_name": "x" * 121},
    {"expected_revision": 0, "status": "succeeded"},
    {"expected_revision": 0},
])
def test_strict_mutation_schema(values):
    with pytest.raises(ValidationError):
        RunMetadataUpdate.model_validate(values)


def test_trimmed_unicode_name_and_explicit_null_are_supported():
    assert RunMetadataUpdate(expected_revision=0, display_name=" " + "界" * 120 + " ").display_name == "界" * 120
    assert RunMetadataUpdate(expected_revision=0, display_name=None).display_name is None


def test_schema_version_is_persisted_and_future_version_is_not_overwritten(stores):
    ledger, metadata = stores
    metadata.close()
    with sqlite3.connect(metadata.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        connection.execute("PRAGMA user_version=2")
    with pytest.raises(RunMetadataError, match="newer"):
        RunMetadataStore(metadata.path, ledger)
    with sqlite3.connect(metadata.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_authenticated_routes_defaults_conflicts_and_restart(config):
    app = create_desktop_app(config)
    with client_for(app) as client:
        ledger = app.state.run_store
        add_run(ledger, managed=False, status="incomplete")
        original = ledger_dump(ledger)
        endpoint = "/api/desktop/run-metadata/existing"
        assert client.get("/api/desktop/run-metadata").status_code == 401
        assert client.get(endpoint).status_code == 401
        assert client.put(endpoint, json={"expected_revision": 0, "display_name": "Denied"}).status_code == 401
        assert client.get(endpoint, headers=AUTH).json() == default_metadata("existing")
        assert client.get("/api/desktop/run-metadata/missing", headers=AUTH).status_code == 404
        response = client.put(endpoint, headers=AUTH, json={"expected_revision": 0, "display_name": "  显示名称  ", "archived": True})
        assert response.status_code == 200, response.text
        saved = response.json()
        assert saved["display_name"] == "显示名称" and saved["revision"] == 1
        assert client.get("/api/desktop/run-metadata", headers=AUTH).json() == {"items": [saved]}
        assert client.put(endpoint, headers=AUTH, json={"expected_revision": 0, "display_name": "Stale"}).status_code == 409
        assert client.put(endpoint, headers=AUTH, json={"expected_revision": 1, "status": "running"}).status_code == 422
        assert client.put(endpoint, headers=AUTH, json={"expected_revision": 1, "archived": "false"}).status_code == 422
        assert ledger_dump(ledger) == original
    with client_for(create_desktop_app(config)) as client:
        assert client.get(endpoint, headers=AUTH).json() == saved
        response = client.put(endpoint, headers=AUTH, json={"expected_revision": 1, "archived": False})
        assert response.status_code == 200 and response.json()["revision"] == 2
