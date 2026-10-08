"""Single-run reads must not load unrelated histories or stale lifecycle state."""

import json
import os
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from webui import server


def history(settings, name):
    directory = settings.logs_root / name
    directory.mkdir()
    (directory / "run_manifest.json").write_text(json.dumps({
        "data_dir": str(settings.datasets_root / "synthetic"),
        "config": {"agent_steps": 10},
    }))
    (directory / "events.jsonl").write_text(
        '{"event":"run_finished","completed_steps":3}\n'
    )
    (directory / f"{name}.log").write_text("synthetic console\n")
    (directory / "report.md").write_text("synthetic report\n")
    return {
        "id": name, "status": "succeeded", "read_only": True,
        "log_dir": str(directory), "console_log": str(directory / f"{name}.log"),
    }


@pytest.mark.parametrize("suffix", ["", "/updates", "/events", "/log"])
def test_single_run_routes_use_indexed_ledger_lookup(settings, monkeypatch, suffix):
    app = server.create_app(settings)
    target = "target # 中文"
    for name in ("unrelated-first", target, "unrelated-last"):
        app.state.run_store.add_record("runs", history(settings, name))

    def no_full_scan(*_args, **_kwargs):
        pytest.fail("A single-run read scanned the entire ledger")

    monkeypatch.setattr(app.state.run_store, "list_records", no_full_scan)
    original = app.state.observations.catalog
    inspected = []

    def selected_only(directory):
        inspected.append(directory.name)
        return original(directory)

    monkeypatch.setattr(app.state.observations, "catalog", selected_only)
    try:
        response = TestClient(app).get(f"/api/runs/{quote(target, safe='')}{suffix}")
        assert response.status_code == 200
        assert inspected == [target]
        missing = TestClient(app).get(f"/api/runs/missing{suffix}")
        assert missing.status_code == 404
        assert inspected == [target]
    finally:
        app.state.run_store.close()


def test_cached_observations_do_not_own_current_ledger_lifecycle(settings):
    app = server.create_app(settings)
    record = history(settings, "lifecycle")
    record.update(status="running", read_only=False)
    store = app.state.run_store
    store.add_record("runs", record)
    client = TestClient(app)
    try:
        for state, canonical, actions in [
            ("running", "running", ["stop"]),
            ("queued", "queued", ["cancel"]),
            ("cancelled", "incomplete", ["requeue"]),
        ]:
            store.update_record("runs", "lifecycle", {"status": state})
            detail = client.get("/api/runs/lifecycle").json()
            assert detail["status"] == canonical
            assert detail["detailed_status"] == state
            assert detail["allowed_actions"] == actions
            assert detail["status_source"] == "ledger"
            assert detail["terminal_event"] == "run_finished"
            assert detail["progress"]["completed_steps"] == 3
            assert client.get("/api/runs").json() == [
                {key: value for key, value in detail.items() if key not in {"observed", "artifacts"}}
            ]
    finally:
        store.close()


def test_update_signal_detects_replacement_with_retained_size_and_mtime(settings):
    app = server.create_app(settings)
    record = history(settings, "replaced")
    app.state.run_store.add_record("runs", record)
    directory = settings.logs_root / "replaced"
    client = TestClient(app)
    try:
        original = client.get("/api/runs/replaced/updates").json()
        assert client.get("/api/runs/replaced").json()["observed"]["last_event"]["completed_steps"] == 3
        source = directory / "events.jsonl"
        metadata = source.stat()
        replacement = directory / "replacement.tmp"
        replacement.write_bytes(source.read_bytes().replace(b'"completed_steps":3', b'"completed_steps":7'))
        os.utime(replacement, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        replacement.replace(source)
        changed = client.get("/api/runs/replaced/updates").json()
        assert source.stat().st_size == metadata.st_size
        assert source.stat().st_mtime_ns == metadata.st_mtime_ns
        assert original["events"]["revision"] != changed["events"]["revision"]
        assert original["run_revision"] != changed["run_revision"]
        detail = client.get("/api/runs/replaced").json()
        assert detail["progress"]["completed_steps"] == 7
        assert detail["observed"]["last_event"]["completed_steps"] == 7
    finally:
        app.state.run_store.close()
