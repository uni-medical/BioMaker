"""Single-run reads must not load unrelated histories or stale lifecycle state."""

import json
import os
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from webui import server
from webui.run_catalog import run_from_log_dir


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


def test_service_and_http_share_fresh_ledger_and_observed_views(settings):
    app = server.create_app(settings)
    record = history(settings, "shared-views")
    record.update(status="running", read_only=False)
    store = app.state.run_store
    service = app.state.run_service
    assert service.observations is app.state.observations
    store.add_record("runs", record)
    results = settings.logs_root / record["id"] / "results.tsv"
    results.write_text("step\tmetric\tmaximize\tstatus\n1\t0.7\ttrue\tsucceeded\n")
    client = TestClient(app)
    try:
        for state, canonical, actions in [
            ("running", "running", ["stop"]),
            ("cancelled", "incomplete", ["requeue"]),
        ]:
            store.update_record("runs", record["id"], {"status": state})
            raw = service.get(record["id"])
            detail = service.detail(record["id"])
            status = service.status(record["id"])
            assert raw["status"] == state
            assert "observed" not in raw and "allowed_actions" not in raw
            assert detail["status"] == canonical
            assert detail["allowed_actions"] == actions
            assert detail["terminal_event"] == "run_finished"
            assert status == {
                key: value for key, value in detail.items()
                if key not in {"observed", "artifacts"}
            }
            assert service.list_runs() == client.get("/api/runs").json() == [status]
            assert detail == client.get(f'/api/runs/{record["id"]}').json()
            assert service.results(record["id"]) == detail["observed"]["results"]
            assert service.results(record["id"])["best_metric"] == 0.7
            assert service.get(record["id"]) == raw
        # The cache used through either adapter must invalidate when results grow.
        with results.open("a") as handle:
            handle.write("2\t0.9\ttrue\tsucceeded\n")
        current = client.get(f'/api/runs/{record["id"]}').json()
        assert service.results(record["id"]) == current["observed"]["results"]
        assert service.results(record["id"])["best_metric"] == 0.9
        assert service.results(record["id"])["count"] == 2
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["status", "detail", "results"])
def test_service_single_run_reads_use_indexed_lookup(settings, monkeypatch, operation):
    app = server.create_app(settings)
    store = app.state.run_store
    service = app.state.run_service
    for name in ("unrelated", "selected"):
        store.add_record("runs", history(settings, name))

    def no_full_scan(*_args, **_kwargs):
        pytest.fail("A service single-run read scanned the entire ledger")

    monkeypatch.setattr(store, "list_records", no_full_scan)
    original = service.observations.catalog
    inspected = []

    def selected_only(directory):
        inspected.append(directory.name)
        return original(directory)

    monkeypatch.setattr(service.observations, "catalog", selected_only)
    try:
        assert getattr(service, operation)("selected") is not None
        assert inspected == ["selected"]
        if operation == "status":
            assert service.status("missing") is None
        else:
            with pytest.raises(KeyError, match="missing"):
                getattr(service, operation)("missing")
        assert inspected == ["selected"]
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


@pytest.mark.parametrize(
    "state,read_only,reason,error,canonical,actions,expected_reason",
    [
        ("succeeded", True, None, None, "complete", [], None),
        ("running", False, None, None, "running", ["stop"], None),
        ("queued", False, None, None, "queued", ["cancel"], None),
        ("failed", False, "ledger failure", "original failure", "incomplete", ["requeue"], "ledger failure"),
        ("failed", False, None, "original failure", "incomplete", ["requeue"], "original failure"),
        ("stopped", False, None, None, "incomplete", ["requeue"], "Stopped by the user."),
        ("succeeded", True, "retained ledger note", None, "complete", [], "retained ledger note"),
    ],
)
def test_tail_evidence_never_supplies_ledger_lifecycle_reason(
    settings, state, read_only, reason, error, canonical, actions, expected_reason
):
    app = server.create_app(settings)
    record = history(settings, "owned-lifecycle")
    record.update(status=state, read_only=read_only, status_reason=reason, error=error,
                  terminal_event="run_finished")
    store = app.state.run_store
    store.add_record("runs", record)
    persisted = store.get_record("runs", record["id"])
    source = settings.logs_root / record["id"] / "events.jsonl"
    client = TestClient(app)
    try:
        # Both missing terminal evidence and a contradictory terminal error are
        # observations. Neither becomes a ledger-owned lifecycle explanation.
        for text, terminal in [
            ('{"event":"observation"}\n', None),
            ('{"event":"run_failed","error":"tail failure"}\n', "run_failed"),
        ]:
            source.write_text(text)
            detail = client.get(f'/api/runs/{record["id"]}').json()
            assert detail["status"] == canonical
            assert detail["detailed_status"] == state
            assert detail["status_source"] == "ledger"
            assert detail["read_only"] is read_only
            assert detail["allowed_actions"] == actions
            assert detail.get("status_reason") == expected_reason
            assert detail.get("error") == error
            assert detail["terminal_event"] == terminal
            assert detail["event_tail"] == {
                "valid_event_limit": 5000, "valid_event_count": 1,
                "terminal_event": terminal,
                "reason_code": "no_terminal_in_tail" if terminal is None else None,
            }
            assert client.get("/api/runs").json() == [
                {key: value for key, value in detail.items() if key not in {"observed", "artifacts"}}
            ]
            assert store.get_record("runs", record["id"]) == persisted
    finally:
        store.close()


def test_completed_history_keeps_lifecycle_when_terminal_leaves_tail(settings):
    app = server.create_app(settings)
    record = history(settings, "long-completed-history")
    directory = settings.logs_root / record["id"]
    store = app.state.run_store
    imported = app.state.manager.import_log_record(run_from_log_dir(settings, directory))
    source = directory / "events.jsonl"
    client = TestClient(app)
    try:
        before = client.get(f'/api/runs/{record["id"]}').json()
        assert before["status"] == "complete"
        assert before["event_tail"]["valid_event_count"] == 1
        with source.open("a") as handle:
            handle.write('{"event":"observation"}\n' * 4999)
            handle.write('not-json\n[]\n\n')
        boundary = client.get(f'/api/runs/{record["id"]}').json()
        assert boundary["event_tail"]["valid_event_count"] == 5000
        assert boundary["event_tail"]["terminal_event"] == "run_finished"
        assert boundary["observed"]["event_count"] == 5002
        with source.open("a") as handle:
            handle.write('{"event":"observation"}\n')
        after = client.get(f'/api/runs/{record["id"]}').json()
        assert after["event_tail"] == {
            "valid_event_limit": 5000, "valid_event_count": 5000,
            "terminal_event": None, "reason_code": "no_terminal_in_tail",
        }
        assert after["terminal_event"] is None  # Never retain the imported observation.
        assert after["observed"]["event_count"] == 5003
        for key in ("status", "detailed_status", "status_source", "read_only",
                    "managed", "status_reason", "error", "allowed_actions"):
            assert after[key] == before[key]
        assert after["status_reason"] is None
        assert store.get_record("runs", record["id"]) == imported
    finally:
        store.close()


@pytest.mark.parametrize("missing", ["events.jsonl", "run_manifest.json"])
def test_missing_history_observations_do_not_resurrect_imported_tail(settings, missing):
    app = server.create_app(settings)
    record = history(settings, "missing-observation")
    directory = settings.logs_root / record["id"]
    store = app.state.run_store
    imported = app.state.manager.import_log_record(run_from_log_dir(settings, directory))
    client = TestClient(app)
    try:
        assert client.get(f'/api/runs/{record["id"]}').json()["terminal_event"] == "run_finished"
        source = directory / missing
        saved = source.read_bytes()
        source.unlink()
        missing_detail = client.get(f'/api/runs/{record["id"]}').json()
        assert missing_detail["status"] == "complete"
        assert missing_detail["status_reason"] is None
        assert missing_detail["terminal_event"] is None
        if missing == "run_manifest.json":
            assert "event_tail" not in missing_detail  # No current catalog, not an empty tail.
        else:
            assert missing_detail["event_tail"] == {
                "valid_event_limit": 5000, "valid_event_count": 0,
                "terminal_event": None, "reason_code": "no_terminal_in_tail",
            }
        source.write_bytes(saved)
        restored = client.get(f'/api/runs/{record["id"]}').json()
        assert restored["event_tail"]["terminal_event"] == "run_finished"
        assert restored["event_tail"]["reason_code"] is None
        assert restored["terminal_event"] == "run_finished"
        assert store.get_record("runs", record["id"]) == imported
    finally:
        store.close()


def test_legacy_import_reason_stays_a_ledger_record_when_tail_changes(settings):
    app = server.create_app(settings)
    record = history(settings, "legacy-incomplete-history")
    legacy_reason = (
        "The process is not known to be alive and no terminal lifecycle event was recorded."
    )
    record.update(status="incomplete", detailed_status="orphaned", managed=False,
                  status_reason=legacy_reason, error=legacy_reason)
    store = app.state.run_store
    store.add_record("runs", record)
    persisted = store.get_record("runs", record["id"])
    try:
        detail = TestClient(app).get(f'/api/runs/{record["id"]}').json()
        assert detail["status"] == "incomplete"
        assert detail["detailed_status"] == "orphaned"
        assert detail["status_source"] == "ledger"
        assert detail["status_reason"] == detail["error"] == legacy_reason
        assert detail["event_tail"]["terminal_event"] == "run_finished"
        assert detail["allowed_actions"] == []
        assert store.get_record("runs", record["id"]) == persisted
    finally:
        store.close()
