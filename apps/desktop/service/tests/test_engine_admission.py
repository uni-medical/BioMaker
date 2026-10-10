"""Desktop maintenance admission against real durable queue writes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading

import pytest

from test_desktop_service import AUTH, client_for, config, create_desktop_app


class ControlledEngine:
    def __init__(self):
        self.operation = None
        self.calls = []
        self.entered = None
        self.release = None

    def snapshot(self):
        return {"ready": self.operation is None, "operation": self.operation,
                "installing": self.operation in {"install", "repair"},
                "checking": self.operation == "check", "health": "unknown",
                "checked_at": None, "busy_external": False, "error": None, "log": ""}

    def perform(self, action):
        self.calls.append(action)
        if self.entered:
            self.entered.set()
            assert self.release.wait(5)
        self.operation = action
        return self.snapshot()

    def start(self): return self.perform("install")
    def check(self): return self.perform("check")
    def repair(self): return self.perform("repair")
    def cancel(self):
        self.operation = None
        return self.snapshot()
    def close(self): pass


def configured_app(config, monkeypatch):
    config.env_file.write_text('OPENAI_API_KEY="local-fake-key"\nBIOMAKER_MAIN_MODEL="fixture"\n')
    engine = ControlledEngine()
    app = create_desktop_app(config, installer=engine)
    monkeypatch.setattr(app.state.manager, "_ensure_supervisor", lambda: None)
    dataset = config.workspace / "synthetic-data"
    dataset.mkdir()
    (dataset / "fixture.bin").write_bytes(b"fixture")
    payload = {"experiment": {"dataset_path": str(dataset), "task_body": "Synthetic admission only."}}
    return app, engine, payload


@pytest.mark.parametrize("action", ["install", "check", "repair"])
def test_maintenance_requires_auth_and_never_calls_worker_without_it(config, monkeypatch, action):
    app, engine, _ = configured_app(config, monkeypatch)
    with client_for(app) as client:
        assert client.post(f"/api/desktop/engine/{action}").status_code == 401
        assert engine.calls == []


@pytest.mark.parametrize("action", ["install", "check", "repair"])
@pytest.mark.parametrize("status,cleanup", [("queued", False), ("running", False), ("stopping", False), ("failed", True)])
def test_maintenance_waits_for_queue_and_process_cleanup(config, monkeypatch, action, status, cleanup):
    app, engine, _ = configured_app(config, monkeypatch)
    with client_for(app) as client:
        app.state.run_store.add_record("runs", {"id": "owned", "managed": True,
            "status": status, "cleanup_pending": cleanup})
        before = app.state.run_store.list_records("runs")
        response = client.post(f"/api/desktop/engine/{action}", headers=AUTH)
        assert response.status_code == 409
        assert "cleanup" in response.text
        assert engine.calls == []
        assert app.state.run_store.list_records("runs") == before


def test_imported_records_do_not_block_maintenance_and_cancel_stays_available(config, monkeypatch):
    app, engine, _ = configured_app(config, monkeypatch)
    with client_for(app) as client:
        app.state.run_store.add_record("runs", {"id": "imported", "managed": False,
            "read_only": True, "status": "running", "cleanup_pending": True})
        assert client.post("/api/desktop/engine/check", headers=AUTH).status_code == 202
        assert client.post("/api/queue/resume", json={"pause_generation": 0}, headers=AUTH).status_code == 409
        assert client.post("/api/desktop/engine/cancel", headers=AUTH).status_code == 202
        assert engine.operation is None


@pytest.mark.parametrize("action", ["check", "repair"])
def test_maintenance_admission_blocks_simultaneous_submission_until_busy(config, monkeypatch, action):
    app, engine, payload = configured_app(config, monkeypatch)
    engine.entered, engine.release = threading.Event(), threading.Event()
    with client_for(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        maintenance = pool.submit(client.post, f"/api/desktop/engine/{action}", headers=AUTH)
        assert engine.entered.wait(3)
        submitted = pool.submit(client.post, "/api/runs", json=payload, headers=AUTH)
        with pytest.raises(TimeoutError):
            submitted.result(.15)
        engine.release.set()
        assert maintenance.result(5).status_code == 202
        assert submitted.result(5).status_code == 409
        assert app.state.run_store.list_records("runs") == []


def test_submission_holds_admission_until_durable_write_before_repair(config, monkeypatch):
    app, engine, payload = configured_app(config, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    enqueue = app.state.manager.enqueue_experiment
    def delayed_enqueue(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return enqueue(*args, **kwargs)
    monkeypatch.setattr(app.state.manager, "enqueue_experiment", delayed_enqueue)
    with client_for(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        submitted = pool.submit(client.post, "/api/runs", json=payload, headers=AUTH)
        assert entered.wait(3)
        maintenance = pool.submit(client.post, "/api/desktop/engine/repair", headers=AUTH)
        with pytest.raises(TimeoutError):
            maintenance.result(.15)
        release.set()
        assert submitted.result(5).status_code == 202
        assert maintenance.result(5).status_code == 409
        assert engine.calls == []
        assert len(app.state.run_store.list_records("runs")) == 1


def test_requeue_and_resume_are_rejected_during_maintenance(config, monkeypatch):
    app, engine, payload = configured_app(config, monkeypatch)
    with client_for(app) as client:
        record = client.post("/api/runs", json=payload, headers=AUTH).json()
        assert client.post(f"/api/runs/{record['id']}/cancel", headers=AUTH).status_code == 200
        assert client.post("/api/desktop/engine/repair", headers=AUTH).status_code == 202
        before = app.state.run_store.list_records("runs")
        assert client.post(f"/api/runs/{record['id']}/requeue", headers=AUTH).status_code == 409
        assert client.post("/api/queue/resume", json={"pause_generation": 0}, headers=AUTH).status_code == 409
        assert app.state.run_store.list_records("runs") == before
