from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.parse import quote

import webui.server as server
from fastapi.testclient import TestClient
from webui.server import create_app


def prepare_fake_run(command, kwargs, output):
    launch_token = kwargs["env"]["BIOMAKER_LAUNCH_TOKEN"]

    def argument_value(name):
        prefix = f"{name}="
        return next(
            argument.removeprefix(prefix)
            for argument in command
            if argument.startswith(prefix)
        )

    logs_root = Path(argument_value("log_dir"))
    log_dir = logs_root / f"api-fake-{launch_token}"
    log_dir.mkdir(parents=True)
    (log_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "launch_token": launch_token,
                "created_at": "2026-08-21T00:00:00+00:00",
                "data_dir": argument_value("data_dir"),
                "workspace_dir": str(
                    Path(argument_value("workspace_dir")) / log_dir.name
                ),
                "config": {},
            }
        ),
        encoding="utf-8",
    )
    events_path = log_dir / "events.jsonl"
    events_path.write_text('{"event":"run_started"}\n', encoding="utf-8")
    kwargs["stdout"].write(output)
    kwargs["stdout"].flush()
    return events_path


def append_fake_event(events_path, event, **details):
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, **details}) + "\n")


class ImmediateProcess:
    pid = None

    def __init__(self, command, **kwargs):
        self.stdout = None
        events_path = prepare_fake_run(command, kwargs, "finished\n")
        append_fake_event(events_path, "run_finished")
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class NonzeroFinishedProcess(ImmediateProcess):
    def __init__(self, command, **kwargs):
        super().__init__(command, **kwargs)
        self.returncode = 1


class BlockingProcess:
    pid = None

    def __init__(self, command, **kwargs):
        self.stdout = None
        self.events_path = prepare_fake_run(command, kwargs, "running\n")
        self.returncode = None
        self.done = threading.Event()
        self.terminal_written = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.done.wait(timeout or 2)
        return self.returncode

    def terminate(self):
        if not self.terminal_written:
            append_fake_event(self.events_path, "run_interrupted", reason="stopped")
            self.terminal_written = True
        self.returncode = -15
        self.done.set()

    def kill(self):
        if not self.terminal_written:
            append_fake_event(self.events_path, "run_interrupted", reason="killed")
            self.terminal_written = True
        self.returncode = -9
        self.done.set()


def wait_for_api_status(client, run_id, expected, *, detailed_status=None, timeout=3):
    deadline = time.monotonic() + timeout
    body = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        body = response.json()
        current = body["status"]
        if current in expected:
            if detailed_status is not None:
                assert body["detailed_status"] == detailed_status
            return body
        time.sleep(0.02)
    raise AssertionError(f"Run did not reach {expected}; current={body}")


def experiment_payload(dataset, template):
    return {
        "dataset_path": str(dataset),
        "task_template_path": str(template),
        "task_body": "Train a robust classifier.",
        "advanced_defaults": {
            "agent_steps": 3,
            "exec_timeout": 30,
            "exec_parallel_workers": 1,
            "agent_total_time_limit": 120,
            "final_retrain_enabled": False,
        },
    }


def prepare_workspace(settings):
    dataset = settings.datasets_root / "dermamnist"
    dataset.mkdir()
    experiment = settings.experiments_root / "dermamnist"
    experiment.mkdir()
    template = experiment / "task.md"
    template.write_text("# Template task\n", encoding="utf-8")
    return dataset, template


def test_env_run_and_artifact_api(settings):
    dataset, template = prepare_workspace(settings)
    app = create_app(settings, process_factory=ImmediateProcess)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok"
        assert health["run_ledger"] == app.state.run_store.database_path
        templates = client.get("/api/task-templates").json()
        assert templates[0]["dataset_path"] == str(dataset.resolve())

        invalid = experiment_payload(dataset, template)
        invalid["dataset_path"] = str(dataset / "missing")
        assert client.post("/api/runs", json={"experiment": invalid}).status_code == 422

        secret = "api-secret-value-9876"
        response = client.put(
            "/api/env",
            json={
                "OPENAI_API_KEY": secret,
                "OPENAI_BASE_URL": "https://api.example.test/v1",
                "BIOMAKER_MAIN_MODEL": "model-one",
            },
        )
        assert response.status_code == 200
        assert secret not in response.text
        environment_response = client.get("/api/env")
        assert secret not in environment_response.text
        assert secret[-4:] not in environment_response.text
        environment = environment_response.json()
        assert environment["values"]["OPENAI_BASE_URL"]["value"] == (
            "https://api.example.test/v1"
        )
        assert environment["values"]["BIOMAKER_MAIN_MODEL"]["value"] == "model-one"
        assert "defaults" not in environment
        assert environment["source"] == {
            "path": str(settings.env_file),
            "exists": True,
        }
        reveal = client.post("/api/env/reveal", json={"key": "OPENAI_API_KEY"})
        assert reveal.status_code == 200
        assert reveal.json() == {
            "key": "OPENAI_API_KEY",
            "configured": True,
            "value": secret,
        }
        assert reveal.headers["cache-control"] == "no-store"
        assert (
            client.post("/api/env/reveal", json={"key": "OPENAI_BASE_URL"}).status_code
            == 422
        )
        assert secret not in json.dumps(client.get("/api/health").json())

        response = client.post(
            "/api/runs", json={"experiment": experiment_payload(dataset, template)}
        )
        assert response.status_code == 202
        run = response.json()
        completed = wait_for_api_status(
            client, run["id"], {"complete"}, detailed_status="succeeded"
        )
        assert completed["terminal_event"] == "run_finished"
        assert "finished" in client.get(f"/api/runs/{run['id']}/log").text

        log_dir = settings.logs_root / "1-api-run"
        log_dir.mkdir()
        (log_dir / "tree_plot.html").write_text("<html>tree</html>", encoding="utf-8")
        app.state.run_store.update_record(
            "runs",
            run["id"],
            {"log_dir": str(log_dir), "actual_exp_name": log_dir.name},
        )
        tree_response = client.get(f"/artifacts/runs/{run['id']}/tree_plot.html")
        assert tree_response.status_code == 200
        assert "tree" in tree_response.text

        outside = settings.workspace_root / "outside"
        outside.mkdir()
        (outside / "tree_plot.html").write_text("secret", encoding="utf-8")
        app.state.run_store.update_record("runs", run["id"], {"log_dir": str(outside)})
        assert (
            client.get(f"/artifacts/runs/{run['id']}/tree_plot.html").status_code == 404
        )


def test_nonzero_exit_cannot_publish_finished_event_as_complete(settings):
    dataset, template = prepare_workspace(settings)
    app = create_app(settings, process_factory=NonzeroFinishedProcess)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post(
            "/api/runs", json={"experiment": experiment_payload(dataset, template)}
        )
        assert response.status_code == 202
        run = response.json()
        incomplete = wait_for_api_status(
            client, run["id"], {"incomplete"}, detailed_status="failed"
        )

        assert incomplete["terminal_event"] == "run_finished"
        assert "Lifecycle mismatch" in incomplete["status_reason"]
        queue = client.get("/api/queue").json()
        assert queue["paused"] is True
        assert "Lifecycle mismatch" in queue["reason"]


def test_run_updates_exposes_revisions_and_terminal_barrier(settings):
    app = create_app(settings, process_factory=ImmediateProcess)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        run_id = "revision-run"
        log_dir = settings.logs_root / "1-revision-run"
        log_dir.mkdir()
        tree_path = log_dir / "tree_plot.html"
        events_path = log_dir / "events.jsonl"
        tree_path.write_text("<html>first tree</html>", encoding="utf-8")
        events_path.write_text('{"event":"started"}\n', encoding="utf-8")
        console_path = settings.state_dir / "console" / f"{run_id}.log"
        console_path.write_text("starting\n", encoding="utf-8")
        app.state.run_store.add_record(
            "runs",
            {
                "id": run_id,
                "status": "running",
                "log_dir": str(log_dir),
                "console_log": str(console_path),
                "timestamps": {"updated_at": "2026-08-20T00:00:00+00:00"},
            },
        )

        response = client.get(f"/api/runs/{run_id}/updates")
        assert response.status_code == 200
        first = response.json()
        assert first["status"] == "running"
        assert client.get(f"/api/runs/{run_id}").json()["detailed_status"] == "running"
        assert first["terminal"] is False
        assert first["final_barrier"] is False
        assert isinstance(first["run_revision"], str)
        assert first["tree"] == {
            "available": True,
            "revision": first["tree"]["revision"],
            "url": f"/artifacts/runs/{run_id}/tree_plot.html",
        }
        assert isinstance(first["tree"]["revision"], str)
        assert isinstance(first["events"]["revision"], str)
        assert isinstance(first["console"]["revision"], str)
        assert isinstance(first["artifacts"]["revision"], str)

        tree_path.write_text("<html>final tree is different</html>", encoding="utf-8")
        events_path.write_text(
            '{"event":"started"}\n{"event":"finished"}\n', encoding="utf-8"
        )
        console_path.write_text("starting\nfinished\n", encoding="utf-8")
        app.state.run_store.update_record(
            "runs",
            run_id,
            {
                "status": "succeeded",
                "timestamps": {"updated_at": "2026-08-20T00:00:01+00:00"},
            },
        )

        final = client.get(f"/api/runs/{run_id}/updates").json()
        assert final["status"] == "complete"
        assert final["terminal"] is True
        assert final["final_barrier"] is True
        assert final["run_revision"] != first["run_revision"]
        assert final["tree"]["revision"] != first["tree"]["revision"]
        assert final["events"]["revision"] != first["events"]["revision"]
        assert final["console"]["revision"] != first["console"]["revision"]
        assert final["artifacts"]["revision"] != first["artifacts"]["revision"]
        assert client.get(f"/api/runs/{run_id}").json()["detailed_status"] == (
            "succeeded"
        )

        assert client.get("/api/runs/missing/updates").status_code == 404


def test_reimported_unmanaged_terminal_event_updates_api_status(settings):
    log_dir = settings.logs_root / "orphan-history"
    log_dir.mkdir()
    (log_dir / "run_manifest.json").write_text(
        json.dumps({"data_dir": str(settings.datasets_root / "fixture")}), encoding="utf-8"
    )
    app = create_app(settings, process_factory=ImmediateProcess)

    with TestClient(app) as client:
        orphan = app.state.manager.import_log_record(
            {
                "id": "orphan-history",
                "status": "incomplete",
                "detailed_status": "incomplete",
                "terminal_event": None,
                "log_dir": str(log_dir),
                "timestamps": {"finished_at": None},
            }
        )
        (log_dir / "events.jsonl").write_text(
            json.dumps({"event": "run_finished", "ts": "2026-08-21T15:38:00Z"}) + "\n", encoding="utf-8"
        )
        app.state.manager.import_log_record(
            {
                **orphan,
                "status": "succeeded",
                "detailed_status": "succeeded",
                "terminal_event": "run_finished",
                "timestamps": {"finished_at": "2026-08-21T15:38:00Z"},
            }
        )

        listed = next(
            run for run in client.get("/api/runs").json() if run["id"] == orphan["id"]
        )
        updates = client.get(f"/api/runs/{orphan['id']}/updates").json()

        assert listed["status"] == "complete"
        assert listed["detailed_status"] == "succeeded"
        assert listed["terminal_event"] == "run_finished"
        assert updates["status"] == "complete"
        assert updates["terminal"] is True
        assert updates["final_barrier"] is True


def test_queue_and_system_resources_api(settings, monkeypatch):
    resource_snapshot = {
        "sampled_at": "2026-08-21T08:09:10Z",
        "stale": False,
        "error": None,
        "cpu": {"utilization_percent": 12.5},
        "memory": {
            "used_bytes": 2_000,
            "total_bytes": 8_000,
            "utilization_percent": 25.0,
        },
        "nvidia_gpus": [],
        "apple_gpu": {
            "source": "ioreg",
            "utilization_percent": 97.0,
            "shared_memory_used_bytes": 1_204_289_536,
        },
    }
    monkeypatch.setattr(server, "get_system_resources", lambda: resource_snapshot)
    app = create_app(settings, process_factory=ImmediateProcess)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        queue = client.get("/api/queue")
        assert queue.status_code == 200
        assert queue.json()["paused"] is False
        assert queue.json()["reason"] is None
        assert queue.json()["pause_generation"] == 0
        assert queue.json()["runner_token"] is None

        resumed = client.post(
            "/api/queue/resume",
            json={"pause_generation": queue.json()["pause_generation"]},
        )
        assert resumed.status_code == 202
        assert resumed.json()["paused"] is False
        assert resumed.json()["reason"] is None

        resources = client.get("/api/system/resources")
        assert resources.status_code == 200
        assert resources.json() == resource_snapshot
        assert resources.headers["cache-control"] == "no-store"

        app.state.run_store.update_queue_control(
            {"paused": True, "reason": "first pause", "pause_generation": 1}
        )
        stale_generation = app.state.run_store.get_queue_control()["pause_generation"]
        app.state.run_store.update_queue_control(
            {"paused": True, "reason": "new failure pause", "pause_generation": 2}
        )
        conflict = client.post(
            "/api/queue/resume",
            json={"pause_generation": stale_generation},
        )
        assert conflict.status_code == 409
        assert "pause changed" in conflict.json()["detail"]
        still_paused = client.get("/api/queue").json()
        assert still_paused["paused"] is True
        assert still_paused["reason"] == "new failure pause"
        assert still_paused["pause_generation"] == 2
        resumed = client.post(
            "/api/queue/resume",
            json={"pause_generation": still_paused["pause_generation"]},
        )
        assert resumed.status_code == 202
        assert resumed.json()["paused"] is False


def test_model_discovery_uses_configured_or_supplied_credentials(settings, monkeypatch):
    settings.env_file.write_text(
        "OPENAI_BASE_URL=https://configured.example/v1\n"
        "OPENAI_API_KEY=configured-secret\n",
        encoding="utf-8",
    )
    calls = []

    def fake_discover(base_url, api_key):
        calls.append((base_url, api_key))
        return {
            "status": "ok",
            "models": ["fallback-model", "another-model"],
            "base_url": base_url,
            "detail": None,
        }

    monkeypatch.setattr(server, "discover_models", fake_discover)
    app = create_app(settings, process_factory=ImmediateProcess)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        configured = client.post("/api/models/discover", json={})
        assert configured.status_code == 200
        assert calls[-1] == (
            "https://configured.example/v1",
            "configured-secret",
        )
        assert "default_model" not in configured.json()
        assert "default_model_source" not in configured.json()
        assert "configured-secret" not in configured.text

        supplied = client.post(
            "/api/models/discover",
            json={
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "temporary-key",
            },
        )
        assert supplied.status_code == 200
        assert calls[-1] == ("http://127.0.0.1:11434/v1", "temporary-key")
        assert "temporary-key" not in supplied.text

        changed_provider = client.post(
            "/api/models/discover",
            json={"base_url": "https://different.example/v1"},
        )
        assert changed_provider.status_code == 200
        assert calls[-1] == ("https://different.example/v1", None)

        marker = "VALIDATION_SECRET_MUST_NOT_APPEAR"
        oversized_secret = marker + ("x" * 20_000)
        for path, payload in (
            ("/api/models/discover", {"api_key": oversized_secret}),
            ("/api/env", {"OPENAI_API_KEY": oversized_secret}),
        ):
            if "models" in path:
                response = client.post(path, json=payload)
            else:
                response = client.put(path, json=payload)
            assert response.status_code == 422
            assert marker not in response.text


def test_secret_bearing_settings_endpoints_require_loopback(settings):
    dataset, template = prepare_workspace(settings)
    settings.env_file.write_text("OPENAI_API_KEY=do-not-return\n", encoding="utf-8")
    app = create_app(settings, process_factory=ImmediateProcess)
    with TestClient(app, client=("192.0.2.10", 50000)) as client:
        environment = client.get("/api/env")
        assert environment.status_code == 200
        assert "do-not-return" not in environment.text
        assert client.put("/api/env", json={"OPENAI_API_KEY": "new"}).status_code == 403
        assert (
            client.post("/api/env/reveal", json={"key": "OPENAI_API_KEY"}).status_code
            == 403
        )
        assert client.post("/api/models/discover", json={}).status_code == 403
        assert (
            client.post("/api/queue/resume", json={"pause_generation": 0}).status_code
            == 403
        )
        assert (
            client.post(
                "/api/runs",
                json={"experiment": experiment_payload(dataset, template)},
            ).status_code
            == 403
        )
        app.state.run_store.add_record(
            "runs",
            {
                "id": "remote-stop-target",
                "status": "running",
                "timestamps": {"created_at": "now", "updated_at": "now"},
            },
        )
        assert client.post("/api/runs/remote-stop-target/stop").status_code == 403


def test_native_directory_picker_is_loopback_only_and_accepts_no_body(
    settings, tmp_path, monkeypatch
):
    selected = tmp_path / "selected dataset"
    selected.mkdir()
    calls = []

    def fake_select_directory():
        calls.append(True)
        return {"status": "selected", "path": str(selected.resolve())}

    monkeypatch.setattr(server, "select_directory", fake_select_directory)
    app = create_app(settings, process_factory=ImmediateProcess)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.post("/api/directories/select").status_code == 403
        response = client.post(
            "/api/directories/select",
            headers={"X-BioMaker-UI": "directory-picker"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "selected",
            "path": str(selected.resolve()),
        }
        assert calls == [True]

    with TestClient(app, client=("192.0.2.10", 50000)) as client:
        response = client.post(
            "/api/directories/select",
            headers={"X-BioMaker-UI": "directory-picker"},
        )
        assert response.status_code == 403
        assert calls == [True]


def test_api_cancel_pending_and_stop_active_run(settings):
    dataset, template = prepare_workspace(settings)
    app = create_app(settings, process_factory=BlockingProcess)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        payload = {"experiment": experiment_payload(dataset, template)}
        first = client.post("/api/runs", json=payload).json()
        wait_for_api_status(client, first["id"], {"running"}, detailed_status="running")
        second = client.post("/api/runs", json=payload).json()
        wait_for_api_status(client, second["id"], {"queued"}, detailed_status="queued")

        cancel_response = client.post(f"/api/runs/{second['id']}/cancel")
        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "incomplete"
        assert cancel_response.json()["detailed_status"] == "cancelled"
        cancelled = wait_for_api_status(
            client, second["id"], {"incomplete"}, detailed_status="cancelled"
        )
        assert cancelled["status_reason"] == "Cancelled before the run started."
        stop_response = client.post(f"/api/runs/{first['id']}/stop")
        assert stop_response.status_code == 202
        assert stop_response.json()["status"] == "stopping"
        assert stop_response.json()["detailed_status"] == "stopping"
        assert stop_response.json()["allowed_actions"] == []
        stopped = wait_for_api_status(
            client, first["id"], {"incomplete"}, detailed_status="stopped"
        )
        assert stopped["status_reason"] == "Stopped by the user."
        assert stopped["terminal_event"] == "run_interrupted"
        paused = client.get("/api/queue")
        assert paused.status_code == 200
        assert paused.json()["paused"] is True
        assert "stopped" in paused.json()["reason"]

        retry = client.post(f"/api/runs/{first['id']}/requeue")
        assert retry.status_code == 202
        assert retry.json()["id"] != first["id"]
        assert retry.json()["status"] == "queued"
        assert retry.json()["detailed_status"] == "queued"
        resumed = client.post(
            "/api/queue/resume",
            json={"pause_generation": paused.json()["pause_generation"]},
        )
        assert resumed.status_code == 202
        assert resumed.json()["paused"] is False
        retry_id = retry.json()["id"]
        wait_for_api_status(client, retry_id, {"running"}, detailed_status="running")
        client.post(f"/api/runs/{retry_id}/stop")
        wait_for_api_status(client, retry_id, {"incomplete"}, detailed_status="stopped")


def test_logs_are_the_run_history_and_keep_cli_directory_names(settings):
    dataset = settings.datasets_root / "dermamnist_224"
    dataset.mkdir()
    run_name = "5-dermamnist-deepseek-v4-flash-35step"
    log_dir = settings.logs_root / run_name
    log_dir.mkdir()
    workspace_dir = settings.workspaces_root / run_name
    manifest = {
        "created_at": "2026-08-20T16:54:26+00:00",
        "exp_name": run_name,
        "data_dir": str(dataset),
        "workspace_dir": str(workspace_dir),
        "task_description": "Train and evaluate DermMNIST.",
        "config": {
            "agent_steps": 35,
            "per_node_timeout_seconds": 600,
            "total_time_limit_seconds": 86400,
            "main_model": "deepseek-v4-flash",
        },
    }
    (log_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    events = [
        {
            "ts": "2026-08-20T16:54:24+00:00",
            "event": "run_started",
            "parallel_workers": 1,
        },
        *(
            {
                "ts": f"2026-08-20T17:{step:02d}:00+00:00",
                "event": "step_saved",
                "step": step,
            }
            for step in range(35)
        ),
        {
            "ts": "2026-08-20T19:52:25+00:00",
            "event": "final_retrain_started",
        },
        {
            "ts": "2026-08-20T19:59:11+00:00",
            "event": "run_finished",
            "completed_steps": 35,
        },
    ]
    (log_dir / "events.jsonl").write_text(
        "".join(f"{json.dumps(event)}\n" for event in events), encoding="utf-8"
    )
    (log_dir / "results.tsv").write_text(
        "step\tmetric\tmaximize\toutcome\n"
        + "".join(
            f"{step}\t{0.5 + step / 100}\ttrue\tvalid_non_improving\n"
            for step in range(34)
        )
        + "34\t0.8783649052841476\ttrue\tnew_best\n",
        encoding="utf-8",
    )

    app = create_app(settings, process_factory=ImmediateProcess)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        runs = client.get("/api/runs").json()
        assert len(runs) == 1
        run = runs[0]
        assert run["id"] == run_name
        assert run["actual_exp_name"] == run_name
        assert run["dataset_name"] == "dermamnist_224"
        assert run["dataset_path"] == str(dataset)
        assert run["status"] == "complete"
        assert run["detailed_status"] == "succeeded"
        assert run["terminal_event"] == "run_finished"
        assert run["progress"] == {
            "completed_steps": 35,
            "total_steps": 35,
            "last_event": "run_finished",
        }
        assert run["options"]["model_for_all_agents"] is None
        assert run["options"]["main_model"] == "deepseek-v4-flash"
        assert run["options"]["exec_parallel_workers"] == 1
        assert run["options"]["final_retrain_enabled"] is True
        assert run["allowed_actions"] == []

        detail = client.get(f"/api/runs/{run_name}").json()
        assert detail["observed"]["event_count"] == 38
        assert detail["observed"]["results"]["count"] == 35
        assert detail["observed"]["results"]["best_metric"] == 0.8783649052841476
        assert client.post(f"/api/runs/{run_name}/requeue").status_code == 409

        duplicate = app.state.manager.import_log_record(
            {
                "id": "current-webui-session",
                "project_snapshot": {
                    "dataset_path": "/tmp/wrong-snapshot",
                    "task_body": "Stale WebUI draft",
                },
                "status": "succeeded",
                "requested_exp_name": "wrong-requested-name",
                "actual_exp_name": "wrong-session-name",
                "options": {"model_for_all_agents": "wrong-session-model"},
                "log_dir": str(log_dir),
                "timestamps": {},
            },
        )
        assert duplicate["id"] == run_name
        merged = client.get("/api/runs").json()
        assert len(merged) == 1
        assert merged[0]["id"] == run_name
        assert merged[0]["actual_exp_name"] == run_name
        assert merged[0]["dataset_path"] == str(dataset)
        assert merged[0]["dataset_name"] == "dermamnist_224"
        assert merged[0]["project_snapshot"]["task_body"] == (
            manifest["task_description"]
        )
        assert merged[0]["options"]["model_for_all_agents"] is None
        assert merged[0]["options"]["main_model"] == "deepseek-v4-flash"


def test_cli_lifecycle_events_drive_status_and_special_names_are_routable(settings):
    dataset = settings.datasets_root / "dermamnist"
    dataset.mkdir()

    def write_run(name, events):
        log_dir = settings.logs_root / name
        log_dir.mkdir()
        (log_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "created_at": "2026-08-21T00:00:00+00:00",
                    "exp_name": name,
                    "data_dir": str(dataset),
                    "workspace_dir": str(settings.workspaces_root / name),
                    "task_description": "Train from the CLI.",
                    "config": {"agent_steps": 35, "main_model": "model-a"},
                }
            ),
            encoding="utf-8",
        )
        (log_dir / "events.jsonl").write_text(
            "".join(f"{json.dumps(event)}\n" for event in events),
            encoding="utf-8",
        )
        (log_dir / "tree_plot.html").write_text("<html>tree</html>", encoding="utf-8")

    running_name = "30-cli # running"
    failed_name = "31-cli ? failed"
    write_run(
        running_name,
        [{"ts": "2026-08-21T00:00:00+00:00", "event": "run_started"}],
    )
    write_run(
        failed_name,
        [
            {"ts": "2026-08-21T00:00:00+00:00", "event": "run_started"},
            {"ts": "2026-08-21T00:01:00+00:00", "event": "run_finished"},
            {
                "ts": "2026-08-21T00:01:01+00:00",
                "event": "run_failed",
                "error": "report failed",
            },
        ],
    )

    app = create_app(settings, process_factory=ImmediateProcess)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        runs = {run["id"]: run for run in client.get("/api/runs").json()}
        assert runs[running_name]["status"] == "incomplete"
        assert runs[running_name]["detailed_status"] == "orphaned"
        assert runs[running_name]["terminal_event"] is None
        assert "no terminal lifecycle event" in runs[running_name]["status_reason"].lower()
        assert runs[failed_name]["status"] == "incomplete"
        assert runs[failed_name]["detailed_status"] == "failed"
        assert runs[failed_name]["terminal_event"] == "run_failed"
        assert runs[failed_name]["status_reason"] == "report failed"
        assert runs[failed_name]["error"] == "report failed"

        running_updates = client.get(
            f"/api/runs/{quote(running_name, safe='')}/updates"
        ).json()
        assert running_updates["terminal"] is True
        assert running_updates["final_barrier"] is True

        failed_updates = client.get(
            f"/api/runs/{quote(failed_name, safe='')}/updates"
        ).json()
        assert failed_updates["terminal"] is True
        assert failed_updates["final_barrier"] is True

        detail = client.get(f"/api/runs/{quote(running_name, safe='')}").json()
        tree = next(
            item for item in detail["artifacts"] if item["name"] == "tree_plot.html"
        )
        assert "%23" in tree["url"]
        assert client.get(tree["url"]).status_code == 200
