from __future__ import annotations

import io
import json
import threading
import time

from fastapi.testclient import TestClient

import webui.server as server
from webui.server import create_app


class ImmediateProcess:
    pid = None

    def __init__(self, *_args, **_kwargs):
        self.stdout = io.StringIO("finished\n")
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class BlockingProcess(ImmediateProcess):
    def __init__(self, *_args, **_kwargs):
        self.stdout = io.StringIO("running\n")
        self.returncode = None
        self.done = threading.Event()

    def wait(self, timeout=None):
        self.done.wait(timeout or 2)
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.done.set()

    def kill(self):
        self.returncode = -9
        self.done.set()


def wait_for_api_status(client, run_id, expected, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        current = response.json()["status"]
        if current in expected:
            return current
        time.sleep(0.02)
    raise AssertionError(f"Run did not reach {expected}")


def project_payload(dataset, template):
    return {
        "name": "Derm project",
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


def test_project_env_run_and_artifact_api(settings):
    dataset, template = prepare_workspace(settings)
    app = create_app(settings, process_factory=ImmediateProcess)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        templates = client.get("/api/task-templates").json()
        assert templates[0]["dataset_path"] == str(dataset.resolve())

        invalid = project_payload(dataset, template)
        invalid["dataset_path"] = str(dataset / "missing")
        assert client.post("/api/projects", json=invalid).status_code == 422

        response = client.post("/api/projects", json=project_payload(dataset, template))
        assert response.status_code == 201
        project = response.json()
        assert client.get("/api/projects").json()[0]["id"] == project["id"]

        secret = "api-secret-value-9876"
        response = client.put(
            "/api/env",
            json={
                "OPENAI_API_KEY": secret,
                "OPENAI_BASE_URL": "https://api.example.test/v1",
                "BIOMAKER_MAIN_MODEL": "model-one",
                "BIOMAKER_HELPER_MODEL": "model-one",
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
        assert environment["defaults"]["main_model"] == "model-one"
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

        response = client.post("/api/runs", json={"project_id": project["id"]})
        assert response.status_code == 202
        run = response.json()
        assert wait_for_api_status(client, run["id"], {"succeeded"}) == "succeeded"
        assert "finished" in client.get(f"/api/runs/{run['id']}/log").text

        log_dir = settings.logs_root / f"1-{run['requested_exp_name']}"
        log_dir.mkdir()
        (log_dir / "tree_plot.html").write_text("<html>tree</html>", encoding="utf-8")
        app.state.registry.update_record(
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
        app.state.registry.update_record("runs", run["id"], {"log_dir": str(outside)})
        assert (
            client.get(f"/artifacts/runs/{run['id']}/tree_plot.html").status_code == 404
        )

        assert client.delete(f"/api/projects/{project['id']}").status_code == 204
        assert client.get(f"/api/projects/{project['id']}").status_code == 404


def test_model_discovery_uses_configured_or_supplied_credentials(settings, monkeypatch):
    settings.env_file.write_text(
        "OPENAI_BASE_URL=https://configured.example/v1\n"
        "OPENAI_API_KEY=configured-secret\n"
        "OPENAI_MODEL=fallback-model\n",
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
        assert configured.json()["default_model"] == "fallback-model"
        assert configured.json()["default_model_source"] == "OPENAI_MODEL"
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
    with TestClient(app) as client:
        project = client.post(
            "/api/projects", json=project_payload(dataset, template)
        ).json()
        first = client.post("/api/runs", json={"project_id": project["id"]}).json()
        assert wait_for_api_status(client, first["id"], {"running"}) == "running"
        second = client.post("/api/runs", json={"project_id": project["id"]}).json()
        assert wait_for_api_status(client, second["id"], {"queued"}) == "queued"

        assert client.post(f"/api/runs/{second['id']}/cancel").status_code == 200
        assert wait_for_api_status(client, second["id"], {"cancelled"}) == "cancelled"
        assert client.post(f"/api/runs/{first['id']}/stop").status_code == 202
        assert wait_for_api_status(client, first["id"], {"stopped"}) == "stopped"

        retry = client.post(f"/api/runs/{first['id']}/requeue")
        assert retry.status_code == 202
        assert retry.json()["id"] != first["id"]
