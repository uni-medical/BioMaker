from __future__ import annotations

import io
import json
import threading
import time

from fastapi.testclient import TestClient

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
    with TestClient(app) as client:
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
        response = client.put("/api/env", json={"OPENAI_API_KEY": secret})
        assert response.status_code == 200
        assert secret not in response.text
        assert secret not in client.get("/api/env").text
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
