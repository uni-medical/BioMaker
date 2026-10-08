"""Offline acceptance of desktop quit/reconnect/stop using the real run supervisor.

This submits a synthetic NumPy loop, not an AI task or a scientific evaluation.
It requires an already installed engine and never accesses a model provider.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import select
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]

WORKER = r"""
import json, os, sys, time
from pathlib import Path
import numpy as np
target = Path(sys.argv[1])
for sequence in range(1200):
    value = int(np.arange(10000, dtype=np.int64).sum())
    stage = target.with_suffix('.part')
    stage.write_text(json.dumps({'sequence': sequence, 'value': value, 'pid': os.getpid()}))
    stage.replace(target)
    time.sleep(0.1)
"""

LAUNCHER = r"""
import json, os, subprocess, sys, time
from pathlib import Path
home = Path(sys.argv[1])
token = os.environ['BIOMAKER_LAUNCH_TOKEN']
log = home / 'workspace' / 'logs' / ('synthetic-' + token[:12])
work = home / 'workspace' / 'workspaces' / ('synthetic-' + token[:12])
log.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
(log / 'run_manifest.json').write_text(json.dumps({
    'launch_token': token, 'data_dir': str(home / 'dataset'),
    'workspace_dir': str(work), 'task_description': 'Synthetic desktop lifecycle check; no AI task.',
    'config': {'agent_steps': 1}
}))
(log / 'events.jsonl').write_text(json.dumps({'event': 'run_started'}) + '\n')
child = subprocess.Popen([sys.executable, str(home / 'worker.py'), str(home / 'heartbeat.json')], process_group=0)
(home / 'owned-child.json').write_text(json.dumps({'pid': child.pid, 'leader': os.getpid()}))
print('Synthetic NumPy worker is running; no model provider is involved.', flush=True)
while child.poll() is None:
    time.sleep(0.1)
"""


def wait_until(callback, label: str, timeout: float = 25):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            value = callback()
            if value:
                return value
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise RuntimeError(
        f"Timed out waiting for {label}: {last_error or 'condition unmet'}"
    )


class Client:
    def __init__(self, runtime: Path, engine: Path, home: Path):
        self.token = secrets.token_hex(32)
        self.home = home
        self.engine = engine
        self.runtime = runtime
        self.process: subprocess.Popen | None = None
        self.port: int | None = None
        self.stderr = (home / "sidecar-diagnostics.log").open("ab")
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if key
            not in {
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "SERPER_API_KEY",
                "JINA_API_KEY",
            }
        }
        self.environment.update(
            BIOMAKER_DESKTOP_TOKEN=self.token,
            BIOMAKER_DESKTOP_HOME=str(home),
            BIOMAKER_DESKTOP_ENGINE_ROOT=str(engine),
            BIOMAKER_DESKTOP_PARENT_PID=str(os.getpid()),
            BIOMAKER_WEBUI_METHOD_ROOT=str(engine),
            BIOMAKER_WEBUI_WORKSPACE_ROOT=str(home / "workspace"),
            BIOMAKER_WEBUI_STATE_DIR=str(home / "state"),
            BIOMAKER_ENV_FILE=str(home / "config.env"),
            BIOMAKER_WEBUI_UV=str(runtime / "bin" / "uv"),
            PYTHONPATH=os.pathsep.join(
                [
                    str(runtime / "site-packages"),
                    str(engine / "apps" / "webui"),
                    str(engine),
                ]
            ),
            UV_PROJECT_ENVIRONMENT=str(engine / ".venv"),
        )

    def start(self):
        self.process = subprocess.Popen(
            [
                str(self.runtime / "python" / "bin" / "python3"),
                "-u",
                str(DESKTOP / "service" / "main.py"),
            ],
            cwd=self.engine,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
        )
        readable, _, _ = select.select([self.process.stdout], [], [], 15)
        if not readable:
            raise RuntimeError("Sidecar did not announce readiness")
        ready = json.loads(self.process.stdout.readline())
        assert ready["type"] == "ready"
        self.port = ready["port"]
        assert self.request("/api/health")["status"] == "ok"

    def request(self, path: str, method: str = "GET"):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            method=method,
            headers={"Authorization": f"Bearer {self.token}"},
        )
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=3
        ) as response:
            return json.load(response)

    def stop(self):
        process = self.process
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process and process.stdout:
            process.stdout.close()
        self.process = None
        self.port = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=DESKTOP / "build" / "runtime")
    parser.add_argument(
        "--report",
        type=Path,
        default=DESKTOP / "output" / "queue-lifecycle-acceptance.json",
    )
    args = parser.parse_args()
    engine = args.engine.resolve()
    runtime = args.runtime.resolve()
    home = Path(tempfile.mkdtemp(prefix="biomaker-queue-lifecycle-")).resolve()
    (home / "dataset").mkdir()
    (home / "worker.py").write_text(WORKER)
    (home / "launcher.py").write_text(LAUNCHER)
    client = Client(runtime, engine, home)
    run_id = None
    proof = {
        "synthetic": True,
        "provider_calls": 0,
        "home": str(home),
        "engine": str(engine),
    }
    try:
        client.start()
        launch_code = """
import json, sys
from pathlib import Path
from webui.settings import Settings
from webui.runner import RunManager
settings = Settings.default()
manager = RunManager(settings)
record = manager.enqueue_command(
    [sys.executable, str(Path(sys.argv[1]) / 'launcher.py'), sys.argv[1]],
    cwd=settings.method_root,
    project_snapshot={'dataset_path': str(Path(sys.argv[1])/'dataset'), 'task_body': 'Synthetic desktop lifecycle check; no AI task.'},
    requested_exp_name='synthetic-desktop-lifecycle',
)
print(json.dumps({'id': record['id']}))
"""
        environment = dict(client.environment)
        environment.pop("BIOMAKER_DESKTOP_TOKEN", None)
        submitted = subprocess.run(
            [str(engine / ".venv" / "bin" / "python"), "-c", launch_code, str(home)],
            cwd=engine,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        run_id = json.loads(submitted.stdout)["id"]
        proof["run_id"] = run_id
        heartbeat = home / "heartbeat.json"
        first = wait_until(
            lambda: json.loads(heartbeat.read_text()), "worker heartbeat"
        )
        record = wait_until(
            lambda: (
                value
                if (value := client.request(f"/api/runs/{run_id}")).get("log_dir")
                else None
            ),
            "recorded manifest",
        )
        assert record["status"] == "running"
        proof["child_pid"] = first["pid"]
        proof["initial_status"] = record["status"]
        client.stop()
        after = wait_until(
            lambda: (
                value
                if (value := json.loads(heartbeat.read_text()))["sequence"]
                >= first["sequence"] + 5
                else None
            ),
            "computation after desktop service quit",
        )
        proof["advanced_while_service_closed"] = after["sequence"] - first["sequence"]
        # A fresh token emulates a new Electron session reconnecting to the ledger.
        client.stderr.close()
        client = Client(runtime, engine, home)
        client.start()
        restored = client.request(f"/api/runs/{run_id}")
        assert restored["status"] == "running" and "stop" in restored["allowed_actions"]
        proof["reconnected_status"] = restored["status"]
        stopping = client.request(f"/api/runs/{run_id}/stop", "POST")
        assert stopping["status"] in {"stopping", "incomplete"}
        stopped = wait_until(
            lambda: (
                value
                if (value := client.request(f"/api/runs/{run_id}"))["status"]
                == "incomplete"
                else None
            ),
            "owned run termination",
        )
        assert stopped["detailed_status"] == "stopped", stopped
        child_check = subprocess.run(
            [
                str(engine / ".venv" / "bin" / "python"),
                "-c",
                "import psutil,sys; p=int(sys.argv[1]); print(not psutil.pid_exists(p) or psutil.Process(p).status()==psutil.STATUS_ZOMBIE)",
                str(first["pid"]),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        assert child_check.stdout.strip() == "True"
        proof.update(
            final_status=stopped["detailed_status"],
            owned_child_terminated=True,
            passed=True,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(proof, indent=2) + "\n")
        print(json.dumps(proof, indent=2))
    finally:
        if run_id and client.port:
            try:
                if client.request(f"/api/runs/{run_id}")["status"] == "running":
                    client.request(f"/api/runs/{run_id}/stop", "POST")
            except (OSError, ValueError):
                pass
        client.stop()
        client.stderr.close()


if __name__ == "__main__":
    main()
