from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVICE = Path(__file__).resolve().parents[1]
REPOSITORY = SERVICE.parents[2]
sys.path.insert(0, str(SERVICE))

from desktop_service import DesktopConfig, EngineInstaller, create_desktop_app
import desktop_service

TOKEN = "desktop-test-session-token-" + "x" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def config(tmp_path, monkeypatch):
    # These tests exercise the installer state machine with a fake interpreter;
    # native Mach-O editing and real library execution have dedicated tests.
    monkeypatch.setattr(desktop_service, "provision_native_runtime", lambda *_args, **_kwargs: {"test_native_verified": True})
    monkeypatch.setattr(desktop_service, "native_receipt_is_current", lambda _root, receipt: receipt == {"test_native_verified": True})
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "pyproject.toml").write_text(
        '[project]\nname = "desktop-test-engine"\nversion = "1.0.0"\n'
    )
    (engine / "apps").mkdir()
    (engine / "apps" / "webui").symlink_to(
        REPOSITORY / "apps" / "webui", target_is_directory=True
    )
    (engine / "biomaker").symlink_to(REPOSITORY / "biomaker", target_is_directory=True)
    config = DesktopConfig(
        tmp_path / "app-data", engine, TOKEN, str(tmp_path / "uv"), sys.executable
    )
    # Keep each test's service environment isolated from later tests.
    for key in (
        "BIOMAKER_WEBUI_METHOD_ROOT",
        "BIOMAKER_WEBUI_WORKSPACE_ROOT",
        "BIOMAKER_WEBUI_STATE_DIR",
        "BIOMAKER_ENV_FILE",
        "BIOMAKER_WEBUI_UV",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON_DOWNLOADS",
        "UV_CACHE_DIR",
    ):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    config.prepare()
    return config


def ready_installer(config):
    engine = EngineInstaller(config)
    engine.interpreter.parent.mkdir(parents=True)
    engine.interpreter.write_text("#!/bin/sh\nexit 0\n")
    engine.interpreter.chmod(0o755)
    engine.receipt.write_text(json.dumps({"fingerprint": engine._fingerprint(), "native_runtime": {"test_native_verified": True}}))
    return EngineInstaller(config)


def wait_finished(engine, seconds=5):
    deadline = time.monotonic() + seconds
    while engine.snapshot()["installing"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not engine.snapshot()["installing"], engine.snapshot()
    return engine.snapshot()


def fake_uv(config, body):
    path = Path(config.uv)
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)


def client_for(app):
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 51234))


def test_configuration_rejects_missing_token_and_relative_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("BIOMAKER_DESKTOP_TOKEN", "short")
    with pytest.raises(ValueError, match="32"):
        DesktopConfig.from_environment()
    monkeypatch.setenv("BIOMAKER_DESKTOP_TOKEN", TOKEN)
    monkeypatch.setenv("BIOMAKER_DESKTOP_HOME", "relative")
    with pytest.raises(ValueError, match="absolute"):
        DesktopConfig.from_environment()


def test_prepare_uses_private_appdata_and_removes_inherited_token(config, monkeypatch):
    monkeypatch.setenv("BIOMAKER_DESKTOP_TOKEN", TOKEN)
    config.prepare()
    assert "BIOMAKER_DESKTOP_TOKEN" not in os.environ
    assert os.environ["BIOMAKER_ENV_FILE"] == str(config.home / "config.env")
    assert os.environ["BIOMAKER_WEBUI_WORKSPACE_ROOT"] == str(config.workspace)


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/api/health",
        "/api/desktop/status",
        "/api/env",
        "/openapi.json",
        "/static/app.js",
        "/artifacts/runs/no-run/tree_plot.html",
    ],
)
def test_every_surface_requires_desktop_token(config, path):
    with client_for(create_desktop_app(config)) as client:
        assert client.get(path).status_code == 401
        assert (
            client.get(path, headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
        response = client.get("/api/desktop/status", headers=AUTH)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"


def test_browser_origins_and_untrusted_hosts_are_rejected(config):
    with client_for(create_desktop_app(config)) as client:
        for origin in ("https://evil.example", "http://localhost:3000", "null"):
            assert (
                client.get(
                    "/api/health", headers={**AUTH, "Origin": origin}
                ).status_code
                == 403
            )
        assert (
            client.get(
                "/api/health", headers={**AUTH, "Host": "evil.example"}
            ).status_code
            == 403
        )
    with client_for(
        create_desktop_app(replace(config, allowed_origins=("app://biomaker",)))
    ) as client:
        assert (
            client.get(
                "/api/health", headers={**AUTH, "Origin": "app://biomaker"}
            ).status_code
            == 200
        )


def test_status_and_saved_settings_do_not_expose_credentials(config):
    config.env_file.write_text(
        'OPENAI_API_KEY="sk-private-test-secret"\nBIOMAKER_MAIN_MODEL="my-model"\n'
    )
    with client_for(create_desktop_app(config)) as client:
        response = client.get("/api/desktop/status", headers=AUTH)
        assert response.json()["configuration"] == {
            "api_key_configured": True,
            "model_configured": True,
        }
        assert "sk-private" not in response.text
        assert TOKEN not in response.text
        assert "sk-private" not in client.get("/api/env", headers=AUTH).text
        assert response.json()["paths"]["workspace"] == str(config.workspace)


def test_run_guard_checks_engine_credentials_and_explicit_model(config, monkeypatch):
    dataset = config.workspace / "data"
    dataset.mkdir()
    payload = {
        "experiment": {"dataset_path": str(dataset), "task_body": "Classify images."}
    }
    with client_for(create_desktop_app(config)) as client:
        response = client.post("/api/runs", json=payload, headers=AUTH)
        assert response.status_code == 409
        assert "Install" in response.text
    installer = ready_installer(config)
    app = create_desktop_app(config, installer=installer)
    monkeypatch.setattr(app.state.manager, "_ensure_supervisor", lambda: None)
    with client_for(app) as client:
        response = client.post("/api/runs", json=payload, headers=AUTH)
        assert response.status_code == 409 and "API key" in response.text
        assert (
            client.put(
                "/api/env", json={"OPENAI_API_KEY": "test-key"}, headers=AUTH
            ).status_code
            == 200
        )
        response = client.post("/api/runs", json=payload, headers=AUTH)
        assert response.status_code == 409 and "Choose a model" in response.text
        payload["options"] = {"model_for_all_agents": "my-model"}
        response = client.post("/api/runs", json=payload, headers=AUTH)
        assert response.status_code == 202, response.text
        record = response.json()
        assert record["status"] == "queued"
        assert record["options"]["model_for_all_agents"] == "my-model"
        assert (
            client.post("/api/desktop/engine/install", headers=AUTH).status_code == 409
        )
        assert (
            client.post(f"/api/runs/{record['id']}/cancel", headers=AUTH).status_code
            == 200
        )
        # Requeue uses the recorded model, even with no saved default.
        assert (
            client.post(f"/api/runs/{record['id']}/requeue", headers=AUTH).status_code
            == 202
        )


def test_failed_installation_reports_redacted_log_and_can_retry(config):
    config.env_file.write_text('OPENAI_API_KEY="custom-private-token"\n')
    fake_uv(
        config,
        "import sys\nprint('Failure custom-private-token sk-secret-secret-secret')\nsys.exit(17)\n",
    )
    engine = EngineInstaller(config)
    try:
        assert engine.start()["installing"]
        state = wait_finished(engine)
        assert state["ready"] is False
        assert "17" in state["error"]
        assert "custom-private-token" not in state["log"]
        assert "sk-secret-secret-secret" not in state["log"]
        assert "[redacted]" in state["log"]
        fake_uv(
            config,
            "import pathlib\np = pathlib.Path('.venv/bin/python')\np.parent.mkdir(parents=True, exist_ok=True)\np.write_text('#!/bin/sh\\nprintf \"Verified test engine\\n\"\\n')\np.chmod(0o755)\n",
        )
        engine.start()
        state = wait_finished(engine)
        assert state["ready"] is True
        assert state["error"] is None
        assert EngineInstaller(config).snapshot()["ready"] is True
        (config.engine_root / "requirements.txt").write_text("changed-dependency\n")
        assert EngineInstaller(config).snapshot()["ready"] is False
    finally:
        engine.close()


def test_older_receipt_without_native_acceptance_requires_setup(config):
    engine = ready_installer(config)
    assert engine.snapshot()["ready"]
    receipt = json.loads(engine.receipt.read_text())
    receipt.pop("native_runtime")
    engine.receipt.write_text(json.dumps(receipt))
    assert not EngineInstaller(config).snapshot()["ready"]


def test_install_cancellation_terminates_real_child_and_preserves_retry(config):
    fake_uv(
        config, "import time\nprint('Download started', flush=True)\ntime.sleep(90)\n"
    )
    engine = EngineInstaller(config)
    try:
        engine.start()
        deadline = time.monotonic() + 3
        while (
            "Download started" not in engine.snapshot()["log"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        process = engine._process
        assert process is not None
        assert engine.start()["installing"]  # Double-click is idempotent.
        engine.cancel()
        state = wait_finished(engine)
        assert state["ready"] is False
        assert state["error"] == "Installation cancelled."
        assert process.poll() is not None
        assert not engine.receipt.exists()
    finally:
        engine.close()


def test_service_lifespan_cancels_install_but_leaves_queue_policy_intact(config):
    fake_uv(config, "import time\nprint('setup started', flush=True)\ntime.sleep(90)\n")
    engine = EngineInstaller(config)
    app = create_desktop_app(config, installer=engine)
    with client_for(app) as client:
        assert (
            client.post("/api/desktop/engine/install", headers=AUTH).status_code == 202
        )
        assert engine.snapshot()["installing"]
    assert engine.snapshot()["installing"] is False
    assert engine.snapshot()["ready"] is False


def test_cancellation_also_stops_environment_verification(config):
    fake_uv(
        config,
        "import pathlib\np = pathlib.Path('.venv/bin/python')\n"
        "p.parent.mkdir(parents=True, exist_ok=True)\n"
        f"p.write_text({('#!' + sys.executable + chr(10) + 'import time; print(\"verification active\", flush=True); time.sleep(90)' + chr(10))!r})\n"
        "p.chmod(0o755)\n",
    )
    engine = EngineInstaller(config)
    try:
        engine.start()
        deadline = time.monotonic() + 5
        while (
            "verification active" not in engine.snapshot()["log"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        process = engine._process
        assert process is not None and process.args[0] == str(engine.interpreter)
        engine.cancel()
        assert wait_finished(engine)["error"] == "Installation cancelled."
        assert process.poll() is not None
        assert not engine.receipt.exists()
    finally:
        engine.close()


def test_real_service_binds_ephemeral_loopback_and_emits_readiness(config):
    environment = os.environ.copy()
    environment.update(
        BIOMAKER_DESKTOP_TOKEN=TOKEN,
        BIOMAKER_DESKTOP_HOME=str(config.home),
        BIOMAKER_DESKTOP_ENGINE_ROOT=str(config.engine_root),
        BIOMAKER_WEBUI_UV=config.uv,
        BIOMAKER_DESKTOP_PARENT_PID=str(os.getpid()),
    )
    process = subprocess.Popen(
        [sys.executable, str(SERVICE / "main.py")],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        import httpx
        import select

        readable, _, _ = select.select([process.stdout], [], [], 10)
        assert readable, "Sidecar did not publish readiness"
        message = json.loads(process.stdout.readline())
        assert message["type"] == "ready"
        assert 0 < message["port"] < 65536
        url = f"http://127.0.0.1:{message['port']}/api/desktop/status"
        assert httpx.get(url, trust_env=False).status_code == 401
        assert httpx.get(url, headers=AUTH, trust_env=False).status_code == 200
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()
    assert process.returncode in (0, -signal.SIGTERM)
