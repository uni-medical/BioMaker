from __future__ import annotations

import io
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import biomaker.run as run_module
from biomaker.utils import queue_client

from biomaker.runtime.run_store import MemoryRunStore
from biomaker.runtime.runner import RunManager
from biomaker.runtime.settings import Settings


class _ClientStore:
    def __init__(self, record):
        self.record = record
        self.closed = False

    def get_record(self, collection, run_id):
        assert collection == "runs"
        assert run_id == self.record["id"]
        return self.record

    def get_queue_control(self):
        return {"paused": False, "reason": None}

    def close(self):
        self.closed = True


class _ClientManager:
    def __init__(self, tmp_path, *, status="queued"):
        self.console_path = tmp_path / "console.log"
        self.console_path.write_text("queued\n", encoding="utf-8")
        self.store = _ClientStore(
            {
                "id": "cli-run",
                "status": status,
                "return_code": None,
                "console_log": str(self.console_path),
            }
        )
        self.started = False
        self.shutdown_called = False
        self.imported = None
        self.enqueue_call = None
        self.cancel_calls = []
        self.stop_calls = []

    def import_history(self, records):
        self.imported = records

    def start(self):
        self.started = True

    def enqueue_command(self, command, **kwargs):
        self.enqueue_call = (command, kwargs)
        return dict(self.store.record)

    def cancel(self, run_id):
        self.cancel_calls.append(run_id)
        self.store.record["status"] = "cancelled"
        return self.store.record

    def stop(self, run_id):
        self.stop_calls.append(run_id)
        self.store.record["status"] = "stopped"
        self.store.record["return_code"] = -signal.SIGTERM
        return self.store.record

    def shutdown(self):
        self.shutdown_called = True


def _raw_cfg(tmp_path):
    return SimpleNamespace(
        log_dir=tmp_path / "logs",
        workspace_dir=tmp_path / "workspaces",
        data_dir=tmp_path / "dataset",
        desc_file=None,
        goal="Train a model",
        exp_name="cli-test",
        model_for_all_agents="test-model",
        agent=SimpleNamespace(steps=4, total_time_limit=100),
        exec=SimpleNamespace(timeout=30, parallel_workers="auto"),
        final_retrain=SimpleNamespace(enabled=True),
    )


def _install_client(monkeypatch, tmp_path, manager):
    cfg = _raw_cfg(tmp_path)
    cfg.data_dir.mkdir()
    settings = SimpleNamespace(
        stop_grace_seconds=0.0,
        logs_root=(tmp_path / "shared" / "logs").resolve(),
        workspaces_root=(tmp_path / "shared" / "workspaces").resolve(),
        ensure_directories=lambda: None,
    )
    monkeypatch.setattr(queue_client, "_load_cfg", lambda **_kwargs: cfg)
    monkeypatch.setattr(
        queue_client,
        "_settings",
        lambda _cfg, _root: (
            lambda _settings: manager,
            lambda _settings: [{"id": "history"}],
            settings,
        ),
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(queue_client.sys, "stdout", stdout)
    monkeypatch.setattr(queue_client.sys, "stderr", stderr)
    return cfg, stdout, stderr


@pytest.mark.parametrize(
    ("record", "interrupted", "expected"),
    [
        ({"status": "succeeded", "return_code": 9}, False, 0),
        ({"status": "failed", "return_code": 7}, False, 7),
        ({"status": "interrupted", "return_code": -signal.SIGKILL}, False, 137),
        ({"status": "failed", "return_code": None}, False, 1),
        ({"status": "cancelled", "return_code": None}, False, 130),
        ({"status": "failed", "return_code": 7}, True, 130),
    ],
)
def test_cli_exit_code_preserves_child_outcome(record, interrupted, expected):
    assert queue_client._exit_code(record, interrupted_by_user=interrupted) == expected


def test_cli_submission_waits_in_fifo_streams_log_and_returns_child_code(
    tmp_path, monkeypatch
):
    manager = _ClientManager(tmp_path)
    cfg, stdout, stderr = _install_client(monkeypatch, tmp_path, manager)
    sleeps = []

    def advance(_delay):
        sleeps.append(_delay)
        if len(sleeps) == 1:
            manager.store.record["status"] = "running"
            with manager.console_path.open("a", encoding="utf-8") as handle:
                handle.write("running\n")
        elif len(sleeps) == 2:
            manager.store.record.update(status="failed", return_code=7)
            with manager.console_path.open("a", encoding="utf-8") as handle:
                handle.write("failed\n")

    monkeypatch.setattr(queue_client.time, "sleep", advance)
    argv = ["data_dir=dataset", "goal=Train a model"]

    code = queue_client.submit_cli_and_wait(argv)

    assert code == 7
    assert len(sleeps) == 2
    command, kwargs = manager.enqueue_call
    assert command == [
        sys.executable,
        "-m",
        "biomaker.run",
        *argv,
        f"log_dir={tmp_path / 'shared' / 'logs'}",
        f"workspace_dir={tmp_path / 'shared' / 'workspaces'}",
    ]
    assert kwargs["cwd"] == Path.cwd().resolve()
    assert kwargs["project_snapshot"] == {
        "dataset_path": str(cfg.data_dir.resolve()),
        "task_template_path": None,
        "task_body": "Train a model",
    }
    assert kwargs["options"]["exec_parallel_workers"] == "auto"
    assert kwargs["requested_exp_name"] == "cli-test"
    assert kwargs["logs_root"] == (tmp_path / "shared" / "logs").resolve()
    assert kwargs["workspaces_root"] == (tmp_path / "shared" / "workspaces").resolve()
    assert stdout.getvalue() == "queued\nrunning\nfailed\n"
    assert "Queued BioMaker run cli-run" in stderr.getvalue()
    assert manager.started
    assert manager.imported == [{"id": "history"}]
    assert manager.shutdown_called
    assert manager.store.closed


def test_cli_and_webui_resolve_the_same_default_queue_domain(tmp_path, monkeypatch):
    shared_workspace = tmp_path / "workspace"
    method_root = shared_workspace / "BioMaker"
    method_root.mkdir(parents=True)
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.delenv("BIOMAKER_WEBUI_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("BIOMAKER_WEBUI_STATE_DIR", raising=False)
    monkeypatch.setenv("BIOMAKER_WEBUI_METHOD_ROOT", str(method_root))
    monkeypatch.setattr(
        queue_client,
        "_load_queue_types",
        lambda _method_root: (RunManager, Settings, lambda _settings: []),
    )
    raw_cfg = SimpleNamespace(log_dir="logs", workspace_dir="workspaces")

    _, _, cli_settings = queue_client._settings(raw_cfg, method_root)
    webui_settings = Settings.default()

    assert cli_settings.workspace_root == webui_settings.workspace_root
    assert cli_settings.state_dir == webui_settings.state_dir
    assert cli_settings.logs_root == webui_settings.logs_root
    assert cli_settings.workspaces_root == webui_settings.workspaces_root
    assert cli_settings.state_dir == (shared_workspace / ".biomaker").resolve()
    assert cli_settings.logs_root == (shared_workspace / "logs").resolve()
    assert cli_settings.workspaces_root == (shared_workspace / "workspaces").resolve()


def test_cli_queue_roots_honor_environment_and_absolute_artifact_paths(
    tmp_path, monkeypatch
):
    method_root = tmp_path / "checkout" / "BioMaker"
    method_root.mkdir(parents=True)
    workspace_override = tmp_path / "shared-domain"
    state_override = tmp_path / "explicit-state"
    absolute_logs = tmp_path / "external-logs"
    absolute_workspaces = tmp_path / "external-workspaces"
    monkeypatch.setenv("BIOMAKER_WEBUI_WORKSPACE_ROOT", str(workspace_override))
    monkeypatch.setenv("BIOMAKER_WEBUI_STATE_DIR", str(state_override))
    monkeypatch.setattr(
        queue_client,
        "_load_queue_types",
        lambda _method_root: (RunManager, Settings, lambda _settings: []),
    )
    raw_cfg = SimpleNamespace(
        log_dir=absolute_logs,
        workspace_dir=absolute_workspaces,
    )

    _, _, settings = queue_client._settings(raw_cfg, method_root)

    assert settings.workspace_root == workspace_override.resolve()
    assert settings.state_dir == state_override.resolve()
    assert settings.logs_root == absolute_logs.resolve()
    assert settings.workspaces_root == absolute_workspaces.resolve()


@pytest.mark.parametrize(
    "failure_stage", ["import_history", "start", "enqueue_command"]
)
def test_cli_closes_manager_when_queue_setup_or_enqueue_fails(
    failure_stage, tmp_path, monkeypatch
):
    manager = _ClientManager(tmp_path)
    _install_client(monkeypatch, tmp_path, manager)

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"failed during {failure_stage}")

    monkeypatch.setattr(manager, failure_stage, fail)

    with pytest.raises(RuntimeError, match=failure_stage):
        queue_client.submit_cli_and_wait([])

    assert manager.shutdown_called
    assert manager.store.closed


@pytest.mark.parametrize(
    ("data_dir", "goal", "expected"),
    [
        (None, "Train a model", "Provide data_dir"),
        ("missing", "Train a model", "Dataset directory does not exist"),
        ("dataset", None, "Provide desc_file or goal"),
    ],
)
def test_invalid_cli_request_is_rejected_before_queue_start(
    data_dir, goal, expected, tmp_path, monkeypatch
):
    cfg = _raw_cfg(tmp_path)
    cfg.data_dir = None if data_dir is None else tmp_path / data_dir
    cfg.goal = goal
    if data_dir == "dataset":
        cfg.data_dir.mkdir()
    monkeypatch.setattr(queue_client, "_load_cfg", lambda **_kwargs: cfg)
    monkeypatch.setattr(
        queue_client,
        "_settings",
        lambda *_args: pytest.fail("invalid requests must not open the queue"),
    )

    with pytest.raises(ValueError, match=expected):
        queue_client.submit_cli_and_wait([])


@pytest.mark.parametrize("initial_status", ["queued", "running"])
def test_ctrl_c_cancels_queued_or_stops_running_run(
    initial_status, tmp_path, monkeypatch
):
    manager = _ClientManager(tmp_path, status=initial_status)
    _install_client(monkeypatch, tmp_path, manager)
    monkeypatch.setattr(
        queue_client.time,
        "sleep",
        lambda _delay: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    code = queue_client.submit_cli_and_wait([])

    assert code == 130
    assert manager.cancel_calls == (["cli-run"] if initial_status == "queued" else [])
    assert manager.stop_calls == (["cli-run"] if initial_status == "running" else [])
    assert manager.store.record["status"] in {"cancelled", "stopped"}
    assert manager.shutdown_called
    assert manager.store.closed


def test_ctrl_c_retries_when_queued_run_is_claimed_during_cancel(tmp_path, monkeypatch):
    manager = _ClientManager(tmp_path, status="queued")
    _install_client(monkeypatch, tmp_path, manager)
    original_cancel = manager.cancel

    def racing_cancel(run_id):
        original_cancel(run_id)
        manager.store.record["status"] = "running"
        raise ValueError("run was claimed")

    manager.cancel = racing_cancel
    monkeypatch.setattr(
        queue_client.time,
        "sleep",
        lambda _delay: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    code = queue_client.submit_cli_and_wait([])

    assert code == 130
    assert manager.cancel_calls == ["cli-run"]
    assert manager.stop_calls == ["cli-run"]
    assert manager.store.record["status"] == "stopped"


@pytest.mark.parametrize("queue_flag", [[], ["--queue"]])
def test_queue_direct_child_bypasses_submission_and_prevents_recursion(
    monkeypatch, queue_flag,
):
    calls = []
    monkeypatch.setenv("BIOMAKER_QUEUE_DIRECT", "1")
    monkeypatch.setattr(run_module.sys, "argv", ["biomaker", *queue_flag])
    monkeypatch.setattr(
        queue_client,
        "submit_cli_and_wait",
        lambda: pytest.fail("supervised child must not re-enter the queue"),
    )
    monkeypatch.setattr(run_module, "_run_direct_entry", lambda: calls.append("direct"))

    run_module.run()

    assert calls == ["direct"]
    assert "--queue" not in run_module.sys.argv


@pytest.mark.parametrize("entrypoint", ["biomaker", "/checkout/biomaker/run.py"])
def test_cli_defaults_to_direct_without_opening_queue(tmp_path, monkeypatch, entrypoint):
    monkeypatch.delenv("BIOMAKER_QUEUE_DIRECT", raising=False)
    monkeypatch.delenv("BIOMAKER_DISABLE_RUN_QUEUE", raising=False)
    argv = [entrypoint, "goal=detect fractures", "agent.steps=3"]
    monkeypatch.setattr(run_module.sys, "argv", argv)
    # Direct mode must also work without an installed queue sidecar.
    monkeypatch.setattr(run_module, "__file__", str(tmp_path / "biomaker" / "run.py"))
    monkeypatch.setattr(
        queue_client, "submit_cli_and_wait",
        lambda: pytest.fail("default CLI must not access the queue"),
    )
    calls = []
    monkeypatch.setattr(
        run_module, "_run_direct_entry", lambda: calls.append(list(run_module.sys.argv))
    )

    run_module.run()

    assert calls == [argv]


@pytest.mark.parametrize("entrypoint", ["biomaker", "/checkout/biomaker/run.py"])
@pytest.mark.parametrize("flag_first", [True, False])
def test_explicit_queue_returns_exit_code_and_consumes_flag(monkeypatch, entrypoint, flag_first):
    monkeypatch.delenv("BIOMAKER_QUEUE_DIRECT", raising=False)
    monkeypatch.delenv("BIOMAKER_DISABLE_RUN_QUEUE", raising=False)
    overrides = ["goal=detect fractures", "agent.steps=3"]
    args = ["--queue", *overrides] if flag_first else [*overrides, "--queue"]
    monkeypatch.setattr(run_module.sys, "argv", [entrypoint, *args])

    def submit():
        assert run_module.sys.argv == [entrypoint, *overrides]
        parsed = queue_client._load_cfg(use_cli_args=True)
        assert parsed.goal == "detect fractures"
        assert parsed.agent.steps == 3
        return 23

    monkeypatch.setattr(queue_client, "submit_cli_and_wait", submit)
    monkeypatch.setattr(
        run_module,
        "_run_direct_entry",
        lambda: pytest.fail("--queue should submit to the queue"),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_module.run()

    assert exc_info.value.code == 23


def test_explicit_queue_failure_does_not_fall_back_to_direct(monkeypatch):
    monkeypatch.delenv("BIOMAKER_QUEUE_DIRECT", raising=False)
    monkeypatch.delenv("BIOMAKER_DISABLE_RUN_QUEUE", raising=False)
    monkeypatch.setattr(run_module.sys, "argv", ["biomaker", "--queue"])
    def unavailable():
        raise RuntimeError("durable run queue is unavailable")

    monkeypatch.setattr(queue_client, "submit_cli_and_wait", unavailable)
    monkeypatch.setattr(
        run_module,
        "_run_direct_entry",
        lambda: pytest.fail("missing queue support must not silently run directly"),
    )

    with pytest.raises(RuntimeError, match="durable run queue is unavailable"):
        run_module.run()


@pytest.mark.parametrize("queue_flag", [[], ["--queue"]])
def test_explicit_queue_opt_out_still_runs_directly(monkeypatch, queue_flag):
    calls = []
    monkeypatch.delenv("BIOMAKER_QUEUE_DIRECT", raising=False)
    monkeypatch.setenv("BIOMAKER_DISABLE_RUN_QUEUE", "1")
    monkeypatch.setattr(run_module.sys, "argv", ["biomaker", *queue_flag])
    monkeypatch.setattr(run_module, "_run_direct_entry", lambda: calls.append("direct"))

    run_module.run()

    assert calls == ["direct"]
    assert "--queue" not in run_module.sys.argv


class _ImmediateQueuedProcess:
    def __init__(self, factory, command, kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = factory.next_pid
        factory.next_pid += 1
        self.returncode = factory.return_code
        self.stdout = None
        launch_token = kwargs["env"]["BIOMAKER_LAUNCH_TOKEN"]
        log_dir = factory.settings.logs_root / f"cli-{launch_token[:8]}"
        log_dir.mkdir()
        (log_dir / "run_manifest.json").write_text(
            json.dumps({"launch_token": launch_token}), encoding="utf-8"
        )
        (log_dir / "events.jsonl").write_text(
            json.dumps(
                {"event": ("run_finished" if self.returncode == 0 else "run_failed")}
            )
            + "\n",
            encoding="utf-8",
        )
        kwargs["stdout"].write("queued command child output\n")
        kwargs["stdout"].flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.returncode = -signal.SIGKILL


class _QueuedProcessFactory:
    def __init__(self, settings, return_code):
        self.settings = settings
        self.return_code = return_code
        self.next_pid = 910_000
        self.calls = []

    def __call__(self, command, **kwargs):
        process = _ImmediateQueuedProcess(self, command, kwargs)
        self.calls.append(process)
        return process


def _wait_for_terminal(store, run_id, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.get_record("runs", run_id)
        if record and record.get("status") in {
            "succeeded",
            "failed",
            "cancelled",
            "stopped",
            "interrupted",
        }:
            return record
        time.sleep(0.01)
    raise AssertionError(store.get_record("runs", run_id))


def test_enqueue_command_runs_through_fifo_with_direct_child_environment(tmp_path):
    method_root = tmp_path / "BioMaker"
    method_root.mkdir()
    settings = SimpleNamespace(
        method_root=method_root,
        workspace_root=tmp_path,
        state_dir=tmp_path / ".biomaker",
        logs_root=tmp_path / "logs",
        workspaces_root=tmp_path / "workspaces",
        experiments_root=tmp_path / "experiments",
        datasets_root=tmp_path / "datasets",
        env_file=tmp_path / ".env",
        uv_executable="uv",
        stop_grace_seconds=0.01,
        ensure_directories=lambda: _ensure_queue_directories(tmp_path),
    )
    settings.ensure_directories()
    store = MemoryRunStore()
    factory = _QueuedProcessFactory(settings, return_code=7)
    manager = RunManager(settings, store, process_factory=factory)
    command = [sys.executable, "-m", "biomaker.run", "goal=test"]

    queued = manager.enqueue_command(
        command,
        cwd=method_root,
        project_snapshot={"task_body": "test"},
        options={"exec_parallel_workers": "auto"},
        requested_exp_name="cli-run",
    )
    saved = _wait_for_terminal(store, queued["id"])

    assert saved["status"] == "failed"
    assert saved["return_code"] == 7
    assert saved["command"] == command
    assert saved["cwd"] == str(method_root.resolve())
    assert saved["project_snapshot"] == {"task_body": "test"}
    assert saved["options"] == {"exec_parallel_workers": "auto"}
    assert saved["logs_root"] == str(settings.logs_root.resolve())
    assert saved["workspaces_root"] == str(settings.workspaces_root.resolve())
    assert len(factory.calls) == 1
    child = factory.calls[0]
    assert child.command == command
    assert child.kwargs["env"]["BIOMAKER_QUEUE_DIRECT"] == "1"
    assert child.kwargs["env"]["BIOMAKER_LAUNCH_TOKEN"] == queued["id"]
    if os.name == "posix":
        assert child.kwargs["start_new_session"] is True
    assert "queued command child output" in Path(saved["console_log"]).read_text(
        encoding="utf-8"
    )
    manager.shutdown()
    store.close()


def _ensure_queue_directories(root):
    for path in (
        root / ".biomaker",
        root / ".biomaker" / "tasks",
        root / ".biomaker" / "console",
        root / "logs",
        root / "workspaces",
        root / "experiments",
        root / "datasets",
    ):
        path.mkdir(parents=True, exist_ok=True)
