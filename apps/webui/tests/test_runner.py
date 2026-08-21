from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest
import webui.runner as runner_module
from biomaker.utils.process_identity import ProcessIdentity, capture_process_identity
from webui.models import AdvancedOptions
from webui.run_store import MemoryRunStore, SQLiteRunStore
from webui.runner import RunManager, build_command


class ControlledProcess:
    def __init__(
        self,
        factory,
        command,
        kwargs,
        *,
        return_code=None,
        terminal_event=None,
    ):
        self.factory = factory
        self.command = command
        self.kwargs = kwargs
        self.pid = factory.next_pid
        factory.next_pid += 1
        self.launch_token = kwargs["env"]["BIOMAKER_LAUNCH_TOKEN"]
        self.returncode = return_code
        self.done = threading.Event()
        self.terminate_calls = 0
        self.kill_calls = 0
        self.stdout = None

        index = len(factory.processes) + 1
        self.log_dir = factory.settings.logs_root / f"{index}-{self.launch_token[:8]}"
        self.log_dir.mkdir()
        (self.log_dir / "run_manifest.json").write_text(
            json.dumps({"launch_token": self.launch_token}), encoding="utf-8"
        )
        self.events_path = self.log_dir / "events.jsonl"
        self._append_event("run_started")
        kwargs["stdout"].write("BioMaker test output\n")
        kwargs["stdout"].flush()
        if terminal_event:
            self._append_event(terminal_event)
        if return_code is not None:
            self.done.set()

    def _append_event(self, event, **details):
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **details}) + "\n")

    def finish(self, return_code=0, terminal_event="run_finished"):
        if terminal_event:
            details = (
                {"error": "test failure"} if terminal_event == "run_failed" else {}
            )
            self._append_event(terminal_event, **details)
        self.returncode = return_code
        self.done.set()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.done.wait(timeout or 3)
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self._append_event("run_interrupted", reason="stopped")
        self.returncode = -signal.SIGTERM
        self.done.set()

    def kill(self):
        self.kill_calls += 1
        self.returncode = -signal.SIGKILL
        self.done.set()


class ProcessFactory:
    def __init__(self, settings, outcomes=None):
        self.settings = settings
        self.outcomes = list(outcomes or [])
        self.processes = []
        self.by_pid = {}
        self.next_pid = 700_000
        self._lock = threading.Lock()

    def __call__(self, command, **kwargs):
        with self._lock:
            outcome = self.outcomes.pop(0) if self.outcomes else (None, None)
            process = ControlledProcess(
                self,
                command,
                kwargs,
                return_code=outcome[0],
                terminal_event=outcome[1],
            )
            self.processes.append(process)
            self.by_pid[process.pid] = process
            return process

    def verify(self, pid, _started_at, launch_token):
        process = self.by_pid.get(pid)
        return bool(
            process and process.launch_token == launch_token and process.poll() is None
        )


@pytest.mark.skipif(os.name != "posix", reason="session cleanup is POSIX-only")
def test_uncaptured_detached_process_drains_the_full_session_before_reaping(
    monkeypatch,
):
    observations = iter(
        [
            {410_000, 410_001},
            {410_002},  # A child-created PGID discovered on the next pass.
            set(),
            set(),
        ]
    )
    inspected = []
    signalled = []
    waited_after_inspections = []

    class DetachedProcess:
        pid = 410_000

        def wait(self, timeout=None):
            waited_after_inspections.append((len(inspected), timeout))
            return -signal.SIGKILL

    def session_groups(sid):
        inspected.append(sid)
        return next(observations)

    monkeypatch.setattr(runner_module, "read_session_process_groups", session_groups)
    monkeypatch.setattr(runner_module.os, "getpgrp", lambda: 999_999)
    monkeypatch.setattr(
        runner_module.os,
        "killpg",
        lambda pgid, sent_signal: signalled.append((pgid, sent_signal)),
    )
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)

    RunManager._drain_uncaptured_detached_process(DetachedProcess())

    assert inspected == [410_000, 410_000, 410_000, 410_000]
    assert signalled == [
        (410_001, signal.SIGKILL),
        (410_000, signal.SIGKILL),
        (410_002, signal.SIGKILL),
    ]
    assert len(waited_after_inspections) == 1
    assert waited_after_inspections[0][0] == 4


@pytest.mark.skipif(os.name != "posix", reason="session cleanup is POSIX-only")
def test_uncaptured_detached_process_does_not_reap_when_session_is_unreadable(
    monkeypatch,
):
    waited = []
    clock = iter([0.0, 0.0, 5.1])

    class DetachedProcess:
        pid = 420_000

        def wait(self, timeout=None):
            waited.append(timeout)
            return -signal.SIGKILL

    monkeypatch.setattr(
        runner_module,
        "read_session_process_groups",
        lambda _sid: (_ for _ in ()).throw(
            runner_module.ProcessIdentityError("inspection denied")
        ),
    )
    monkeypatch.setattr(runner_module.os, "getpgrp", lambda: 999_999)
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="Could not confirm.*session exited"):
        RunManager._drain_uncaptured_detached_process(DetachedProcess())

    assert waited == []


def wait_for_status(store, run_id, expected, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.get_record("runs", run_id)
        if record and record.get("status") in expected:
            return record
        time.sleep(0.02)
    current = store.get_record("runs", run_id)
    raise AssertionError(f"Run {run_id} did not reach {expected}; current={current}")


def wait_for_processes(factory, count, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(factory.processes) >= count:
            return factory.processes
        time.sleep(0.02)
    raise AssertionError(f"Expected {count} processes; found {len(factory.processes)}")


def wait_for_runner_exit(store, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if store.get_queue_control().get("runner_token") is None:
            return
        time.sleep(0.02)
    raise AssertionError(f"Runner did not exit: {store.get_queue_control()}")


def assert_safe_signal_target(*, pid=None, pgid=None):
    """Keep test cleanup from ever signalling pytest or its process group."""

    if pid is not None:
        assert isinstance(pid, int) and not isinstance(pid, bool) and pid > 1
        assert pid != os.getpid()
    if pgid is not None:
        assert isinstance(pgid, int) and not isinstance(pgid, bool) and pgid > 1
        assert pgid != os.getpgrp()


def experiment_record(dataset_path):
    return {
        "dataset_path": str(dataset_path),
        "task_template_path": None,
        "task_body": "Train a model and write predictions.",
        "advanced_defaults": AdvancedOptions().model_dump(),
    }


def test_build_command_is_argument_list_with_supported_overrides(settings, tmp_path):
    command = build_command(
        settings,
        dataset_path="/data/path with spaces",
        task_file=tmp_path / "task.md",
        requested_exp_name="test-run",
        options={
            **AdvancedOptions().model_dump(),
            "model_for_all_agents": "openai/gpt-test",
            "agent_steps": 7,
            "final_retrain_enabled": False,
        },
    )

    assert command[:3] == ["uv", "run", "biomaker"]
    assert "data_dir=/data/path with spaces" in command
    assert "agent.steps=7" in command
    assert "final_retrain.enabled=false" in command
    assert "model_for_all_agents=openai/gpt-test" in command


def test_default_manager_uses_state_dir_sqlite(settings):
    manager = RunManager(settings)
    try:
        assert manager.store.database_path == str(
            (settings.state_dir / "runs.sqlite3").resolve()
        )
    finally:
        manager.store.close()


def test_file_store_runs_a_real_detached_temporary_supervisor(settings):
    database = settings.state_dir / "runs.sqlite3"
    store = SQLiteRunStore(database)
    log_dir = settings.logs_root / "detached-run"
    console_log = settings.state_dir / "console" / "detached.log"
    script = (
        "import json, os; from pathlib import Path; "
        f"root=Path({str(log_dir)!r}); root.mkdir(parents=True); "
        "token=os.environ['BIOMAKER_LAUNCH_TOKEN']; "
        "(root/'run_manifest.json').write_text(json.dumps({'launch_token':token})); "
        "(root/'events.jsonl').write_text(json.dumps({'event':'run_finished'})+'\\n'); "
        "print('detached child output')"
    )
    record = {
        "id": "detached-run",
        "managed": True,
        "status": "queued",
        "command": [sys.executable, "-c", script],
        "console_log": str(console_log),
        "launch_token": "detached-run",
        "pid": None,
        "pgid": None,
        "process_started_at": None,
        "return_code": None,
        "exit_signal": None,
        "timestamps": {
            "created_at": datetime.now(UTC).isoformat(),
            "queued_at": datetime.now(UTC).isoformat(),
        },
    }
    store.add_record("runs", record)
    manager = RunManager(settings, store)

    manager.start()
    saved = wait_for_status(store, record["id"], {"succeeded"}, timeout=10)
    wait_for_runner_exit(store, timeout=10)

    assert saved["pid"] != os.getpid()
    assert saved["return_code"] == 0
    assert "detached child output" in console_log.read_text(encoding="utf-8")
    store.close()


def test_pre_manifest_exec_failure_is_failed_not_interrupted(settings):
    settings.ensure_directories()
    store = SQLiteRunStore(settings.state_dir / "runs.sqlite3")
    manager = RunManager(settings, store)

    run = manager.enqueue_command(
        [str(settings.method_root / "definitely-missing-command")],
        cwd=settings.method_root,
    )
    saved = wait_for_status(store, run["id"], {"failed"}, timeout=10)
    wait_for_runner_exit(store, timeout=10)

    assert saved["return_code"] == 127
    assert saved["launch_state"] == "failed"
    assert "Run exec failed" in saved["launch_error"]
    assert saved["log_dir"] is None
    assert "Run exec failed" in Path(saved["console_log"]).read_text(encoding="utf-8")
    store.close()


@pytest.mark.skipif(os.name != "posix", reason="bootstrap recovery is POSIX-only")
def test_bootstrap_self_registers_when_launching_manager_never_attaches(settings):
    settings.ensure_directories()
    database = settings.state_dir / "runs.sqlite3"
    first_store = SQLiteRunStore(database)
    first_manager = RunManager(settings, first_store)
    run_id = "bootstrap-orphan"
    launch_token = "bootstrap-orphan-token"
    release = settings.state_dir / "release-bootstrap"
    launch_counter = settings.state_dir / "bootstrap-launches.txt"
    log_dir = settings.logs_root / "bootstrap-orphan"
    console_path = settings.state_dir / "console" / f"{run_id}.log"
    script = """
import json
import os
import sys
import time
from pathlib import Path

log_dir = Path(sys.argv[1])
release = Path(sys.argv[2])
counter = Path(sys.argv[3])
log_dir.mkdir(parents=True, exist_ok=True)
token = os.environ["BIOMAKER_LAUNCH_TOKEN"]
with counter.open("a", encoding="utf-8") as handle:
    handle.write(f"{os.getpid()} {token}\\n")
(log_dir / "run_manifest.json").write_text(
    json.dumps({"launch_token": token}), encoding="utf-8"
)
(log_dir / "events.jsonl").write_text(
    json.dumps({"event": "run_started"}) + "\\n", encoding="utf-8"
)
while not release.exists():
    print("bootstrap child alive", flush=True)
    time.sleep(0.05)
with (log_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"event": "run_finished"}) + "\\n")
"""
    record = first_store.add_record(
        "runs",
        {
            "id": run_id,
            "managed": True,
            "status": "running",
            "command": [
                sys.executable,
                "-u",
                "-c",
                script,
                str(log_dir),
                str(release),
                str(launch_counter),
            ],
            "cwd": str(settings.method_root),
            "console_log": str(console_path),
            "launch_token": launch_token,
            "launch_state": "pending",
            "launch_error": None,
            "supervisor_token": "abandoned-supervisor",
            "pid": None,
            "pgid": None,
            "process_started_at": None,
            "log_dir": None,
            "return_code": None,
            "exit_signal": None,
            "timestamps": {
                "created_at": datetime.now(UTC).isoformat(),
                "started_at": datetime.now(UTC).isoformat(),
            },
        },
    )
    command, cwd, environment = first_manager._run_process_spec(record)
    console_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen | None = None
    second_store: SQLiteRunStore | None = None
    replacement_identity: ProcessIdentity | None = None
    try:
        with console_path.open("a", encoding="utf-8") as console:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=console,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                start_new_session=True,
            )

        # Deliberately do not call _await_bootstrap_identity: this is the exact
        # old Popen-to-ledger crash window. The child must register itself.
        deadline = time.monotonic() + 10
        registered = None
        while time.monotonic() < deadline:
            candidate = first_store.get_record("runs", run_id)
            if candidate and candidate.get("launch_state") == "registered":
                registered = candidate
                break
            time.sleep(0.02)
        assert registered is not None
        assert registered["pid"] == process.pid
        assert registered["pgid"] == process.pid

        second_store = SQLiteRunStore(database)
        second_manager = RunManager(settings, second_store)
        second_manager.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            control = second_store.get_queue_control()
            runner_pid = control.get("runner_pid")
            runner_token = control.get("runner_token")
            if (
                isinstance(runner_pid, int)
                and runner_pid != os.getpid()
                and runner_token
            ):
                candidate_identity = ProcessIdentity(
                    runner_pid,
                    int(control["runner_pgid"]),
                    str(control["runner_process_started_at"]),
                    str(runner_token),
                )
                if (
                    candidate_identity.pgid != os.getpgrp()
                    and runner_module.verify_os_process_identity(
                        candidate_identity,
                        launch_token=candidate_identity.launch_token,
                    ).matched
                ):
                    replacement_identity = candidate_identity
                    break
            time.sleep(0.02)
        assert replacement_identity is not None
        time.sleep(0.2)
        assert first_store.get_record("runs", run_id)["pid"] == process.pid
        assert len(launch_counter.read_text(encoding="utf-8").splitlines()) == 1

        release.touch()
        assert process.wait(timeout=5) == 0
        saved = wait_for_status(second_store, run_id, {"succeeded"}, timeout=10)
        wait_for_runner_exit(second_store, timeout=10)
        assert saved["pid"] == process.pid
        assert len(launch_counter.read_text(encoding="utf-8").splitlines()) == 1
        assert "bootstrap child alive" in console_path.read_text(encoding="utf-8")
    finally:
        release.touch()
        if process is not None and process.poll() is None:
            assert_safe_signal_target(pid=process.pid, pgid=process.pid)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
        if (
            replacement_identity is not None
            and runner_module.verify_os_process_identity(
                replacement_identity,
                launch_token=replacement_identity.launch_token,
            ).matched
        ):
            assert_safe_signal_target(
                pid=replacement_identity.pid,
                pgid=replacement_identity.pgid,
            )
            try:
                os.kill(replacement_identity.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if second_store is not None:
            second_store.close()
        first_store.close()


@pytest.mark.skipif(os.name != "posix", reason="detached recovery is POSIX-only")
def test_hard_killed_supervisor_preserves_child_and_recovery_does_not_relaunch(
    settings,
):
    settings.ensure_directories()
    database = settings.state_dir / "runs.sqlite3"
    first_store = SQLiteRunStore(database)
    first_manager = RunManager(settings, first_store)
    launch_counter = settings.state_dir / "launches.txt"
    release = settings.state_dir / "release-hard-kill-run"
    log_dir = settings.logs_root / "hard-kill-run"
    script = """
import json
import os
import sys
import time
from pathlib import Path

log_dir = Path(sys.argv[1])
counter = Path(sys.argv[2])
release = Path(sys.argv[3])
log_dir.mkdir(parents=True, exist_ok=True)
token = os.environ["BIOMAKER_LAUNCH_TOKEN"]
with counter.open("a", encoding="utf-8") as handle:
    handle.write(f"{os.getpid()} {token}\\n")
(log_dir / "run_manifest.json").write_text(
    json.dumps({"launch_token": token}), encoding="utf-8"
)
(log_dir / "events.jsonl").write_text(
    json.dumps({"event": "run_started"}) + "\\n", encoding="utf-8"
)
index = 0
while not release.exists():
    print(f"durable tick {index}", flush=True)
    index += 1
    time.sleep(0.05)
with (log_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"event": "run_finished"}) + "\\n")
"""
    child_pid: int | None = None
    child_identity: ProcessIdentity | None = None
    supervisor_identities: list[ProcessIdentity] = []
    try:
        run = first_manager.enqueue_command(
            [
                sys.executable,
                "-u",
                "-c",
                script,
                str(log_dir),
                str(launch_counter),
                str(release),
            ],
            cwd=settings.method_root,
        )
        deadline = time.monotonic() + 10
        running = None
        while time.monotonic() < deadline:
            candidate = first_store.get_record("runs", run["id"])
            if candidate and candidate.get("pid") and candidate.get("log_dir"):
                running = candidate
                break
            time.sleep(0.02)
        assert running is not None
        child_pid = int(running["pid"])
        child_identity = ProcessIdentity(
            child_pid,
            int(running["pgid"]),
            str(running["process_started_at"]),
            str(running["launch_token"]),
        )

        deadline = time.monotonic() + 5
        supervisor_identity = None
        while time.monotonic() < deadline:
            control = first_store.get_queue_control()
            runner_pid = control.get("runner_pid")
            runner_token = control.get("runner_token")
            if (
                isinstance(runner_pid, int)
                and runner_pid != os.getpid()
                and runner_token
            ):
                candidate_identity = ProcessIdentity(
                    runner_pid,
                    int(control["runner_pgid"]),
                    str(control["runner_process_started_at"]),
                    str(runner_token),
                )
                if (
                    candidate_identity.pgid != os.getpgrp()
                    and runner_module.verify_os_process_identity(
                        candidate_identity,
                        launch_token=candidate_identity.launch_token,
                    ).matched
                ):
                    supervisor_identity = candidate_identity
                    break
            time.sleep(0.02)
        assert supervisor_identity is not None
        supervisor_pid = supervisor_identity.pid
        supervisor_identities.append(supervisor_identity)
        console_path = Path(str(running["console_log"]))
        before_kill_size = console_path.stat().st_size

        assert_safe_signal_target(
            pid=supervisor_identity.pid,
            pgid=supervisor_identity.pgid,
        )
        os.kill(supervisor_pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            verification = runner_module.verify_os_process_identity(
                supervisor_identity,
                launch_token=supervisor_identity.launch_token,
            )
            if verification.reason == "missing":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("queue supervisor was not reaped after SIGKILL")

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if console_path.stat().st_size > before_kill_size:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("run console stopped growing with the supervisor")
        assert runner_module.verify_os_process_identity(
            child_identity,
            launch_token=child_identity.launch_token,
        ).matched
        assert first_store.get_record("runs", run["id"])["pid"] == child_pid

        deadline = time.monotonic() + 10
        replacement_identity = None
        while time.monotonic() < deadline:
            replacement = first_store.get_queue_control()
            replacement_pid = replacement.get("runner_pid")
            replacement_token = replacement.get("runner_token")
            if (
                isinstance(replacement_pid, int)
                and replacement_pid != supervisor_pid
                and replacement_pid != os.getpid()
                and replacement_token
            ):
                candidate_identity = ProcessIdentity(
                    replacement_pid,
                    int(replacement["runner_pgid"]),
                    str(replacement["runner_process_started_at"]),
                    str(replacement_token),
                )
                if (
                    candidate_identity.pgid != os.getpgrp()
                    and runner_module.verify_os_process_identity(
                        candidate_identity,
                        launch_token=candidate_identity.launch_token,
                    ).matched
                ):
                    replacement_identity = candidate_identity
                    supervisor_identities.append(replacement_identity)
                    break
            time.sleep(0.02)
        assert replacement_identity is not None
        assert runner_module.verify_os_process_identity(
            replacement_identity,
            launch_token=replacement_identity.launch_token,
        ).matched
        assert first_store.get_record("runs", run["id"])["pid"] == child_pid

        release.touch()
        saved = wait_for_status(first_store, run["id"], {"succeeded"}, timeout=10)
        wait_for_runner_exit(first_store, timeout=10)
        assert saved["pid"] == child_pid
        assert len(launch_counter.read_text(encoding="utf-8").splitlines()) == 1
        terminal_events = [
            json.loads(line)
            for line in (log_dir / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if json.loads(line).get("event")
            in {"run_finished", "run_failed", "run_interrupted"}
        ]
        assert [event["event"] for event in terminal_events] == ["run_finished"]
    finally:
        release.touch()
        for supervisor_identity in supervisor_identities:
            if runner_module.verify_os_process_identity(
                supervisor_identity,
                launch_token=supervisor_identity.launch_token,
            ).matched:
                assert_safe_signal_target(
                    pid=supervisor_identity.pid,
                    pgid=supervisor_identity.pgid,
                )
                try:
                    os.kill(supervisor_identity.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if (
            child_identity is not None
            and runner_module.verify_os_process_identity(
                child_identity,
                launch_token=child_identity.launch_token,
            ).matched
        ):
            assert_safe_signal_target(
                pid=child_identity.pid,
                pgid=child_identity.pgid,
            )
            try:
                os.killpg(child_identity.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        first_store.close()


def test_output_discovery_uses_roots_persisted_on_the_run(settings, tmp_path):
    custom_logs = tmp_path / "custom-logs"
    custom_workspaces = tmp_path / "custom-workspaces"
    log_dir = custom_logs / "1-custom-root"
    log_dir.mkdir(parents=True)
    (log_dir / "run_manifest.json").write_text(
        json.dumps({"launch_token": "custom-token"}), encoding="utf-8"
    )
    store = MemoryRunStore()
    store.add_record(
        "runs",
        {
            "id": "custom-root-run",
            "managed": True,
            "status": "running",
            "launch_token": "custom-token",
            "logs_root": str(custom_logs),
            "workspaces_root": str(custom_workspaces),
            "log_dir": None,
            "timestamps": {"started_at": datetime.now(UTC).isoformat()},
        },
    )
    store.add_record(
        "runs",
        {
            "id": "imported-duplicate",
            "managed": False,
            "status": "running",
            "log_dir": str(log_dir),
            "timestamps": {},
        },
    )
    manager = RunManager(settings, store, process_factory=ProcessFactory(settings))

    assert manager._discover_output_dir("custom-root-run") is True

    saved = store.get_record("runs", "custom-root-run")
    assert saved["log_dir"] == str(log_dir.resolve())
    assert saved["workspace_dir"] == str(
        (custom_workspaces / "1-custom-root").resolve()
    )
    assert [record["id"] for record in store.list_records("runs")] == [
        "custom-root-run"
    ]


def test_child_stdio_is_direct_file_and_success_requires_terminal_event(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(settings, outcomes=[(0, "run_finished")])
    manager = RunManager(settings, store, process_factory=factory)

    run = manager.enqueue_experiment(experiment_record(dataset))
    saved = wait_for_status(store, run["id"], {"succeeded"})

    process = factory.processes[0]
    assert process.kwargs["stdout"] is not subprocess.PIPE
    assert process.kwargs["stderr"] is subprocess.STDOUT
    assert process.kwargs["start_new_session"] is True
    assert saved["return_code"] == 0
    assert saved["exit_signal"] is None
    assert saved["pid"] == process.pid
    assert saved["pgid"] == process.pid
    assert saved["process_started_at"]
    assert saved["launch_token"] == run["id"]
    assert "BioMaker test output" in Path(saved["console_log"]).read_text(
        encoding="utf-8"
    )
    wait_for_runner_exit(store)


def test_zero_exit_without_terminal_event_is_interrupted_and_pauses(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(settings, outcomes=[(0, None)])
    manager = RunManager(settings, store, process_factory=factory)

    run = manager.enqueue_experiment(experiment_record(dataset))
    saved = wait_for_status(store, run["id"], {"interrupted"})

    assert saved["return_code"] == 0
    assert store.get_queue_control()["paused"] is True
    events = [
        json.loads(line)
        for line in factory.processes[0].events_path.read_text().splitlines()
    ]
    terminal = [event for event in events if event["event"].startswith("run_")][1:]
    assert terminal == [
        {
            "ts": terminal[0]["ts"],
            "event": "run_interrupted",
            "reason": "supervisor_lost_process",
        }
    ]


def test_positive_exit_without_terminal_event_is_failed_and_records_event(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(settings, outcomes=[(64, None)])
    manager = RunManager(settings, store, process_factory=factory)

    run = manager.enqueue_experiment(experiment_record(dataset))
    saved = wait_for_status(store, run["id"], {"failed"})

    assert saved["return_code"] == 64
    assert "exited with code 64" in saved["error"]
    events = [
        json.loads(line)
        for line in factory.processes[0].events_path.read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["run_started", "run_failed"]


def test_fifo_failure_pauses_and_resume_continues(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = SQLiteRunStore(settings.state_dir / "runs.sqlite3")
    factory = ProcessFactory(settings)
    manager = RunManager(
        settings, store, process_factory=factory, identity_verifier=factory.verify
    )

    first = manager.enqueue_experiment(experiment_record(dataset))
    wait_for_status(store, first["id"], {"running"})
    second = manager.enqueue_experiment(experiment_record(dataset))
    assert store.get_record("runs", second["id"])["status"] == "queued"

    factory.processes[0].finish(7, "run_failed")
    failed = wait_for_status(store, first["id"], {"failed"})
    assert failed["return_code"] == 7
    assert store.get_queue_control()["paused"] is True
    time.sleep(0.1)
    assert len(factory.processes) == 1
    assert store.get_record("runs", second["id"])["status"] == "queued"

    manager.resume_queue(store.get_queue_control()["pause_generation"])
    wait_for_status(store, second["id"], {"running"})
    wait_for_processes(factory, 2)[1].finish(0, "run_finished")
    wait_for_status(store, second["id"], {"succeeded"})
    wait_for_runner_exit(store)


def test_cancelled_queued_run_pauses_queue(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(settings)
    manager = RunManager(settings, store, process_factory=factory)

    first = manager.enqueue_experiment(experiment_record(dataset))
    wait_for_status(store, first["id"], {"running"})
    second = manager.enqueue_experiment(experiment_record(dataset))
    cancelled = manager.cancel(second["id"])
    assert cancelled["status"] == "cancelled"
    control = store.get_queue_control()
    assert control["paused"] is True
    assert second["id"] in control["reason"]

    factory.processes[0].finish(0, "run_finished")
    wait_for_status(store, first["id"], {"succeeded"})
    wait_for_runner_exit(store)
    assert len(factory.processes) == 1


def test_shutdown_is_non_destructive_and_run_survives(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(settings)
    manager = RunManager(settings, store, process_factory=factory)

    run = manager.enqueue_experiment(experiment_record(dataset))
    wait_for_status(store, run["id"], {"running"})
    control = store.get_queue_control()
    manager.shutdown()
    time.sleep(0.1)

    assert store.get_record("runs", run["id"])["status"] == "running"
    assert store.get_queue_control() == control
    assert factory.processes[0].poll() is None
    assert factory.processes[0].terminate_calls == 0

    factory.processes[0].finish(0, "run_finished")
    wait_for_status(store, run["id"], {"succeeded"})


def test_restart_and_concurrent_submission_do_not_duplicate_launch(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    database = settings.state_dir / "runs.sqlite3"
    first_store = SQLiteRunStore(database)
    second_store = SQLiteRunStore(database)
    factory = ProcessFactory(settings)
    first_manager = RunManager(
        settings,
        first_store,
        process_factory=factory,
        identity_verifier=factory.verify,
    )
    second_manager = RunManager(
        settings,
        second_store,
        process_factory=factory,
        identity_verifier=factory.verify,
    )

    first = first_manager.enqueue_experiment(experiment_record(dataset))
    wait_for_status(second_store, first["id"], {"running"})
    first_manager.shutdown()
    second_manager.start()
    second = second_manager.enqueue_experiment(experiment_record(dataset))
    time.sleep(0.1)
    assert len(factory.processes) == 1
    assert second_store.get_record("runs", second["id"])["status"] == "queued"

    factory.processes[0].finish(0, "run_finished")
    wait_for_processes(factory, 2)
    factory.processes[1].finish(0, "run_finished")
    wait_for_status(second_store, second["id"], {"succeeded"})
    wait_for_runner_exit(second_store)
    first_store.close()
    second_store.close()


def test_concurrent_managers_launch_in_persisted_fifo_order(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    database = settings.state_dir / "runs.sqlite3"
    first_store = SQLiteRunStore(database)
    second_store = SQLiteRunStore(database)
    factory = ProcessFactory(settings)
    managers = [
        RunManager(
            settings,
            first_store,
            process_factory=factory,
            identity_verifier=factory.verify,
        ),
        RunManager(
            settings,
            second_store,
            process_factory=factory,
            identity_verifier=factory.verify,
        ),
    ]
    barrier = threading.Barrier(2)

    def submit(manager):
        barrier.wait()
        return manager.enqueue_experiment(experiment_record(dataset))

    with ThreadPoolExecutor(max_workers=2) as executor:
        submitted = list(executor.map(submit, managers))

    wait_for_processes(factory, 1)
    records = first_store.list_records("runs")
    assert {record["id"] for record in records} == {run["id"] for run in submitted}
    assert factory.processes[0].launch_token == records[0]["id"]
    assert first_store.get_record("runs", records[1]["id"])["status"] == "queued"

    factory.processes[0].finish(0, "run_finished")
    wait_for_processes(factory, 2)
    assert factory.processes[1].launch_token == records[1]["id"]
    factory.processes[1].finish(0, "run_finished")
    wait_for_status(first_store, records[1]["id"], {"succeeded"})
    wait_for_runner_exit(first_store)
    first_store.close()
    second_store.close()


def test_stale_supervisor_cannot_claim_and_pause_is_checked_atomically(settings):
    store = MemoryRunStore()
    store.add_record(
        "runs",
        {
            "id": "queued-run",
            "managed": True,
            "status": "queued",
            "timestamps": {},
        },
    )
    store.update_queue_control(
        {"runner_token": "current-owner", "paused": False, "reason": None}
    )
    manager = RunManager(
        settings,
        store,
        process_factory=ProcessFactory(settings),
    )

    assert manager._claim_next_run("stale-owner") is None
    assert store.get_record("runs", "queued-run")["status"] == "queued"

    store.update_queue_control({"paused": True, "reason": "previous run failed"})
    assert manager._claim_next_run("current-owner") is None
    assert store.get_record("runs", "queued-run")["status"] == "queued"

    store.update_queue_control({"paused": False, "reason": None})
    claimed = manager._claim_next_run("current-owner")
    assert claimed is not None
    assert claimed["id"] == "queued-run"
    assert claimed["status"] == "running"


def test_supervisor_claim_records_live_provisional_owner_before_launch(settings):
    store = MemoryRunStore()
    store.add_record(
        "runs",
        {
            "id": "queued-run",
            "managed": True,
            "status": "queued",
            "timestamps": {},
        },
    )
    factory = ProcessFactory(settings)
    owner = RunManager(settings, store, process_factory=factory)
    challenger = RunManager(settings, store, process_factory=factory)

    try:
        assert owner._claim_supervisor("owner-token") is True
        control = store.get_queue_control()
        assert control["runner_token"] == "owner-token"
        assert control["runner_pid"] == os.getpid()
        assert control["runner_process_started_at"]
        assert challenger._claim_supervisor("challenger-token") is False
    finally:
        owner._release_supervisor("owner-token")


def test_process_group_is_cleaned_before_success_is_persisted(settings, monkeypatch):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(settings, outcomes=[(0, "run_finished")])
    manager = RunManager(settings, store, process_factory=factory)
    group_alive = True
    sent_signals = []

    def fake_group_alive(_pgid):
        return group_alive

    def fake_killpg(_pgid, sent_signal):
        nonlocal group_alive
        record = store.list_records("runs")[0]
        assert record["status"] == "running"
        sent_signals.append(sent_signal)
        group_alive = False

    monkeypatch.setattr(manager, "_process_group_alive", fake_group_alive)
    monkeypatch.setattr("webui.runner.os.killpg", fake_killpg)

    run = manager.enqueue_experiment(experiment_record(dataset))
    saved = wait_for_status(store, run["id"], {"succeeded"})

    assert sent_signals == [signal.SIGTERM]
    assert saved["return_code"] == 0


def test_cleanup_failure_pauses_queue_until_retry_then_fifo_continues(
    settings, monkeypatch
):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(
        settings,
        outcomes=[(0, "run_finished"), (0, "run_finished")],
    )
    manager = RunManager(settings, store, process_factory=factory)
    cleanup_allowed = threading.Event()
    first_failure = threading.Event()
    cleanup_calls = 0

    def cleanup(_record, _identity):
        nonlocal cleanup_calls
        cleanup_calls += 1
        if not cleanup_allowed.is_set():
            first_failure.set()
            return False
        return True

    monkeypatch.setattr(manager, "_cleanup_process_group", cleanup)

    first = manager.enqueue_experiment(experiment_record(dataset))
    assert first_failure.wait(3)
    deadline = time.monotonic() + 3
    pending = None
    while time.monotonic() < deadline:
        candidate = store.get_record("runs", first["id"])
        if candidate and candidate.get("cleanup_pending"):
            pending = candidate
            break
        time.sleep(0.02)
    assert pending is not None
    second = manager.enqueue_experiment(experiment_record(dataset))

    control = store.get_queue_control()
    assert pending["status"] == "running"
    assert pending["status_reason"] == runner_module.CLEANUP_PENDING_REASON
    assert control["paused"] is True
    assert control["reason"] == runner_module.CLEANUP_PENDING_REASON
    assert len(factory.processes) == 1
    assert store.get_record("runs", second["id"])["status"] == "queued"
    with pytest.raises(ValueError, match="cleanup could not be verified"):
        manager.resume_queue(control["pause_generation"])

    cleanup_allowed.set()
    saved = wait_for_status(store, first["id"], {"succeeded"})
    assert saved["cleanup_pending"] is False
    assert cleanup_calls >= 2
    wait_for_status(store, second["id"], {"succeeded"})
    assert len(factory.processes) == 2
    control = store.get_queue_control()
    assert control["paused"] is False
    assert control["reason"] is None
    assert control["pause_generation"] >= 1
    wait_for_runner_exit(store)


def test_recovery_cleanup_failure_stays_active_and_is_retryable(settings, monkeypatch):
    settings.ensure_directories()
    log_dir = settings.logs_root / "cleanup-recovery"
    log_dir.mkdir()
    (log_dir / "run_manifest.json").write_text(
        json.dumps({"launch_token": "cleanup-token"}), encoding="utf-8"
    )
    (log_dir / "events.jsonl").write_text('{"event":"run_started"}\n', encoding="utf-8")
    store = MemoryRunStore()
    store.add_record(
        "runs",
        {
            "id": "cleanup-recovery",
            "managed": True,
            "status": "running",
            "pid": 812345,
            "pgid": 812345,
            "process_started_at": "old-start",
            "launch_token": "cleanup-token",
            "log_dir": str(log_dir),
            "return_code": 0,
            "exit_signal": None,
            "timestamps": {"started_at": "2026-08-20T00:00:00+00:00"},
        },
    )
    manager = RunManager(settings, store)
    cleanup_results = iter([False, True])
    monkeypatch.setattr(manager, "_record_identity_state", lambda _record: "missing")
    monkeypatch.setattr(
        manager,
        "_cleanup_process_group",
        lambda *_args: next(cleanup_results),
    )

    assert manager.recover_active_runs() == 0
    pending = store.get_record("runs", "cleanup-recovery")
    control = store.get_queue_control()
    assert pending["status"] == "running"
    assert pending["cleanup_pending"] is True
    assert control["paused"] is True
    assert control["reason"] == runner_module.CLEANUP_PENDING_REASON
    assert (log_dir / "events.jsonl").read_text(encoding="utf-8") == (
        '{"event":"run_started"}\n'
    )

    assert manager.recover_active_runs() == 1
    saved = store.get_record("runs", "cleanup-recovery")
    assert saved["status"] == "interrupted"
    assert saved["cleanup_pending"] is False
    terminal = [
        json.loads(line)
        for line in (log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in terminal] == [
        "run_started",
        "run_interrupted",
    ]


def test_process_group_cleanup_rejects_reused_leader_identity(settings, monkeypatch):
    manager = RunManager(settings, MemoryRunStore())
    identity = ProcessIdentity(812345, 812345, "old-start", "run-token")
    record = {
        "pid": identity.pid,
        "pgid": identity.pgid,
        "process_started_at": identity.start_time,
        "launch_token": identity.launch_token,
    }
    monkeypatch.setattr(
        runner_module,
        "terminate_process_session",
        lambda *_args, **_kwargs: type(
            "Result", (), {"accepted": False, "outcome": "rejected"}
        )(),
    )
    monkeypatch.setattr(
        runner_module.os,
        "killpg",
        lambda *_args: pytest.fail("a reused process group must not be signalled"),
    )

    assert manager._cleanup_process_group(record, identity) is False


def test_rejected_stop_verification_cannot_fall_through_to_group_kill(
    settings, monkeypatch
):
    store = MemoryRunStore()
    record = {
        "id": "reused",
        "status": "stopping",
        "pid": 812345,
        "pgid": 812345,
        "process_started_at": "old-start",
        "launch_token": "run-token",
    }
    store.add_record("runs", record)
    manager = RunManager(settings, store)
    monkeypatch.setattr(manager, "_record_identity_state", lambda _record: "match")
    monkeypatch.setattr(
        runner_module,
        "terminate_process_session",
        lambda *_args, **_kwargs: type(
            "Result", (), {"accepted": False, "outcome": "rejected"}
        )(),
    )

    assert manager._terminate_record("reused") is False


def test_stop_restarts_supervisor_when_missing_leader_cleanup_is_pending(
    settings, monkeypatch
):
    store = MemoryRunStore()
    store.add_record(
        "runs",
        {
            "id": "cleanup-stop",
            "managed": True,
            "status": "running",
            "pid": 812345,
            "pgid": 812345,
            "process_started_at": "old-start",
            "launch_token": "cleanup-stop-token",
            "return_code": None,
            "exit_signal": None,
            "timestamps": {"started_at": "2026-08-20T00:00:00+00:00"},
        },
    )
    manager = RunManager(settings, store)
    supervisor_requested = threading.Event()
    monkeypatch.setattr(manager, "_record_identity_state", lambda _record: "missing")
    monkeypatch.setattr(manager, "_cleanup_process_group", lambda *_args: False)
    monkeypatch.setattr(manager, "_ensure_supervisor", supervisor_requested.set)

    result = manager.stop("cleanup-stop")

    assert result["status"] == "stopping"
    assert result["cleanup_pending"] is True
    assert supervisor_requested.is_set()
    control = store.get_queue_control()
    assert control["paused"] is True
    assert control["reason"] == runner_module.CLEANUP_PENDING_REASON


def test_stale_pid_is_interrupted_without_launch_or_signal(settings, monkeypatch):
    store = MemoryRunStore()
    log_dir = settings.logs_root / "stale"
    log_dir.mkdir()
    (log_dir / "run_manifest.json").write_text(
        json.dumps({"launch_token": "stale-token"}), encoding="utf-8"
    )
    (log_dir / "events.jsonl").write_text('{"event":"run_started"}\n')
    store.add_record(
        "runs",
        {
            "id": "stale",
            "managed": True,
            "status": "running",
            "pid": 812345,
            "pgid": 812345,
            "process_started_at": "old process",
            "launch_token": "stale-token",
            "log_dir": str(log_dir),
            "return_code": None,
            "exit_signal": None,
            "timestamps": {"started_at": "2026-08-20T00:00:00+00:00"},
        },
    )
    factory = ProcessFactory(settings)
    manager = RunManager(
        settings,
        store,
        process_factory=factory,
        identity_verifier=lambda *_: False,
    )
    monkeypatch.setattr(manager, "_cleanup_process_group", pytest.fail)

    manager.start()

    saved = store.get_record("runs", "stale")
    assert saved["status"] == "interrupted"
    assert store.get_queue_control()["paused"] is True
    assert factory.processes == []
    terminal = [
        json.loads(line)
        for line in (log_dir / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event"] != "run_started"
    ]
    assert len(terminal) == 1
    assert terminal[0]["event"] == "run_interrupted"


def test_stop_does_not_mutate_or_signal_when_identity_is_temporarily_unreadable(
    settings, monkeypatch
):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(settings)
    manager = RunManager(settings, store, process_factory=factory)
    run = manager.enqueue_experiment(experiment_record(dataset))
    wait_for_status(store, run["id"], {"running"})
    monkeypatch.setattr(manager, "_record_identity_state", lambda _record: "unreadable")

    with pytest.raises(ValueError, match="retry Stop"):
        manager.stop(run["id"])

    assert store.get_record("runs", run["id"])["status"] == "running"
    assert factory.processes[0].terminate_calls == 0
    factory.processes[0].finish(0, "run_finished")
    wait_for_status(store, run["id"], {"succeeded"})


@pytest.mark.parametrize("manifest_state", ["mismatch", "unreadable"])
@pytest.mark.skipif(os.name != "posix", reason="guarded process stop is POSIX-only")
def test_live_os_identity_with_untrusted_manifest_stays_active_and_pauses(
    settings, manifest_state
):
    settings.ensure_directories()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        identity = capture_process_identity(process.pid, "ledger-token")
        log_dir = settings.logs_root / "wrong-manifest"
        log_dir.mkdir()
        manifest_content = (
            json.dumps({"launch_token": "different-token"})
            if manifest_state == "mismatch"
            else "{not-json"
        )
        (log_dir / "run_manifest.json").write_text(manifest_content, encoding="utf-8")
        (log_dir / "events.jsonl").write_text(
            '{"event":"run_started"}\n', encoding="utf-8"
        )
        store = MemoryRunStore()
        store.add_record(
            "runs",
            {
                "id": "manifest-mismatch",
                "managed": True,
                "status": "running",
                "pid": identity.pid,
                "pgid": identity.pgid,
                "process_started_at": identity.start_time,
                "launch_token": identity.launch_token,
                "log_dir": str(log_dir),
                "return_code": None,
                "exit_signal": None,
                "timestamps": {"started_at": datetime.now(UTC).isoformat()},
            },
        )
        manager = RunManager(settings, store)
        manager._ensure_supervisor = lambda: None

        assert manager.recover_active_runs() == 0
        saved = store.get_record("runs", "manifest-mismatch")
        control = store.get_queue_control()

        assert saved["status"] == "running"
        assert "run_manifest.json" in saved["status_reason"]
        assert control["paused"] is True
        assert control["reason"] == saved["status_reason"]
        with pytest.raises(ValueError, match="Stop is blocked"):
            manager.resume_queue(control["pause_generation"])

        with pytest.raises(ValueError, match="Stop is blocked"):
            manager.stop("manifest-mismatch")

        assert store.get_record("runs", "manifest-mismatch")["status"] == "running"
        assert process.poll() is None
        assert (log_dir / "events.jsonl").read_text(encoding="utf-8") == (
            '{"event":"run_started"}\n'
        )
    finally:
        assert_safe_signal_target(pid=process.pid, pgid=process.pid)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def test_resume_queue_returns_control_after_supervisor_ensure(settings, monkeypatch):
    store = MemoryRunStore()
    store.add_record(
        "runs",
        {
            "id": "queued",
            "managed": True,
            "status": "queued",
            "timestamps": {},
        },
    )
    manager = RunManager(settings, store, process_factory=ProcessFactory(settings))
    monkeypatch.setattr(manager, "recover_active_runs", lambda: 0)

    def ensure_supervisor():
        store.update_queue_control(
            {
                "runner_pid": 12345,
                "runner_token": "fresh-supervisor",
            }
        )

    monkeypatch.setattr(manager, "_ensure_supervisor", ensure_supervisor)

    control = manager.resume_queue(store.get_queue_control()["pause_generation"])

    assert control["paused"] is False
    assert control["runner_pid"] == 12345
    assert control["runner_token"] == "fresh-supervisor"


def test_stale_resume_generation_cannot_clear_new_pause(settings, monkeypatch):
    store = MemoryRunStore()
    manager = RunManager(settings, store, process_factory=ProcessFactory(settings))
    monkeypatch.setattr(manager, "recover_active_runs", lambda: 0)
    monkeypatch.setattr(manager, "_ensure_supervisor", pytest.fail)
    store.update_queue_control(
        lambda control: runner_module._set_queue_paused(
            control,
            "first pause",
            force_generation=True,
        )
    )
    stale_generation = store.get_queue_control()["pause_generation"]
    store.update_queue_control(
        lambda control: runner_module._set_queue_paused(
            control,
            "new failure pause",
            force_generation=True,
        )
    )

    with pytest.raises(ValueError, match="pause changed"):
        manager.resume_queue(stale_generation)

    control = store.get_queue_control()
    assert control["paused"] is True
    assert control["reason"] == "new failure pause"
    assert control["pause_generation"] == stale_generation + 1


def test_detached_reaper_does_not_restart_after_normal_release(settings, monkeypatch):
    store = MemoryRunStore()
    manager = RunManager(settings, store)

    class ExitedProcess:
        @staticmethod
        def wait():
            return 0

    monkeypatch.setattr(
        manager,
        "_ensure_supervisor",
        lambda: pytest.fail("a normally released supervisor must not restart"),
    )

    manager._reap_detached_supervisor(ExitedProcess(), "released-token")


def test_detached_reaper_recovers_nonzero_supervisor_exit(settings, monkeypatch):
    store = MemoryRunStore()
    manager = RunManager(settings, store)
    restarted = threading.Event()

    class FailedProcess:
        @staticmethod
        def wait():
            return 1

    monkeypatch.setattr(manager, "_ensure_supervisor", restarted.set)

    manager._reap_detached_supervisor(FailedProcess(), "released-after-error")

    assert restarted.is_set()


def test_detached_reaper_exits_safely_after_store_close(settings):
    store = MemoryRunStore()
    manager = RunManager(settings, store)
    store.close()

    class ExitedProcess:
        @staticmethod
        def wait():
            return -signal.SIGKILL

    manager._reap_detached_supervisor(ExitedProcess(), "orphaned-token")


def test_external_monitor_retries_transient_unreadable_identity(settings, monkeypatch):
    settings.ensure_directories()
    store = MemoryRunStore()
    log_dir = settings.logs_root / "transient-inspection"
    log_dir.mkdir()
    (log_dir / "run_manifest.json").write_text(
        json.dumps({"launch_token": "transient-token"}), encoding="utf-8"
    )
    (log_dir / "events.jsonl").write_text('{"event":"run_started"}\n', encoding="utf-8")
    record = store.add_record(
        "runs",
        {
            "id": "transient",
            "managed": True,
            "status": "running",
            "pid": 812345,
            "pgid": 812345,
            "process_started_at": "process-start",
            "launch_token": "transient-token",
            "log_dir": str(log_dir),
            "return_code": None,
            "exit_signal": None,
            "timestamps": {"started_at": datetime.now(UTC).isoformat()},
        },
    )
    manager = RunManager(settings, store, process_factory=ProcessFactory(settings))
    observed_states = iter(["unreadable", "unreadable", "missing"])
    calls = []

    def identity_state(_record):
        state = next(observed_states)
        calls.append(state)
        return state

    monkeypatch.setattr(manager, "_record_identity_state", identity_state)
    monkeypatch.setattr(manager, "_cleanup_process_group", lambda *_args: True)
    monkeypatch.setattr(runner_module.time, "sleep", lambda _delay: None)

    manager._monitor_external_run(record)

    assert calls == ["unreadable", "unreadable", "missing"]
    assert store.get_record("runs", "transient")["status"] == "interrupted"


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-only")
@pytest.mark.parametrize("cleanup_path", ["recovery", "stop_worker"])
def test_dead_leader_cleanup_drains_live_descendants(settings, cleanup_path):
    settings.ensure_directories()
    ready = settings.state_dir / "leader-ready"
    release = settings.state_dir / "leader-release"
    script = (
        "import subprocess, sys, time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "Path(sys.argv[1]).write_text(str(child.pid)); "
        "gate=Path(sys.argv[2]); "
        "\nwhile not gate.exists(): time.sleep(0.01)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", script, str(ready), str(release)],
        start_new_session=True,
    )
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 3
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file()
        child_pid = int(ready.read_text(encoding="utf-8"))
        identity = capture_process_identity(leader.pid, "recovery-token")
        assert identity.pgid == identity.pid

        log_dir = settings.logs_root / "leader-exited"
        log_dir.mkdir()
        (log_dir / "run_manifest.json").write_text(
            json.dumps({"launch_token": identity.launch_token}), encoding="utf-8"
        )
        (log_dir / "events.jsonl").write_text(
            '{"event":"run_started"}\n', encoding="utf-8"
        )
        store = MemoryRunStore()
        store.add_record(
            "runs",
            {
                "id": "leader-exited",
                "managed": True,
                "status": "stopping" if cleanup_path == "stop_worker" else "running",
                "pid": identity.pid,
                "pgid": identity.pgid,
                "process_started_at": identity.start_time,
                "launch_token": identity.launch_token,
                "log_dir": str(log_dir),
                "return_code": None,
                "exit_signal": None,
                "timestamps": {"started_at": datetime.now(UTC).isoformat()},
            },
        )

        release.touch()
        assert leader.wait(timeout=3) == 0
        os.kill(child_pid, 0)

        manager = RunManager(settings, store)
        if cleanup_path == "recovery":
            assert manager.recover_active_runs() == 1
            assert store.get_record("runs", "leader-exited")["status"] == "interrupted"
        else:
            assert manager._terminate_record("leader-exited") is True
            assert store.get_record("runs", "leader-exited")["status"] == "stopping"

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            raise AssertionError(f"descendant process {child_pid} survived recovery")
    finally:
        release.touch()
        if leader.poll() is None:
            assert_safe_signal_target(pid=leader.pid)
            leader.kill()
            leader.wait(timeout=3)
        assert_safe_signal_target(pid=leader.pid, pgid=leader.pid)
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if child_pid is not None:
            assert_safe_signal_target(pid=child_pid)
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_dead_supervisor_pid_is_not_kept_alive_by_recent_claim(settings):
    store = MemoryRunStore()
    log_dir = settings.logs_root / "dead-supervisor"
    log_dir.mkdir()
    (log_dir / "events.jsonl").write_text('{"event":"run_started"}\n')
    now = datetime.now(UTC).isoformat()
    store.add_record(
        "runs",
        {
            "id": "unlaunched",
            "managed": True,
            "status": "running",
            "pid": None,
            "pgid": None,
            "process_started_at": None,
            "launch_token": "run-token",
            "supervisor_token": "dead-runner",
            "log_dir": str(log_dir),
            "timestamps": {"started_at": now},
        },
    )
    store.update_queue_control(
        {
            "runner_pid": 899999,
            "runner_process_started_at": "dead",
            "runner_token": "dead-runner",
            "runner_claimed_at": now,
        }
    )
    manager = RunManager(
        settings,
        store,
        process_factory=ProcessFactory(settings),
        identity_verifier=lambda *_: False,
    )

    manager.start()

    assert store.get_record("runs", "unlaunched")["status"] == "interrupted"
    assert store.get_queue_control()["runner_token"] is None


def test_stop_validates_then_terminates_and_persists_signal(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(settings)
    manager = RunManager(settings, store, process_factory=factory)

    run = manager.enqueue_experiment(experiment_record(dataset))
    wait_for_status(store, run["id"], {"running"})
    stopping = manager.stop(run["id"])
    assert stopping["status"] == "stopping"

    saved = wait_for_status(store, run["id"], {"stopped"})
    assert factory.processes[0].terminate_calls == 1
    assert saved["return_code"] == -signal.SIGTERM
    assert saved["exit_signal"] == signal.SIGTERM
    assert store.get_queue_control()["paused"] is True


@pytest.mark.parametrize(
    ("return_code", "terminal_event", "expected", "expected_signal"),
    [
        (5, "run_failed", "failed", None),
        (-signal.SIGKILL, "run_interrupted", "interrupted", signal.SIGKILL),
    ],
)
def test_exit_code_and_signal_are_audit_fields(
    settings, return_code, terminal_event, expected, expected_signal
):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()
    factory = ProcessFactory(settings, outcomes=[(return_code, terminal_event)])
    manager = RunManager(settings, store, process_factory=factory)

    run = manager.enqueue_experiment(experiment_record(dataset))
    saved = wait_for_status(store, run["id"], {expected})

    assert saved["return_code"] == return_code
    assert saved["exit_signal"] == expected_signal
    assert store.get_queue_control()["paused"] is True


def test_history_import_is_idempotent_and_never_changes_existing(settings):
    store = MemoryRunStore()
    manager = RunManager(settings, store, process_factory=ProcessFactory(settings))
    log_dir = settings.logs_root / "history"
    log_dir.mkdir()
    original = {
        "id": "history-1",
        "status": "succeeded",
        "log_dir": str(log_dir),
        "timestamps": {},
    }

    imported = manager.import_log_record(original)
    duplicate = manager.import_log_record(
        {**original, "id": "different-id", "status": "failed"}
    )
    changed = manager.import_log_record({**original, "status": "failed"})

    assert imported["managed"] is False
    assert duplicate == imported
    assert changed == imported
    assert manager.import_history([original]) == 0
    assert store.list_records("runs") == [imported]
    with pytest.raises(ValueError, match="view-only"):
        manager.requeue(imported["id"])


def test_history_import_reuses_managed_run_with_same_launch_token(settings):
    store = MemoryRunStore()
    manager = RunManager(settings, store, process_factory=ProcessFactory(settings))
    managed = store.add_record(
        "runs",
        {
            "id": "managed-run",
            "managed": True,
            "status": "running",
            "launch_token": "shared-launch-token",
            "log_dir": None,
            "timestamps": {},
        },
    )

    imported = manager.import_log_record(
        {
            "id": "historical-row",
            "status": "running",
            "launch_token": "shared-launch-token",
            "log_dir": str(settings.logs_root / "historical-row"),
            "timestamps": {},
        }
    )

    assert imported == managed
    assert store.list_records("runs") == [managed]


def test_requeue_preserves_model_resolved_from_settings(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    settings.env_file.write_text(
        "BIOMAKER_MAIN_MODEL=deepseek-v4-pro\n", encoding="utf-8"
    )
    store = MemoryRunStore()
    factory = ProcessFactory(settings, outcomes=[(0, "run_finished")])
    manager = RunManager(settings, store, process_factory=factory)

    first = manager.enqueue_experiment(experiment_record(dataset))
    wait_for_status(store, first["id"], {"succeeded"})
    settings.env_file.write_text(
        "BIOMAKER_MAIN_MODEL=deepseek-v4-flash\n", encoding="utf-8"
    )
    factory.outcomes.append((0, "run_finished"))
    retry = manager.requeue(first["id"])

    assert retry["options"]["model_for_all_agents"] == "deepseek-v4-pro"
    assert "model_for_all_agents=deepseek-v4-pro" in retry["command"]
    wait_for_status(store, retry["id"], {"succeeded"})
