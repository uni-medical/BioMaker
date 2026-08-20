from __future__ import annotations

import io
import threading
import time
from pathlib import Path

from webui.models import AdvancedOptions
from webui.registry import JsonRegistry
from webui.runner import RunManager, build_command


class ImmediateProcess:
    pid = None

    def __init__(self, *_args, **_kwargs):
        self.stdout = io.StringIO("BioMaker test output\n")
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class BlockingProcess:
    pid = None

    def __init__(self, *_args, **_kwargs):
        self.stdout = io.StringIO("BioMaker started\n")
        self.returncode = None
        self.terminated = threading.Event()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.terminated.wait(timeout or 2)
        return self.returncode

    def terminate(self):
        self.returncode = -15
        self.terminated.set()

    def kill(self):
        self.returncode = -9
        self.terminated.set()


def wait_for_status(registry, run_id, expected, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = registry.get_record("runs", run_id)["status"]
        if status in expected:
            return status
        time.sleep(0.02)
    raise AssertionError(
        f"Run {run_id} did not reach {expected}; "
        f"current={registry.get_record('runs', run_id)['status']}"
    )


def project_record(dataset_path):
    return {
        "id": "project-1",
        "name": "Test project",
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


def test_queue_runs_one_process_and_supports_cancel_and_stop(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    registry = JsonRegistry(settings.state_dir / "registry.json")
    processes = []

    def factory(*args, **kwargs):
        process = BlockingProcess(*args, **kwargs)
        processes.append(process)
        return process

    manager = RunManager(settings, registry, process_factory=factory)
    manager.start()
    try:
        first = manager.enqueue_project(project_record(dataset))
        wait_for_status(registry, first["id"], {"running"})
        second = manager.enqueue_project(project_record(dataset))
        wait_for_status(registry, second["id"], {"queued"})

        manager.cancel(second["id"])
        assert wait_for_status(registry, second["id"], {"cancelled"}) == "cancelled"
        manager.stop(first["id"])
        assert wait_for_status(registry, first["id"], {"stopped"}) == "stopped"
        assert len(processes) == 1
        assert processes[0].terminated.is_set()
    finally:
        manager.shutdown()


def test_successful_run_captures_console(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    registry = JsonRegistry(settings.state_dir / "registry.json")
    manager = RunManager(settings, registry, process_factory=ImmediateProcess)
    manager.start()
    try:
        run = manager.enqueue_project(project_record(dataset))
        assert wait_for_status(registry, run["id"], {"succeeded"}) == "succeeded"
        console = Path(run["console_log"]).read_text(encoding="utf-8")
        assert "BioMaker test output" in console
        assert "exited with code 0" in console
    finally:
        manager.shutdown()


def test_run_discovers_biomaker_numbered_output_directory(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    registry = JsonRegistry(settings.state_dir / "registry.json")

    def factory(command, **kwargs):
        requested_name = next(
            item.removeprefix("exp_name=")
            for item in command
            if item.startswith("exp_name=")
        )
        log_dir = settings.logs_root / f"17-{requested_name}"
        log_dir.mkdir()
        (log_dir / "tree_plot.html").write_text("tree", encoding="utf-8")
        return ImmediateProcess(command, **kwargs)

    manager = RunManager(settings, registry, process_factory=factory)
    manager.start()
    try:
        run = manager.enqueue_project(project_record(dataset))
        assert wait_for_status(registry, run["id"], {"succeeded"}) == "succeeded"
        saved = registry.get_record("runs", run["id"])
        assert saved["actual_exp_name"].startswith("17-webui-Test-project-")
        assert saved["log_dir"].endswith(saved["actual_exp_name"])
        assert saved["workspace_dir"].endswith(saved["actual_exp_name"])
    finally:
        manager.shutdown()


def test_restart_marks_active_records_interrupted(settings):
    registry = JsonRegistry(settings.state_dir / "registry.json")
    for run_id, run_status in (("queued-run", "queued"), ("active-run", "running")):
        registry.add_record(
            "runs",
            {
                "id": run_id,
                "status": run_status,
                "return_code": None,
                "timestamps": {"created_at": "before", "finished_at": None},
            },
        )
    manager = RunManager(settings, registry, process_factory=ImmediateProcess)

    assert manager.recover_incomplete_runs() == 2
    assert registry.get_record("runs", "queued-run")["status"] == "interrupted"
    assert registry.get_record("runs", "active-run")["status"] == "interrupted"
