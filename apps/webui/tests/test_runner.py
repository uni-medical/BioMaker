from __future__ import annotations

import io
import threading
import time
from pathlib import Path

from webui.models import AdvancedOptions
from webui.run_store import MemoryRunStore
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


def test_build_command_uses_settings_model_unless_task_overrides(settings, tmp_path):
    settings.env_file.write_text(
        "BIOMAKER_MAIN_MODEL=deepseek-v4-flash\n", encoding="utf-8"
    )
    base_options = AdvancedOptions().model_dump()
    command = build_command(
        settings,
        dataset_path="/data/dermamnist",
        task_file=tmp_path / "task.md",
        requested_exp_name="settings-model",
        options=base_options,
    )
    assert "model_for_all_agents=deepseek-v4-flash" in command

    override_command = build_command(
        settings,
        dataset_path="/data/dermamnist",
        task_file=tmp_path / "task.md",
        requested_exp_name="task-model",
        options={**base_options, "model_for_all_agents": "deepseek-v4-pro"},
    )
    assert "model_for_all_agents=deepseek-v4-pro" in override_command
    assert "model_for_all_agents=deepseek-v4-flash" not in override_command


def test_build_command_leaves_naming_to_cli_when_name_is_blank(settings, tmp_path):
    command = build_command(
        settings,
        dataset_path="/data/dermamnist",
        task_file=tmp_path / "task.md",
        requested_exp_name=None,
        options=AdvancedOptions().model_dump(),
    )

    assert not any(argument.startswith("exp_name=") for argument in command)


def test_requeue_preserves_model_resolved_from_settings(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    settings.env_file.write_text(
        "BIOMAKER_MAIN_MODEL=deepseek-v4-pro\n", encoding="utf-8"
    )
    registry = MemoryRunStore()
    manager = RunManager(settings, registry, process_factory=ImmediateProcess)
    manager.start()
    try:
        first = manager.enqueue_experiment(experiment_record(dataset))
        assert wait_for_status(registry, first["id"], {"succeeded"}) == "succeeded"
        saved = registry.get_record("runs", first["id"])
        assert saved["options"]["model_for_all_agents"] == "deepseek-v4-pro"

        settings.env_file.write_text(
            "BIOMAKER_MAIN_MODEL=deepseek-v4-flash\n", encoding="utf-8"
        )
        retry = manager.requeue(first["id"])

        assert retry["options"]["model_for_all_agents"] == "deepseek-v4-pro"
        assert "model_for_all_agents=deepseek-v4-pro" in retry["command"]
        assert "model_for_all_agents=deepseek-v4-flash" not in retry["command"]
    finally:
        manager.shutdown()


def test_queue_runs_one_process_and_supports_cancel_and_stop(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    registry = MemoryRunStore()
    processes = []

    def factory(*args, **kwargs):
        process = BlockingProcess(*args, **kwargs)
        processes.append(process)
        return process

    manager = RunManager(settings, registry, process_factory=factory)
    manager.start()
    try:
        first = manager.enqueue_experiment(experiment_record(dataset))
        wait_for_status(registry, first["id"], {"running"})
        second = manager.enqueue_experiment(experiment_record(dataset))
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
    registry = MemoryRunStore()
    manager = RunManager(settings, registry, process_factory=ImmediateProcess)
    manager.start()
    try:
        run = manager.enqueue_experiment(experiment_record(dataset))
        assert wait_for_status(registry, run["id"], {"succeeded"}) == "succeeded"
        console = Path(run["console_log"]).read_text(encoding="utf-8")
        assert "BioMaker test output" in console
        assert "exited with code 0" in console
    finally:
        manager.shutdown()


def test_run_discovers_biomaker_numbered_output_directory(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    registry = MemoryRunStore()

    def factory(command, **kwargs):
        requested_name = next(
            item.removeprefix("exp_name=")
            for item in command
            if item.startswith("exp_name=")
        )
        log_dir = settings.logs_root / f"17-{requested_name}"
        log_dir.mkdir()
        (log_dir / "run_manifest.json").write_text(
            '{"launch_token": "' + kwargs["env"]["BIOMAKER_LAUNCH_TOKEN"] + '"}',
            encoding="utf-8",
        )
        (log_dir / "tree_plot.html").write_text("tree", encoding="utf-8")
        return ImmediateProcess(command, **kwargs)

    manager = RunManager(settings, registry, process_factory=factory)
    manager.start()
    try:
        run = manager.enqueue_experiment(
            experiment_record(dataset), requested_exp_name="named-run"
        )
        assert wait_for_status(registry, run["id"], {"succeeded"}) == "succeeded"
        saved = registry.get_record("runs", run["id"])
        assert saved["actual_exp_name"] == "17-named-run"
        assert saved["log_dir"].endswith(saved["actual_exp_name"])
        assert saved["workspace_dir"].endswith(saved["actual_exp_name"])
    finally:
        manager.shutdown()


def test_run_uses_launch_token_when_external_cli_creates_another_log(settings):
    dataset = settings.datasets_root / "sample"
    dataset.mkdir()
    store = MemoryRunStore()

    def factory(command, **kwargs):
        external = settings.logs_root / "20-external-cli"
        external.mkdir()
        (external / "run_manifest.json").write_text(
            '{"launch_token": null}', encoding="utf-8"
        )

        owned = settings.logs_root / "21-cli-generated-name"
        owned.mkdir()
        (owned / "run_manifest.json").write_text(
            '{"launch_token": "' + kwargs["env"]["BIOMAKER_LAUNCH_TOKEN"] + '"}',
            encoding="utf-8",
        )
        return ImmediateProcess(command, **kwargs)

    manager = RunManager(settings, store, process_factory=factory)
    manager.start()
    try:
        run = manager.enqueue_experiment(experiment_record(dataset))
        assert wait_for_status(store, run["id"], {"succeeded"}) == "succeeded"
        saved = store.get_record("runs", run["id"])
        assert saved["actual_exp_name"] == "21-cli-generated-name"
        assert saved["log_dir"].endswith("21-cli-generated-name")
    finally:
        manager.shutdown()


def test_new_webui_session_does_not_inherit_run_control_state(settings):
    registry = MemoryRunStore()
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
    restarted_store = MemoryRunStore()
    manager = RunManager(settings, restarted_store, process_factory=ImmediateProcess)

    assert manager.recover_active_runs() == 0
    assert restarted_store.list_records("runs") == []
    assert len(registry.list_records("runs")) == 2
