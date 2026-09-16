from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import run_biomaker_main9_queue as main9  # noqa: E402
import run_fig2_medmnist_queue as fig2  # noqa: E402


@pytest.fixture(autouse=True)
def clean_compute_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "BIOMAKER_GPU_UUIDS", "BIOMAKER_PARALLEL_WORKERS", "CUDA_VISIBLE_DEVICES",
        "BIOMAKER_STORAGE_ROOT", "BIOMAKER_PYTHON", "UV_PROJECT_ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("queue", [main9, fig2])
@pytest.mark.parametrize("gpu_count", [1, 4, 8])
def test_gpu_allocation_scales_to_local_and_server(
    monkeypatch: pytest.MonkeyPatch, queue, gpu_count: int,
) -> None:
    inventory = tuple(f"GPU-{index}" for index in range(gpu_count))
    monkeypatch.setattr(fig2, "gpu_inventory", lambda: inventory)
    monkeypatch.setattr(fig2, "gpu_compute_processes", lambda _gpus: [])
    assert queue.resolve_compute_profile() == (inventory[:4], min(gpu_count, 4))


@pytest.mark.parametrize("mask", ["3,1", "GPU-3,GPU-1"])
def test_auto_assignment_respects_cuda_allocation_and_busy_devices(
    monkeypatch: pytest.MonkeyPatch, mask: str,
) -> None:
    monkeypatch.setattr(fig2, "gpu_inventory", lambda: tuple(f"GPU-{i}" for i in range(4)))
    monkeypatch.setattr(fig2, "gpu_compute_processes", lambda _gpus: [("123", "GPU-1")])
    monkeypatch.setattr(fig2.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(
        returncode=0, stdout='["GPU-3", "GPU-1"]', stderr="",
    ))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert fig2.resolve_compute_profile() == (("GPU-3",), 1)


@pytest.mark.parametrize("mask", ["", "-1", "5", "GPU-missing", "0,0"])
def test_unavailable_cuda_mask_never_selects_another_gpu(
    monkeypatch: pytest.MonkeyPatch, mask: str,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    with pytest.raises(fig2.QueueConfigurationError, match="CUDA_VISIBLE_DEVICES"):
        fig2.visible_gpu_inventory(("GPU-0", "GPU-1"))


def test_explicit_gpu_cannot_escape_allocated_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIOMAKER_GPU_UUIDS", "GPU-1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-0")
    with pytest.raises(fig2.QueueConfigurationError, match="stay within"):
        fig2.resolve_compute_profile(lambda: ("GPU-0", "GPU-1"))


def test_numeric_cuda_mask_uses_cuda_order_not_smi_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    observed = []

    def probe(command, **_kwargs):
        observed.append(command)
        return SimpleNamespace(returncode=0, stdout='["GPU-3", "GPU-1"]', stderr="")

    monkeypatch.setattr(fig2.subprocess, "run", probe)
    python = tmp_path / "runtime" / ".venv" / "bin" / "python"
    assert fig2.visible_gpu_inventory(
        ("GPU-0", "GPU-1", "GPU-2", "GPU-3"), python=python,
    ) == ("GPU-3", "GPU-1")
    assert observed[0][0] == str(python)
    assert "torch.cuda.get_device_properties" in observed[0][2]


@pytest.mark.parametrize("queue", [main9, fig2])
def test_default_paths_are_relative_to_runtime_workspace(tmp_path: Path, queue) -> None:
    runtime = tmp_path / "checkout" / "BioMaker"
    args = queue.parse_arguments(["--runtime-root", str(runtime)])
    paths = queue.resolve_paths(args)
    assert paths.data_root.is_relative_to(runtime.parent)
    assert paths.logs_root.is_relative_to(runtime.parent)
    assert paths.workspaces_root.is_relative_to(runtime.parent)
    assert paths.resource_root.is_relative_to(runtime.parent)
    assert paths.state_dir.is_relative_to(runtime.parent)
    assert queue.DEFAULT_REGISTRY.is_relative_to(queue.REPOSITORY_ROOT.parent / "experiments")
    assert paths.python == runtime / ".venv" / "bin" / "python"


def test_uv_environment_symlink_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    python = runtime / "custom-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "custom-env")
    assert fig2.runtime_python(runtime) == python
    monkeypatch.setenv("BIOMAKER_PYTHON", str(tmp_path / "override-python"))
    assert fig2.runtime_python(runtime) == tmp_path / "override-python"
    assert fig2.runtime_python(runtime, python) == python


def test_child_threads_respect_cpu_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fig2.os, "sched_getaffinity", lambda _pid: {0, 2, 3, 4, 6, 7})
    for name in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
        monkeypatch.delenv(f"BIOMAKER_{name}_NUM_THREADS", raising=False)
    env = fig2.child_environment(("GPU-0", "GPU-1"), 2)
    assert env["OMP_NUM_THREADS"] == "3"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-0,GPU-1"
    assert env["BIOMAKER_DISABLE_RUN_QUEUE"] == "1"


def test_suite_overrides_match_current_cli_schema(tmp_path: Path) -> None:
    from biomaker.utils.config import Config, _load_cfg

    paths = fig2.resolve_paths(fig2.parse_arguments(["--runtime-root", str(tmp_path)]))
    command = fig2.build_run_command(
        paths=paths, registry_path=tmp_path / "tasks.json",
        task={"desc_file": "task.md"}, stage="canary", steps=4,
        total_time_limit=600, node_timeout=120, workers=1,
        model="deepseek-v4-flash", requested_name="test", data_dir=tmp_path / "data",
        generate_report=False,
    )
    configured = OmegaConf.merge(_load_cfg(use_cli_args=False), OmegaConf.from_dotlist(command[3:]))
    cfg = OmegaConf.merge(OmegaConf.structured(Config), configured)
    assert cfg.private_evaluation.per_node is False
    assert cfg.final_retrain.enabled is False
    assert cfg.exec.parallel_workers == 1
    assert cfg.agent.search.max_children_per_node == 4


def test_audit_accepts_parallel_completion_and_checks_scheduled_mapping() -> None:
    nodes = [
        {"id": f"node-{planned}", "step": step, "scheduled_step": planned}
        for step, planned in enumerate([2, 0, 3, 1])
    ]
    events = [
        {"node_id": node["id"], "step": node["step"], "scheduled_step": node["scheduled_step"]}
        for node in nodes
    ]
    assert fig2._step_order_errors(nodes, events, 4) == []
    events[0]["scheduled_step"] = 0
    assert "mapping differs" in " ".join(fig2._step_order_errors(nodes, events, 4))
    events[0]["scheduled_step"] = 2
    nodes[0]["scheduled_step"] = 0
    assert "complete and unique" in " ".join(fig2._step_order_errors(nodes, events, 4))


def test_relative_credential_path_matches_child_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "runtime"
    paths = fig2.resolve_paths(fig2.parse_arguments(["--runtime-root", str(runtime)]))
    monkeypatch.setenv("BIOMAKER_ENV_FILE", "config/provider.env")
    assert fig2.runtime_env_file(paths) == runtime / "config" / "provider.env"


@pytest.mark.parametrize("queue", [main9, fig2])
def test_missing_registry_is_a_configuration_error(tmp_path: Path, queue) -> None:
    with pytest.raises(queue.QueueConfigurationError, match="cannot read task registry"):
        queue.load_registry(tmp_path / "missing.json")


def test_retina_frozen_ordinal_metric_is_blocked_without_changing_it(tmp_path: Path) -> None:
    task = {"task_id": "retinamnist", "primary_metric": "ordinal_accuracy"}
    check = fig2.ReadinessTracker().check(task, tmp_path, {"status": "ready"})
    assert check.ready is False
    assert "unsupported evaluation" in check.reason
    with pytest.raises(fig2.QueueConfigurationError, match="ordinal_accuracy"):
        fig2.require_supported_tasks([task])
    assert task["primary_metric"] == "ordinal_accuracy"
