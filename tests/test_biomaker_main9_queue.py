from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import run_biomaker_main9_queue as queue  # noqa: E402


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _registry(root: Path) -> Path:
    tasks = []
    for order, task_id in enumerate(queue.FROZEN_TASK_ORDER, start=1):
        family, metric = queue.FROZEN_TASK_CONTRACT[task_id]
        _write(root / "tasks" / f"{task_id}.md", f"# {task_id}\n")
        tasks.append(
            {
                "order": order,
                "task_id": task_id,
                "status": "active",
                "task_family": family,
                "primary_metric": metric,
                "desc_file": f"tasks/{task_id}.md",
            }
        )
    path = root / "tasks.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "protocol_id": queue.PROTOCOL_ID,
            "defaults": {
                "model": "deepseek-v4-flash",
                "canary_steps": 4,
                "full_steps": 35,
                "canary_total_time_limit_seconds": 30,
                "full_total_time_limit_seconds": 100,
                "canary_hard_wall_clock_seconds": 40,
                "full_hard_wall_clock_seconds": 121,
                "per_node_timeout_seconds": 10,
                "progress_timeout_seconds": 20,
                "final_retrain_progress_timeout_seconds": 16,
                "shutdown_grace_seconds": 1,
                "generate_report_for_canary": False,
                "generate_report_for_full": False,
            },
            "tasks": tasks,
        },
    )
    return path


def _prepared_fixture(
    root: Path,
    task: dict,
    *,
    cache_root: Path,
    status: str = "ready",
    record_family_key: str = "task_family",
) -> tuple[Path, dict]:
    registry_path = Path(str(task["_registry_path"]))
    desc = registry_path.parent / str(task["desc_file"])
    desc_hash = queue._harness.sha256_file(desc)
    data_dir = root / "data" / str(task["task_id"])
    _write_json(data_dir / "task_metadata.json", {"task_id": task["task_id"]})
    counts = {"train": 6, "score_val": 2, "test": 3}
    ready = {
        "status": status,
        "task_id": task["task_id"],
        "task_family": task["task_family"],
        "primary_metric": task["primary_metric"],
        "task_text_sha256": desc_hash,
        "counts": counts,
    }
    ready_path = data_dir / "_READY.json"
    _write_json(ready_path, ready)
    cache_key = f"cache-{task['task_id']}"
    cache = cache_root / cache_key
    _write_json(
        cache / "manifest.json",
        {
            "verifier_ok": True,
            "dataset": task["task_id"],
            "task_text_sha256": desc_hash,
            "primary_metric": task["primary_metric"],
        },
    )
    _write_json(cache / "verifier.json", {"ok": True})
    for name in (
        "split_manifest.json",
        "biomaker_splits/agent_train.csv",
        "biomaker_splits/score_val.csv",
        "biomaker_splits/agent_test.csv",
        "private_gt/score_val.csv",
        "private_gt/test.csv",
        "private_gt/private_gt.json",
    ):
        _write(cache / name)
    record = {
        "task_id": task["task_id"],
        "status": status,
        "data_dir": str(data_dir),
        "ready_marker": str(ready_path),
        "primary_metric": task["primary_metric"],
        "task_text_sha256": desc_hash,
        "counts": counts,
        "cache_key": cache_key,
        "cache_entry": str(cache),
        "_manifest_root": str(root),
        "_manifest_path": str(root / "suite_manifest.json"),
    }
    record[record_family_key] = task["task_family"]
    return data_dir, record


def test_registry_freezes_exact_main9_and_rejects_forbidden(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    registry = queue.load_registry(registry_path)
    assert [task["task_id"] for task in queue.active_tasks(registry)] == list(
        queue.FROZEN_TASK_ORDER
    )
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["tasks"][-1]["task_id"] = "aapm-low-dose-ct"
    _write_json(registry_path, payload)
    with pytest.raises(queue.QueueConfigurationError):
        queue.load_registry(registry_path)


def test_only_mode_requires_only_selected_task_text(tmp_path: Path) -> None:
    registry_path = _registry(tmp_path)
    (tmp_path / "tasks" / "kvasir_seg.md").unlink()
    registry = queue.load_registry(
        registry_path,
        require_descriptions_for={"monuseg"},
    )
    assert [task["task_id"] for task in queue.active_tasks(registry, {"monuseg"})] == [
        "monuseg"
    ]


def test_readiness_accepts_score_val_and_rejects_family_alias(tmp_path: Path) -> None:
    registry = queue.load_registry(_registry(tmp_path / "registry"))
    task = queue.active_tasks(registry, {"monuseg"})[0]
    cache_root = tmp_path / "shared" / "split_builder"
    data_dir, record = _prepared_fixture(tmp_path, task, cache_root=cache_root)
    check = queue.ReadinessTracker().check(
        task, data_dir, record, expected_cache_root=cache_root
    )
    assert check.ready
    assert check.row_counts == {"train": 6, "val": 2, "test": 3}

    _data, alias_record = _prepared_fixture(
        tmp_path / "alias", task, cache_root=cache_root, record_family_key="family"
    )
    alias_check = queue.ReadinessTracker().check(
        task,
        Path(str(alias_record["data_dir"])),
        alias_record,
        expected_cache_root=cache_root,
    )
    assert not alias_check.ready
    assert "task family" in alias_check.reason


def test_resolve_dataset_dir_never_invents_unprepared_path(tmp_path: Path) -> None:
    task = {"task_id": "monuseg"}
    assert queue.resolve_dataset_dir(task, {}) == (None, None)
    record = {
        "task_id": "monuseg",
        "data_dir": "packages/monuseg",
        "_manifest_root": str(tmp_path),
    }
    path, observed = queue.resolve_dataset_dir(task, {"monuseg": record})
    assert path == (tmp_path / "packages" / "monuseg").resolve()
    assert observed is record


@pytest.mark.parametrize("task_id", ["bccd", "fracatlas", "dentex", "mp_idb", "fmd"])
def test_ready_unsupported_task_is_blocked_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_id: str,
) -> None:
    registry_path = _registry(tmp_path / "registry")
    task = queue.active_tasks(queue.load_registry(registry_path), {task_id})[0]
    cache_root = tmp_path / "cache"
    data_dir, record = _prepared_fixture(tmp_path, task, cache_root=cache_root)
    check = queue.ReadinessTracker().check(
        task, data_dir, record, expected_cache_root=cache_root,
    )
    assert check.ready is False
    assert "unsupported evaluation" in check.reason
    monkeypatch.setattr(queue, "run_one_stage", lambda **_kwargs: pytest.fail("stage must not launch"))
    monkeypatch.setattr(queue, "resolve_compute_profile", lambda **_kwargs: pytest.fail("preflight must precede GPU probe"))
    with pytest.raises(queue.QueueConfigurationError, match="unsupported evaluation"):
        queue.queue_main(["--registry", str(registry_path), "--only", task_id])


def test_scheduler_skips_protocol_blocked_task_and_uses_next_ready(tmp_path: Path) -> None:
    registry = queue.load_registry(_registry(tmp_path / "registry"))
    tasks = queue.active_tasks(registry, {"monuseg", "kvasir_seg"})
    cache_root = tmp_path / "shared" / "split_builder"
    _dir, ready_record = _prepared_fixture(tmp_path, tasks[1], cache_root=cache_root)
    prepared = {
        "monuseg": {
            "task_id": "monuseg",
            "status": "protocol_blocked",
            "blocked_reason": "fixed evaluator not yet sealed",
            "_manifest_root": str(tmp_path),
        },
        "kvasir_seg": ready_record,
    }
    state = {
        "tasks": {
            task["task_id"]: {"status": "pending", "stages": {}} for task in tasks
        }
    }
    chosen = queue.choose_next_ready(
        tasks,
        state,
        prepared,
        queue.ReadinessTracker(),
        expected_cache_root=cache_root,
    )
    assert chosen is not None
    assert chosen[0]["task_id"] == "kvasir_seg"
    assert state["tasks"]["monuseg"]["status"] == "waiting_data"
    assert "protocol_blocked" in state["tasks"]["monuseg"]["data_status_reason"]


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_compute_profile_uses_one_candidate_per_available_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    del tmp_path
    inventory = tuple(f"GPU-{index}" for index in range(6))
    monkeypatch.delenv("BIOMAKER_GPU_UUIDS", raising=False)
    monkeypatch.setenv("BIOMAKER_PARALLEL_WORKERS", str(count))
    monkeypatch.setattr(queue, "_gpu_inventory", lambda: inventory)
    monkeypatch.setattr(
        queue._harness,
        "gpu_compute_processes",
        lambda _uuids: [("99", "GPU-0")],
    )
    selected, workers = queue.resolve_compute_profile()
    assert workers == count
    assert selected == inventory[1 : count + 1]


def test_eight_gpus_cap_one_search_at_four(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory = tuple(f"GPU-{index}" for index in range(8))
    monkeypatch.delenv("BIOMAKER_GPU_UUIDS", raising=False)
    monkeypatch.delenv("BIOMAKER_PARALLEL_WORKERS", raising=False)
    monkeypatch.setattr(queue, "_gpu_inventory", lambda: inventory)
    monkeypatch.setattr(queue._harness, "gpu_compute_processes", lambda _uuids: [])
    selected, workers = queue.resolve_compute_profile()
    assert workers == 4
    assert selected == inventory[:4]


def test_runtime_python_symlink_is_not_resolved(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    link = runtime / ".venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(Path(sys.executable))
    args = SimpleNamespace(
        runtime_root=runtime,
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        workspaces_root=tmp_path / "workspaces",
        resource_root=tmp_path / "resource",
        state_dir=tmp_path / "state",
        python=None,
    )
    paths = queue.resolve_paths(args)
    assert paths.python == link.absolute()
    assert paths.python != link.resolve()


def test_stage_command_keeps_branching_limit_four_and_never_uses_ddp(tmp_path: Path) -> None:
    desc = tmp_path / "tasks" / "monuseg.md"
    _write(desc)
    paths = queue.QueuePaths(
        runtime_root=tmp_path,
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        workspaces_root=tmp_path / "workspaces",
        resource_root=tmp_path / "resource",
        state_dir=tmp_path / "state",
        python=Path(sys.executable),
    )
    task = {"desc_file": "tasks/monuseg.md"}
    command = queue._harness.build_run_command(
        paths=paths,
        registry_path=tmp_path / "tasks.json",
        task=task,
        stage="canary",
        steps=4,
        total_time_limit=30,
        node_timeout=10,
        workers=2,
        model="deepseek-v4-flash",
        requested_name="main9-test",
        data_dir=tmp_path / "data",
        generate_report=False,
    )
    assert "agent.search.max_children_per_node=4" in command
    assert not any("max_round_candidates" in item for item in command)
    assert "agent.search.max_children_per_node=4" in command
    assert "exec.parallel_workers=2" in command
    assert "final_retrain.enabled=false" in command
    assert not any("torchrun" in item or "DistributedDataParallel" in item for item in command)


@pytest.mark.parametrize(
    ("metric", "private_metric"),
    sorted(queue.PRIVATE_METRIC_BY_PRIMARY.items()),
)
def test_audit_accepts_exact_main9_private_metric_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric: str,
    private_metric: str,
) -> None:
    errors = (
        f"final_retrain_result metric_name is {private_metric!r}, expected None",
        f"private_metrics metric_name is {private_metric!r}, expected None",
    )
    monkeypatch.setattr(
        queue,
        "BASE_VALIDATE_RUN_ARTIFACTS",
        lambda *_args, **_kwargs: queue.RunAudit(False, errors, {}),
    )
    result = {
        "name": "private_test",
        "metric_name": private_metric,
        "value": 0.5,
        "n": 3,
    }
    _write_json(tmp_path / "final_retrain_result.json", {"private_metrics": {"results": [result]}})
    _write_json(
        tmp_path / "private_metrics.json",
        {"history": [{"stage": "final_retrain", "private_metrics": {"results": [result]}}]},
    )
    _write_json(tmp_path / "source_state.json", {"protocol_id": queue.PROTOCOL_ID})
    audit = queue.validate_run_artifacts(
        tmp_path,
        stage="full",
        expected_steps=35,
        expected_model="deepseek-v4-flash",
        expected_metric=metric,
        expected_test_rows=3,
        expected_workers=1,
        expected_gpu_uuids=("GPU-1",),
        require_report=False,
    )
    assert audit.ok, audit.errors


def test_audit_rejects_wrong_detection_private_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        queue,
        "BASE_VALIDATE_RUN_ARTIFACTS",
        lambda *_args, **_kwargs: queue.RunAudit(True, (), {}),
    )
    wrong = {"metric_name": "private_accuracy"}
    _write_json(tmp_path / "final_retrain_result.json", {"private_metrics": {"results": [wrong]}})
    _write_json(
        tmp_path / "private_metrics.json",
        {"history": [{"private_metrics": {"results": [wrong]}}]},
    )
    _write_json(tmp_path / "source_state.json", {"protocol_id": queue.PROTOCOL_ID})
    audit = queue.validate_run_artifacts(
        tmp_path,
        stage="full",
        expected_steps=35,
        expected_model="deepseek-v4-flash",
        expected_metric="coco_map",
        expected_test_rows=3,
        expected_workers=1,
        expected_gpu_uuids=("GPU-1",),
        require_report=False,
    )
    assert not audit.ok
    assert any("private_coco_map" in error for error in audit.errors)


def test_private_claim_is_main9_namespaced_and_one_shot(tmp_path: Path) -> None:
    path = queue.claim_private_evaluation_once(
        resource_root=tmp_path,
        task_id="monuseg",
        repeat_index=1,
        queue_id="main9-q",
        requested_name="main9-run",
        source_state_sha256="a" * 64,
    )
    assert ".biomaker_main9_queue" in str(path)
    with pytest.raises(queue.QueueConfigurationError):
        queue.claim_private_evaluation_once(
            resource_root=tmp_path,
            task_id="monuseg",
            repeat_index=1,
            queue_id="main9-q2",
            requested_name="main9-run2",
            source_state_sha256="b" * 64,
        )


def test_queue_runs_four_step_canary_then_35_step_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path / "registry")
    registry = queue.load_registry(registry_path)
    task = queue.active_tasks(registry, {"monuseg"})[0]
    resource_root = tmp_path / "resource"
    cache_root = resource_root / "split_builder"
    _data_dir, record = _prepared_fixture(tmp_path, task, cache_root=cache_root)
    manifest_path = tmp_path / "suite_manifest.json"
    record_for_json = {key: value for key, value in record.items() if not key.startswith("_")}
    _write_json(manifest_path, {"tasks": [record_for_json]})

    monkeypatch.setenv("BIOMAKER_GPU_UUIDS", "GPU-1,GPU-2")
    monkeypatch.setenv("BIOMAKER_PARALLEL_WORKERS", "2")
    monkeypatch.setenv("BIOMAKER_REPEAT_INDEX", "1")
    monkeypatch.setattr(queue._harness, "validate_runtime", lambda _paths: None)
    monkeypatch.setattr(queue._harness, "validate_gpu_inventory", lambda *_args: None)
    monkeypatch.setattr(queue._harness, "install_signal_handlers", lambda: None)
    queue._harness.STOP_REQUESTED.clear()
    observed: list[tuple[str, int, int]] = []

    def fake_run_one_stage(*, state, task, stage, steps, workers, **_kwargs):
        observed.append((stage, steps, workers))
        task_state = state["tasks"][task["task_id"]]
        task_state.setdefault("stages", {})[stage] = {"status": "succeeded"}
        task_state["status"] = "canary_succeeded" if stage == "canary" else "succeeded"
        return True

    monkeypatch.setattr(queue, "run_one_stage", fake_run_one_stage)
    rc = queue.queue_main(
        [
            "--registry",
            str(registry_path),
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--data-root",
            str(tmp_path / "data"),
            "--logs-root",
            str(tmp_path / "logs"),
            "--workspaces-root",
            str(tmp_path / "workspaces"),
            "--resource-root",
            str(resource_root),
            "--state-dir",
            str(tmp_path / "state"),
            "--prepared-manifest",
            str(manifest_path),
            "--python",
            sys.executable,
            "--only",
            "monuseg",
        ]
    )
    assert rc == 0
    assert observed == [("canary", 4, 2), ("full", 35, 2)]


def test_dangerous_runtime_overrides_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path / "registry")
    monkeypatch.setenv("BIOMAKER_GPU_UUIDS", "GPU-1")
    with pytest.raises(queue.QueueConfigurationError, match="extra-override"):
        queue.queue_main(
            [
                "--registry",
                str(registry_path),
                "--extra-override",
                "final_retrain.enabled=true",
                "--dry-run",
            ]
        )
