"""Blocking CLI bridge to BioMaker's durable local run queue."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from .config import _load_cfg, _validate_exec_config
from .queue_paths import resolve_queue_paths


def _task_body(raw_cfg: Any, launch_cwd: Path) -> str:
    raw_desc = getattr(raw_cfg, "desc_file", None)
    if raw_desc:
        path = Path(str(raw_desc)).expanduser()
        if not path.is_absolute():
            path = launch_cwd / path
        resolved = path.resolve()
        if not resolved.is_file():
            raise ValueError(f"Task description file does not exist: {resolved}")
        return resolved.read_text(encoding="utf-8")
    goal = getattr(raw_cfg, "goal", None)
    task_body = str(goal or "").strip()
    if not task_body:
        raise ValueError("Provide desc_file or goal before submitting a run.")
    return task_body


def _dataset_path(raw_cfg: Any, launch_cwd: Path) -> Path:
    raw_data_dir = getattr(raw_cfg, "data_dir", None)
    if raw_data_dir is None:
        raise ValueError("Provide data_dir before submitting a run.")
    candidate = Path(str(raw_data_dir)).expanduser()
    resolved = (
        (launch_cwd / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    if not resolved.is_dir():
        raise ValueError(f"Dataset directory does not exist: {resolved}")
    return resolved


def _load_queue_types(method_root: Path):
    webui_root = method_root / "apps" / "webui"
    if not (webui_root / "webui" / "runner.py").is_file():
        raise RuntimeError("The durable queue sidecar is not installed.")
    if str(webui_root) not in sys.path:
        sys.path.insert(0, str(webui_root))
    from webui.run_catalog import list_log_runs
    from webui.runner import RunManager
    from webui.settings import Settings

    return RunManager, Settings, list_log_runs


def _settings(raw_cfg: Any, method_root: Path):
    RunManager, Settings, list_log_runs = _load_queue_types(method_root)
    queue_paths = resolve_queue_paths(
        method_root,
        log_dir=raw_cfg.log_dir,
        workspace_dir=raw_cfg.workspace_dir,
    )
    from .env import resolve_env_file

    settings = Settings(
        method_root=method_root,
        workspace_root=queue_paths.workspace_root,
        state_dir=queue_paths.state_dir,
        logs_root=queue_paths.logs_root,
        workspaces_root=queue_paths.workspaces_root,
        experiments_root=queue_paths.experiments_root,
        datasets_root=queue_paths.datasets_root,
        env_file=resolve_env_file(method_root),
        uv_executable=os.environ.get("BIOMAKER_WEBUI_UV", "uv"),
    )
    return RunManager, list_log_runs, settings


def _write_stream(stream: Any, content: str) -> bool:
    try:
        stream.write(content)
        stream.flush()
        return True
    except (BrokenPipeError, OSError, ValueError):
        return False


def _tail_console(path: Path, offset: int, stream: Any) -> tuple[int, bool]:
    try:
        size = path.stat().st_size
        if size < offset:
            offset = 0
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
    except OSError:
        return offset, True
    if not chunk:
        return offset, True
    try:
        target = getattr(stream, "buffer", stream)
        if target is stream:
            content = chunk.decode("utf-8", errors="replace")
            return offset, _write_stream(stream, content)
        target.write(chunk)
        target.flush()
        return offset, True
    except (BrokenPipeError, OSError, ValueError):
        return offset, False


def _exit_code(record: dict[str, Any], *, interrupted_by_user: bool) -> int:
    if interrupted_by_user:
        return 130
    if record.get("status") == "succeeded":
        return 0
    return_code = record.get("return_code")
    if isinstance(return_code, int):
        return 128 + abs(return_code) if return_code < 0 else max(1, return_code)
    if record.get("status") in {"cancelled", "stopped"}:
        return 130
    return 1


def _request_interrupt(manager: Any, run_id: str) -> None:
    """Cancel or stop a run, tolerating a queue-to-running race."""

    for _attempt in range(3):
        current = manager.store.get_record("runs", run_id)
        if current is None:
            return
        status = current.get("status")
        if status in {"succeeded", "failed", "cancelled", "stopped", "interrupted"}:
            return
        if status == "stopping":
            return
        try:
            if status == "queued":
                manager.cancel(run_id)
            elif status == "running":
                manager.stop(run_id)
            return
        except (KeyError, ValueError):
            # The supervisor can claim or finish the run between the read and the
            # mutation. Re-read before deciding whether another action is needed.
            continue


def _close_manager(manager: Any) -> None:
    """Release local queue handles without affecting detached run processes."""

    try:
        manager.shutdown()
    finally:
        manager.store.close()


def submit_cli_and_wait(argv: list[str] | None = None) -> int:
    """Submit the CLI invocation, follow its durable log, and return its code."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    method_root = Path(__file__).resolve().parents[2]
    launch_cwd = Path.cwd().resolve()
    raw_cfg = _load_cfg(use_cli_args=True)
    _validate_exec_config(raw_cfg)
    dataset_path = _dataset_path(raw_cfg, launch_cwd)
    task_body = _task_body(raw_cfg, launch_cwd)
    RunManager, list_log_runs, settings = _settings(raw_cfg, method_root)
    settings.ensure_directories()
    manager = RunManager(settings)
    try:
        manager.import_history(list_log_runs(settings))
        manager.start()

        requested_exp_name = getattr(raw_cfg, "exp_name", None)
        options = {
            "agent_steps": getattr(raw_cfg.agent, "steps", None),
            "exec_timeout": getattr(raw_cfg.exec, "timeout", None),
            "exec_parallel_workers": getattr(raw_cfg.exec, "parallel_workers", None),
            "agent_total_time_limit": getattr(raw_cfg.agent, "total_time_limit", None),
            "final_retrain_enabled": getattr(raw_cfg.final_retrain, "enabled", None),
            "model_for_all_agents": getattr(raw_cfg, "model_for_all_agents", None),
        }
        child_arguments = [
            *arguments,
            f"log_dir={settings.logs_root}",
            f"workspace_dir={settings.workspaces_root}",
        ]
        record = manager.enqueue_command(
            [sys.executable, "-m", "biomaker.run", *child_arguments],
            cwd=launch_cwd,
            project_snapshot={
                "dataset_path": str(dataset_path),
                "task_template_path": (
                    str(getattr(raw_cfg, "desc_file", None))
                    if getattr(raw_cfg, "desc_file", None)
                    else None
                ),
                "task_body": task_body,
            },
            options=options,
            requested_exp_name=(
                str(requested_exp_name) if requested_exp_name else None
            ),
            logs_root=settings.logs_root,
            workspaces_root=settings.workspaces_root,
        )
    except BaseException:
        _close_manager(manager)
        raise
    run_id = str(record["id"])
    console_path = Path(str(record["console_log"]))
    _write_stream(
        sys.stderr,
        f"Queued BioMaker run {run_id}. Durable log: {console_path}\n",
    )

    offset = 0
    stream_open = True
    interrupted_by_user = False
    announced_pause: str | None = None
    terminal_statuses = {"succeeded", "failed", "cancelled", "stopped", "interrupted"}
    try:
        while True:
            if stream_open:
                offset, stream_open = _tail_console(console_path, offset, sys.stdout)
            current = manager.store.get_record("runs", run_id)
            if current is None:
                raise RuntimeError(f"Run disappeared from the durable ledger: {run_id}")
            if current.get("status") in terminal_statuses:
                if stream_open:
                    offset, stream_open = _tail_console(
                        console_path, offset, sys.stdout
                    )
                return _exit_code(current, interrupted_by_user=interrupted_by_user)
            control = manager.store.get_queue_control()
            if current.get("status") == "queued" and control.get("paused"):
                reason = str(control.get("reason") or "an incomplete run")
                if reason != announced_pause:
                    _write_stream(
                        sys.stderr,
                        "Queue paused: "
                        f"{reason}. Resume it in the WebUI to continue.\n",
                    )
                    announced_pause = reason
            time.sleep(0.2)
    except KeyboardInterrupt:
        interrupted_by_user = True
        _request_interrupt(manager, run_id)
        deadline = time.monotonic() + settings.stop_grace_seconds + 5.0
        while time.monotonic() < deadline:
            if stream_open:
                offset, stream_open = _tail_console(console_path, offset, sys.stdout)
            current = manager.store.get_record("runs", run_id)
            if current is not None and current.get("status") in terminal_statuses:
                break
            time.sleep(0.1)
        return 130
    finally:
        _close_manager(manager)
