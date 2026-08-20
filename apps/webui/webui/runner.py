from __future__ import annotations

import os
import queue
import re
import shlex
import signal
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any

from .env_file import subprocess_environment
from .models import AdvancedOptions, AdvancedOptionsPatch
from .registry import JsonRegistry
from .settings import Settings

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "stopped", "interrupted"}
ACTIVE_STATUSES = {"queued", "running", "stopping"}
EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def slugify(value: str, fallback: str = "run") -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return (value or fallback)[:48]


def validate_experiment_name(value: str) -> str:
    value = value.strip()
    if not EXPERIMENT_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "Experiment name must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    return value


def merge_options(
    defaults: dict[str, Any] | None,
    patch: AdvancedOptionsPatch | dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = AdvancedOptions.model_validate(defaults or {}).model_dump()
    if patch is None:
        return options
    if isinstance(patch, AdvancedOptionsPatch):
        changes = patch.model_dump(exclude_unset=True)
    else:
        changes = {key: value for key, value in patch.items() if value is not None}
    options.update({key: value for key, value in changes.items() if value is not None})
    return AdvancedOptions.model_validate(options).model_dump()


def build_command(
    settings: Settings,
    *,
    dataset_path: str,
    task_file: Path,
    requested_exp_name: str,
    options: dict[str, Any],
) -> list[str]:
    validated = AdvancedOptions.model_validate(options)
    command = [
        settings.uv_executable,
        "run",
        "biomaker",
        f"data_dir={dataset_path}",
        f"desc_file={task_file}",
        f"log_dir={settings.logs_root}",
        f"workspace_dir={settings.workspaces_root}",
        f"exp_name={validate_experiment_name(requested_exp_name)}",
        f"agent.steps={validated.agent_steps}",
        f"exec.timeout={validated.exec_timeout}",
        f"exec.parallel_workers={validated.exec_parallel_workers}",
        f"agent.total_time_limit={validated.agent_total_time_limit}",
        f"final_retrain.enabled={str(validated.final_retrain_enabled).lower()}",
    ]
    if validated.model_for_all_agents:
        command.append(f"model_for_all_agents={validated.model_for_all_agents}")
    return command


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class RunManager:
    def __init__(
        self,
        settings: Settings,
        registry: JsonRegistry,
        *,
        process_factory: Callable[..., Any] | None = None,
    ):
        self.settings = settings
        self.registry = registry
        self.process_factory = process_factory or subprocess.Popen
        self._queue: queue.Queue[str] = queue.Queue()
        self._shutdown = Event()
        self._worker: Thread | None = None
        self._lock = RLock()
        self._current_run_id: str | None = None
        self._current_process: Any | None = None
        self._stop_requested: set[str] = set()

    def start(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self.settings.ensure_directories()
            self.recover_incomplete_runs()
            self._shutdown.clear()
            self._worker = Thread(
                target=self._worker_loop, name="biomaker-webui-runner", daemon=True
            )
            self._worker.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        with self._lock:
            process = self._current_process
            run_id = self._current_run_id
            if run_id is not None:
                self._stop_requested.add(run_id)
        if process is not None and run_id is not None:
            self._start_terminator(process)
        worker = self._worker
        if worker is not None:
            worker.join(timeout=self.settings.stop_grace_seconds + 3)
        self.recover_incomplete_runs(reason="WebUI stopped before the run completed")

    def recover_incomplete_runs(
        self, reason: str = "WebUI restarted before the run completed"
    ) -> int:
        now = utc_now()

        def recover(data: dict[str, Any]) -> int:
            count = 0
            for record in data["runs"].values():
                if record.get("status") not in ACTIVE_STATUSES:
                    continue
                record["status"] = "interrupted"
                record["error"] = reason
                record["return_code"] = None
                timestamps = record.setdefault("timestamps", {})
                timestamps["finished_at"] = now
                timestamps["updated_at"] = now
                count += 1
            return count

        _, count = self.registry.mutate(recover)
        return count

    def enqueue_project(
        self,
        project: dict[str, Any],
        *,
        requested_exp_name: str | None = None,
        option_patch: AdvancedOptionsPatch | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = {
            "project_name": project["name"],
            "dataset_path": project["dataset_path"],
            "task_template_path": project.get("task_template_path"),
            "task_body": project["task_body"],
        }
        options = merge_options(project.get("advanced_defaults"), option_patch)
        return self._enqueue_snapshot(
            project_id=project["id"],
            snapshot=snapshot,
            options=options,
            requested_exp_name=requested_exp_name,
        )

    def _enqueue_snapshot(
        self,
        *,
        project_id: str,
        snapshot: dict[str, Any],
        options: dict[str, Any],
        requested_exp_name: str | None,
    ) -> dict[str, Any]:
        dataset_path = Path(str(snapshot["dataset_path"])).expanduser().resolve()
        if not dataset_path.is_dir():
            raise ValueError(f"Dataset directory does not exist: {dataset_path}")
        task_body = str(snapshot.get("task_body", "")).strip()
        if not task_body:
            raise ValueError("Task body must not be blank")

        run_id = uuid.uuid4().hex
        if requested_exp_name:
            exp_name = validate_experiment_name(requested_exp_name)
        else:
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            exp_name = validate_experiment_name(
                f"webui-{slugify(str(snapshot['project_name']))}-{timestamp}-{run_id[:6]}"
            )
        task_file = (self.settings.state_dir / "tasks" / f"{run_id}.md").resolve()
        console_log = (self.settings.state_dir / "console" / f"{run_id}.log").resolve()
        _atomic_write_text(task_file, task_body)
        command = build_command(
            self.settings,
            dataset_path=str(dataset_path),
            task_file=task_file,
            requested_exp_name=exp_name,
            options=options,
        )
        now = utc_now()
        record = {
            "id": run_id,
            "project_id": project_id,
            "project_snapshot": {**snapshot, "dataset_path": str(dataset_path)},
            "status": "queued",
            "requested_exp_name": exp_name,
            "actual_exp_name": None,
            "command": command,
            "options": options,
            "task_file": str(task_file),
            "console_log": str(console_log),
            "log_dir": None,
            "workspace_dir": None,
            "return_code": None,
            "pid": None,
            "error": None,
            "timestamps": {
                "created_at": now,
                "queued_at": now,
                "started_at": None,
                "finished_at": None,
                "updated_at": now,
            },
        }
        self.registry.add_record("runs", record)
        self._queue.put(run_id)
        return record

    def cancel(self, run_id: str) -> dict[str, Any]:
        now = utc_now()

        def cancel_record(record: dict[str, Any]) -> None:
            if record.get("status") != "queued":
                raise ValueError("Only queued runs can be cancelled")
            record["status"] = "cancelled"
            record["timestamps"]["finished_at"] = now
            record["timestamps"]["updated_at"] = now

        return self.registry.update_record("runs", run_id, cancel_record)

    def stop(self, run_id: str) -> dict[str, Any]:
        now = utc_now()

        def request_stop(record: dict[str, Any]) -> None:
            if record.get("status") not in {"running", "stopping"}:
                raise ValueError("Only a running run can be stopped")
            record["status"] = "stopping"
            record["timestamps"]["stop_requested_at"] = now
            record["timestamps"]["updated_at"] = now

        record = self.registry.update_record("runs", run_id, request_stop)
        with self._lock:
            self._stop_requested.add(run_id)
            process = self._current_process if self._current_run_id == run_id else None
        if process is not None:
            self._start_terminator(process)
        return record

    def requeue(self, run_id: str) -> dict[str, Any]:
        source = self.registry.get_record("runs", run_id)
        if source is None:
            raise KeyError(run_id)
        if source.get("status") not in TERMINAL_STATUSES:
            raise ValueError("Only completed runs can be requeued")
        retry_name = (
            f"{str(source['requested_exp_name'])[:62].rstrip('-._')}"
            f"-retry-{uuid.uuid4().hex[:5]}"
        )
        return self._enqueue_snapshot(
            project_id=source["project_id"],
            snapshot=source["project_snapshot"],
            options=source["options"],
            requested_exp_name=retry_name,
        )

    def summary(self) -> dict[str, Any]:
        records = self.registry.list_records("runs")
        with self._lock:
            current = self._current_run_id
        return {
            "active_run_id": current,
            "queued": sum(record.get("status") == "queued" for record in records),
            "running": sum(
                record.get("status") in {"running", "stopping"} for record in records
            ),
        }

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                run_id = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                record = self.registry.get_record("runs", run_id)
                if record and record.get("status") == "queued":
                    try:
                        self._execute_run(record)
                    # This is the queue's supervision boundary: every unexpected
                    # child failure must become a persisted terminal state.
                    except Exception as exc:  # noqa: BLE001
                        self._append_console_error(record, exc)
                        self._finish_run(run_id, "failed", None, str(exc))
            finally:
                with self._lock:
                    if self._current_run_id == run_id:
                        self._current_run_id = None
                        self._current_process = None
                    self._stop_requested.discard(run_id)
                self._queue.task_done()

    def _execute_run(self, record: dict[str, Any]) -> None:
        run_id = record["id"]
        started_at = utc_now()
        started_epoch = time.time()
        baseline = {
            path.name for path in self.settings.logs_root.iterdir() if path.is_dir()
        }
        record = self.registry.update_record(
            "runs",
            run_id,
            {
                "status": "running",
                "timestamps": {
                    **record["timestamps"],
                    "started_at": started_at,
                    "updated_at": started_at,
                },
            },
        )
        console_path = Path(record["console_log"])
        console_path.parent.mkdir(parents=True, exist_ok=True)
        with console_path.open("a", encoding="utf-8") as console:
            console.write(f"$ {shlex.join(record['command'])}\n\n")
            console.flush()
            try:
                kwargs: dict[str, Any] = {
                    "cwd": self.settings.method_root,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                    "bufsize": 1,
                    "env": subprocess_environment(self.settings.env_file),
                }
                if os.name == "posix":
                    kwargs["start_new_session"] = True
                process = self.process_factory(record["command"], **kwargs)
            except Exception as exc:  # noqa: BLE001 - injected process boundary
                console.write(f"Failed to start BioMaker: {exc}\n")
                self._finish_run(run_id, "failed", None, str(exc))
                return

            with self._lock:
                self._current_run_id = run_id
                self._current_process = process
            self.registry.update_record(
                "runs", run_id, {"pid": getattr(process, "pid", None)}
            )
            reader = Thread(
                target=self._capture_output,
                args=(process, console_path),
                name=f"biomaker-console-{run_id[:8]}",
                daemon=True,
            )
            reader.start()
            if run_id in self._stop_requested or self._shutdown.is_set():
                self._start_terminator(process)

            discovered = False
            shutdown_termination_started = False
            while process.poll() is None:
                if not discovered:
                    discovered = self._discover_output_dir(
                        run_id, record["requested_exp_name"], baseline, started_epoch
                    )
                if self._shutdown.is_set():
                    if not shutdown_termination_started:
                        shutdown_termination_started = True
                        self._start_terminator(process)
                    time.sleep(0.1)
                else:
                    self._shutdown.wait(0.35)
            return_code = process.wait()
            reader.join(timeout=2)
            if not discovered:
                self._discover_output_dir(
                    run_id, record["requested_exp_name"], baseline, started_epoch
                )
            console.write(f"\n[BioMaker exited with code {return_code}]\n")
            console.flush()

        current = self.registry.get_record("runs", run_id) or record
        if self._shutdown.is_set():
            status = "interrupted"
            error = "WebUI stopped before the run completed"
        elif current.get("status") == "stopping" or run_id in self._stop_requested:
            status = "stopped"
            error = None
        elif return_code == 0:
            status = "succeeded"
            error = None
        else:
            status = "failed"
            error = f"BioMaker exited with code {return_code}"
        self._finish_run(run_id, status, return_code, error)

    @staticmethod
    def _capture_output(process: Any, console_path: Path) -> None:
        stream = getattr(process, "stdout", None)
        if stream is None:
            return
        try:
            with console_path.open("a", encoding="utf-8") as console:
                for line in stream:
                    line = ANSI_ESCAPE_PATTERN.sub("", str(line)).replace("\r", "")
                    console.write(line)
                    console.flush()
        except (OSError, ValueError) as exc:
            try:
                with console_path.open("a", encoding="utf-8") as console:
                    console.write(f"\n[Console capture ended: {exc}]\n")
            except OSError:
                pass
        finally:
            try:
                stream.close()
            except (OSError, ValueError):
                # The child may close its pipe while the reader is unwinding.
                return

    @staticmethod
    def _append_console_error(record: dict[str, Any], exc: Exception) -> None:
        raw_path = record.get("console_log")
        if not raw_path:
            return
        try:
            with Path(str(raw_path)).open("a", encoding="utf-8") as console:
                console.write(f"\n[Run manager error: {exc}]\n")
        except OSError:
            pass

    def _discover_output_dir(
        self,
        run_id: str,
        requested_exp_name: str,
        baseline: set[str],
        started_epoch: float,
    ) -> bool:
        candidates: list[Path] = []
        try:
            for path in self.settings.logs_root.iterdir():
                if not path.is_dir():
                    continue
                _, separator, suffix = path.name.partition("-")
                if (
                    separator
                    and suffix == requested_exp_name
                    and (
                        path.name not in baseline
                        or path.stat().st_mtime >= started_epoch - 2
                    )
                ):
                    candidates.append(path)
        except OSError:
            return False
        if not candidates:
            return False
        log_dir = max(candidates, key=lambda path: path.stat().st_mtime).resolve()
        actual_name = log_dir.name
        self.registry.update_record(
            "runs",
            run_id,
            {
                "actual_exp_name": actual_name,
                "log_dir": str(log_dir),
                "workspace_dir": str(
                    (self.settings.workspaces_root / actual_name).resolve()
                ),
                "timestamps": {
                    **(self.registry.get_record("runs", run_id) or {}).get(
                        "timestamps", {}
                    ),
                    "updated_at": utc_now(),
                },
            },
        )
        return True

    def _finish_run(
        self,
        run_id: str,
        status: str,
        return_code: int | None,
        error: str | None,
    ) -> None:
        now = utc_now()

        def finish(record: dict[str, Any]) -> None:
            record["status"] = status
            record["return_code"] = return_code
            record["error"] = error
            record["pid"] = None
            record["timestamps"]["finished_at"] = now
            record["timestamps"]["updated_at"] = now

        try:
            self.registry.update_record("runs", run_id, finish)
        except KeyError:
            pass

    def _start_terminator(self, process: Any) -> None:
        Thread(
            target=self._terminate_process,
            args=(process,),
            name="biomaker-process-terminator",
            daemon=True,
        ).start()

    def _terminate_process(self, process: Any) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix" and getattr(process, "pid", None):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
        except (OSError, ProcessLookupError):
            try:
                process.terminate()
            except (AttributeError, OSError):
                return
        deadline = time.monotonic() + self.settings.stop_grace_seconds
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is not None:
            return
        try:
            if os.name == "posix" and getattr(process, "pid", None):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except (AttributeError, OSError):
                # The process can exit between the final poll and kill call.
                return
