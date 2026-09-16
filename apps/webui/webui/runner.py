from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock, Thread
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - BioMaker WebUI targets macOS/Linux.
    fcntl = None  # type: ignore[assignment]

from .env_file import read_env_values, subprocess_environment
from .models import AdvancedOptions, AdvancedOptionsPatch
from .run_store import SQLiteRunStore
from .settings import Settings

_METHOD_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(_METHOD_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_METHOD_PACKAGE_ROOT))

from biomaker.utils.process_identity import (  # noqa: E402
    IdentityVerification,
    ProcessIdentity,
    ProcessIdentityError,
    capture_process_identity,
    read_process_snapshot,
    read_session_process_groups,
    terminate_process_session,
    verify_process_identity as verify_os_process_identity,
)

TERMINAL_STATUSES = {
    "succeeded",
    "failed",
    "cancelled",
    "stopped",
    "interrupted",
}
ACTIVE_STATUSES = {"running", "stopping"}
PAUSING_STATUSES = {"failed", "cancelled", "stopped", "interrupted"}
CLAIMED_LAUNCH_STATE = "claimed"
EXPERIMENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SUPERVISOR_CONFIG_ENV = "BIOMAKER_QUEUE_SUPERVISOR_CONFIG"
RUN_BOOTSTRAP_ENV = "BIOMAKER_QUEUE_RUN_BOOTSTRAP"
COMPUTE_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
LAUNCH_GRACE_SECONDS = 10.0
RUN_BOOTSTRAP_TIMEOUT_SECONDS = 10.0
UNCAPTURED_SESSION_DRAIN_TIMEOUT_SECONDS = 5.0
UNCAPTURED_CLEANUP_RETRY_SECONDS = 0.3
CLEANUP_PENDING_REASON = (
    "Run process cleanup could not be verified. BioMaker is retrying cleanup "
    "before recording a terminal state or starting the next run."
)

ProcessFactory = Callable[..., Any]
IdentityVerifier = Callable[[int, str | None, str | None], bool]

_INLINE_SUPERVISORS: set[str] = set()
_INLINE_SUPERVISORS_LOCK = RLock()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _compute_environment_snapshot(
    environment: Mapping[str, str | None] | None = None,
) -> dict[str, str | None]:
    """Keep per-submission CUDA selection without persisting credentials.

    A missing variable (None) and an empty CUDA mask have different meanings;
    both must override the environment of an already-running supervisor.
    """

    source = os.environ if environment is None else environment
    snapshot = {key: source.get(key) for key in COMPUTE_ENV_KEYS}
    if any(
        value is not None and not isinstance(value, str) for value in snapshot.values()
    ):
        raise ValueError("Compute environment values must be strings or None")
    return snapshot


def _is_claimed_run(record: dict[str, Any]) -> bool:
    return bool(
        record.get("managed", True)
        and record.get("status") == "queued"
        and record.get("launch_state") == CLAIMED_LAUNCH_STATE
    )


def _occupies_fifo(record: dict[str, Any]) -> bool:
    return bool(
        record.get("managed", True)
        and (record.get("status") in ACTIVE_STATUSES or _is_claimed_run(record))
    )


def _pause_generation(control: dict[str, Any]) -> int:
    value = control.get("pause_generation", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _set_queue_paused(
    control: dict[str, Any],
    reason: str,
    *,
    force_generation: bool = False,
) -> int:
    """Pause the queue and advance the CAS generation for a new pause fact."""

    changed = not control.get("paused") or control.get("reason") != reason
    generation = _pause_generation(control)
    if force_generation or changed:
        generation += 1
    control.update(
        {
            "paused": True,
            "reason": reason,
            "pause_generation": generation,
        }
    )
    return generation


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


def resolve_run_options(settings: Settings, options: dict[str, Any]) -> dict[str, Any]:
    resolved = AdvancedOptions.model_validate(options).model_dump()
    if resolved["model_for_all_agents"]:
        return resolved
    configured_models = read_env_values(settings.env_file, ("BIOMAKER_MAIN_MODEL",))
    resolved["model_for_all_agents"] = configured_models.get("BIOMAKER_MAIN_MODEL")
    return resolved


def build_command(
    settings: Settings,
    *,
    dataset_path: str,
    task_file: Path,
    requested_exp_name: str | None,
    options: dict[str, Any],
) -> list[str]:
    validated = AdvancedOptions.model_validate(resolve_run_options(settings, options))
    command = [
        settings.uv_executable,
        "run",
        "biomaker",
        f"data_dir={dataset_path}",
        f"desc_file={task_file}",
        f"log_dir={settings.logs_root}",
        f"workspace_dir={settings.workspaces_root}",
    ]
    if requested_exp_name:
        command.append(f"exp_name={validate_experiment_name(requested_exp_name)}")
    command.extend(
        [
            f"agent.steps={validated.agent_steps}",
            f"exec.timeout={validated.exec_timeout}",
            f"exec.parallel_workers={validated.exec_parallel_workers}",
            f"agent.total_time_limit={validated.agent_total_time_limit}",
            f"final_retrain.enabled={str(validated.final_retrain_enabled).lower()}",
        ]
    )
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


def verify_process_identity(
    pid: int, process_started_at: str | None, launch_token: str | None
) -> bool:
    """Compatibility adapter backed by the shared OS identity reader."""

    if pid <= 1 or not process_started_at or not launch_token:
        return False
    try:
        snapshot = read_process_snapshot(pid)
    except (OSError, ProcessLookupError, ValueError):
        return False
    return snapshot.pid == pid and snapshot.start_time == process_started_at


def _settings_payload(settings: Settings) -> dict[str, Any]:
    return {
        "method_root": str(settings.method_root),
        "workspace_root": str(settings.workspace_root),
        "state_dir": str(settings.state_dir),
        "logs_root": str(settings.logs_root),
        "workspaces_root": str(settings.workspaces_root),
        "experiments_root": str(settings.experiments_root),
        "datasets_root": str(settings.datasets_root),
        "env_file": str(settings.env_file),
        "uv_executable": settings.uv_executable,
        "stop_grace_seconds": settings.stop_grace_seconds,
    }


def _settings_from_payload(payload: dict[str, Any]) -> Settings:
    return Settings(
        method_root=Path(payload["method_root"]),
        workspace_root=Path(payload["workspace_root"]),
        state_dir=Path(payload["state_dir"]),
        logs_root=Path(payload["logs_root"]),
        workspaces_root=Path(payload["workspaces_root"]),
        experiments_root=Path(payload["experiments_root"]),
        datasets_root=Path(payload["datasets_root"]),
        env_file=Path(payload["env_file"]),
        uv_executable=str(payload["uv_executable"]),
        stop_grace_seconds=float(payload["stop_grace_seconds"]),
    )


def _record_bootstrap_failure(
    database_path: str,
    run_id: str,
    launch_token: str,
    error: str,
) -> None:
    """Persist a pre-exec failure so recovery never guesses it was an interrupt."""

    store = SQLiteRunStore(database_path)
    try:

        def fail(record: dict[str, Any]) -> None:
            if record.get("launch_token") != launch_token:
                return
            if record.get("status") in TERMINAL_STATUSES:
                return
            record["launch_state"] = "failed"
            record["launch_error"] = error
            record.setdefault("timestamps", {})["updated_at"] = utc_now()

        try:
            store.update_record("runs", run_id, fail)
        except KeyError:
            return
    finally:
        store.close()


def _run_bootstrap_main(database_path: str, run_id: str, launch_token: str) -> int:
    """Register this detached process before replacing it with the run command."""

    store = SQLiteRunStore(database_path)
    command: list[str] | None = None
    working_directory: str | None = None
    should_exec = False
    was_cancelled = False
    try:
        identity = capture_process_identity(os.getpid(), launch_token)

        def register(data: dict[str, Any]) -> None:
            nonlocal command, working_directory, should_exec, was_cancelled
            record = data["runs"].get(run_id)
            if record is None:
                raise RuntimeError(f"Run {run_id} no longer exists")
            if not record.get("managed", True):
                raise RuntimeError("Historical runs cannot be launched")
            if record.get("launch_token") != launch_token:
                raise RuntimeError("Run launch token changed before bootstrap")
            if record.get("status") == "cancelled":
                was_cancelled = True
                return
            if not _is_claimed_run(record):
                raise RuntimeError(
                    f"Run is no longer launchable: {record.get('status')}"
                )
            recorded_pid = record.get("pid")
            if recorded_pid not in {None, identity.pid}:
                raise RuntimeError("Another process already owns this run")
            raw_command = record.get("command")
            if (
                not isinstance(raw_command, list)
                or not raw_command
                or any(not isinstance(part, str) or not part for part in raw_command)
            ):
                raise RuntimeError("Run command is invalid")
            command = list(raw_command)
            working_directory = str(record.get("cwd") or "")
            now = utc_now()
            record["pid"] = identity.pid
            record["pgid"] = identity.pgid
            record["process_started_at"] = identity.start_time
            record["launch_state"] = "registered"
            record["launch_error"] = None
            record["status"] = "running"
            timestamps = record.setdefault("timestamps", {})
            timestamps["started_at"] = now
            timestamps["updated_at"] = now
            should_exec = True

        store.mutate_state(register)
    except Exception as exc:  # noqa: BLE001 - bootstrap must leave durable evidence
        error = f"Run bootstrap failed: {type(exc).__name__}: {exc}"
        try:
            _record_bootstrap_failure(database_path, run_id, launch_token, error)
        except Exception:
            pass
        print(error, file=sys.stderr, flush=True)
        return 126
    finally:
        store.close()

    if was_cancelled:
        return 130
    if not should_exec:
        return 130
    assert command is not None
    try:
        if working_directory:
            os.chdir(working_directory)
        environment = os.environ.copy()
        environment.pop(RUN_BOOTSTRAP_ENV, None)
        os.execvpe(command[0], command, environment)
    except Exception as exc:  # noqa: BLE001 - os.exec boundary
        error = f"Run exec failed: {type(exc).__name__}: {exc}"
        try:
            _record_bootstrap_failure(database_path, run_id, launch_token, error)
        except Exception:
            pass
        print(error, file=sys.stderr, flush=True)
        return 127


class RunManager:
    """Durable FIFO submission facade and temporary queue supervisor."""

    def __init__(
        self,
        settings: Settings,
        store: SQLiteRunStore | None = None,
        *,
        process_factory: ProcessFactory | None = None,
        identity_verifier: IdentityVerifier | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or SQLiteRunStore(settings.state_dir / "runs.sqlite3")
        self.process_factory = process_factory or subprocess.Popen
        self.identity_verifier = identity_verifier or verify_process_identity
        self._identity_verifier_injected = identity_verifier is not None
        self._uses_injected_process_factory = process_factory is not None
        self._lock = RLock()
        self._local_processes: dict[str, tuple[str, Any]] = {}

    def start(self) -> None:
        self.settings.ensure_directories()
        self.recover_active_runs()
        self._ensure_supervisor()

    def shutdown(self) -> None:
        """Detach the WebUI without changing queue or child process state."""

    def recover_active_runs(
        self, reason: str = "Run process exited without a terminal lifecycle event"
    ) -> int:
        """Reconcile stale identities while preserving every verified process."""

        self._reconcile_runner_identity()
        count = 0
        for record in self.store.list_records("runs"):
            if not record.get("managed", True):
                continue
            if _is_claimed_run(record):
                if record.get("cleanup_pending"):
                    self._mark_uncaptured_run_cleanup_pending(
                        str(record["id"]),
                        str(record.get("cleanup_error") or "cleanup owner is retrying"),
                    )
                    continue
                if self._claimed_launch_in_progress(record):
                    continue
                error = (
                    "Queue supervisor exited while launching the run, before its "
                    "process identity was registered"
                )
                self._finish_run(str(record["id"]), "interrupted", None, error, None)
                count += 1
                continue
            if record.get("status") not in ACTIVE_STATUSES:
                continue
            identity_state = self._record_identity_state(record)
            if identity_state == "match":
                self._pause_for_manifest_safety(record)
                continue
            if identity_state == "unreadable" or self._launch_in_progress(record):
                continue
            if identity_state not in {"missing", "mismatch"}:
                continue
            leader_identity = self._identity_from_record(record)
            if record.get("cleanup_pending") and leader_identity is None:
                # The launch supervisor still owns an unreaped Popen whose
                # identity never became durable. A second manager cannot prove
                # that session empty and must not synthesize a terminal state.
                self._mark_uncaptured_run_cleanup_pending(
                    str(record["id"]),
                    str(record.get("cleanup_error") or "cleanup owner is retrying"),
                )
                continue
            if (
                leader_identity is not None
                and (
                    record.get("cleanup_pending")
                    or (
                        identity_state == "missing"
                        and not self._uses_injected_process_factory
                    )
                )
                and not self._cleanup_before_terminal(record, leader_identity)
            ):
                # A missing session leader can still leave candidate/DataLoader/GPU
                # groups behind. Session cleanup accepts only the exact identity or
                # a missing leader; PID reuse and mismatches remain fail-closed.
                continue
            status, error = self._settle_dead_run(record, reason)
            self._finish_run(
                str(record["id"]),
                status,
                record.get("return_code"),
                error,
                record.get("exit_signal"),
            )
            count += 1
        return count

    def reconcile(self) -> int:
        return self.recover_active_runs()

    def enqueue_experiment(
        self,
        experiment: dict[str, Any],
        *,
        requested_exp_name: str | None = None,
        option_patch: AdvancedOptionsPatch | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot = {
            "dataset_path": experiment["dataset_path"],
            "task_template_path": experiment.get("task_template_path"),
            "task_body": experiment["task_body"],
        }
        options = merge_options(experiment.get("advanced_defaults"), option_patch)
        return self._enqueue_snapshot(
            snapshot=snapshot,
            options=options,
            requested_exp_name=requested_exp_name,
        )

    def import_log_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Idempotently add one read-only historical log record."""

        candidate = json.loads(json.dumps(record, allow_nan=False))
        candidate["managed"] = False
        record_id = str(candidate["id"])
        candidate["id"] = record_id
        launch_token = candidate.get("launch_token")
        raw_log_dir = candidate.get("log_dir")
        log_dir = (
            str(Path(str(raw_log_dir)).expanduser().resolve()) if raw_log_dir else None
        )
        if log_dir:
            candidate["log_dir"] = log_dir
            candidate.setdefault("logs_root", str(Path(log_dir).parent))
        raw_workspace_dir = candidate.get("workspace_dir")
        if raw_workspace_dir:
            workspace_dir = Path(str(raw_workspace_dir)).expanduser().resolve()
            candidate["workspace_dir"] = str(workspace_dir)
            candidate.setdefault("workspaces_root", str(workspace_dir.parent))

        def import_once(data: dict[str, Any]) -> dict[str, Any]:
            def refresh_unmanaged(
                existing_id: str, existing: dict[str, Any]
            ) -> dict[str, Any]:
                if existing.get("managed", True):
                    return existing
                refreshed = {**existing, **candidate}
                refreshed["id"] = existing_id
                refreshed["managed"] = False
                data["runs"][existing_id] = refreshed
                return refreshed

            existing = data["runs"].get(record_id)
            if existing is not None:
                return refresh_unmanaged(record_id, existing)
            if launch_token:
                for current in data["runs"].values():
                    if (
                        current.get("managed", True)
                        and current.get("launch_token") == launch_token
                    ):
                        return current
            if log_dir:
                for current in data["runs"].values():
                    current_log_dir = current.get("log_dir")
                    if (
                        current_log_dir
                        and str(Path(str(current_log_dir)).expanduser().resolve())
                        == log_dir
                    ):
                        return refresh_unmanaged(str(current["id"]), current)
            data["runs"][record_id] = candidate
            return candidate

        _, imported = self.store.mutate(import_once)
        return imported

    def import_history(self, records: list[dict[str, Any]]) -> int:
        before = len(self.store.list_records("runs"))
        for record in records:
            self.import_log_record(record)
        return len(self.store.list_records("runs")) - before

    def _enqueue_snapshot(
        self,
        *,
        snapshot: dict[str, Any],
        options: dict[str, Any],
        requested_exp_name: str | None,
        compute_environment: Mapping[str, str | None] | None = None,
    ) -> dict[str, Any]:
        dataset_path = Path(str(snapshot["dataset_path"])).expanduser().resolve()
        if not dataset_path.is_dir():
            raise ValueError(f"Dataset directory does not exist: {dataset_path}")
        task_body = str(snapshot.get("task_body", "")).strip()
        if not task_body:
            raise ValueError("Task body must not be blank")

        exp_name = (
            validate_experiment_name(requested_exp_name) if requested_exp_name else None
        )
        run_id = uuid.uuid4().hex
        task_file = (self.settings.state_dir / "tasks" / f"{run_id}.md").resolve()
        _atomic_write_text(task_file, task_body)
        options = resolve_run_options(self.settings, options)
        command = build_command(
            self.settings,
            dataset_path=str(dataset_path),
            task_file=task_file,
            requested_exp_name=exp_name,
            options=options,
        )
        return self._add_queued_run(
            run_id=run_id,
            command=command,
            cwd=self.settings.method_root,
            project_snapshot={**snapshot, "dataset_path": str(dataset_path)},
            options=options,
            requested_exp_name=exp_name,
            task_file=task_file,
            logs_root=self.settings.logs_root,
            workspaces_root=self.settings.workspaces_root,
            compute_environment=compute_environment,
        )

    def enqueue_command(
        self,
        command: list[str],
        *,
        cwd: Path | str,
        project_snapshot: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
        requested_exp_name: str | None = None,
        logs_root: Path | str | None = None,
        workspaces_root: Path | str | None = None,
        compute_environment: Mapping[str, str | None] | None = None,
    ) -> dict[str, Any]:
        """Submit a CLI command through the same durable FIFO as the WebUI."""

        if not command or any(
            not isinstance(part, str) or not part for part in command
        ):
            raise ValueError("command must be a non-empty argument list")
        working_directory = Path(cwd).expanduser().resolve()
        if not working_directory.is_dir():
            raise ValueError(f"Working directory does not exist: {working_directory}")
        exp_name = (
            validate_experiment_name(requested_exp_name) if requested_exp_name else None
        )
        return self._add_queued_run(
            run_id=uuid.uuid4().hex,
            command=list(command),
            cwd=working_directory,
            project_snapshot=project_snapshot or {},
            options=options or {},
            requested_exp_name=exp_name,
            task_file=None,
            logs_root=logs_root or self.settings.logs_root,
            workspaces_root=workspaces_root or self.settings.workspaces_root,
            compute_environment=compute_environment,
        )

    def _add_queued_run(
        self,
        *,
        run_id: str,
        command: list[str],
        cwd: Path,
        project_snapshot: dict[str, Any],
        options: dict[str, Any],
        requested_exp_name: str | None,
        task_file: Path | None,
        logs_root: Path | str,
        workspaces_root: Path | str,
        compute_environment: Mapping[str, str | None] | None = None,
    ) -> dict[str, Any]:
        console_log = (self.settings.state_dir / "console" / f"{run_id}.log").resolve()
        now = utc_now()
        record = {
            "id": run_id,
            "managed": True,
            "project_snapshot": project_snapshot,
            "status": "queued",
            "requested_exp_name": requested_exp_name,
            "actual_exp_name": None,
            "command": command,
            "cwd": str(cwd),
            "options": options,
            "compute_environment": _compute_environment_snapshot(compute_environment),
            "task_file": str(task_file) if task_file is not None else None,
            "console_log": str(console_log),
            "logs_root": str(Path(logs_root).expanduser().resolve()),
            "workspaces_root": str(Path(workspaces_root).expanduser().resolve()),
            "log_dir": None,
            "workspace_dir": None,
            "return_code": None,
            "exit_signal": None,
            "pid": None,
            "pgid": None,
            "process_started_at": None,
            "launch_token": run_id,
            "launch_state": "pending",
            "launch_error": None,
            "cleanup_pending": False,
            "cleanup_error": None,
            "supervisor_token": None,
            "error": None,
            "timestamps": {
                "created_at": now,
                "queued_at": now,
                "started_at": None,
                "finished_at": None,
                "updated_at": now,
            },
        }
        saved = self.store.add_record("runs", record)
        self._ensure_supervisor()
        return saved

    def cancel(self, run_id: str) -> dict[str, Any]:
        now = utc_now()

        def cancel_record(data: dict[str, Any]) -> dict[str, Any]:
            record = data["runs"].get(run_id)
            if record is None:
                raise KeyError(run_id)
            if record.get("status") != "queued":
                raise ValueError("Only queued runs can be cancelled")
            record["status"] = "cancelled"
            timestamps = record.setdefault("timestamps", {})
            timestamps["finished_at"] = now
            timestamps["updated_at"] = now
            run_name = record.get("requested_exp_name") or record.get("actual_exp_name")
            reason = (
                f"{run_name} was cancelled before it started"
                if run_name
                else "A queued run was cancelled before it started"
            )
            _set_queue_paused(
                data["queue_control"],
                reason,
                force_generation=True,
            )
            return record

        _, cancelled = self.store.mutate_state(cancel_record)
        return cancelled

    def stop(self, run_id: str) -> dict[str, Any]:
        record = self.store.get_record("runs", run_id)
        if record is None:
            raise KeyError(run_id)
        if record.get("status") != "running":
            raise ValueError("Only a running run can be stopped")

        identity_state = self._record_identity_state(record)
        if identity_state == "match":
            manifest_state = self._manifest_launch_state(record)
            if manifest_state != "match":
                self._pause_for_manifest_safety(record, manifest_state)
                raise ValueError(self._manifest_safety_reason(manifest_state))
        else:
            if self._launch_in_progress(record):
                return self._mark_stopping(run_id)
            if identity_state == "unreadable":
                raise ValueError(
                    "Cannot safely inspect this run's process identity; retry Stop"
                )
            record = self._mark_stopping(run_id)
            leader_identity = self._identity_from_record(record)
            if (
                identity_state == "missing"
                and leader_identity is not None
                and not self._uses_injected_process_factory
                and not self._cleanup_before_terminal(record, leader_identity)
            ):
                self._ensure_supervisor()
                return self.store.get_record("runs", run_id) or record
            status, error = self._settle_dead_run(
                record,
                "Run process was already gone when Stop was requested",
                return_code=record.get("return_code"),
            )
            self._finish_run(
                run_id,
                status,
                record.get("return_code"),
                error,
                record.get("exit_signal"),
            )
            return self.store.get_record("runs", run_id) or record

        updated = self._mark_stopping(run_id)
        Thread(
            target=self._terminate_record,
            args=(run_id,),
            name=f"biomaker-stop-{run_id[:8]}",
            daemon=True,
        ).start()
        return updated

    def _mark_stopping(self, run_id: str) -> dict[str, Any]:
        now = utc_now()

        def request_stop(record: dict[str, Any]) -> None:
            if record.get("status") != "running":
                raise ValueError("Only a running run can be stopped")
            record["status"] = "stopping"
            timestamps = record.setdefault("timestamps", {})
            timestamps["stop_requested_at"] = now
            timestamps["updated_at"] = now

        return self.store.update_record("runs", run_id, request_stop)

    def resume_queue(self, expected_pause_generation: int) -> dict[str, Any]:
        if (
            isinstance(expected_pause_generation, bool)
            or not isinstance(expected_pause_generation, int)
            or expected_pause_generation < 0
        ):
            raise ValueError("A non-negative pause generation is required")
        control = self.store.get_queue_control()
        if _pause_generation(control) != expected_pause_generation:
            raise ValueError("Queue pause changed; refresh before resuming")
        self.recover_active_runs()
        if self._active_manifest_safety_issue():
            control = self.store.get_queue_control()
            if _pause_generation(control) != expected_pause_generation:
                raise ValueError("Queue pause changed; refresh before resuming")
            raise ValueError(str(control.get("reason") or "Queue safety issue remains"))

        def resume(data: dict[str, Any]) -> None:
            control = data["queue_control"]
            if _pause_generation(control) != expected_pause_generation:
                raise ValueError("Queue pause changed; refresh before resuming")
            if any(
                _occupies_fifo(record) and record.get("cleanup_pending")
                for record in data["runs"].values()
            ):
                raise ValueError(CLEANUP_PENDING_REASON)
            control.update({"paused": False, "reason": None})

        self.store.mutate_state(resume)
        self._ensure_supervisor()
        return self.store.get_queue_control()

    def requeue(self, run_id: str) -> dict[str, Any]:
        source = self.store.get_record("runs", run_id)
        if source is None:
            raise KeyError(run_id)
        if not source.get("managed", True):
            raise ValueError("Historical runs are view-only")
        if source.get("status") not in TERMINAL_STATUSES:
            raise ValueError("Only completed runs can be requeued")
        retry_name = source.get("requested_exp_name")
        if not retry_name:
            _, separator, suffix = str(source.get("actual_exp_name") or "").partition(
                "-"
            )
            retry_name = suffix if separator else None
        if source.get("task_file") is None:
            return self.enqueue_command(
                list(source.get("command") or []),
                cwd=source.get("cwd") or self.settings.method_root,
                project_snapshot=source.get("project_snapshot") or {},
                options=source.get("options") or {},
                requested_exp_name=retry_name,
                logs_root=source.get("logs_root") or self.settings.logs_root,
                workspaces_root=(
                    source.get("workspaces_root") or self.settings.workspaces_root
                ),
                compute_environment=source.get("compute_environment"),
            )
        return self._enqueue_snapshot(
            snapshot=source["project_snapshot"],
            options=source["options"],
            requested_exp_name=retry_name,
            compute_environment=source.get("compute_environment"),
        )

    def summary(self) -> dict[str, Any]:
        records = self.store.list_records("runs")
        control = self.store.get_queue_control()
        active = next(
            (
                str(record["id"])
                for record in records
                if record.get("managed", True)
                and record.get("status") in ACTIVE_STATUSES
            ),
            None,
        )
        return {
            "active_run_id": active,
            "queued": sum(
                record.get("managed", True) and record.get("status") == "queued"
                for record in records
            ),
            "running": sum(
                record.get("managed", True) and record.get("status") in ACTIVE_STATUSES
                for record in records
            ),
            "paused": bool(control.get("paused")),
            "pause_reason": control.get("reason"),
        }

    def _ensure_supervisor(self) -> None:
        if not self._needs_supervisor():
            return
        token = uuid.uuid4().hex
        if self._should_run_inline():
            if not self._claim_supervisor(token):
                return
            thread = Thread(
                target=self._supervisor_loop,
                args=(token,),
                name="biomaker-temporary-queue",
                daemon=True,
            )
            thread.start()
            return
        control = self.store.get_queue_control()
        if self._runner_identity_state(control) in {"match", "unreadable"}:
            return
        try:
            self._launch_detached_supervisor(token)
        except Exception as exc:  # noqa: BLE001 - process launch boundary
            self._pause_for_supervisor_start_failure(
                f"Unable to start queue supervisor: {type(exc).__name__}: {exc}"
            )

    def _should_run_inline(self) -> bool:
        return (
            self._uses_injected_process_factory
            or self.store.database_path == ":memory:"
        )

    def _needs_supervisor(self) -> bool:
        records = self.store.list_records("runs")
        has_active = any(_occupies_fifo(record) for record in records)
        has_queued = any(
            record.get("managed", True) and record.get("status") == "queued"
            for record in records
        )
        control = self.store.get_queue_control()
        return has_active or (has_queued and not control.get("paused"))

    def _claim_supervisor(self, token: str) -> bool:
        try:
            identity = capture_process_identity(os.getpid(), token)
        except (
            OSError,
            ProcessLookupError,
            ProcessIdentityError,
            ValueError,
        ) as exc:
            self._pause_for_supervisor_start_failure(
                "Unable to register queue supervisor identity: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        inline = self._should_run_inline()
        if inline:
            # Test/embedded supervisors run in a thread in this process. Publish
            # the token locally before the transaction so another manager in the
            # same process cannot replace the exact claim before the thread starts.
            with _INLINE_SUPERVISORS_LOCK:
                _INLINE_SUPERVISORS.add(token)
        claimed = False
        now = utc_now()

        def claim(data: dict[str, Any]) -> None:
            nonlocal claimed
            records = data["runs"].values()
            control = data["queue_control"]
            has_active = any(_occupies_fifo(record) for record in records)
            has_queued = any(
                record.get("managed", True) and record.get("status") == "queued"
                for record in data["runs"].values()
            )
            if not has_active and (not has_queued or control.get("paused")):
                return
            if self._runner_identity_state(control) in {
                "match",
                "unreadable",
            }:
                return
            control.update(
                {
                    "runner_pid": identity.pid,
                    "runner_pgid": identity.pgid,
                    "runner_process_started_at": identity.start_time,
                    "runner_token": token,
                    "runner_claimed_at": now,
                }
            )
            claimed = True

        try:
            self.store.mutate_state(claim)
        finally:
            if inline and not claimed:
                with _INLINE_SUPERVISORS_LOCK:
                    _INLINE_SUPERVISORS.discard(token)
        return claimed

    def _pause_for_supervisor_start_failure(self, reason: str) -> None:
        """Pause only when no verified supervisor can own the pending work."""

        def pause(data: dict[str, Any]) -> None:
            records = data["runs"].values()
            control = data["queue_control"]
            if self._runner_identity_state(control) in {"match", "unreadable"}:
                return
            has_active = any(_occupies_fifo(record) for record in records)
            has_queued = any(
                record.get("managed", True) and record.get("status") == "queued"
                for record in data["runs"].values()
            )
            if not has_active and not has_queued:
                return
            _set_queue_paused(control, reason)

        self.store.mutate_state(pause)

    def _launch_detached_supervisor(self, token: str) -> None:
        supervisor_log = self.settings.state_dir / "queue-supervisor.log"
        supervisor_log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment[SUPERVISOR_CONFIG_ENV] = json.dumps(
            _settings_payload(self.settings), separators=(",", ":")
        )
        command = [
            sys.executable,
            "-m",
            "webui.runner",
            "--supervise",
            self.store.database_path,
            token,
        ]
        with supervisor_log.open("ab", buffering=0) as console:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.DEVNULL,
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=environment,
            )
        Thread(
            target=self._reap_detached_supervisor,
            args=(process, token),
            name="biomaker-queue-reaper",
            daemon=True,
        ).start()

    def _reap_detached_supervisor(self, process: Any, token: str) -> None:
        """Recover an unexpectedly lost temporary supervisor while this UI lives."""

        try:
            return_code = process.wait()
            control = self.store.get_queue_control()
            owned = control.get("runner_token") == token
            if owned:
                self._release_supervisor(token)
            if not owned and return_code == 0:
                return
            self._ensure_supervisor()
        except (OSError, RuntimeError):
            # The owning WebUI may have closed its store while this daemon reaper
            # was waiting. Shutdown must remain non-destructive and quiet.
            return

    def _supervisor_loop(self, token: str) -> None:
        try:
            control = self.store.get_queue_control()
            if (
                control.get("runner_token") != token
                or control.get("runner_pid") != os.getpid()
                or self._runner_identity_state(control) != "match"
            ):
                return
            # A replacement supervisor must settle an abandoned claimed launch
            # before it can consider any later FIFO item.
            self.recover_active_runs()
            while True:
                active = self._first_active_record()
                if active is not None:
                    self._monitor_external_run(active)
                    continue
                if self.store.get_queue_control().get("paused"):
                    return
                record = self._claim_next_run(token)
                if record is not None:
                    self._execute_claimed_run(record)
                    continue

                self._release_supervisor(token)
                if not self._needs_supervisor() or not self._claim_supervisor(token):
                    return
        finally:
            self._release_supervisor(token)

    def _first_active_record(self) -> dict[str, Any] | None:
        return next(
            (
                record
                for record in self.store.list_records("runs")
                if record.get("managed", True)
                and record.get("status") in ACTIVE_STATUSES
            ),
            None,
        )

    def _claim_next_run(self, supervisor_token: str) -> dict[str, Any] | None:
        now = utc_now()

        def claim(data: dict[str, Any]) -> dict[str, Any] | None:
            control = data["queue_control"]
            if (
                control.get("paused")
                or control.get("runner_token") != supervisor_token
                or control.get("runner_pid") != os.getpid()
                or self._runner_identity_state(control) != "match"
            ):
                return None
            if any(_occupies_fifo(current) for current in data["runs"].values()):
                return None
            for record in data["runs"].values():
                if not record.get("managed", True) or record.get("status") != "queued":
                    continue
                record["launch_state"] = CLAIMED_LAUNCH_STATE
                record["supervisor_token"] = supervisor_token
                timestamps = record.setdefault("timestamps", {})
                timestamps["claimed_at"] = now
                timestamps["updated_at"] = now
                return record
            return None

        _, record = self.store.mutate_state(claim)
        return record

    @staticmethod
    def _drain_uncaptured_detached_process(process: Any) -> None:
        """Drain a just-created session while ``Popen`` still reserves its SID.

        Identity capture can fail after the bootstrap has already exec'd BioMaker.
        By then candidates or DataLoaders may lead process groups other than the
        bootstrap's original PGID.  Reaping the session leader before those groups
        are gone would release its numeric SID and make further cleanup unsafe.
        """

        pid = getattr(process, "pid", None)
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            raise RuntimeError("Detached process has no safe session id")
        if os.name != "posix":
            try:
                process.kill()
                process.wait(timeout=UNCAPTURED_SESSION_DRAIN_TIMEOUT_SECONDS)
            except (AttributeError, OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(
                    "Could not confirm that the detached process exited"
                ) from exc
            return

        own_pgid = os.getpgrp()
        deadline = time.monotonic() + UNCAPTURED_SESSION_DRAIN_TIMEOUT_SECONDS
        empty_observations = 0
        last_inspection_error: BaseException | None = None
        while empty_observations < 2:
            try:
                groups = read_session_process_groups(pid)
                last_inspection_error = None
            except (OSError, ProcessIdentityError, ValueError) as exc:
                groups = set()
                last_inspection_error = exc
                empty_observations = 0
            else:
                unsafe_groups = {
                    pgid for pgid in groups if pgid <= 1 or pgid == own_pgid
                }
                if unsafe_groups:
                    raise RuntimeError(
                        "Detached session overlaps an unsafe process group: "
                        f"{sorted(unsafe_groups)}"
                    )
                if groups:
                    empty_observations = 0
                    # Kill non-root groups first so the bootstrap cannot keep
                    # launching work after its already-discovered children die.
                    for pgid in sorted(groups, key=lambda value: (value == pid, value)):
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            continue
                        except OSError as exc:
                            raise RuntimeError(
                                f"Could not kill detached process group {pgid}"
                            ) from exc
                else:
                    empty_observations += 1

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = (
                    f": {last_inspection_error}"
                    if last_inspection_error is not None
                    else ""
                )
                raise RuntimeError(
                    "Could not confirm that the detached process session exited"
                    + detail
                )
            time.sleep(min(0.05, remaining))

        # Only now is it safe to reap the leader and release its numeric PID/SID.
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except (AttributeError, OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Detached session was empty but its leader could not be reaped"
            ) from exc

    def _run_process_spec(
        self,
        record: dict[str, Any],
    ) -> tuple[list[str], Path, dict[str, str]]:
        environment = {
            **subprocess_environment(self.settings.env_file),
            "BIOMAKER_LAUNCH_TOKEN": str(record["launch_token"]),
            "BIOMAKER_QUEUE_DIRECT": "1",
        }
        # Older records predate per-run CUDA snapshots and retain their original
        # inheritance behavior. New submissions must not inherit another run's
        # GPU mask merely because that run started the shared supervisor first.
        if record.get("compute_environment") is not None:
            snapshot = _compute_environment_snapshot(record["compute_environment"])
            for key, value in snapshot.items():
                if value is None:
                    environment.pop(key, None)
                else:
                    environment[key] = value
        if self._uses_injected_process_factory:
            return (
                list(record["command"]),
                Path(str(record.get("cwd") or self.settings.method_root)),
                environment,
            )
        environment[RUN_BOOTSTRAP_ENV] = "1"
        return (
            [
                sys.executable,
                "-m",
                "webui.runner",
                "--launch-run",
                self.store.database_path,
                str(record["id"]),
                str(record["launch_token"]),
            ],
            Path(__file__).resolve().parents[1],
            environment,
        )

    def _await_bootstrap_identity(
        self,
        record: dict[str, Any],
        process: Any,
    ) -> tuple[dict[str, Any], ProcessIdentity]:
        pid = getattr(process, "pid", None)
        if not isinstance(pid, int) or pid <= 1:
            raise RuntimeError("Run bootstrap returned no valid process id")
        deadline = time.monotonic() + RUN_BOOTSTRAP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            current = self.store.get_record("runs", str(record["id"])) or record
            if current.get("status") == "cancelled":
                raise RuntimeError("Run was cancelled before bootstrap registration")
            identity = self._identity_from_record(current)
            if identity is not None:
                if identity.pid != pid:
                    raise RuntimeError(
                        "Run bootstrap registered an unexpected process id"
                    )
                verification = verify_os_process_identity(
                    identity,
                    launch_token=str(record["launch_token"]),
                )
                if verification.matched:
                    return current, identity
                if verification.reason not in {"missing", "unreadable"}:
                    raise RuntimeError(
                        "Run bootstrap registered an invalid identity: "
                        f"{verification.reason}"
                    )
            # poll()/wait() reap a dead child and release its numeric PID/SID.
            # Until identity is registered, the Popen owner must retain that SID
            # so emergency session cleanup cannot be redirected by PID reuse.
            return_code = getattr(process, "returncode", None)
            if return_code is not None:
                error = current.get("launch_error") or (
                    f"Run bootstrap exited with code {return_code} before registration"
                )
                raise RuntimeError(str(error))
            time.sleep(0.02)
        raise RuntimeError("Timed out waiting for run bootstrap registration")

    def _execute_claimed_run(self, record: dict[str, Any]) -> None:
        run_id = str(record["id"])
        try:
            self._execute_claimed_run_inner(record)
        finally:
            with self._lock:
                self._local_processes.pop(run_id, None)

    def _execute_claimed_run_inner(self, record: dict[str, Any]) -> None:
        run_id = str(record["id"])
        console_path = Path(record["console_log"])
        console_path.parent.mkdir(parents=True, exist_ok=True)
        process: Any | None = None
        leader_identity: ProcessIdentity | None = None
        with console_path.open("a", encoding="utf-8", errors="replace") as console:
            console.write(f"$ {shlex.join(record['command'])}\n\n")
            console.flush()
            try:
                command, working_directory, environment = self._run_process_spec(record)
                kwargs: dict[str, Any] = {
                    "cwd": working_directory,
                    "stdin": subprocess.DEVNULL,
                    "stdout": console,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                    "env": environment,
                }
                if os.name == "posix":
                    kwargs["start_new_session"] = True
                process = self.process_factory(command, **kwargs)
            except Exception as exc:  # noqa: BLE001 - child process boundary
                console.write(f"Failed to start BioMaker: {exc}\n")
                current = self.store.get_record("runs", run_id) or record
                status = "stopped" if current.get("status") == "stopping" else "failed"
                self._finish_run(run_id, status, None, str(exc), None)
                return

            pid = getattr(process, "pid", None)
            pid = int(pid) if isinstance(pid, int) and pid > 1 else None
            if not self._uses_injected_process_factory:
                try:
                    attached, leader_identity = self._await_bootstrap_identity(
                        record,
                        process,
                    )
                except (OSError, ProcessLookupError, RuntimeError, ValueError) as exc:
                    self._finish_uncaptured_run_after_cleanup(
                        run_id,
                        process,
                        str(exc),
                    )
                    return
                pid = leader_identity.pid
                pgid = leader_identity.pgid
                process_started_at = leader_identity.start_time
            else:
                pgid = self._resolve_process_group(pid)
                process_started_at = utc_now()
                if isinstance(pid, int) and isinstance(pgid, int):
                    leader_identity = ProcessIdentity(
                        pid=pid,
                        pgid=pgid,
                        start_time=process_started_at,
                        launch_token=str(record["launch_token"]),
                    )
            with self._lock:
                self._local_processes[run_id] = (str(record["launch_token"]), process)

            if self._uses_injected_process_factory:

                def attach_injected(data: dict[str, Any]) -> dict[str, Any]:
                    current = data["runs"].get(run_id)
                    if current is None:
                        raise KeyError(run_id)
                    if current.get("status") in TERMINAL_STATUSES:
                        return current
                    if not _is_claimed_run(current):
                        raise RuntimeError(
                            "Run lost its launch claim before registration"
                        )
                    current["pid"] = pid
                    current["pgid"] = pgid
                    current["process_started_at"] = process_started_at
                    current["launch_state"] = "registered"
                    current["launch_error"] = None
                    current["status"] = "running"
                    now = utc_now()
                    timestamps = current.setdefault("timestamps", {})
                    timestamps["started_at"] = now
                    timestamps["updated_at"] = now
                    return current

                _, attached = self.store.mutate_state(attach_injected)
                if attached.get("status") in TERMINAL_STATUSES:
                    try:
                        process.terminate()
                        process.wait(timeout=self.settings.stop_grace_seconds)
                    except (AttributeError, OSError, subprocess.TimeoutExpired):
                        try:
                            process.kill()
                            process.wait(timeout=self.settings.stop_grace_seconds)
                        except (AttributeError, OSError, subprocess.TimeoutExpired):
                            pass
                    return
            if attached.get("status") == "stopping":
                Thread(
                    target=self._terminate_record,
                    args=(run_id,),
                    name=f"biomaker-stop-{run_id[:8]}",
                    daemon=True,
                ).start()

            discovered = False
            while process.poll() is None:
                if not discovered:
                    discovered = self._discover_output_dir(run_id)
                time.sleep(0.25)
            return_code = process.wait()
            if not discovered:
                self._discover_output_dir(run_id)
            self._append_injected_output(process, console)
            console.write(f"\n[BioMaker exited with code {return_code}]\n")
            console.flush()

        current = self.store.get_record("runs", run_id) or record
        exit_signal = (
            -return_code if isinstance(return_code, int) and return_code < 0 else None
        )
        if not self._cleanup_before_terminal(
            current,
            leader_identity,
            return_code=return_code,
            exit_signal=exit_signal,
        ):
            return
        default_error = (
            f"BioMaker terminated by signal {exit_signal}"
            if exit_signal is not None
            else f"BioMaker exited with code {return_code} without a terminal event"
        )
        status, error = self._settle_dead_run(
            current,
            default_error,
            return_code=return_code,
        )
        self._finish_run(run_id, status, return_code, error, exit_signal)

    def _finish_uncaptured_run_after_cleanup(
        self,
        run_id: str,
        process: Any,
        launch_error: str,
    ) -> None:
        """Hold the Popen owner and queue barrier until emergency drain succeeds."""

        while True:
            try:
                self._drain_uncaptured_detached_process(process)
                break
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup boundary
                self._mark_uncaptured_run_cleanup_pending(
                    run_id,
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
                time.sleep(UNCAPTURED_CLEANUP_RETRY_SECONDS)

        current = self.store.get_record("runs", run_id)
        if current is None:
            return
        status = "stopped" if current.get("status") == "stopping" else "failed"
        return_code = getattr(process, "returncode", None)
        exit_signal = (
            -return_code if isinstance(return_code, int) and return_code < 0 else None
        )
        self._finish_run(
            run_id,
            status,
            return_code if isinstance(return_code, int) else None,
            launch_error,
            exit_signal,
        )

    def _mark_uncaptured_run_cleanup_pending(
        self,
        run_id: str,
        cleanup_error: str,
    ) -> None:
        """Atomically retain the active run and pause FIFO for cleanup retry."""

        now = utc_now()

        def mark(data: dict[str, Any]) -> None:
            current = data["runs"].get(run_id)
            if current is None or not _occupies_fifo(current):
                return
            current["cleanup_pending"] = True
            current["cleanup_error"] = cleanup_error
            current["status_reason"] = CLEANUP_PENDING_REASON
            current.setdefault("timestamps", {})["updated_at"] = now
            _set_queue_paused(data["queue_control"], CLEANUP_PENDING_REASON)

        self.store.mutate_state(mark)

    @staticmethod
    def _append_injected_output(process: Any, console: Any) -> None:
        stream = getattr(process, "stdout", None)
        if stream is None or stream is console:
            return
        try:
            content = stream.read()
        except (AttributeError, OSError, ValueError):
            return
        if content:
            console.write(str(content).replace("\r", ""))

    @staticmethod
    def _resolve_process_group(pid: int | None) -> int | None:
        if pid is None or os.name != "posix":
            return pid
        try:
            return os.getpgid(pid)
        except (OSError, ProcessLookupError):
            return pid

    def _monitor_external_run(self, record: dict[str, Any]) -> None:
        run_id = str(record["id"])
        leader_identity = self._identity_from_record(record)
        identity_state: str | None = None
        while True:
            current = self.store.get_record("runs", run_id)
            if current is None or current.get("status") not in ACTIVE_STATUSES:
                return
            if current.get("cleanup_pending"):
                leader_identity = self._identity_from_record(current) or leader_identity
                if leader_identity is None:
                    self._mark_uncaptured_run_cleanup_pending(
                        run_id,
                        str(
                            current.get("cleanup_error") or "cleanup owner is retrying"
                        ),
                    )
                    time.sleep(UNCAPTURED_CLEANUP_RETRY_SECONDS)
                    continue
                if not self._cleanup_before_terminal(
                    current,
                    leader_identity,
                    return_code=current.get("return_code"),
                    exit_signal=current.get("exit_signal"),
                ):
                    time.sleep(0.3)
                    continue
                leader_identity = None
                identity_state = "missing"
                break
            identity_state = self._record_identity_state(current)
            if identity_state == "unreadable" or self._launch_in_progress(current):
                time.sleep(0.3)
                continue
            if identity_state == "match":
                manifest_state = self._manifest_launch_state(current)
                if manifest_state != "match":
                    self._pause_for_manifest_safety(current, manifest_state)
                    time.sleep(0.3)
                    continue
            if current.get("status") == "stopping" and identity_state == "match":
                leader_identity = self._identity_from_record(current)
                if not self._terminate_record(run_id):
                    time.sleep(0.3)
                    continue
                time.sleep(0.05)
                continue
            if current.get("status") == "stopping" and identity_state == "missing":
                leader_identity = self._identity_from_record(current)
                if not self._terminate_record(run_id):
                    time.sleep(0.3)
                    continue
                break
            if identity_state in {"missing", "mismatch"}:
                break
            if identity_state != "match":
                time.sleep(0.3)
                continue
            leader_identity = self._identity_from_record(current)
            self._discover_output_dir(run_id)
            time.sleep(0.3)

        current = self.store.get_record("runs", run_id) or record
        if identity_state != "mismatch" and not self._cleanup_before_terminal(
            current, leader_identity
        ):
            time.sleep(0.3)
            return
        self._discover_output_dir(run_id)
        status, error = self._settle_dead_run(
            current,
            "Run process exited without a terminal lifecycle event",
        )
        self._finish_run(
            run_id,
            status,
            current.get("return_code"),
            error,
            current.get("exit_signal"),
        )

    def _discover_output_dir(self, run_id: str) -> bool:
        candidates: list[Path] = []
        record = self.store.get_record("runs", run_id)
        logs_root = self._record_root(record, "logs_root", self.settings.logs_root)
        try:
            paths = list(logs_root.iterdir())
        except OSError:
            return False
        launch_token = str((record or {}).get("launch_token") or run_id)
        for path in paths:
            try:
                if path.is_symlink() or not path.is_dir():
                    continue
                manifest_path = path / "run_manifest.json"
                if (
                    manifest_path.is_symlink()
                    or not manifest_path.is_file()
                    or manifest_path.stat().st_size > 5_000_000
                ):
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if (
                    isinstance(manifest, dict)
                    and manifest.get("launch_token") == launch_token
                ):
                    candidates.append(path)
            except (OSError, json.JSONDecodeError):
                continue
        if not candidates:
            return False
        try:
            log_dir = max(candidates, key=lambda path: path.stat().st_mtime).resolve()
        except OSError:
            return False
        actual_name = log_dir.name
        workspaces_root = self._record_root(
            record, "workspaces_root", self.settings.workspaces_root
        )

        def attach(data: dict[str, Any]) -> dict[str, Any]:
            current = data["runs"].get(run_id)
            if current is None:
                raise KeyError(run_id)
            for other_id, other in list(data["runs"].items()):
                if other_id == run_id or other.get("managed", True):
                    continue
                raw_other_log_dir = other.get("log_dir")
                if not raw_other_log_dir:
                    continue
                try:
                    other_log_dir = str(
                        Path(str(raw_other_log_dir)).expanduser().resolve()
                    )
                except (OSError, RuntimeError, ValueError):
                    continue
                if other_log_dir == str(log_dir):
                    del data["runs"][other_id]
            current["actual_exp_name"] = actual_name
            current["log_dir"] = str(log_dir)
            current["workspace_dir"] = str((workspaces_root / actual_name).resolve())
            current.setdefault("timestamps", {})["updated_at"] = utc_now()
            return current

        try:
            self.store.mutate_state(attach)
        except KeyError:
            return False
        return True

    @staticmethod
    def _record_root(
        record: dict[str, Any] | None,
        field: str,
        fallback: Path,
    ) -> Path:
        raw_value = (record or {}).get(field)
        if not raw_value:
            return fallback
        try:
            return Path(str(raw_value)).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return fallback

    def _settle_dead_run(
        self,
        record: dict[str, Any],
        default_error: str,
        *,
        return_code: int | None = None,
    ) -> tuple[str, str | None]:
        """Classify a dead run from lifecycle evidence plus an observed exit code."""

        run_id = str(record["id"])
        self._discover_output_dir(run_id)
        current = self.store.get_record("runs", run_id) or record
        status, event_error = self._terminal_status_from_events(current)
        effective_return_code = (
            return_code if isinstance(return_code, int) else current.get("return_code")
        )
        if status == "interrupted" and current.get("status") == "stopping":
            return "stopped", None
        if status == "succeeded":
            if (
                current.get("managed", True)
                and isinstance(effective_return_code, int)
                and effective_return_code != 0
            ):
                return (
                    "failed",
                    "Lifecycle mismatch: run_finished was recorded, but BioMaker "
                    f"exited with code {effective_return_code}",
                )
            return "succeeded", None
        if status in {"failed", "interrupted"}:
            return status, event_error or default_error
        if current.get("status") == "stopping":
            self._append_interrupted_event(current, "stopped")
            return "stopped", None

        launch_error = current.get("launch_error")
        if launch_error:
            error = str(launch_error)
            self._append_failed_event(current, error)
            return "failed", error
        if isinstance(effective_return_code, int) and effective_return_code > 0:
            error = (
                f"BioMaker exited with code {effective_return_code} before recording "
                "a terminal lifecycle event"
            )
            self._append_failed_event(current, error)
            return "failed", error

        if current.get("managed", True):
            self._append_interrupted_event(current, "supervisor_lost_process")
        return "interrupted", default_error

    @staticmethod
    def _append_interrupted_event(record: dict[str, Any], reason: str) -> bool:
        return RunManager._append_terminal_event(
            record,
            "run_interrupted",
            {"reason": reason},
        )

    @staticmethod
    def _append_failed_event(record: dict[str, Any], error: str) -> bool:
        return RunManager._append_terminal_event(
            record,
            "run_failed",
            {"error": error},
        )

    @staticmethod
    def _append_terminal_event(
        record: dict[str, Any],
        event_name: str,
        details: dict[str, Any],
    ) -> bool:
        raw_log_dir = record.get("log_dir")
        if not raw_log_dir:
            return False
        if (
            record.get("managed", True)
            and RunManager._manifest_launch_state(record) != "match"
        ):
            return False
        events_path = Path(str(raw_log_dir)) / "events.jsonl"
        try:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a+", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                handle.seek(0)
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") in {
                        "run_finished",
                        "run_failed",
                        "run_interrupted",
                    }:
                        return False
                handle.seek(0, os.SEEK_END)
                handle.write(
                    json.dumps(
                        {
                            "ts": utc_now(),
                            "event": event_name,
                            **details,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                return True
        except OSError:
            return False

    @staticmethod
    def _terminal_status_from_events(
        record: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        raw_log_dir = record.get("log_dir")
        if not raw_log_dir:
            return None, None
        if (
            record.get("managed", True)
            and RunManager._manifest_launch_state(record) != "match"
        ):
            return None, None
        events_path = Path(str(raw_log_dir)) / "events.jsonl"
        if not events_path.is_file():
            return None, None
        terminal: tuple[str | None, str | None] = (None, None)
        mapping = {
            "run_finished": "succeeded",
            "run_failed": "failed",
            "run_interrupted": "interrupted",
        }
        try:
            with events_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    status = mapping.get(event.get("event"))
                    if status:
                        terminal = (status, event.get("error") or event.get("reason"))
        except OSError:
            return None, None
        return terminal

    def _cleanup_process_group(
        self,
        record: dict[str, Any],
        leader_identity: ProcessIdentity | None,
    ) -> bool:
        """Drain the exact run session, including candidate-owned process groups."""

        current_identity = self._identity_from_record(record)
        if current_identity is None or current_identity != leader_identity:
            return False
        if self._manifest_launch_state(record) != "match":
            return False
        if not self._uses_injected_process_factory and os.name == "posix":
            result = terminate_process_session(
                current_identity,
                launch_token=current_identity.launch_token,
                grace_seconds=self.settings.stop_grace_seconds,
            )
            return result.accepted and result.outcome != "signal_error"

        pid = current_identity.pid
        pgid = current_identity.pgid
        if (
            os.name != "posix"
            or pid <= 1
            or pgid != pid
            or not self._process_group_alive(pgid)
        ):
            return True
        if not self._uses_injected_process_factory:
            verification = verify_os_process_identity(
                current_identity,
                launch_token=current_identity.launch_token,
            )
            if verification.reason not in {"match", "missing"}:
                return False
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            return True
        deadline = time.monotonic() + self.settings.stop_grace_seconds
        while self._process_group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.1)
        if not self._process_group_alive(pgid):
            return True
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            return not self._process_group_alive(pgid)
        deadline = time.monotonic() + max(self.settings.stop_grace_seconds, 1.0)
        while self._process_group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.1)
        return not self._process_group_alive(pgid)

    def _cleanup_before_terminal(
        self,
        record: dict[str, Any],
        leader_identity: ProcessIdentity | None,
        *,
        return_code: int | None = None,
        exit_signal: int | None = None,
    ) -> bool:
        """Keep a run active until its complete process session is gone."""

        if leader_identity is None:
            return True
        cleanup_error = None
        try:
            cleaned = self._cleanup_process_group(record, leader_identity)
        except Exception as exc:  # noqa: BLE001 - OS cleanup boundary
            cleaned = False
            cleanup_error = f"{type(exc).__name__}: {exc}"
        run_id = str(record["id"])
        now = utc_now()

        if cleaned:

            def mark_clean(data: dict[str, Any]) -> None:
                current = data["runs"].get(run_id)
                if current is None or current.get("status") not in ACTIVE_STATUSES:
                    return
                changed = bool(
                    current.get("cleanup_pending")
                    or current.get("cleanup_error") is not None
                    or current.get("status_reason") == CLEANUP_PENDING_REASON
                )
                current["cleanup_pending"] = False
                current["cleanup_error"] = None
                if current.get("status_reason") == CLEANUP_PENDING_REASON:
                    current["status_reason"] = None
                if changed:
                    current.setdefault("timestamps", {})["updated_at"] = now
                control = data["queue_control"]
                if control.get("reason") == CLEANUP_PENDING_REASON:
                    control.update({"paused": False, "reason": None})

            self.store.mutate_state(mark_clean)
            return True

        def mark_pending(data: dict[str, Any]) -> None:
            current = data["runs"].get(run_id)
            if current is None:
                return
            if current.get("status") not in ACTIVE_STATUSES:
                return
            changed = (
                not current.get("cleanup_pending")
                or current.get("cleanup_error") != cleanup_error
                or current.get("status_reason") != CLEANUP_PENDING_REASON
            )
            current["cleanup_pending"] = True
            current["cleanup_error"] = cleanup_error
            current["status_reason"] = CLEANUP_PENDING_REASON
            if return_code is not None:
                current["return_code"] = return_code
            if exit_signal is not None:
                current["exit_signal"] = exit_signal
            if changed:
                current.setdefault("timestamps", {})["updated_at"] = now
            _set_queue_paused(data["queue_control"], CLEANUP_PENDING_REASON)

        self.store.mutate_state(mark_pending)
        return False

    @staticmethod
    def _process_group_alive(pgid: int) -> bool:
        if os.name != "posix" or pgid <= 1:
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _finish_run(
        self,
        run_id: str,
        status: str,
        return_code: int | None,
        error: str | None,
        exit_signal: int | None,
    ) -> None:
        now = utc_now()

        def finish(data: dict[str, Any]) -> None:
            record = data["runs"].get(run_id)
            if record is None:
                raise KeyError(run_id)
            if record.get("status") in TERMINAL_STATUSES:
                return
            record["status"] = status
            record["return_code"] = return_code
            record["exit_signal"] = exit_signal
            record["error"] = error
            record["status_reason"] = error
            record["cleanup_pending"] = False
            record["cleanup_error"] = None
            timestamps = record.setdefault("timestamps", {})
            timestamps["finished_at"] = now
            timestamps["updated_at"] = now
            if status in PAUSING_STATUSES:
                _set_queue_paused(
                    data["queue_control"],
                    error or f"A run ended as {status}",
                    force_generation=True,
                )

        try:
            self.store.mutate_state(finish)
        except KeyError:
            return

    def _terminate_record(self, run_id: str) -> bool:
        record = self.store.get_record("runs", run_id)
        if record is None:
            return False
        identity_state = self._record_identity_state(record)
        if identity_state not in {"match", "missing"}:
            return False
        if self._manifest_launch_state(record) != "match":
            self._pause_for_manifest_safety(record)
            return False
        if not self._uses_injected_process_factory:
            identity = self._identity_from_record(record)
            if identity is None:
                return False
            return self._cleanup_before_terminal(record, identity)
        identity = self._identity_from_record(record)
        self._send_signal(record, signal.SIGTERM)
        deadline = time.monotonic() + self.settings.stop_grace_seconds
        while time.monotonic() < deadline:
            current = self.store.get_record("runs", run_id)
            if current is None:
                return True
            current_state = self._record_identity_state(current)
            if current_state != "match":
                if current is not None and identity is not None:
                    cleaned = self._cleanup_before_terminal(current, identity)
                    return current_state == "missing" and cleaned
                return current_state == "missing"
            time.sleep(0.1)
        current = self.store.get_record("runs", run_id)
        if current is not None and self._record_identity_state(current) == "match":
            self._send_signal(current, signal.SIGKILL)
        if current is not None and identity is not None:
            return self._cleanup_before_terminal(current, identity)
        return True

    def _send_signal(self, record: dict[str, Any], sig: signal.Signals) -> None:
        run_id = str(record["id"])
        with self._lock:
            local = self._local_processes.get(run_id)
        pid = record.get("pid")
        pgid = record.get("pgid")
        if (
            os.name == "posix"
            and isinstance(pid, int)
            and isinstance(pgid, int)
            and pid > 1
            and pgid == pid
            and self._process_group_alive(pgid)
        ):
            os.killpg(pgid, sig)
            return
        if local is not None and local[0] == str(record.get("launch_token")):
            process = local[1]
            if sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
            return
        if not isinstance(pid, int) or pid <= 1:
            return
        if os.name == "posix" and isinstance(pgid, int) and pgid == pid and pgid > 1:
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)

    @staticmethod
    def _manifest_launch_state(record: dict[str, Any]) -> str:
        raw_log_dir = record.get("log_dir")
        if not raw_log_dir:
            return "match"
        manifest_path = Path(str(raw_log_dir)) / "run_manifest.json"
        try:
            if (
                manifest_path.is_symlink()
                or not manifest_path.is_file()
                or manifest_path.stat().st_size > 5_000_000
            ):
                return "unreadable"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "unreadable"
        if not isinstance(manifest, dict):
            return "unreadable"
        return (
            "match"
            if manifest.get("launch_token") == record.get("launch_token")
            else "mismatch"
        )

    @staticmethod
    def _verification_state(verification: IdentityVerification) -> str:
        if verification.matched:
            return "match"
        if verification.reason == "missing":
            return "missing"
        if verification.reason == "unreadable":
            return "unreadable"
        return "mismatch"

    def _record_identity_state(self, record: dict[str, Any]) -> str:
        """Return only OS/process ownership state; manifest trust is separate."""

        run_id = str(record.get("id"))
        launch_token = str(record.get("launch_token") or "")
        with self._lock:
            local = self._local_processes.get(run_id)
        if local is not None and local[0] == launch_token:
            try:
                state = "match" if local[1].poll() is None else "missing"
            except (AttributeError, OSError):
                return "unreadable"
            return state
        supervisor_token = record.get("supervisor_token")
        if supervisor_token:
            with _INLINE_SUPERVISORS_LOCK:
                if str(supervisor_token) in _INLINE_SUPERVISORS:
                    return "match"
        identity = self._identity_from_record(record)
        if identity is None:
            return "missing"
        if not self._identity_verifier_injected:
            state = self._verification_state(
                verify_os_process_identity(identity, launch_token=launch_token)
            )
        else:
            state = (
                "match"
                if self.identity_verifier(
                    identity.pid,
                    identity.start_time,
                    record.get("launch_token"),
                )
                else "mismatch"
            )
        return state

    @staticmethod
    def _manifest_safety_reason(manifest_state: str) -> str:
        if manifest_state == "mismatch":
            detail = "run_manifest.json launch token does not match the run ledger"
        else:
            detail = "run_manifest.json is missing or unreadable"
        return (
            f"Process identity matches, but {detail}. The run remains active and "
            "Stop is blocked until manifest identity can be verified."
        )

    def _pause_for_manifest_safety(
        self,
        record: dict[str, Any],
        manifest_state: str | None = None,
    ) -> bool:
        manifest_state = manifest_state or self._manifest_launch_state(record)
        if manifest_state == "match":
            return False
        run_id = str(record["id"])
        reason = self._manifest_safety_reason(manifest_state)
        now = utc_now()

        def pause(data: dict[str, Any]) -> None:
            current = data["runs"].get(run_id)
            if current is None or current.get("status") not in ACTIVE_STATUSES:
                return
            if current.get("status_reason") != reason:
                current["status_reason"] = reason
                current.setdefault("timestamps", {})["updated_at"] = now
            control = data["queue_control"]
            if not control.get("paused") or control.get("reason") != reason:
                _set_queue_paused(control, reason)

        self.store.mutate_state(pause)
        return True

    def _active_manifest_safety_issue(self) -> bool:
        for record in self.store.list_records("runs"):
            if (
                record.get("managed", True)
                and record.get("status") in ACTIVE_STATUSES
                and self._record_identity_state(record) == "match"
                and self._manifest_launch_state(record) != "match"
            ):
                self._pause_for_manifest_safety(record)
                return True
        return False

    @staticmethod
    def _identity_from_record(record: dict[str, Any]) -> ProcessIdentity | None:
        try:
            return ProcessIdentity(
                pid=record["pid"],
                pgid=record["pgid"],
                start_time=record["process_started_at"],
                launch_token=record["launch_token"],
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _launch_in_progress(self, record: dict[str, Any]) -> bool:
        if record.get("pid") is not None or not record.get("supervisor_token"):
            return False
        timestamps = record.get("timestamps") or {}
        if not self._timestamp_is_recent(timestamps.get("started_at")):
            return False
        control = self.store.get_queue_control()
        if control.get("runner_token") != record.get("supervisor_token"):
            return False
        return self._runner_identity_state(control) in {
            "match",
            "unreadable",
        }

    def _claimed_launch_in_progress(self, record: dict[str, Any]) -> bool:
        if not _is_claimed_run(record) or record.get("pid") is not None:
            return False
        control = self.store.get_queue_control()
        if control.get("runner_token") != record.get("supervisor_token"):
            return False
        return self._runner_identity_state(control) in {
            "match",
            "unreadable",
        }

    def _runner_identity_state(self, control: dict[str, Any]) -> str:
        token = control.get("runner_token")
        if token:
            with _INLINE_SUPERVISORS_LOCK:
                if str(token) in _INLINE_SUPERVISORS:
                    return "match"
        pid = control.get("runner_pid")
        pgid = control.get("runner_pgid")
        started_at = control.get("runner_process_started_at")
        if not isinstance(pid, int):
            return "unregistered"
        if not self._identity_verifier_injected:
            try:
                identity = ProcessIdentity(
                    pid=pid,
                    pgid=int(pgid),
                    start_time=str(started_at),
                    launch_token=str(token),
                )
            except (TypeError, ValueError):
                return "mismatch"
            return self._verification_state(
                verify_os_process_identity(identity, launch_token=str(token))
            )
        return (
            "match"
            if self.identity_verifier(
                pid,
                started_at,
                control.get("runner_token"),
            )
            else "mismatch"
        )

    @staticmethod
    def _timestamp_is_recent(raw_timestamp: Any) -> bool:
        if not raw_timestamp:
            return False
        try:
            timestamp = datetime.fromisoformat(str(raw_timestamp))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
        except ValueError:
            return False
        age_seconds = (datetime.now(UTC) - timestamp).total_seconds()
        return age_seconds < LAUNCH_GRACE_SECONDS

    def _reconcile_runner_identity(self) -> None:
        def reconcile(control: dict[str, Any]) -> None:
            if self._runner_identity_state(control) in {
                "match",
                "unreadable",
            }:
                return
            control.update(
                {
                    "runner_pid": None,
                    "runner_pgid": None,
                    "runner_process_started_at": None,
                    "runner_token": None,
                    "runner_claimed_at": None,
                }
            )

        self.store.update_queue_control(reconcile)

    def _release_supervisor(self, token: str) -> None:
        with _INLINE_SUPERVISORS_LOCK:
            _INLINE_SUPERVISORS.discard(token)

        def release(control: dict[str, Any]) -> None:
            if control.get("runner_token") != token:
                return
            control.update(
                {
                    "runner_pid": None,
                    "runner_pgid": None,
                    "runner_process_started_at": None,
                    "runner_token": None,
                    "runner_claimed_at": None,
                }
            )

        self.store.update_queue_control(release)


def _supervisor_main(database_path: str, token: str) -> int:
    raw_config = os.environ.get(SUPERVISOR_CONFIG_ENV)
    if not raw_config:
        raise RuntimeError(f"Missing {SUPERVISOR_CONFIG_ENV}")
    settings = _settings_from_payload(json.loads(raw_config))
    store = SQLiteRunStore(database_path)
    try:
        manager = RunManager(settings, store)
        if manager._claim_supervisor(token):
            manager._supervisor_loop(token)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--supervise":
        raise SystemExit(_supervisor_main(sys.argv[2], sys.argv[3]))
    if len(sys.argv) == 5 and sys.argv[1] == "--launch-run":
        raise SystemExit(_run_bootstrap_main(sys.argv[2], sys.argv[3], sys.argv[4]))
