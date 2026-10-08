"""Authenticated desktop adapter for the existing BioMaker WebUI service.

The desktop process owns the session token. This adapter leaves queue state and
experiment execution with BioMaker's durable runner, including background runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import platform
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, ConfigDict, Field
from history_import import HistoryImporter, HistoryImportError
from run_metadata import RunMetadataStore, RunMetadataUpdate, RunMetadataError
from native_runtime import NATIVE_RUNTIME_REVISION, native_receipt_is_current, provision_native_runtime
from provider_check import ProviderChecker, check_request
from provider_settings import ProviderSettings, provider_settings_request
from dataset_inspection import DatasetInspector, inspect_request, ADMISSION_MESSAGES
from artifact_preview import ArtifactPreviewError, preview_artifact
from result_overview import build_result_overview
from run_streams import RunStreams, StreamError

VERSION = os.environ.get("BIOMAKER_DESKTOP_VERSION", "0.1.11")


class HistoryPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=8192)


class HistoryImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_ids: list[str] = Field(min_length=1, max_length=100)


@dataclass(frozen=True)
class DesktopConfig:
    home: Path
    engine_root: Path
    token: str
    uv: str
    python: str
    allowed_origins: tuple[str, ...] = ()

    @classmethod
    def from_environment(cls) -> "DesktopConfig":
        token = os.environ.get("BIOMAKER_DESKTOP_TOKEN", "")
        if len(token) < 32:
            raise ValueError(
                "BIOMAKER_DESKTOP_TOKEN must contain at least 32 characters"
            )
        required = ("BIOMAKER_DESKTOP_HOME", "BIOMAKER_DESKTOP_ENGINE_ROOT")
        for key in required:
            if not os.environ.get(key) or not Path(os.environ[key]).is_absolute():
                raise ValueError(f"{key} must be an absolute path")
        return cls(
            home=Path(os.environ[required[0]]).resolve(),
            engine_root=Path(os.environ[required[1]]).resolve(),
            token=token,
            uv=os.environ.get("BIOMAKER_WEBUI_UV", "uv"),
            python=os.environ.get("BIOMAKER_DESKTOP_PYTHON", sys.executable),
            allowed_origins=tuple(
                value.strip()
                for value in os.environ.get(
                    "BIOMAKER_DESKTOP_ALLOWED_ORIGINS", ""
                ).split(",")
                if value.strip()
            ),
        )

    @property
    def workspace(self) -> Path:
        return self.home / "workspace"

    @property
    def env_file(self) -> Path:
        return self.home / "config.env"

    def prepare(self) -> None:
        if not (self.engine_root / "pyproject.toml").is_file():
            raise ValueError("The bundled BioMaker engine is missing pyproject.toml")
        for path in (self.home, self.workspace, self.home / "state"):
            path.mkdir(parents=True, exist_ok=True)
        # Environment is also the existing detached supervisor/CLI contract.
        os.environ.update(
            BIOMAKER_WEBUI_METHOD_ROOT=str(self.engine_root),
            BIOMAKER_WEBUI_WORKSPACE_ROOT=str(self.workspace),
            BIOMAKER_WEBUI_STATE_DIR=str(self.home / "state"),
            BIOMAKER_ENV_FILE=str(self.env_file),
            BIOMAKER_WEBUI_UV=self.uv,
            UV_PROJECT_ENVIRONMENT=str(self.engine_root / ".venv"),
            UV_PYTHON_DOWNLOADS="never",
            UV_CACHE_DIR=str(self.home / "cache" / "uv"),
        )
        # Generated experiment code must not inherit the desktop API credential.
        os.environ.pop("BIOMAKER_DESKTOP_TOKEN", None)
        for path in (self.engine_root, self.engine_root / "apps" / "webui"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))


class DesktopAccess:
    """Gate every HTTP route, including artifacts, docs, and static resources."""

    def __init__(self, app, *, token: str, allowed_origins: tuple[str, ...]):
        self.app = app
        self.expected = f"Bearer {token}".encode()
        self.allowed_origins = set(allowed_origins)

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        origin = headers.get(b"origin")
        authorized = hmac.compare_digest(
            headers.get(b"authorization", b""), self.expected
        )
        origin_allowed = (
            origin is None or origin.decode("latin-1") in self.allowed_origins
        )
        host = headers.get(b"host", b"").decode("latin-1").lower()
        hostname = (
            host.split(":", 1)[0]
            if not host.startswith("[")
            else host.split("]", 1)[0] + "]"
        )
        host_allowed = hostname in {"127.0.0.1", "localhost", "[::1]"}
        if not authorized or not origin_allowed or not host_allowed:
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            code = 401 if not authorized else 403
            response = JSONResponse(
                {
                    "detail": (
                        "Desktop authorization required"
                        if code == 401
                        else "Request origin is not allowed"
                    )
                },
                status_code=code,
                headers={"Cache-Control": "no-store"},
            )
            return await response(scope, receive, send)
        await self.app(scope, receive, send)


class EngineInstaller:
    """Own bounded engine maintenance, including its cross-process worker lease."""

    VERIFY_TIMEOUT_SECONDS = 90.0
    SETUP_TIMEOUT_SECONDS = 1800.0
    TERMINATE_GRACE_SECONDS = 3.0
    MAX_OUTPUT_LINE_BYTES = 8192
    FAILURE_MARKER = b"BioMaker engine health failed v1\n"
    MAX_MARKER_BYTES = 128

    def __init__(self, config: DesktopConfig):
        self.config = config
        self._lock = threading.RLock()
        self._termination_lock = threading.Lock()
        self._log: deque[str] = deque(maxlen=120)
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()
        self._operation: str | None = None
        self._claim_fd: int | None = None
        self._external_seen = False
        self._maintenance_unavailable = False
        self._installing = False
        self._error: str | None = None
        self._health = "unknown"
        self._checked_at: str | None = None
        self._ready = self._has_receipt()
        self._last_marker_state = self._marker_state()
        if self._last_marker_state != "clear":
            self._ready = False
            self._health = "failed"
            self._error = self._marker_error(self._last_marker_state)

    @property
    def receipt(self) -> Path:
        return self.config.engine_root / ".desktop-engine.json"

    @property
    def interpreter(self) -> Path:
        folder, executable = (("Scripts", "python.exe") if os.name == "nt" else ("bin", "python"))
        return self.config.engine_root / ".venv" / folder / executable

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        for filename in ("pyproject.toml", "requirements.txt", "uv.lock"):
            path = self.config.engine_root / filename
            digest.update(filename.encode())
            if path.is_file():
                digest.update(path.read_bytes())
        digest.update(platform.machine().encode())
        digest.update(sys.platform.encode())
        digest.update(NATIVE_RUNTIME_REVISION.encode())
        return digest.hexdigest()

    def _has_receipt(self) -> bool:
        try:
            if self._marker_state() != "clear":
                return False
            receipt = json.loads(self.receipt.read_text())
            return (
                isinstance(receipt, dict)
                and self.interpreter.is_file()
                and receipt.get("fingerprint") == self._fingerprint()
                and native_receipt_is_current(self.config.engine_root, receipt.get("native_runtime"))
            )
        except (OSError, ValueError, RuntimeError):
            return False

    def _marker_state(self) -> str:
        import stat

        descriptor = None
        try:
            descriptor = os.open(
                self.config.engine_root / ".desktop-engine-operation.lock",
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            )
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > self.MAX_MARKER_BYTES:
                return "invalid"
            data = os.read(descriptor, self.MAX_MARKER_BYTES)
            if data == b"":
                return "clear"
            return "failed" if data == self.FAILURE_MARKER else "invalid"
        except FileNotFoundError:
            return "clear"  # Legacy engines have no operation lock yet.
        except OSError:
            return "unavailable"
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @staticmethod
    def _marker_error(state: str) -> str:
        if state == "unavailable":
            return "The engine health record is unavailable. Check local disk access before repairing the runtime."
        if state == "invalid":
            return "The engine health record is invalid. Repair the runtime before starting an experiment."
        return "A previous engine check or maintenance failed. Repair the runtime before starting an experiment."

    def _write_marker(self, failed: bool) -> None:
        if self._claim_fd is None:
            raise RuntimeError("Engine health cannot be saved without owning its maintenance lock.")
        if failed:
            os.lseek(self._claim_fd, 0, os.SEEK_SET)
            remaining = self.FAILURE_MARKER
            while remaining:
                written = os.write(self._claim_fd, remaining)
                if written <= 0:
                    raise OSError("The engine health record could not be written.")
                remaining = remaining[written:]
            os.ftruncate(self._claim_fd, len(self.FAILURE_MARKER))
        else:
            os.ftruncate(self._claim_fd, 0)
        os.fsync(self._claim_fd)
        self._last_marker_state = "failed" if failed else "clear"

    def _open_claim(self) -> int | None:
        import fcntl
        import stat

        # Keep this inode permanently: unlinking a lock file would split the
        # ownership domain between old descriptors and a newly created file.
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(self.config.engine_root / ".desktop-engine-operation.lock", flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("The engine operation lock is not a regular file.")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                return None
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _external_busy(self) -> bool:
        if self._claim_fd is not None or os.name == "nt":
            self._maintenance_unavailable = False
            return False
        try:
            descriptor = self._open_claim()
        except OSError:
            self._maintenance_unavailable = True
            return False
        self._maintenance_unavailable = False
        if descriptor is None:
            return True
        # Close only: LOCK_UN would also unlock an inherited worker descriptor.
        os.close(descriptor)
        return False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            external = self._external_busy()
            marker = self._marker_state()
            if external or self._maintenance_unavailable:
                self._external_seen = True
            elif self._operation is None and (self._external_seen or marker != self._last_marker_state):
                self._ready = self._has_receipt()
                self._health = "unknown" if marker == "clear" else "failed"
                self._error = None if marker == "clear" else self._marker_error(marker)
                self._external_seen = False
                self._last_marker_state = marker
            return {
                "ready": self._ready and self._operation is None and not external and not self._maintenance_unavailable and self.receipt.is_file() and self.interpreter.is_file(),
                "installing": self._installing,
                "checking": self._operation == "check",
                "operation": self._operation,
                "health": "failed" if self._maintenance_unavailable else "unknown" if external else self._health,
                "checked_at": self._checked_at,
                "busy_external": external,
                "maintenance_unavailable": self._maintenance_unavailable,
                "error": "The engine operation lock is unavailable. Check local disk access." if self._maintenance_unavailable else "Another BioMaker engine operation is active. Wait for it to finish." if external else self._error,
                "log": "\n".join(self._log),
            }

    def _redact(self, text: str) -> str:
        from webui.env_file import SECRET_ENV_KEYS, read_env_values

        try:
            stored = read_env_values(self.config.env_file, SECRET_ENV_KEYS)
        except OSError:
            stored = {}
        secrets = [self.config.token, *stored.values()]
        secrets.extend(os.environ.get(key, "") for key in SECRET_ENV_KEYS)
        for secret in sorted((value for value in secrets if value), key=len, reverse=True):
            text = text.replace(secret, "[redacted]")
        text = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[redacted]", text)
        text = re.sub(r"\b(?:sk-|sk_|jina_)[A-Za-z0-9_-]{8,}\b", "[redacted]", text)
        text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", text)
        return text

    def _append(self, text: str) -> None:
        sanitized = self._redact(text.strip())
        if sanitized:
            with self._lock:
                self._log.append(sanitized[:2000])

    def start(self) -> dict[str, Any]:
        return self._begin("install")

    def check(self) -> dict[str, Any]:
        return self._begin("check")

    def repair(self) -> dict[str, Any]:
        return self._begin("repair")

    def _begin(self, operation: str) -> dict[str, Any]:
        if os.name == "nt":
            raise ValueError("Local engine execution currently requires macOS or Linux; Windows support is planned.")
        with self._lock:
            current = self.snapshot()
            if self._operation is not None or current["busy_external"] or (operation == "install" and current["ready"]):
                return current
            try:
                claim = self._open_claim()
            except OSError as exc:
                raise ValueError("The engine operation lock is unavailable. Check local disk access.") from exc
            if claim is None:
                return self.snapshot()
            self._claim_fd = claim
            self._operation = operation
            self._installing = operation in {"install", "repair"}
            self._error = None
            self._ready = False
            self._health = "unknown"
            self._cancelled.clear()
            self._log.clear()
            self._thread = threading.Thread(target=self._execute, args=(operation,), name=f"biomaker-engine-{operation}", daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._finish_operation()
                raise
            return self.snapshot()

    def _cancel_message(self) -> str:
        return {"check": "Engine check cancelled.", "repair": "Engine repair cancelled."}.get(self._operation, "Installation cancelled.")

    def _run(self, command: list[str], environment: dict[str, str], timeout: float | None = None) -> None:
        import selectors
        import time

        if self._cancelled.is_set():
            raise InterruptedError(self._cancel_message())
        process = subprocess.Popen(
            command, cwd=self.config.engine_root, env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, start_new_session=True,
            # The worker keeps the lease even if the sidecar is killed abruptly.
            pass_fds=(self._claim_fd,) if self._claim_fd is not None else (),
        )
        with self._lock:
            self._process = process
        deadline = time.monotonic() + (self.SETUP_TIMEOUT_SECONDS if timeout is None else timeout)
        pending = bytearray()
        skipping_line = False
        confirmed = False
        try:
            assert process.stdout is not None
            os.set_blocking(process.stdout.fileno(), False)
            with selectors.DefaultSelector() as reader:
                reader.register(process.stdout, selectors.EVENT_READ)
                pipe_open = True
                while pipe_open or process.poll() is None:
                    if self._cancelled.is_set():
                        raise InterruptedError(self._cancel_message())
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Engine verification timed out." if timeout == self.VERIFY_TIMEOUT_SECONDS else "Engine setup timed out.")
                    for key, _ in reader.select(min(0.1, remaining)):
                        try:
                            chunk = os.read(key.fd, 8192)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            reader.unregister(key.fileobj)
                            pipe_open = False
                            continue
                        for piece in chunk.splitlines(keepends=True):
                            ended = piece.endswith((b"\n", b"\r"))
                            if skipping_line:
                                skipping_line = not ended
                                continue
                            pending.extend(piece)
                            if len(pending) > self.MAX_OUTPUT_LINE_BYTES:
                                pending.clear()
                                skipping_line = not ended
                                self._append("[An oversized output line was omitted.]")
                            elif ended:
                                self._append(pending.decode("utf-8", errors="replace"))
                                pending.clear()
                if pending:
                    self._append(pending.decode("utf-8", errors="replace"))
            code = process.wait()
            if self._cancelled.is_set():
                raise InterruptedError(self._cancel_message())
            if code != 0:
                raise RuntimeError(f"Engine {'verification' if self._operation == 'check' else 'setup'} exited with code {code}. See the engine log.")
        finally:
            # A reaped leader is insufficient if a descendant still owns its
            # process group or the inherited operation descriptor.
            with self._termination_lock:
                confirmed = self._terminate(process)
            if process.stdout:
                process.stdout.close()
            with self._lock:
                if confirmed and self._process is process:
                    self._process = None
            if not confirmed:
                raise RuntimeError("The engine command did not stop completely. Quit BioMaker and wait for it to stop before retrying.")

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def _verify(self, environment: dict[str, str]) -> None:
        self._append("Checking the installed BioMaker engine and scientific packages…")
        self._run([str(self.interpreter), "-I", "-B", str(Path(__file__).with_name("verify_engine.py"))], environment, self.VERIFY_TIMEOUT_SECONDS)

    def _save_receipt(self, native_runtime: dict[str, Any]) -> None:
        fd, name = tempfile.mkstemp(prefix=".desktop-engine-", dir=self.config.engine_root)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump({"version": VERSION, "fingerprint": self._fingerprint(), "native_runtime": native_runtime}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, self.receipt)
        finally:
            Path(name).unlink(missing_ok=True)

    def _execute(self, operation: str) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("BIOMAKER_DESKTOP_TOKEN", None)
        environment.update(UV_NO_PROGRESS="1", UV_PROJECT_ENVIRONMENT=str(self.config.engine_root / ".venv"), UV_PYTHON_DOWNLOADS="never", PYTHONDONTWRITEBYTECODE="1")
        try:
            if operation != "check":
                self.receipt.unlink(missing_ok=True)
            if self._cancelled.is_set():
                raise InterruptedError(self._cancel_message())
            if operation == "check":
                if not self._has_receipt():
                    raise RuntimeError("The engine interpreter or installation receipt is missing or outdated. Install or repair the engine.")
                self._verify(environment)
            else:
                # Invalidate before uv or native patching can change any files.
                # A cancelled repair must never leave a trusted old receipt.
                self._append("Repairing the locked BioMaker engine; installed packages will be reinstalled." if operation == "repair" else "Installing the locked BioMaker engine. The first setup downloads scientific computing packages.")
                command = [self.config.uv, "sync", "--frozen"]
                if operation == "repair":
                    command.append("--reinstall")
                command.extend(["--project", str(self.config.engine_root), "--python", self.config.python])
                self._run(command, environment)
                if not self.interpreter.is_file():
                    raise RuntimeError("Setup did not create the engine Python environment.")
                if self._cancelled.is_set():
                    raise InterruptedError(self._cancel_message())
                native_runtime = provision_native_runtime(self.config.engine_root, run=lambda command: self._run(command, environment, 30.0))
                self._verify(environment)
            # Serialize cancellation with the successful receipt/state commit.
            with self._lock:
                if self._cancelled.is_set():
                    raise InterruptedError(self._cancel_message())
                if operation != "check":
                    self._save_receipt(native_runtime)
                    # Verify and commit the receipt first. A crash before this
                    # fsync leaves the durable failure state blocking readiness.
                    self._write_marker(False)
                self._ready = True
                self._health = "healthy"
                self._checked_at = self._now()
        except Exception as exc:
            cancelled = isinstance(exc, InterruptedError)
            if not cancelled:
                try:
                    # The held writable inode survives even if the directory
                    # refuses receipt deletion. Unknown/partial marker bytes
                    # are also rejected on the next startup.
                    self._write_marker(True)
                except OSError:
                    self._append("The engine health failure could not be saved; check local disk access.")
                try:
                    self.receipt.unlink(missing_ok=True)
                except OSError:
                    # Preserve a deterministic failure, never re-enable execution.
                    self._append("The failed engine receipt could not be removed; check local disk access.")
            message = self._redact(str(exc))
            self._append(message)
            with self._lock:
                self._error = message
                self._ready = False
                self._health = "unknown" if cancelled else "failed"
                if not cancelled:
                    self._checked_at = self._now()
        finally:
            with self._lock:
                self._finish_operation()

    def _finish_operation(self) -> None:
        if self._process is not None:
            return  # Keep the lease and busy state until child exit is confirmed.
        self._operation = None
        self._installing = False
        if self._claim_fd is not None:
            os.close(self._claim_fd)
            self._claim_fd = None

    def _terminate(self, process: subprocess.Popen) -> bool:
        import time

        def group_alive() -> bool:
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False
            except OSError:
                return True

        process.poll()
        if not group_alive():
            try:
                process.wait(timeout=self.TERMINATE_GRACE_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                return False
            return True
        for action in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, action)
            except ProcessLookupError:
                pass
            except OSError:
                return False
            deadline = time.monotonic() + self.TERMINATE_GRACE_SECONDS
            while time.monotonic() < deadline:
                process.poll()
                if not group_alive():
                    return process.poll() is not None
                time.sleep(0.025)
        return process.poll() is not None and not group_alive()

    def _cancel_process(self, process: subprocess.Popen) -> None:
        with self._termination_lock:
            confirmed = self._terminate(process)
        with self._lock:
            if confirmed and self._process is process and (self._thread is None or not self._thread.is_alive()):
                self._process = None
                self._finish_operation()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._operation is not None:
                self._cancelled.set()
                process = self._process
                if process is not None:
                    threading.Thread(target=self._cancel_process, args=(process,), daemon=True).start()
            return self.snapshot()

    def close(self) -> None:
        self.cancel()
        if self._thread is not None:
            self._thread.join(timeout=7)

def create_desktop_app(
    config: DesktopConfig, *, installer: EngineInstaller | None = None
) -> FastAPI:
    config.prepare()
    from contextlib import asynccontextmanager

    from webui.env_file import read_env_values
    from webui.models import RunCreate
    from webui.server import create_app
    from webui.settings import Settings

    settings = Settings.default()
    app = create_app(settings)
    # Replace inherited routes before adding desktop policy; duplicate routes
    # appended after the shared handlers would never intercept those requests.
    app.router.routes = [route for route in app.router.routes if not (
        getattr(route, "path", None) == "/api/env"
        and getattr(route, "methods", set()) & {"GET", "PUT"}
    )]
    provider_settings = ProviderSettings(config.env_file)

    @app.get("/api/env")
    @app.put("/api/env")
    @app.get("/api/env/", include_in_schema=False)
    @app.put("/api/env/", include_in_schema=False)
    async def desktop_provider_settings(request: Request):
        return await provider_settings_request(request, provider_settings)

    app.state.provider_settings = provider_settings
    engine = installer or EngineInstaller(config)
    history = HistoryImporter(settings, app.state.manager)
    metadata = RunMetadataStore(settings.state_dir / "desktop-metadata.sqlite3", app.state.run_store)
    provider = ProviderChecker(config.env_file)
    datasets = DatasetInspector(settings.workspace_root)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with original_lifespan(application):
                history.recover()
                yield
        finally:
            # One failed cleanup must not strand another owned worker.
            for component in (provider, datasets, history, metadata, engine):
                try:
                    component.close()
                except Exception:
                    pass

    app.router.lifespan_context = lifespan
    app.state.desktop_engine = engine
    app.state.history_importer = history
    app.state.run_metadata = metadata
    app.state.provider_checker = provider
    app.state.dataset_inspector = datasets

    @app.post("/api/desktop/provider/check")
    async def check_provider(request: Request):
        return await check_request(request, provider)

    @app.post("/api/desktop/dataset/inspect")
    async def inspect_dataset(request: Request):
        return await inspect_request(request, datasets)

    @app.get("/api/desktop/runs/{run_id}/result")
    def result_overview(run_id: str):
        record = app.state.run_store.get_record("runs", run_id)
        if record is None:
            return JSONResponse({"detail": "Run not found."}, status_code=404, headers={"Cache-Control": "no-store"})
        return JSONResponse(build_result_overview(record, logs_root=settings.logs_root), headers={"Cache-Control": "no-store"})

    streams = RunStreams(settings.logs_root, settings.state_dir, app.state.run_store)
    app.state.run_streams = streams

    @app.exception_handler(StreamError)
    async def stream_error(_request: Request, exc: StreamError):
        return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=exc.status_code,
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/desktop/runs/{run_id}/console")
    def console_window(run_id: str, before: str | None = None):
        return JSONResponse(streams.window(run_id, "console", before), headers={"Cache-Control": "no-store"})

    @app.get("/api/desktop/runs/{run_id}/activity")
    def activity_window(run_id: str, before: str | None = None):
        return JSONResponse(streams.window(run_id, "activity", before), headers={"Cache-Control": "no-store"})

    @app.get("/api/desktop/runs/{run_id}/console/download")
    def download_console(run_id: str):
        download = streams.download(run_id)
        return StreamingResponse(download.chunks(), media_type="application/octet-stream",
                                 headers={"Cache-Control": "no-store", "Content-Length": str(download.size),
                                          "Content-Disposition": 'attachment; filename="console.log"',
                                          "X-BioMaker-Console-SHA256": download.sha256},
                                 background=BackgroundTask(download.close))

    @app.get("/api/desktop/runs/{run_id}/artifacts/{name}/preview")
    def preview_result_artifact(run_id: str, name: str):
        try:
            return JSONResponse(preview_artifact(settings.logs_root, app.state.run_store, run_id, name), headers={"Cache-Control": "no-store"})
        except ArtifactPreviewError as exc:
            return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=exc.status_code, headers={"Cache-Control": "no-store"})

    @app.exception_handler(RunMetadataError)
    async def metadata_error(_request: Request, exc: RunMetadataError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)

    @app.get("/api/desktop/run-metadata")
    def list_run_metadata():
        return metadata.list_items()

    @app.get("/api/desktop/run-metadata/{run_id}")
    def get_run_metadata(run_id: str):
        return metadata.get(run_id)

    @app.put("/api/desktop/run-metadata/{run_id}")
    def update_run_metadata(run_id: str, payload: RunMetadataUpdate):
        try:
            return metadata.update(run_id, payload)
        except sqlite3.Error as exc:
            raise HTTPException(status_code=503, detail="Organization settings could not be saved. Check local disk access and retry.") from exc

    @app.post("/api/desktop/history/preview")
    def preview_history(payload: HistoryPreviewRequest):
        try:
            return history.preview(payload.path)
        except HistoryImportError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=422, detail="The selected folder could not be read. Check access and scan again.") from exc

    @app.post("/api/desktop/history/import", status_code=202)
    def import_history(payload: HistoryImportRequest):
        try:
            return history.submit(payload.preview_id, payload.source_ids)
        except HistoryImportError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=422, detail="The import could not start. Check local disk space and folder access.") from exc

    @app.get("/api/desktop/history/imports/{job_id}")
    def history_import_status(job_id: str):
        try:
            return history.get_job(job_id)
        except HistoryImportError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @app.get("/api/desktop/status")
    def desktop_status():
        values = read_env_values(config.env_file)
        return {
            "version": VERSION,
            "engine": engine.snapshot(),
            "paths": {
                "workspace": str(config.workspace),
                "logs": str(settings.logs_root),
                "engine": str(config.engine_root),
            },
            "configuration": {
                "api_key_configured": bool(values.get("OPENAI_API_KEY")),
                "model_configured": bool(values.get("BIOMAKER_MAIN_MODEL")),
            },
            "capabilities": {
                "local_execution": os.name != "nt",
                "platform": sys.platform,
                "background_runs": True,
            },
        }

    def engine_operation(action: str):
        # Include unsettled descendants even after their leader is terminal.
        # Admission middleware below keeps enqueue/requeue/resume from racing
        # this inspection and the installer's synchronous busy-state transition.
        records = app.state.run_store.list_records("runs")
        if any(
            record.get("managed", True)
            and (
                record.get("status") in {"queued", "running", "stopping"}
                or record.get("cleanup_pending")
            )
            for record in records
        ):
            return JSONResponse(
                {
                    "detail": "Finish or cancel queued experiments and wait for process cleanup before checking or repairing the runtime."
                },
                status_code=409,
            )
        try:
            return getattr(engine, action)()
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.post("/api/desktop/engine/install", status_code=202)
    def install_engine():
        return engine_operation("start")

    @app.post("/api/desktop/engine/check", status_code=202)
    def check_engine():
        return engine_operation("check")

    @app.post("/api/desktop/engine/repair", status_code=202)
    def repair_engine():
        return engine_operation("repair")

    @app.post("/api/desktop/engine/cancel", status_code=202)
    def cancel_install():
        return engine.cancel()

    async def checked_dispatch(request: Request, call_next):
        path = request.url.path
        control = re.fullmatch(r"/api/runs/([^/]+)/(?:stop|cancel|requeue)", path)
        if request.method == "POST" and control:
            record = app.state.run_store.get_record("runs", control.group(1))
            if record and (record.get("read_only") or not record.get("managed", True)):
                return JSONResponse({"detail": "Imported experiment records are view-only and cannot be controlled."}, status_code=409)
        is_create = request.method == "POST" and path == "/api/runs"
        is_requeue = (
            request.method == "POST"
            and path.startswith("/api/runs/")
            and path.endswith("/requeue")
        )
        is_resume = request.method == "POST" and path == "/api/queue/resume"
        if is_create or is_requeue or is_resume:
            if os.name == "nt" or not engine.snapshot()["ready"]:
                return JSONResponse(
                    {
                        "detail": "Install or repair the BioMaker runtime, and wait for maintenance to finish before starting an experiment."
                    },
                    status_code=409,
                )
            values = read_env_values(config.env_file)
            if not values.get("OPENAI_API_KEY"):
                return JSONResponse(
                    {
                        "detail": "Add an API key in Settings before starting an experiment."
                    },
                    status_code=409,
                )
            model = values.get("BIOMAKER_MAIN_MODEL")
            dataset_path = None
            if is_create:
                try:
                    payload = await request.json()
                    try:
                        validated_run = RunCreate.model_validate(payload)
                        dataset_path = validated_run.experiment.dataset_path
                    except ValueError:
                        pass  # Let the original route report schema errors.
                    model = (
                        (payload.get("options") or {}).get("model_for_all_agents")
                        or (
                            payload.get("experiment", {}).get("advanced_defaults") or {}
                        ).get("model_for_all_agents")
                        or model
                    )
                except (ValueError, AttributeError):
                    pass  # Existing Pydantic validation owns malformed request errors.
            elif is_requeue:
                record = app.state.run_store.get_record("runs", path.split("/")[-2])
                model = (record or {}).get("options", {}).get(
                    "model_for_all_agents"
                ) or model
                dataset_path = ((record or {}).get("project_snapshot") or {}).get("dataset_path")
            if not is_resume and (not isinstance(model, str) or not model.strip()):
                return JSONResponse(
                    {
                        "detail": "Choose a model in Settings or the experiment options before starting."
                    },
                    status_code=409,
                )
            if not is_resume and dataset_path is not None:
                checked = await asyncio.to_thread(datasets.inspect, {"path": dataset_path}, mode="admission")
                if checked["status"] == "error":
                    code = checked["code"]
                    return JSONResponse({"detail": f"[dataset:{code}] {ADMISSION_MESSAGES.get(code, ADMISSION_MESSAGES['internal'])}", "code": code}, status_code=409 if code == "busy" else 422)
        return await call_next(request)

    admission = asyncio.Lock()

    @app.middleware("http")
    async def preflight(request: Request, call_next):
        path = request.url.path.rstrip("/")
        exclusive = request.method == "POST" and (
            path in {
                "/api/runs", "/api/queue/resume",
                "/api/desktop/engine/install", "/api/desktop/engine/check",
                "/api/desktop/engine/repair",
            }
            or re.fullmatch(r"/api/runs/[^/]+/requeue", path)
        )
        if exclusive:
            async with admission:
                return await checked_dispatch(request, call_next)
        return await checked_dispatch(request, call_next)

    app.add_middleware(
        DesktopAccess, token=config.token, allowed_origins=config.allowed_origins
    )
    return app
