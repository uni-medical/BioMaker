"""Authenticated desktop adapter for the existing BioMaker WebUI service.

The desktop process owns the session token. This adapter leaves queue state and
experiment execution with BioMaker's durable runner, including background runs.
"""

from __future__ import annotations

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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from history_import import HistoryImporter, HistoryImportError
from run_metadata import RunMetadataStore, RunMetadataUpdate, RunMetadataError
from native_runtime import NATIVE_RUNTIME_REVISION, native_receipt_is_current, provision_native_runtime

VERSION = os.environ.get("BIOMAKER_DESKTOP_VERSION", "0.1.6")


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
    """Install the locked Python engine off the request thread, with cancellation."""

    def __init__(self, config: DesktopConfig):
        self.config = config
        self._lock = threading.RLock()
        self._log: deque[str] = deque(maxlen=120)
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._cancelled = threading.Event()
        self._installing = False
        self._error: str | None = None
        self._ready = self._has_receipt()

    @property
    def receipt(self) -> Path:
        return self.config.engine_root / ".desktop-engine.json"

    @property
    def interpreter(self) -> Path:
        folder, executable = (
            ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
        )
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
            receipt = json.loads(self.receipt.read_text())
            return (
                self.interpreter.is_file()
                and receipt.get("fingerprint") == self._fingerprint()
                and native_receipt_is_current(self.config.engine_root, receipt.get("native_runtime"))
            )
        except (OSError, ValueError):
            return False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self._ready and self.interpreter.is_file(),
                "installing": self._installing,
                "error": self._error,
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
        for secret in sorted(
            (value for value in secrets if value), key=len, reverse=True
        ):
            text = text.replace(secret, "[redacted]")
        text = re.sub(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+", r"\1[redacted]", text
        )
        text = re.sub(r"\b(?:sk-|sk_|jina_)[A-Za-z0-9_-]{8,}\b", "[redacted]", text)
        text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", text)
        return text

    def _append(self, text: str) -> None:
        sanitized = self._redact(text.strip())
        if sanitized:
            with self._lock:
                self._log.append(sanitized[:2000])

    def start(self) -> dict[str, Any]:
        if os.name == "nt":
            raise ValueError(
                "Local engine execution currently requires macOS or Linux; Windows support is planned."
            )
        with self._lock:
            if self._installing or self.snapshot()["ready"]:
                return self.snapshot()
            self._installing = True
            self._error = None
            self._ready = False
            self._cancelled.clear()
            self._log.clear()
            self._thread = threading.Thread(
                target=self._install, name="biomaker-engine-install", daemon=True
            )
            self._thread.start()
            return self.snapshot()

    def _run(self, command: list[str], environment: dict[str, str]) -> None:
        if self._cancelled.is_set():
            raise InterruptedError("Installation cancelled.")
        process = subprocess.Popen(
            command,
            cwd=self.config.engine_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=os.name != "nt",
        )
        with self._lock:
            self._process = process
        if self._cancelled.is_set():
            self._terminate(process)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                self._append(line)
            code = process.wait()
            if self._cancelled.is_set():
                raise InterruptedError("Installation cancelled.")
            if code != 0:
                raise RuntimeError(
                    f"Engine setup exited with code {code}. See the installation log."
                )
        finally:
            if process.poll() is None:
                self._terminate(process)
            with self._lock:
                self._process = None
            if process.stdout:
                process.stdout.close()

    def _install(self) -> None:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("BIOMAKER_DESKTOP_TOKEN", None)
        environment["UV_NO_PROGRESS"] = "1"
        environment["UV_PROJECT_ENVIRONMENT"] = str(self.config.engine_root / ".venv")
        environment["UV_PYTHON_DOWNLOADS"] = "never"
        try:
            self._append(
                "Installing the locked BioMaker engine. The first setup downloads scientific computing packages."
            )
            self._run(
                [
                    self.config.uv,
                    "sync",
                    "--frozen",
                    "--project",
                    str(self.config.engine_root),
                    "--python",
                    self.config.python,
                ],
                environment,
            )
            if not self.interpreter.is_file():
                raise RuntimeError(
                    "Setup did not create the engine Python environment."
                )
            self._append("Verifying the BioMaker engine and scientific packages…")
            native_runtime = provision_native_runtime(self.config.engine_root, run=lambda command: self._run(command, environment))
            self._run(
                [
                    str(self.interpreter),
                    "-I",
                    str(Path(__file__).with_name("verify_engine.py")),
                ],
                environment,
            )
            if self._cancelled.is_set():
                raise InterruptedError("Installation cancelled.")
            fd, name = tempfile.mkstemp(
                prefix=".desktop-engine-", dir=self.config.engine_root
            )
            try:
                with os.fdopen(fd, "w") as handle:
                    json.dump(
                        {"version": VERSION, "fingerprint": self._fingerprint(), "native_runtime": native_runtime}, handle
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(name, self.receipt)
            finally:
                Path(name).unlink(missing_ok=True)
            with self._lock:
                self._ready = True
        except Exception as exc:
            message = self._redact(str(exc))
            self._append(message)
            with self._lock:
                self._error = message
                self._ready = False
        finally:
            with self._lock:
                self._installing = False

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
        except ProcessLookupError:
            pass

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._installing:
                self._cancelled.set()
                process = self._process
                if process is not None:
                    threading.Thread(
                        target=self._terminate, args=(process,), daemon=True
                    ).start()
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
    from webui.server import create_app
    from webui.settings import Settings

    settings = Settings.default()
    app = create_app(settings)
    engine = installer or EngineInstaller(config)
    history = HistoryImporter(settings, app.state.manager)
    metadata = RunMetadataStore(settings.state_dir / "desktop-metadata.sqlite3", app.state.run_store)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with original_lifespan(application):
                history.recover()
                yield
        finally:
            history.close()
            metadata.close()
            engine.close()

    app.router.lifespan_context = lifespan
    app.state.desktop_engine = engine
    app.state.history_importer = history
    app.state.run_metadata = metadata

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

    @app.post("/api/desktop/engine/install", status_code=202)
    def install_engine():
        queue = app.state.manager.summary()
        if queue["running"] or queue["queued"]:
            return JSONResponse(
                {
                    "detail": "Finish or cancel queued experiments before changing the engine."
                },
                status_code=409,
            )
        try:
            return engine.start()
        except ValueError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.post("/api/desktop/engine/cancel", status_code=202)
    def cancel_install():
        return engine.cancel()

    @app.middleware("http")
    async def preflight(request: Request, call_next):
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
        if is_create or is_requeue:
            if os.name == "nt" or not engine.snapshot()["ready"]:
                return JSONResponse(
                    {
                        "detail": "Install the BioMaker engine before starting an experiment."
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
            if is_create:
                try:
                    payload = await request.json()
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
            if not isinstance(model, str) or not model.strip():
                return JSONResponse(
                    {
                        "detail": "Choose a model in Settings or the experiment options before starting."
                    },
                    status_code=409,
                )
        return await call_next(request)

    app.add_middleware(
        DesktopAccess, token=config.token, allowed_origins=config.allowed_origins
    )
    return app
