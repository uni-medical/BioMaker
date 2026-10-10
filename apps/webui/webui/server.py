from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from biomaker.application import AdmissionError, RunService
from biomaker.runtime.queries import CANONICAL_TERMINAL_STATUSES

from .artifacts import (
    ArtifactError,
    artifact_collection_revision,
    file_revision,
    list_artifacts,
    opaque_revision,
    read_events,
    resolve_artifact,
    resolve_run_log_dir,
    tail_text,
)
from .directory_picker import (
    DirectoryPickerBusy,
    DirectoryPickerError,
    DirectoryPickerTimeout,
    DirectoryPickerUnsupported,
    select_directory,
)
from .env_file import SECRET_ENV_KEYS, masked_env, read_env_values, update_env_file
from .model_discovery import discover_models, normalize_base_url
from .models import (
    EnvRevealRequest,
    EnvUpdate,
    ExperimentInput,
    ModelDiscoveryRequest,
    QueueResumeRequest,
    RunCreate,
)
from .run_catalog import list_log_runs
from .run_store import SQLiteRunStore
from .runner import RunManager
from .settings import Settings
from .system_resources import get_system_resources

STATIC_DIR = Path(__file__).resolve().parent / "static"




def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False



def _task_templates(settings: Settings) -> list[dict[str, Any]]:
    templates = []
    datasets = (
        {
            "".join(
                character for character in path.name.lower() if character.isalnum()
            ): path
            for path in settings.datasets_root.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        }
        if settings.datasets_root.is_dir()
        else {}
    )
    if not settings.experiments_root.is_dir():
        return templates
    for path in sorted(settings.experiments_root.glob("*/task.md")):
        if not path.is_file():
            continue
        normalized_name = "".join(
            character for character in path.parent.name.lower() if character.isalnum()
        )
        dataset_path = datasets.get(normalized_name)
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        templates.append(
            {
                "id": path.parent.name,
                "name": path.parent.name.replace("_", " ").title(),
                "path": str(path.resolve()),
                "dataset_path": str(dataset_path.resolve()) if dataset_path else None,
                "body": body,
            }
        )
    return templates


def _datasets(settings: Settings) -> list[dict[str, str]]:
    if not settings.datasets_root.is_dir():
        return []
    return [
        {"name": path.name, "path": str(path.resolve())}
        for path in sorted(
            settings.datasets_root.iterdir(), key=lambda item: item.name.lower()
        )
        if path.is_dir() and not path.name.startswith("_")
    ]


def _console_path(settings: Settings, run: dict[str, Any]) -> Path:
    raw = run.get("console_log")
    if not raw:
        raise HTTPException(status_code=404, detail="Console log is not available")
    path = Path(str(raw)).resolve()
    state_root = (settings.state_dir / "console").resolve()
    if _inside(path, state_root) and path.parent == state_root and path.is_file():
        return path
    try:
        log_dir = resolve_run_log_dir(settings, run)
    except ArtifactError:
        log_dir = None
    if (
        log_dir is not None
        and _inside(path, log_dir)
        and path.parent == log_dir
        and path.is_file()
    ):
        return path
    raise HTTPException(status_code=404, detail="Invalid console log path")


def _environment_response(settings: Settings) -> dict[str, Any]:
    return {
        "values": masked_env(settings.env_file),
        "source": {
            "path": str(settings.env_file),
            "exists": settings.env_file.is_file(),
        },
    }


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    is_loopback = bool(address and address.is_loopback)
    if address and address.version == 6 and address.ipv4_mapped:
        is_loopback = address.ipv4_mapped.is_loopback
    if not is_loopback:
        raise HTTPException(
            status_code=403,
            detail="This action is available only from this computer.",
        )


def _require_directory_picker_request(request: Request) -> None:
    _require_loopback(request)
    if request.headers.get("x-biomaker-ui") != "directory-picker":
        raise HTTPException(
            status_code=403,
            detail="The system folder picker can be opened only from the BioMaker UI.",
        )


def _same_base_url(first: str | None, second: str | None) -> bool:
    if not first or not second:
        return False
    try:
        return normalize_base_url(first) == normalize_base_url(second)
    except ValueError:
        return False


def create_app(
    settings: Settings | None = None,
    *,
    process_factory: Callable[..., Any] | None = None,
    admission_policy: Callable | None = None,
    env_handler: Callable | None = None,
) -> FastAPI:
    app_settings = settings or Settings.default()
    app_settings.ensure_directories()
    run_store = SQLiteRunStore(app_settings.state_dir / "runs.sqlite3")
    manager = RunManager(app_settings, run_store, process_factory=process_factory)
    service = RunService(manager, app_settings, admission_policy=admission_policy)
    observations = service.observations

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.import_history(list_log_runs(app_settings))
        manager.start()
        try:
            yield
        finally:
            try:
                manager.shutdown()
            finally:
                observations.clear()

    app = FastAPI(
        title="BioMaker WebUI",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.run_store = run_store
    app.state.manager = manager
    app.state.run_service = service
    app.state.observations = observations
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def response_headers(request: Request, call_next: Callable):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AdmissionError)
    async def admission_error(_: Request, exc: AdmissionError):
        payload = {"detail": exc.detail}
        if exc.code is not None:
            payload["code"] = exc.code
        return JSONResponse(payload, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(_: Request, exc: RequestValidationError):
        safe_errors = [
            {
                "type": error.get("type", "value_error"),
                "loc": error.get("loc", ()),
                "msg": error.get("msg", "Invalid value"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "queue": manager.summary(),
            "run_ledger": run_store.database_path,
            "workspace_root": str(app_settings.workspace_root),
            "method_root": str(app_settings.method_root),
        }

    @app.get("/api/system/resources")
    def system_resources() -> dict[str, Any]:
        return get_system_resources()

    @app.get("/api/queue")
    def queue_state() -> dict[str, Any]:
        return run_store.get_queue_control()

    @app.post("/api/queue/resume", status_code=status.HTTP_202_ACCEPTED)
    def resume_queue(request: Request, payload: QueueResumeRequest) -> dict[str, Any]:
        _require_loopback(request)
        try:
            return service.resume_queue(payload.pause_generation)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    if env_handler is not None:
        @app.api_route("/api/env", methods=["GET", "PUT"])
        @app.api_route("/api/env/", methods=["GET", "PUT"], include_in_schema=False)
        async def configured_env(request: Request):
            return await env_handler(request)
    else:
        @app.get("/api/env")
        def get_env() -> dict[str, Any]:
            return _environment_response(app_settings)

        @app.put("/api/env")
        def put_env(request: Request, payload: EnvUpdate) -> dict[str, Any]:
            _require_loopback(request)
            updates = {key: getattr(payload, key) for key in payload.model_fields_set}
            update_env_file(app_settings.env_file, updates)
            return _environment_response(app_settings)

    @app.post("/api/env/reveal")
    def reveal_env(request: Request, payload: EnvRevealRequest) -> dict[str, Any]:
        _require_loopback(request)
        values = read_env_values(app_settings.env_file, SECRET_ENV_KEYS)
        value = values.get(payload.key)
        return {
            "key": payload.key,
            "configured": value is not None,
            "value": value,
        }

    @app.post("/api/models/discover")
    def models_discover(
        request: Request, payload: ModelDiscoveryRequest
    ) -> dict[str, Any]:
        _require_loopback(request)
        values = read_env_values(app_settings.env_file)
        # OpenAI's client defaults to this URL when the setting is blank. Use
        # the same effective provider when deciding whether to reuse its key.
        configured_base_url = (
            values.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        )
        base_url = payload.base_url or configured_base_url
        if payload.api_key:
            api_key = payload.api_key
        elif payload.base_url is None or _same_base_url(
            payload.base_url, configured_base_url
        ):
            api_key = values.get("OPENAI_API_KEY")
        else:
            api_key = None
        return discover_models(base_url, api_key)

    @app.post("/api/directories/select")
    def directory_select(request: Request) -> dict[str, Any]:
        _require_directory_picker_request(request)
        try:
            return select_directory()
        except DirectoryPickerBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DirectoryPickerUnsupported as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except DirectoryPickerTimeout as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except DirectoryPickerError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/task-templates")
    def task_templates() -> list[dict[str, Any]]:
        return _task_templates(app_settings)

    @app.get("/api/datasets")
    def datasets() -> list[dict[str, str]]:
        return _datasets(app_settings)

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return service.list_runs()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return service.detail(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None

    @app.get("/api/runs/{run_id}/updates")
    def run_updates(run_id: str) -> dict[str, Any]:
        """Return only the revisions needed to update an open run view.

        A true ``final_barrier`` means the logs-backed record was already
        terminal before artifact revisions were sampled. Clients can therefore
        apply this response once before stopping their update scheduler.
        """

        run = service.status(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        artifacts = list_artifacts(app_settings, run)
        artifacts_by_name = {item["name"]: item for item in artifacts}
        tree = artifacts_by_name.get("tree_plot.html")
        events = artifacts_by_name.get("events.jsonl")
        try:
            console_revision = file_revision(_console_path(app_settings, run))
        except HTTPException:
            console_revision = None

        run_status = str(run.get("status") or "unknown")
        terminal = run_status in CANONICAL_TERMINAL_STATUSES
        return {
            "run_id": run_id,
            "status": run_status,
            "terminal": terminal,
            "final_barrier": terminal,
            "run_revision": opaque_revision(run),
            "tree": {
                "available": tree is not None,
                "revision": tree.get("revision") if tree else None,
                "url": tree.get("url") if tree else None,
            },
            "events": {
                "available": events is not None,
                "revision": events.get("revision") if events else None,
            },
            "console": {
                "available": console_revision is not None,
                "revision": console_revision,
            },
            "artifacts": {
                "revision": artifact_collection_revision(artifacts),
            },
        }

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    def create_run(request: Request, payload: RunCreate) -> dict[str, Any]:
        _require_loopback(request)
        try:
            return service.present_run(service.submit(payload))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(request: Request, run_id: str) -> dict[str, Any]:
        _require_loopback(request)
        try:
            return service.present_run(service.cancel(run_id))
        except KeyError:
            if service.status(run_id) is not None:
                raise HTTPException(
                    status_code=409, detail="This run is view-only"
                ) from None
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/stop", status_code=status.HTTP_202_ACCEPTED)
    def stop_run(request: Request, run_id: str) -> dict[str, Any]:
        _require_loopback(request)
        try:
            return service.present_run(service.stop(run_id))
        except KeyError:
            if service.status(run_id) is not None:
                raise HTTPException(
                    status_code=409, detail="This run is view-only"
                ) from None
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/requeue", status_code=status.HTTP_202_ACCEPTED)
    def requeue_run(request: Request, run_id: str) -> dict[str, Any]:
        _require_loopback(request)
        try:
            return service.present_run(service.retry(run_id))
        except KeyError:
            if service.status(run_id) is not None:
                raise HTTPException(
                    status_code=409, detail="This run is view-only"
                ) from None
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str, tail: int = Query(default=200, ge=1, le=2000)
    ) -> dict[str, Any]:
        run = service.status(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return read_events(app_settings, run, tail=tail)

    @app.get("/api/runs/{run_id}/log", response_class=PlainTextResponse)
    def run_log(
        run_id: str, tail: int = Query(default=400, ge=1, le=10_000)
    ) -> PlainTextResponse:
        run = service.status(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return PlainTextResponse(tail_text(_console_path(app_settings, run), tail))

    @app.get("/artifacts/runs/{run_id}/tree_plot.html", include_in_schema=False)
    def tree_plot(run_id: str) -> FileResponse:
        run = service.status(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            path = resolve_artifact(app_settings, run, "tree_plot.html")
        except ArtifactError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path, media_type="text/html", headers={"Cache-Control": "no-store"}
        )

    @app.get("/artifacts/runs/{run_id}/{artifact_name}", include_in_schema=False)
    def artifact(run_id: str, artifact_name: str) -> FileResponse:
        run = service.status(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            path = resolve_artifact(app_settings, run, artifact_name)
        except ArtifactError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        text_suffixes = {
            ".md",
            ".py",
            ".json",
            ".jsonl",
            ".tsv",
            ".yaml",
            ".log",
        }
        media_type = "text/plain" if path.suffix in text_suffixes else None
        return FileResponse(
            path, media_type=media_type, headers={"Cache-Control": "no-store"}
        )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local BioMaker WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
