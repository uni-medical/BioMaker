from __future__ import annotations

import argparse
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .artifacts import (
    ArtifactError,
    detailed_run,
    read_events,
    resolve_artifact,
    tail_text,
)
from .env_file import masked_env, update_env_file
from .models import EnvUpdate, ProjectInput, RunCreate
from .registry import JsonRegistry, RegistryError
from .runner import RunManager, utc_now
from .settings import Settings

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _normalized_project(
    settings: Settings,
    payload: ProjectInput,
    *,
    project_id: str,
    created_at: str,
) -> dict[str, Any]:
    dataset_path = Path(payload.dataset_path).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = settings.workspace_root / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_dir():
        raise HTTPException(
            status_code=422,
            detail=f"Dataset directory does not exist: {dataset_path}",
        )

    template_path = None
    if payload.task_template_path:
        candidate = Path(payload.task_template_path).expanduser()
        if not candidate.is_absolute():
            candidate = settings.workspace_root / candidate
        candidate = candidate.resolve()
        experiments_root = settings.experiments_root.resolve()
        if (
            not _inside(candidate, experiments_root)
            or not candidate.is_file()
            or candidate.name != "task.md"
        ):
            raise HTTPException(status_code=422, detail="Invalid task template path")
        template_path = str(candidate)

    now = utc_now()
    return {
        "id": project_id,
        "name": payload.name,
        "dataset_path": str(dataset_path),
        "task_template_path": template_path,
        "task_body": payload.task_body,
        "advanced_defaults": payload.advanced_defaults.model_dump(),
        "created_at": created_at,
        "updated_at": now,
    }


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
    root = (settings.state_dir / "console").resolve()
    if not _inside(path, root) or path.parent != root:
        raise HTTPException(status_code=404, detail="Invalid console log path")
    return path


def create_app(
    settings: Settings | None = None,
    *,
    process_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    app_settings = settings or Settings.default()
    app_settings.ensure_directories()
    registry = JsonRegistry(app_settings.state_dir / "registry.json")
    manager = RunManager(app_settings, registry, process_factory=process_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        manager.start()
        yield
        manager.shutdown()

    app = FastAPI(
        title="BioMaker WebUI",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.registry = registry
    app.state.manager = manager
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def response_headers(request: Request, call_next: Callable):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RegistryError)
    async def registry_error(_: Request, exc: RegistryError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

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
            "workspace_root": str(app_settings.workspace_root),
            "method_root": str(app_settings.method_root),
        }

    @app.get("/api/env")
    def get_env() -> dict[str, Any]:
        return {"values": masked_env(app_settings.env_file)}

    @app.put("/api/env")
    def put_env(payload: EnvUpdate) -> dict[str, Any]:
        updates = {key: getattr(payload, key) for key in payload.model_fields_set}
        update_env_file(app_settings.env_file, updates)
        return {"values": masked_env(app_settings.env_file)}

    @app.get("/api/task-templates")
    def task_templates() -> list[dict[str, Any]]:
        return _task_templates(app_settings)

    @app.get("/api/datasets")
    def datasets() -> list[dict[str, str]]:
        return _datasets(app_settings)

    @app.get("/api/projects")
    def list_projects() -> list[dict[str, Any]]:
        return sorted(
            registry.list_records("projects"),
            key=lambda item: item.get("updated_at", ""),
            reverse=True,
        )

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> dict[str, Any]:
        project = registry.get_record("projects", project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return project

    @app.post("/api/projects", status_code=status.HTTP_201_CREATED)
    def create_project(payload: ProjectInput) -> dict[str, Any]:
        now = utc_now()
        project = _normalized_project(
            app_settings,
            payload,
            project_id=uuid.uuid4().hex[:12],
            created_at=now,
        )
        return registry.add_record("projects", project)

    @app.put("/api/projects/{project_id}")
    def update_project(project_id: str, payload: ProjectInput) -> dict[str, Any]:
        existing = registry.get_record("projects", project_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Project not found")
        project = _normalized_project(
            app_settings,
            payload,
            project_id=project_id,
            created_at=existing["created_at"],
        )
        return registry.replace_record("projects", project_id, project)

    @app.delete("/api/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_project(project_id: str) -> Response:
        try:
            registry.delete_record("projects", project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found") from None
        return Response(status_code=204)

    @app.get("/api/runs")
    def list_runs(project_id: str | None = None) -> list[dict[str, Any]]:
        runs = registry.list_records("runs")
        if project_id:
            runs = [run for run in runs if run.get("project_id") == project_id]
        return sorted(
            runs,
            key=lambda item: item.get("timestamps", {}).get("created_at", ""),
            reverse=True,
        )

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = registry.get_record("runs", run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return detailed_run(app_settings, run)

    @app.post("/api/runs", status_code=status.HTTP_202_ACCEPTED)
    def create_run(payload: RunCreate) -> dict[str, Any]:
        project = registry.get_record("projects", payload.project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        try:
            return manager.enqueue_project(
                project,
                requested_exp_name=payload.requested_exp_name,
                option_patch=payload.options,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            return manager.cancel(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/stop", status_code=status.HTTP_202_ACCEPTED)
    def stop_run(run_id: str) -> dict[str, Any]:
        try:
            return manager.stop(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/requeue", status_code=status.HTTP_202_ACCEPTED)
    def requeue_run(run_id: str) -> dict[str, Any]:
        try:
            return manager.requeue(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/events")
    def run_events(
        run_id: str, tail: int = Query(default=200, ge=1, le=2000)
    ) -> dict[str, Any]:
        run = registry.get_record("runs", run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return read_events(app_settings, run, tail=tail)

    @app.get("/api/runs/{run_id}/log", response_class=PlainTextResponse)
    def run_log(
        run_id: str, tail: int = Query(default=400, ge=1, le=10_000)
    ) -> PlainTextResponse:
        run = registry.get_record("runs", run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return PlainTextResponse(tail_text(_console_path(app_settings, run), tail))

    @app.get("/artifacts/runs/{run_id}/tree_plot.html", include_in_schema=False)
    def tree_plot(run_id: str) -> FileResponse:
        run = registry.get_record("runs", run_id)
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
        run = registry.get_record("runs", run_id)
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
