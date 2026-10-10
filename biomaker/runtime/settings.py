from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .env_file import resolve_env_file

from biomaker.utils.queue_paths import resolve_queue_paths


@dataclass(frozen=True, slots=True)
class Settings:
    method_root: Path
    workspace_root: Path
    state_dir: Path
    logs_root: Path
    workspaces_root: Path
    experiments_root: Path
    datasets_root: Path
    env_file: Path
    uv_executable: str = "uv"
    stop_grace_seconds: float = 8.0

    @classmethod
    def default(cls) -> Settings:
        method_root = (
            Path(
                os.environ.get(
                    "BIOMAKER_WEBUI_METHOD_ROOT",
                    Path(__file__).resolve().parents[2],
                )
            )
            .expanduser()
            .resolve()
        )
        queue_paths = resolve_queue_paths(method_root)
        env_file = resolve_env_file(method_root)
        return cls(
            method_root=method_root,
            workspace_root=queue_paths.workspace_root,
            state_dir=queue_paths.state_dir,
            logs_root=queue_paths.logs_root,
            workspaces_root=queue_paths.workspaces_root,
            experiments_root=queue_paths.experiments_root,
            datasets_root=queue_paths.datasets_root,
            env_file=env_file,
            uv_executable=os.environ.get("BIOMAKER_WEBUI_UV", "uv"),
        )

    def ensure_directories(self) -> None:
        for path in (
            self.state_dir,
            self.state_dir / "tasks",
            self.state_dir / "console",
            self.logs_root,
            self.workspaces_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
