from __future__ import annotations

from pathlib import Path

import pytest

from webui.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    workspace_root = tmp_path / "workspace"
    method_root = workspace_root / "BioMaker"
    state_dir = method_root / "apps" / "webui" / "state"
    for path in (
        method_root,
        workspace_root / "logs",
        workspace_root / "workspaces",
        workspace_root / "experiments",
        workspace_root / "datasets",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return Settings(
        method_root=method_root,
        workspace_root=workspace_root,
        state_dir=state_dir,
        logs_root=workspace_root / "logs",
        workspaces_root=workspace_root / "workspaces",
        experiments_root=workspace_root / "experiments",
        datasets_root=workspace_root / "datasets",
        env_file=workspace_root / ".env",
        stop_grace_seconds=0.1,
    )
