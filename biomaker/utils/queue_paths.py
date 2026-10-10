"""Deterministic filesystem roots shared by the BioMaker queue and WebUI."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

STATE_DIR_ENV = "BIOMAKER_WEBUI_STATE_DIR"
WORKSPACE_ROOT_ENV = "BIOMAKER_WEBUI_WORKSPACE_ROOT"


@dataclass(frozen=True, slots=True)
class QueuePaths:
    """Resolved roots that define one local BioMaker queue domain."""

    workspace_root: Path
    state_dir: Path
    logs_root: Path
    workspaces_root: Path
    experiments_root: Path
    datasets_root: Path


def _resolve_path(value: str | os.PathLike[str], *, anchor: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path.resolve()


def resolve_queue_paths(
    method_root: str | os.PathLike[str],
    *,
    log_dir: str | os.PathLike[str] = "logs",
    workspace_dir: str | os.PathLike[str] = "workspaces",
    environment: Mapping[str, str] | None = None,
) -> QueuePaths:
    """Resolve queue state and artifact roots without depending on the CWD.

    ``BIOMAKER_WEBUI_STATE_DIR`` directly selects the ledger directory. Without
    it, the ledger is ``.biomaker`` under ``BIOMAKER_WEBUI_WORKSPACE_ROOT`` or,
    by default, the parent of the method repository. Installed wheels instead
    use ``$XDG_DATA_HOME/biomaker`` (or ``~/.local/share/biomaker``). Relative state and artifact
    paths are anchored to that shared workspace; explicit absolute paths remain
    unchanged.
    """

    env = os.environ if environment is None else environment
    resolved_method_root = Path(method_root).expanduser().resolve()
    # Installed wheels live in a shared interpreter directory, which must never
    # receive user queue state or experiment artifacts.
    if resolved_method_root.name in {"site-packages", "dist-packages"}:
        data_home = Path(
            env.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
        ).expanduser()
        if not data_home.is_absolute():
            data_home = Path.home() / ".local" / "share"
        default_workspace_root = (data_home / "biomaker").resolve()
    else:
        default_workspace_root = resolved_method_root.parent
    raw_workspace_root = env.get(WORKSPACE_ROOT_ENV)
    workspace_root = (
        _resolve_path(raw_workspace_root, anchor=default_workspace_root)
        if raw_workspace_root
        else default_workspace_root
    )

    raw_state_dir = env.get(STATE_DIR_ENV)
    state_dir = (
        _resolve_path(raw_state_dir, anchor=workspace_root)
        if raw_state_dir
        else (workspace_root / ".biomaker").resolve()
    )
    return QueuePaths(
        workspace_root=workspace_root,
        state_dir=state_dir,
        logs_root=_resolve_path(log_dir, anchor=workspace_root),
        workspaces_root=_resolve_path(workspace_dir, anchor=workspace_root),
        experiments_root=(workspace_root / "experiments").resolve(),
        datasets_root=(workspace_root / "datasets").resolve(),
    )
