"""Run-level manifest writing.

The manifest records fixed facts about one BioMaker run. It is not a prompt and
does not make decisions for the search agent.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .environment import environment_manifest_text


def _read_json(path: Path) -> Any:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _split_files(workspace_dir: Path) -> dict[str, str]:
    split_dir = workspace_dir / "input" / "biomaker_splits"
    if not split_dir.is_dir():
        return {}
    return {
        path.stem: str(path.relative_to(workspace_dir))
        for path in sorted(split_dir.glob("*.csv"))
    }


def _selected_cfg(cfg: Config) -> dict[str, Any]:
    return {
        "agent_steps": cfg.agent.steps,
        "total_time_limit_seconds": cfg.agent.total_time_limit,
        "per_node_timeout_seconds": cfg.exec.timeout,
        "main_model": cfg.agent.code.model,
        "feedback_model": cfg.agent.feedback.model,
        "enable_node_tools": cfg.agent.enable_node_tools,
        "enable_monai_knowledge_base": cfg.agent.enable_monai_knowledge_base,
        "enable_web_search": cfg.agent.enable_web_search,
        "copy_data": cfg.copy_data,
        "preprocess_data": cfg.preprocess_data,
    }


def write_run_manifest(
    cfg: Config,
    task_desc: Any,
    *,
    split_protocol_text: str | None = None,
    prediction_spec: dict[str, Any] | None = None,
) -> Path:
    """Write run_manifest.json with stable run facts."""
    spec = prediction_spec or _read_json(
        Path(cfg.workspace_dir) / "evaluator_prediction_spec.json"
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exp_name": cfg.exp_name,
        "data_dir": str(cfg.data_dir),
        "workspace_dir": str(cfg.workspace_dir),
        "log_dir": str(cfg.log_dir),
        "task_description": task_desc,
        "split_protocol": split_protocol_text,
        "split_files": _split_files(Path(cfg.workspace_dir)),
        "prediction_spec": spec,
        "config": _selected_cfg(cfg),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "facts": environment_manifest_text(
                include_medical=cfg.exec.include_medical_packages
            ),
        },
    }
    out_path = Path(cfg.log_dir) / "run_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return out_path
