"""Run-level manifest writing.

The manifest records fixed facts about one BioMaker run. It is not a prompt and
does not make decisions for the search agent.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
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


def _monai_rag_index_provenance() -> dict[str, Any] | None:
    manifest_path = (
        Path(__file__).resolve().parent.parent
        / "monai_rag"
        / "monai_rag_db"
        / "manifest.json"
    )
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        return None
    fields = (
        "schema_version",
        "version",
        "document_versions",
        "mode",
        "embedding_model",
        "record_count",
        "dimension",
        "embedding_dtype",
        "chunks_sha256",
        "embeddings_sha256",
    )
    return {field: manifest.get(field) for field in fields}


def _selected_cfg(cfg: Config) -> dict[str, Any]:
    knowledge_cfg = cfg.agent.knowledge_base
    requested_workers = getattr(
        cfg.exec, "requested_parallel_workers", cfg.exec.parallel_workers
    )
    effective_workers = getattr(cfg.exec, "effective_parallel_workers", None)
    return {
        "agent_steps": cfg.agent.steps,
        "total_time_limit_seconds": cfg.agent.total_time_limit,
        "per_node_timeout_seconds": cfg.exec.timeout,
        "parallel_workers": effective_workers or cfg.exec.parallel_workers,
        "requested_parallel_workers": requested_workers,
        "effective_parallel_workers": effective_workers,
        "compute_slots": getattr(cfg.exec, "compute_slots", None),
        "final_retrain_enabled": cfg.final_retrain.enabled,
        "model_for_all_agents": cfg.model_for_all_agents,
        "main_model": cfg.agent.code.model,
        "feedback_model": cfg.agent.feedback.model,
        "knowledge_base_model": cfg.agent.knowledge_base.model,
        "report_model": cfg.report.model,
        "enable_node_tools": cfg.agent.enable_node_tools,
        "enable_monai_knowledge_base": cfg.agent.enable_monai_knowledge_base,
        "monai_rag_mode": getattr(knowledge_cfg, "retrieval_mode", "keyword"),
        "monai_rag_embedding_model": getattr(
            knowledge_cfg, "embedding_model", "text-embedding-3-large"
        ),
        "monai_rag_topk": getattr(knowledge_cfg, "topk", 5),
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
    out_path = Path(cfg.log_dir) / "run_manifest.json"
    existing = _read_json(out_path)
    created_at = (
        existing.get("created_at")
        if isinstance(existing, dict) and existing.get("created_at")
        else datetime.now(timezone.utc).isoformat()
    )
    spec = prediction_spec or _read_json(
        Path(cfg.workspace_dir) / "evaluator_prediction_spec.json"
    )
    manifest = {
        "created_at": created_at,
        "launch_token": os.getenv("BIOMAKER_LAUNCH_TOKEN") or None,
        "exp_name": cfg.exp_name,
        "data_dir": str(cfg.data_dir),
        "workspace_dir": str(cfg.workspace_dir),
        "log_dir": str(cfg.log_dir),
        "task_description": task_desc,
        "split_protocol": split_protocol_text,
        "split_files": _split_files(Path(cfg.workspace_dir)),
        "prediction_spec": spec,
        "config": _selected_cfg(cfg),
        "monai_rag_index": _monai_rag_index_provenance(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "facts": environment_manifest_text(
                include_medical=cfg.exec.include_medical_packages
            ),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, out_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return out_path
