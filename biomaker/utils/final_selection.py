"""Freeze validation selection and evaluate one delivered model once."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .artifact_integrity import artifact_snapshot
from .config import evaluator_timeout_seconds, evaluator_workers
from .evaluator_setup import EvaluatorInfrastructureError
from .events import append_event
from .experiment_contract import load_experiment_contract
from .private_evaluation import evaluate_private_outputs


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _artifact_digest(workspace, snapshot):
    return _digest({str(Path(k).relative_to(workspace)): v[-1] if v else None for k, v in snapshot.items()})


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def freeze_selection(cfg, journal, best_node):
    if best_node is None:
        raise RuntimeError("No valid scored candidate exists for final selection.")
    workspace = Path(best_node.execution_workspace or "")
    if not best_node.execution_workspace or not workspace.is_dir():
        raise RuntimeError("Selected candidate workspace is missing; final delivery cannot use another candidate's outputs.")
    contract = load_experiment_contract(workspace)
    ranking = journal.ranking_snapshot()
    evidence = artifact_snapshot(workspace)
    payload = {
        "version": 1, "protocol": "post_selection_once", "node_id": best_node.id,
        "node_step": best_node.step, "ranking_revision": ranking["version"],
        "validation_selection": ranking,
        "code_sha256": hashlib.sha256(best_node.code.encode()).hexdigest(),
        "selected_artifact_digest": _artifact_digest(workspace, evidence),
        "contract_digest": _digest(contract.prediction.raw),
        "evaluator_sha256": (evidence.get(str(workspace / "evaluator.py")) or [None])[-1],
    }
    payload["selection_id"] = _digest(payload)
    path = cfg.log_dir / "selection.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError("An existing frozen selection disagrees with this candidate or contract.")
    else:
        _atomic_json(path, payload)
    append_event(cfg.log_dir, "selection_frozen", node_id=best_node.id, selection_id=payload["selection_id"], ranking_revision=ranking["version"])
    return payload


def evaluate_selected_delivery(cfg, best_node, workspace_dir, private_dir, selection, *, stage):
    """Reserve the evaluation before execution; retries cannot rescore test labels."""
    if stage not in {"selected_candidate", "final_retrain"}:
        raise ValueError("Unknown final delivery stage")
    workspace = Path(workspace_dir)
    if not workspace.is_dir():
        raise RuntimeError("Selected delivery workspace is missing.")
    contract = load_experiment_contract(workspace, hidden_answers_dir=private_dir)
    snapshot = artifact_snapshot(workspace)
    artifact_digest = _artifact_digest(workspace, snapshot)
    evaluator_digest = (snapshot.get(str(workspace / "evaluator.py")) or [None])[-1]
    if evaluator_digest != selection["evaluator_sha256"] or _digest(contract.prediction.raw) != selection["contract_digest"]:
        raise RuntimeError("Final delivery disagrees with the frozen evaluator or contract.")
    if stage == "selected_candidate" and artifact_digest != selection["selected_artifact_digest"]:
        raise RuntimeError("Selected candidate artifacts changed after selection was frozen.")
    entry = {
        "node_id": best_node.id, "node_step": best_node.step, "stage": stage,
        "selection_id": selection["selection_id"], "artifact_digest": artifact_digest,
        "contract_digest": _digest(contract.prediction.raw),
        "evaluator_sha256": (snapshot.get(str(workspace / "evaluator.py")) or [None])[-1],
        "ranking_revision": selection["ranking_revision"], "is_current_best": True,
        "private_metrics_available": False, "private_metrics": None,
        "private_error": None, "private_failure_kind": None,
    }
    receipt_path = cfg.log_dir / "private_metrics.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        prior = previous.get("latest", {})
        if previous.get("protocol") != "post_selection_once" or prior.get("selection_id") != entry["selection_id"] or prior.get("artifact_digest") != artifact_digest:
            raise RuntimeError("An existing private evaluation belongs to different selected artifacts.")
        if prior.get("status") in {"completed", "skipped"}:
            return prior
        raise EvaluatorInfrastructureError("Private evaluation was already attempted; retained artifacts require explicit recovery without repeating selection.", failure_kind=prior.get("private_failure_kind") or "interrupted")

    def save():
        _atomic_json(receipt_path, {
            "version": 2, "protocol": "post_selection_once", "visible_to_agent": False,
            "validation_selection": selection["validation_selection"],
            "latest": entry, "history": [entry],
        })

    job = contract.prediction_jobs.get("test")
    hidden = contract.hidden_answers.get("test")
    if job is None or hidden is None:
        entry.update(status="skipped", skip_reason="no_test_job" if job is None else "no_private_test_answers")
        save()
        append_event(cfg.log_dir, "private_evaluation_skipped", node_id=best_node.id, reason=entry["skip_reason"])
        return entry
    entry["status"] = "started"
    save()
    append_event(cfg.log_dir, "private_evaluation_started", node_id=best_node.id, stage=stage)
    try:
        result = evaluate_private_outputs(workspace, private_dir,
            timeout=evaluator_timeout_seconds(cfg), env_overrides={
                "CUDA_VISIBLE_DEVICES": "", "BIOMAKER_EVALUATOR_WORKERS": str(evaluator_workers(cfg)),
                "BIOMAKER_EVALUATOR_PROGRESS_PATH": str(cfg.log_dir / "final-private-scoring.jsonl"),
            })
        if result is None or result.get("error") or not result.get("results"):
            entry.update(status="failed", private_error=(result or {}).get("error", "Private evaluation returned no result."), private_failure_kind=(result or {}).get("failure_kind", "protocol"))
        else:
            entry.update(status="completed", private_metrics_available=True, private_metrics=result)
    except Exception as exc:
        entry.update(status="failed", private_error=str(exc), private_failure_kind=getattr(exc, "failure_kind", "execution"))
    save()
    append_event(cfg.log_dir, "private_evaluation_finished", node_id=best_node.id, status=entry["status"], failure_kind=entry["private_failure_kind"])
    if entry["status"] != "completed":
        raise EvaluatorInfrastructureError("Selected model and predictions were retained, but final private evaluation did not complete: " + str(entry["private_error"]), failure_kind=entry["private_failure_kind"])
    return entry
