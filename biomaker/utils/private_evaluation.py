"""User-only test evaluation through the run's frozen scoring implementation.

This module selects artifacts and packages results. It does not infer task types,
interpret prediction cells, or implement metrics. Test labels are supplied only
to the evaluator subprocess and its results are never used for solution search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluator_setup import score_artifacts_with_evaluator
from .experiment_contract import load_experiment_contract
from .split_contract import SplitContractError


def evaluate_private_outputs(
    workspace_dir: Path | str,
    private_dir: Path | str,
    *,
    timeout: int = 3600,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    workspace_dir, private_dir = Path(workspace_dir), Path(private_dir)
    payload: dict[str, Any] = {
        "visible_to_agent": False,
        "note": "Private metrics are for user inspection only and are not passed to the agent.",
        "results": [],
    }
    try:
        contract = load_experiment_contract(workspace_dir, hidden_answers_dir=private_dir)
    except SplitContractError as exc:
        payload.update(error=str(exc), failure_kind="protocol")
        return payload
    job = contract.prediction_jobs.get("test")
    if job is None:
        return None
    hidden = contract.hidden_answers.get("test")
    if hidden is None:
        return None
    if not job.expected.is_file() or not hidden.path.is_file():
        payload["error"] = "Missing private test answers or declared public test sample list."
        payload["failure_kind"] = "protocol"
        return payload
    if not job.artifact.is_file():
        payload["error"] = "Missing declared test prediction artifact."
        payload["failure_kind"] = "protocol"
        return payload
    result = score_artifacts_with_evaluator(
        workspace_dir,
        job.artifact,
        hidden.path,
        job.expected,
        timeout=timeout,
        env_overrides=env_overrides,
    )
    if not result.ok:
        payload["error"] = result.error or "Frozen evaluator private scoring failed."
        payload["failure_kind"] = result.failure_kind
        return payload
    score = result.details["score"]
    item = {
        "name": "private_test",
        "source": hidden.path.name,
        "visible_to_agent": False,
        "metrics": score["metrics"],
        "metric_specs": contract.metric.metrics,
        "n": score.get("n"),
    }
    definitions = contract.metric.metrics
    if contract.prediction.raw.get("evaluator_api_version", 2) == 2:
        # Original frozen v2 runs retain their scalar reporting contract.
        item.update(
            metric_name="private_" + str(contract.metric.primary_metric),
            value=result.metric,
            maximize=result.maximize,
        )
    elif len(definitions) == 1:
        spec = definitions[0]
        item.update(
            metric_name="private_" + spec["name"],
            value=score["metrics"].get(spec["name"]),
            maximize=spec["maximize"],
        )
    payload["results"] = [item]
    return payload
