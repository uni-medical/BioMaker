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


def evaluate_private_outputs(
    workspace_dir: Path | str,
    private_dir: Path | str,
    *,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    workspace_dir, private_dir = Path(workspace_dir), Path(private_dir)
    contract = load_experiment_contract(workspace_dir, hidden_answers_dir=private_dir)
    hidden = contract.hidden_answers.get("test")
    if hidden is None:
        return None
    submission = contract.prediction.test_submission
    if not submission.is_file():
        return None
    payload: dict[str, Any] = {
        "visible_to_agent": False,
        "note": "Private metrics are for user inspection only and are not passed to the agent.",
        "results": [],
    }
    expected = contract.split_path("test")
    if expected is None or not hidden.path.is_file():
        payload["error"] = "Missing private test answers or public test sample list."
        return payload
    result = score_artifacts_with_evaluator(
        workspace_dir,
        submission,
        hidden.path,
        expected,
        timeout=120,
        env_overrides=env_overrides,
    )
    if not result.ok:
        payload["error"] = result.error or "Frozen evaluator private scoring failed."
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
