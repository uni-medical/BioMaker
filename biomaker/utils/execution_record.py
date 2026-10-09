"""Helpers for attaching BioMaker post-processing results to executions."""

from __future__ import annotations

from typing import Any

from ..interpreter import ExecutionResult
from .artifact_validation import ArtifactValidationResult
from .evaluator_setup import EvaluatorRunResult


def attach_artifact_validation_result(
    exec_result: ExecutionResult,
    artifact_check: ArtifactValidationResult,
    *,
    failure_reason: str,
) -> None:
    """Attach deterministic artifact validation facts to an execution."""
    exec_result.artifact_validation = {
        "ok": artifact_check.ok,
        "messages": artifact_check.messages,
    }
    if getattr(artifact_check, "model_artifact", None) is not None:
        exec_result.artifact_validation["model_artifact"] = (
            artifact_check.model_artifact
        )
    if artifact_check.messages:
        exec_result.append_system_output(
            "artifact_validation",
            "\n".join(artifact_check.messages),
            details=exec_result.artifact_validation,
        )
    if not artifact_check.ok:
        exec_result.exc_type = "ArtifactValidationError"
        exec_result.exc_info = {"reason": failure_reason}
        exec_result.failure_phase = "artifact_validation"


def attach_evaluator_run_result(
    exec_result: ExecutionResult,
    evaluator_result: EvaluatorRunResult,
    *,
    failure_phase: str,
    failure_type: str,
    failure_message_prefix: str,
    extra_details: dict[str, Any] | None = None,
) -> None:
    """Attach frozen evaluator facts to an execution."""
    details = {
        "ok": evaluator_result.ok,
        "metric": evaluator_result.metric,
        "maximize": evaluator_result.maximize,
        "details": evaluator_result.details,
        "error": evaluator_result.error,
    }
    if extra_details:
        details.update(extra_details)
    exec_result.evaluator_result = details

    if evaluator_result.output:
        exec_result.append_system_output(
            "evaluator_output",
            evaluator_result.output,
            details={
                "ok": evaluator_result.ok,
                "metric": evaluator_result.metric,
                "maximize": evaluator_result.maximize,
            },
        )
    if evaluator_result.ok:
        exec_result.append_system_output(
            "evaluator_result",
            evaluator_result.log_line(),
            details=exec_result.evaluator_result,
        )
        return

    exec_result.exc_type = failure_type
    exec_result.exc_info = {"reason": evaluator_result.error}
    exec_result.failure_phase = failure_phase
    exec_result.append_system_output(
        failure_phase,
        f"{failure_message_prefix}: "
        + (evaluator_result.error or "external evaluator failed"),
        details=exec_result.evaluator_result,
    )
