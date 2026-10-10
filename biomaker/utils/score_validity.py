"""Canonical scoring outcomes and a bounded legacy-result compatibility boundary."""

import math
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ScoringOutcome:
    status: Literal["valid", "invalid_prediction", "evaluator_unavailable"]
    metrics: dict[str, float]
    failure_kind: str | None = None
    error: str | None = None

    def __post_init__(self):
        if self.status not in {"valid", "invalid_prediction", "evaluator_unavailable"}:
            raise ValueError("Unknown scoring outcome")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("Ranking metrics must be finite numbers")
        if self.status != "valid" and self.metrics:
            raise ValueError("A failed scoring outcome cannot publish ranking metrics")


def require_valid_score_payload(value: Any) -> None:
    """New envelopes have explicit semantics; legacy markers are interpreted once.

    A new envelope's diagnostics are opaque observations, not result-status fields.
    Historical untyped evaluators retain their conservative marker adapter.
    """
    if not isinstance(value, dict):
        raise ValueError("Scoring must return an object")
    if value.get("score_schema_version") == 1:
        if value.get("status") != "ok" or value.get("failure") is not None:
            raise ValueError("Scoring did not explicitly succeed")
        if any(value.get(name) for name in ("error", "fallback", "fallback_used", "used_fallback")) or value.get("valid") is False or value.get("ok") is False:
            raise ValueError("Scoring returned contradictory result markers")
    elif "score_schema_version" in value:
        raise ValueError("Unknown score schema version")
    elif has_invalid_result(value):
        raise ValueError("Legacy scoring returned a failure or fallback result")


def require_submission_success(value: Any) -> None:
    if value is True:
        return
    if isinstance(value, dict) and not has_invalid_result(value):
        if value.get("valid") is True or value.get("ok") is True:
            return
    raise ValueError("validate_submission() did not explicitly confirm valid predictions")


def has_fallback_marker(value: object) -> bool:
    if isinstance(value, dict):
        if any(
            value.get(key) for key in ("fallback", "fallback_used", "used_fallback")
        ):
            return True
        return any(has_fallback_marker(child) for child in value.values())
    if isinstance(value, list):
        return any(has_fallback_marker(child) for child in value)
    return False


def has_invalid_result(value: object) -> bool:
    """An explicit failure anywhere in a metric result cannot become a score."""
    if isinstance(value, dict):
        if (
            any(
                value.get(key)
                for key in ("fallback", "fallback_used", "used_fallback", "error")
            )
            or value.get("valid") is False
            or value.get("ok") is False
            or ("maximize" in value and not isinstance(value["maximize"], bool))
        ):
            return True
        return any(has_invalid_result(child) for child in value.values())
    if isinstance(value, list):
        return any(has_invalid_result(child) for child in value)
    return False
