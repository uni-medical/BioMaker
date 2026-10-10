"""Structured evaluator receipts must preserve the same validity as runtime scores."""
from __future__ import annotations

import json
import pytest

from biomaker.agents.base_agent import parse_factual_exec_result
from biomaker.interpreter import ExecutionResult
from biomaker.journal import Journal, Node
from biomaker.utils import evaluator_setup as evaluator


def _parse_structured(payload):
    node = Node(code="print('candidate')")
    execution = ExecutionResult(
        # A rejected structured receipt must never fall back to this valid log line.
        term_out=[evaluator.EXTERNAL_VALIDATION_PREFIX + '{"metric": 0.99}'],
        exec_time=0.1,
        exc_type=None,
        evaluator_result=payload,
    )
    parse_factual_exec_result(node, execution)
    return node


@pytest.mark.parametrize("overrides", [
    {"details": {"score": {"fallback_used": True}}},
    {"details": {"score": {"metadata": [{"fallback": "chance"}]}}},
    {"details": {"score": {"ok": False}}},
    {"details": {"score": {"valid": False}}},
    {"details": {"score": {"error": "undefined metric"}}},
    {"ok": False}, {"ok": 1}, {"ok": "true"}, {"ok": None},
    {"metric": "0.9"}, {"metric": True}, {"metric": None},
    {"metric": float("nan")}, {"metric": float("inf")},
])
def test_structured_invalid_receipt_marks_candidate_buggy(overrides):
    node = _parse_structured({"ok": True, "metric": 0.9, **overrides})
    assert node.is_buggy is True
    assert node.metric.value is None
    assert node.outcome == "bug"
    assert Journal(nodes=[node]).get_best_node() is None


def test_structured_receipt_requires_explicit_true_ok():
    node = _parse_structured({"metric": 0.9})
    assert node.is_buggy is True
    assert node.metric.value is None


@pytest.mark.parametrize("direction", [None, 0, 1, "false", "true", [], {}])
def test_malformed_metric_direction_is_rejected_by_receipt_and_legacy_parsers(direction):
    score = {"metric": 0.9, "maximize": direction}
    structured = _parse_structured({"ok": True, **score})
    assert structured.is_buggy is True
    assert structured.evaluator_details["maximize"] == direction
    nested = _parse_structured({"ok": True, "metric": 0.9, "details": {"score": score}})
    assert nested.is_buggy is True
    assert evaluator.parse_external_validation(evaluator.EXTERNAL_VALIDATION_PREFIX + json.dumps(score)) is None
    spec = {"test_submission": "./working/submission.csv", "key_columns": ["id"], "prediction_columns": ["prob"], "primary_metric": "auc", "maximize": direction}
    with pytest.raises(evaluator.EvaluatorSetupError, match="maximize must be a boolean"):
        evaluator._normalize_prediction_spec_json(json.dumps(spec))


@pytest.mark.parametrize("score,expected_direction", [
    ({"metric": 0.9}, True),
    ({"metric": 0.9, "maximize": True}, True),
    ({"metric": 0.9, "maximize": False}, False),
    ({"metric": 1}, True),
])
def test_valid_structured_scores_preserve_direction_and_missing_default(score, expected_direction):
    node = _parse_structured({"ok": True, **score, "details": {"score": score}})
    assert node.is_buggy is False
    assert node.metric.value == score["metric"]
    assert node.metric.maximize is expected_direction
    assert evaluator.parse_external_validation(evaluator.EXTERNAL_VALIDATION_PREFIX + json.dumps(score)) == (score["metric"], expected_direction)


def test_typed_score_diagnostics_cannot_override_official_validity():
    node = _parse_structured({
        "ok": True, "outcome": "valid", "metric": 0.8,
        "details": {"score": {
            "score_schema_version": 1, "status": "ok", "failure": None,
            "metrics": {"accuracy": 0.8},
            "diagnostics": {"error": "diagnostic measurement", "fallback_used": True},
        }},
    })
    assert node.is_buggy is False
    assert node.metrics == {"accuracy": 0.8}


def test_typed_unavailable_evaluator_keeps_candidate_unscored():
    node = _parse_structured({
        "ok": False, "outcome": "evaluator_unavailable",
        "metric": None, "failure_kind": "timeout", "error": "scoring timeout",
    })
    assert node.is_buggy is None
    assert node.outcome == "unscored"
    assert node.metrics == {} and node.metric is None
    assert Journal(nodes=[node]).get_best_node() is None


def test_typed_invalid_prediction_cannot_publish_metrics():
    node = _parse_structured({
        "ok": False, "outcome": "invalid_prediction", "metric": 0.9,
        "metrics": {"accuracy": 0.9}, "failure_kind": "prediction_contract",
    })
    assert node.is_buggy is True
    assert node.metrics == {}
    assert Journal(nodes=[node]).get_best_node() is None
