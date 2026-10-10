from types import SimpleNamespace

from biomaker.journal import Journal, Node, _format_resource_usage
from biomaker.journal2report import journal2report
from biomaker.utils.metric import MetricValue
from biomaker.interpreter import _format_usage_line


def candidate(id, value=0.9, **kwargs):
    return Node(
        id=id,
        code="# generated",
        metric=MetricValue(value, maximize=True),
        is_buggy=False,
        exec_time=12.5,
        **kwargs,
    )


def render(*nodes):
    journal = Journal()
    for node in nodes:
        journal.append(node)
    return journal2report(journal, {"task": "Classify images"}, SimpleNamespace())


def test_report_preserves_failed_attempts_and_nested_validation_evidence():
    nodes = [candidate(f"valid-{i}", 0.6 + i * 0.01) for i in range(7)]
    winner = candidate(
        "winner",
        0.9272656048223907,
        evaluator_details={
            "details": {
                "score": {
                    "metric": 0.9272656048223907,
                    "per_class_auroc": {"0": 0.934, "6": 0.981},
                    "accuracy": 0.7777,
                }
            }
        },
        resource_usage={
            "peak_rss_mb": 2377.65625,
            "peak_gpu_memory_mb": 2377.65625,
            "gpu_backend": "mps",
            "gpu_memory_source": "unified_memory_rss_proxy",
        },
        execution={
            "candidate": {"stdout": "DONE pipeline=frozen MobileNetV3 + LightGBM"}
        },
    )
    failed = Node(
        id="failed-missing-split",
        code="",
        is_buggy=True,
        exc_type="ArtifactValidationError",
        exc_info={"missing": ["split"]},
    )
    report = render(*nodes, winner, failed)
    assert "Recorded attempts: **9**" in report
    assert "| #9 | failed-missing-split | failed / invalid | not valid" in report
    assert "ArtifactValidationError" in report
    assert "Best valid recorded exploration candidate: **#8**" in report
    assert "per_class_auroc.6 | 0.981" in report
    assert "DONE pipeline=frozen MobileNetV3 + LightGBM" in report
    assert "GPU memory not independently measured" in report
    assert "peak_mps" not in report
    assert "0.927265604822" in report


def test_reports_are_deterministic_and_work_without_provider(monkeypatch):
    monkeypatch.setattr(
        "biomaker.backend.query",
        lambda **kw: (_ for _ in ()).throw(AssertionError("network")),
    )
    node = candidate("same")
    assert render(node) == render(node)


def test_report_never_ranks_nan_fallback_or_failed_scores():
    fallback = candidate(
        "fallback", evaluator_details={"details": {"score": {"fallback_used": True}}}
    )
    nan = candidate("nan", float("nan"))
    failed = candidate("failed", 0.999, exc_type="ArtifactValidationError")
    report = render(fallback, nan, failed)
    assert "Valid scored attempts: **0**" in report
    assert "No valid scored candidate is recorded" in report
    assert "invalid: fallback metric" in report
    assert "Best valid" not in report
    journal = Journal(nodes=[fallback, nan, failed])
    assert journal.get_best_node() is None
    assert journal._reference_nodes(5) == []


def test_empty_and_mixed_direction_journals_do_not_invent_selection():
    assert "Recorded attempts: **0**" in render()
    first, second = candidate("a"), candidate("b")
    second.metric.maximize = False
    assert "No cross-candidate ranking" in render(first, second)


def test_resource_source_is_preserved_for_all_measurement_kinds():
    assert "total_delta_approx" in _format_resource_usage(
        {
            "peak_gpu_memory_mb": 12,
            "gpu_memory_source": "nvidia-smi_total_used_delta_approx",
        }
    )
    assert "process_tree" in _format_resource_usage(
        {
            "peak_gpu_memory_mb": 12,
            "gpu_memory_source": "nvidia-smi_compute_apps_process_tree",
        }
    )
    assert "source unconfirmed" in _format_resource_usage({"peak_gpu_memory_mb": 12})
    line = _format_usage_line(
        {
            "peak_rss_mb": 12,
            "peak_gpu_memory_mb": 12,
            "gpu_backend": "mps",
            "gpu_memory_source": "unified_memory_rss_proxy",
        }
    )
    assert "not independently measured" in line
    assert "peak_mps" not in line


def test_report_redacts_credentials_and_quotes_untrusted_plans(monkeypatch):
    monkeypatch.setenv("REPORT_QA_API_KEY", "synthetic-credential.with/punctuation=")
    report = render(
        candidate(
            "a",
            plan="```\n# invented header\n```",
            execution={
                "candidate": {"stdout": "synthetic-credential.with/punctuation="}
            },
        )
    )
    assert "synthetic-credential.with/punctuation=" not in report
    assert "[REDACTED]" in report
    assert "````text\n```\n# invented header\n```\n````" in report


def test_report_marks_truncated_output_and_retains_tail():
    report = render(
        candidate(
            "a", execution={"candidate": {"stdout": "begin\n" + "x" * 20000 + "\nend"}}
        )
    )
    assert "omitted; see saved journal/code/output for the full record" in report
    assert "begin" in report and "end" in report


def test_validation_table_excludes_private_subtrees():
    report = render(
        candidate(
            "a",
            evaluator_details={
                "details": {
                    "score": {
                        "accuracy": 0.8,
                        "private_metrics": {"test_auroc": "private-sentinel"},
                    }
                }
            },
        )
    )
    assert "accuracy | 0.8" in report
    assert "private-sentinel" not in report


def test_saved_model_receipt_is_reported_without_inventing_legacy_success():
    checked = candidate(
        "checked",
        execution={
            "artifact_validation": {
                "model_artifact": {
                    "ok": True,
                    "framework": "sklearn",
                    "checkpoint_count": 1,
                    "probe_examples": 2,
                    "output_shape": [2, 3],
                    "guard_scope": "python_audit_and_profile_not_os_sandbox",
                }
            }
        },
    )
    report = render(checked, candidate("legacy"))
    assert '"checkpoint_count": 1' in report
    assert '"probe_examples": 2' in report
    assert "python_audit_and_profile_not_os_sandbox" in report
    assert "No supervised model reload receipt is recorded" in report


def test_runtime_ranking_and_report_share_invalid_result_semantics():
    from biomaker.utils.evaluator_setup import EXTERNAL_VALIDATION_PREFIX, parse_external_validation
    import json

    assert parse_external_validation(EXTERNAL_VALIDATION_PREFIX + json.dumps({'metric': 0.99, 'maximize': True})) == (0.99, True)

    examples = [
        {"fallback": True},
        {"used_fallback": True},
        {"fallback_used": "yes"},
        {"details": {"fallback_used": True}},
        {"error": "undefined"},
        {"valid": False},
        {"details": {"error": "undefined"}},
        {"details": {"ok": False}},
    ]
    for markers in examples:
        score = {"metric": 0.99, "maximize": True, **markers}
        node = candidate(
            "invalid-marker", 0.99, evaluator_details={"details": {"score": score}}
        )
        assert Journal(nodes=[node]).get_best_node() is None, markers
        assert "Best valid" not in render(node), markers
        assert (
            parse_external_validation(EXTERNAL_VALIDATION_PREFIX + json.dumps(score))
            is None
        ), markers
