import copy
import json
from types import SimpleNamespace

import pytest

from biomaker.agents.base_agent import BaseAgent, parse_factual_exec_result
from biomaker.interpreter import ExecutionResult
from biomaker.journal import Journal, Node
from biomaker.utils.metric import MetricValue
from biomaker.utils.ranking import average_rank_snapshot, validated_metric_specs
from biomaker.utils.serialize import dumps_json, loads_json
from biomaker.utils.evaluator_setup import EXTERNAL_VALIDATION_PREFIX

SPECS = [{"name": "x", "maximize": True}, {"name": "y", "maximize": True}]


def candidate(node_id, x, y, **kwargs):
    return Node(
        code="pass", id=node_id, metrics={"x": x, "y": y}, is_buggy=False, **kwargs
    )


def ranked_journal(specs=None):
    return Journal(selection_policy="average_rank", metric_specs=specs or SPECS)


def test_old_node_becomes_best_and_commit_context_is_historical():
    journal = ranked_journal()
    a, b, c = candidate("A", 9, 1), candidate("B", 5, 5), candidate("C", 3, 4)
    journal.append(a)
    journal.append(b)
    assert journal.get_best_node() is a
    frozen = copy.deepcopy(journal.ranking_snapshot())
    b_context = copy.deepcopy(b.rank_context)
    journal.append(c)
    assert journal.get_best_node() is b
    assert journal.ranking_snapshot()["ordered_node_ids"] == ["B", "A", "C"]
    assert frozen["best_node_id"] == "A"
    assert frozen["candidate_count"] == 2
    assert b.rank_context == b_context
    assert a.ranking["average_rank"] == 2
    assert b.ranking["average_rank"] == 1.5
    assert b.ranking["snapshot_version"] != frozen["version"]


def test_reading_an_older_cohort_does_not_overwrite_current_node_ranks():
    journal = ranked_journal()
    a, b, c = candidate("A", 9, 1), candidate("B", 5, 5), candidate("C", 3, 4)
    journal.append(a)
    journal.append(b)
    old_view = Journal(
        nodes=[a, b], metric_specs=SPECS, selection_policy="average_rank"
    )
    journal.append(c)
    current_ranks = copy.deepcopy([node.ranking for node in journal.nodes])
    assert old_view.ranking_snapshot()["best_node_id"] == "A"
    assert "Avg rank 1.5" in old_view.generate_tree_index()
    assert [node.ranking for node in journal.nodes] == current_ranks


def test_workflow_agent_inherits_frozen_global_selection_policy(monkeypatch):
    import biomaker.agent as workflow_module

    monkeypatch.setattr(
        workflow_module, "ProposalAgent", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    journal = ranked_journal()
    workflow = workflow_module.BioMakerWorkflow(
        task_desc="task",
        cfg=SimpleNamespace(agent=SimpleNamespace(steps=2)),
        journal=journal,
    )
    agent_journal = workflow.proposal_agent.journal
    assert agent_journal.selection_policy == "average_rank"
    assert agent_journal.metric_specs == journal.metric_specs
    assert agent_journal.metric_specs is not journal.metric_specs


def test_midrank_ties_and_mixed_directions():
    specs = [{"name": "x", "maximize": True}, {"name": "y", "maximize": False}]
    journal = ranked_journal(specs)
    for node in (
        candidate("A", 4, 1),
        candidate("B", 3, 2),
        candidate("C", 3, 2),
        candidate("D", 1, 4),
    ):
        journal.append(node)
    ranks = journal.ranking_snapshot()["nodes"]
    assert ranks["A"]["metric_ranks"] == {"x": 1, "y": 1}
    assert ranks["B"]["metric_ranks"] == {"x": 2.5, "y": 2.5}
    assert ranks["C"]["average_rank"] == 2.5
    assert ranks["D"]["worst_rank"] == 4


def test_average_tie_prefers_smaller_worst_rank_before_step():
    specs = [{"name": name, "maximize": True} for name in ("x", "y", "z")]
    journal = ranked_journal(specs)
    for node_id, values in (
        ("A", (4, 1, 1)),
        ("B", (2, 2, 2)),
        ("C", (3, 3, 3)),
        ("D", (1, 4, 4)),
    ):
        journal.append(
            Node(
                code="pass",
                id=node_id,
                metrics=dict(zip(("x", "y", "z"), values)),
                is_buggy=False,
            )
        )
    snapshot = journal.ranking_snapshot()
    assert snapshot["ordered_node_ids"] == ["C", "D", "B", "A"]
    assert (
        snapshot["nodes"]["A"]["average_rank"]
        == snapshot["nodes"]["B"]["average_rank"]
        == 3
    )
    assert snapshot["nodes"]["B"]["worst_rank"] == 3
    assert snapshot["nodes"]["A"]["worst_rank"] == 4


def test_metric_order_and_scale_do_not_change_selection():
    journal = ranked_journal()
    for node in (candidate("A", 9, 1), candidate("B", 5, 5), candidate("C", 3, 4)):
        journal.append(node)
    original = journal.ranking_snapshot()
    journal.metric_specs.reverse()
    for node in journal.nodes:
        node.metrics["x"] *= 1e12
        node.metrics["y"] = node.metrics["y"] / 100 + 91
    changed = journal.ranking_snapshot()
    assert changed["ordered_node_ids"] == original["ordered_node_ids"]
    assert {
        node_id: row["average_rank"] for node_id, row in changed["nodes"].items()
    } == {node_id: row["average_rank"] for node_id, row in original["nodes"].items()}


@pytest.mark.parametrize("maximize", [True, False])
def test_single_metric_uses_original_value_order(maximize):
    journal = ranked_journal([{"name": "x", "maximize": maximize}])
    for idx, value in enumerate((8, 2, 5)):
        journal.append(
            Node(code="pass", id=str(idx), metrics={"x": value}, is_buggy=False)
        )
    assert [node.metrics["x"] for node in journal.ranked_nodes()] == sorted(
        (8, 2, 5), reverse=maximize
    )


@pytest.mark.parametrize(
    "value", [None, float("nan"), float("inf"), -float("inf"), True]
)
def test_invalid_or_missing_target_excludes_whole_node(value):
    journal = ranked_journal()
    invalid = candidate("invalid", 100, value, metric=MetricValue(100, maximize=True))
    if value is None:
        del invalid.metrics["y"]
    journal.append(invalid)
    journal.append(candidate("valid", 1, 1))
    assert journal.ranking_snapshot()["candidate_count"] == 1
    assert journal.get_best_node().id == "valid"
    assert invalid.ranking is None


def test_running_failed_uncommitted_and_private_test_are_excluded():
    journal = ranked_journal()
    valid = candidate("valid", 1, 1)
    journal.append(valid)
    journal.append(candidate("running", 100, 100, tag="running"))
    failed = candidate("failed", 200, 200)
    failed.is_buggy = True
    journal.append(failed)
    journal.append(candidate("crashed", 300, 300, exc_type="RuntimeError"))
    journal.nodes.append(candidate("uncommitted", 400, 400))
    before = journal.ranking_snapshot()
    valid.metrics["private_test"] = 0.99
    valid.evaluator_details = {"private_test": {"x": 900, "y": 900}}
    after = journal.ranking_snapshot()
    assert before == after
    assert before["ordered_node_ids"] == ["valid"]


def test_missing_named_scores_cannot_fall_back_to_legacy_scalar():
    journal = ranked_journal()
    node = Node(code="pass", metric=MetricValue(0.9, maximize=True), is_buggy=False)
    journal.append(node)
    assert journal.get_best_node() is None
    assert node.metrics == {}


def test_unprepared_scalar_caller_is_adapted_to_one_metric():
    journal = Journal(selection_policy="average_rank")
    node = Node(code="pass", metric=MetricValue(0.9, maximize=True), is_buggy=False)
    journal.append(node)
    assert journal.metric_specs == [{"name": "metric", "maximize": True}]
    assert node.metrics == {"metric": 0.9}
    assert journal.get_best_node() is node


def test_legacy_display_snapshot_excludes_uncommitted_result():
    journal = Journal()
    node = Node(code="pass", metric=MetricValue(0.8, maximize=True), is_buggy=False)
    journal.append(node)
    transient = Node(
        code="pass",
        tag="completed",
        metric=MetricValue(0.99, maximize=True),
        is_buggy=False,
    )
    journal.nodes.append(transient)
    assert journal.ranking_snapshot()["best_node_id"] == node.id
    assert journal.ranking_snapshot()["candidate_count"] == 1


def test_same_snapshot_deltas_compare_parent_and_child():
    journal = ranked_journal()
    parent = candidate("A", 9, 1)
    journal.append(parent)
    child = candidate("B", 5, 5, parent=parent)
    journal.append(child)
    assert (
        child.rank_context["node"]["snapshot_version"]
        == child.rank_context["parent"]["snapshot_version"]
    )
    assert child.delta_vs_parent == 0
    assert child.metric_deltas["parent"] == {"x": -4, "y": 4}
    assert child.outcome == "matched_best"
    frozen = copy.deepcopy(child.rank_context)
    journal.append(candidate("C", 3, 4))
    assert child.rank_context == frozen


def test_round_freezes_snapshot_and_ranks_before_expandability_filter():
    journal = ranked_journal()
    a, b, c = candidate("A", 9, 1), candidate("B", 5, 5), candidate("C", 3, 4)
    journal.append(a)
    journal.append(b)
    agent = BaseAgent.__new__(BaseAgent)
    agent.journal = journal
    agent.acfg = SimpleNamespace(
        search=SimpleNamespace(
            max_children_per_node=1, topk_improve=1, max_debug_depth=0
        )
    )
    tasks = agent.select_round_tasks()
    assert tasks[0].parent is a
    journal.append(c)
    journal.append(Node(code="pass", parent=c, is_buggy=True))
    later = agent.select_round_tasks()
    assert later[0].parent is b  # Re-ranking only A/B would incorrectly select A.
    assert later[0].selection_snapshot["candidate_count"] == 3
    assert tasks[0].parent is a
    assert tasks[0].selection_snapshot["candidate_count"] == 2


def test_serialization_preserves_snapshot_and_old_policy():
    journal = ranked_journal()
    parent = candidate("A", 9, 1)
    journal.append(parent)
    journal.append(candidate("B", 5, 5, parent=parent))
    journal.append(candidate("C", 3, 4))
    expected = journal.ranking_snapshot()
    restored = loads_json(dumps_json(journal), Journal)
    assert restored.ranking_snapshot() == expected
    assert restored.nodes[1].parent is restored.nodes[0]
    assert restored.nodes[1].rank_context == journal.nodes[1].rank_context
    legacy = Journal()
    legacy.append(
        Node(
            code="pass",
            id="old",
            is_buggy=False,
            metric=MetricValue(0.8, maximize=True),
        )
    )
    payload = json.loads(dumps_json(legacy))
    del payload["selection_policy"]
    restored_legacy = loads_json(json.dumps(payload), Journal)
    assert restored_legacy.selection_policy == "legacy_primary"
    assert restored_legacy.ranking_snapshot()["best_node_id"] == "old"


def test_declarations_require_at_least_one_consistent_direction():
    with pytest.raises(ValueError, match="At least one"):
        average_rank_snapshot([], [])
    with pytest.raises(ValueError, match="maximize"):
        validated_metric_specs([{"name": "custom"}])
    with pytest.raises(ValueError, match="Conflicting"):
        validated_metric_specs(
            [{"name": "x", "maximize": True}, {"name": "x", "maximize": False}]
        )
    assert validated_metric_specs(SPECS + [SPECS[0]]) == SPECS


def test_frozen_metrics_reject_conflicting_result_specs():
    journal = ranked_journal()
    node = candidate(
        "A", 9, 1, metric_specs=[{"name": "x", "maximize": False}, SPECS[1]]
    )
    with pytest.raises(ValueError, match="frozen"):
        journal.append(node)
    assert not journal.nodes


@pytest.mark.parametrize("legacy_envelope", [False, True])
def test_parser_accepts_named_metrics_and_retains_diagnostics(legacy_envelope):
    envelope = {"ok": True, "metric_specs": SPECS}
    metrics = {"x": 9, "y": 1, "diagnostic_count": 20}
    if legacy_envelope:
        envelope.update(
            metric=9, maximize=True, details={"score": {"metrics": metrics}}
        )
    else:
        envelope.update(metrics=metrics)
    node = Node(code="pass")
    parse_factual_exec_result(
        node,
        ExecutionResult(
            term_out=[], exec_time=1, exc_type=None, evaluator_result=envelope
        ),
        record_context=True,
    )
    assert node.is_buggy is False
    assert node.metrics == metrics
    assert node.metric is None
    assert node.metric_specs == SPECS
    journal = ranked_journal()
    journal.append(node)
    assert set(journal.ranking_snapshot()["nodes"][node.id]["metric_ranks"]) == {
        "x",
        "y",
    }


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), True])
def test_parser_rejects_invalid_objective_even_with_valid_legacy_scalar(bad_value):
    node = Node(code="pass")
    envelope = {
        "ok": True,
        "metric": 9,
        "metrics": {"x": 9, "y": bad_value},
        "metric_specs": SPECS,
    }
    parse_factual_exec_result(
        node,
        ExecutionResult(
            term_out=[], exec_time=1, exc_type=None, evaluator_result=envelope
        ),
    )
    assert node.is_buggy is True


def test_parser_accepts_v3_log_only_result_and_clears_stale_metrics():
    node = Node(code="pass", metrics={"old_objective": 7})
    payload = {"metrics": {"x": 2, "y": 3}, "metric_specs": SPECS}
    parse_factual_exec_result(
        node,
        ExecutionResult(
            term_out=[EXTERNAL_VALIDATION_PREFIX + " " + json.dumps(payload)],
            exec_time=1,
            exc_type=None,
        ),
    )
    assert node.metrics == {"x": 2, "y": 3}
    assert node.metric is None
    assert node.is_buggy is False
    parse_factual_exec_result(
        node, ExecutionResult(term_out=[], exec_time=1, exc_type="RuntimeError")
    )
    assert node.metrics == {}
    assert node.is_buggy is True
