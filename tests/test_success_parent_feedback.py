from __future__ import annotations

import json

from biomaker.journal import Journal, Node, _format_detail_record
from biomaker.utils.metric import MetricValue


OBSERVATION_HEADER = (
    "Candidate Execution Observations (not official validation metrics):\n```\n"
)


def completed_node(output: str, *, parent: Node | None = None) -> Node:
    node = Node(code="print('candidate')", parent=parent)
    node.step = 0 if parent is None else parent.step + 1
    node.is_buggy = False
    node.metric = MetricValue(0.72, maximize=True)
    node.metrics = {"accuracy": 0.72}
    node._term_out = [output, "EVALUATOR_RESULT_JSON={\"accuracy\": 0.72}\n"]
    node.execution = {
        "candidate": {"output": output, "returncode": 0},
        "system_outputs": ["SYSTEM_OUTPUT_SENTINEL"],
        "evaluator": {
            "ok": True,
            "metrics": {"accuracy": 0.72},
            "details": {"private": "PRIVATE_DETAIL_SENTINEL"},
        },
    }
    return node


def observation(records: str) -> str:
    return records.split(OBSERVATION_HEADER, 1)[1].split("\n```", 1)[0]


def test_only_selected_successful_parent_receives_execution_observations() -> None:
    root = completed_node("ROOT_LOG_SENTINEL\n")
    parent = completed_node("SELECTED_LOG_SENTINEL\n", parent=root)
    sibling = completed_node("SIBLING_LOG_SENTINEL\n", parent=root)
    journal = Journal(nodes=[root, parent, sibling])

    records = journal.generate_tree_records("improve", parent)

    assert records.count(OBSERVATION_HEADER) == 1
    assert "SELECTED_LOG_SENTINEL" in records
    assert "ROOT_LOG_SENTINEL" not in records
    assert "SIBLING_LOG_SENTINEL" not in records


def test_candidate_observations_preserve_official_score_and_exclude_system_data() -> None:
    parent = completed_node("epoch 2: loss=0.4; self_reported_accuracy=0.99\n")
    parent._term_out.append("PRIVATE_TERM_SENTINEL\n")
    journal = Journal(nodes=[parent])

    records = journal.generate_tree_records("improve", parent)
    output = observation(records)

    assert "loss=0.4" in output
    assert "self_reported_accuracy=0.99" in output
    official_metrics = records.split("validation_metrics: ", 1)[1].split("\n", 1)[0]
    assert json.loads(official_metrics) == {"accuracy": 0.72}
    assert "not official validation metrics" in records
    for sentinel in (
        "SYSTEM_OUTPUT_SENTINEL",
        "PRIVATE_DETAIL_SENTINEL",
        "PRIVATE_TERM_SENTINEL",
        "EVALUATOR_RESULT_JSON=",
    ):
        assert sentinel not in records
    assert parent.metric.value == 0.72
    assert parent.metrics == {"accuracy": 0.72}


def test_legacy_observations_strip_status_lines_and_redact_local_paths() -> None:
    parent = completed_node("")
    parent.execution = None
    parent._term_out = [
        "loaded /home/user/secret/train.csv\n",
        "epoch 1: loss=0.5\n",
        'EVALUATOR_RESULT_JSON={"sentinel":"OFFICIAL_LOG_SENTINEL"}\n',
        'Resource usage: {"sentinel":"RESOURCE_LOG_SENTINEL"}\n',
        "Execution time: 10 seconds\n",
    ]

    output = observation(Journal(nodes=[parent]).generate_tree_records("improve", parent))

    assert "epoch 1: loss=0.5" in output
    assert "<local_path>" in output
    assert "/home/user/secret" not in output
    assert "OFFICIAL_LOG_SENTINEL" not in output
    assert "RESOURCE_LOG_SENTINEL" not in output
    assert "Execution time:" not in output


def test_successful_parent_observations_are_bounded_after_redaction() -> None:
    parent = completed_node(("measurement /tmp/x: " + "value " * 50 + "\n") * 100)

    output = observation(Journal(nodes=[parent]).generate_tree_records("improve", parent))

    assert len(output) <= 4000
    assert "/tmp/x" not in output


def test_empty_candidate_output_adds_no_observation_block() -> None:
    parent = completed_node("")
    parent._term_out = ["EVALUATOR_RESULT_JSON={\"accuracy\": 0.72}\n"]

    records = Journal(nodes=[parent]).generate_tree_records("improve", parent)

    assert OBSERVATION_HEADER not in records


def test_empty_structured_candidate_does_not_fall_back_to_system_output() -> None:
    parent = completed_node("")
    parent._term_out.append("NON_CANDIDATE_SYSTEM_SENTINEL\n")

    records = Journal(nodes=[parent]).generate_tree_records("improve", parent)

    assert OBSERVATION_HEADER not in records
    assert "NON_CANDIDATE_SYSTEM_SENTINEL" not in records


def test_candidate_stdout_and_stderr_are_available_without_combined_output() -> None:
    parent = completed_node("")
    parent.execution["candidate"].update(
        stdout="TRAINING_STDOUT_SENTINEL\n", stderr="TRAINING_WARNING_SENTINEL\n"
    )

    records = Journal(nodes=[parent]).generate_tree_records("improve", parent)

    assert "TRAINING_STDOUT_SENTINEL" in observation(records)
    assert "TRAINING_WARNING_SENTINEL" in observation(records)


def test_failed_parent_retains_existing_failure_record() -> None:
    parent = completed_node("Traceback: CANDIDATE_FAILURE_SENTINEL\n")
    parent.is_buggy = True
    parent.exc_type = "ValueError"
    journal = Journal(nodes=[parent])
    expected = _format_detail_record(
        parent, "root-to-parent lineage; selected parent", journal.ranking_snapshot()
    )

    records = journal.generate_tree_records("debug", parent)

    assert expected in records
    assert "CANDIDATE_FAILURE_SENTINEL" in records
    assert OBSERVATION_HEADER not in records


def test_pending_and_unscored_parents_do_not_receive_success_observations() -> None:
    for state in ("pending", "unscored"):
        parent = completed_node("UNCONFIRMED_LOG_SENTINEL\n")
        if state == "pending":
            parent.step = None
        else:
            parent.is_buggy = None
            parent.outcome = "unscored"

        records = Journal(nodes=[parent]).generate_tree_records("improve", parent)

        assert OBSERVATION_HEADER not in records
        assert "UNCONFIRMED_LOG_SENTINEL" not in records


def test_draft_without_parent_does_not_add_successful_sibling_logs() -> None:
    root = completed_node("SUCCESSFUL_DRAFT_LOG_SENTINEL\n")

    records = Journal(nodes=[root]).generate_tree_records("draft")

    assert OBSERVATION_HEADER not in records
    assert "SUCCESSFUL_DRAFT_LOG_SENTINEL" not in records
