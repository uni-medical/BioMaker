"""Candidate population changes must propagate to artifacts and public views."""

import io
import json
from types import SimpleNamespace

import pytest
from rich.console import Console

from biomaker.journal import Journal, Node
from biomaker.run import (
    _ranking_payload,
    _sync_selected_best,
    best_metrics_table,
    journal_to_rich_tree,
)
from biomaker.utils.experiment_contract import load_experiment_contract


def render(item, width=100):
    stream = io.StringIO()
    Console(file=stream, width=width, force_terminal=False).print(item)
    return stream.getvalue()


def append_candidate(journal, tmp_path, node_id, values):
    directory = tmp_path / node_id
    (directory / "working").mkdir(parents=True)
    (directory / "working" / "model.txt").write_text(node_id)
    node = Node(
        code=f"print('{node_id}')",
        id=node_id,
        execution_workspace=str(directory),
        metrics=values,
        is_buggy=False,
    )
    journal.append(node)
    return node


def test_old_winner_promotion_updates_artifacts_terminal_and_frozen_selection(tmp_path, monkeypatch):
    cfg = SimpleNamespace(workspace_dir=tmp_path / "run", log_dir=tmp_path / "logs")
    cfg.log_dir.mkdir()
    journal = Journal(
        selection_policy="average_rank",
        metric_specs=[{"name": "x", "maximize": True}, {"name": "y", "maximize": True}],
    )
    a = append_candidate(journal, tmp_path, "A", {"x": 9, "y": 1})
    b = append_candidate(journal, tmp_path, "B", {"x": 5, "y": 5})
    _, initial = _sync_selected_best(cfg, journal, None)
    assert initial["best_node_id"] == "A"
    assert (cfg.workspace_dir / "working_best" / "model.txt").read_text() == "A"
    c = append_candidate(journal, tmp_path, "C", {"x": 3, "y": 4})
    winner, current = _sync_selected_best(cfg, journal, initial["best_node_id"])
    assert winner is b
    assert current["version"] != initial["version"]
    assert current["nodes"]["A"]["average_rank"] == 2
    assert (cfg.workspace_dir / "working_best" / "model.txt").read_text() == "B"
    from biomaker.utils import final_selection
    monkeypatch.setattr(final_selection, "load_experiment_contract", lambda *_a, **_k:
        SimpleNamespace(prediction=SimpleNamespace(raw={"metrics": journal.metric_specs})))
    frozen = final_selection.freeze_selection(cfg, journal, winner)
    assert frozen["node_id"] == "B"
    assert frozen["validation_selection"]["nodes"]["B"]["average_rank"] == 1.5
    assert not (cfg.log_dir / "private_metrics.json").exists()
    assert "1: x 5 · y 5 (best)" in render(journal_to_rich_tree(journal))
    assert "3 candidates" in " ".join(render(best_metrics_table(journal)).split())


@pytest.mark.parametrize("count", [1, 2, 3, 9])
def test_summary_display_rule_and_full_metric_table(tmp_path, count):
    specs = [{"name": f"score_{i}", "maximize": i % 2 == 0} for i in range(count)]
    journal = Journal(selection_policy="average_rank", metric_specs=specs)
    append_candidate(
        journal,
        tmp_path,
        "node",
        {item["name"]: i + 0.123 for i, item in enumerate(specs)},
    )
    text = render(journal_to_rich_tree(journal), width=120)
    if count < 3:
        assert "Avg rank" not in text
        if count == 2:
            assert "score_0 0.123 · score_1 1.123" in text
    else:
        assert "Avg rank 1.00 · score_0 0.123" in text
        assert "score_2" not in text
    full = render(best_metrics_table(journal), width=50)
    assert all(item["name"] in full for item in specs)
    if count > 6:
        assert "+3 metrics" in render(best_metrics_table(journal, max_rows=6))


def test_frozen_contract_exposes_all_metrics_and_directions(tmp_path):
    specs = [
        {"name": "binary_dice", "maximize": True},
        {"name": "hd95", "maximize": False},
    ]
    (tmp_path / "evaluator_prediction_spec.json").write_text(
        json.dumps(
            {
                "evaluator_api_version": 3,
                "metrics": specs,
                "prediction_columns": ["mask"],
                "key_columns": ["id"],
            }
        )
    )
    contract = load_experiment_contract(tmp_path)
    assert contract.metric.metrics == specs
    assert contract.prediction.metrics == specs
    assert contract.metric.primary_metric is None


def test_terminal_handles_no_valid_results():
    journal = Journal(
        selection_policy="average_rank",
        metric_specs=[{"name": "auc", "maximize": True}],
    )
    assert "0 valid" in render(journal_to_rich_tree(journal))
    assert "No valid" in render(best_metrics_table(journal))
