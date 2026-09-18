from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from biomaker.journal import Journal, Node
from biomaker.utils import config, tree_export
from biomaker.utils.metric import MetricValue


def _cfg(tmp_path):
    return SimpleNamespace(
        exp_name="rank-view", workspace_dir=tmp_path, log_dir=tmp_path
    )


def _journal(count=2):
    specs = [
        {"name": f"score_{index}", "maximize": index != 2} for index in range(count)
    ]
    return Journal(metric_specs=specs, selection_policy="average_rank")


def _append(journal, node_id, values, **kwargs):
    node = Node(
        code=f"print({node_id!r})",
        id=node_id,
        metrics=dict(zip((spec["name"] for spec in journal.metric_specs), values)),
        is_buggy=False,
        tag="completed",
        **kwargs,
    )
    journal.append(node)
    return node


@pytest.mark.parametrize("count", [1, 2, 3, 12])
def test_tree_labels_and_details_use_all_declared_metrics(tmp_path, count):
    journal = _journal(count)
    node = _append(journal, "first-node", range(1, count + 1))
    node.metrics["extra_diagnostic"] = 0.2
    tree = tree_export.cfg_to_tree_struct(_cfg(tmp_path), journal)
    detail = tree["summaries"][1]
    label = tree["display_labels"][1]
    assert detail["is_best"]
    assert detail["candidate_count"] == 1
    assert detail["average_rank"] == 1
    assert len(detail["metric_ranks"]) == count
    assert detail["diagnostic_metrics"] == {"extra_diagnostic": 0.2}
    assert len(tree["candidates"]) == 1
    if count == 1:
        assert "Avg rank" not in label
        assert "1.000" in label
    elif count == 2:
        assert "score_0 ↑ 1" in label
        assert "score_1 ↑ 2" in label
        assert "Avg rank" not in label
    else:
        assert "Avg rank 1.00" in label
        assert "score_0 ↑ 1" in label
        assert "score_1" not in label


def test_old_node_winner_and_exports_refresh_from_same_snapshot(tmp_path):
    journal = _journal()
    a = _append(journal, "aaaaaaaa-full", (9, 1))
    b = _append(journal, "bbbbbbbb-full", (5, 5))
    cfg = _cfg(tmp_path)
    first = tree_export.generate(cfg, journal, tmp_path / "tree_plot.html")
    assert journal.ranking_snapshot()["best_node_id"] == a.id
    _append(journal, "cccccccc-full", (3, 4))
    second = tree_export.generate(cfg, journal, tmp_path / "tree_plot.html")
    tree = tree_export.cfg_to_tree_struct(cfg, journal)
    snapshot = tree["ranking_snapshot"]
    assert snapshot["best_node_id"] == b.id
    assert tree["is_best"] == [False, False, True, False]
    assert tree["candidates"][0]["node_id"] == b.id[:8]
    assert first["revision"] != second["revision"]
    assert (
        tree_export.generate(cfg, journal, tmp_path / "tree_plot.html")["changed"]
        is False
    )

    config._write_results_table(tmp_path, journal, snapshot)
    with (tmp_path / "results.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["node_id"] for row in rows if row["is_best"] == "True"] == [b.id[:8]]
    assert all(row["metric"] == row["maximize"] == "" for row in rows)
    assert all(row["ranking_version"] == snapshot["version"] for row in rows)
    assert json.loads(rows[1]["metrics"]) == {"score_0": 5, "score_1": 5}
    config._write_node_metadata(tmp_path, b, snapshot)
    metadata = json.loads((tmp_path / "node_metadata.json").read_text())
    assert metadata["is_best"] is True
    assert metadata["metric_specs"] == journal.metric_specs
    assert metadata["metric"] is None
    assert metadata["average_rank"] == snapshot["nodes"][b.id]["average_rank"]


def test_zero_candidates_and_uncommitted_nodes(tmp_path):
    journal = _journal(3)
    tree = tree_export.cfg_to_tree_struct(_cfg(tmp_path), journal)
    assert tree["candidates"] == []
    assert tree["run_summary"]["candidate_count"] == 0
    live = Node(
        code="",
        metrics={"score_0": 1, "score_1": 1, "score_2": 1},
        tag="running",
        is_buggy=False,
    )
    journal.nodes.append(live)
    tree = tree_export.cfg_to_tree_struct(_cfg(tmp_path), journal, live=True)
    assert tree["candidates"] == []
    assert tree["is_best"] == [False, False]


def test_long_metric_names_are_preserved_and_inline_data_is_safe(tmp_path):
    name = "very_long_metric_name_" * 12 + "</script><script>unsafe()</script>"
    journal = Journal(
        metric_specs=[
            {"name": name, "maximize": True},
            {"name": "second", "maximize": False},
        ],
        selection_policy="average_rank",
    )
    _append(journal, "long-name", (0.6, 500))
    tree = tree_export.cfg_to_tree_struct(_cfg(tmp_path), journal)
    assert name in tree["labels"][1]
    assert name in tree["summaries"][1]["metrics"]
    assert name not in tree["display_labels"][1]
    html = tree_export.generate_html(json.dumps(tree))
    assert "</script><script>unsafe()" not in html
    assert "candidate-sort" in html
    assert "overflow-x: auto" in html


def test_legacy_tree_keeps_scalar_selection(tmp_path):
    a = Node(
        code="a",
        is_buggy=False,
        step=0,
        metric=MetricValue(0.9, maximize=True),
        metrics={"other": 0},
    )
    b = Node(
        code="b",
        is_buggy=False,
        step=1,
        metric=MetricValue(0.8, maximize=True),
        metrics={"other": 100},
    )
    journal = Journal(nodes=[a, b])
    tree = tree_export.cfg_to_tree_struct(_cfg(tmp_path), journal)
    assert tree["is_best"] == [False, True, False]
    assert tree["summaries"][1]["metric"] == "0.9"
    assert tree["ranking_snapshot"]["selection_policy"] == "legacy_primary"


def test_new_selected_best_removes_stale_legacy_artifact_aliases(tmp_path):
    workspace = tmp_path / "workspace"
    log = tmp_path / "log"
    current = workspace / "working_best"
    current.mkdir(parents=True)
    log.mkdir()
    cfg = SimpleNamespace(workspace_dir=workspace, log_dir=log)
    (current / "submission.csv").write_text("id,prediction\none,1\n")
    (current / "best_ckpt.pth").write_text("old checkpoint")
    (current / "pred_masks").mkdir()
    (current / "pred_masks" / "old.png").write_text("old mask")
    archive = log / "nodes" / "old"
    archive.mkdir(parents=True)
    (archive / "best_ckpt.pth").write_text("preserved checkpoint")
    config._copy_best_working_outputs(cfg)
    (current / "best_ckpt.pth").unlink()
    (current / "pred_masks" / "old.png").unlink()
    (current / "pred_masks").rmdir()
    (current / "submission.csv").unlink()
    config._copy_best_working_outputs(cfg)
    assert not (log / "best_ckpt.pth").exists()
    assert not (log / "pred_masks").exists()
    assert not (log / "submission.csv").exists()
    assert (archive / "best_ckpt.pth").read_text() == "preserved checkpoint"
