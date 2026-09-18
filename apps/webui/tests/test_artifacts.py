from __future__ import annotations

import pytest
import csv
import json
from webui.artifacts import (
    ArtifactError,
    artifact_collection_revision,
    list_artifacts,
    resolve_artifact,
    read_results_summary,
)


def test_artifact_resolution_is_confined_to_logs_root(settings, tmp_path):
    log_dir = settings.logs_root / "1-safe-run"
    log_dir.mkdir()
    tree = log_dir / "tree_plot.html"
    tree.write_text("<html>tree</html>", encoding="utf-8")
    run = {"id": "run", "log_dir": str(log_dir)}

    assert resolve_artifact(settings, run, "tree_plot.html") == tree.resolve()
    with pytest.raises(ArtifactError, match="Invalid artifact name"):
        resolve_artifact(settings, run, "../tree_plot.html")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tree_plot.html").write_text("outside", encoding="utf-8")
    with pytest.raises(ArtifactError, match="outside"):
        resolve_artifact(
            settings,
            {"id": "bad", "log_dir": str(outside)},
            "tree_plot.html",
        )


def test_artifact_listing_excludes_direct_symlinks(settings, tmp_path):
    log_dir = settings.logs_root / "2-safe-run"
    log_dir.mkdir()
    (log_dir / "results.tsv").write_text("metric\n1\n", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (log_dir / "linked-secret.txt").symlink_to(outside)

    run = {"id": "r", "log_dir": str(log_dir)}
    artifacts = list_artifacts(settings, run)
    names = [item["name"] for item in artifacts]
    assert names == ["results.tsv"]
    assert isinstance(artifacts[0]["revision"], str)

    first_revision = artifacts[0]["revision"]
    first_collection_revision = artifact_collection_revision(artifacts)
    (log_dir / "results.tsv").write_text("metric\n10\n", encoding="utf-8")
    changed = list_artifacts(settings, run)

    assert changed[0]["revision"] != first_revision
    assert artifact_collection_revision(changed) != first_collection_revision
    assert artifact_collection_revision([]) is None


def test_result_summary_uses_exported_winner_and_raw_metrics(settings):
    log_dir = settings.logs_root / "ranked-run"
    log_dir.mkdir()
    specs = [
        {"name": "dice", "maximize": True},
        {"name": "distance", "maximize": False},
    ]
    fields = [
        "node_id",
        "step",
        "metric",
        "maximize",
        "is_best",
        "metrics",
        "metric_specs",
        "metric_ranks",
        "average_rank",
        "worst_rank",
        "candidate_count",
        "ranking_version",
        "selection_policy",
    ]
    with (log_dir / "results.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index, (node_id, is_best, dice) in enumerate(
            [("old-a", False, 9), ("old-b", True, 5), ("new-c", False, 3)]
        ):
            writer.writerow(
                {
                    "node_id": node_id,
                    "step": index,
                    "metric": 999 if not is_best else 0,
                    "maximize": True,
                    "is_best": is_best,
                    "metrics": json.dumps({"dice": dice, "distance": 10}),
                    "metric_specs": json.dumps(specs),
                    "metric_ranks": json.dumps({"dice": 2, "distance": 1}),
                    "average_rank": 1.5,
                    "worst_rank": 2,
                    "candidate_count": 3,
                    "ranking_version": "snapshot-3",
                    "selection_policy": "average_rank",
                }
            )
    result = read_results_summary(settings, {"id": "rank", "log_dir": str(log_dir)})
    assert result["best"]["node_id"] == "old-b"
    assert result["best"]["metrics"] == {"dice": 5, "distance": 10}
    assert result["best"]["metric_specs"] == specs
    assert result["best"]["average_rank"] == 1.5
    assert result["best"]["ranking_version"] == "snapshot-3"


def test_result_summary_does_not_choose_unmarked_candidate(settings):
    log_dir = settings.logs_root / "invalid-run"
    log_dir.mkdir()
    (log_dir / "results.tsv").write_text(
        "node_id\tmetric\tis_best\ninvalid\t999\tFalse\n", encoding="utf-8"
    )
    result = read_results_summary(settings, {"id": "invalid", "log_dir": str(log_dir)})
    assert result["best"] is None
    assert result["best_metric"] is None


def test_result_summary_reads_historical_scalar_tsv(settings):
    log_dir = settings.logs_root / "legacy-run"
    log_dir.mkdir()
    (log_dir / "results.tsv").write_text(
        "node_id\tmetric\tmaximize\na\t7\tFalse\nb\t2\tFalse\nc\tnan\tFalse\n",
        encoding="utf-8",
    )
    result = read_results_summary(settings, {"id": "legacy", "log_dir": str(log_dir)})
    assert result["best_metric"] == 2
    assert result["best"]["node_id"] == "b"
