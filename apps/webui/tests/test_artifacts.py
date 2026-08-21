from __future__ import annotations

import pytest
from webui.artifacts import (
    ArtifactError,
    artifact_collection_revision,
    list_artifacts,
    resolve_artifact,
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
