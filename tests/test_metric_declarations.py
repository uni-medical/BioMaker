"""Task prose must identify requested metrics without changing their meaning."""

from pathlib import Path

import pytest

from biomaker.utils.evaluator_protocol import requested_metrics


@pytest.mark.parametrize("declaration,expected", [
    ("MAE, in months (lower is better). Compute the mean absolute difference.", "mae"),
    ("Mean absolute error (MAE), in months.", "mae"),
    ("Binary Dice, treating all nonzero pixels as foreground.", "binary_dice"),
    ("Aggregated Jaccard Index (AJI).", "aji"),
    ("macro one-vs-rest ROC AUC.", "auc"),
    ("mean one-vs-rest ROC AUC.", "auc"),
    ("mean binary accuracy across labels.", "accuracy"),
    ("mAP@0.5 (higher is better).", "ap50"),
    ("Recall at IoU=0.50 (higher is better).", "recall"),
    ("mAP@0.5:0.95.", "map50_95"),
    ("PSNR.", "psnr"),
    ("SSIM.", "ssim"),
])
def test_exact_names_and_explanatory_text(declaration, expected):
    assert requested_metrics("Primary evaluation metric: " + declaration) == (expected, [])


@pytest.mark.parametrize("declaration,expected", [
    ("PR-AUC.", "pr_auc"),
    ("precision-recall AUC.", "pr_auc"),
    ("partial ROC AUC (max_fpr=0.1).", "partial_roc_auc_(max_fpr=0.1)"),
    ("ROC AUC (max_fpr=0.1).", "roc_auc_(max_fpr=0.1)"),
    ("Generalized Dice.", "generalized_dice"),
    ("mean Dice across cardiac structures.", "mean_dice_across_cardiac_structures"),
    ("macro accuracy.", "macro_accuracy"),
    ("Recall at IoU=0.75.", "recall_at_iou=0.75"),
    ("AUC, micro average.", "auc,_micro_average"),
    ("custom_research_metric.", "custom_research_metric"),
])
def test_unsupported_variants_are_not_replaced_with_standard_metrics(declaration, expected):
    assert requested_metrics("Primary metric: " + declaration) == (expected, [])


@pytest.mark.parametrize("declaration,expected", [
    ("AUC, F1.", ["auc", "f1"]),
    ("AUC; F1; balanced accuracy.", ["auc", "f1", "balanced_accuracy"]),
    ("AUC and F1.", ["auc", "f1"]),
    ("macro one-vs-rest ROC AUC, IoU.", ["auc", "binary_iou"]),
    ("AUC, custom_score.", ["auc", "custom_score"]),
    ("Binary Dice, treating nonzero labels as foreground.", ["binary_dice"]),
])
def test_secondary_metric_lists_preserve_every_unambiguous_name(declaration, expected):
    assert requested_metrics("Secondary metrics: " + declaration) == (None, expected)


def test_repeated_declarations_and_generic_metric_lines():
    assert requested_metrics("Metric: MAE, in months.") == ("mae", [])
    assert requested_metrics("Metrics: accuracy; AUC.") == ("accuracy", ["auc"])
    assert requested_metrics("Metric: mean Dice/IoU across cardiac structures.") == (
        "mean_dice/iou_across_cardiac_structures", [])
    assert requested_metrics("Secondary metric: AUC.\nSecondary metric: F1.") == (None, ["auc", "f1"])
    assert requested_metrics("Metric: F1.\nPrimary metric: accuracy.") == ("accuracy", [])


def test_existing_workspace_task_declarations():
    experiments = Path(__file__).resolve().parents[2] / "experiments"
    if not experiments.is_dir():
        pytest.skip("Workspace-level experiment definitions are not part of the method repository.")
    assert requested_metrics((experiments / "rsna_bone_age/task.md").read_text()) == ("mae", [])
    assert requested_metrics((experiments / "monuseg/task.md").read_text()) == ("aji", ["binary_dice"])
    assert requested_metrics((experiments / "kvasirseg/task.md").read_text()) == ("binary_dice", ["binary_iou"])
    assert requested_metrics((experiments / "camus/task.md").read_text())[0] != "binary_dice"
