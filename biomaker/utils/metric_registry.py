"""Central metric support table for deterministic evaluators."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class MetricInfo:
    name: str
    maximize: bool
    task_family: str


SUPPORTED_METRICS: dict[str, MetricInfo] = {
    "accuracy": MetricInfo("accuracy", True, "classification"),
    "acc": MetricInfo("accuracy", True, "classification"),
    "auc": MetricInfo("auc", True, "classification"),
    "roc_auc": MetricInfo("auc", True, "classification"),
    "macro_auc": MetricInfo("auc", True, "classification"),
    "macro_auroc": MetricInfo("auc", True, "classification"),
    "ovr_auc": MetricInfo("auc", True, "classification"),
    "macro_auroc_ovr": MetricInfo("auc", True, "classification"),
    "macro_roc_auc": MetricInfo("auc", True, "classification"),
    "auroc": MetricInfo("auc", True, "classification"),
    "mean_roc_auc": MetricInfo("mean_roc_auc", True, "classification"),
    "mean_binary_accuracy": MetricInfo("mean_binary_accuracy", True, "classification"),
    "binary_dice": MetricInfo("binary_dice", True, "segmentation"),
    "dice": MetricInfo("binary_dice", True, "segmentation"),
    "mean_dice": MetricInfo("binary_dice", True, "segmentation"),
    "aji": MetricInfo("aji", True, "segmentation"),
    "mean_aji": MetricInfo("aji", True, "segmentation"),
    "hd95": MetricInfo("hd95", False, "segmentation"),
    "hausdorff95": MetricInfo("hd95", False, "segmentation"),
    "hausdorff_95": MetricInfo("hd95", False, "segmentation"),
}

# These are finite spelling equivalents of the same unweighted one-vs-rest
# calculation. Do not match arbitrary names containing "auc": micro, weighted,
# one-vs-one and per-label/multilabel contracts have different semantics.
for _ovr in ("ovr", "one_vs_rest", "one_versus_rest"):
    for _auc in ("auc", "auroc", "roc_auc"):
        for _alias in (f"macro_{_ovr}_{_auc}", f"macro_{_auc}_{_ovr}"):
            SUPPORTED_METRICS[_alias] = MetricInfo("auc", True, "classification")

_TASK_METRICS = {
    "classification": {"accuracy", "auc", "mean_binary_accuracy", "mean_roc_auc"},
    "single_label": {"accuracy", "auc"},
    "multilabel": {"mean_binary_accuracy", "mean_roc_auc"},
    "segmentation": {"binary_dice", "aji", "hd95"},
}
_TASK_KINDS = {
    "classification": "classification",
    **dict.fromkeys(
        (
            "binary",
            "binary_classification",
            "multiclass",
            "multiclass_classification",
            "ordinal_multiclass",
            "single_label_classification",
        ),
        "single_label",
    ),
    **dict.fromkeys(
        ("multilabel", "multilabel_classification", "multi_label_classification"),
        "multilabel",
    ),
    **dict.fromkeys(
        (
            "segmentation",
            "segmentation_2d",
            "segmentation_3d",
            "binary_segmentation",
            "instance_segmentation",
            "volume_segmentation",
            "medical_volume_segmentation",
        ),
        "segmentation",
    ),
}
METRIC_CONTRACT_REQUIREMENTS = (
    "Use canonical primary_metric names: binary/multiclass classification uses "
    "accuracy or auc (unweighted macro one-vs-rest ROC AUC); multilabel uses "
    "mean_binary_accuracy or mean_roc_auc (unweighted mean of per-label ROC AUC); "
    "segmentation uses binary_dice, aji, or hd95. Declare task_type and maximize "
    "(false only for hd95). Micro, weighted, one-vs-one, and unknown metrics are "
    "unsupported: preserve the requested metric and report the incompatibility, "
    "never replace it with accuracy, auc, or another default."
)


def _metric_key(metric: str) -> str:
    return re.sub(r"[\s-]+", "_", str(metric).strip().lower())


def metric_info(metric: str | None) -> MetricInfo | None:
    if metric is None:
        return None
    return SUPPORTED_METRICS.get(_metric_key(metric))


def canonical_metric_name(metric: str | None) -> str | None:
    """Normalize only known equivalents; retain unknown names, never default."""
    if metric is None or not str(metric).strip():
        return None
    info = metric_info(metric)
    return info.name if info else _metric_key(metric)


def require_supported_metric(
    metric: str | None, task_type: str | None = None
) -> MetricInfo:
    """Admission check shared by generated, deterministic and cached evaluators."""
    info = metric_info(metric)
    if info is None:
        raise ValueError(
            f"Unsupported primary metric {metric!r}. {METRIC_CONTRACT_REQUIREMENTS}"
        )
    if task_type:
        kind = _TASK_KINDS.get(_metric_key(task_type))
        if kind is None or info.name not in _TASK_METRICS[kind]:
            raise ValueError(
                f"Primary metric {metric!r} is unsupported for task_type {task_type!r}. {METRIC_CONTRACT_REQUIREMENTS}"
            )
    return info


def is_classification_metric_supported(metric: str | None) -> bool:
    info = metric_info(metric)
    return bool(info and info.task_family == "classification")


def is_segmentation_metric_supported(metric: str | None) -> bool:
    info = metric_info(metric)
    return bool(info and info.task_family == "segmentation")


def metric_maximize(metric: str | None, default: bool = True) -> bool:
    info = metric_info(metric)
    return info.maximize if info is not None else default
