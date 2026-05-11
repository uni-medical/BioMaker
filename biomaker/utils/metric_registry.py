"""Central metric support table for deterministic evaluators."""

from __future__ import annotations

from dataclasses import dataclass


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
    "ovr_auc": MetricInfo("auc", True, "classification"),
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


def metric_info(metric: str | None) -> MetricInfo | None:
    if metric is None:
        return None
    return SUPPORTED_METRICS.get(str(metric).strip().lower())


def is_classification_metric_supported(metric: str | None) -> bool:
    info = metric_info(metric)
    return bool(info and info.task_family == "classification")


def is_segmentation_metric_supported(metric: str | None) -> bool:
    info = metric_info(metric)
    return bool(info and info.task_family == "segmentation")


def metric_maximize(metric: str | None, default: bool = True) -> bool:
    info = metric_info(metric)
    return info.maximize if info is not None else default
