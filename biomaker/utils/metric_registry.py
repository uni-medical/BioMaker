"""Central metric support table for deterministic evaluators."""

from __future__ import annotations

from dataclasses import dataclass

from .trusted_metrics import metric_capabilities


@dataclass(frozen=True)
class MetricInfo:
    name: str
    maximize: bool
    task_family: str


# Compatibility facade: the executable kernel catalog owns metric names,
# aliases, task families and direction. No independent support table can drift.
_CAPABILITIES = metric_capabilities()
SUPPORTED_METRICS: dict[str, MetricInfo] = {
    alias: MetricInfo(name, _CAPABILITIES["metrics"][name]["maximize"],
                      _CAPABILITIES["metrics"][name]["family"])
    for alias, name in _CAPABILITIES["aliases"].items()
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
