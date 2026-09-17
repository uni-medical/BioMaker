"""Shared admission checks for prepared metric declarations.

Data preparation can repair invalid declarations before they are frozen. Both
preparation and evaluator setup use this pure gate; it never rewrites a manifest
or substitutes unsupported scoring semantics.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .evaluator_protocol import metric_name, requested_metrics
from .metric_registry import metric_info
from .trusted_metrics import metric_capabilities, normalize_metric_spec

PROTOCOL_DOCUMENTATION_FIELDS = {"notes", "description"}


def manifest_head_types(manifest: dict[str, Any]) -> set[str]:
    heads = manifest.get("task_heads")
    if isinstance(heads, str):
        return {heads.lower()}
    result: set[str] = set()
    if isinstance(heads, list):
        for head in heads:
            if isinstance(head, str):
                result.add(head.lower())
            elif isinstance(head, dict) and isinstance(head.get("type"), str):
                result.add(str(head["type"]).lower())
    return result


def manifest_primary_metric(manifest: dict[str, Any]) -> str | None:
    metric_contract = manifest.get("metric_contract")
    if isinstance(metric_contract, dict) and metric_contract.get("primary_metric"):
        metric = metric_contract["primary_metric"]
        if isinstance(metric, dict):
            for key in ("name", "metric", "id"):
                value = metric.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None
        return str(metric).strip() or None
    return None


def declared_metric_names(
    manifest: dict[str, Any], task_text: str, *, fallback_primary: str | None = None,
) -> tuple[str | None, list[str]]:
    """Resolve task/manifest declarations without reading private data or files."""
    contract = manifest.get("metric_contract", {})
    if not isinstance(contract, dict):
        raise ValueError("metric_contract must be an object.")
    declared_primary = contract.get("primary_metric")
    if "primary_metric" in contract and not isinstance(declared_primary, (str, dict)):
        raise ValueError("metric_contract.primary_metric must be a metric name or named object.")
    manifest_primary = manifest_primary_metric(manifest)
    if declared_primary is not None and not manifest_primary:
        raise ValueError("metric_contract.primary_metric must declare a nonempty metric name.")
    declared_secondary = contract.get("secondary_metrics", [])
    if not isinstance(declared_secondary, list) or any(
        not isinstance(name, str) or not name.strip() for name in declared_secondary
    ):
        raise ValueError("metric_contract.secondary_metrics must be a list of nonempty metric names.")
    primary, secondary = requested_metrics(task_text)
    if primary and manifest_primary and metric_name(primary) != metric_name(manifest_primary):
        raise ValueError("Task and prepared metric_contract disagree on the primary metric; repair data preparation before compiling an evaluator.")
    primary = primary or manifest_primary or fallback_primary
    secondary = list(dict.fromkeys(metric_name(name) for name in [*secondary, *declared_secondary]))
    return metric_name(primary) if primary else None, secondary


def validate_declared_capabilities(manifest: dict[str, Any], task_text: str, *, fallback_primary: str | None = None) -> None:
    """Reject invalid supplied settings while leaving absent data details to adapters."""
    primary, secondary = declared_metric_names(manifest, task_text, fallback_primary=fallback_primary)
    names = [m for m in [primary, *secondary] if m]
    if not names or any(metric_info(m) is None for m in names):
        return
    families = {metric_info(m).task_family for m in names}
    if len(families) != 1:
        raise ValueError("Requested standard metrics require different canonical sample families; a mixed-head evaluation protocol is not yet supported.")
    family = families.pop()
    capabilities = metric_capabilities()["families"][family]
    contract = dict(manifest.get("metric_contract", {}))
    if family == "image_restoration":
        blocks = re.findall(r"```restoration-evaluation\s*\n(.*?)```", task_text, re.S)
        if len(blocks) > 1:
            raise ValueError("Declare exactly one restoration-evaluation contract.")
        if blocks:
            task_protocol = json.loads(blocks[0])
            if not isinstance(task_protocol, dict) or not isinstance(contract.get("restoration_evaluation", {}), dict):
                raise ValueError("restoration_evaluation must be an object.")
            contract["restoration_evaluation"] = {**contract.get("restoration_evaluation", {}), **task_protocol}
    unknown = set(contract) - set(capabilities["allowed_contract_fields"]) - PROTOCOL_DOCUMENTATION_FIELDS
    if unknown:
        raise ValueError("Unsupported declared metric protocol fields: " + ", ".join(sorted(unknown)))

    def check_fields(values, fields, prefix=""):
        for name, field in fields.items():
            keys = [key for key in [name, *field.get("aliases", [])] if key in values]
            for key in keys:
                value = values[key]
                value = field.get("value_aliases", {}).get(value, value) if isinstance(value, str) else value
                unsupported = ("fixed" in field and value != field["fixed"] or
                               "values" in field and value not in field["values"] or
                               "fixed_by_task_type" in field and value not in field["fixed_by_task_type"].values())
                if unsupported:
                    raise ValueError("Unsupported declared metric setting " + prefix + name + "; the Harness requires a validated implementation for this protocol, not another data adapter.")
                if "fields" in field:
                    if not isinstance(value, dict):
                        raise ValueError("Declared metric setting " + prefix + name + " must be an object.")
                    nested_unknown = set(value) - {k for n, f in field["fields"].items() for k in [n, *f.get("aliases", [])]}
                    if nested_unknown:
                        raise ValueError("Unsupported declared metric settings: " + prefix + name + "." + ", ".join(sorted(nested_unknown)))
                    check_fields(value, field["fields"], prefix + name + ".")
    check_fields(contract, capabilities["protocol_fields"])
    if family == "image_restoration":
        fields = capabilities["protocol_fields"]["restoration_evaluation"]["fields"]
        check_fields(contract, fields)
    # Check numeric/range and cross-field constraints as well. Placeholder data
    # dimensions/labels are used only to validate supplied protocol settings;
    # they are never frozen or used for dataset interpretation.
    probe = {key: value for key, value in contract.items() if key not in PROTOCOL_DOCUMENTATION_FIELDS}
    probe.update(primary_metric=primary, secondary_metrics=secondary)
    if family == "classification":
        output = manifest.get("prediction_contract", {})
        for key in ("class_labels", "positive_class"):
            if key not in probe and key in output:
                probe[key] = output[key]
        heads = manifest_head_types(manifest)
        inferred_type = ("multilabel" if heads & {"multilabel", "multi_label_classification"} or probe.get("aggregation") == "mean_labels"
                         else "multiclass" if "multiclass" in heads or len(probe.get("class_labels", [])) > 2 else "binary")
        probe.setdefault("task_type", inferred_type)
        if probe["task_type"] != "multilabel" and "class_labels" not in probe:
            positive = probe.get("positive_class", "__protocol_positive__")
            probe["class_labels"] = ["__protocol_negative__", positive]
    if family == "image_restoration":
        protocol = {key: probe.pop(key) for key in list(probe) if key in fields and key != "ssim_parameters"}
        protocol.update(probe.get("restoration_evaluation", {}))
        # Unspecified required values are still resolved from data by adapters.
        clipping = probe.get("prediction_clipping")
        default_range = clipping[1] if isinstance(clipping, (list, tuple)) and len(clipping) == 2 else 1
        crop_mode = protocol.get("crop_mode")
        defaults = dict(
            data_range=default_range,
            crop_size=1_000_000 if crop_mode in {"four_corners", "four_corner_crops_no_center"} else None,
            groups=["__unresolved_public_group__"] if protocol.get("aggregation") == "mean_groups" else None,
            aggregation="mean_images",
        )
        probe["restoration_evaluation"] = {**defaults, **protocol}
    try:
        normalize_metric_spec(probe)
    except ValueError as exc:
        raise ValueError("Unsupported declared metric protocol: " + str(exc)) from exc
