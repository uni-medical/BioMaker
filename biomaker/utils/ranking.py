"""Deterministic, population-relative ranking of frozen validation objectives.

This module never reads test scores or combines raw values across objectives.
Snapshots are detached JSON data, so a selected scheduler round can retain its
original decision even after another candidate commits.
"""

import hashlib
import json
import math
from typing import Any, Iterable
from .score_validity import require_valid_score_payload


def finite_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def validated_metric_specs(specs: list[dict]) -> list[dict]:
    """Validate already-normalized objectives without guessing directions."""
    if not isinstance(specs, list) or not specs:
        raise ValueError(
            "At least one metric must be declared for average-rank selection"
        )
    result: list[dict] = []
    by_name: dict[str, dict] = {}
    for spec in specs:
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("name"), str)
            or not spec["name"]
        ):
            raise ValueError("Each metric must have a nonempty name")
        if not isinstance(spec.get("maximize"), bool):
            raise ValueError(
                f"Metric {spec['name']!r} must declare a boolean maximize direction"
            )
        item = json.loads(json.dumps(spec, sort_keys=True, allow_nan=False))
        if item["name"] in by_name:
            if item != by_name[item["name"]]:
                raise ValueError(
                    f"Conflicting declarations for metric {item['name']!r}"
                )
            continue
        by_name[item["name"]] = item
        result.append(item)
    return result


def completed_candidate(node: Any) -> bool:
    """A live node cannot enter the population before its result is committed."""
    details = getattr(node, "evaluator_details", None)
    if isinstance(details, dict):
        if details.get("ok") is False or details.get("outcome") in {"invalid_prediction", "evaluator_unavailable"}:
            return False
        score = details.get("details", details)
        score = score.get("score") if isinstance(score, dict) else None
        try:
            require_valid_score_payload(score if isinstance(score, dict) and score.get("score_schema_version") == 1 else details)
        except ValueError:
            return False
    return (
        node.step is not None
        and node.is_buggy is False
        and node.exc_type is None
        and node.tag
        not in {"generating", "generated", "running", "evaluating", "queued", "scoring"}
    )


def average_rank_snapshot(nodes: Iterable[Any], metric_specs: list[dict]) -> dict:
    specs = validated_metric_specs(metric_specs)
    candidates: list[Any] = []
    raw: dict[str, dict[str, float]] = {}
    for node in nodes:
        if not completed_candidate(node):
            continue
        values = {
            spec["name"]: finite_metric(node.metrics.get(spec["name"]))
            for spec in specs
        }
        if any(value is None for value in values.values()):
            continue
        candidates.append(node)
        raw[node.id] = values

    ranks: dict[str, dict[str, float]] = {node.id: {} for node in candidates}
    for spec in specs:
        name = spec["name"]
        ordered = sorted(
            candidates, key=lambda node: raw[node.id][name], reverse=spec["maximize"]
        )
        start = 0
        while start < len(ordered):
            stop = start + 1
            while (
                stop < len(ordered)
                and raw[ordered[stop].id][name] == raw[ordered[start].id][name]
            ):
                stop += 1
            # Positions are one-based; a tied block receives its mean position.
            midrank = (start + 1 + stop) / 2
            for node in ordered[start:stop]:
                ranks[node.id][name] = midrank
            start = stop

    records = {
        node.id: {
            "metrics": raw[node.id],
            "metric_ranks": ranks[node.id],
            "average_rank": sum(ranks[node.id].values()) / len(specs),
            "worst_rank": max(ranks[node.id].values()),
            "step": node.step,
        }
        for node in candidates
    }
    ordered_ids = sorted(
        records,
        key=lambda node_id: (
            records[node_id]["average_rank"],
            records[node_id]["worst_rank"],
            records[node_id]["step"],
            node_id,
        ),
    )
    snapshot = {
        "selection_policy": "average_rank",
        "candidate_count": len(candidates),
        "metric_specs": specs,
        "best_node_id": ordered_ids[0] if ordered_ids else None,
        "ordered_node_ids": ordered_ids,
        "nodes": records,
    }
    # The revision depends on the complete population and frozen objectives,
    # including changes to old candidates. It is stable across serialization.
    revision = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()[:16]
    snapshot["version"] = revision
    for position, node_id in enumerate(ordered_ids, 1):
        records[node_id].update(
            position=position,
            is_best=position == 1,
            snapshot_version=revision,
            candidate_count=len(candidates),
        )
    return snapshot
