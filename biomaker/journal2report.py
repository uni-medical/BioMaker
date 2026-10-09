"""Reproducible reports from saved facts, without model paraphrases.

Compact agent memory drops failures and nested metrics. It is not a report source.
"""

import json
import math
import os
import re

from .interpreter import _StreamingLogRedactor
from .journal import (
    Journal,
    Node,
    _candidate_execution_text,
    _format_resource_usage,
    _has_rankable_metric,
)
from .utils.config import StageConfig
from .utils.response import truncate_text
from .utils.score_validity import has_fallback_marker


def _text(value: object) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, indent=2, default=str)
    )
    return _StreamingLogRedactor(os.environ).feed(text, final=True)


def _cell(value: object) -> str:
    return _text(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _block(value: object, language: str = "text", limit: int = 12000) -> str:
    text = truncate_text(
        _text(value),
        max_chars=limit,
        marker="omitted; see saved journal/code/output for the full record",
    )
    # Generated logs/plans cannot terminate their literal block and inject markup.
    fence = "`" * max(3, 1 + max((len(m) for m in re.findall(r"`+", text)), default=0))
    return f"{fence}{language}\n{text}\n{fence}"


def _score_details(node: Node) -> dict:
    details = node.evaluator_details
    if not isinstance(details, dict):
        return {}
    payload = details.get("details", details)
    score = payload.get("score") if isinstance(payload, dict) else None
    return score if isinstance(score, dict) else {}


def _metric(node: Node) -> float | None:
    value = getattr(node.metric, "value", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _valid(node: Node) -> bool:
    return node.is_buggy is False and _has_rankable_metric(node)


def _number(node: Node, index: int) -> int:
    return node.step + 1 if isinstance(node.step, int) and node.step >= 0 else index + 1


def _flatten_metrics(value: object, path: str = "", depth: int = 0):
    """Keep nested public metrics such as per-class AUROC visible."""
    if depth > 8:
        yield path, "Nested value omitted; see journal.json"
    elif isinstance(value, dict):
        for key, item in value.items():
            # Do not publish a misplaced private-score subtree.
            if re.search(
                r"private|test(?:_|$)|secret|token|password|credential", str(key), re.I
            ):
                continue
            yield from _flatten_metrics(
                item, f"{path}.{key}" if path else str(key), depth + 1
            )
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _flatten_metrics(item, f"{path}[{i}]", depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        yield path, "undefined (non-finite recorded value)"
    elif value is None:
        yield path, "not recorded"
    else:
        yield path, str(value)


def journal2report(journal: Journal, task_desc: dict, rcfg: StageConfig):
    """Render every attempt and its validation evidence in stable Markdown.

    rcfg remains for API/config compatibility. A report no longer needs an external
    model request or depends on model output being factual.
    """
    nodes = list(journal.nodes)
    valid = [node for node in nodes if _valid(node)]
    directions = {node.metric.maximize for node in valid}
    best = None
    if valid and len(directions) == 1 and next(iter(directions)) in (True, False):
        best = sorted(
            valid, key=lambda node: _metric(node), reverse=bool(next(iter(directions)))
        )[0]
    sections = [
        "# BioMaker experiment report",
        "This report is rendered directly from the saved exploration journal. Values are recorded runtime evidence, not an independent re-evaluation. All attempts, including failures, are included below.",
        "## Task",
        _block(task_desc, "json" if not isinstance(task_desc, str) else "text"),
        "## Results",
        f"Recorded attempts: **{len(nodes)}**. Valid scored attempts: **{len(valid)}**. Failed, invalid or unconfirmed attempts: **{len(nodes) - len(valid)}**.",
    ]
    if best is not None:
        idx = nodes.index(best)
        direction = "maximize" if best.metric.maximize else "minimize"
        sections.append(
            f"Best valid recorded exploration candidate: **#{_number(best, idx)}** (`{best.id}`), validation metric **{_metric(best):.12g}** ({direction}). This describes exploration; consult the separate final-retraining receipt for any later model."
        )
    elif valid:
        sections.append(
            "No cross-candidate ranking is reported because the recorded metric directions are missing or inconsistent."
        )
    else:
        sections.append(
            "No valid scored candidate is recorded; successful process exit alone does not establish artifact validity."
        )
    table = [
        "| Candidate | ID | Status | Validation metric | Execution seconds | Failure |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for index, node in enumerate(nodes):
        value = _metric(node)
        status = (
            "valid"
            if _valid(node)
            else "failed / invalid" if node.is_buggy or node.exc_type else "unconfirmed"
        )
        if has_fallback_marker(node.evaluator_details):
            status = "invalid: fallback metric"
        runtime = node.exec_time
        runtime = (
            f"{runtime:.3f}"
            if isinstance(runtime, (int, float)) and math.isfinite(runtime)
            else "not recorded"
        )
        table.append(
            "| "
            + " | ".join(
                _cell(x)
                for x in [
                    f"#{_number(node, index)}",
                    node.id,
                    status,
                    (
                        f"{value:.12g}"
                        if value is not None and _valid(node)
                        else "not valid"
                    ),
                    runtime,
                    node.exc_type or "—",
                ]
            )
            + " |"
        )
    sections.append("\n".join(table))

    for index, node in enumerate(nodes):
        sections.extend(
            [
                f"## Candidate #{_number(node, index)} · {_cell(node.id)}",
                "### Recorded resources",
                _format_resource_usage(node.resource_usage),
            ]
        )
        if isinstance(node.resource_usage, dict):
            scope = _cell(node.resource_usage.get("scope", "not recorded"))
            source = _cell(node.resource_usage.get("gpu_memory_source", "not recorded"))
            sections.append(
                f"Measurement scope: `{scope}`. GPU-memory source: `{source}`."
            )
        score = _score_details(node)
        if score:
            rows = ["| Recorded validation field | Value |", "| --- | --- |"]
            rows.extend(
                f"| {_cell(key)} | {_cell(value)} |"
                for key, value in _flatten_metrics(score)
            )
            sections.extend(["### Validation evaluator record", "\n".join(rows)])
            if not _valid(node):
                sections.append(
                    "These are diagnostic recorded fields. This attempt has no valid ranking score."
                )
        if node.exc_type or node.exc_info:
            sections.extend(
                [
                    "### Failure",
                    _block({"type": node.exc_type, "info": node.exc_info}, "json"),
                ]
            )
        execution = node.execution if isinstance(node.execution, dict) else {}
        artifact = execution.get("artifact_validation")
        model = artifact.get("model_artifact") if isinstance(artifact, dict) else None
        sections.append("### Saved-model check")
        if isinstance(model, dict):
            facts = {
                key: model[key]
                for key in (
                    "ok",
                    "framework",
                    "checkpoint_count",
                    "probe_examples",
                    "output_shape",
                    "offline",
                    "relocated",
                    "training_disabled",
                    "guard_scope",
                    "detection_scope",
                )
                if key in model
            }
            sections.append(_block(facts, "json"))
        else:
            sections.append(
                "No supervised model reload receipt is recorded for this attempt; saved code or predictions alone do not confirm a reusable fitted model."
            )
        if node.plan:
            sections.extend(
                [
                    "### Generated plan",
                    "The plan states intent. It does not prove which branch executed.",
                    _block(node.plan),
                ]
            )
        output = _candidate_execution_text(node)
        sections.extend(
            [
                "### Recorded candidate execution output",
                "Use these recorded messages to distinguish executed branches from alternatives in the source. Candidate messages are not independent verification.",
                _block(output) if output else "No candidate output is recorded.",
            ]
        )
        if node.analysis:
            sections.extend(
                [
                    "### Generated analysis",
                    "Agent-authored interpretation, retained separately from recorded metrics.",
                    _block(node.analysis, limit=4000),
                ]
            )
    sections.extend(
        [
            "## Evidence and limitations",
            "- Full candidate source, execution output and evaluator records are retained in `journal.json` and the candidate artifacts. Long quoted fields above explicitly indicate omitted text.",
            "- Only public validation evaluator fields are tabulated. No private test evaluation file is read, and no metrics are independently recomputed for this report.",
            "- An RSS proxy on Apple unified memory is process-tree resident memory, not a separate measurement of MPS/GPU allocation. An NVIDIA total-memory delta is an approximation, not exact per-process allocation.",
            "- Generated plans, code alternatives and agent analysis do not establish which method executed. Refer to the recorded candidate output and saved artifacts.",
            "- A saved prediction file does not itself prove that a reloadable fitted model was saved. Consult the model artifact validation receipt where available.",
            "- This run does not establish multi-seed reproducibility, patient-level independence, or clinical validity.",
        ]
    )
    return "\n\n".join(sections) + "\n"
