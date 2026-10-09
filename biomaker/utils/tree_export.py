"""Export journal to HTML visualization of tree + code."""

import hashlib
import json
import os
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from ..backend import compile_prompt_to_md
from ..journal import Journal
from .response import format_human_execution_output, truncate_text

_TREE_EXPORT_LOCK = RLock()
_VENDOR_ASSETS = (
    ("/* __BIOMAKER_VENDOR_P5_JS__ */", "p5-1.9.0.min.js"),
    ("/* __BIOMAKER_VENDOR_HIGHLIGHT_CSS__ */", "highlight-github-11.9.0.min.css"),
    ("/* __BIOMAKER_VENDOR_HIGHLIGHT_JS__ */", "highlight-11.9.0.min.js"),
    ("__BIOMAKER_LICENSE_P5__", "LICENSE.p5-1.9.0.txt"),
    ("__BIOMAKER_LICENSE_HIGHLIGHT__", "LICENSE.highlight-11.9.0.txt"),
)


@dataclass
class VisualNode:
    source_node: object | None
    children: list[int] = field(default_factory=list)
    x: float | None = None
    y: int = 0


def _sorted_children(node) -> list:
    return sorted(
        node.children,
        key=lambda child: (
            _display_step(child) if _display_step(child) is not None else 10**9
        ),
    )


def _build_visual_tree(journal: Journal) -> tuple[list[VisualNode], list[list[int]]]:
    """Build a visual tree with an explicit virtual root at index 0."""
    visual_nodes = [VisualNode(source_node=None)]
    index_by_node = {}
    for node in journal:
        index_by_node[node] = len(visual_nodes)
        visual_nodes.append(VisualNode(source_node=node))

    edges: list[list[int]] = []
    ordered_nodes = sorted(
        list(journal),
        key=lambda node: (
            _display_step(node) if _display_step(node) is not None else 10**9
        ),
    )
    for node in ordered_nodes:
        child_index = index_by_node[node]
        if node.parent is None:
            parent_index = 0
        elif node.parent in index_by_node:
            parent_index = index_by_node[node.parent]
        else:
            parent_index = 0
        visual_nodes[parent_index].children.append(child_index)
        edges.append([parent_index, child_index])
    return visual_nodes, edges


def generate_layout(visual_nodes: list[VisualNode]) -> list[list[float]]:
    """Generate a deterministic top-down tidy tree layout normalized to [0, 1]."""

    if not visual_nodes:
        return []
    if len(visual_nodes) == 1:
        return [[0.5, 0.0]]

    next_leaf_x = 0

    def assign(index: int, depth: int) -> float:
        nonlocal next_leaf_x
        node = visual_nodes[index]
        node.y = depth
        if not node.children:
            node.x = float(next_leaf_x)
            next_leaf_x += 1
            return node.x
        child_x = [assign(child, depth + 1) for child in node.children]
        node.x = sum(child_x) / len(child_x)
        return node.x

    assign(0, 0)
    max_x = max(1.0, max(float(node.x or 0.0) for node in visual_nodes))
    max_y = max(1, max(node.y for node in visual_nodes))
    return [
        [
            0.5 if max_x == 0 else float(node.x or 0.0) / max_x,
            node.y / max_y,
        ]
        for node in visual_nodes
    ]


def strip_code_markers(code: str) -> str:
    """Remove markdown code block markers if present."""
    code = (code or "").strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def _node_metric(node) -> str:
    if node.metric is None:
        return "n/a"
    return str(node.metric.value_npsafe)


def _node_metric_short(node) -> str:
    if node is None or node.metric is None or node.metric.value is None:
        return ""
    try:
        return f"{float(node.metric.value):.3f}"
    except Exception:
        return str(node.metric.value_npsafe)


def _display_step(node) -> int | None:
    if node is None:
        return None
    if node.step is not None:
        return node.step
    value = getattr(node, "scheduled_step", None)
    return value if isinstance(value, int) else None


def _node_label(node) -> str:
    if node is None:
        return ""
    display_step = _display_step(node)
    if node.step is None:
        step = "?" if display_step is None else str(display_step)
        if node.tag == "completed":
            if node.is_buggy:
                return f"{step}: bug"
            metric = _node_metric(node)
            return (
                f"{step}: completed {metric}"
                if metric != "n/a"
                else f"{step}: completed"
            )
        return f"{step}: {_transient_status(node)}"
    prefix = "bug" if node.is_buggy else node.node_status
    metric = _node_metric(node)
    step = "na" if display_step is None else str(display_step)
    return f"{step}: {prefix} {metric}"


def _node_display_label(node) -> str:
    if node is None:
        return ""
    display_step = _display_step(node)
    step = "?" if display_step is None else str(display_step)
    if node.step is None:
        if node.tag == "completed":
            if node.is_buggy:
                return f"{step}\nbug"
            metric = _node_metric_short(node)
            return f"{step}\n{metric}" if metric else f"{step}\ncompleted"
        return f"{step}\n{_transient_status(node)}"
    if node.is_buggy:
        return f"{step}\nbug"
    metric = _node_metric_short(node)
    return f"{step}\n{metric}" if metric else step


def _format_seconds(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.1f}s"
    except Exception:
        return str(value)


def _format_mb(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.0f} MiB"
    except Exception:
        return str(value)


def _estimated_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _json_section(name: str, data: dict[str, Any], max_chars: int = 6000) -> dict:
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str) if data else ""
    text = truncate_text(text, max_chars=max_chars, marker=f"{name} trimmed")
    return {
        "name": name,
        "chars": len(text),
        "estimated_tokens": _estimated_tokens(text),
        "text": text or "Not available.",
    }


def _root_setup_prompt(cfg) -> dict:
    workspace_dir = Path(getattr(cfg, "workspace_dir", ""))
    log_dir = Path(getattr(cfg, "log_dir", ""))
    split_dir = workspace_dir / "input" / "biomaker_splits"
    sections = [
        _json_section(
            "Run Setup",
            {
                "exp_name": getattr(cfg, "exp_name", ""),
                "workspace_dir": str(workspace_dir),
                "log_dir": str(log_dir),
                "data_dir": str(getattr(cfg, "data_dir", "")),
                "desc_file": str(getattr(cfg, "desc_file", "")),
            },
            max_chars=3000,
        ),
        _json_section("Run Manifest", _read_json(log_dir / "run_manifest.json")),
        _json_section("Experiment Guide", _read_json(workspace_dir / "exp_guide.json")),
        _json_section(
            "Evaluator Prediction Spec",
            _read_json(workspace_dir / "evaluator_prediction_spec.json"),
        ),
        _json_section("Split Plan", _read_json(split_dir / "split_plan.json")),
        _json_section("Split Manifest", _read_json(split_dir / "split_manifest.json")),
    ]
    sections = [section for section in sections if section["text"] != "Not available."]
    if not sections:
        sections = [
            {
                "name": "Run Setup",
                "chars": 0,
                "estimated_tokens": 0,
                "text": "No pre-node setup artifacts were available when this tree was exported.",
            }
        ]
    return {
        "available": True,
        "stats": {
            "model": "pre-node setup",
            "estimated_input_tokens": sum(
                section.get("estimated_tokens", 0) for section in sections
            ),
        },
        "sections": sections,
        "messages": [],
    }


def _transient_status(node) -> str:
    if node is None:
        return "virtual"
    if node.step is None:
        if node.tag == "completed" and node.is_buggy:
            return "bug"
        return str(node.tag or "pending")
    if node.is_buggy:
        return "bug"
    return node.node_status


def _node_prompt(node, cfg=None) -> dict:
    if node is None:
        return _root_setup_prompt(cfg) if cfg is not None else {"available": False}

    prompt = node.prompt
    messages = node.prompt_messages or []
    stats = node.prompt_stats or {}
    sections = []
    if isinstance(prompt, dict):
        section_stats = {
            str(item.get("name")): item
            for item in stats.get("sections", [])
            if isinstance(item, dict)
        }
        for key, value in prompt.items():
            text = compile_prompt_to_md(value)
            saved_stats = section_stats.get(str(key), {})
            sections.append(
                {
                    "name": str(key),
                    "chars": saved_stats.get("chars", len(text)),
                    "estimated_tokens": saved_stats.get(
                        "estimated_tokens", _estimated_tokens(text)
                    ),
                    "text": text,
                }
            )

    return {
        "available": bool(prompt or messages),
        "stats": stats,
        "sections": sections,
        "messages": [],
    }


def _artifact_summary(report: dict | None) -> str:
    if not isinstance(report, dict):
        return ""
    parts = []
    available = report.get("available")
    if available is not None:
        parts.append(f"available: {available}")
    alignment = report.get("alignment")
    if isinstance(alignment, dict):
        method = alignment.get("method")
        matched = alignment.get("matched_rows")
        if method or matched is not None:
            parts.append(f"alignment: {method or 'n/a'} ({matched or 0} rows)")
    predictions = report.get("prediction_column_summaries")
    if isinstance(predictions, dict):
        for key, value in list(predictions.items())[:3]:
            if isinstance(value, dict):
                parts.append(
                    f"{key}: nulls={value.get('null_count', 'n/a')}, "
                    f"unique={value.get('sample_unique_count', 'n/a')}"
                )
    warnings = report.get("warnings")
    if warnings:
        parts.append("warnings: " + truncate_text(str(warnings), max_chars=160))
    return "\n".join(parts)


def _node_summary(node, best_node) -> dict:
    if node is None:
        return {
            "title": "Search start",
            "node_id": "",
            "step": "",
            "stage": "virtual",
            "status": "virtual",
            "metric": "",
            "runtime": "",
            "memory": "",
            "gpu_memory": "",
            "cpu": "",
            "outcome": "",
            "plan": (
                "Pre-node setup and run contract. Use Input Prompt to inspect "
                "the split protocol, evaluator prediction spec, run manifest, "
                "and experiment guide that existed before node generation."
            ),
            "error": "",
        }

    resources = node.resource_usage or {}
    metric = _node_metric(node)
    status = _transient_status(node)
    return {
        "title": _node_label(node),
        "node_id": node.id[:8],
        "step": "" if _display_step(node) is None else str(_display_step(node)),
        "stage": node.node_status,
        "status": status,
        "metric": "" if metric == "n/a" else metric,
        "runtime": _format_seconds(node.exec_time),
        "memory": _format_mb(resources.get("peak_rss_mb")),
        "gpu_memory": (
            "not independently measured (RSS proxy)"
            if resources.get("gpu_memory_source") == "unified_memory_rss_proxy"
            else _format_mb(resources.get("peak_gpu_memory_mb"))
            + (" (total delta, approximate)" if resources.get("gpu_memory_source") == "nvidia-smi_total_used_delta_approx"
               else " (process tree)" if resources.get("gpu_memory_source") == "nvidia-smi_compute_apps_process_tree"
               else " (source unconfirmed)" if resources.get("peak_gpu_memory_mb") is not None else "")
        ),
        "cpu": (
            ""
            if resources.get("avg_cpu_percent") is None
            else f"{float(resources.get('avg_cpu_percent')):.1f}%"
        ),
        "outcome": node.outcome or ("not yet appended" if node.step is None else ""),
        "plan": str(node.plan or ""),
        "error": truncate_text(
            str(node.exc_info or node.exc_type or ""), max_chars=600
        ),
        "is_best": bool(node == best_node),
    }


def _node_plan_text(node) -> str:
    if node is None:
        return "Search start\n" "Draft nodes are direct children of this start point."
    original_lines = [
        f"node:{node.id[:8]}",
        f"stage:{node.node_status}",
        f"metric:{_node_metric(node)}",
        f"outcome:{node.outcome}",
        f"selected:{truncate_text(str(node.selection_reason or ''), max_chars=220)}",
        f"plan:{str(node.plan or '')}",
    ]
    return "\n".join(textwrap.fill(str(line), width=88) for line in original_lines)


def _node_term_out(node) -> str:
    if node is None or getattr(node, "_term_out", None) is None:
        return ""
    execution = getattr(node, "execution", None)
    if isinstance(execution, dict):
        candidate = execution.get("candidate")
        if isinstance(candidate, dict) and isinstance(candidate.get("output"), str):
            return format_human_execution_output(candidate["output"], max_chars=12000)
    return format_human_execution_output("".join(node._term_out), max_chars=12000)


def cfg_to_tree_struct(cfg, jou: Journal, live: bool = False):
    visual_nodes, edges = _build_visual_tree(jou)
    layout = generate_layout(visual_nodes)
    committed_nodes = [node for node in jou.nodes if node.step is not None]
    good_nodes = [node for node in committed_nodes if node.is_buggy is False]
    bug_nodes = [node for node in committed_nodes if node.is_buggy is True]
    scored_good_nodes = [node for node in good_nodes if node.metric is not None]
    best_node = (
        sorted(scored_good_nodes, key=lambda node: node.metric, reverse=True)[0]
        if scored_good_nodes
        else None
    )

    return dict(
        edges=edges,
        layout=layout,
        plan=[_node_plan_text(v.source_node) for v in visual_nodes],
        code=[
            "" if v.source_node is None else strip_code_markers(v.source_node.code)
            for v in visual_nodes
        ],
        term_out=[_node_term_out(v.source_node) for v in visual_nodes],
        artifact_reports=[
            None if v.source_node is None else v.source_node.artifact_report
            for v in visual_nodes
        ],
        exp_name=cfg.exp_name,
        metrics=[0 for _ in visual_nodes],
        labels=[_node_label(v.source_node) for v in visual_nodes],
        display_labels=[_node_display_label(v.source_node) for v in visual_nodes],
        summaries=[_node_summary(v.source_node, best_node) for v in visual_nodes],
        prompts=[_node_prompt(v.source_node, cfg) for v in visual_nodes],
        statuses=[_transient_status(v.source_node) for v in visual_nodes],
        is_best=[
            bool(
                v.source_node is not None
                and v.source_node.step is not None
                and v.source_node == best_node
            )
            for v in visual_nodes
        ],
        run_summary={
            "exp_name": cfg.exp_name,
            "steps": len(committed_nodes),
            "valid": len(good_nodes),
            "bug": len(bug_nodes),
            "best_metric": _node_metric_short(best_node),
            "live": live,
        },
    )


def _script_safe_json(value: str) -> str:
    """Keep JSON data from terminating the inline script element."""

    return (
        value.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def generate_html(tree_graph_str: str) -> str:
    template_dir = Path(__file__).parent / "viz_templates"

    js = (template_dir / "template.js").read_text(encoding="utf-8")
    js = js.replace("<placeholder>", _script_safe_json(tree_graph_str))

    html = (template_dir / "template.html").read_text(encoding="utf-8")
    html = html.replace("<!-- placeholder -->", js)
    for placeholder, filename in _VENDOR_ASSETS:
        asset = (template_dir / filename).read_text(encoding="utf-8")
        html = html.replace(placeholder, asset, 1)
    return html


def _atomic_write_text(out_path: Path, text: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        dir=out_path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, out_path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def generate(
    cfg, jou: Journal, out_path: Path, live: bool = False
) -> dict[str, bool | str]:
    """Export the current tree only when its canonical content has changed.

    The returned revision describes the tree data rather than filesystem
    timestamps.  Callers should treat it as opaque and use ``changed`` to avoid
    work that is only driven by a periodic refresh attempt.
    """

    with _TREE_EXPORT_LOCK:
        tree_graph_str = json.dumps(
            cfg_to_tree_struct(cfg, jou, live=live),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        revision = hashlib.sha256(tree_graph_str.encode("utf-8")).hexdigest()
        html = generate_html(tree_graph_str)
        out_path = Path(out_path)
        try:
            current_html = out_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            current_html = None
        changed = current_html != html
        if changed:
            _atomic_write_text(out_path, html)
        return {"changed": changed, "revision": revision}
