"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from ..journal import Journal
from .response import truncate_text


@dataclass
class VisualNode:
    source_node: object | None
    children: list[int] = field(default_factory=list)
    x: float | None = None
    y: int = 0


def _sorted_children(node) -> list:
    return sorted(
        node.children,
        key=lambda child: child.step if child.step is not None else 10**9,
    )


def _build_visual_tree(journal: Journal) -> tuple[list[VisualNode], list[list[int]]]:
    """Build a visual tree with an explicit virtual root at index 0."""
    visual_nodes = [VisualNode(source_node=None)]
    index_by_node = {}
    for node in journal:
        index_by_node[node] = len(visual_nodes)
        visual_nodes.append(VisualNode(source_node=node))

    edges: list[list[int]] = []
    root_children = sorted(
        journal.draft_nodes,
        key=lambda node: node.step if node.step is not None else 10**9,
    )
    for node in root_children:
        child_index = index_by_node[node]
        visual_nodes[0].children.append(child_index)
        edges.append([0, child_index])

    for node in journal:
        parent_index = index_by_node[node]
        for child in _sorted_children(node):
            child_index = index_by_node[child]
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


def _node_label(node) -> str:
    if node is None:
        return ""
    prefix = "bug" if node.is_buggy else node.node_status
    metric = _node_metric(node)
    step = "na" if node.step is None else str(node.step)
    return f"{step}: {prefix} {metric}"


def _node_display_label(node) -> str:
    if node is None:
        return ""
    step = "?" if node.step is None else str(node.step)
    if node.is_buggy:
        reason = "timeout" if node.exc_type == "TimeoutError" else "bug"
        return f"{step}\n{reason}"
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
        return f"{float(value):.0f} MB"
    except Exception:
        return str(value)


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
            "stage": "virtual",
            "status": "virtual",
            "metric": "",
            "runtime": "",
            "memory": "",
            "cpu": "",
            "outcome": "",
            "parent": "",
            "selected": "",
            "plan": "Draft nodes are direct children of this start point.",
            "analysis": "",
            "artifact": "",
            "error": "",
        }

    resources = node.resource_usage or {}
    metric = _node_metric(node)
    status = "bug" if node.is_buggy else node.node_status
    return {
        "title": _node_label(node),
        "stage": node.node_status,
        "status": status,
        "metric": "" if metric == "n/a" else metric,
        "runtime": _format_seconds(node.exec_time),
        "memory": _format_mb(resources.get("peak_rss_mb")),
        "cpu": (
            ""
            if resources.get("avg_cpu_percent") is None
            else f"{float(resources.get('avg_cpu_percent')):.1f}%"
        ),
        "outcome": node.outcome or "",
        "parent": node.parent.id[:8] if node.parent is not None else "",
        "selected": truncate_text(str(node.selection_reason or ""), max_chars=220),
        "plan": truncate_text(str(node.plan or ""), max_chars=900),
        "analysis": truncate_text(str(node.analysis or ""), max_chars=900),
        "artifact": _artifact_summary(node.artifact_report),
        "error": truncate_text(str(node.exc_info or node.exc_type or ""), max_chars=600),
        "is_best": bool(node == best_node),
    }


def _node_plan_text(node) -> str:
    if node is None:
        return (
            "Search start\n"
            "Draft nodes are direct children of this start point."
        )
    original_lines = [
        f"node:{node.id[:8]}",
        f"stage:{node.node_status}",
        f"metric:{_node_metric(node)}",
        f"outcome:{node.outcome}",
        f"selected:{truncate_text(str(node.selection_reason or ''), max_chars=220)}",
        f"plan:{truncate_text(str(node.plan or ''), max_chars=900)}",
        f"analysis:{truncate_text(str(node.analysis or ''), max_chars=900)}",
    ]
    return "\n".join(textwrap.fill(str(line), width=88) for line in original_lines)


def cfg_to_tree_struct(cfg, jou: Journal):
    visual_nodes, edges = _build_visual_tree(jou)
    layout = generate_layout(visual_nodes)
    best_node = jou.get_best_node(only_good=True, topk=1)
    good_nodes = [node for node in jou.nodes if node.is_buggy is False]
    bug_nodes = [node for node in jou.nodes if node.is_buggy is True]
    timeout_nodes = [node for node in bug_nodes if node.exc_type == "TimeoutError"]

    return dict(
        edges=edges,
        layout=layout,
        plan=[_node_plan_text(v.source_node) for v in visual_nodes],
        code=[
            "" if v.source_node is None else strip_code_markers(v.source_node.code)
            for v in visual_nodes
        ],
        term_out=[
            ""
            if v.source_node is None
            else truncate_text(v.source_node.term_out, max_chars=12000)
            for v in visual_nodes
        ],
        analysis=[
            "" if v.source_node is None else v.source_node.analysis
            for v in visual_nodes
        ],
        exp_name=cfg.exp_name,
        metrics=[0 for _ in visual_nodes],
        labels=[_node_label(v.source_node) for v in visual_nodes],
        display_labels=[_node_display_label(v.source_node) for v in visual_nodes],
        summaries=[_node_summary(v.source_node, best_node) for v in visual_nodes],
        statuses=[
            (
                "virtual"
                if v.source_node is None
                else "bug"
                if v.source_node.is_buggy
                else v.source_node.node_status
            )
            for v in visual_nodes
        ],
        is_best=[
            bool(v.source_node is not None and v.source_node == best_node)
            for v in visual_nodes
        ],
        run_summary={
            "exp_name": cfg.exp_name,
            "steps": len(jou.nodes),
            "valid": len(good_nodes),
            "bug": len(bug_nodes),
            "timeout": len(timeout_nodes),
            "best_metric": _node_metric_short(best_node),
        },
    )


def generate_html(tree_graph_str: str):
    template_dir = Path(__file__).parent / "viz_templates"

    with open(template_dir / "template.js") as f:
        js = f.read()
        js = js.replace("<placeholder>", tree_graph_str)

    with open(template_dir / "template.html") as f:
        html = f.read()
        html = html.replace("<!-- placeholder -->", js)

        return html


def generate(cfg, jou: Journal, out_path: Path):
    tree_graph_str = json.dumps(cfg_to_tree_struct(cfg, jou))
    html = generate_html(tree_graph_str)
    with open(out_path, "w") as f:
        f.write(html)
