"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from pathlib import Path

import numpy as np
from igraph import Graph

from ..journal import Journal


def get_edges(journal: Journal):
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)


def generate_layout(n_nodes, edges, layout_type="rt"):
    """Generate visual layout of graph"""
    layout = Graph(
        n_nodes,
        edges=edges,
        directed=True,
    ).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)


def normalize_layout(layout: np.ndarray):
    """Normalize layout to [0, 1]"""
    min_vals = layout.min(axis=0)
    max_vals = layout.max(axis=0)
    range_vals = max_vals - min_vals

    # 避免除以0
    nonzero_mask = range_vals != 0
    layout[:, nonzero_mask] = (layout[:, nonzero_mask] - min_vals[nonzero_mask]) / range_vals[nonzero_mask]
    layout[:, 1] = 1 - layout[:, 1]
    layout[:, 1] = np.nan_to_num(layout[:, 1], nan=0)
    layout[:, 0] = np.nan_to_num(layout[:, 0], nan=0.5)
    return layout


def cfg_to_tree_struct(cfg, jou: Journal):
    edges = list(get_edges(jou))
    layout = normalize_layout(generate_layout(len(jou), edges))

    metrics = np.array([str(n.metric.value_npsafe) for n in jou])
    agent_names = np.array([str(n.agent_name) for n in jou])
    term_outs = [n.term_out for n in jou]

    plan_with_metric_list = []
    for n, t, m, a in zip(jou.nodes, term_outs, metrics, agent_names):
        # 构建原始字符串
        original_string = [
            f"agent:{a}",
            f"metric:{m}",
            f"plan:{n.plan}",
            f"term_out:{t}",
        ]

        # 按换行符分割字符串，然后对每一行进行换行处理
        wrapped_lines = []
        for line in original_string: # 按 \n 分割
            wrapped_lines.append(textwrap.fill(line, width=80)) # 对每一行进行 fill

        # 将处理过的行用 \n 重新连接
        plan_with_metric_list.append('\n'.join(wrapped_lines))

    return dict(
        edges=edges,
        layout=layout.tolist(),
        plan=plan_with_metric_list,
        code=[n.code for n in jou],
        term_out=[n.term_out for n in jou],
        analysis=[n.analysis for n in jou],
        exp_name=cfg.exp_name,
        metrics=np.array([0 for n in jou]).tolist(),
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
