"""Export journal to HTML visualization of tree + code."""

import json
import textwrap
from pathlib import Path

from ..journal import Journal


def get_edges(journal: Journal):
    for node in journal:
        for c in node.children:
            yield (node.step, c.step)


def generate_layout(journal: Journal) -> list[list[float]]:
    """Generate a small dependency-free tree layout normalized to [0, 1]."""

    if not journal:
        return []

    depths = {}
    for node in journal:
        depth = 0
        parent = node.parent
        while parent is not None:
            depth += 1
            parent = parent.parent
        depths[node.step] = depth

    max_step = max(1, len(journal) - 1)
    max_depth = max(1, max(depths.values()))
    return [[node.step / max_step, depths[node.step] / max_depth] for node in journal]


def cfg_to_tree_struct(cfg, jou: Journal):
    edges = list(get_edges(jou))
    layout = generate_layout(jou)

    metrics = [str(n.metric.value_npsafe) for n in jou]
    agent_names = [str(n.agent_name) for n in jou]
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
        for line in original_string:  # 按 \n 分割
            wrapped_lines.append(textwrap.fill(line, width=80))  # 对每一行进行 fill

        # 将处理过的行用 \n 重新连接
        plan_with_metric_list.append("\n".join(wrapped_lines))

    return dict(
        edges=edges,
        layout=layout,
        plan=plan_with_metric_list,
        code=[n.code for n in jou],
        term_out=[n.term_out for n in jou],
        analysis=[n.analysis for n in jou],
        exp_name=cfg.exp_name,
        metrics=[0 for n in jou],
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
