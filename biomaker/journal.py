"""
The journal is the core datastructure in BioMaker that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from dataclasses_json import DataClassJsonMixin

from .interpreter import ExecutionResult
from .utils.metric import MetricValue
from .utils.response import trim_execution_output, truncate_text


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)  # type: ignore
    agent_name: str = field(default=None, kw_only=True)  # type: ignore
    tag: str = field(default=None, kw_only=True)  # type: ignore
    prompt: dict[str, Any] | None = field(default=None, kw_only=True)
    prompt_messages: list[dict[str, str]] | None = field(default=None, kw_only=True)
    prompt_stats: dict[str, Any] | None = field(default=None, kw_only=True)
    execution_workspace: str | None = field(default=None, kw_only=True)

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)
    selection_reason: str | None = field(default=None, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    resource_usage: dict | None = field(default=None, kw_only=True)
    artifact_report: dict | None = field(default=None, kw_only=True)
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    parent_metric: float | None = field(default=None, kw_only=True)
    best_metric_before: float | None = field(default=None, kw_only=True)
    delta_vs_parent: float | None = field(default=None, kw_only=True)
    delta_vs_best: float | None = field(default=None, kw_only=True)
    outcome: str | None = field(default=None, kw_only=True)
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def node_status(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.resource_usage = exec_result.resource_usage
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    def record_experiment_context(self, best_node_before: Optional["Node"]) -> None:
        """Record simple experiment deltas after execution, before appending."""
        if self.is_buggy or self.metric is None or self.metric.value is None:
            self.outcome = "bug"
            return

        maximize = bool(self.metric.maximize)

        if (
            self.parent is not None
            and self.parent.metric is not None
            and self.parent.metric.value is not None
        ):
            self.parent_metric = float(self.parent.metric.value)
            self.delta_vs_parent = _better_delta(
                current=float(self.metric.value),
                baseline=self.parent_metric,
                maximize=maximize,
            )

        if best_node_before is not None and best_node_before.metric.value is not None:
            self.best_metric_before = float(best_node_before.metric.value)
            self.delta_vs_best = _better_delta(
                current=float(self.metric.value),
                baseline=self.best_metric_before,
                maximize=maximize,
            )

        eps = 1e-12
        if self.best_metric_before is None:
            self.outcome = "first_success"
        elif self.delta_vs_best is not None and self.delta_vs_best > eps:
            self.outcome = "new_best"
        elif self.delta_vs_parent is not None and self.delta_vs_parent > eps:
            self.outcome = "improved_parent"
        elif self.delta_vs_parent is not None and self.delta_vs_parent < -eps:
            self.outcome = "regressed_parent"
        elif self.delta_vs_best is not None and abs(self.delta_vs_best) <= eps:
            self.outcome = "matched_best"
        else:
            self.outcome = "valid_non_improving"

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution after semantic trimming."""
        return trim_execution_output("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.node_status != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    def pop(self) -> Optional["Node"]:
        """
        Remove and return the last node from the journal.
        Returns None if the journal is empty.
        """
        if not self.nodes:
            return None
        node = self.nodes.pop()
        if node.parent is not None:
            node.parent.children.discard(node)
        return node

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy is True]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy is False]

    def get_best_node(self, only_good=True, topk=1) -> None | Node:
        """Return one of the top-k best solutions found so far (randomly chosen)."""

        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes

        top_nodes = sorted(nodes, key=lambda n: n.metric, reverse=True)[:topk]
        return random.choice(top_nodes) if top_nodes else None

    def get_nearest_successful_ancestor(self, node: Node) -> Node | None:
        """Return the closest runnable ancestor for local debug reference."""
        current = node.parent
        while current is not None:
            if current.is_buggy is False:
                return current
            current = current.parent
        return None

    def _reference_nodes(self, topk: int) -> list[Node]:
        """Return deterministic high-scoring successful nodes for factual memory."""
        good_nodes = sorted(self.good_nodes, key=lambda n: n.metric, reverse=True)
        return good_nodes[:topk]

    def generate_tree_index(self) -> str:
        """Return a compact factual index of every node in the search tree."""
        if not self.nodes:
            return "No previous nodes."

        header = (
            "step | id | parent | stage | status | metric | "
            "delta_parent | delta_best | runtime | idea"
        )
        rows = [header]
        for n in self.nodes:
            rows.append(
                " | ".join(
                    [
                        "" if n.step is None else str(n.step),
                        n.id[:8],
                        n.parent.id[:8] if n.parent is not None else "root",
                        n.node_status,
                        _node_status_text(n),
                        _metric_text(n),
                        _format_delta(n.delta_vs_parent),
                        _format_delta(n.delta_vs_best),
                        _format_runtime(n.exec_time),
                        _one_line(str(n.plan or n.analysis or ""), max_chars=90),
                    ]
                )
            )
        return "\n".join(rows)

    def select_detail_nodes(
        self,
        stage: Literal["draft", "debug", "improve"],
        selected_parent: Node | None = None,
    ) -> list[tuple[Node, str]]:
        """Choose which previous nodes receive full detail records.

        The policy is deterministic. It only chooses nodes; every chosen node is
        represented by a full record that includes its complete plan and code.
        """
        if not self.nodes:
            return []

        selected: list[tuple[Node, str]] = []
        seen: set[str] = set()

        def add(node: Node | None, reason: str) -> None:
            if node is None or node.id in seen:
                return
            if node not in self.nodes:
                return
            if stage == "draft" and node.parent is not None:
                return
            if stage in {"debug", "improve"} and selected_parent is None:
                return
            if (
                stage in {"debug", "improve"}
                and selected_parent is not None
                and not _is_local_to_selected_parent(node, selected_parent)
            ):
                return
            seen.add(node.id)
            selected.append((node, reason))

        def add_many(nodes, reason: str) -> None:
            for node in nodes:
                add(node, reason)

        if stage == "draft":
            add_many(
                sorted(self.draft_nodes, key=_node_sort_step),
                "existing draft sibling under the virtual root",
            )
            return selected

        if selected_parent is not None:
            add(selected_parent, "selected parent; direct parent of the new node")
            if stage == "debug":
                if selected_parent.parent is None:
                    sibling_nodes = [
                        node for node in self.draft_nodes if node is not selected_parent
                    ]
                else:
                    sibling_nodes = [
                        node
                        for node in selected_parent.parent.children
                        if node is not selected_parent
                    ]
                add_many(
                    sorted(sibling_nodes, key=_node_sort_step),
                    "sibling of the selected buggy parent; alternative attempt from the same parent",
                )

        if selected_parent is not None:
            add_many(
                _ancestor_chain(selected_parent),
                "ancestor of the selected parent, nearest first",
            )
            add_many(
                sorted(selected_parent.children, key=_node_sort_step),
                "existing sibling of the new node; already generated from the same selected parent",
            )

        return selected

    def generate_detail_records(
        self,
        stage: Literal["draft", "debug", "improve"],
        selected_parent: Node | None = None,
    ) -> str:
        """Return full detail records for deterministically selected nodes."""
        selected = self.select_detail_nodes(
            stage=stage, selected_parent=selected_parent
        )
        if not selected:
            return "No previous node details."
        return "\n\n" + ("\n\n---\n\n").join(
            _format_detail_record(node, reason) for node, reason in selected
        )

    def _avoid_summary(self, max_items: int = 4) -> str:
        failures = []
        for n in reversed(self.buggy_nodes):
            if len(failures) >= max_items:
                break
            reason = _format_node_header(n)
            if n.parent is not None:
                reason += f" | parent: {n.parent.id[:8]}"
            reason += f" | error: {n.exc_type or 'BuggyResult'}"
            if n.analysis:
                reason += (
                    f"\n  Finding: {truncate_text(str(n.analysis), max_chars=180)}"
                )
            elif n.exc_info:
                reason += (
                    f"\n  Error info: {truncate_text(str(n.exc_info), max_chars=180)}"
                )
            if n.plan:
                reason += f"\n  Attempt: {truncate_text(str(n.plan), max_chars=180)}"
            failures.append(f"- {reason}")
        if not failures:
            return ""
        return "Recent execution failures:\n" + "\n".join(failures)

    def _regression_summary(self, max_items: int = 4) -> str:
        regressions = []
        for n in reversed(self.good_nodes):
            if len(regressions) >= max_items:
                break
            if n.delta_vs_parent is None or n.delta_vs_parent >= 0:
                continue
            regressions.append(
                "- "
                + _format_node_header(n)
                + f" | delta_vs_parent: {_format_delta(n.delta_vs_parent)}\n"
                + f"  Tried: {truncate_text(str(n.plan), max_chars=180)}"
            )
        if not regressions:
            return ""
        return "Previous valid regressions:\n" + "\n".join(regressions)

    def generate_branch_history(
        self,
        node: Node,
        max_children: int = 6,
        max_chars: int = 3000,
    ) -> str:
        """Summarize factual child experiments already tried from this node."""
        parts = ["Selected parent\n" + _format_memory_node(node)]
        children = sorted(
            node.children,
            key=lambda n: n.step if n.step is not None else 10**9,
        )
        if not children:
            parts.append("Direct child experiments from this parent: none yet.")
        else:
            recent_children = children[-max_children:]
            parts.append(
                "Direct child experiments from this parent\n"
                + "\n".join(_format_child_experiment(n) for n in recent_children)
            )

        if node.parent is not None:
            siblings = sorted(
                [n for n in node.parent.children if n != node],
                key=lambda n: n.step if n.step is not None else 10**9,
            )
            if siblings:
                parts.append(
                    "Sibling experiments under the same grandparent\n"
                    + "\n".join(_format_child_experiment(n) for n in siblings[-3:])
                )

        return truncate_text(
            "\n-------------------------------\n".join(parts),
            max_chars=max_chars,
        )

    def generate_debug_summary(
        self,
        buggy_node: Node,
        max_chars: int = 2500,
    ) -> str:
        """Summarize factual context useful for fixing the current buggy node."""
        parts = ["Current buggy node\n" + _format_memory_node(buggy_node)]
        reference_node = self.get_nearest_successful_ancestor(buggy_node)
        if reference_node is not None:
            parts.append(
                "Nearest successful ancestor\n" + _format_memory_node(reference_node)
            )

        avoid_summary = self._avoid_summary(max_items=3)
        if avoid_summary:
            parts.append(avoid_summary)

        return truncate_text(
            "\n-------------------------------\n".join(parts),
            max_chars=max_chars,
        )

    def generate_summary(
        self,
        include_code: bool = False,
        topk: int = 3,
        max_chars: int = 6000,
        include_avoid_list: bool = True,
    ) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        best_node = self.get_best_node(only_good=True, topk=1)
        if best_node is not None:
            summary.append("CURRENT BEST\n" + _format_memory_node(best_node))

        if include_avoid_list:
            regression_summary = self._regression_summary()
            if regression_summary:
                summary.append(regression_summary)
            avoid_summary = self._avoid_summary()
            if avoid_summary:
                summary.append(avoid_summary)

        candidates = [n for n in self._reference_nodes(topk=topk) if n != best_node]
        if candidates:
            summary.append(
                "Other high-scoring successful nodes\n"
                + "\n\n".join(_format_memory_node(n) for n in candidates)
            )

        if include_code:
            code_nodes = [n for n in [best_node, *candidates] if n is not None]
            if code_nodes:
                summary.append(
                    "Reference code snippets\n"
                    + "\n\n".join(
                        f"Node {n.id[:8]} code:\n{truncate_text(n.code, max_chars=4000)}"
                        for n in code_nodes
                    )
                )

        if not summary:
            return "No successful attempts yet." if self.nodes else "No attempts yet."

        return truncate_text(
            "\n-------------------------------\n".join(summary), max_chars=max_chars
        )


def _better_delta(current: float, baseline: float, maximize: bool) -> float:
    """Positive means current is better than baseline."""
    return current - baseline if maximize else baseline - current


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.4f}"


def _format_runtime(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {rem:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _format_resource_usage(resource_usage: dict | None) -> str:
    if not resource_usage:
        return "n/a"
    peak_rss = resource_usage.get("peak_rss_mb")
    if not isinstance(peak_rss, (int, float)):
        return "n/a"
    if peak_rss >= 1024:
        return f"peak_rss {peak_rss / 1024:.2f}GB"
    return f"peak_rss {peak_rss:.0f}MB"


def _node_sort_step(n: Node) -> int:
    return n.step if n.step is not None else 10**9


def _recent_nodes(nodes) -> list[Node]:
    return sorted(list(nodes), key=_node_sort_step, reverse=True)


def _is_problem_node(n: Node) -> bool:
    return (
        n.is_buggy is True
        or n.exc_type is not None
        or n.outcome == "bug"
        or (n.delta_vs_parent is not None and n.delta_vs_parent < 0)
    )


def _problem_nodes_from(nodes) -> list[Node]:
    return [n for n in nodes if _is_problem_node(n)]


def _ancestor_chain(node: Node) -> list[Node]:
    ancestors = []
    current = node.parent
    while current is not None:
        ancestors.append(current)
        current = current.parent
    return ancestors


def _is_local_to_selected_parent(node: Node, selected_parent: Node) -> bool:
    if node == selected_parent:
        return True
    if node in _ancestor_chain(selected_parent):
        return True
    if node.parent == selected_parent:
        return True
    if selected_parent.parent is None:
        return node.parent is None
    return node.parent == selected_parent.parent


def _top_successful_nodes(nodes) -> list[Node]:
    return sorted(
        [n for n in nodes if n.metric is not None],
        key=lambda n: n.metric,
        reverse=True,
    )


def _best_node_deterministic(nodes) -> Node | None:
    top_nodes = _top_successful_nodes(nodes)
    return top_nodes[0] if top_nodes else None


def _metric_text(n: Node) -> str:
    metric = n.metric.value if n.metric is not None else None
    if metric is None:
        return "n/a"
    try:
        return f"{float(metric):.6g}"
    except Exception:
        return str(metric)


def _node_status_text(n: Node) -> str:
    if n.is_buggy is True:
        return "bug"
    if n.exc_type:
        return "error"
    return n.outcome or ("success" if n.is_buggy is False else "unknown")


def _artifact_status(report: dict | None) -> str:
    if not isinstance(report, dict):
        return "n/a"
    if report.get("available") is False:
        return "missing"
    warnings = report.get("warnings")
    alignment = report.get("alignment")
    if warnings:
        return "valid_with_warnings"
    if isinstance(alignment, dict):
        matched = alignment.get("matched_rows")
        method = alignment.get("method") or "aligned"
        return f"{method}:{matched}" if matched is not None else str(method)
    return "available" if report.get("available") else "n/a"


def _one_line(text: str, max_chars: int) -> str:
    return truncate_text(" ".join(text.split()), max_chars=max_chars)


def json_dumps_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _format_artifact_report(report: dict) -> str:
    compact = {
        "warnings": report.get("warnings"),
        "alignment": report.get("alignment"),
        "validation_artifact": report.get("validation_artifact"),
        "target_artifact": report.get("target_artifact"),
    }
    samples = report.get("sample_reports")
    if isinstance(samples, list) and samples:
        compact["sample_0"] = samples[0]
    return json_dumps_compact(compact)


def _format_node_header(n: Node) -> str:
    metric = n.metric.value if n.metric is not None else None
    metric_text = "n/a" if metric is None else f"{float(metric):.4f}"
    return (
        f"Node {n.id[:8]} | status: {n.node_status} | "
        f"metric: {metric_text} | runtime: {_format_runtime(n.exec_time)} | "
        f"resources: {_format_resource_usage(n.resource_usage)} | "
        f"outcome: {n.outcome or 'unknown'}"
    )


def _format_memory_node(n: Node) -> str:
    artifact_text = ""
    if n.artifact_report:
        artifact_text = "\nArtifact facts: " + truncate_text(
            _format_artifact_report(n.artifact_report), max_chars=520
        )
    selection_text = ""
    if n.selection_reason:
        selection_text = "\nSelected because: " + truncate_text(
            str(n.selection_reason), max_chars=180
        )
    return (
        f"{_format_node_header(n)}\n"
        f"delta_vs_parent: {_format_delta(n.delta_vs_parent)} | "
        f"delta_vs_best_before: {_format_delta(n.delta_vs_best)}\n"
        f"Core idea: {truncate_text(str(n.plan), max_chars=320)}\n"
        f"Empirical finding: {truncate_text(str(n.analysis), max_chars=260)}"
        f"{selection_text}"
        f"{artifact_text}"
    )


def _format_child_experiment(n: Node) -> str:
    line = (
        "- "
        + _format_node_header(n)
        + f" | delta_vs_parent: {_format_delta(n.delta_vs_parent)}"
    )
    if n.plan:
        line += f"\n  Attempt: {truncate_text(str(n.plan), max_chars=180)}"
    if n.analysis:
        line += f"\n  Finding: {truncate_text(str(n.analysis), max_chars=180)}"
    elif n.exc_info:
        line += f"\n  Error info: {truncate_text(str(n.exc_info), max_chars=180)}"
    if n.artifact_report:
        line += "\n  Artifact facts: " + truncate_text(
            _format_artifact_report(n.artifact_report), max_chars=320
        )
    return line


def _format_detail_record(n: Node, reason: str) -> str:
    artifact_report = (
        json_dumps_compact(n.artifact_report)
        if n.artifact_report is not None
        else "n/a"
    )
    resource_usage = (
        json_dumps_compact(n.resource_usage) if n.resource_usage is not None else "n/a"
    )
    exception_info = {
        "exc_type": n.exc_type,
        "exc_info": n.exc_info,
        "exc_stack": n.exc_stack,
    }
    return (
        f"Node {n.id[:8]} full record\n"
        f"Selection reason for including this record: {reason}\n"
        f"step: {n.step}\n"
        f"id: {n.id}\n"
        f"parent_id: {n.parent.id if n.parent is not None else None}\n"
        f"children_ids: {[child.id for child in sorted(n.children, key=_node_sort_step)]}\n"
        f"stage: {n.node_status}\n"
        f"status: {_node_status_text(n)}\n"
        f"metric: {_metric_text(n)}\n"
        f"maximize: {n.metric.maximize if n.metric is not None else None}\n"
        f"delta_vs_parent: {_format_delta(n.delta_vs_parent)}\n"
        f"delta_vs_best_before: {_format_delta(n.delta_vs_best)}\n"
        f"outcome: {n.outcome}\n"
        f"selected_by_search_policy: {n.selection_reason}\n"
        f"exec_time: {_format_runtime(n.exec_time)}\n"
        f"resource_usage: {resource_usage}\n"
        f"artifact_status: {_artifact_status(n.artifact_report)}\n"
        f"artifact_report: {artifact_report}\n"
        f"exception: {json_dumps_compact(exception_info)}\n\n"
        f"Plan:\n{n.plan or ''}\n\n"
        f"Code:\n```python\n{n.code or ''}\n```\n\n"
        f"Compact execution output:\n```\n{n.term_out or ''}\n```\n\n"
        f"Empirical finding:\n{n.analysis or ''}"
    )
