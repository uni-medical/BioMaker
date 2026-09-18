"""
The journal is the core datastructure in BioMaker that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

import json
import copy
import hashlib
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from dataclasses_json import DataClassJsonMixin

from .interpreter import ExecutionResult
from .utils.metric import MetricValue
from .utils.ranking import (
    average_rank_snapshot,
    completed_candidate,
    finite_metric,
    validated_metric_specs,
)
from .utils.response import (
    format_prompt_execution_output,
    redact_local_paths,
    trim_execution_output,
    truncate_text,
)


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
    round_index: int = field(default=0, kw_only=True)
    scheduled_step: int | None = field(default=None, kw_only=True)
    status_started_at: float | None = field(default=None, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    resource_usage: dict | None = field(default=None, kw_only=True)
    execution: dict | None = field(default=None, kw_only=True)
    artifact_report: dict | None = field(default=None, kw_only=True)
    evaluator_details: dict | None = field(default=None, kw_only=True)
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    metrics: dict[str, float] = field(default_factory=dict, kw_only=True)
    metric_specs: list[dict[str, Any]] = field(default_factory=list, kw_only=True)
    ranking: dict[str, Any] | None = field(default=None, kw_only=True)
    selection_snapshot: dict[str, Any] | None = field(default=None, kw_only=True)
    rank_context: dict[str, Any] | None = field(default=None, kw_only=True)
    metric_deltas: dict[str, dict[str, float]] = field(
        default_factory=dict, kw_only=True
    )
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
        self.execution = exec_result.execution_record()
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    def assign_schedule(
        self,
        *,
        round_index: int | None = None,
        scheduled_step: int | None = None,
    ) -> None:
        """Record scheduler metadata used for live, in-flight nodes."""
        if round_index is not None:
            self.round_index = int(round_index)
        self.scheduled_step = scheduled_step

    def mark_status(self, status: str, started_at: float | None = None) -> None:
        """Record a live execution/generation status and optional start time."""
        self.tag = status
        self.status_started_at = started_at

    @property
    def _round_index(self) -> int:
        return self.round_index

    @_round_index.setter
    def _round_index(self, value: int) -> None:
        self.round_index = int(value)

    @property
    def _scheduled_step(self) -> int | None:
        return self.scheduled_step

    @_scheduled_step.setter
    def _scheduled_step(self, value: int | None) -> None:
        self.scheduled_step = value

    @property
    def _status_started_at(self) -> float | None:
        return self.status_started_at

    @_status_started_at.setter
    def _status_started_at(self, value: float | None) -> None:
        self.status_started_at = value

    def record_experiment_context(self, best_node_before: Optional["Node"]) -> None:
        """Record simple experiment deltas after execution, before appending."""
        if self.metrics and self.metric_specs:
            # Population-relative comparisons belong to Journal.append, after
            # this result and its baselines share one ranking snapshot.
            if self.is_buggy:
                self.outcome = "bug"
            return
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

        if (
            best_node_before is not None
            and best_node_before.metric is not None
            and best_node_before.metric.value is not None
        ):
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
        return trim_execution_output("".join(self._term_out or []))

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
    metric_specs: list[dict[str, Any]] = field(default_factory=list)
    # A missing policy in historical JSON preserves its original selection.
    selection_policy: str = "legacy_primary"

    def __post_init__(self) -> None:
        if self.selection_policy not in {"legacy_primary", "average_rank"}:
            raise ValueError(f"Unknown selection policy {self.selection_policy!r}")
        if self.metric_specs:
            self.metric_specs = validated_metric_specs(self.metric_specs)

    def _prepare_metrics(self, node: Node) -> None:
        """Bind a result to the already frozen objectives before committing."""
        if self.selection_policy != "average_rank":
            return
        if node.metric_specs:
            incoming = validated_metric_specs(node.metric_specs)
            if not self.metric_specs:
                self.metric_specs = copy.deepcopy(incoming)
            elif {item["name"]: item for item in incoming} != {
                item["name"]: item for item in self.metric_specs
            }:
                raise ValueError(
                    "Candidate metrics disagree with the journal's frozen objectives"
                )
        if (
            not self.metric_specs
            and not node.metrics
            and node.metric is not None
            and finite_metric(node.metric.value) is not None
        ):
            # Compatibility for scalar-only callers without a prepared task.
            self.metric_specs = [
                {"name": "metric", "maximize": bool(node.metric.maximize)}
            ]
        if node.metric is not None and self.metric_specs == [
            {"name": "metric", "maximize": bool(node.metric.maximize)}
        ]:
            if not node.metrics and finite_metric(node.metric.value) is not None:
                node.metrics = {"metric": float(node.metric.value)}
        if node.metrics and not self.metric_specs:
            raise ValueError("Named candidate metrics require frozen metric directions")
        if self.metric_specs:
            node.metric_specs = copy.deepcopy(self.metric_specs)

    def ranking_snapshot(self) -> dict[str, Any]:
        """Return detached current ranks over all committed valid candidates."""
        if self.selection_policy == "average_rank" and self.metric_specs:
            snapshot = average_rank_snapshot(self.nodes, self.metric_specs)
        elif self.selection_policy == "average_rank":
            snapshot = {
                "selection_policy": "average_rank",
                "version": "unprepared",
                "candidate_count": 0,
                "metric_specs": [],
                "best_node_id": None,
                "ordered_node_ids": [],
                "nodes": {},
            }
        else:
            # Historical reports retain the original scalar ordering. Do not
            # retrospectively fold old auxiliary scores into their selection.
            ordered = sorted(
                (
                    n
                    for n in self.good_nodes
                    if completed_candidate(n)
                    and n.metric is not None
                    and finite_metric(n.metric.value) is not None
                ),
                key=lambda n: n.metric,
                reverse=True,
            )
            records = {
                n.id: {
                    "metrics": copy.deepcopy(n.metrics),
                    "metric_ranks": {},
                    "average_rank": None,
                    "worst_rank": None,
                    "step": n.step,
                    "position": index,
                    "is_best": index == 1,
                    "candidate_count": len(ordered),
                }
                for index, n in enumerate(ordered, 1)
            }
            revision = hashlib.sha256(
                json.dumps([(n.id, n.step, n.metric.value) for n in ordered]).encode()
            ).hexdigest()[:16]
            for record in records.values():
                record["snapshot_version"] = revision
            snapshot = {
                "selection_policy": "legacy_primary",
                "version": revision,
                "candidate_count": len(ordered),
                "metric_specs": copy.deepcopy(self.metric_specs),
                "best_node_id": ordered[0].id if ordered else None,
                "ordered_node_ids": [n.id for n in ordered],
                "nodes": records,
            }
        return snapshot

    def _apply_ranking_snapshot(self, snapshot: dict) -> None:
        """Refresh persisted current ranks only when the cohort is committed."""
        for node in self.nodes:
            node.ranking = copy.deepcopy(snapshot["nodes"].get(node.id))

    def ranked_nodes(self, snapshot: dict[str, Any] | None = None) -> list[Node]:
        """Order globally first; callers may then filter expandable nodes."""
        ranking = snapshot if snapshot is not None else self.ranking_snapshot()
        by_id = {node.id: node for node in self.nodes}
        return [
            by_id[node_id]
            for node_id in ranking["ordered_node_ids"]
            if node_id in by_id
        ]

    def _record_rank_context(self, node: Node, before: dict, current: dict) -> None:
        record = current["nodes"].get(node.id)
        best_before_id = before["best_node_id"]
        parent_id = node.parent.id if node.parent is not None else None
        node.rank_context = {
            "snapshot_version": current["version"],
            "candidate_count": current["candidate_count"],
            "best_node_before_id": best_before_id,
            "best_node_id": current["best_node_id"],
            "node": copy.deepcopy(record),
            "parent": copy.deepcopy(current["nodes"].get(parent_id)),
            "best_before": copy.deepcopy(current["nodes"].get(best_before_id)),
        }
        if record is None:
            node.outcome = "bug" if node.is_buggy else "invalid_metrics"
            return
        node.metric_deltas = {}
        for relation, baseline_id in (
            ("parent", parent_id),
            ("best_before", best_before_id),
        ):
            baseline = current["nodes"].get(baseline_id)
            if baseline is None:
                continue
            node.metric_deltas[relation] = {
                name: value - baseline["metrics"][name]
                for name, value in record["metrics"].items()
            }
            delta = baseline["average_rank"] - record["average_rank"]
            if relation == "parent":
                node.delta_vs_parent = delta
            else:
                node.delta_vs_best = delta
        pair = lambda item: (item["average_rank"], item["worst_rank"])
        parent_record = current["nodes"].get(parent_id)
        best_record = current["nodes"].get(best_before_id)
        if best_before_id is None:
            node.outcome = "first_success"
        elif current["best_node_id"] == node.id and best_before_id != node.id:
            node.outcome = "new_best"
        elif parent_record is not None and pair(record) < pair(parent_record):
            node.outcome = "improved_parent"
        elif parent_record is not None and pair(record) > pair(parent_record):
            node.outcome = "regressed_parent"
        elif best_record is not None and pair(record) == pair(best_record):
            node.outcome = "matched_best"
        else:
            node.outcome = "valid_non_improving"

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        self._prepare_metrics(node)
        before = (
            self.ranking_snapshot() if self.selection_policy == "average_rank" else None
        )
        node.step = len(self.nodes)
        self.nodes.append(node)
        if before is not None:
            current = self.ranking_snapshot()
            self._apply_ranking_snapshot(current)
            self._record_rank_context(node, before, current)

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
        if self.selection_policy == "average_rank":
            self._apply_ranking_snapshot(self.ranking_snapshot())
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

        if self.selection_policy == "average_rank":
            top_nodes = self.ranked_nodes()[:topk]
            return random.choice(top_nodes) if top_nodes else None
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
        good_nodes = self.ranked_nodes()
        return good_nodes[:topk]

    def generate_tree_index(self) -> str:
        """Return a compact factual index of every node in the search tree."""
        if not self.nodes:
            return "No previous nodes."
        snapshot = self.ranking_snapshot()

        legend = (
            "Column legend: step=completed node order; id=short node id; "
            "parent=short parent id or root; stage=draft/debug/improve; "
            "status=execution outcome; val_metric=current average rank (lower is better) for multiple objectives, raw validation value for one objective; "
            "delta_parent=change versus parent where positive means better; "
            "delta_best=change versus best node before this attempt where positive means better; average-rank deltas compare both nodes in their shared commit-time population; "
            "runtime=execution time."
        )
        header = (
            "step | id | parent | stage | status | val_metric | "
            "delta_parent | delta_best | runtime"
        )
        rows = [legend]
        if self.selection_policy == "average_rank":
            rows.append(
                f"Current ranking snapshot {snapshot['version']}: {snapshot['candidate_count']} valid candidates; "
                f"objectives={json_dumps_compact(snapshot['metric_specs'])}. "
                "Raw metric deltas are current minus baseline; interpret signs using each objective's direction."
            )
        rows.append(header)
        for n in self.nodes:
            rows.append(
                " | ".join(
                    [
                        "" if n.step is None else str(n.step),
                        n.id[:8],
                        n.parent.id[:8] if n.parent is not None else "root",
                        n.node_status,
                        _node_status_text(n),
                        _metric_text(n, snapshot),
                        _format_delta(n.delta_vs_parent),
                        _format_delta(n.delta_vs_best),
                        _format_runtime(n.exec_time),
                    ]
                )
            )
        return "\n".join(rows)

    def generate_tree_records(
        self,
        stage: Literal["draft", "debug", "improve"],
        selected_parent: Node | None = None,
        pending_same_round_nodes: list[Node] | None = None,
    ) -> str:
        """Return the single prompt section that describes tree context.

        The top overview stays compact. Full node records are only emitted in
        the ordered lineage and sibling sections so the selected parent is not
        repeated as a full record in multiple places.
        """
        sep = "\n\n---\n\n"
        snapshot = self.ranking_snapshot()
        parts = [
            "Overall tree\n"
            + _format_current_target(stage, selected_parent)
            + "\n\n"
            + self.generate_tree_index(),
        ]

        def append(part: str) -> None:
            parts.append(part)

        lineage = _lineage_from_root(selected_parent) if selected_parent else []
        if lineage:
            append(_tree_record_lineage_intro(stage, selected_parent))
            for node in lineage:
                append(
                    _format_detail_record(
                        node,
                        (
                            "root-to-parent lineage; selected parent"
                            if node is selected_parent
                            else "root-to-parent lineage ancestor"
                        ),
                        snapshot,
                    )
                )
        else:
            append(_tree_record_lineage_intro(stage, selected_parent))

        sibling_nodes = [
            node
            for node in _tree_record_siblings(stage, selected_parent, self.draft_nodes)
            if node in self.nodes
        ]
        if pending_same_round_nodes:
            sibling_nodes.extend(
                node for node in pending_same_round_nodes if node not in sibling_nodes
            )
        completed_sibling_nodes = [
            node for node in sibling_nodes if node.step is not None
        ]
        pending_sibling_nodes = [node for node in sibling_nodes if node.step is None]

        if completed_sibling_nodes:
            append(_tree_record_sibling_intro(stage, selected_parent))
            for node in sorted(completed_sibling_nodes, key=_node_context_sort_key):
                append(
                    _format_detail_record(
                        node, _tree_record_sibling_reason(stage), snapshot
                    )
                )
        else:
            append(
                _tree_record_sibling_intro(stage, selected_parent)
                + "\nNo completed sibling records were available when this prompt was built."
            )

        if pending_sibling_nodes:
            append(_tree_record_pending_sibling_intro(stage, selected_parent))
            for node in sorted(pending_sibling_nodes, key=_node_context_sort_key):
                append(
                    _format_detail_record(
                        node, _tree_record_pending_sibling_reason(stage), snapshot
                    )
                )

        return sep.join(parts)

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
            if n.outcome != "regressed_parent" and (
                n.delta_vs_parent is None or n.delta_vs_parent >= 0
            ):
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

    def generate_summary(
        self,
        include_code: bool = False,
        topk: int = 3,
        max_chars: int = 6000,
        include_avoid_list: bool = True,
    ) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        snapshot = self.ranking_snapshot()
        best_node = self.get_best_node(only_good=True, topk=1)
        if best_node is not None:
            summary.append("CURRENT BEST\n" + _format_memory_node(best_node, snapshot))

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
                + "\n\n".join(_format_memory_node(n, snapshot) for n in candidates)
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
    parts = []
    peak_rss = resource_usage.get("peak_rss_mb")
    if isinstance(peak_rss, (int, float)):
        if peak_rss >= 1024:
            parts.append(f"peak_rss {peak_rss / 1024:.2f}GB")
        else:
            parts.append(f"peak_rss {peak_rss:.0f}MB")
    peak_gpu = resource_usage.get("peak_gpu_memory_mb")
    if isinstance(peak_gpu, (int, float)):
        backend = resource_usage.get("gpu_backend")
        label = f"peak_{backend}" if backend else "peak_gpu"
        if peak_gpu >= 1024:
            parts.append(f"{label} {peak_gpu / 1024:.2f}GB")
        else:
            parts.append(f"{label} {peak_gpu:.0f}MB")
    if not parts:
        return "n/a"
    return ", ".join(parts)


def _node_sort_step(n: Node) -> int:
    return n.step if n.step is not None else 10**9


def _node_display_step(n: Node) -> int | None:
    if n.step is not None:
        return n.step
    value = n.scheduled_step
    return value if isinstance(value, int) else None


def _node_context_sort_key(n: Node) -> tuple[int, int | float]:
    display_step = _node_display_step(n)
    if display_step is not None:
        return (0, display_step)
    return (1, getattr(n, "ctime", 0) or 0)


def _recent_nodes(nodes) -> list[Node]:
    return sorted(list(nodes), key=_node_sort_step, reverse=True)


def _is_problem_node(n: Node) -> bool:
    return (
        n.is_buggy is True
        or n.exc_type is not None
        or n.outcome == "bug"
        or n.outcome == "regressed_parent"
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


def _lineage_from_root(node: Node | None) -> list[Node]:
    if node is None:
        return []
    return list(reversed(_ancestor_chain(node))) + [node]


def _tree_record_siblings(
    stage: Literal["draft", "debug", "improve"],
    selected_parent: Node | None,
    draft_nodes,
) -> list[Node]:
    if stage == "draft":
        return sorted(list(draft_nodes), key=_node_sort_step)
    if selected_parent is None:
        return []
    if stage == "debug":
        if selected_parent.parent is None:
            return [
                node
                for node in sorted(list(draft_nodes), key=_node_sort_step)
                if node is not selected_parent
            ]
        return [
            node
            for node in sorted(selected_parent.parent.children, key=_node_sort_step)
            if node is not selected_parent
        ]
    if stage == "improve":
        return sorted(selected_parent.children, key=_node_sort_step)
    return []


def _tree_record_sibling_title(stage: Literal["draft", "debug", "improve"]) -> str:
    if stage == "debug":
        return "Siblings of the Buggy Parent"
    if stage == "improve":
        return "Existing Children of the Selected Parent"
    return "Existing Root Drafts"


def _tree_record_lineage_intro(
    stage: Literal["draft", "debug", "improve"], selected_parent: Node | None
) -> str:
    title = "Parent Lineage"
    if selected_parent is None:
        return (
            f"{title}\n"
            "None. This is a new root draft, so there is no parent lineage to inherit."
        )
    if stage == "debug":
        instruction = (
            "This is the chain of implementations that led to the buggy parent. "
            "Keep that method's useful parts and fix the concrete failure."
        )
    else:
        instruction = (
            "This is the chain of implementations from the root to the selected "
            "parent. Build on the selected parent while keeping the useful context "
            "from its ancestors."
        )
    return f"{title}\n{instruction}"


def _tree_record_sibling_intro(
    stage: Literal["draft", "debug", "improve"], selected_parent: Node | None
) -> str:
    title = _tree_record_sibling_title(stage)
    if stage == "debug":
        if selected_parent is None:
            instruction = "There is no selected parent, so no buggy-parent sibling context is available."
        else:
            instruction = (
                "These records are siblings of the buggy parent: they were generated "
                "from the same parent as the node being debugged. Use them as nearby "
                "evidence, but the new debug node should fix the selected buggy parent "
                "rather than become a different sibling strategy."
            )
    elif stage == "improve":
        instruction = (
            "These are previous children of the same selected parent. Compare their "
            "plans, code, and results, then make this new child try a distinct "
            "improvement rather than repeating one of them."
        )
    else:
        instruction = (
            "These are previous root drafts. Use them as alternative starting points "
            "and avoid duplicating an existing attempt. If reusing any component, "
            "make the overall attempt substantively different."
        )
    return f"{title}\n{instruction}"


def _tree_record_pending_sibling_intro(
    stage: Literal["draft", "debug", "improve"], selected_parent: Node | None
) -> str:
    if stage == "debug":
        title = "Same-Round Siblings Not Yet Scored"
        instruction = (
            "These related sibling nodes had code generated when this prompt was built, "
            "but their execution result and validation metric were not available yet. "
            "Use their plan and code only as tentative context."
        )
    elif stage == "improve":
        title = "Same-Parent Children Not Yet Scored"
        instruction = (
            "These are other children of the selected parent whose code was already "
            "generated, but whose execution result and validation metric were not "
            "available yet. Use them as tentative sibling context and avoid repeating "
            "the same direction."
        )
    else:
        title = "Same-Round Root Drafts Not Yet Scored"
        instruction = (
            "These root drafts had code generated when this prompt was built, but "
            "their execution result and validation metric were not available yet. "
            "Use their plan and code as factual but unscored context; avoid "
            "duplicating the same attempt."
        )
    return f"{title}\n{instruction}"


def _tree_record_sibling_reason(stage: Literal["draft", "debug", "improve"]) -> str:
    if stage == "debug":
        return (
            "sibling of the selected parent; use as context while fixing the "
            "selected parent"
        )
    if stage == "improve":
        return (
            "existing child of the selected parent; sibling of the new improve " "node"
        )
    return "existing root draft; sibling of the new draft node"


def _tree_record_pending_sibling_reason(
    stage: Literal["draft", "debug", "improve"],
) -> str:
    if stage == "debug":
        return "same-round sibling of the buggy parent; generated but not scored yet"
    if stage == "improve":
        return "same-parent child; generated but not scored yet"
    return "same-round root draft; generated but not scored yet"


def _format_current_target(
    stage: Literal["draft", "debug", "improve"], selected_parent: Node | None
) -> str:
    if selected_parent is None:
        return (
            "Current target: generate a new root draft with no parent. Treat any "
            "existing root drafts as sibling alternatives, not as parents."
        )
    if stage == "debug":
        relation = (
            "Current target: generate a debug child of the selected buggy parent "
            "node. Preserve the selected parent's intended method and fix the "
            "observed failure."
        )
    else:
        relation = (
            "Current target: generate a child of the selected parent node. Base the "
            "new implementation on that parent and try to improve it."
        )
    return (
        f"{relation}\n"
        f"Selected parent node: {selected_parent.id} ({selected_parent.id[:8]})."
    )


def _top_successful_nodes(nodes) -> list[Node]:
    nodes = list(nodes)
    ranked = [
        node
        for node in nodes
        if node.ranking and node.ranking.get("average_rank") is not None
    ]
    if ranked:
        return sorted(
            ranked,
            key=lambda node: (
                node.ranking["average_rank"],
                node.ranking["worst_rank"],
                _node_sort_step(node),
                node.id,
            ),
        )
    return sorted(
        [n for n in nodes if n.metric is not None],
        key=lambda n: n.metric,
        reverse=True,
    )


def _best_node_deterministic(nodes) -> Node | None:
    top_nodes = _top_successful_nodes(nodes)
    return top_nodes[0] if top_nodes else None


def _metric_text(n: Node, snapshot: dict | None = None) -> str:
    ranking = snapshot["nodes"].get(n.id) if snapshot is not None else n.ranking
    specs = snapshot["metric_specs"] if snapshot is not None else n.metric_specs
    legacy = snapshot is not None and snapshot["selection_policy"] == "legacy_primary"
    if n.metrics and not legacy:
        if len(specs) > 1:
            return f"Avg rank {ranking['average_rank']:.4g}" if ranking else "unranked"
        if len(specs) == 1:
            value = n.metrics.get(specs[0]["name"])
            return "n/a" if value is None else f"{value:.6g}"
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


def _format_artifact_facts(
    report: dict | None, include_raw_report: bool = False
) -> str:
    if not isinstance(report, dict):
        return "artifact_status: n/a"

    status = _artifact_status(report)
    warnings = report.get("warnings")

    if not include_raw_report and not warnings and report.get("available") is not False:
        lines = ["validation output: valid"]
        alignment = report.get("alignment")
        if isinstance(alignment, dict):
            method = alignment.get("method", "n/a")
            matched = alignment.get("matched_rows", "n/a")
            lines.append(f"alignment: {method}, matched_rows={matched}")
        else:
            lines.append(f"alignment: {status}")
        lines.append("warnings: none")
        return "\n".join(lines)

    lines = [
        "Local evaluator artifact check; format/alignment facts only, no hidden labels.",
        f"artifact_status: {status}",
    ]

    alignment = report.get("alignment")
    if isinstance(alignment, dict):
        method = alignment.get("method", "n/a")
        matched = alignment.get("matched_rows", "n/a")
        lines.append(f"alignment: method={method}, matched_rows={matched}")

    validation = report.get("validation_artifact")
    if isinstance(validation, dict):
        columns = validation.get("columns")
        column_count = len(columns) if isinstance(columns, list) else "n/a"
        lines.append(
            "validation_artifact: "
            f"exists={validation.get('exists')}, "
            f"rows={validation.get('rows')}, "
            f"columns={column_count}"
        )

    target = report.get("target_artifact")
    if isinstance(target, dict):
        lines.append(
            "target_artifact: "
            f"exists={target.get('exists')}, rows={target.get('rows')}"
        )

    spec = report.get("prediction_spec")
    if isinstance(spec, dict):
        pred_cols = spec.get("prediction_columns")
        pred_col_count = len(pred_cols) if isinstance(pred_cols, list) else "n/a"
        lines.append(
            "prediction_spec: "
            f"type={spec.get('prediction_type')}, "
            f"metrics={spec.get('metrics', spec.get('primary_metric'))}, "
            f"prediction_columns={pred_col_count}"
        )

    summaries = report.get("prediction_column_summaries")
    if isinstance(summaries, dict) and summaries:
        nulls = 0
        mins = []
        maxes = []
        dtypes = set()
        for summary in summaries.values():
            if not isinstance(summary, dict):
                continue
            null_count = summary.get("null_count")
            if isinstance(null_count, (int, float)):
                nulls += int(null_count)
            min_value = summary.get("min")
            max_value = summary.get("max")
            if isinstance(min_value, (int, float)):
                mins.append(float(min_value))
            if isinstance(max_value, (int, float)):
                maxes.append(float(max_value))
            dtype = summary.get("dtype")
            if dtype:
                dtypes.add(str(dtype))
        range_text = (
            f", global_min={min(mins):.4g}, global_max={max(maxes):.4g}"
            if mins and maxes
            else ""
        )
        dtype_text = ",".join(sorted(dtypes)) if dtypes else "n/a"
        lines.append(
            "prediction_columns: " f"dtypes={dtype_text}, nulls={nulls}{range_text}"
        )

    if warnings:
        lines.append("warnings: " + truncate_text(str(warnings), max_chars=500))
    else:
        lines.append("warnings: none")

    if include_raw_report:
        lines.append("raw_artifact_report: " + json_dumps_compact(report))

    return "\n".join(lines)


def _format_evaluator_details(details: dict | None) -> str:
    if not isinstance(details, dict):
        return ""
    metrics = details.get("metrics")
    if isinstance(metrics, dict) and metrics:
        return ", ".join(
            f"{name}={float(value):.6g}"
            for name, value in metrics.items()
            if finite_metric(value) is not None
        )

    payload = details.get("details")
    if not isinstance(payload, dict):
        payload = details
    score = payload.get("score")
    if not isinstance(score, dict):
        return ""
    if isinstance(score.get("metrics"), dict):
        return ", ".join(
            f"{name}={float(value):.6g}"
            for name, value in score["metrics"].items()
            if finite_metric(value) is not None
        )

    parts = []
    skip_keys = {"metric", "maximize"}
    for key, value in score.items():
        if key in skip_keys or isinstance(value, bool):
            continue
        if isinstance(value, int):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        if len(parts) >= 8:
            break
    return ", ".join(parts)


def _execution_dict(n: Node) -> dict:
    return n.execution if isinstance(n.execution, dict) else {}


def _candidate_execution_text(n: Node) -> str:
    execution = _execution_dict(n)
    candidate = execution.get("candidate")
    if isinstance(candidate, dict):
        output = candidate.get("output")
        if isinstance(output, str) and output.strip():
            return output
        stdout = str(candidate.get("stdout") or "")
        stderr = str(candidate.get("stderr") or "")
        if stdout or stderr:
            return stdout + stderr
    return "".join(n._term_out or [])


def _format_failure(failure: object) -> str:
    if not isinstance(failure, dict):
        return "none"
    phase = failure.get("phase") or "unknown"
    failure_type = failure.get("type") or "unknown"
    info = failure.get("info")
    return redact_local_paths(
        truncate_text(
            f"{phase}: {failure_type}; info={json_dumps_compact(info)}",
            max_chars=700,
            marker="failure details trimmed",
        )
    )


def _format_execution_summary(n: Node, *, not_executed: bool) -> str:
    if not_executed:
        return "execution_status: not_executed"

    execution = _execution_dict(n)
    lines = [
        "execution_status: " + ("bug" if n.is_buggy is True else "success"),
        f"failure: {_format_failure(execution.get('final_failure'))}",
    ]

    candidate = execution.get("candidate")
    if isinstance(candidate, dict):
        returncode = candidate.get("returncode")
        if returncode is not None:
            lines.append(f"candidate_returncode: {returncode}")
        candidate_failure = candidate.get("failure")
        if candidate_failure:
            lines.append(f"candidate_failure: {_format_failure(candidate_failure)}")

    artifact_validation = execution.get("artifact_validation")
    if isinstance(artifact_validation, dict):
        lines.append(f"artifact_validation_ok: {artifact_validation.get('ok')}")
        messages = artifact_validation.get("messages")
        if messages:
            lines.append(
                "artifact_validation_messages: "
                + redact_local_paths(
                    truncate_text(
                        str(messages),
                        max_chars=800,
                        marker="artifact validation messages trimmed",
                    )
                )
            )

    evaluator = execution.get("evaluator")
    if isinstance(evaluator, dict):
        lines.append(
            "evaluator: "
            f"ok={evaluator.get('ok')}, "
            f"metrics={evaluator.get('metrics', evaluator.get('metric'))}"
        )
        if evaluator.get("error"):
            lines.append(
                "evaluator_error: "
                + redact_local_paths(
                    truncate_text(
                        str(evaluator.get("error")),
                        max_chars=500,
                        marker="evaluator error trimmed",
                    )
                )
            )
    return "\n".join(lines)


def _format_node_header(n: Node, snapshot: dict | None = None) -> str:
    metric_text = _metric_text(n, snapshot)
    detail_text = (
        json_dumps_compact(n.metrics)
        if n.metrics
        else _format_evaluator_details(n.evaluator_details)
    )
    if detail_text:
        detail_text = f" | validation_metrics: {detail_text}"
    return (
        f"Node {n.id[:8]} | status: {n.node_status} | "
        f"metric: {metric_text} | runtime: {_format_runtime(n.exec_time)} | "
        f"resources: {_format_resource_usage(n.resource_usage)} | "
        f"outcome: {n.outcome or 'unknown'}"
        f"{detail_text}"
    )


def _format_memory_node(n: Node, snapshot: dict | None = None) -> str:
    finding_text = (
        f"\nEmpirical finding: {truncate_text(str(n.analysis), max_chars=260)}"
        if n.analysis
        else ""
    )
    return (
        f"{_format_node_header(n, snapshot)}\n"
        f"delta_vs_parent: {_format_delta(n.delta_vs_parent)} | "
        f"delta_vs_best_before: {_format_delta(n.delta_vs_best)}\n"
        f"Core idea: {truncate_text(str(n.plan), max_chars=320)}"
        f"{finding_text}"
    )


def _format_detail_record(
    n: Node,
    reason: str,
    snapshot: dict | None = None,
) -> str:
    ranking = snapshot["nodes"].get(n.id) if snapshot is not None else n.ranking
    not_executed = n.step is None or n._term_out is None
    include_output_check = bool(
        n.is_buggy
        or (
            isinstance(n.artifact_report, dict)
            and (
                n.artifact_report.get("available") is False
                or bool(n.artifact_report.get("warnings"))
            )
        )
    )
    artifact_facts = (
        _format_artifact_facts(n.artifact_report, include_raw_report=False)
        if include_output_check
        else ""
    )
    resource_usage = (
        json_dumps_compact(n.resource_usage) if n.resource_usage is not None else "n/a"
    )
    evaluator_details = _format_evaluator_details(n.evaluator_details)
    exception_info = redact_local_paths(
        json_dumps_compact(
            {
                "exc_type": n.exc_type,
                "exc_info": n.exc_info,
                "exc_stack": n.exc_stack,
            }
        )
    )
    if not_executed:
        result_text = (
            "generated_not_scored_yet"
            if str(n.code or "").strip()
            else "generation_in_progress"
        )
    else:
        result_text = "bug" if n.is_buggy is True else "success"
    metric_text = "unknown_not_executed" if not_executed else _metric_text(n, snapshot)
    exec_time = "unknown_not_executed" if not_executed else _format_runtime(n.exec_time)
    parent_id = n.parent.id[:8] if n.parent is not None else "root"
    display_step = _node_display_step(n)
    if display_step is not None:
        step_title = f"Step {display_step}"
        step_value = str(display_step)
    elif str(n.code or "").strip():
        step_title = "generated, not scored"
        step_value = "generated_not_scored"
    else:
        step_title = "generating"
        step_value = "generating"
    children_ids = [child.id[:8] for child in sorted(n.children, key=_node_sort_step)]
    execution_output = (
        "Not executed yet. Execution output is unknown."
        if not_executed
        else format_prompt_execution_output(
            _candidate_execution_text(n), is_buggy=bool(n.is_buggy)
        )
    )
    execution_summary = _format_execution_summary(n, not_executed=not_executed)
    include_execution_output = bool(not_executed or n.is_buggy is True)
    output_check_block = (
        f"\n\nOutput Check:\n{artifact_facts}" if artifact_facts.strip() else ""
    )
    execution_output_block = (
        f"\n\nExecution Output:\n```\n{execution_output}\n```"
        if include_execution_output
        else ""
    )
    return (
        f"Node {n.id[:8]} · {step_title} · {n.node_status} · parent {parent_id}\n"
        f"step: {step_value}\n"
        f"node_id: {n.id[:8]}\n"
        f"parent_id: {parent_id}\n"
        f"children_ids: {children_ids}\n"
        f"stage: {n.node_status}\n"
        f"result: {result_text}\n"
        f"val_metric: {metric_text}\n"
        f"validation_metrics: {json_dumps_compact(n.metrics) if n.metrics else evaluator_details or 'n/a'}\n"
        f"metric_definitions: {json_dumps_compact(snapshot['metric_specs'] if snapshot is not None else n.metric_specs)}\n"
        f"current_ranking: {json_dumps_compact(ranking) if ranking else 'n/a'}\n"
        f"commit_rank_context: {json_dumps_compact(n.rank_context) if n.rank_context else 'n/a'}\n"
        f"metric_deltas_raw: {json_dumps_compact(n.metric_deltas)}\n"
        f"maximize: {n.metric.maximize if n.metric is not None else None}\n"
        f"delta_vs_parent: {_format_delta(n.delta_vs_parent)}\n"
        f"execution_time: {exec_time}\n"
        f"resource_usage: {resource_usage}\n"
        f"exception: {exception_info}\n"
        f"Execution Summary:\n{execution_summary}"
        f"{output_check_block}\n\n"
        f"Plan:\n{n.plan or ''}\n\n"
        f"Code:\n```python\n{n.code or ''}\n```"
        f"{execution_output_block}"
    )
