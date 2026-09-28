import logging
import json
import copy
from dataclasses import dataclass
from typing import Any, Callable, cast

from ..backend import compile_prompt_to_md, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..utils import data_summary
from ..utils.config import Config
from ..utils.events import append_event
from ..utils.evaluator_setup import (
    parse_artifact_report,
    parse_external_validation,
    parse_external_validation_details,
)
from ..utils.environment import environment_manifest_text
from ..utils.metric import MetricValue, WorstMetricValue
from ..utils.ranking import finite_metric, validated_metric_specs
from ..utils.response import (
    extract_code_blocks,
    extract_text_up_to_code,
    truncate_text,
    wrap_code,
)

logger = logging.getLogger("biomaker")
ExecCallbackType = Callable[..., ExecutionResult]
CODE_PROMPT_CHAR_BUDGET = 4_000_000
CODE_PROMPT_SECTION_FALLBACKS = [
    ("Tool Observations", 40_000),
    ("Data and Evaluation Summary", 32_000),
    ("Task Description", 32_000),
]


@dataclass(frozen=True)
class RoundTask:
    parent: Node | None
    selection_reason: str
    draft_plan: str | None = None
    selection_snapshot: dict[str, Any] | None = None


def parse_factual_exec_result(
    node: Node,
    exec_result: ExecutionResult,
    *,
    best_node_before: Node | None = None,
    record_context: bool = False,
) -> None:
    """Parse factual execution/evaluator output into a node."""
    node.absorb_exec_result(exec_result)
    node.metrics = {}
    node.metric_specs = []
    evaluator_result = (
        exec_result.evaluator_result
        if isinstance(exec_result.evaluator_result, dict)
        else None
    )
    if evaluator_result is None:
        logged = parse_external_validation_details(exec_result.term_out)
        if isinstance(logged, dict) and isinstance(logged.get("metrics"), dict):
            evaluator_result = dict(logged)
            # The legacy EXTERNAL_VALIDATION log is emitted only after scoring
            # succeeds. An explicit failure remains authoritative if present.
            evaluator_result.setdefault("ok", True)
    external_metric = None
    external_metrics = None
    metric_specs = []
    if evaluator_result is not None:
        details = evaluator_result.get("details")
        if not isinstance(details, dict):
            details = {}
        node.artifact_report = details.get("artifact_report") or parse_artifact_report(
            exec_result.term_out
        )
        score_details = {k: v for k, v in details.items() if k != "artifact_report"}
        node.evaluator_details = {
            "metric": evaluator_result.get("metric"),
            "maximize": bool(evaluator_result.get("maximize", True)),
            "details": score_details,
            "failure_kind": evaluator_result.get("failure_kind"),
            "error": evaluator_result.get("error"),
        }
        values = evaluator_result.get("metrics")
        if not isinstance(values, dict):
            legacy_score = score_details.get("score", {})
            values = (
                legacy_score.get("metrics") if isinstance(legacy_score, dict) else None
            )
        legacy_score = score_details.get("score", {})
        metric_specs = (
            evaluator_result.get("metric_specs")
            or (
                legacy_score.get("metric_specs")
                if isinstance(legacy_score, dict)
                else None
            )
            or []
        )
        if metric_specs:
            metric_specs = validated_metric_specs(metric_specs)
        if isinstance(values, dict):
            # Preserve finite diagnostic values as well as declared objectives.
            # Completeness is checked against the frozen target names below.
            external_metrics = {
                name: finite_metric(value) for name, value in values.items()
            }
            node.metrics = {
                name: value
                for name, value in external_metrics.items()
                if value is not None
            }
            node.metric_specs = metric_specs
            node.evaluator_details["metrics"] = node.metrics
            node.evaluator_details["metric_specs"] = metric_specs
        metric = evaluator_result.get("metric")
        if evaluator_result.get("ok") and not isinstance(metric, bool):
            try:
                external_metric = (
                    finite_metric(metric),
                    bool(evaluator_result.get("maximize", True)),
                )
            except (TypeError, ValueError):
                external_metric = None
    else:
        node.evaluator_details = parse_external_validation_details(exec_result.term_out)
        node.artifact_report = parse_artifact_report(exec_result.term_out)
        external_metric = parse_external_validation(exec_result.term_out)

    node.analysis = ""
    if (
        evaluator_result is not None
        and not evaluator_result.get("ok")
        and evaluator_result.get("failure_kind") in {
            "timeout", "resource", "interrupted", "execution", "protocol",
        }
        and exec_result.candidate_exc_type is None
        and exec_result.candidate_returncode in (None, 0)
        and (exec_result.artifact_validation or {}).get("ok", True)
    ):
        # No official score exists, and there is no demonstrated candidate bug.
        # Keep its saved outputs for re-scoring; do not offer it for debugging
        # or rank a partial metric vector as a completed experiment.
        node.is_buggy = None
        node.outcome = "unscored"
        node.metric = None
        node.metrics = {}
        return
    if external_metrics is not None:
        declared_names = [spec["name"] for spec in metric_specs] or list(
            external_metrics
        )
        valid_result = (
            bool(evaluator_result.get("ok"))
            and bool(declared_names)
            and all(external_metrics.get(name) is not None for name in declared_names)
        )
    else:
        valid_result = (
            external_metric is not None
            and finite_metric(external_metric[0]) is not None
        )
    node.is_buggy = node.exc_type is not None or not valid_result
    if node.is_buggy:
        node.metric = WorstMetricValue()
        if not record_context:
            node.outcome = "bug"
    else:
        if external_metrics is not None and len(metric_specs) == 1:
            spec = metric_specs[0]
            node.metric = MetricValue(
                node.metrics[spec["name"]], maximize=spec["maximize"]
            )
        elif external_metrics is not None and len(metric_specs) > 1:
            node.metric = None
        elif external_metric is not None:
            metric, maximize = external_metric
            node.metric = MetricValue(metric, maximize=maximize)
        else:
            node.metric = None

    if record_context:
        node.record_experiment_context(best_node_before=best_node_before)


def _estimated_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class BaseAgent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_summary: str | None = None
        self._last_selection_reason: str | None = None
        self.environment_facts = environment_manifest_text(
            include_medical=self.cfg.exec.include_medical_packages
        )
        self.profile = "You are a factual biomedical machine learning assistant. "

    def has_empty_journal(self) -> bool:
        """Check if the journal is empty."""
        return not self.journal.nodes or len(self.journal.nodes) <= 0

    def update_with_external_journal(self, external_journal: Journal):
        if not isinstance(external_journal, Journal):
            raise ValueError("external_journal must be an instance of Journal")
        for n in external_journal.nodes:
            self.journal.append(n)

    def select_round_tasks(self) -> list[RoundTask]:
        """Select the next batch of tree-search tasks from the current journal.

        This freezes the empirical search frontier for one scheduler round. Code
        generation may happen serially afterward, but the selected parent set is
        based only on completed nodes already present in the journal.
        """
        search_cfg = self.acfg.search
        snapshot = self.journal.ranking_snapshot()

        def journal_child_count(node: Node) -> int:
            return sum(1 for child in self.journal.nodes if child.parent is node)

        if len(self.journal.draft_nodes) < search_cfg.max_children_per_node:
            current = len(self.journal.draft_nodes)
            missing = search_cfg.max_children_per_node - current
            return [
                RoundTask(
                    parent=None,
                    selection_snapshot=copy.deepcopy(snapshot),
                    selection_reason=(
                        "bootstrap selected virtual root draft "
                        f"({current + idx + 1}/{search_cfg.max_children_per_node})"
                    ),
                )
                for idx in range(missing)
            ]

        tasks: list[RoundTask] = []
        improve_candidates = [
            n
            for n in self.journal.ranked_nodes(snapshot)
            if (journal_child_count(n) < search_cfg.max_children_per_node)
        ]
        if improve_candidates:
            ranked_candidates = improve_candidates
            top_candidates = ranked_candidates[: search_cfg.topk_improve]
            for selected in top_candidates:
                rank = snapshot["ordered_node_ids"].index(selected.id)
                tasks.append(
                    RoundTask(
                        parent=selected,
                        selection_snapshot=copy.deepcopy(snapshot),
                        selection_reason=(
                            "round global frontier selected node "
                            f"{selected.id[:8]} rank {rank + 1} "
                            f"of {snapshot['candidate_count']} valid candidates; "
                            f"within top {len(top_candidates)} successful expandable nodes "
                            f"(snapshot {snapshot['version']})"
                        ),
                    )
                )

        debuggable_nodes = [
            n
            for n in self.journal.buggy_nodes
            if (
                n.debug_depth < search_cfg.max_debug_depth
                and journal_child_count(n) == 0
            )
        ]
        for selected_bug in sorted(
            debuggable_nodes,
            key=lambda n: n.step if n.step is not None else 10**9,
        ):
            tasks.append(
                RoundTask(
                    parent=selected_bug,
                    selection_snapshot=copy.deepcopy(snapshot),
                    selection_reason=(
                        "round frontier debug selected buggy node "
                        f"(depth {selected_bug.debug_depth + 1}/"
                        f"{search_cfg.max_debug_depth})"
                    ),
                )
            )

        return tasks

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        tasks = self.select_round_tasks()
        if not tasks:
            self._last_selection_reason = "no search task selected"
            return None

        selected_task = tasks[0]
        self._last_selection_reason = selected_task.selection_reason
        return selected_task.parent

    @property
    def _prompt_environment(self):
        return {"Environment contract": self.environment_facts}

    @property
    def _prompt_impl_guideline(self):
        per_node_timeout = getattr(getattr(self.cfg, "exec", None), "timeout", None)
        timeout_text = (
            f"{per_node_timeout} seconds"
            if per_node_timeout is not None
            else "the configured per-node timeout"
        )
        impl_guideline = [
            "The code should **implement the proposed solution** and write final validation predictions in the format required by `./evaluator.py`.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped. Do not terminate before finishing the script.",
            "Your response should contain a brief plan followed by exactly one Python code block.",
            "If you use PyTorch, the code must run on CPU, CUDA, and Apple MPS by selecting the available device dynamically; do not hard-code CUDA or MPS-only execution.",
            "Do not import unlisted third-party packages, and do not install packages.",
            "You may use pretrained model weights from installed libraries when useful, including downloading those weights at runtime. Do not install packages or modify the uv environment.",
            "Use all observed validation metrics, runtime, and resource history when deciding training budget; if you change the budget, make it an explicit part of the proposed experiment.",
            "When relevant, consider reducing execution time and improving GPU utilization while preserving validation quality.",
            f"The per-node execution timeout is {timeout_text}. This is a hard safety limit, not a target budget. Choose the experiment scale from the task, data, observed validation metric, runtime, and resource history so the full script is likely to finish well before the timeout.",
            'All the provided input data is stored in "./input" directory.',
            "When building data loading code, base it on the actual existing data files rather than relying entirely on the annotation file.",
            "A frozen evaluator is available as `./evaluator.py`; strictly follow `evaluator.prediction_spec()` and the Experiment guide's prediction_jobs. Write the artifact for each declared prediction job using exactly its expected sample list.",
            "The Experiment guide's evaluation_protocol provides the complete frozen output and metric contract, including thresholds, class order, units, crops and aggregation when applicable. Harness owns final metric calculation; your program produces predictions. Choose the model, loss, training strategy, augmentation and preprocessing yourself within the task requirements, and convert final predictions back to the declared evaluation representation.",
            "BioMaker computes the final validation metrics after your script exits using hidden score labels that are not available to generated code. For a declared test prediction job, you may call `evaluator.validate_submission()` after writing its artifact. Do not call `evaluator.score_val()` inside generated code because the hidden score labels are only supplied to BioMaker's external scoring process.",
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
        ]
        if (
            getattr(getattr(self, "journal", None), "selection_policy", None)
            == "average_rank"
        ):
            impl_guideline.append(
                "Every declared metric is an equally important objective with its own optimization direction. "
                "BioMaker ranks all completed valid candidates independently per metric, then selects by mean rank, "
                "worst individual rank, and earlier step. Use raw metric values to explain tradeoffs; ranks are relative "
                "to the stated candidate population, so changes across different populations are not performance gains."
            )
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response must start with a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by exactly one markdown Python code block whose opening fence is exactly ```python. "
                "The code block must implement this solution and write the required prediction artifacts. "
                "There should be no additional headings, extra code blocks, or text after the code block. "
                "Historical code blocks in the prompt are context only; do not copy their fences into the final answer except for the single final ```python block. "
            )
        }

    def _compiled_prompt_chars(self, prompt: dict[str, Any]) -> int:
        return len(compile_prompt_to_md(prompt))

    def _budget_code_prompt(
        self, prompt: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Apply a last-resort prompt budget while preserving full context when possible."""
        if self._compiled_prompt_chars(prompt) <= CODE_PROMPT_CHAR_BUDGET:
            return prompt, []

        budgeted = dict(prompt)
        notes = [
            (
                f"Compiled prompt exceeded {CODE_PROMPT_CHAR_BUDGET} chars; "
                "BioMaker applied last-resort section caps."
            )
        ]
        for key, section_cap in CODE_PROMPT_SECTION_FALLBACKS:
            if self._compiled_prompt_chars(budgeted) <= CODE_PROMPT_CHAR_BUDGET:
                break
            if key not in budgeted:
                continue
            section_text = compile_prompt_to_md(budgeted[key]).strip()
            if len(section_text) <= section_cap:
                continue
            budgeted[key] = truncate_text(
                section_text,
                max_chars=section_cap,
                marker=f"{key} trimmed by prompt budget",
            )
            notes.append(f"{key} capped at {section_cap} chars.")

        if notes:
            budgeted["Prompt Budget Notes"] = notes
        return budgeted, notes

    def _prompt_capture(self, prompt: dict, metadata: dict | None) -> dict[str, Any]:
        sections = []
        for key, value in prompt.items():
            section_text = compile_prompt_to_md(value)
            sections.append(
                {
                    "name": str(key),
                    "chars": len(section_text),
                    "estimated_tokens": _estimated_tokens(section_text),
                }
            )

        metadata = metadata or {}
        sent_messages = metadata.get("sent_messages") or []
        sent_chars = sum(
            len(str(message.get("content") or "")) for message in sent_messages
        )
        return {
            "prompt": _json_safe(prompt),
            "messages": _json_safe(sent_messages),
            "stats": {
                "model": metadata.get("model", self.acfg.code.model),
                "provider": metadata.get("provider"),
                "request_time_seconds": metadata.get("request_time_seconds"),
                "input_tokens": metadata.get("prompt_tokens"),
                "output_tokens": metadata.get("completion_tokens"),
                "estimated_input_tokens": _estimated_tokens(
                    "\n".join(
                        str(message.get("content") or "") for message in sent_messages
                    )
                ),
                "sent_chars": sent_chars,
                "compiled_system_chars": metadata.get("compiled_system_chars"),
                "compiled_user_chars": metadata.get("compiled_user_chars"),
                "sections": sections,
                "model_info": metadata.get("model_info"),
            },
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str, dict[str, Any]]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        query_prompt, budget_notes = self._budget_code_prompt(prompt)
        if budget_notes:
            append_event(
                self.cfg.log_dir,
                "prompt_budget_applied",
                notes=budget_notes,
                compiled_chars=self._compiled_prompt_chars(query_prompt),
            )
        prompt_capture: dict[str, Any] = self._prompt_capture(query_prompt, None)
        last_reason = "no attempt made"
        for attempt in range(1, retries + 1):
            completion_text, metadata = cast(
                tuple[str, dict[str, Any]],
                query(
                    system_message=query_prompt,
                    user_message=None,
                    model=self.acfg.code.model,
                    temperature=self.acfg.code.temp,
                    return_metadata=True,
                ),
            )
            prompt_capture = self._prompt_capture(query_prompt, metadata)

            completion_text = cast(str, completion_text)
            code_blocks = extract_code_blocks(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if len(code_blocks) == 1:
                return (
                    nl_text or "No separate natural-language plan was returned.",
                    code_blocks[0],
                    prompt_capture,
                )

            if not code_blocks:
                last_reason = "no valid Python code block"
            else:
                last_reason = f"{len(code_blocks)} valid Python code blocks"

            append_event(
                self.cfg.log_dir,
                "plan_code_extraction_retry",
                attempt=attempt,
                reason=last_reason,
                response_chars=len(completion_text),
            )
            logger.warning(
                "Plan + code extraction failed on attempt %d/%d: %s",
                attempt,
                retries,
                last_reason,
            )
        raise RuntimeError(
            "Plan + code extraction failed after "
            f"{retries} attempts ({last_reason}). Last response length: "
            f"{len(completion_text or '')} chars."
        )

    def wrap_execution_output(self, node: Node) -> str:
        """Return the execution output in the form shown to debug prompts."""
        return wrap_code(node.term_out, lang="")

    def update_data_summary(self):
        self.data_summary = data_summary.data_and_evaluation_summary(
            self.cfg.workspace_dir,
            max_chars=None,
        )

    def get_data_summary(self) -> str:
        """Return deterministic data/evaluator facts that are stable for the run."""
        if self.data_summary is None:
            self.update_data_summary()
        return self.data_summary or ""

    def attach_data_summary(
        self,
        prompt: dict,
        max_chars: int | None = None,
    ) -> None:
        if not self.acfg.data_summary:
            return
        if self.data_summary is None:
            self.data_summary = data_summary.data_and_evaluation_summary(
                self.cfg.workspace_dir,
                max_chars=None,
            )
        if not self.data_summary:
            return
        prompt["Data and Evaluation Summary"] = (
            truncate_text(
                self.data_summary,
                max_chars=max_chars,
                marker="data and evaluation summary trimmed",
            )
            if max_chars is not None
            else self.data_summary
        )

    def parse_exec_result(
        self,
        node: Node,
        exec_result: ExecutionResult,
        best_node_before: Node | None = None,
    ):
        """Parse factual execution results and evaluator output for a node."""
        logger.debug(f"Agent is parsing execution results for node {node.id}")
        if best_node_before is None:
            best_node_before = self.journal.get_best_node(only_good=True, topk=1)
        parse_factual_exec_result(
            node,
            exec_result,
            best_node_before=best_node_before,
            record_context=True,
        )
        append_event(
            self.cfg.log_dir,
            "node_evaluated",
            node_id=node.id,
            parent_id=node.parent.id if node.parent is not None else None,
            stage=node.node_status,
            is_buggy=node.is_buggy,
            metric=None if node.metric is None else node.metric.value,
            metrics=node.metrics,
            metric_specs=node.metric_specs,
            selection_snapshot=node.selection_snapshot,
            maximize=None if node.metric is None else node.metric.maximize,
            outcome=node.outcome,
            exec_time=node.exec_time,
            resource_usage=node.resource_usage,
            exc_type=node.exc_type,
            artifact_available=(
                node.artifact_report.get("available")
                if isinstance(node.artifact_report, dict)
                else None
            ),
        )

    def step(self, exec_callback: ExecCallbackType):
        """core logic of the actions of this Agent"""
        raise NotImplementedError
