import logging
import numbers
import json
from dataclasses import dataclass
from typing import Any, Callable, cast

import humanize

from ..backend import FunctionSpec, compile_prompt_to_md, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..utils import data_preview
from ..utils.config import Config
from ..utils.events import append_event
from ..utils.evaluator_setup import parse_artifact_report, parse_external_validation
from ..utils.environment import environment_manifest_text
from ..utils.metric import MetricValue, WorstMetricValue
from ..utils.response import (
    extract_code,
    extract_text_up_to_code,
    truncate_text,
    wrap_code,
)

logger = logging.getLogger("biomaker")
ExecCallbackType = Callable[..., ExecutionResult]
FEEDBACK_CODE_CHARS = 6000


@dataclass(frozen=True)
class RoundTask:
    parent: Node | None
    selection_reason: str

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "If there is a bug, write one concise factual sentence naming where execution failed and the direct cause; do not propose a fix. Otherwise, write 2-3 concise sentences describing the metric, runtime/resource facts when relevant, and the empirical finding.",
            },
            "metric": {
                "type": ["number", "null"],
                "description": "If BioMakerExternalValidation is present, report that validation metric. If the code failed, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the authoritative metric should be minimized, false if it should be maximized.",
            },
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the training script.",
)


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
        self.data_preview: str | None = None
        self.run_facts: str | None = None
        self._last_selection_reason: str | None = None
        self.environment_facts = environment_manifest_text(
            include_medical=self.cfg.exec.include_medical_packages
        )
        self.profile = "You are a helpful assistance. "

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
        if len(self.journal.draft_nodes) < search_cfg.max_children_per_node:
            current = len(self.journal.draft_nodes)
            return [
                RoundTask(
                    parent=None,
                    selection_reason=(
                        "bootstrap selected virtual root draft "
                        f"({current + idx + 1}/{search_cfg.max_children_per_node})"
                    ),
                )
                for idx in range(search_cfg.max_children_per_node - current)
            ]

        tasks: list[RoundTask] = []
        improve_candidates = [
            n
            for n in self.journal.good_nodes
            if (
                n.metric is not None
                and len(n.children) < search_cfg.max_children_per_node
            )
        ]
        if improve_candidates:
            ranked_candidates = sorted(
                improve_candidates,
                key=lambda n: (
                    n.metric,
                    -(n.step if n.step is not None else 10**9),
                ),
                reverse=True,
            )
            top_candidates = ranked_candidates[: search_cfg.topk_improve]
            for selected in top_candidates:
                rank = ranked_candidates.index(selected)
                tasks.append(
                    RoundTask(
                        parent=selected,
                        selection_reason=(
                            "round global frontier selected node "
                            f"{selected.id[:8]} rank {rank + 1} "
                            f"within top {len(top_candidates)} successful expandable nodes"
                        ),
                    )
                )

        debuggable_nodes = [
            n
            for n in self.journal.buggy_nodes
            if (
                n.debug_depth < search_cfg.max_debug_depth
                and len(n.children) < search_cfg.num_bugs
            )
        ]
        for selected_bug in sorted(
            debuggable_nodes,
            key=lambda n: n.step if n.step is not None else 10**9,
        ):
            tasks.append(
                RoundTask(
                    parent=selected_bug,
                    selection_reason=(
                        "round frontier debug selected buggy node "
                        f"(depth {selected_bug.debug_depth + 1}/"
                        f"{search_cfg.max_debug_depth})"
                    ),
                )
            )

        if not tasks:
            return [
                RoundTask(
                    parent=None,
                    selection_reason=(
                        "no successful expandable node remains after bootstrap drafts"
                    ),
                )
            ]

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
        impl_guideline = [
            "The code should **implement the proposed solution** and write final validation predictions in the format required by `./evaluator.py`.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped. Do not terminate before finishing the script.",
            "Your response should contain a brief plan followed by exactly one Python code block.",
            "If you use PyTorch, the code must run on CPU, CUDA, and Apple MPS by selecting the available device dynamically; do not hard-code CUDA or MPS-only execution.",
            "Do not import unlisted third-party packages, and do not install packages.",
            "You may use pretrained model weights from installed libraries when useful, including downloading those weights at runtime. Do not install packages or modify the uv environment.",
            "Use observed validation metric, runtime, and resource history when deciding training budget; if you change the budget, make it an explicit part of the proposed experiment.",
            f"The per-node execution timeout is a hard safety limit of {humanize.naturaldelta(self.cfg.exec.timeout)}, not a target budget. Choose the experiment scale from the task, data, observed validation metric, runtime, and resource history.",
            'All the provided input data is stored in "./input" directory.',
            "When building data loading code, base it on the actual existing data files rather than relying entirely on the annotation file.",
            "A frozen evaluator is available as `./evaluator.py`; strictly follow `evaluator.prediction_spec()` for score-validation predictions and `./working/submission.csv`.",
            "BioMaker computes the final node score after your script exits using hidden score labels that are not available to generated code. You may call `evaluator.validate_submission()` yourself after writing the required files, but do not call `evaluator.score_val()` inside generated code because the hidden score labels are only supplied to BioMaker's external scoring process.",
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and writes the required prediction artifacts. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

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
        prompt_capture: dict[str, Any] = self._prompt_capture(prompt, None)
        for _ in range(retries):
            completion_text, metadata = cast(
                tuple[str, dict[str, Any]],
                query(
                    system_message=prompt,
                    user_message=None,
                    model=self.acfg.code.model,
                    temperature=self.acfg.code.temp,
                    return_metadata=True,
                ),
            )
            prompt_capture = self._prompt_capture(prompt, metadata)

            completion_text = cast(str, completion_text)
            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code, prompt_capture

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text, prompt_capture  # type: ignore

    def wrap_execution_output(self, node: Node) -> str:
        """Return the execution output in the form shown to debug prompts."""
        return wrap_code(node.term_out, lang="")

    def update_data_preview(self):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)
        self.run_facts = data_preview.biomaker_essentials(self.cfg.workspace_dir)

    def get_run_facts(self) -> str:
        """Return deterministic task facts that should be stable for the whole run."""
        if self.run_facts is None:
            self.run_facts = data_preview.biomaker_essentials(self.cfg.workspace_dir)
        return self.run_facts

    def attach_data_overview(
        self,
        prompt: dict,
        max_chars: int | None = None,
        essentials_first: bool = False,
    ) -> None:
        if self.acfg.data_preview and self.data_preview:
            overview = self.data_preview
            if essentials_first and max_chars is not None:
                essentials = self.get_run_facts()
                if essentials:
                    remaining = max_chars - len(essentials) - 80
                    if remaining > 300:
                        overview = (
                            essentials
                            + "\n\nAdditional dataset preview excerpt:\n"
                            + truncate_text(
                                self.data_preview,
                                max_chars=remaining,
                                marker="data overview excerpt trimmed",
                            )
                        )
                    else:
                        overview = essentials
            prompt["Data Overview"] = (
                truncate_text(
                    overview,
                    max_chars=max_chars,
                    marker="data overview trimmed",
                )
                if max_chars is not None
                else overview
            )

    def parse_exec_result(
        self,
        node: Node,
        exec_result: ExecutionResult,
        best_node_before: Node | None = None,
    ):
        """core function for feedback for this Agent"""
        logger.debug(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)
        external_metric = parse_external_validation(node.term_out)
        artifact_report = parse_artifact_report(exec_result.term_out)
        node.artifact_report = artifact_report

        prompt = {
            "Introduction": (
                self.profile
                + "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs and summarize the empirical findings. "
                "If a line beginning with BioMakerExternalValidation is present, it is the authoritative validation score. "
                "For failed runs, the summary should be a factual failure note only, not repair advice."
            ),
            "Task description": self.task_desc,
            "Execution facts": {
                "exception_type": node.exc_type,
                "exception_info": node.exc_info,
                "execution_time_seconds": node.exec_time,
                "resource_usage": node.resource_usage,
                "authoritative_external_metric": external_metric,
                "artifact_report": (
                    truncate_text(
                        json.dumps(
                            artifact_report,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ),
                        max_chars=6000,
                        marker="artifact report trimmed",
                    )
                    if artifact_report is not None
                    else None
                ),
            },
            "Implementation excerpt": wrap_code(
                truncate_text(
                    node.code,
                    max_chars=FEEDBACK_CODE_CHARS,
                    marker="feedback code excerpt trimmed",
                )
            ),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response = query(
            system_message=prompt,
            user_message=None,
            func_spec=review_func_spec,
            model=self.acfg.feedback.model,
            temperature=self.acfg.feedback.temp,
        )
        if not isinstance(response, dict):
            raise TypeError(
                f"Expected review response to be a dict, got {type(response)}"
            )
        response = cast(dict, response)

        metric = response["metric"]
        if isinstance(metric, bool) or not isinstance(metric, numbers.Real):
            response["metric"] = None
        if external_metric is not None:
            response["metric"], maximize = external_metric
            response["lower_is_better"] = not maximize
            response["is_bug"] = False

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )
        if best_node_before is None:
            best_node_before = self.journal.get_best_node(only_good=True, topk=1)
        node.record_experiment_context(best_node_before=best_node_before)
        append_event(
            self.cfg.log_dir,
            "node_evaluated",
            node_id=node.id,
            parent_id=node.parent.id if node.parent is not None else None,
            stage=node.node_status,
            is_buggy=node.is_buggy,
            metric=None if node.metric is None else node.metric.value,
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
