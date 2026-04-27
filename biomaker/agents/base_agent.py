import logging
import numbers
import random
from typing import Callable, cast

import humanize

from ..backend import FunctionSpec, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..utils import data_preview
from ..utils.config import Config
from ..utils.evaluator_setup import parse_external_validation
from ..utils.environment import environment_manifest_text
from ..utils.metric import MetricValue, WorstMetricValue
from ..utils.response import (
    extract_code,
    extract_text_up_to_code,
    truncate_text,
    wrap_code,
)

logger = logging.getLogger("biomaker")
ExecCallbackType = Callable[[str, bool], ExecutionResult]
FEEDBACK_CODE_CHARS = 6000

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

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search
        debuggable_nodes = [
            n
            for n in self.journal.buggy_nodes
            if (n.is_leaf and n.debug_depth < search_cfg.max_debug_depth)
        ]
        good_nodes = self.journal.good_nodes

        if not good_nodes:
            if debuggable_nodes:
                logger.debug("[search policy] debugging shallow buggy leaf")
                return random.choice(debuggable_nodes)

            if len(self.journal.draft_nodes) < search_cfg.num_drafts:
                logger.debug("[search policy] drafting new node (no good nodes)")
                return None

            logger.debug(
                "[search policy] drafting new node (no good or debuggable nodes)"
            )
            return None

        if random.random() < search_cfg.draft_prob_after_success:
            logger.debug("[search policy] drafting new node for diversity")
            return None

        if debuggable_nodes and random.random() < search_cfg.debug_prob:
            logger.debug("[search policy] debugging buggy leaf")
            return random.choice(debuggable_nodes)

        clean_frontier_nodes = [
            n for n in good_nodes if n.is_leaf and n.outcome != "regressed_parent"
        ]
        strong_nodes = [n for n in good_nodes if n.outcome != "regressed_parent"]
        leaf_nodes = [n for n in good_nodes if n.is_leaf]
        best_node = self.journal.get_best_node(only_good=True, topk=1)
        candidate_nodes = []
        for candidate in [
            best_node,
            *clean_frontier_nodes,
            *strong_nodes,
            *leaf_nodes,
            *good_nodes,
        ]:
            if candidate is not None and candidate not in candidate_nodes:
                candidate_nodes.append(candidate)
        top_nodes = sorted(candidate_nodes, key=lambda n: n.metric, reverse=True)[
            : search_cfg.topk_improve
        ]
        # Bias toward the current best, but keep other strong branches alive.
        weights = [1.0 / (rank + 1) for rank in range(len(top_nodes))]
        node = random.choices(top_nodes, weights=weights, k=1)[0]
        logger.debug(
            "[search policy] improving one of the top-%d frontier good nodes",
            len(top_nodes),
        )
        return node

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
            "A frozen evaluator is available as `./evaluator.py`; strictly follow `evaluator.prediction_spec()` for validation predictions and `./working/submission.csv`.",
            "BioMaker computes the final node score after your script exits by calling `evaluator.score_val()` with no arguments. You may call `evaluator.score_val()` or `evaluator.validate_submission()` yourself only after writing the required files.",
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

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

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

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        """core function for feedback for this Agent"""
        logger.debug(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)
        external_metric = parse_external_validation(node.term_out)

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
        best_node_before = self.journal.get_best_node(only_good=True, topk=1)
        node.record_experiment_context(best_node_before=best_node_before)

    def step(self, exec_callback: ExecCallbackType):
        """core logic of the actions of this Agent"""
        raise NotImplementedError
