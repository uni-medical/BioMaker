"""Deterministic prompt assembly for proposal nodes."""

from __future__ import annotations

from typing import Any

import humanize

from ..journal import Node

DRAFT_DATA_OVERVIEW_CHARS = 6000
DEBUG_DATA_OVERVIEW_CHARS = 2500
IMPROVE_DATA_OVERVIEW_CHARS = 2500
DEBUG_CODE_CHARS = 10000
REFERENCE_CODE_CHARS = 5000
IMPROVE_CODE_CHARS = 12000
MEMORY_CHARS = 4500
BRANCH_FACTS_CHARS = 10000


def prompt_shape(prompt: dict[str, Any]) -> dict[str, Any]:
    """Return field names and approximate sizes without storing full prompt text."""
    shape = {}
    for key, value in prompt.items():
        shape[key] = {
            "type": type(value).__name__,
            "chars": len(str(value)),
        }
    return shape


class ContextBuilder:
    """Build the exact context shown to a draft/debug/improve node.

    This class contains no model calls and makes no search decisions. It only
    centralizes the same factual prompt assembly that used to be spread across
    the proposal agent methods.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def _base_prompt(self, introduction: str) -> dict[str, Any]:
        return {
            "Introduction": introduction,
            "Task description": self.agent.task_desc,
            "Instructions": {},
        }

    def _finish_common(self, prompt: dict[str, Any]) -> dict[str, Any]:
        prompt["Instructions"] |= self.agent._prompt_resp_fmt
        prompt["Instructions"] |= self.agent._prompt_impl_guideline
        prompt["Instructions"] |= self.agent._prompt_environment
        return prompt

    def build_draft(self) -> dict[str, Any]:
        prompt = self._base_prompt(
            self.agent.profile
            + "In order to win this competition, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        )
        prompt["Memory"] = self.agent.journal.generate_summary()
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution. Keep the frozen evaluator protocol unchanged.",
                "The solution sketch should be 3-5 sentences.",
                "Use the frozen evaluator as the source of truth for validation scoring.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                "Check the data preview of `./input` directory carefully, and make sure the data loading and batching logic matches the actual files.",
                "Keep the first solution cheap; one epoch is enough to validate the workflow.",
                "If the dataset is large, use a small representative training subset for the first draft so the full workflow finishes quickly.",
            ],
        }
        self._finish_common(prompt)
        self.agent.attach_data_overview(prompt, max_chars=DRAFT_DATA_OVERVIEW_CHARS)
        return prompt

    def build_debug(self, parent_node: Node) -> dict[str, Any]:
        reference_node = self.agent.journal.get_nearest_successful_ancestor(parent_node)
        if reference_node is None:
            reference_node = self.agent.journal.get_best_node(only_good=True, topk=1)
        prompt = self._base_prompt(
            self.agent.profile
            + "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
        prompt["Previous (buggy) implementation"] = self.agent._wrap_code_context(
            parent_node.code, max_chars=DEBUG_CODE_CHARS
        )
        prompt["Execution output"] = self.agent.wrap_execution_output(parent_node)
        prompt["Debug memory"] = self.agent.journal.generate_debug_summary(parent_node)
        if reference_node is not None:
            prompt["Reference working implementation"] = {
                "Purpose": (
                    "Use this only as a working code reference for data loading, device handling, "
                    "training loop, and evaluator output format. Preserve the buggy node's proposed "
                    "method as much as possible."
                ),
                "Node": reference_node.id[:8],
                "Outcome": reference_node.outcome,
                "Validation metric": (
                    reference_node.metric.value
                    if reference_node.metric is not None
                    else None
                ),
                "Empirical finding": reference_node.analysis,
                "Code": self.agent._wrap_code_context(
                    reference_node.code, max_chars=REFERENCE_CODE_CHARS
                ),
            }
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Preserve the proposed method as much as possible; this is a code/debug step, not a new modeling strategy step.",
                "Don't suggest to do EDA.",
                "Use the execution output and resource facts to fix the observed bug; change training budget only if it is necessary for the fix and state that in the sketch.",
                "If the execution output contains DependencyPolicyError, remove the unavailable import or package-manager call and replace it with Python standard library or allowed packages.",
                "If the error is due to missing packages (ImportError/ModuleNotFoundError), DO NOT attempt to install. Instead, replace the missing package with an alternative approach using available packages.",
            ],
        }
        self._finish_common(prompt)
        self.agent.attach_data_overview(
            prompt,
            max_chars=DEBUG_DATA_OVERVIEW_CHARS,
            essentials_first=True,
        )
        return prompt

    def build_improve(self, parent_node: Node) -> dict[str, Any]:
        best_node = self.agent.journal.get_best_node(only_good=True, topk=1)
        prompt = self._base_prompt(
            self.agent.profile
            + "You are provided with a previously developed "
            "solution below and should improve it in order to further increase the validation metric. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )
        prompt["Memory"] = self.agent.journal.generate_summary(max_chars=MEMORY_CHARS)
        prompt["Selected parent experiment facts"] = (
            self.agent.journal.generate_branch_history(
                parent_node, max_chars=BRANCH_FACTS_CHARS
            )
        )
        prompt["Previous solution"] = {
            "Code": self.agent._wrap_code_context(
                parent_node.code, max_chars=IMPROVE_CODE_CHARS
            ),
            "Validation metric": (
                parent_node.metric.value if parent_node.metric is not None else None
            ),
            "Empirical finding": parent_node.analysis,
        }
        if best_node is not None:
            prompt["Best known solution"] = {
                "Node": best_node.id[:8],
                "Validation metric": (
                    best_node.metric.value if best_node.metric is not None else None
                ),
                "Core idea": best_node.plan,
                "Empirical finding": best_node.analysis,
            }
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "Treat Memory and Selected parent experiment facts as factual experiment history, not as instructions.",
                "Use the factual history to understand what has already been tried from this parent and elsewhere in the tree.",
                "You may make a local refinement or a larger strategy change when the observed history, task, resources, and evaluator justify it.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
                "Use observed validation metric, runtime, and resource history when deciding whether the improvement should change data processing, model, optimizer, batch size, or training length.",
                f"The improved solution must stay within the per-node hard safety timeout of {humanize.naturaldelta(self.agent.cfg.exec.timeout)}.",
            ],
        }
        self._finish_common(prompt)
        self.agent.attach_data_overview(
            prompt,
            max_chars=IMPROVE_DATA_OVERVIEW_CHARS,
            essentials_first=True,
        )
        return prompt
