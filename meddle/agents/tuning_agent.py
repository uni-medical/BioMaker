import copy
import logging
from typing import Any, Callable, cast

import humanize

from ..backend import FunctionSpec, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..monai_rag.query_rag_db import MONAI_RAG
from ..utils.config import Config
from ..utils.response import wrap_code
from .base_agent import BaseAgent

logger = logging.getLogger("meddle-tuning")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

class TuningAgent(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__(task_desc, cfg, journal)
        self.profile = "You are an expert for optimizing deep learning model training pipelines, especially for medical imaging tasks. "
    
    def update_with_external_journal(self, external_journal: Journal):
        super().update_with_external_journal(external_journal)
        # prune the journal and keep only the last nodes to save memory
        self.journal.nodes = self.journal.nodes[-1:]
        for node in self.journal.nodes:
            node.parent = None
            node.children = set()
    
    def search_policy(self) -> Node | None:
        # always choose the best-as-known node for further tuning
        best_node = self.journal.get_best_node(only_good=True)
        if best_node is None:
            best_node = self.journal.get_best_node(only_good=False)
        return best_node

    def _tuning(self, parent_node: Node, exec_callback: ExecCallbackType) -> Node:
        prompt: Any = {
            "Introduction": (
                self.profile + " You are provided with a previously developed solution below."
                " Your task is to improve this solution by refining training codes (e.g., learning rate, optimizers, epoch, ...)."
                " 1. If the model has not used a pretrain, consider using a suitable one. Libraries like `timm` are recommended."
                " 2. If you want to tune hyperparameters, **only tune fewer than 3 hyperparameters at a time**. Libraries like `optuna` are recommended."
                " 3. If hyperparameter optimization shows minor improvement, you are encouraged to try other tricks you know."
                " Focus exclusively on these aspects, ignoring data preparation."
                " Use feedback from the training results and validation metrics of previous solutions in your memory to guide your improvements."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution performance tuning sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved."
                "You should be very specific and should only propose a single actionable improvement."
                "This tuning plan should be atomic so that we can experimentally evaluate the effect of the proposed change."
                "Take the Memory section into consideration when proposing the improvement."
                "The solution sketch should be 3-5 sentences."
                "Don't suggest to do EDA."
                "If the epoch number is 1, change it to a suitable epoch number according to the previous execution time in Memory."
                f"The improved soultion should run **suitable epoch numbers within {humanize.naturaldelta(self.cfg.exec.timeout)}**."
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        result_node = self._tuning(parent_node, exec_callback)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)
        return result_node
        