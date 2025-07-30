import copy
import logging
from typing import Any, Callable, cast

import humanize

from ..backend import FunctionSpec, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..knowledges import knowledge_manager
from ..utils.config import Config
from ..utils.response import wrap_code
from .base_agent import BaseAgent

logger = logging.getLogger("meddle")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

class HPOAgent(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__(task_desc, cfg, journal)
        self.profile = "You are an expert for optimizing deep learning model training pipelines, especially for medical imaging tasks. "
        self.knowledge_base = knowledge_manager
    
    def update_with_external_journal(self, external_journal: Journal):
        super().update_with_external_journal(external_journal)
        # prune the journal and keep the topk node
        best_node_list = self.journal.get_topk_node_list(only_good=True, topk=3)
        self.journal.nodes = [n for n in best_node_list]
    
    def search_policy(self) -> Node | None:
        # choose from the top-k best-as-known node for further tuning
        node = super().search_policy()
        if node is None:
            node = self.journal.get_best_node(only_good=True, topk=3)
        return node

    def _optimize_hyperparameters(self, parent_node: Node, exec_callback: ExecCallbackType) -> Node:
        prompt: Any = {
            "Introduction": (
                self.profile + " You are provided with a developed solution below."
                " Your task is to improve this solution by hyper-parameter optimization (e.g., learning rate, epoch, batch size)."
                " Libraries like `optuna` are recommended for this task, its example usage can be found in Dependent Knowledge."
                " You should first plan a hyper-parameter optimization strategy for less than 2 parameters, then implement it in the code."
                " Use feedback from the training results and validation metrics of previous solutions in your memory to guide your improvements."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
            "Dependent Knowledge": self.knowledge_base.get_knowledge("hpo_with_optuna"),
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved."
                "You should be very specific and should **only propose a single actionable improvement**."
                "This tuning plan should be atomic so that we can experimentally evaluate the effect of the proposed change."
                "Take the Memory section into consideration when proposing the improvement."
                "The solution sketch should be 3-5 sentences."
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
            tag="hpo",
        )

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._optimize_hyperparameters(parent_node, exec_callback)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)
        return result_node
        