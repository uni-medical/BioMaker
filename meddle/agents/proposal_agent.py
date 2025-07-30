import logging
from typing import Any, Callable, cast

import humanize

from ..backend import FunctionSpec, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..monai_rag.query_rag_db import MED_KNOWLEDGE_RAG
from ..utils.config import Config
from ..utils.response import wrap_code
from .base_agent import BaseAgent

logger = logging.getLogger("meddle")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

class ProposalAgent(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__(task_desc, cfg, journal)
        self.profile = "You are a medical deep learning grandmaster attending a competition. "
        self.knowledge_base = MED_KNOWLEDGE_RAG(model=self.acfg.knowledge_base.model)

    def query_knowledge_base(self, query: str, topk: int=3) -> str: 
        result = ""
        if self.acfg.enable_monai_knowledge_base:
            logger.trace(f"Querying knowledge base with query: {query}")
            result = self.knowledge_base.search(query, k=topk, enable_query2doc=self.acfg.enable_query2doc)
            logger.debug(f"Querying knowledge base get result:\n {result[:100]}")
        return result

    def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                self.profile +
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
                "Check the data preview of `./input` directory carefully, and make sure the correct Dataset and DataLoader is created.",
                "This first soultion should **only run 1 training epoch** to help validate the code quickly.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        # if self.acfg.enable_monai_knowledge_base:
        #     prompt["Dependent Knowledge"] = self.query_knowledge_base("The basic knowledge of MONAI Data Pipeline usage")

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code)

    def _debug(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                self.profile + 
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }

        if self.acfg.enable_monai_knowledge_base:
            knowledge_retrieve_context = prompt.copy()
            knowledge_retrieve_context["Query"] = "What topic should be queried to debug this code?"
            topic_of_knowledge_for_debug = query(
                system_message=knowledge_retrieve_context,
                user_message=None,
                model=self.acfg.knowledge_base.model,
                temperature=self.acfg.knowledge_base.temp,
            )
            prompt["Dependent Knowledge"] = self.query_knowledge_base(topic_of_knowledge_for_debug)

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
                "The debugged soultion should **only run 1 training epoch** to help validate the code quickly.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def _improve(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                self.profile + "You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "You should pay more attention on data preparation/augmentation and model architecture."
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        if self.acfg.enable_monai_knowledge_base:
            knowledge_retrieve_context = prompt.copy()
            knowledge_retrieve_context["Query"] = "What topic should be queried to improve this code?"
            topic_of_knowledge_for_debug = query(
                system_message=knowledge_retrieve_context,
                user_message=None,
                model=self.acfg.knowledge_base.model,
                temperature=self.acfg.knowledge_base.temp,
            )
            prompt["Dependent Knowledge"] = self.query_knowledge_base(topic_of_knowledge_for_debug)

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
                f"The improved soultion should **run suitable training epochs (no more than 10)** to help get best performance **within {humanize.naturaldelta(self.cfg.exec.timeout)}**.",
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

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)
        return result_node
