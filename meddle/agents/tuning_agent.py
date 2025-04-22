import logging
from typing import Any, Callable, cast

import humanize
from ..backend import FunctionSpec, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..utils.config import Config
from ..utils.response import wrap_code
from ..monai_rag.query_rag_db import MONAI_RAG
from .base_agent import BaseAgent

logger = logging.getLogger("meddle-tuning")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

# hardware_func_spec = FunctionSpec(
#     name="submit_review",
#     json_schema={
#         "type": "object",
#         "properties": {
#             "model_gpu_ua": {
#                 "type": "boolean",
#                 "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
#             },
#             "code": {
#                 "type": "string",
#                 "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.",
#             },
#         },
#         "required": ["is_bug", "summary", "metric", "lower_is_better"],
#     },
#     description="Submit a hardware evaluating the output of the training script.",
# )

class TuningAgent(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__(task_desc, cfg, journal)
        self.profile = "You are an expert for optimizing training code for maximum performance. "
        self.knowledge_base = MONAI_RAG(model=self.acfg.knowledge_base.model)

    def query_knowledge_base(self, query: str, topk: int=3) -> str: 
        result = ""
        if self.acfg.enable_monai_knowledge_base:
            try:
                result = self.knowledge_base.search(query, k=topk, enable_query2doc=self.acfg.enable_query2doc)
            except:
                print("Retrieve Failed! Plz check the knowledge base status")
        return result

    def search_policy(self) -> Node | None:
        # always choose the best-as-known node for further tuning
        best_node = self.journal.get_best_node(only_good=True)
        if best_node is None:
            best_node = self.journal.get_best_node(only_good=False)
        logger.debug("[search policy] best-as-known node selected")
        return best_node

    # def _tuning_batch_size_in_code(self, code, exec_callback: ExecCallbackType):
    #     prompt = {
    #         "Introduction": (
    #             self.profile + "You are provided with a previously developed solution below."
    #             "Your task is to generate a debug version of the code which only runs 10 iterations then break."
    #             "Notice that the generated code should report the GPU memory usage of model and data and then report current batch size of the code."
    #         ),
    #         "Previous Solution": wrap_code(code),
    #     }

    #     response = cast(
    #         dict,
    #         query(
    #             system_message=prompt,
    #             user_message=None,
    #             func_spec=hardware_func_spec,
    #             model=self.acfg.feedback.model,
    #             temperature=self.acfg.feedback.temp,
    #         ),
    #     )
    #     # get gpu memory usage & current batch size of the code
    #     # TODO

    #     # get available gpu memory of the machine
    #     # TODO
        
    #     # compute 
    #     # TODO
    #     return code

    def _tuning(self, parent_node: Node, exec_callback: ExecCallbackType) -> Node:
        prompt: Any = {
            "Introduction": (
                self.profile + "You are provided with a previously developed solution below."
                "Your task is to improve the provided solution by refining data augmentation (transforms) and training hyperparameters. "
                "Focus exclusively on these aspects, ignoring data preparation or model architecture changes. "
                "Use feedback from training results and validation metric of previous solutions in your memory to guide your improvements. "
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
            knowledge_retrieve_context["Query"] = "What topic should be queried to improve this code by refining data augmentation (transforms) and training hyperparameters?"
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
                "The solution sketch should be a brief natural language description of how the previous solution can be improved."
                "You should be very specific and should only propose a single actionable improvement."
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change."
                "Take the Memory section into consideration when proposing the improvement."
                "The solution sketch should be 3-5 sentences."
                "Don't suggest to do EDA."
                "If the epoch number is 1, change it to a suitable epoch number according to the previous execution time in Memory."
                f"The improved soultion should run **suitable epoch numbers within {humanize.naturaldelta(self.cfg.exec.timeout)}**."
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        # code = self._tuning_batch_size_in_code(code, exec_callback)
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
        