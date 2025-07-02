import logging
from typing import Any, Callable, cast

import humanize
from ..backend import FunctionSpec, query
from ..interpreter import ExecutionResult
from ..journal import Journal, Node
from ..utils.config import Config
from ..utils.response import wrap_code
from ..utils.metric import ZeroMetricValue
from .base_agent import BaseAgent

logger = logging.getLogger("meddle")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

class FunctionalAgent(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        solution_file: str | None = None,
    ):
        super().__init__(task_desc, cfg, journal)
        self.profile = "You are an expert for optimizing training code for maximum performance. "
        self.solution_file = solution_file

    def search_policy(self) -> Node | None:
        # always choose the best-as-known node for further tuning
        best_node = self.journal.get_best_node(only_good=True)
        if best_node is None:
            best_node = self.journal.get_best_node(only_good=False)
        logger.debug("[search policy] best-as-known node selected")
        return best_node

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
        ]
        impl_guideline.append(
            "The implementation should include a predict() function, "
            "allowing users to seamlessly reuse the code to make predictions on new data. "
            "The prediction function should be well-documented, especially the function signature."
        )

        return {"Implementation guideline": impl_guideline}

    def _functional_fix(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                self.profile + "Below, you will find a previously developed solution. "
                "Your task is to review the code and make minor modifications if necessary to ensure that it meets the following requirements:\n\n"
                "- The number of epochs is set to 1, which should allow the code to run quickly.\n"
                "- The code includes the functionality to generate predictions on the test data and save them to `working/submission.csv`.\n\n"
                "Please verify these points and make any necessary adjustments to the code."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        # code = self._tuning_batch_size_in_code(code, exec_callback)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _validate_existed_solution_code(self) -> bool:
        code = open(self.solution_file, "r").read()
        solution_node = Node(
            plan="Try to validate this existed solution code to check whether it is suitable for this task.", 
            code=code
        )
        solution_node.metric = ZeroMetricValue()
        return solution_node

    def step(self, exec_callback: ExecCallbackType):
        # if a solution file is provided, validate it
        if self.solution_file is not None:
            solution_node = self._validate_existed_solution_code()
            self.solution_file = None
        else:
            if not self.journal.nodes or self.data_preview is None:
                self.update_data_preview()
            parent_node = self.search_policy()
            logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")
            solution_node = self._functional_fix(parent_node)

        self.parse_exec_result(
            node=solution_node,
            exec_result=exec_callback(solution_node.code, True),
        )
        self.journal.append(solution_node)
        return solution_node
        