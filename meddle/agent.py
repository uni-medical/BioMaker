import logging
from dataclasses import dataclass
from .agents import BaseAgent, ProposalAgent, FunctionalAgent, TuningAgent
from .journal import Journal, Node
from .utils.metric import ZeroMetricValue
from .utils.config import Config
from .interpreter import ExecutionResult
from typing import Any, Callable
from pathlib import Path

logger = logging.getLogger("meddle")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

@dataclass
class StepsLeft:
    proposal: int
    tuning: int

    def __init__(self, global_steps: int, steps_for_tuning: int):
        self.proposal = global_steps - steps_for_tuning
        self.tuning = steps_for_tuning

class MeddleAgentSys(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__(task_desc, cfg, journal)
        
        # check for the number of steps
        self.steps_left = StepsLeft(global_steps=self.acfg.steps, steps_for_tuning=self.acfg.steps_for_tuning)
        assert self.steps_left.proposal >= 0, "agent.steps can not be smaller than agent.steps_for_tuning"
    
        self.proposal_agent = ProposalAgent(task_desc=task_desc, cfg=cfg, journal=Journal())
        self.functional_agent = FunctionalAgent(task_desc=task_desc, cfg=cfg, journal=Journal())
        self.tuning_agent = TuningAgent(task_desc=task_desc, cfg=cfg, journal=Journal())
        
        self.solution_node_to_be_exec = []
        if self.cfg.proposed_solution_file is not None:
            if self.steps_left.proposal > 0:
                raise NotImplementedError("Proposed solution file is not supported for proposal agent for now.")
            proposed_solution_node = self._check_and_create_solution_node(self.cfg.proposed_solution_file)
            self.solution_node_to_be_exec.append(proposed_solution_node)

    def _check_and_create_solution_node(self, proposed_solution_file: Path) -> Node:
        # read the proposed solution file and create a node
        code = open(proposed_solution_file, "r").read()
        solution_node = Node(plan="To solve this task, we start from this proposed code.", code=code)
        solution_node.metric = ZeroMetricValue()
        return solution_node

    def step(self, exec_callback: ExecCallbackType) -> None:
        """Step the agent forward by one step."""
        if len(self.solution_node_to_be_exec) > 0:
            solution_node = self.solution_node_to_be_exec.pop()
            ''' use functional agent to check and update the solution code '''
            self.functional_agent.journal.append(solution_node)
            refined_solution_node = self.functional_agent.step(exec_callback=exec_callback)
            # if the refined version is buggy, we return to original solution
            if refined_solution_node.is_buggy:
                self.parse_exec_result(
                    node=solution_node,
                    exec_result=exec_callback(solution_node.code, True),
                )
                refined_solution_node = solution_node
            refined_solution_node.parent = None
            self.tuning_agent.journal.append(refined_solution_node)
            self.journal.append(refined_solution_node)
        elif self.steps_left.proposal > 0:
            self.steps_left.proposal -= 1
            proposal_node = self.proposal_agent.step(exec_callback=exec_callback)
            self.journal.append(proposal_node)
        elif self.steps_left.tuning > 0:
            self.steps_left.tuning -= 1
            tuning_node = self.tuning_agent.step(exec_callback=exec_callback)
            self.journal.append(tuning_node)
        else:
            pass
