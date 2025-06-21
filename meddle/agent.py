import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agents import BaseAgent, FunctionalAgent, ProposalAgent, TuningAgent
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils.config import Config
from .utils.metric import ZeroMetricValue

logger = logging.getLogger("meddle")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

@dataclass 
class WorkflowAgent:
    """Status of an agent."""
    name: str
    steps_left: int
    agent: BaseAgent

    def __post_init__(self):
        self.name = self.name.lower()
        assert isinstance(self.steps_left, int), "steps_left must be an integer"
        assert self.steps_left >= 0, "steps_left must be >= 0"
    
    def step(self, exec_callback: ExecCallbackType) -> Node:
        """Run the agent for one step and return the solution node."""
        solution_node = self.agent.step(exec_callback)
        steps_left -= 1
        return solution_node


class MeddleWorkflow(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__(task_desc, cfg, journal)

        # initialize all agents needed for the workflow        
        self.proposal_agent = ProposalAgent(task_desc=task_desc, cfg=cfg, journal=Journal())
        self.func_agent = FunctionalAgent(task_desc=task_desc, cfg=cfg, journal=Journal(), solution_file=self.cfg.proposed_solution_file)
        self.tuning_agent = TuningAgent(task_desc=task_desc, cfg=cfg, journal=Journal())

        # create the squenetial workflow of agents
        func_val_steps = 1 if (cfg.agent.steps_for_tuning>0) else 0
        self.workflow = [
            WorkflowAgent("proposal", steps_left=cfg.agent.steps - cfg.agent.steps_for_tuning, agent=self.proposal_agent),
            WorkflowAgent("proposal_func_val", steps_left=func_val_steps, agent=self.func_agent),
            WorkflowAgent("tuning", steps_left=cfg.agent.steps_for_tuning, agent=self.tuning_agent),
        ]
    
    def recompute_steps_left(self) -> int:
        """Return the number of steps left in the workflow."""
        return sum(node.steps_left for node in self.workflow)

    def step(self, exec_callback: ExecCallbackType) -> None:
        """Step the agent forward by one step."""
        # walkthrough the workflow and run the first node that has steps left
        for idx, workflow_agent in enumerate(self.workflow):
            if workflow_agent.steps_left > 0:
                solution_node = workflow_agent.agent.step(exec_callback)
                self.journal.append(solution_node)

                if workflow_agent.steps_left == 0 and idx < len(self.workflow) - 1:
                    # for workflow node switch, update the journal for the next agent
                    self.workflow[idx + 1].agent.journal = self.journal
                break
