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
            WorkflowAgent(name="proposal", steps_left=cfg.agent.steps - cfg.agent.steps_for_tuning, agent=self.proposal_agent),
            WorkflowAgent(name="proposal_func_val", steps_left=func_val_steps, agent=self.func_agent),
            WorkflowAgent(name="tuning", steps_left=cfg.agent.steps_for_tuning, agent=self.tuning_agent),
        ]
    
    def recompute_steps_left(self) -> int:
        """Return the number of steps left in the workflow."""
        return sum(node.steps_left for node in self.workflow)

    def step(self, exec_callback: ExecCallbackType) -> None:
        """Step the agent forward by one step."""
        # walkthrough the workflow and run the first node that has steps left
        for idx, workflow_agent in enumerate(self.workflow):
            if workflow_agent.steps_left > 0:
                logger.info("Current agent: %s, steps left: %d", workflow_agent.name, workflow_agent.steps_left)
                solution_node = workflow_agent.agent.step(exec_callback)
                solution_node.agent_name = workflow_agent.name
                workflow_agent.steps_left -= 1
                
                # update global journal
                self.journal.append(solution_node)

                if workflow_agent.steps_left == 0 and idx < len(self.workflow)-1:
                    # for workflow node switch, update the journal for the next agent
                    logger.info("Passing journal (%d nodes) to the next agent: %s", len(self.journal), self.workflow[idx + 1].name)
                    self.workflow[idx + 1].agent.update_with_external_journal(self.journal)
                break
