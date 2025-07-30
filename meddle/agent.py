import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agents import (BaseAgent, FunctionalAgent, HPOAgent, ProposalAgent,
                     TuningAgent)
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
        interpreter: Any = None,  # type: ignore
    ):
        super().__init__(task_desc, cfg, journal)
        self.interpreter = interpreter
        self.cfg = cfg

        # initialize all agents needed for the workflow        
        self.proposal_agent = ProposalAgent(task_desc=task_desc, cfg=cfg, journal=Journal())
        self.func_agent = FunctionalAgent(task_desc=task_desc, cfg=cfg, journal=Journal(), solution_file=self.cfg.proposed_solution_file)
        self.tuning_agent = TuningAgent(task_desc=task_desc, cfg=cfg, journal=Journal())
        self.hpo_agent = HPOAgent(task_desc=task_desc, cfg=cfg, journal=Journal())
        # create the squenetial workflow of agents
        self.workflow = [
            WorkflowAgent(name="func_val", steps_left=1 if (self.cfg.proposed_solution_file is not None) else 0, agent=self.func_agent),
            WorkflowAgent(name="proposal", steps_left=cfg.agent.step_plan.proposal, agent=self.proposal_agent),
            WorkflowAgent(name="tuning", steps_left=cfg.agent.step_plan.tuning, agent=self.tuning_agent),
            WorkflowAgent(name="hpo", steps_left=cfg.agent.step_plan.hpo, agent=self.hpo_agent),
        ]
    
    @property
    def step_plan(self) -> int:
        step_plan = [f"{wa.name}-step:{wa.steps_left}" for wa in self.workflow]
        return " | ".join(step_plan)

    def update_steps_left(self) -> int:
        """Return the number of steps left in the workflow."""
        return sum(node.steps_left for node in self.workflow)

    def step(self, exec_callback: ExecCallbackType) -> None:
        """Step the agent forward by one step."""
        # walkthrough the workflow and run the first node that has steps left
        for idx, workflow_agent in enumerate(self.workflow):
            if workflow_agent.name == "hpo":
                self.interpreter.timeout = self.cfg.exec.timeout * 2 
                logger.info("Setting interpreter timeout to %d seconds for HPO agent.", self.interpreter.timeout)
            if workflow_agent.steps_left > 0:
                if workflow_agent.agent.has_empty_journal() and idx > 0:
                    logger.info("Agent '%s' has no memory, getting initial memory from global memory.", workflow_agent.name)
                    workflow_agent.agent.update_with_external_journal(self.journal)

                logger.info("Current Agent: '%s', steps left: %d", workflow_agent.name, workflow_agent.steps_left)
                solution_node = workflow_agent.agent.step(exec_callback)
                solution_node.agent_name = workflow_agent.name
                workflow_agent.steps_left -= 1
                self.journal.append(solution_node) # update global journal
                break

