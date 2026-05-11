import logging
from dataclasses import dataclass
from typing import Any, Callable

from .agents import BaseAgent, ProposalAgent
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils.config import Config

logger = logging.getLogger("biomaker")
ExecCallbackType = Callable[[str, bool], ExecutionResult]


@dataclass
class WorkflowAgent:
    """Status of an agent."""

    name: str
    steps_left: int
    agent: BaseAgent

    def __post_init__(self):
        self.name = self.name.lower()
        if not isinstance(self.steps_left, int):
            raise TypeError("steps_left must be an integer")
        if self.steps_left < 0:
            raise ValueError("steps_left must be >= 0")


@dataclass
class RoundNode:
    workflow_agent: WorkflowAgent
    node: Node
    round_index: int
    round_item_index: int
    round_item_count: int


class BioMakerWorkflow:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        interpreter: Any = None,  # type: ignore
    ):
        self.task_desc = task_desc
        self.journal = journal
        self.interpreter = interpreter
        self.cfg = cfg

        self.proposal_agent = ProposalAgent(
            task_desc=task_desc, cfg=cfg, journal=Journal()
        )
        self.workflow = [
            WorkflowAgent(
                name="proposal",
                steps_left=cfg.agent.steps,
                agent=self.proposal_agent,
            ),
        ]

    @property
    def workflow_status(self) -> str:
        status = [f"{wa.name}-steps:{wa.steps_left}" for wa in self.workflow]
        return " | ".join(status)

    def update_steps_left(self) -> int:
        """Return the number of steps left in the workflow."""
        return sum(node.steps_left for node in self.workflow)

    def step(self, exec_callback: ExecCallbackType):
        """Step the agent forward by one step."""
        # walkthrough the workflow and run the first node that has steps left
        for idx, workflow_agent in enumerate(self.workflow):
            if workflow_agent.steps_left > 0:
                if workflow_agent.agent.has_empty_journal() and idx > 0:
                    logger.info(
                        "Agent '%s' has no memory, getting initial memory from global memory.",
                        workflow_agent.name,
                    )
                    workflow_agent.agent.update_with_external_journal(self.journal)

                logger.info(
                    "Current Agent: '%s', steps left: %d",
                    workflow_agent.name,
                    workflow_agent.steps_left,
                )
                solution_node = workflow_agent.agent.step(exec_callback)
                solution_node.agent_name = workflow_agent.name
                workflow_agent.steps_left -= 1
                self.journal.append(solution_node)  # update global journal
                return solution_node
        return None

    def parse_round_result(
        self,
        round_node: RoundNode,
        exec_result: ExecutionResult,
        best_node_before: Node | None = None,
    ) -> None:
        round_node.workflow_agent.agent.parse_exec_result(
            round_node.node,
            exec_result,
            best_node_before=best_node_before,
        )

    def append_round_node(self, round_node: RoundNode) -> Node:
        """Append one parsed round node in deterministic order."""
        node = round_node.node
        if node.parent is not None:
            node.parent.children.add(node)
        round_node.workflow_agent.agent.journal.append(node)
        round_node.workflow_agent.steps_left -= 1
        self.journal.append(node)
        return node
