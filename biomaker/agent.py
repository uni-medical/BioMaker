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
        assert isinstance(self.steps_left, int), "steps_left must be an integer"
        assert self.steps_left >= 0, "steps_left must be >= 0"


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

    def generate_round_nodes(
        self,
        max_nodes: int,
        round_index: int = 0,
        on_task_start: Callable | None = None,
        initial_pending_nodes: list[Node] | None = None,
    ) -> list[RoundNode]:
        return list(
            self.iter_round_nodes(
                max_nodes=max_nodes,
                round_index=round_index,
                on_task_start=on_task_start,
                initial_pending_nodes=initial_pending_nodes,
            )
        )

    def iter_round_nodes(
        self,
        max_nodes: int,
        round_index: int = 0,
        on_task_start: Callable | None = None,
        initial_pending_nodes: list[Node] | None = None,
    ):
        """Generate plan/code for one scheduler round without executing nodes."""
        remaining = max_nodes
        yielded = 0
        for idx, workflow_agent in enumerate(self.workflow):
            if remaining <= 0:
                break
            if workflow_agent.steps_left <= 0:
                continue
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
            tasks = workflow_agent.agent.select_round_tasks()
            if not tasks:
                continue
            tasks = tasks[: min(remaining, workflow_agent.steps_left)]
            round_item_count = len(tasks)
            pending_nodes: list[Node] = list(initial_pending_nodes or [])
            if isinstance(workflow_agent.agent, ProposalAgent):
                workflow_agent.agent.clear_pending_round_nodes()
            try:
                for task in tasks:
                    if not isinstance(workflow_agent.agent, ProposalAgent):
                        raise TypeError(
                            "Async round generation currently requires ProposalAgent."
                        )
                    item_index = yielded + 1
                    placeholder_node = None
                    if on_task_start is not None:
                        placeholder_node = on_task_start(
                            workflow_agent,
                            task,
                            round_index,
                            item_index,
                            round_item_count,
                        )
                    workflow_agent.agent.set_pending_round_nodes(pending_nodes)
                    node = workflow_agent.agent.generate_node(task)
                    node.agent_name = workflow_agent.name
                    if node.parent is not None:
                        node.parent.children.discard(node)
                    if placeholder_node is not None:
                        node.id = placeholder_node.id
                        node.ctime = placeholder_node.ctime
                    pending_nodes.append(node)
                    yielded += 1
                    yield RoundNode(
                        workflow_agent=workflow_agent,
                        node=node,
                        round_index=round_index,
                        round_item_index=item_index,
                        round_item_count=round_item_count,
                    )
                    remaining -= 1
                    if remaining <= 0:
                        break
            finally:
                if isinstance(workflow_agent.agent, ProposalAgent):
                    workflow_agent.agent.clear_pending_round_nodes()

    def parse_round_results(
        self,
        round_nodes: list[RoundNode],
        exec_results: list[ExecutionResult],
    ) -> None:
        """Parse execution results against the pre-round journal snapshot."""
        for round_node, exec_result in zip(round_nodes, exec_results, strict=True):
            round_node.workflow_agent.agent.parse_exec_result(
                round_node.node, exec_result
            )

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
