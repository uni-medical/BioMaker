from .agents import BaseAgent, ProposalAgent
from .journal import Journal, Node
from .utils.config import Config
from .interpreter import ExecutionResult
from typing import Any, Callable
import logging

logger = logging.getLogger("meddle")
ExecCallbackType = Callable[[str, bool], ExecutionResult]

class MeddleAgentSys(BaseAgent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__(task_desc, cfg, journal)
        self.proposal_agent = ProposalAgent(task_desc=task_desc, cfg=cfg, journal=Journal())
    
    def step(self, exec_callback: ExecCallbackType) -> None:
        """Step the agent forward by one step."""
        proposal_node = self.proposal_agent.step(exec_callback=exec_callback)
        
        self.journal.append(proposal_node)

