import json
from dataclasses import dataclass

from omegaconf import OmegaConf
from rich.status import Status

from .agent import BioMakerWorkflow
from .interpreter import Interpreter
from .journal import Journal
from .utils.config import (
    _load_cfg,
    load_task_desc,
    prep_agent_workspace,
    prep_cfg,
    save_run,
)
from .utils.evaluator_setup import (
    clean_candidate_outputs,
    prepare_evaluator,
    score_candidate_with_evaluator,
)
from .utils.artifact_validation import (
    normalize_working_outputs,
    validate_working_outputs,
)
from .utils.private_evaluation import evaluate_private_outputs
from .utils.split_protocol import prepare_split_protocol


@dataclass
class Solution:
    code: str
    valid_metric: float


class Experiment:

    def __init__(self, data_dir: str, goal: str, eval: str | None = None):
        """Initialize a new experiment run.

        Args:
            data_dir (str): Path to the directory containing the data files.
            goal (str): Description of the goal of the task.
            eval (str | None, optional): Optional description of the preferred way for the agent to evaluate its solutions.
        """

        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval
        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(self.cfg)
        self.private_dir = self.cfg.log_dir / "private_truth"
        split_protocol = prepare_split_protocol(
            self.cfg.workspace_dir / "input", private_dir=self.private_dir
        )
        if isinstance(self.task_desc, dict):
            self.task_desc["Data split protocol"] = split_protocol.prompt_text
        else:
            self.task_desc = {
                "Task description": self.task_desc,
                "Data split protocol": split_protocol.prompt_text,
            }

        prepared_evaluator = prepare_evaluator(
            workspace_dir=self.cfg.workspace_dir,
            log_dir=self.cfg.log_dir,
            task_desc=self.task_desc,
            model=self.cfg.agent.feedback.model,
            temperature=0.0,
            timeout=min(180, self.cfg.exec.timeout),
        )
        self.task_desc["Evaluation protocol"] = (
            "A frozen evaluator has been generated and smoke-tested at `./evaluator.py`. "
            "Strictly follow `evaluator.prediction_spec()` for validation predictions and test submission. "
            "BioMaker computes the final node score after the script exits via no-argument `evaluator.score_val()`. "
            "Key columns in prediction artifacts must exactly match the split CSV values. "
            "If a DataLoader returns key values as tensors, convert them to ordinary Python scalars with `.item()` or read the key column directly from the dataframe; never write values such as `tensor(1)`. "
            "Classification probability columns must be complete softmax probabilities for every class and each row must sum to 1; do not fill missing rows with zeros. "
            f"Prediction spec: {prepared_evaluator.prediction_spec}"
        )

        self.journal = Journal()
        self.interpreter = Interpreter(
            self.cfg.workspace_dir, **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )
        self.agent = BioMakerWorkflow(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
            interpreter=self.interpreter,
        )

    def _exec_callback(self, *args, **kwargs):
        clean_candidate_outputs(self.cfg.workspace_dir)
        res = self.interpreter.run(*args, **kwargs)
        if res.exc_type is None:
            normalization_messages = normalize_working_outputs(
                self.cfg.workspace_dir, str(self.task_desc)
            )
            if normalization_messages:
                res.term_out.append("\n" + "\n".join(normalization_messages))
            artifact_check = validate_working_outputs(
                self.cfg.workspace_dir, str(self.task_desc)
            )
            if artifact_check.messages:
                res.term_out.append("\n" + "\n".join(artifact_check.messages))
            if not artifact_check.ok:
                res.exc_type = "ArtifactValidationError"
                res.exc_info = {"reason": "generated artifacts failed validation"}
        if res.exc_type is None:
            evaluator_result = score_candidate_with_evaluator(
                self.cfg.workspace_dir, timeout=min(180, self.cfg.exec.timeout)
            )
            if evaluator_result.output:
                res.term_out.append("\n" + evaluator_result.output)
            if evaluator_result.ok:
                res.term_out.append("\n" + evaluator_result.log_line())
            else:
                res.exc_type = "EvaluatorValidationError"
                res.exc_info = {"reason": evaluator_result.error}
                res.term_out.append(
                    "\nEvaluatorValidationError: "
                    + (evaluator_result.error or "external evaluator failed")
                )
        return res

    def run(self, steps: int) -> Solution:
        for _i in range(steps):
            solution_node = self.agent.step(exec_callback=self._exec_callback)
            best_node = self.journal.get_best_node(only_good=True)
            if solution_node is not None and solution_node is best_node:
                try:
                    private_metrics = evaluate_private_outputs(
                        self.cfg.workspace_dir, self.private_dir
                    )
                except Exception:
                    private_metrics = None
                if private_metrics:
                    self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
                    (
                        self.cfg.log_dir
                        / f"private_metrics_step_{solution_node.step}.json"
                    ).write_text(
                        json.dumps(private_metrics, indent=2), encoding="utf-8"
                    )
                    (self.cfg.log_dir / "private_metrics_latest.json").write_text(
                        json.dumps(private_metrics, indent=2), encoding="utf-8"
                    )
            save_run(self.cfg, self.journal)
        self.interpreter.cleanup_session()

        best_node = self.journal.get_best_node(only_good=True)
        if best_node is None:
            raise RuntimeError("No runnable solution was generated.")
        return Solution(code=best_node.code, valid_metric=best_node.metric.value)
