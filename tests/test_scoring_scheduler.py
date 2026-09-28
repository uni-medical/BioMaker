"""Exercise real scheduler threads without launching models or API requests."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace

from biomaker.interpreter import ExecutionResult
from biomaker.utils.compute import ComputePool, ComputeSlot
import biomaker.run as runner
from test_presentation_failure_lifecycle import _install_minimal_run, _MinimalWorkflow


def test_scoring_releases_gpu_and_bounds_backlog_before_commit(tmp_path, monkeypatch):
    cfg, events, order, _ = _install_minimal_run(
        monkeypatch, tmp_path, retrain_ok=True, break_on_stop=False)
    cfg.agent.steps = 4
    second_execution = Event()
    third_execution = Event()
    lock = Lock()
    executions, scores, commits = [], [], []
    pool = ComputePool((ComputeSlot(index=0, backend='cuda', selector='7'),))

    class Workflow(_MinimalWorkflow):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.workflow[0].steps_left = 4
            self.proposal_agent.select_round_tasks = lambda: [
                runner.RoundTask(parent=None, selection_reason=str(i)) for i in range(4)]

        def update_steps_left(self):
            return 4

        def parse_round_result(self, node, result, **kwargs):
            assert len(scores) > len(commits), 'unscored nodes must not be ranked'
            commits.append(node.node.id)
            super().parse_round_result(node, result, **kwargs)

    class Interpreter:
        def __init__(self, workspace, environment=None, **kwargs):
            if environment is not None:
                assert environment['CUDA_VISIBLE_DEVICES'] == '7'

        def run(self, *args, **kwargs):
            with lock:
                executions.append(1)
                if len(executions) == 2:
                    second_execution.set()
                if len(executions) == 3:
                    third_execution.set()
            return ExecutionResult(['trained'], .01, None)

    def score(workspace, env_overrides, **kwargs):
        assert env_overrides['CUDA_VISIBLE_DEVICES'] == ''
        if not scores:
            assert second_execution.wait(5), 'GPU must run next candidate during scoring'
            assert not third_execution.wait(.15), 'backlog must be bounded'
            assert commits == [], 'scoring has not finished yet'
        scores.append(1)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(runner, 'BioMakerWorkflow', Workflow)
    monkeypatch.setattr(runner, 'Interpreter', Interpreter)
    monkeypatch.setattr(runner, 'score_candidate_with_evaluator', score)
    monkeypatch.setattr(runner, 'ThreadPoolExecutor', ThreadPoolExecutor)
    monkeypatch.setattr(runner.ComputePool, 'detect', staticmethod(lambda _: pool))
    monkeypatch.setattr(runner, 'SHARED_CONSOLE', SimpleNamespace(is_terminal=False))
    runner._run_direct_entry()
    assert len(executions) == len(scores) == len(commits) == 4
    assert events[-1] == ('run_finished', {'completed_steps': 4})
    assert len([e for e, _ in events if e == 'scoring_finished']) == 4
    assert 'final_retrain' in order
