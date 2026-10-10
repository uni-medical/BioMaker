"""Malformed provider output consumes a slot without executing empty code."""
import pytest

import biomaker.run as run
from biomaker.agents.base_agent import CandidateGenerationError
from biomaker.utils.final_selection import freeze_selection


def test_exhausted_generation_is_saved_once_and_never_executed(tmp_path, monkeypatch):
    from test_presentation_failure_lifecycle import _install_minimal_run, _GenerationProposalAgent, _SuccessfulInterpreter
    cfg, events, order, _ = _install_minimal_run(monkeypatch, tmp_path, retrain_ok=True, break_on_stop=False)
    saved = []
    monkeypatch.setattr(run, 'freeze_selection', freeze_selection)
    monkeypatch.setattr(run, 'save_run', lambda _cfg, journal, *a, **kw: saved.append(journal))
    monkeypatch.setattr(_SuccessfulInterpreter, 'run', lambda *a, **kw: pytest.fail('empty code must not execute'))
    def malformed(self, task):
        raise CandidateGenerationError(reason='missing_python', attempts=3, response_chars=24, prompt_capture={'prompt': 'bounded prompt', 'messages': [{'role': 'system', 'content': 'bounded message'}], 'stats': {'output_tokens': 7}})
    monkeypatch.setattr(_GenerationProposalAgent, 'generate_node', malformed)
    with pytest.raises(RuntimeError, match='No valid scored candidate'):
        run._run_direct_entry()
    assert len(saved[-1].nodes) == 1
    node = saved[-1].nodes[0]
    assert node.code == '' and node.outcome == 'generation_format_failed'
    assert node.prompt_messages[0]['content'] == 'bounded message' and node.prompt_stats['output_tokens'] == 7
    assert node.execution_workspace is None and node.exc_info['execution_skipped'] is True
    assert node.scheduled_step == 0 and node.step == 0
    assert [name for name, _ in events].count('node_generation_failed') == 1
    assert not any(name == 'execution_started' for name, _ in events)
    assert 'final_retrain' not in order
    assert events[-1][0] == 'run_failed'
