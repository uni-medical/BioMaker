"""Trusted scoring budgets are validated and separate from generated code."""

from types import SimpleNamespace

import pytest

from biomaker.run import _interpreter_kwargs
from biomaker.utils import config as config_module
from biomaker.utils import private_evaluation


@pytest.mark.parametrize("candidate_timeout", [1, 10800])
def test_scoring_defaults_do_not_depend_on_candidate_budget(candidate_timeout):
    cfg = SimpleNamespace(
        exec=SimpleNamespace(timeout=candidate_timeout, parallel_workers=1)
    )
    config_module._validate_exec_config(cfg)
    assert config_module.evaluator_timeout_seconds(cfg) == 3600
    assert config_module.evaluator_workers(cfg) == 1


@pytest.mark.parametrize("field", ["evaluator_timeout", "evaluator_workers"])
@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "180", None])
def test_scoring_config_rejects_invalid_budget_values(field, invalid):
    cfg = SimpleNamespace(exec=SimpleNamespace(timeout=60, parallel_workers=1))
    setattr(cfg.exec, field, invalid)
    with pytest.raises(ValueError, match=rf"exec\.{field} must be an integer >= 1"):
        config_module._validate_exec_config(cfg)


def test_scoring_config_is_not_forwarded_to_candidate_interpreter():
    cfg = config_module._load_cfg(use_cli_args=False)
    assert cfg.exec.evaluator_timeout == 3600
    assert cfg.exec.evaluator_workers == 1
    cfg.exec.timeout = 7
    cfg.exec.evaluator_timeout = 7200
    cfg.exec.evaluator_workers = 2
    config_module._validate_exec_config(cfg)
    assert config_module.evaluator_timeout_seconds(cfg) == 7200
    assert config_module.evaluator_workers(cfg) == 2
    candidate = _interpreter_kwargs(cfg)
    assert candidate["timeout"] == 7
    assert "evaluator_timeout" not in candidate
    assert "evaluator_workers" not in candidate


@pytest.mark.parametrize("configured", [None, 7200])
def test_private_evaluation_uses_the_requested_scoring_budget(
    tmp_path, monkeypatch, configured
):
    submission = tmp_path / "submission.csv"
    hidden = tmp_path / "test.csv"
    submission.write_text("id,prediction\na,1\n")
    hidden.write_text("id,label\na,1\n")
    expected = tmp_path / "expected.csv"
    expected.write_text("id\na\n")
    contract = SimpleNamespace(
        hidden_answers={"test": SimpleNamespace(path=hidden)},
        prediction_jobs={"test": SimpleNamespace(artifact=submission, expected=expected)},
    )
    monkeypatch.setattr(private_evaluation, "load_experiment_contract", lambda *_a, **_k: contract)
    calls = []

    def score(*_args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(ok=False, error="intentional scoring failure", failure_kind="execution")

    monkeypatch.setattr(private_evaluation, "score_artifacts_with_evaluator", score)
    kwargs = {} if configured is None else {"timeout": configured}
    result = private_evaluation.evaluate_private_outputs(tmp_path, tmp_path, **kwargs)
    assert result["error"] == "intentional scoring failure"
    assert result["results"] == []
    assert result["failure_kind"] == "execution"
    assert calls[0]["timeout"] == (3600 if configured is None else configured)
