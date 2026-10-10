"""The validation winner is immutable and private evaluation is once per delivery."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from biomaker.journal import Journal, Node
from biomaker.utils import final_selection as final
from biomaker.utils.metric import MetricValue


def setup(tmp_path, *, test=True, answers=True):
    workspace = tmp_path / 'candidate'
    working = workspace / 'working'
    split = workspace / 'input/biomaker_splits'
    working.mkdir(parents=True)
    split.mkdir(parents=True)
    private = tmp_path / 'logs/private_gt'
    private.mkdir(parents=True)
    spec = {'evaluator_api_version': 3, 'metrics': [{'name': 'accuracy', 'maximize': True}],
            'key_columns': ['id'], 'prediction_columns': ['prediction'],
            'validation_artifact': './working/validation_predictions.csv',
            'test_submission': './working/submission.csv' if test else None}
    (split / 'agent_train.csv').write_text('id,label\na,1\n')
    (split / 'score_val.csv').write_text('id\nb\n')
    (working / 'validation_predictions.csv').write_text('id,prediction\nb,1\n')
    (working / 'model.bin').write_bytes(b'synthetic fitted state')
    manifest = {'public_splits': ['train', 'score_val'] + (['test'] if test else [])}
    (split / 'split_manifest.json').write_text(json.dumps(manifest))
    if test:
        (split / 'agent_test.csv').write_text('id\nc\n')
        (working / 'submission.csv').write_text('id,prediction\nc,1\n')
    if answers:
        (private / 'private_gt.json').write_text(json.dumps({'test': {'file': 'test.csv', 'answer_columns': ['label']}}))
        (private / 'test.csv').write_text('id,label\nc,1\n')
    (workspace / 'evaluator.py').write_text('# frozen synthetic scorer')
    (workspace / 'evaluator_prediction_spec.json').write_text(json.dumps(spec))
    cfg = SimpleNamespace(log_dir=tmp_path / 'logs', exec=SimpleNamespace(evaluator_timeout=17, evaluator_workers=2))
    best = Node(code='print("trained")', id='selected', is_buggy=False, execution_workspace=str(workspace),
                metric=MetricValue(.8, maximize=True), metrics={'accuracy': .8}, metric_specs=spec['metrics'])
    journal = Journal(selection_policy='average_rank', metric_specs=spec['metrics'])
    journal.append(best)
    return cfg, best, journal, workspace, private


def test_once_cpu_scoring_receipt_and_opaque_private_results(tmp_path, monkeypatch):
    cfg, best, journal, workspace, private = setup(tmp_path)
    selection = final.freeze_selection(cfg, journal, best)
    calls = []
    def score(*args, **kw):
        calls.append(kw)
        assert json.loads((cfg.log_dir / 'private_metrics.json').read_text())['latest']['status'] == 'started'
        return {'results': [{'name': 'private_test', 'metrics': {'accuracy': .5}}]}
    monkeypatch.setattr(final, 'evaluate_private_outputs', score)
    first = final.evaluate_selected_delivery(cfg, best, workspace, private, selection, stage='selected_candidate')
    second = final.evaluate_selected_delivery(cfg, best, workspace, private, selection, stage='selected_candidate')
    assert first == second and len(calls) == 1
    assert calls[0]['timeout'] == 17
    assert calls[0]['env_overrides']['CUDA_VISIBLE_DEVICES'] == ''
    assert calls[0]['env_overrides']['BIOMAKER_EVALUATOR_WORKERS'] == '2'
    assert 'private_test' not in (cfg.log_dir / 'selection.json').read_text()
    assert journal.get_best_node() is best
    assert len(json.loads((cfg.log_dir / 'private_metrics.json').read_text())['history']) == 1


@pytest.mark.parametrize('test,answers,reason', [(False, False, 'no_test_job'), (True, False, 'no_private_test_answers')])
def test_undeclared_private_evaluation_skips_without_call(tmp_path, monkeypatch, test, answers, reason):
    cfg, best, journal, workspace, private = setup(tmp_path, test=test, answers=answers)
    monkeypatch.setattr(final, 'evaluate_private_outputs', lambda *a, **kw: pytest.fail('must skip'))
    receipt = final.evaluate_selected_delivery(cfg, best, workspace, private, final.freeze_selection(cfg, journal, best), stage='selected_candidate')
    assert receipt['status'] == 'skipped' and receipt['skip_reason'] == reason


@pytest.mark.parametrize('damage', ['model', 'prediction', 'evaluator', 'contract'])
def test_changed_artifacts_cannot_be_scored_as_frozen_selection(tmp_path, monkeypatch, damage):
    cfg, best, journal, workspace, private = setup(tmp_path)
    selection = final.freeze_selection(cfg, journal, best)
    if damage == 'model':
        (workspace / 'working/model.bin').write_bytes(b'other model')
    elif damage == 'prediction':
        (workspace / 'working/submission.csv').write_text('id,prediction\nc,0\n')
    elif damage == 'evaluator':
        (workspace / 'evaluator.py').write_text('# changed')
    else:
        path = workspace / 'evaluator_prediction_spec.json'
        spec = json.loads(path.read_text()); spec['extra'] = True; path.write_text(json.dumps(spec))
    monkeypatch.setattr(final, 'evaluate_private_outputs', lambda *a, **kw: pytest.fail('changed artifact'))
    with pytest.raises(RuntimeError, match='changed|disagrees'):
        final.evaluate_selected_delivery(cfg, best, workspace, private, selection, stage='selected_candidate')


def test_failed_evaluation_retains_delivery_and_cannot_repeat(tmp_path, monkeypatch):
    cfg, best, journal, workspace, private = setup(tmp_path)
    selection = final.freeze_selection(cfg, journal, best)
    calls = []
    def fail(*a, **kw):
        calls.append(1)
        return {'error': 'temporary scoring outage', 'failure_kind': 'resource', 'results': []}
    monkeypatch.setattr(final, 'evaluate_private_outputs', fail)
    for _ in range(2):
        with pytest.raises(final.EvaluatorInfrastructureError):
            final.evaluate_selected_delivery(cfg, best, workspace, private, selection, stage='selected_candidate')
    assert len(calls) == 1
    assert (workspace / 'working/model.bin').is_file()
    assert (cfg.log_dir / 'selection.json').is_file()


def test_declared_missing_truth_is_failure_not_skip(tmp_path, monkeypatch):
    cfg, best, journal, workspace, private = setup(tmp_path)
    (private / 'test.csv').unlink()
    selection = final.freeze_selection(cfg, journal, best)
    with pytest.raises(final.EvaluatorInfrastructureError):
        final.evaluate_selected_delivery(cfg, best, workspace, private, selection, stage='selected_candidate')
    receipt = json.loads((cfg.log_dir / 'private_metrics.json').read_text())['latest']
    assert receipt['status'] == 'failed'


def test_missing_selected_workspace_has_no_fallback(tmp_path):
    cfg, best, journal, workspace, private = setup(tmp_path)
    best.execution_workspace = str(tmp_path / 'missing')
    with pytest.raises(RuntimeError, match='workspace is missing'):
        final.freeze_selection(cfg, journal, best)
