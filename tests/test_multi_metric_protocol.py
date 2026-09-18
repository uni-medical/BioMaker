"""Multi-objective declarations and frozen evaluator v2/v3 interoperability."""

import copy
import json

import pytest

from biomaker.utils.evaluator_adapter import compile_adapter
from biomaker.utils.evaluator_protocol import (
    metric_definitions, requested_metric_definitions, requested_metrics, validate_score,
)
from biomaker.utils.evaluator_setup import (
    EvaluatorRunResult, _resolve_metric_contract, _validate_scoring_contract,
    score_artifacts_with_evaluator, _validate_custom_metric_contract,
)
from biomaker.utils.metric_declarations import (
    canonical_metric_contract, declared_metric_definitions, validate_declared_capabilities,
)
from biomaker.utils.trusted_metrics import normalize_metric_spec, score_samples


def test_arbitrary_objectives_infer_directions_and_deduplicate_aliases():
    spec = {'metrics': [{'name': 'mae'}, {'name': 'mse'}, {'name': 'rmse'},
                        {'name': 'mean_absolute_error', 'maximize': False}]}
    before = copy.deepcopy(spec)
    assert metric_definitions(spec) == [dict(name=name, maximize=False) for name in ('mae', 'mse', 'rmse')]
    assert before == spec


@pytest.mark.parametrize('spec,error', [
    ({'metrics': []}, 'nonempty'),
    ({'metrics': {}}, 'nonempty'),
    ({'metrics': [{'name': ''}]}, 'nonempty'),
    ({'metrics': [{'name': 'custom'}]}, 'maximize'),
    ({'metrics': [{'name': 'mae', 'maximize': True}]}, 'direction'),
    ({'metrics': [{'name': 'custom', 'maximize': 1}]}, 'maximize'),
    ({'metrics': [{'name': 'dice'}, {'name': 'binary_dice', 'threshold': .3}]}, 'Conflicting'),
    ({'metrics': [{'name': 'mae'}, {'name': 'dice'}]}, 'family'),
    ({'metrics': [{'name': 'mae'}], 'primary_metric': 'mse'}, 'disagree'),
    ({'primary_metric': 'dice', 'secondary_metrics': ['research_metric']}, 'maximize'),
])
def test_invalid_metric_declarations(spec, error):
    with pytest.raises(ValueError, match=error):
        metric_definitions(spec)


def test_legacy_primary_and_all_secondaries_are_equal_objectives():
    assert metric_definitions({'primary_metric': 'dice', 'maximize': True, 'secondary_metrics': ['hd95', 'aji']}) == [
        {'name': 'binary_dice', 'maximize': True}, {'name': 'hd95', 'maximize': False}, {'name': 'aji', 'maximize': True}]


def test_structured_task_and_manifest_preserve_custom_parameters():
    definitions = [{'name': 'research_dice', 'maximize': True, 'matching': 'strict'}, {'name': 'hd95'}]
    task = '```metrics\n' + json.dumps(definitions) + '\n```'
    manifest = {'metric_contract': {'metrics': definitions}}
    resolved = declared_metric_definitions(manifest, task)
    assert resolved[0]['matching'] == 'strict'
    assert resolved[1]['maximize'] is False
    assert requested_metrics(task) == ('research_dice', ['hd95'])
    assert requested_metric_definitions('```yaml\nmetrics:\n  - name: hd95\n```') == [{'name': 'hd95', 'maximize': False}]
    manifest['metric_contract']['metrics'][0]['maximize'] = False
    with pytest.raises(ValueError, match='disagree'):
        declared_metric_definitions(manifest, task)


def test_reordered_objectives_match_task_and_prepared_evaluator(tmp_path):
    definitions = [{'name': 'dice', 'maximize': True}, {'name': 'hd95', 'maximize': False}]
    task = '```metrics\n' + json.dumps(definitions[::-1]) + '\n```'
    manifest = {'metric_contract': {'metrics': definitions}}
    resolved = declared_metric_definitions(manifest, task)
    assert [item['name'] for item in resolved] == ['binary_dice', 'hd95']
    path = tmp_path / 'input/biomaker_splits/split_manifest.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    actual = _resolve_metric_contract({'metrics': definitions[::-1]}, tmp_path, task)
    assert actual['metrics'] == resolved


@pytest.mark.parametrize('definition', [
    {'name': 'accuracy', 'decision_threshold': .3},
    {'name': 'ssim', 'ssim_parameters': {'window_size': 9}},
])
def test_task_metric_parameter_aliases_match_canonical_manifest_and_evaluator(definition):
    task = '```metrics\n' + json.dumps([definition]) + '\n```'
    canonical = canonical_metric_contract({'metrics': [definition]})
    assert declared_metric_definitions({'metric_contract': canonical}, task) == canonical['metrics']
    _validate_scoring_contract({'evaluator_api_version': 3, **canonical}, task)
    _validate_scoring_contract({'evaluator_api_version': 3, 'metrics': [definition]}, task)


def test_duplicate_metric_parameter_aliases_are_equivalent():
    assert metric_definitions({'metrics': [{'name': 'accuracy', 'threshold': .3},
                                          {'name': 'acc', 'decision_threshold': .3}]}) == [
        {'name': 'accuracy', 'maximize': True, 'threshold': .3}]


def test_conflicting_legacy_named_parameters_are_not_discarded():
    with pytest.raises(ValueError, match='Conflicting'):
        canonical_metric_contract({'metrics': [{'name': 'accuracy', 'threshold': .3}],
                                   'primary_metric': {'name': 'accuracy', 'threshold': .7}, 'maximize': True})


def test_multiple_yaml_metric_blocks_are_rejected():
    with pytest.raises(ValueError, match='exactly one'):
        requested_metric_definitions('```yaml\nmetrics: [{name: mae}]\n```\n```json\n{"metrics": [{"name": "rmse"}]}\n```')


def test_task_top_level_custom_settings_cannot_silently_drift(tmp_path):
    definitions = [{'name': 'research_metric', 'maximize': True}]
    task = '```metrics\n' + json.dumps({'metrics': definitions, 'threshold': .3}) + '\n```'
    path = tmp_path / 'input/biomaker_splits/split_manifest.json'
    path.parent.mkdir(parents=True)
    for threshold in (.3, .7):
        contract = {'metrics': definitions, 'threshold': threshold}
        path.write_text(json.dumps({'metric_contract': contract}))
        checks = [
            lambda: declared_metric_definitions({'metric_contract': contract}, task),
            lambda: _validate_scoring_contract({'evaluator_api_version': 3, **contract}, task),
            lambda: _validate_custom_metric_contract({'evaluator_api_version': 3, **contract}, tmp_path, task),
        ]
        for check in checks:
            if threshold == .3:
                check()
            else:
                with pytest.raises((ValueError, RuntimeError), match='threshold'):
                    check()


def test_task_top_level_builtin_parameters_are_frozen_without_manifest(tmp_path):
    task = '```metrics\n{"metrics": [{"name": "accuracy"}], "decision_threshold": 0.3}\n```'
    spec = _resolve_metric_contract({'metrics': [{'name': 'accuracy'}], 'task_type': 'binary',
                                    'class_labels': [0, 1], 'prediction_type': 'probability_columns'},
                                   tmp_path, task, require_match=False)
    assert spec['threshold'] == .3


def test_v3_normalization_freezes_parameters_without_legacy_roles(tmp_path):
    contract = {'metrics': [{'name': 'mae', 'units': 'months'}, {'name': 'rmse'}]}
    canonical = canonical_metric_contract(contract)
    assert canonical['metrics'][0] == {'name': 'mae', 'maximize': False, 'units': 'months'}
    assert canonical['units'] == 'months'
    validate_declared_capabilities({'metric_contract': contract}, '')
    path = tmp_path / 'input/biomaker_splits/split_manifest.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'metric_contract': contract}))
    spec = _resolve_metric_contract(dict(primary_metric='mae', prediction_type='continuous_scalar',
                                        prediction_columns=['estimate'], answer_columns=['target']),
                                    tmp_path, '', require_match=False)
    assert spec['evaluator_api_version'] == 3
    assert spec['metrics'] == canonical['metrics']
    assert not {'primary_metric', 'secondary_metrics', 'maximize'} & spec.keys()


def test_v3_raw_values_and_diagnostics_are_not_ranks_or_extra_objectives():
    spec = {'evaluator_api_version': 3, 'metrics': [{'name': 'dice'}, {'name': 'hd95'}]}
    result = validate_score({'metrics': {'dice': .73, 'hd95': 2.4, 'dice/group': .81}, 'n': 2}, spec)
    assert set(result['metrics']) == {'binary_dice', 'hd95', 'dice/group'}
    assert {item['name'] for item in result['metric_specs']} == {'binary_dice', 'hd95'}
    assert not {'metric', 'maximize', 'average_rank'} & result.keys()
    envelope = json.loads(EvaluatorRunResult(True, None, True, {'score': result}, '').log_line().split(': ', 1)[1])
    assert envelope['metrics'] == result['metrics']
    assert envelope['metric_specs'] == result['metric_specs']


@pytest.mark.parametrize('values', [
    {'binary_dice': .5}, {'binary_dice': float('nan'), 'hd95': 2},
    {'binary_dice': .5, 'hd95': float('inf')}, {'binary_dice': True, 'hd95': 2},
])
def test_invalid_v3_results_fail_the_entire_candidate(values):
    with pytest.raises(ValueError):
        validate_score({'metrics': values}, {'evaluator_api_version': 3, 'metrics': [{'name': 'dice'}, {'name': 'hd95'}]})


def test_three_regression_objectives_compute_original_values():
    result = score_samples([{'prediction': 3, 'target': 1}, {'prediction': 0, 'target': 1}],
                           {'evaluator_api_version': 3, 'metrics': [{'name': n} for n in ('mae', 'mse', 'rmse')]})
    assert result['metrics'] == {'mae': 1.5, 'mse': 2.5, 'rmse': pytest.approx(2.5 ** .5)}
    assert 'metric' not in result


def test_standalone_adapter_compiles_legacy_declaration_as_v3():
    source = '''
def prediction_spec():
    return {'evaluator_api_version': 2, 'primary_metric': 'mae', 'secondary_metrics': ['rmse'], 'maximize': False,
            'key_columns': ['id'], 'prediction_columns': ['p'], 'answer_columns': ['y'],
            'validation_artifact': './working/validation_predictions.csv', 'test_submission': './working/submission.csv'}
def load_sample(prediction_row, target_row, expected_row):
    return {'prediction': float(prediction_row['p']), 'target': float(target_row['y'])}
def write_dummy_predictions(expected_rows, artifact_path):
    pass
'''
    namespace = {}
    exec(compile_adapter(source), namespace)
    spec = namespace['prediction_spec']()
    _validate_scoring_contract(spec)
    assert spec['metrics'] == [{'name': 'mae', 'maximize': False}, {'name': 'rmse', 'maximize': False}]
    assert 'primary_metric' not in spec


@pytest.mark.parametrize('version', [2, 3])
def test_frozen_custom_evaluator_rescores_without_rewriting_source(tmp_path, version):
    definitions = [{'name': 'custom_dice', 'maximize': True}, {'name': 'custom_distance', 'maximize': False}]
    spec = dict(evaluator_api_version=version, key_columns=['id'], prediction_columns=['prediction'],
                validation_artifact='./working/validation_predictions.csv', test_submission='./working/submission.csv')
    if version == 3:
        spec['metrics'] = definitions
        score = {'metrics': {'custom_dice': .8123, 'custom_distance': 17.9}}
    else:
        spec.update(primary_metric='custom_dice', secondary_metrics=['custom_distance'], maximize=True)
        score = {'metric': .8123, 'maximize': True, 'metrics': {'custom_dice': .8123, 'custom_distance': 17.9}}
    (tmp_path / 'evaluator_prediction_spec.json').write_text(json.dumps(spec))
    source = 'def score(*paths):\n    return ' + repr(score) + '\n'
    evaluator = tmp_path / 'evaluator.py'
    evaluator.write_text(source)
    prediction, truth, expected = [tmp_path / name for name in ('prediction.csv', 'truth.csv', 'expected.csv')]
    prediction.write_text('id,prediction\na,1\n')
    truth.write_text('id,answer\na,1\n')
    expected.write_text('id\na\n')
    result = score_artifacts_with_evaluator(tmp_path, prediction, truth, expected)
    assert result.ok, result.error or result.output
    assert result.metrics == score['metrics']
    assert result.metric_specs == (definitions if version == 3 else definitions[:1])
    assert evaluator.read_text() == source
    assert json.loads((tmp_path / 'evaluator_prediction_spec.json').read_text()) == spec
