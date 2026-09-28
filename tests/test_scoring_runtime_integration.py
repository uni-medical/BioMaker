import json
from pathlib import Path

import pytest

from biomaker.score import main
from biomaker.utils.evaluator_adapter import compile_adapter
from biomaker.utils import evaluator_setup


@pytest.fixture
def frozen_run(tmp_path):
    workspace = tmp_path / 'run'
    (workspace / 'working').mkdir(parents=True)
    spec = dict(evaluator_api_version=3, metrics=[dict(name='binary_dice', maximize=True)],
                task_family='segmentation', key_columns=['case'], prediction_columns=['mask'],
                answer_columns=['answer'], validation_artifact='./working/validation_predictions.csv',
                test_submission='./working/submission.csv', test_split=None,
                public_splits=['train', 'score_val'])
    source = 'import numpy as np\nADAPTER_SPEC = ' + repr(spec) + '''
def prediction_spec():
    return ADAPTER_SPEC
def load_sample(prediction_row, target_row, expected_row):
    sample = {'prediction': np.full((2, 2), int(prediction_row['mask']), dtype=np.uint8), 'input_shape': (2, 2)}
    if target_row is not None:
        sample['target'] = np.full((2, 2), int(target_row['answer']), dtype=np.uint8)
    return sample
def write_dummy_predictions(expected_rows, artifact_path):
    pass
'''
    frozen = compile_adapter(source, spec)
    namespace = {}
    exec(frozen, namespace)
    (workspace / 'evaluator.py').write_text(frozen)
    (workspace / 'evaluator_prediction_spec.json').write_text(json.dumps(namespace['prediction_spec']()))
    prediction = workspace / 'working/validation_predictions.csv'
    prediction.write_text('case,mask\nc,1\nb,0\na,1\n')
    (workspace / 'working/model.bin').write_bytes(b'saved weights')
    truth, expected = tmp_path / 'truth.csv', tmp_path / 'expected.csv'
    truth.write_text('case,answer\na,1\nb,1\nc,1\n')
    expected.write_text('case\na\nb\nc\n')
    return workspace, prediction, truth, expected


def command(inputs, output, *options):
    workspace, pred, truth, expected = inputs
    return ['--workspace', str(workspace), '--prediction', str(pred), '--truth', str(truth),
            '--expected', str(expected), '--output', str(output), *options]


def test_frozen_evaluator_parallel_rescore_logs_private_progress_and_preserves_inputs(frozen_run, tmp_path):
    workspace, pred, truth, expected = frozen_run
    before = {p: p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
    output = tmp_path / 'private/result.json'
    assert main(command(frozen_run, output, '--workers', '2', '--timeout', '30')) == 0
    payload = json.loads(output.read_text())
    assert payload['result']['details']['score']['metrics'] == {'binary_dice': pytest.approx(2 / 3)}
    assert payload['result']['details']['score']['n'] == 3
    events = [json.loads(s) for s in Path(payload['progress_path']).read_text().splitlines()]
    done = [e for e in events if e['stage'] == 'case' and e['event'] == 'finished']
    assert sorted(e['case_index'] for e in done) == [0, 1, 2]
    assert events[-1]['stage'] == 'score' and events[-1]['event'] == 'finished'
    assert all('prediction' not in e and 'target' not in e for e in events)
    assert not payload['training_started'] and not payload['journal_updated']
    assert all(p.read_bytes() == value for p, value in before.items())
    with pytest.raises(SystemExit):
        main(command(frozen_run, output))


@pytest.mark.parametrize('temporary_input', ['prediction', 'truth', 'expected', None])
def test_rescore_preserves_existing_temporary_paths_and_scoring_inputs(
        frozen_run, tmp_path, temporary_input):
    output = tmp_path / 'result.json'
    previous_temporary = output.with_suffix(output.suffix + '.tmp')
    inputs = list(frozen_run)
    if temporary_input is None:
        previous_temporary.write_text('existing unrelated evidence')
    else:
        index = {'prediction': 1, 'truth': 2, 'expected': 3}[temporary_input]
        previous_temporary.write_bytes(inputs[index].read_bytes())
        inputs[index] = previous_temporary
    before = previous_temporary.read_bytes()
    assert main(command(inputs, output, '--timeout', '30')) == 0
    assert previous_temporary.read_bytes() == before
    assert json.loads(output.read_text())['status'] == 'completed'


@pytest.mark.parametrize('bad_prediction', [
    'case,mask\nwrong,1\nb,0\nc,1\n',
    'case,mask\na,1\na,0\nc,1\n',
    'case,mask\na,1\n',
    'case,wrong-column\na,1\nb,0\nc,1\n',
])
def test_frozen_submission_row_violations_are_candidate_contract_failures(
        frozen_run, bad_prediction):
    workspace, pred, _truth, expected = frozen_run
    source = (workspace / 'evaluator.py').read_text()
    source += '\n_h_jobs["test"] = ' + repr({
        'expected': str(expected), 'artifact': str(pred),
    }) + '\n'
    (workspace / 'evaluator.py').write_text(source)
    pred.write_text(bad_prediction)
    result = evaluator_setup.validate_submission_with_evaluator(workspace, timeout=30)
    assert not result.ok and result.failure_kind == 'prediction_contract'


def test_frozen_adapter_answer_alignment_does_not_implicate_candidate(frozen_run):
    workspace, pred, truth, expected = frozen_run
    namespace = {}
    exec((workspace / 'evaluator.py').read_text(), namespace)
    truth.write_text('case,answer\nwrong,1\nb,1\nc,1\n')
    with pytest.raises(ValueError) as failure:
        namespace['score'](pred, truth, expected)
    assert not isinstance(failure.value, namespace['PredictionValidationError'])


def test_timeout_can_be_rescored_without_regeneration_or_retraining(frozen_run, tmp_path):
    workspace, pred, _, _ = frozen_run
    frozen = "def score(*args):\n    import time\n    time.sleep(1.5)\n    return {'metrics': {'binary_dice': 0.5}}\n"
    (workspace / 'evaluator.py').write_text(frozen)
    before = {p: p.read_bytes() for p in workspace.rglob('*') if p.is_file()}
    failed = tmp_path / 'private/failed.json'
    assert main(command(frozen_run, failed, '--timeout', '1')) == 124
    assert json.loads(failed.read_text())['result']['failure_kind'] == 'timeout'
    success = tmp_path / 'private/success.json'
    assert main(command(frozen_run, success, '--timeout', '10')) == 0
    assert all(p.read_bytes() == value for p, value in before.items())
    assert json.loads(failed.read_text())['status'] == 'failed'


@pytest.mark.parametrize('fail', [False, True])
def test_preparation_execution_environment_reaches_subprocess_and_is_restored(tmp_path, monkeypatch, fail):
    monkeypatch.setenv('BIOMAKER_EVALUATOR_WORKERS', '1')
    observed = []

    def prepare(*args):
        child = evaluator_setup._run_python(tmp_path,
            "import os; print(os.environ['BIOMAKER_EVALUATOR_WORKERS'])", timeout=10)
        observed.append(child.stdout.strip())
        if fail:
            raise RuntimeError('test failure')
        return evaluator_setup.PreparedEvaluator(tmp_path / 'evaluator.py', '{}')

    monkeypatch.setattr(evaluator_setup, '_prepare_evaluator', prepare)
    if fail:
        with pytest.raises(RuntimeError, match='test failure'):
            evaluator_setup.prepare_evaluator(tmp_path, tmp_path, {}, 'unused',
                                             env_overrides={'BIOMAKER_EVALUATOR_WORKERS': '2'})
    else:
        evaluator_setup.prepare_evaluator(tmp_path, tmp_path, {}, 'unused',
                                         env_overrides={'BIOMAKER_EVALUATOR_WORKERS': '2'})
    assert observed == ['2']
    child = evaluator_setup._run_python(tmp_path,
        "import os; print(os.environ['BIOMAKER_EVALUATOR_WORKERS'])", timeout=10)
    assert child.stdout.strip() == '1'


@pytest.mark.parametrize('bad_field,expected_kind', [('prediction', 'prediction_contract'), ('target', 'execution')])
def test_invalid_public_prediction_is_distinguished_from_bad_private_target(frozen_run, tmp_path, bad_field, expected_kind):
    workspace, pred, truth, expected = frozen_run
    # Unsupported label sign is rejected by the same trusted validation path.
    # Use signed arrays in the decoder so invalid values cannot wrap to uint8.
    source = (workspace / 'evaluator.py').read_text().replace('dtype=np.uint8', 'dtype=np.int16')
    (workspace / 'evaluator.py').write_text(source)
    if bad_field == 'prediction':
        pred.write_text('case,mask\na,-1\nb,0\nc,1\n')
    else:
        truth.write_text('case,answer\na,-1\nb,1\nc,1\n')
    result = evaluator_setup.score_artifacts_with_evaluator(workspace, pred, truth, expected, timeout=30)
    assert not result.ok and result.failure_kind == expected_kind


def test_frozen_adapter_releases_previous_volume_before_loading_next(frozen_run):
    import weakref
    import numpy as np
    import pandas as pd

    workspace, _, _, _ = frozen_run
    namespace = {}
    exec((workspace / 'evaluator.py').read_text(), namespace)
    references = []

    def load(*args):
        assert all(ref() is None for ref in references), 'decoder retained the previous volume'
        prediction = np.ones((2, 2), dtype=np.uint8)
        references.append(weakref.ref(prediction))
        return dict(prediction=prediction)

    namespace['_h_load_sample'] = load
    frame = pd.DataFrame({'case': ['a', 'b', 'c']})
    samples = namespace['_h_samples'](frame, None, frame)
    for _ in range(3):
        sample = next(samples)
        assert sample['prediction'].shape == (2, 2)
        del sample
    with pytest.raises(StopIteration):
        next(samples)
    assert all(ref() is None for ref in references)
