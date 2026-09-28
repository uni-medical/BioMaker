"""Scoring execution changes must preserve the frozen mathematical contract."""

import json
import math
import threading
import weakref

import numpy as np
import pytest

from biomaker.utils import segmentation_metrics as seg
from biomaker.utils import trusted_metrics as trusted
from biomaker.utils.metric_source import frozen_metric_source


def line_sample():
    target = np.zeros((5, 5, 12), dtype=np.uint8)
    target[2, 2, 1:11] = 1
    prediction = target.copy()
    prediction[0, 0, 0] = 1
    parsing = np.zeros_like(target)
    parsing[2, 2, 1:5] = 1
    parsing[2, 2, 5:11] = 2
    return dict(prediction=prediction, target=target, skeleton=target.copy(), parsing=parsing)


def tree_spec(betti_reference='one'):
    return dict(metrics=[{'name': name, 'maximize': name != 'betti_error'} for name in (
        'binary_dice', 'cldice', 'tree_length_detected', 'branch_detected', 'betti_error')],
        betti_reference=betti_reference)


def test_full_tree_input_validation_runs_once_but_public_case_api_still_validates(monkeypatch):
    calls = []
    original = trusted.validate_tree_sample

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(trusted, 'validate_tree_sample', counted)
    result = trusted.score_samples([line_sample()], tree_spec())
    assert len(calls) == 1
    assert result['metrics']['binary_dice'] == pytest.approx(20 / 21)
    invalid = line_sample()
    invalid['skeleton'][2, 2, 0] = 1
    with pytest.raises(ValueError, match='positive parsing'):
        trusted.score_samples([invalid], tree_spec())
    with pytest.raises(ValueError, match='positive parsing'):
        seg.segmentation_score_case(invalid, ['branch_detected'])


@pytest.mark.parametrize('reference,expected_calls', [('one', 1), ('target', 2)])
def test_components_reused_for_coverage_and_betti(monkeypatch, reference, expected_calls):
    calls = []
    original = seg._seg_ndi.label

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(seg._seg_ndi, 'label', counted)
    result = trusted.score_samples([line_sample()], tree_spec(reference))
    assert len(calls) == expected_calls
    assert result['metrics']['tree_length_detected'] == 1.0
    assert result['metrics']['branch_detected'] == 1.0
    assert result['metrics']['betti_error'] == 1.0


def test_progress_has_timings_without_changing_payload_or_exposing_sample_fields():
    sample = line_sample()
    sample['id'] = '/secret/private/case-name'
    for value in sample.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    expected = trusted.score_samples([sample], tree_spec())
    events = []
    assert trusted.score_samples([sample], tree_spec(), progress_callback=events.append) == expected
    stages = {event['stage'] for event in events}
    assert {'score', 'load', 'validate', 'metric', 'case', 'components'} <= stages
    assert [event['case_index'] for event in events
            if event['stage'] == 'case' and event['event'] == 'finished'] == [0]
    timed = [event for event in events if 'elapsed_seconds' in event]
    assert timed and all(math.isfinite(event['elapsed_seconds']) and event['elapsed_seconds'] >= 0
                         for event in timed)
    assert '/secret' not in json.dumps(events)
    assert all(set(event) <= {'stage', 'event', 'metric_names', 'case_index',
                             'elapsed_seconds', 'error_type'} for event in events)
    assert events[-1]['stage'] == 'score' and events[-1]['event'] == 'finished'


@pytest.mark.parametrize('workers', [0, -1, True, 1.5, '2', None])
def test_invalid_workers_are_rejected(workers):
    with pytest.raises(ValueError, match='workers'):
        trusted.score_samples([], {'primary_metric': 'binary_dice'}, workers=workers)


def test_splitwise_metrics_cannot_be_changed_to_casewise_means():
    with pytest.raises(ValueError, match='Parallel case'):
        trusted.score_samples([], {'primary_metric': 'auc', 'task_type': 'binary', 'class_labels': [0, 1]}, workers=2)
    with pytest.raises(ValueError, match='Parallel case'):
        trusted.score_samples([], {'primary_metric': 'mae'}, workers=2)


def test_parallel_result_order_and_average_match_serial_even_with_reversed_completion(monkeypatch):
    samples = [dict(prediction=np.full((2, 2), i % 2, dtype=np.uint8),
                    target=np.ones((2, 2), dtype=np.uint8)) for i in range(5)]
    protocol = {'primary_metric': 'binary_dice'}
    expected = trusted.score_samples(iter(samples), protocol)
    second_completed = threading.Event()
    original = trusted._score_image_case

    def reverse_first_two(index, *args, **kwargs):
        if index == 0:
            assert second_completed.wait(timeout=5), 'second worker did not execute concurrently'
        result = original(index, *args, **kwargs)
        if index == 1:
            second_completed.set()
        return result

    monkeypatch.setattr(trusted, '_score_image_case', reverse_first_two)
    events = []
    actual = trusted.score_samples(iter(samples), protocol, workers=2, progress_callback=events.append)
    assert actual == expected
    completions = [event['case_index'] for event in events
                   if event['stage'] == 'case' and event['event'] == 'finished']
    # The second case must finish before the first, but another queued case
    # may also finish before the scheduler wakes the first worker.
    assert completions.index(1) < completions.index(0)
    assert [row['binary_dice'] for row in actual['per_case']] == [0., 1., 0., 1., 0.]


@pytest.mark.parametrize('workers', [1, 2, 3])
def test_loaded_arrays_are_bounded_and_released(workers):
    references = []
    peak = 0

    def volumes():
        nonlocal peak
        for index in range(18):
            prediction = np.full((5, 5), index % 2, dtype=np.uint8)
            target = np.ones_like(prediction)
            references.extend((weakref.ref(prediction), weakref.ref(target)))
            peak = max(peak, sum(ref() is not None for ref in references))
            # One loader is advancing; all earlier pending calls fit the cap.
            assert peak <= workers * 2
            yield dict(prediction=prediction, target=target)

    result = trusted.score_samples(volumes(), {'primary_metric': 'binary_dice'}, workers=workers)
    assert result['n'] == 18
    assert result['metric'] == 0.5
    assert all(ref() is None for ref in references)


def test_parallel_restoration_keeps_corner_and_group_weighting():
    protocol = dict(primary_metric='psnr', restoration_evaluation=dict(
        data_range=255, crop_size=2, crop_mode='four_corner_crops_no_center',
        groups=['a', 'b'], aggregation='mean_groups'))
    samples = [dict(prediction=np.full((4, 4), error, dtype=float),
                    target=np.zeros((4, 4)), group=group)
               for error, group in [(1, 'a'), (3, 'a'), (20, 'b')]]
    serial = trusted.score_samples(iter(samples), protocol)
    parallel = trusted.score_samples(iter(samples), protocol, workers=2)
    assert parallel == serial
    assert parallel['metric'] == pytest.approx(((20 * math.log10(255) + 20 * math.log10(85)) / 2
                                                + 20 * math.log10(255 / 20)) / 2)


@pytest.mark.parametrize('workers', [1, 2])
def test_invalid_sample_fails_whole_split_without_partial_result(workers):
    good = dict(prediction=np.ones((2, 2)), target=np.ones((2, 2)))
    invalid = dict(prediction=np.full((2, 2), np.nan), target=np.ones((2, 2)))
    events = []
    with pytest.raises(ValueError, match='non-finite'):
        trusted.score_samples(iter([good, invalid, good]), {'primary_metric': 'binary_dice'},
                              workers=workers, progress_callback=events.append)
    assert any(event['stage'] == 'validate' and event['event'] == 'failed' for event in events)
    assert events[-1]['stage'] == 'score' and events[-1]['event'] == 'failed'
    assert not any(event['stage'] == 'score' and event['event'] == 'finished' for event in events)


def test_loader_failure_is_distinguishable_from_invalid_prediction():
    events = []

    def unreadable():
        raise OSError('private label filename should not appear in timing output')
        yield  # pragma: no cover

    with pytest.raises(OSError):
        trusted.score_samples(unreadable(), {'primary_metric': 'binary_dice'}, progress_callback=events.append)
    assert any(event['stage'] == 'load' and event['event'] == 'failed' and
               event['error_type'] == 'OSError' for event in events)
    assert 'private label filename' not in json.dumps(events)


def test_callback_failure_is_not_hidden():
    def broken(event):
        raise RuntimeError('progress sink unavailable')

    with pytest.raises(RuntimeError, match='progress sink'):
        trusted.score_samples([line_sample()], tree_spec(), progress_callback=broken)


def test_frozen_kernel_supports_parallel_execution_and_progress():
    namespace = {}
    exec(frozen_metric_source(), namespace)
    expected = trusted.score_samples([line_sample(), line_sample()], tree_spec())
    events = []
    assert namespace['score_samples']([line_sample(), line_sample()], tree_spec(),
                                     workers=2, progress_callback=events.append) == expected
    assert sorted(event['case_index'] for event in events if event['stage'] == 'case'
                  and event['event'] == 'finished') == [0, 1]


def test_idle_worker_takes_next_case_while_first_case_is_still_running(monkeypatch):
    third_started = threading.Event()
    first_done = threading.Event()
    original = trusted._score_image_case

    def uneven(index, *args, **kwargs):
        if index == 0:
            assert third_started.wait(timeout=5), 'completed later case left its worker idle'
        elif index == 2:
            assert not first_done.is_set()
            third_started.set()
        result = original(index, *args, **kwargs)
        if index == 0:
            first_done.set()
        return result

    monkeypatch.setattr(trusted, '_score_image_case', uneven)
    rows = [dict(prediction=np.full((2, 2), i % 2), target=np.ones((2, 2))) for i in range(4)]
    result = trusted.score_samples(iter(rows), {'primary_metric': 'binary_dice'}, workers=2)
    assert [row['binary_dice'] for row in result['per_case']] == [0., 1., 0., 1.]


def test_observed_later_case_failure_does_not_load_more_inputs(monkeypatch):
    release_first = threading.Event()
    loaded = []
    original = trusted._score_image_case
    original_shutdown = trusted.ThreadPoolExecutor.shutdown

    def delayed_first(index, *args, **kwargs):
        if index == 0:
            assert release_first.wait(timeout=5), 'failure was not observed before draining threads'
            return original(index, *args, **kwargs)
        raise RuntimeError('second case failed')

    def releasing_shutdown(pool, *args, **kwargs):
        # Entering cleanup proves future.result() observed case 1's exception
        # while case 0 was still pending; no additional input may be consumed.
        assert loaded == [0, 1]
        release_first.set()
        return original_shutdown(pool, *args, **kwargs)

    def rows():
        for index in range(5):
            loaded.append(index)
            yield dict(prediction=np.ones((2, 2)), target=np.ones((2, 2)))

    monkeypatch.setattr(trusted, '_score_image_case', delayed_first)
    monkeypatch.setattr(trusted.ThreadPoolExecutor, 'shutdown', releasing_shutdown)
    with pytest.raises(RuntimeError, match='second case failed'):
        trusted.score_samples(rows(), {'primary_metric': 'binary_dice'}, workers=2)
    assert loaded == [0, 1]


@pytest.mark.parametrize('family,protocol,good_prediction,bad_prediction,good_target,bad_target', [
    ('segmentation', {'primary_metric': 'binary_dice'}, np.ones((2, 2)),
     np.full((2, 2), np.nan), np.ones((2, 2)), np.full((2, 2), np.nan)),
    ('classification', {'primary_metric': 'accuracy', 'task_type': 'binary', 'class_labels': [0, 1]},
     np.array([0.25, 0.75]), np.array([2., -1.]), 1, 2),
    ('regression', {'primary_metric': 'mae'}, np.array([1., 2.]), np.array([[1., 2.]]),
     np.array([1., 3.]), np.array([1., 2., 3.])),
])
def test_prediction_error_type_requires_public_only_evidence(
        family, protocol, good_prediction, bad_prediction, good_target, bad_target):
    with pytest.raises(trusted.PredictionValidationError):
        trusted.score_samples([dict(prediction=bad_prediction, target=good_target)], protocol)
    with pytest.raises(trusted.PredictionValidationError):
        trusted.validate_prediction(dict(prediction=bad_prediction), protocol)
    with pytest.raises(ValueError) as private_failure:
        trusted.score_samples([dict(prediction=good_prediction, target=bad_target)], protocol)
    assert not isinstance(private_failure.value, trusted.PredictionValidationError)


def test_bad_private_skeleton_is_not_attributed_to_prediction():
    sample = line_sample()
    sample['skeleton'][2, 2, 0] = 1
    with pytest.raises(ValueError, match='positive parsing') as failure:
        trusted.score_samples([sample], tree_spec())
    assert not isinstance(failure.value, trusted.PredictionValidationError)
    # A caller cannot manufacture validation provenance by setting a flag.
    sample['prediction_validated'] = False
    sample['prediction'][0, 0, 0] = 255
    contract = tree_spec()
    contract['label_ids'] = [0, 1]
    with pytest.raises(trusted.PredictionValidationError, match='label_ids'):
        trusted.score_samples([sample], contract)


def test_public_geometry_violation_is_prediction_contract_error():
    sample = dict(prediction=np.ones((2, 2)), target=np.ones((2, 2)), input_shape=(3, 3))
    with pytest.raises(trusted.PredictionValidationError, match='input_shape'):
        trusted.score_samples([sample], {'primary_metric': 'binary_dice'})
