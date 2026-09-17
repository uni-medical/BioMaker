"""Metric references independent of evaluator file layouts and adapter code."""

import copy
import json
import math
import weakref

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from biomaker.utils.trusted_metrics import (
    metric_capabilities,
    normalize_metric_spec,
    normalize_ssim_parameters,
    restoration_metrics,
    score_samples,
    validate_prediction,
)


def restoration_spec(primary='psnr', secondary=('ssim',), **protocol):
    return dict(primary_metric=primary, secondary_metrics=list(secondary),
                restoration_evaluation=dict(data_range=255, crop_size=None,
                                            groups=None, aggregation='mean_images', **protocol))


def sliding_ssim(a, b, data_range, window=7, covariance=True, k1=.01, k2=.03):
    """Direct patch statistics, without scipy filtering or skimage."""
    values = []
    for i in range(a.shape[0] - window + 1):
        for j in range(a.shape[1] - window + 1):
            x, y = a[i:i+window, j:j+window], b[i:i+window, j:j+window]
            mx, my = x.mean(), y.mean()
            ddof = 1 if covariance else 0
            vx, vy = x.var(ddof=ddof), y.var(ddof=ddof)
            cross = ((x-mx)*(y-my)).sum() / (x.size-ddof)
            c1, c2 = (k1*data_range)**2, (k2*data_range)**2
            values.append(((2*mx*my+c1)*(2*cross+c2))/((mx*mx+my*my+c1)*(vx+vy+c2)))
    return np.mean(values)


@pytest.mark.parametrize('window,covariance', [(3, True), (5, False), (7, True)])
@pytest.mark.parametrize('constant', [False, True])
def test_restoration_independent_reference(window, covariance, constant):
    rng = np.random.default_rng(42)
    a = rng.uniform(0, 255, (11, 13)) if not constant else np.full((11, 13), 70.)
    b = rng.uniform(0, 255, (11, 13)) if not constant else a + 3
    params = dict(window_size=window, use_sample_covariance=covariance, K1=.02, K2=.04)
    result = restoration_metrics(a, b, 255, params)
    assert result['psnr'] == pytest.approx(10 * math.log10(255**2 / np.mean((a-b)**2)), abs=1e-12)
    assert result['ssim'] == pytest.approx(sliding_ssim(a, b, 255, window, covariance, .02, .04), abs=1e-12)


@pytest.mark.parametrize('params', [dict(gaussian_weights=True), dict(gaussian_weights=True, sigma=1.2, win_size=7, use_sample_covariance=False)])
def test_gaussian_against_independent_skimage(params):
    skimage = pytest.importorskip('skimage.metrics')
    rng = np.random.default_rng(82)
    a, b = rng.uniform(0, 1, (2, 27, 31))
    result = restoration_metrics(a, b, 1, params, ['ssim'])
    assert result['ssim'] == pytest.approx(skimage.structural_similarity(a, b, data_range=1, **params), abs=1e-12)


@pytest.mark.parametrize('sigma,window,covariance', [(1.5, 11, True), (1.2, 7, False)])
def test_gaussian_against_direct_weighted_patches(sigma, window, covariance):
    rng = np.random.default_rng(6)
    a, b = rng.uniform(0, 1, (2, 19, 21))
    radius = int(3.5 * sigma + .5)
    weights = np.exp(-np.arange(-radius, radius+1, dtype=float)**2 / (2*sigma*sigma))
    weights /= weights.sum()
    weights = np.outer(weights, weights)
    xpad, ypad = [np.pad(arr, radius, mode='symmetric') for arr in (a, b)]
    pad = (window-1)//2
    values = []
    norm = window**2/(window**2-1) if covariance else 1
    for i in range(pad, a.shape[0]-pad):
        for j in range(pad, a.shape[1]-pad):
            x, y = [arr[i:i+2*radius+1, j:j+2*radius+1] for arr in (xpad, ypad)]
            mx, my = np.sum(weights*x), np.sum(weights*y)
            vx, vy = [np.sum(weights*(arr-mean)**2)*norm for arr, mean in ((x, mx), (y, my))]
            cross = np.sum(weights*(x-mx)*(y-my))*norm
            values.append(((2*mx*my+.01**2)*(2*cross+.03**2))/((mx**2+my**2+.01**2)*(vx+vy+.03**2)))
    params = dict(gaussian_weights=True, sigma=sigma, win_size=window, use_sample_covariance=covariance)
    got = restoration_metrics(a, b, 1, params, ['ssim'])
    assert got['ssim'] == pytest.approx(np.mean(values), abs=1e-12)


def test_aliases_are_semantic_and_normalization_is_idempotent():
    original = restoration_spec()
    original['ssim_parameters'] = dict(window_size=5, K1=.02)
    before = copy.deepcopy(original)
    spec = normalize_metric_spec(original)
    assert original == before
    assert spec['ssim_parameters']['win_size'] == 5
    assert 'window_size' not in spec['ssim_parameters']
    assert normalize_metric_spec(spec) == spec
    gaussian = normalize_ssim_parameters(dict(gaussian_weights=True))
    assert gaussian['win_size'] == 11
    assert normalize_ssim_parameters(gaussian) == gaussian


@pytest.mark.parametrize('params', [
    dict(window_size=7, win_size=5), dict(window_size=4), dict(window_size=True),
    dict(gaussian_weights='false'), dict(use_sample_covariance=1), dict(K1=0),
    dict(unknown_parameter=True), dict(sigma=1), dict(gaussian_weights=True, sigma=float('inf')),
])
def test_unsupported_ssim_settings_fail_before_scoring(params):
    with pytest.raises(ValueError):
        normalize_ssim_parameters(params)


def test_group_weighting_and_corner_weighting():
    spec = restoration_spec(secondary=())
    spec['restoration_evaluation'].update(crop_size=2, groups=['a', 'b'], aggregation='mean_groups')
    samples = []
    for group, error in [('a', 1), ('a', 3), ('b', 20)]:
        pred = np.full((4, 4), float(error))
        pred[2:, 2:] *= 2
        samples.append(dict(prediction=pred, target=np.zeros((4, 4)), group=group))
    got = score_samples(iter(samples), spec)
    image_scores = [np.mean([20*math.log10(255/error)]*3 + [20*math.log10(255/(2*error))]) for error in [1, 3, 20]]
    assert got['metric'] == pytest.approx((np.mean(image_scores[:2])+image_scores[2])/2)
    assert got['metrics']['psnr/a'] == pytest.approx(np.mean(image_scores[:2]))
    assert got['n'] == 3
    spec['restoration_evaluation']['aggregation'] = 'mean_images'
    assert score_samples(samples, spec)['metric'] == pytest.approx(np.mean(image_scores))


def test_requested_metrics_only_allow_small_psnr_and_perfect_ssim():
    sample = dict(prediction=np.ones((2, 2)), target=np.zeros((2, 2)))
    got = score_samples([sample], restoration_spec(secondary=()))
    assert set(got['metrics']) == {'psnr'}
    sample = dict(prediction=np.zeros((8, 8)), target=np.zeros((8, 8)))
    got = score_samples([sample], restoration_spec('ssim', ()))
    assert got['metrics'] == {'ssim': 1.0}
    with pytest.raises(ValueError, match='infinite'):
        score_samples([sample], restoration_spec())


def test_prediction_clipping_and_format_check_do_not_require_truth():
    spec = restoration_spec('ssim', ())
    pred = np.full((8, 8), 300.)
    assert validate_prediction(dict(prediction=pred, input_shape=(8, 8)), spec) == {'valid': True}
    assert score_samples([dict(prediction=pred, target=np.full((8, 8), 255.))], spec)['metric'] == 1.0
    with pytest.raises(ValueError, match='target'):
        score_samples([dict(prediction=pred, target=pred)], spec)
    with pytest.raises(ValueError, match='input_shape'):
        validate_prediction(dict(prediction=pred, input_shape=(9, 8)), spec)


@pytest.mark.parametrize('family', ['restoration', 'segmentation'])
def test_arrays_are_consumed_incrementally(family):
    references = []

    def samples():
        for _ in range(20):
            prediction, target = np.ones((8, 8), dtype=int), np.zeros((8, 8), dtype=int)
            references.extend([weakref.ref(prediction), weakref.ref(target)])
            # The previous loop sample and newly yielded sample may overlap in
            # lifetime. A whole-split list would retain 6+ arrays on the third row.
            assert sum(ref() is not None for ref in references) <= 4
            yield dict(prediction=prediction, target=target)

    spec = restoration_spec(secondary=()) if family == 'restoration' else dict(primary_metric='dice')
    assert score_samples(samples(), spec)['n'] == 20


@pytest.mark.parametrize('change', [
    dict(data_range=float('nan')), dict(crop_size=0), dict(groups=['a', 'a']),
    dict(aggregation='median'), dict(unknown=True), dict(crop_mode='whole_image', crop_size=7),
])
def test_invalid_restoration_settings(change):
    spec = restoration_spec()
    spec['restoration_evaluation'].update(change)
    with pytest.raises(ValueError):
        normalize_metric_spec(spec)


def test_groups_must_be_declared_and_complete():
    spec = restoration_spec('ssim', ())
    sample = dict(prediction=np.zeros((8, 8)), target=np.zeros((8, 8)), group='a')
    with pytest.raises(ValueError, match='undeclared'):
        score_samples([sample], spec)
    spec['restoration_evaluation'].update(groups=['a', 'b'], aggregation='mean_groups')
    with pytest.raises(ValueError, match='all and only'):
        score_samples([sample], spec)


def reference_aji(truth, prediction):
    intersection_sum = union_sum = 0
    used = set()
    for label in sorted(set(truth.ravel()) - {0}):
        a = truth == label
        best = (0, None, 0, int(a.sum()))
        for other in sorted(set(prediction.ravel()) - {0}):
            b = prediction == other
            intersection, union = np.count_nonzero(a & b), np.count_nonzero(a | b)
            if intersection / union > best[0]:
                best = (intersection / union, other, intersection, union)
        _, other, intersection, union = best
        intersection_sum += intersection
        union_sum += union
        if other is not None:
            used.add(other)
    for other in set(prediction.ravel()) - {0} - used:
        union_sum += np.count_nonzero(prediction == other)
    return intersection_sum / union_sum if union_sum else 1.0


def test_instance_ids_and_official_aji_reuse_match_independent_reference():
    spec = dict(primary_metric='aji', secondary_metrics=['dice'])
    rng = np.random.default_rng(48)
    samples = [dict(prediction=rng.integers(0, 5, (9, 13)), target=rng.integers(0, 7, (9, 13))) for _ in range(10)]
    samples.append(dict(target=np.array([[0, 1, 1, 2, 2]]), prediction=np.array([[0, 7, 7, 7, 7]])))
    expected = np.mean([reference_aji(row['target'], row['prediction']) for row in samples])
    result = score_samples(samples, spec)
    assert result['metric'] == pytest.approx(expected)
    assert set(result['metrics']) == {'aji', 'binary_dice'}


def test_hd95_matches_independent_surface_point_distance():
    truth = np.zeros((5, 7), dtype=int)
    pred = truth.copy()
    truth[1, 1:4] = 1
    pred[3, 2:5] = 1
    spacing = (2., 0.5)
    distances = cdist(np.argwhere(truth)*spacing, np.argwhere(pred)*spacing)
    reference = np.percentile(np.r_[distances.min(axis=0), distances.min(axis=1)], 95)
    sample = dict(target=truth, prediction=pred, spacing=spacing, prediction_spacing=spacing)
    got = score_samples([sample], dict(primary_metric='hd95'))
    assert got['metric'] == pytest.approx(reference)
    assert got['maximize'] is False
    sample['prediction_spacing'] = (1., 1.)
    with pytest.raises(ValueError, match='spacing'):
        score_samples([sample], dict(primary_metric='hd95'))


def test_unrequested_hd95_does_not_reject_empty_prediction():
    row = dict(target=np.ones((4, 4), dtype=int), prediction=np.zeros((4, 4), dtype=int))
    assert score_samples([row], dict(primary_metric='dice'))['metric'] == 0
    with pytest.raises(ValueError, match='infinite'):
        score_samples([row], dict(primary_metric='dice', secondary_metrics=['hd95']))
    with pytest.raises(ValueError, match='integer'):
        validate_prediction(dict(prediction=np.full((4, 4), 0.7)), dict(primary_metric='dice'))


def test_binary_auc_respects_probability_order_and_positive_class():
    spec = dict(primary_metric='auc', secondary_metrics=['accuracy'], task_type='binary',
                class_labels=['positive', 'negative'], positive_class='positive')
    y = np.array([1, 1, 0, 0])
    positive_prob = np.array([.9, .4, .7, .2])
    samples = [dict(target='positive' if t else 'negative', prediction=[p, 1-p]) for t, p in zip(y, positive_prob)]
    got = score_samples(samples, spec)
    assert got['metric'] == pytest.approx(roc_auc_score(y, positive_prob))
    assert got['metrics']['accuracy'] == .5
    for sample in samples:
        sample['prediction'] = sample['prediction'][:1]
    assert score_samples(samples, spec) == got


def test_multilabel_scores_and_single_class_accuracy_only():
    spec = dict(primary_metric='mean_roc_auc', secondary_metrics=['mean_binary_accuracy'], task_type='multilabel')
    y = np.array([[1, 0], [0, 1], [1, 1]])
    pred = np.array([[.8, .2], [.3, .4], [.4, .9]])
    rows = [dict(target=a, prediction=b) for a, b in zip(y, pred)]
    got = score_samples(rows, spec)
    assert got['metric'] == pytest.approx(roc_auc_score(y, pred, average='macro'))
    assert got['metrics']['mean_binary_accuracy'] == pytest.approx(np.mean((pred >= .5) == y))
    spec = dict(primary_metric='accuracy', task_type='binary', class_labels=[0, 1])
    rows = [dict(target=0, prediction=[.9, .1])]
    assert score_samples(rows, spec)['metric'] == 1
    spec['secondary_metrics'] = ['auc']
    with pytest.raises(ValueError, match='undefined'):
        score_samples(rows, spec)


def test_explicit_threshold_and_semantic_settings():
    spec = dict(primary_metric='accuracy', task_type='binary', class_labels=[0, 1], decision_threshold=.8)
    row = dict(target=0, prediction=[.3, .7])
    assert score_samples([row], spec)['metric'] == 1
    row['prediction'] = [.7]
    assert score_samples([row], spec)['metric'] == 1
    with pytest.raises(ValueError, match='Conflicting'):
        normalize_metric_spec(dict(spec, threshold=.7))
    for change in [dict(auc_average='micro'), dict(multi_class='ovo'), dict(aggregation='macro_classes')]:
        with pytest.raises(ValueError, match='Unsupported'):
            normalize_metric_spec(dict(spec, **change))
    for change in [dict(aji_matching='hungarian'), dict(hd95_percentile=90), dict(hd95_connectivity=2), dict(foreground='label_1'), dict(aggregation='pooled_pixels')]:
        with pytest.raises(ValueError, match='Unsupported'):
            normalize_metric_spec(dict(primary_metric='dice', **change))


def box(x, confidence=1, cls=0):
    return (cls, (x, 0, x+2, 2), confidence)


def test_detection_ap_duplicate_and_missed_ground_truth():
    # TP, FP duplicate, TP: all-points AP = .5*1 + .5*(2/3).
    samples = [dict(target=[box(0), box(5)], prediction=[box(0, .9), box(0, .8), box(5, .7)])]
    spec = dict(primary_metric='ap50', secondary_metrics=['recall'], recall_confidence_threshold=.8)
    result = score_samples(samples, spec)
    assert result['metric'] == pytest.approx(5/6)
    assert result['metrics']['recall'] == .5
    # Classes absent from GT are excluded; macro AP differs from micro recall.
    samples[0]['target'].append(box(10, cls=1))
    samples[0]['prediction'].append(box(50, .99, cls=9))
    result = score_samples(samples, spec)
    assert result['metric'] == pytest.approx(5/12)
    assert result['metrics']['recall'] == pytest.approx(1/3)


def test_detection_empty_prediction_and_invalid_contract():
    row = dict(target=[box(0)], prediction=[])
    assert score_samples([row], dict(primary_metric='ap50'))['metrics'] == {'ap50': 0}
    assert validate_prediction(dict(prediction=[]), dict(primary_metric='ap50'))['valid']
    for change in [dict(iou_threshold=.7), dict(ap_interpolation='coco'), dict(max_detections=100), dict(recall_average='macro')]:
        with pytest.raises(ValueError):
            normalize_metric_spec(dict(primary_metric='ap50', **change))


@pytest.mark.parametrize('sample', [dict(prediction=[(0, (0, 0, 0, 2), .5)]), dict(prediction=[(True, (0, 0, 2, 2), .5)]), dict(prediction=[box(0, float('nan'))])])
def test_invalid_canonical_detections(sample):
    with pytest.raises(ValueError):
        validate_prediction(sample, dict(primary_metric='ap50'))


def test_unknown_metric_mixed_families_and_direction_rejected():
    for spec in [dict(primary_metric='custom'), dict(primary_metric='dice', secondary_metrics=['auc']), dict(primary_metric='hd95', maximize=True)]:
        with pytest.raises(ValueError):
            normalize_metric_spec(spec)


def minimal_registered_case(name):
    family = metric_capabilities()['metrics'][name]['family']
    if family == 'image_restoration':
        return restoration_spec(name, ()), dict(prediction=np.ones((8, 8)), target=np.zeros((8, 8)))
    if family == 'segmentation':
        return dict(primary_metric=name), dict(prediction=np.ones((4, 4), dtype=int), target=np.ones((4, 4), dtype=int))
    if family == 'detection':
        return dict(primary_metric=name), dict(prediction=[box(0)], target=[box(0)])
    if family == 'regression':
        return dict(primary_metric=name, task_type='regression'), dict(prediction=1.5, target=1.0)
    return dict(primary_metric=name, task_type='multilabel'), dict(prediction=[.2, .8], target=[0, 1])


def test_capability_catalog_is_json_serializable_and_isolated():
    first = metric_capabilities()
    assert json.loads(json.dumps(first)) == first
    first['metrics']['psnr']['maximize'] = False
    first['families']['image_restoration']['protocol_fields']['ssim_parameters']['fields']['win_size']['default'] = 99
    second = metric_capabilities()
    assert second['metrics']['psnr']['maximize'] is True
    assert normalize_ssim_parameters()['win_size'] == 7
    assert second['extension_steps']


def test_legacy_registry_is_derived_from_executable_catalog():
    from biomaker.utils.metric_registry import SUPPORTED_METRICS, metric_info

    catalog = metric_capabilities()
    assert set(SUPPORTED_METRICS) == set(catalog['aliases'])
    for alias, canonical in catalog['aliases'].items():
        info = metric_info(alias)
        definition = catalog['metrics'][canonical]
        assert (info.name, info.maximize, info.task_family) == (canonical, definition['maximize'], definition['family'])
    assert metric_info('unknown_research_metric') is None


@pytest.mark.parametrize('name', list(metric_capabilities()['metrics']))
def test_every_registered_metric_has_an_executable_shared_kernel(name):
    catalog = metric_capabilities()
    definition = catalog['metrics'][name]
    family = catalog['families'][definition['family']]
    assert name in family['metrics']
    spec, sample = minimal_registered_case(name)
    samples = [sample]
    if definition['family'] == 'classification':
        # Both outcomes occur in every label so all registered AUCs are defined.
        samples.append(dict(target=[1, 0], prediction=[.8, .2]))
    normalized = normalize_metric_spec(spec)
    result = score_samples(samples, spec)
    assert result['maximize'] == definition['maximize']
    assert set(result['metrics']) == {name}
    for alias in definition['aliases']:
        assert normalize_metric_spec(dict(spec, primary_metric=alias)) == normalized
    for field_name, field in family['protocol_fields'].items():
        if 'fixed' in field:
            assert normalized[field_name] == field['fixed']
            with pytest.raises(ValueError):
                normalize_metric_spec(dict(spec, **{field_name: '__unsupported_setting__'}))
        assert field_name in family['allowed_contract_fields']
        for alias in field.get('aliases', []):
            assert alias in family['allowed_contract_fields']
            assert family['parameter_aliases'][alias] == field_name


def test_capability_contract_admission_and_crop_aliases():
    families = metric_capabilities()['families']
    assert {'threshold', 'decision_threshold', 'classification_threshold'} <= set(families['classification']['allowed_contract_fields'])
    assert {'data_range', 'crop_size', 'groups', 'ssim_parameters'} <= set(families['image_restoration']['allowed_contract_fields'])
    assert 'unknown_parameter' not in families['detection']['allowed_contract_fields']
    spec = restoration_spec(secondary=())
    spec['restoration_evaluation']['crop_size'] = 4
    canonical = normalize_metric_spec(spec)
    assert canonical['restoration_evaluation']['crop_mode'] == 'four_corner_crops_no_center'
    for mode in ['four_corners', 'four_corner_crops_no_center']:
        altered = copy.deepcopy(spec)
        altered['restoration_evaluation']['crop_mode'] = mode
        assert normalize_metric_spec(altered) == canonical
    assert normalize_metric_spec(restoration_spec())['restoration_evaluation']['crop_mode'] == 'whole_image'
