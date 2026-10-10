"""Metric aliases and parameters enter the shared executable kernel contract."""
import json

import pytest

from biomaker.utils.evaluator_protocol import metric_definitions, metric_name
from biomaker.utils import evaluator_setup
from biomaker.utils.metric_declarations import canonical_metric_contract
from biomaker.utils.metric_registry import metric_info
from biomaker.utils.trusted_metrics import metric_capabilities, normalize_metric_spec


@pytest.mark.parametrize('name', ['auc', 'AUROC', 'macro_auroc', 'macro_one_vs_rest_auroc', 'macro_ovr_roc_auc', 'macro_roc_auc_ovr', 'Macro one-vs-rest AUROC', 'macro_auroc_one_versus_rest'])
def test_precise_macro_ovr_aliases_share_canonical_contract(name):
    assert metric_name(name) == 'auc'
    assert metric_info(name) == metric_info('auc')
    assert canonical_metric_contract({'primary_metric': name})['primary_metric'] == 'auc'
    assert metric_definitions({'metrics': [{'name': name, 'maximize': True}]}) == [{'name': 'auc', 'maximize': True}]


@pytest.mark.parametrize('name', ['micro_auc', 'weighted_auroc', 'macro_ovo_auroc', 'macro_one_vs_one_auroc', 'macro_one_vs_rest_weighted_auroc', 'auc_pr', 'balanced_accuracy', 'top_5_accuracy', 'custom_auroc'])
def test_non_equivalent_names_are_never_replaced_by_another_metric(name):
    assert metric_name(name) == name
    assert metric_info(name) is None
    with pytest.raises((ValueError, KeyError)):
        normalize_metric_spec({'metrics': [{'name': name, 'maximize': True}], 'task_type': 'binary', 'class_labels': [0, 1]})


@pytest.mark.parametrize('direction', [None, 0, 1, 'false', 'true', [], {}])
def test_direction_is_boolean_not_coerced(direction):
    with pytest.raises(ValueError, match='boolean'):
        metric_definitions({'metrics': [{'name': 'auc', 'maximize': direction}]})


def test_direction_conflicts_and_mixed_families_are_rejected():
    with pytest.raises(ValueError, match='direction'):
        metric_definitions({'metrics': [{'name': 'auc', 'maximize': False}]})
    with pytest.raises(ValueError, match='famil'):
        metric_definitions({'metrics': ['auc', 'binary_dice']})


def test_named_metrics_keep_every_objective_and_direction():
    assert metric_definitions({'metrics': ['binary_dice', 'hd95']}) == [
        {'name': 'binary_dice', 'maximize': True}, {'name': 'hd95', 'maximize': False}]
    with pytest.raises(ValueError, match='task_family'):
        normalize_metric_spec({'metrics': ['auc'], 'task_family': 'segmentation', 'task_type': 'binary', 'class_labels': [0, 1]})


def test_aliases_live_in_the_executable_catalog():
    capabilities = metric_capabilities()
    assert capabilities['aliases']['macro_one_vs_rest_auroc'] == 'auc'
    assert capabilities['metrics']['auc']['maximize'] is True
    assert capabilities['metrics']['hd95']['maximize'] is False


def test_semantics_and_manifest_participate_in_cache_identity(tmp_path):
    path = tmp_path / 'input/biomaker_splits/split_manifest.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'metric_contract': {'metrics': ['auc']}}))
    first = evaluator_setup._stable_evaluator_cache_key({}, 'synthetic', tmp_path, None)
    path.write_text(json.dumps({'metric_contract': {'metrics': ['accuracy']}}))
    assert first != evaluator_setup._stable_evaluator_cache_key({}, 'synthetic', tmp_path, None)
    meta = evaluator_setup._evaluator_cache_metadata({}, 'synthetic', tmp_path, None)
    for name in ['trusted_metrics.py', 'evaluator_protocol.py', 'prediction_contract.py', 'metric_registry.py', 'score_validity.py']:
        assert meta['code_fingerprints'][name]
    assert meta['cache_version'] == 8
