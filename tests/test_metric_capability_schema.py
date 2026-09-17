"""The public catalog separates executable declarations from reference prose."""

import copy
import json

import pytest

from biomaker.utils.metric_declarations import validate_declared_capabilities
from biomaker.utils.trusted_metrics import metric_capabilities, normalize_metric_spec


@pytest.mark.parametrize('family', list(metric_capabilities()['families']))
def test_published_examples_are_accepted_by_declaration_and_scoring_contracts(family):
    catalog = metric_capabilities()
    example = catalog['families'][family]['metric_contract_example']
    before = copy.deepcopy(example)
    validate_declared_capabilities({'metric_contract': example}, '')
    normalized = normalize_metric_spec(example)
    assert normalized['task_family'] == family
    assert example == before
    assert set(example) <= set(catalog['families'][family]['allowed_contract_fields'])


def test_reference_prose_is_not_presented_as_settable_protocol():
    catalog = metric_capabilities()
    for family in catalog['families'].values():
        assert not {'empty_policy', 'matching', 'score_scope', 'unsupported'} & set(family)
        assert not set(family['documentation']) & set(family['allowed_contract_fields'])
        assert family['sample_schema']['prediction']
        assert family['sample_schema']['target']
    segmentation = catalog['families']['segmentation']
    assert 'Both empty:' in segmentation['documentation']['empty_policy']
    assert segmentation['protocol_fields']['aji_matching']['fixed'] == 'per_gt_best_iou'
    assert 'matching' in catalog['families']['detection']['documentation']


def test_agent_facing_examples_and_documentation_remain_independent_json_values():
    catalog = metric_capabilities()
    assert json.loads(json.dumps(catalog)) == catalog
    catalog['families']['regression']['metric_contract_example']['primary_metric'] = 'custom'
    catalog['families']['segmentation']['documentation']['empty_policy'] = 'changed'
    fresh = metric_capabilities()
    assert fresh['families']['regression']['metric_contract_example']['primary_metric'] == 'mae'
    assert 'Both empty:' in fresh['families']['segmentation']['documentation']['empty_policy']
