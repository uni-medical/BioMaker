"""Trusted, file-format-independent metric kernels for frozen evaluators.

Adapters supply one mapping per sample: ``prediction``, ``target`` and optional
``group``, ``input_shape``, ``spacing`` and ``prediction_spacing``. Restoration
arrays are grayscale 2D intensities; segmentation arrays are 2D/3D nonnegative
integer labels (zero is background). Classification predictions are probability
vectors in class_labels order; binary one-column probabilities refer to
positive_class. Regression predictions and targets are finite numeric scalars
or fixed-width vectors. Detection boxes are (class_id, (x1, y1, x2, y2), confidence).

This module and its segmentation dependency are bundled by metric_source into
the frozen evaluator, so scoring does not depend on a mutable installed package.
Only requested metrics are computed. Undefined requested metrics raise rather
than silently changing the protocol or dropping samples.
"""

import copy
import math
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Lock
from time import perf_counter
from collections.abc import Mapping

import numpy as np
from scipy import ndimage as ndi

from .segmentation_metrics import (segmentation_score_case, validate_tree_sample,
                                   normalize_segmentation_parameters, _seg_score_validated_case)


class PredictionValidationError(ValueError):
    """A violation demonstrated using only public prediction/input fields."""


class UndefinedMetricError(ValueError):
    """Valid canonical inputs for which the requested metric is undefined."""

    def __init__(self, metric, reason):
        self.metric = str(metric)
        self.reason = str(reason)
        super().__init__('Requested %s is undefined: %s' % (self.metric, self.reason))


_METRIC_DEFINITIONS = {
    'mae': dict(family='regression', maximize=False, aliases=['mean_absolute_error'],
                description='Mean absolute error, equally weighted across samples and output dimensions; no rounding.'),
    'mse': dict(family='regression', maximize=False, aliases=['mean_squared_error'],
                description='Mean squared error, equally weighted across samples and output dimensions; no rounding.'),
    'rmse': dict(family='regression', maximize=False, aliases=['root_mean_squared_error'],
                 description='Square root of the mean squared error over all samples and output dimensions; no rounding.'),
    'psnr': dict(family='image_restoration', maximize=True, aliases=[],
                 description='10 log10(data_range squared / per-crop MSE); zero MSE raises.'),
    'ssim': dict(family='image_restoration', maximize=True, aliases=[],
                 description='Local SSIM, valid-interior mean, declared uniform/Gaussian window and covariance.'),
    'ap50': dict(family='detection', maximize=True, aliases=['map@0.5', 'map@0.50', 'map50'],
                 description='Class-aware IoU .5 matching and VOC all-points AP, macro over ground-truth classes.'),
    'recall': dict(family='detection', maximize=True, aliases=[],
                   description='Micro detection recall at the declared confidence threshold and IoU .5.'),
    'accuracy': dict(family='classification', maximize=True, aliases=['acc'],
                     description='Mean correct single-label predictions, or mean binary decisions for multilabel.'),
    'auc': dict(family='classification', maximize=True,
                aliases=['roc_auc', 'auroc', 'macro_auc', 'macro_roc_auc', 'ovr_auc', 'macro_auroc', 'macro_auroc_ovr', 'macro_roc_auc_ovr', 'macro_ovr_roc_auc', 'macro_one_vs_rest_auroc', 'macro one-vs-rest auroc', 'macro_auroc_one_versus_rest'],
                description='Binary ROC AUC, macro OVR multiclass AUC, or macro multilabel ROC AUC.'),
    'mean_roc_auc': dict(family='classification', maximize=True, aliases=[],
                         description='Macro binary ROC AUC for multilabel classification.', task_types=['multilabel']),
    'mean_binary_accuracy': dict(family='classification', maximize=True, aliases=[],
                                 description='Mean per-label binary accuracy for multilabel classification.', task_types=['multilabel']),
    'binary_dice': dict(family='segmentation', maximize=True, aliases=['dice', 'mean_dice'],
                        description='Per-image foreground Dice, then image mean; both-empty score 1.'),
    'binary_iou': dict(family='segmentation', maximize=True, aliases=['iou', 'jaccard'],
                       description='Per-image foreground intersection over union, then image mean; both-empty score 1.'),
    'aji': dict(family='segmentation', maximize=True, aliases=['mean_aji'],
                description='Official MoNuSeg per-GT best-IoU AJI, prediction reuse allowed; then image mean.'),
    'hd95': dict(family='segmentation', maximize=False, aliases=['hausdorff95', 'hausdorff_95'],
                 description='95th percentile of concatenated bidirectional surface distances, then image mean.'),
}
# Metric identity describes the mathematical quantity, never a dataset.
_SEGMENTATION_TREE_METRICS = {
    'cldice': (True, 'Centerline Dice against a supplied target skeleton, raw prediction.'),
    'tree_length_detected': (True, 'Fraction of target skeleton voxels covered by the prediction.'),
    'branch_detected': (True, 'Fraction of parsed target branches meeting the skeleton coverage threshold.'),
    'betti_error': (False, 'Absolute connected-component count difference from the target or declared one-component reference.'),
    'multiclass_dice': (True, 'Macro Dice over nonzero classes present in the target; no target classes scores zero.'),
    'multiclass_cldice': (True, 'Macro centerline Dice over target classes with target skeleton support; no supported classes scores zero.'),
    'branch_accuracy': (True, 'Accuracy of majority-vote semantic labels on target branches.'),
    'branch_f1': (True, 'Macro F1 over nonzero semantic classes present in target branches.'),
    'subtree_consistency': (True, 'Fraction of nontrivial target subtrees assigned one consistent predicted label.'),
    'topological_distance': (False, 'Mean nearest graph-hop distance to a target branch with the predicted semantic label.'),
}
for _name, (_maximize, _description) in _SEGMENTATION_TREE_METRICS.items():
    _METRIC_DEFINITIONS[_name] = dict(family='segmentation', maximize=_maximize,
                                     aliases=[], description=_description + ' Equal-image mean.')

_COMMON_PROTOCOL_FIELDS = {
    'metrics': dict(required=True, description='Nonempty list of {name, maximize}; all objectives have equal rank weight.'),
    'primary_metric': dict(description='Legacy first metric; converted into metrics for new runs.'),
    'secondary_metrics': dict(default=[], description='Legacy additional metrics; all become equal objectives for new runs.'),
    'maximize': dict(description='Legacy first metric direction. New declarations use maximize inside each metrics entry.'),
    'task_family': dict(derived='declared metrics family', description='An explicitly supplied family must agree with all objectives.'),
    'task_type': dict(description='Family-specific task subtype; classification requires binary/multiclass/multilabel.'),
}
_SSIM_PROTOCOL_FIELDS = {
    'win_size': dict(aliases=['window_size'], default=7, type='odd integer >= 3',
                     derived_default='For Gaussian weights: 2 * int(3.5 * sigma + 0.5) + 1.'),
    'gaussian_weights': dict(default=False, type='boolean'),
    'use_sample_covariance': dict(default=True, type='boolean'),
    'K1': dict(default=0.01, type='finite positive number'),
    'K2': dict(default=0.03, type='finite positive number'),
    'sigma': dict(default=1.5, type='finite positive number', requires='gaussian_weights=true',
                  description='Gaussian truncate is fixed at 3.5; sigma is omitted for uniform weights.'),
}
_RESTORATION_PROTOCOL_FIELDS = {
    'data_range': dict(required=True, type='finite positive number'),
    'crop_size': dict(required=True, type='null or positive integer',
                      description='Null means whole image; integer means four equal-weight corner crops, no center.'),
    'crop_mode': dict(derived='whole_image if crop_size=null, otherwise four_corner_crops_no_center',
                      values=['whole_image', 'four_corner_crops_no_center'],
                      value_aliases={'four_corners': 'four_corner_crops_no_center'}),
    'groups': dict(required=True, type='null or nonempty list of unique strings'),
    'aggregation': dict(required=True, values=['mean_images', 'mean_groups'],
                        description='Average corners within images first; mean_groups requires all declared groups.'),
    'group_column': dict(type='optional public metadata column', description='Adapter maps this to canonical sample group.'),
    'input_column': dict(type='optional public input column', description='Data adapter metadata, no prescribed name.'),
    'target_column': dict(type='optional hidden target column', description='Data adapter metadata, no prescribed name.'),
    'ssim_parameters': dict(type='mapping', fields=_SSIM_PROTOCOL_FIELDS,
                            description='Equivalent nested/top-level declarations accepted; conflicts rejected.'),
}
_FAMILY_CAPABILITIES = {
    'regression': {
        'sample_schema': {
            'prediction': 'Finite real numeric scalar or nonempty 1D vector; no rounding, clipping, or discretization.',
            'target': 'Finite real numeric value with exactly the same shape as prediction; shape is fixed across samples.',
        },
        'protocol_fields': {
            'target_shape': dict(type='optional [] for a scalar or [positive integer] for a vector',
                                 description='When omitted, infer from the first sample and require the same shape throughout the split.'),
            'aggregation': dict(fixed='mean_samples'),
            'multioutput': dict(fixed='uniform_average',
                                description='Equal output weights. RMSE takes the root after averaging squared errors across samples and outputs.'),
            'rounding': dict(fixed='none'),
            'units': dict(type='optional nonempty string',
                          description='Identity metadata for target/prediction units; no scaling is performed. MAE/RMSE have these units; MSE has squared units.'),
        },
        'task_types': ['regression'],
        'score_scope': 'stream samples; equal sample and output weights',
        'unsupported': ['sample weights', 'output weights', 'target rounding', 'variable-length outputs'],
    },
    'classification': {
        'sample_schema': {
            'prediction': '1D finite probability vector in class_labels order; binary length 1 means positive_class probability.',
            'target': 'Declared scalar class label, or same-length binary vector for multilabel.',
        },
        'protocol_fields': {
            'class_labels': dict(required_for=['binary', 'multiclass'], type='unique labels in probability-column order'),
            'positive_class': dict(derived_default='last class_labels entry', description='Must be a declared class.'),
            'threshold': dict(default=0.5, aliases=['decision_threshold', 'classification_threshold'],
                              type='finite number in [0,1]', description='Binary/multilabel decision threshold; multiclass always uses argmax.'),
            'auc_average': dict(fixed='macro'),
            'multi_class': dict(fixed='ovr'),
            'aggregation': dict(fixed_by_task_type={'binary': 'mean_samples', 'multiclass': 'mean_samples', 'multilabel': 'mean_labels'}),
        },
        'task_types': ['binary', 'multiclass', 'multilabel'],
        'score_scope': 'entire split',
    },
    'segmentation': {
        'sample_schema': {
            'prediction': '2D/3D nonnegative integer labels; 0 background; preserve instance IDs.',
            'target': 'Same-shape nonnegative integer labels.',
            'input_shape': 'Optional exact prediction shape.',
            'spacing': 'Optional positive spacing per array axis; default unit spacing.',
            'prediction_spacing': 'Optional spacing matching supplied target/input spacing within absolute 1e-4.',
            'skeleton': 'Optional supplied target skeleton, required by centerline and branch-detection metrics.',
            'parsing': 'Optional supplied target branch IDs, contiguous 1..N, zero background.',
            'graph': 'Optional integer (2,E) directed parent-to-child tree edges, node = parsing ID - 1.',
            'input_geometry': 'Optional original size/spacing/origin/direction in SimpleITK XYZ convention.',
            'prediction_geometry': 'Required alongside input_geometry; must preserve the original grid.',
            'target_geometry': 'Required when scoring with geometry metadata.',
            'skeleton_geometry': 'Geometry of requested target skeleton when geometry is supplied.',
            'parsing_geometry': 'Geometry of requested target branch parsing when geometry is supplied.',

        },
        'protocol_fields': {
            'foreground': dict(fixed='nonzero'),
            'aggregation': dict(fixed='mean_images'),
            'aji_matching': dict(fixed='per_gt_best_iou'),
            'hd95_percentile': dict(fixed=95),
            'hd95_connectivity': dict(fixed=1),
            'hd95_empty_policy': dict(fixed='error_if_one_empty'),
            'spacing_order': dict(fixed='array_axes'),
            'dice_smooth': dict(default=0.0, type='finite nonnegative number', description='Add to Dice numerator and denominator, binary and multiclass.'),
            'topology_smooth': dict(default=0.0, type='finite nonnegative number', description='Add to tree-length coverage numerator and denominator.'),
            'branch_coverage_threshold': dict(default=0.8, type='finite number in [0,1]'),
            'topology_component_policy': dict(default='largest', values=['largest', 'raw'], description='Applied to coverage and branch-label projection only; Dice/clDice remain raw.'),
            'connectivity': dict(default=3, values=[1, 2, 3], description='Connectivity for components; capped by array dimension.'),
            'betti_reference': dict(default='target', values=['target', 'one']),
            'label_ids': dict(default=None, type='null or nonempty unique nonnegative integer labels', description='Optional declared label set; no dataset-specific upper limit.'),
            'subtree_ignore_labels': dict(default=[], type='unique nonnegative integer labels'),
            'missing_label_penalty': dict(default=50.0, type='finite nonnegative number'),
            'class_aggregation': dict(fixed='gt_present'),
            'reference_protocol': dict(type='optional nonempty reference provenance string'),
            'image_column': dict(type='optional public image path column'),
            'target_column': dict(type='optional private original mask column'),
            'auxiliary_columns': dict(type='optional mapping from skeleton/parsing/graph to private answer columns'),
            'geometry_policy': dict(default='shape_spacing', values=['shape_spacing', 'physical'], description='Built-in reader checks shape/spacing or full medical geometry, respectively.'),

        },
        'score_scope': 'stream samples; average per-image scores',
        'empty_policy': 'Both empty: Dice/IoU/AJI 1, HD95 0. Exactly one empty: Dice/IoU/AJI 0; requested HD95 raises.',
    },
    'image_restoration': {
        'sample_schema': {
            'prediction': '2D finite numeric grayscale intensities; harness clips to [0,data_range].',
            'target': 'Same-shape finite grayscale intensities in [0,data_range].',
            'input_shape': 'Optional exact prediction shape.',
            'group': 'Required canonical declared group value when groups is non-null.',
        },
        'protocol_fields': {
            'restoration_evaluation': dict(required=True, type='mapping', fields=_RESTORATION_PROTOCOL_FIELDS),
            'ssim_parameters': dict(type='mapping', fields=_SSIM_PROTOCOL_FIELDS),
            'prediction_clipping': dict(derived='[0,data_range]', description='Any conflicting clipping policy is rejected.'),
            'zero_mse_policy': dict(fixed='error: PSNR is infinite'),
        },
        'flattened_contract_fields': list(_RESTORATION_PROTOCOL_FIELDS),
        'score_scope': 'stream samples; mean corners per image, then declared image/group aggregation',
    },
    'detection': {
        'sample_schema': {
            'prediction': 'List of (nonnegative integer class_id, (x1,y1,x2,y2), confidence in [0,1]); finite positive-size boxes.',
            'target': 'Same canonical box tuples; confidence ignored. [] represents no boxes.',
        },
        'protocol_fields': {
            'iou_threshold': dict(fixed=0.5),
            'ap_interpolation': dict(fixed='voc_all_points'),
            'ap_average': dict(fixed='macro_over_gt_classes'),
            'recall_average': dict(fixed='micro'),
            'max_detections': dict(fixed=None),
            'recall_confidence_threshold': dict(default=0.0, type='finite number in [0,1]'),
        },
        'score_scope': 'entire split',
        'matching': 'Confidence descending; ties by sample then box order; highest-IoU GT; duplicate matches are false positives.',
        'unsupported': ['crowd', 'ignore', 'difficult', 'class-agnostic matching', 'COCO or 11-point interpolation'],
    },
}
_METRIC_ALIASES = {alias: name for name, definition in _METRIC_DEFINITIONS.items()
                   for alias in [name, *definition['aliases']]}
_METRIC_FAMILIES = {name: definition['family'] for name, definition in _METRIC_DEFINITIONS.items()}
SUPPORTED_STANDARD_METRICS = dict(_METRIC_FAMILIES)

_METRIC_CONTRACT_EXAMPLES = {
    'regression': dict(primary_metric='mae', units='months'),
    'classification': dict(primary_metric='auc', secondary_metrics=['accuracy'],
                           task_type='binary', class_labels=['negative', 'positive']),
    'segmentation': dict(primary_metric='aji', secondary_metrics=['binary_dice']),
    'image_restoration': dict(primary_metric='psnr', secondary_metrics=['ssim'],
                              restoration_evaluation=dict(data_range=1.0, crop_size=None,
                                                           groups=None, aggregation='mean_images')),
    'detection': dict(primary_metric='ap50', secondary_metrics=['recall']),
}


def metric_capabilities():
    """Return the executable registry's JSON-friendly public protocol catalog.

    Routing, Agent prompts and strict manifest-field admission should consume
    this catalog; adding an alias must not require changing an evaluator or an
    independent routing table. Returned data is isolated from the trusted source.
    """
    families = copy.deepcopy(_FAMILY_CAPABILITIES)
    for family, definition in families.items():
        # Keep reference prose out of the parameter namespace presented to Agents.
        definition['documentation'] = {name: definition.pop(name) for name in
                                       ('score_scope', 'empty_policy', 'matching', 'unsupported')
                                       if name in definition}
        definition['metric_contract_example'] = copy.deepcopy(_METRIC_CONTRACT_EXAMPLES[family])
        definition['metrics'] = [name for name, item in _METRIC_DEFINITIONS.items() if item['family'] == family]
        aliases = {alias: name for name, field in definition['protocol_fields'].items() for alias in field.get('aliases', [])}
        definition['parameter_aliases'] = aliases
        definition['allowed_contract_fields'] = sorted(set(_COMMON_PROTOCOL_FIELDS) | set(definition['protocol_fields']) |
                                                       set(aliases) | set(definition.get('flattened_contract_fields', [])))
    for definition in families.values():
        example = definition['metric_contract_example']
        example['metrics'] = normalize_metric_definitions(example)
        example.pop('primary_metric', None)
        example.pop('secondary_metrics', None)
    return dict(api_version=3, metrics=copy.deepcopy(_METRIC_DEFINITIONS),
                aliases=dict(_METRIC_ALIASES), common_protocol_fields=copy.deepcopy(_COMMON_PROTOCOL_FIELDS),
                families=families,
                catalog_usage=[
                    'metric_contract declares requested metrics and settings using common_protocol_fields and the family protocol_fields; allowed_contract_fields includes accepted aliases and flattened forms.',
                    'Fixed settings and defaults may be omitted; Harness resolves and freezes them. Explicit task requirements take precedence over illustrative examples.',
                    'metric_contract_example is a valid illustration, not a dataset prescription: choose the requested metrics and adapt data-specific values such as class labels, units and intensity range.',
                    'sample_schema describes canonical values returned by the data adapter. Agent chooses source columns, paths, file formats and data loading; the schema does not prescribe the original dataset layout.',
                    'documentation explains scoring behavior and is not a set of metric_contract fields. Optional notes/description carry explanatory prose; they cannot change scoring behavior or replace task-required settings.',
                ],
                unsupported_policy='Reject unknown semantic fields, conflicting aliases, unsupported settings and undefined requested metrics; never replace the scoring definition or drop samples.',
                extension_steps=[
                    'Register the canonical metric, direction, family, aliases and supported protocol in trusted_metrics.py.',
                    'Implement its kernel and wire it into score_samples; reuse the canonical family sample schema or document a new one.',
                    'Add independent reference tests, invalid-setting tests, requested-only tests and adapter integration coverage.',
                    'The shared harness freezes the registered protocol and code automatically; existing evaluator interfaces and data adapters remain reusable.',
                    'Truly unregistered research metrics use the explicit custom-evaluator extension; known metric schema misses require only a data adapter.',
                ])


def _fixed_protocol_values(family, task_type=None):
    result = {}
    for name, field in _FAMILY_CAPABILITIES[family]['protocol_fields'].items():
        if 'fixed' in field:
            result[name] = copy.deepcopy(field['fixed'])
        elif 'fixed_by_task_type' in field:
            result[name] = field['fixed_by_task_type'][task_type]
    return result


METRIC_API_DESCRIPTION = """Harness metric API v3: prediction_spec() declares a nonempty
metrics list of {name, maximize} objectives, and family settings below. All
objectives participate equally in average-rank selection outside the evaluator.
score() returns raw metrics and diagnostics; never ranks or a scalar aggregate.
The harness canonicalizes
known metric aliases and freezes all settings. load_sample returns a mapping with
prediction and, when target_row is provided, target; optional keys are group,
input_shape, spacing, prediction_spacing. Supply arrays/values, never metric code.
Regression: task_type='regression'; finite numeric scalar or nonempty 1D vector
prediction and same-shaped target. No rounding, clipping, or discretization.
Optional target_shape=[] declares a scalar, [N] a vector of N outputs; otherwise
the first sample determines the shape and every sample must match it.
MAE/MSE average absolute/squared errors equally over samples and outputs.
RMSE is the square root of that global MSE, not the mean of per-output roots.
aggregation='mean_samples', multioutput='uniform_average', rounding='none';
all three minimize. Optional units declares target/prediction units without scaling.
Restoration: 2D numeric grayscale arrays; restoration_evaluation requires
data_range>0, crop_size (null=whole image or positive integer=four corner crops),
groups (null or unique nonempty strings), aggregation (mean_images or mean_groups).
The harness clips predictions to [0,data_range], checks targets are in range,
averages corners per image, then images or groups equally. Declare group when
groups are used. Optional ssim_parameters accepts win_size or window_size (same
alias), gaussian_weights, use_sample_covariance, K1, K2, sigma. Defaults are
7x7 uniform, sample covariance, K1=.01, K2=.03; Gaussian defaults sigma=1.5,
truncate=3.5, win_size=11. Unknown/conflicting parameters raise. Perfect-image
PSNR is infinite and raises only when PSNR is requested; perfect SSIM is 1.
Segmentation: 2D/3D integer nonnegative label arrays, 0 background. binary_dice
and binary_iou (aliases iou/jaccard) use labels>0 and average per-image scores;
they do not compute class-wise multiclass scores. AJI uses supplied instance IDs with official per-GT best IoU
matching (prediction reuse allowed). HD95 uses the 95th percentile of concatenated
bidirectional surface distances, connectivity 1, optional spacing in array-axis
order (default unit spacing). Both-empty Dice/IoU/AJI=1, HD95=0; one-empty HD95 raises.
Additional segmentation metrics: cldice, multiclass_dice, multiclass_cldice,
tree_length_detected, branch_detected, betti_error, branch_accuracy, branch_f1,
subtree_consistency, topological_distance. All use the segmentation family;
betti_error and topological_distance minimize. Other objectives maximize.
Supply precomputed target skeleton/parsing/graph references only when requested;
never build target references from predictions. Ground-truth-present classes
have equal weight for multiclass metrics (empty class set gives zero).
Generic parameters include dice_smooth, topology_smooth, connectivity,
branch_coverage_threshold, topology_component_policy, betti_reference,
subtree_ignore_labels, missing_label_penalty, and optional label_ids.
No dataset's label range, trachea class, smoothing or reference component count
is implicit. Protocol defaults and supported values are in metric_capabilities.
Medical geometry metadata, when supplied, must match exactly within declared
numeric tolerances; automatic resampling is forbidden. auxiliary_columns maps
private skeleton/parsing/graph paths for the built-in reader. Agents implement
readers only; all metric formulas come from trusted kernels.
Classification: task_type binary/multiclass/multilabel. Single-label needs distinct
class_labels in prediction-probability order; target is a scalar class label.
Binary one-entry probability refers to positive_class (default final label).
Multilabel target/prediction are equal-length binary/probability vectors.
Accuracy uses argmax or >=threshold (default .5; decision_threshold and
classification_threshold are aliases). AUC uses auc_average='macro' and
multi_class='ovr'; other averaging/OVR settings are rejected. Single-label
aggregation='mean_samples'; multilabel aggregation='mean_labels'. Segmentation
supports only foreground='nonzero', aggregation='mean_images',
aji_matching='per_gt_best_iou', hd95_percentile=95, hd95_connectivity=1,
hd95_empty_policy='error_if_one_empty', spacing_order='array_axes'.
Detection: lists of (nonnegative integer class_id, (x1,y1,x2,y2), confidence),
positive-size boxes, confidence in [0,1]; target confidence is ignored. AP50 uses
IoU .5 and VOC all-points interpolation, macro over GT classes. Recall is micro
at recall_confidence_threshold (default 0). Score-descending, highest-IoU matching;
ties use expected sample order then box order. No crowd/ignore/max-detections.
Only requested metrics are computed. Missing/undefined/nonfinite requested
metrics raise rather than dropping samples. Prediction-only validation never
requires target; input_shape and supplied spacing must match predictions.
"""


def _metric_name(value):
    name = _METRIC_ALIASES.get(str(value).strip().lower())
    if name is None:
        raise ValueError('Unsupported trusted metric: %r' % value)
    return name


def _canonical_metric_parameters(values, fields):
    aliases = {alias: name for name, field in fields.items() for alias in field.get('aliases', [])}
    result = {}
    for key, value in values.items():
        key = aliases.get(key, key)
        field = fields.get(key, {})
        if isinstance(value, str):
            value = field.get('value_aliases', {}).get(value, value)
        if isinstance(value, Mapping) and 'fields' in field:
            value = _canonical_metric_parameters(value, field['fields'])
        if key in result and result[key] != value:
            raise ValueError('Conflicting metric protocol field: ' + key)
        result[key] = value
    return result


def normalize_metric_definitions(spec, *, allow_legacy_unknown=False):
    """Canonical objectives, shared with the portable frozen scoring runtime."""
    if not isinstance(spec, Mapping):
        raise ValueError('Metric specification must be a mapping.')
    legacy = 'metrics' not in spec
    if legacy:
        primary = spec.get('primary_metric')
        if isinstance(primary, Mapping):
            primary = dict(primary)
        else:
            primary = {'name': primary}
        if 'maximize' in spec:
            if 'maximize' in primary and primary['maximize'] != spec['maximize']:
                raise ValueError('Conflicting primary metric maximize direction.')
            primary['maximize'] = spec['maximize']
        secondary = spec.get('secondary_metrics', [])
        if not isinstance(secondary, list):
            raise ValueError('secondary_metrics must be a list.')
        items = [primary, *secondary]
    else:
        items = spec['metrics']
        if not isinstance(items, list) or not items:
            raise ValueError('metrics must be a nonempty list (at least one metric).')
    result = {}
    for index, raw in enumerate(items):
        item = {'name': raw} if isinstance(raw, str) else copy.deepcopy(raw)
        if not isinstance(item, dict):
            raise ValueError('Each metric must declare a nonempty name and boolean maximize.')
        identifiers = [item.pop(key) for key in ('name', 'metric', 'id') if key in item]
        if not identifiers or any(not isinstance(name, str) or not name.strip() for name in identifiers):
            raise ValueError('Each metric must declare a nonempty name.')
        names = {_METRIC_ALIASES.get(name.strip().lower().strip('`*_ .'), name.strip().lower().strip('`*_ .')) for name in identifiers}
        if len(names) != 1:
            raise ValueError('Conflicting metric name/metric/id aliases.')
        name = names.pop()
        # Preserve legacy kernel names for historical direct calls. New
        # objective lists use the same canonical aliases as the public API.
        if not legacy:
            name = {'mean_roc_auc': 'auc', 'mean_binary_accuracy': 'accuracy'}.get(name, name)
        definition = _METRIC_DEFINITIONS.get(name)
        direction = item.get('maximize', definition['maximize'] if definition else None)
        if type(direction) is not bool:
            if legacy and index > 0 and allow_legacy_unknown and direction is None:
                continue
            raise ValueError('Metric %r must explicitly declare boolean maximize.' % name)
        if definition and direction != definition['maximize']:
            raise ValueError('maximize conflicts with registered direction for metric %r.' % name)
        if definition:
            item = _canonical_metric_parameters(item, _FAMILY_CAPABILITIES[definition['family']]['protocol_fields'])
        item = {'name': name, **item, 'maximize': direction}
        if name in result and result[name] != item:
            raise ValueError('Conflicting duplicate metric declaration for %r.' % name)
        result[name] = item
    definitions = list(result.values())
    families = {_METRIC_DEFINITIONS[item['name']]['family'] for item in definitions if item['name'] in _METRIC_DEFINITIONS}
    if len(families) > 1:
        raise ValueError('Requested metrics require different canonical sample families; metrics must belong to one task family, and mixed-head evaluation is not supported.')
    if not legacy and 'primary_metric' in spec:
        old = {key: value for key, value in spec.items() if key != 'metrics'}
        old_definitions = normalize_metric_definitions(old, allow_legacy_unknown=allow_legacy_unknown)
        if {item['name']: item['maximize'] for item in old_definitions} != {item['name']: item['maximize'] for item in definitions}:
            raise ValueError('Legacy primary/secondary metrics disagree with metrics declarations.')
        for old_item in old_definitions:
            new_item = next(item for item in definitions if item['name'] == old_item['name'])
            for key in set(old_item) & set(new_item):
                if old_item[key] != new_item[key]:
                    raise ValueError('Conflicting legacy and metrics parameter declarations: ' + key)
    return definitions


def _finite_number(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(name + ' must be a finite number.')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + ' must be finite' + (' and positive.' if positive else '.'))
    return value


def normalize_ssim_parameters(parameters=None):
    """Normalize SSIM spelling, with explicit skimage-compatible semantics.

    Uniform defaults: 7x7, sample covariance, K1=.01, K2=.03. Gaussian
    weights use sigma=1.5 and truncate=3.5; omitted win_size follows the Gaussian
    radius (11 at sigma=1.5). win_size sets valid-mean cropping and covariance
    normalization, as in skimage. Unknown parameters are rejected.
    """
    if parameters is not None and not isinstance(parameters, Mapping):
        raise ValueError('ssim_parameters must be a mapping.')
    params = dict(parameters or {})
    for name, field in _SSIM_PROTOCOL_FIELDS.items():
        for alias in field.get('aliases', []):
            if alias in params:
                value = params.pop(alias)
                if name in params and params[name] != value:
                    raise ValueError('Conflicting SSIM %s and %s.' % (name, alias))
                params[name] = value
    allowed = set(_SSIM_PROTOCOL_FIELDS)
    unknown = set(params) - allowed
    if unknown:
        raise ValueError('Unsupported SSIM parameters: ' + ', '.join(sorted(unknown)))
    out = {name: field['default'] for name, field in _SSIM_PROTOCOL_FIELDS.items()
           if name not in ('win_size', 'sigma')}
    out.update(params)
    for key in ('gaussian_weights', 'use_sample_covariance'):
        if type(out[key]) is not bool:
            raise ValueError(key + ' must be a boolean.')
    for key in ('K1', 'K2'):
        out[key] = _finite_number(out[key], key, positive=True)
    if 'sigma' in out or out['gaussian_weights']:
        out['sigma'] = _finite_number(out.get('sigma', _SSIM_PROTOCOL_FIELDS['sigma']['default']), 'sigma', positive=True)
        if not out['gaussian_weights']:
            raise ValueError('sigma requires gaussian_weights=true.')
    out.setdefault('win_size', 2 * int(3.5 * out['sigma'] + 0.5) + 1 if out['gaussian_weights'] else _SSIM_PROTOCOL_FIELDS['win_size']['default'])
    window = out['win_size']
    if isinstance(window, bool) or not isinstance(window, int) or window < 3 or window % 2 != 1:
        raise ValueError('SSIM win_size must be an odd integer >= 3.')
    return out


def normalize_metric_spec(spec):
    """Return a canonical independent copy, rejecting unsupported semantics."""
    if not isinstance(spec, Mapping):
        raise ValueError('Metric specification must be a mapping.')
    out = copy.deepcopy(dict(spec))
    definitions = normalize_metric_definitions(out)
    names = [_metric_name(item['name']) for item in definitions]
    legacy = 'metrics' not in out and out.get('evaluator_api_version', 2) < 3
    if not legacy:
        out['metrics'] = definitions
        for key in ('primary_metric', 'secondary_metrics', 'maximize'):
            out.pop(key, None)
    else:
        out['primary_metric'] = names[0]
        out['secondary_metrics'] = names[1:]
        out['maximize'] = definitions[0]['maximize']
    # The existing kernels use shared family parameters. Lift per-metric
    # settings, rejecting conflicts instead of silently ignoring parameters.
    for item in definitions:
        for key, value in item.items():
            if key not in {'name', 'maximize'}:
                if key in out and out[key] != value:
                    raise ValueError('Conflicting metric protocol field: ' + key)
                out[key] = copy.deepcopy(value)
    family = _METRIC_FAMILIES[names[0]]
    allowed_parameters = set(_FAMILY_CAPABILITIES[family]['protocol_fields']) | set(_COMMON_PROTOCOL_FIELDS)
    for item in definitions:
        unknown_parameters = set(item) - {'name', 'maximize'} - allowed_parameters
        if unknown_parameters:
            raise ValueError('Unsupported metric parameters for %s: %s' % (item['name'], ', '.join(sorted(unknown_parameters))))
    if any(_METRIC_FAMILIES[name] != family for name in names):
        raise ValueError('Requested metrics must belong to one task family.')
    if out.get('task_family', family) != family:
        raise ValueError('task_family conflicts with requested metrics.')
    out['task_family'] = family
    if family == 'image_restoration':
        protocol = out.get('restoration_evaluation')
        required = {name for name, field in _RESTORATION_PROTOCOL_FIELDS.items() if field.get('required')}
        allowed = set(_RESTORATION_PROTOCOL_FIELDS)
        if not isinstance(protocol, Mapping) or required - set(protocol):
            raise ValueError('restoration_evaluation requires data_range, crop_size, groups, aggregation.')
        if set(protocol) - allowed:
            raise ValueError('Unsupported restoration parameters: ' + ', '.join(sorted(set(protocol) - allowed)))
        protocol = dict(protocol)
        protocol['data_range'] = _finite_number(protocol['data_range'], 'data_range', positive=True)
        crop = protocol['crop_size']
        if crop is not None and (isinstance(crop, bool) or not isinstance(crop, int) or crop <= 0):
            raise ValueError('crop_size must be null or a positive integer.')
        crop_field = _RESTORATION_PROTOCOL_FIELDS['crop_mode']
        mode = crop_field['value_aliases'].get(protocol.get('crop_mode'), protocol.get('crop_mode'))
        if mode is not None and mode not in crop_field['values']:
            raise ValueError('Unsupported restoration crop_mode.')
        if (mode == 'whole_image' and crop is not None) or (mode == 'four_corner_crops_no_center' and crop is None):
            raise ValueError('crop_mode conflicts with crop_size.')
        protocol['crop_mode'] = 'whole_image' if crop is None else 'four_corner_crops_no_center'
        groups = protocol['groups']
        if groups is not None and (not isinstance(groups, list) or not groups or
                any(not isinstance(group, str) or not group for group in groups) or len(set(groups)) != len(groups)):
            raise ValueError('groups must be null or a nonempty list of unique strings.')
        if protocol['aggregation'] not in ('mean_images', 'mean_groups') or (protocol['aggregation'] == 'mean_groups' and groups is None):
            raise ValueError('aggregation must be mean_images, or mean_groups with declared groups.')
        params = normalize_ssim_parameters(out.get('ssim_parameters', protocol.get('ssim_parameters')))
        if 'ssim_parameters' in out and 'ssim_parameters' in protocol and params != normalize_ssim_parameters(protocol['ssim_parameters']):
            raise ValueError('Conflicting top-level and restoration SSIM parameters.')
        if 'ssim' in names and crop is not None and crop < params['win_size']:
            raise ValueError('crop_size is smaller than the SSIM window.')
        protocol.pop('ssim_parameters', None)
        out['restoration_evaluation'] = protocol
        out['ssim_parameters'] = params
        clipping = out.get('prediction_clipping', [0, protocol['data_range']])
        if clipping != [0, protocol['data_range']]:
            raise ValueError('Trusted restoration scoring clips predictions to [0, data_range].')
        out['prediction_clipping'] = clipping
        policy = _FAMILY_CAPABILITIES[family]['protocol_fields']['zero_mse_policy']['fixed']
        if out.get('zero_mse_policy', policy) != policy:
            raise ValueError('Unsupported zero_mse_policy; requested infinite PSNR raises an error.')
        out['zero_mse_policy'] = policy
    elif family == 'regression':
        if out.get('task_type', 'regression') != 'regression':
            raise ValueError('Regression requires task_type regression.')
        out['task_type'] = 'regression'
        if 'target_shape' in out:
            shape = out['target_shape']
            if (not isinstance(shape, list) or len(shape) > 1 or
                    any(type(size) is not int or size <= 0 for size in shape)):
                raise ValueError('Regression target_shape must be [] for a scalar or [positive integer] for a vector.')
        if 'units' in out and (not isinstance(out['units'], str) or not out['units'].strip()):
            raise ValueError('Regression units must be a nonempty string when declared.')
        for key, value in _fixed_protocol_values(family).items():
            if out.get(key, value) != value:
                raise ValueError('Unsupported regression parameter %s: expected %r.' % (key, value))
            out[key] = value
    elif family == 'classification':
        task_type = out.get('task_type')
        if task_type not in _FAMILY_CAPABILITIES[family]['task_types']:
            raise ValueError('Classification requires task_type binary, multiclass, or multilabel.')
        labels = out.get('class_labels')
        if task_type != 'multilabel':
            if not isinstance(labels, list) or len(labels) < 2 or len(set(map(str, labels))) != len(labels):
                raise ValueError('Single-label classification requires distinct class_labels in probability order.')
            if task_type == 'binary' and len(labels) != 2:
                raise ValueError('Binary classification requires exactly two class_labels.')
            positive = out.get('positive_class', labels[-1])
            if str(positive) not in set(map(str, labels)):
                raise ValueError('positive_class must occur in class_labels.')
            out['positive_class'] = positive
        if any(task_type not in _METRIC_DEFINITIONS[name].get('task_types', [task_type]) for name in names):
            raise ValueError('Requested metric does not support this classification task_type.')
        threshold_field = _FAMILY_CAPABILITIES[family]['protocol_fields']['threshold']
        for alias in threshold_field['aliases']:
            if alias in out:
                value = out.pop(alias)
                if 'threshold' in out and out['threshold'] != value:
                    raise ValueError('Conflicting classification threshold aliases.')
                out['threshold'] = value
        threshold = _finite_number(out.get('threshold', threshold_field['default']), 'threshold')
        if not 0 <= threshold <= 1:
            raise ValueError('Classification threshold must be in [0,1].')
        if task_type == 'multiclass' and threshold != 0.5:
            raise ValueError('Multiclass accuracy uses argmax; a probability threshold is unsupported.')
        out['threshold'] = threshold
        fixed = _fixed_protocol_values(family, task_type)
        for key, value in fixed.items():
            if out.get(key, value) != value:
                raise ValueError('Unsupported classification parameter %s: expected %r.' % (key, value))
            out[key] = value
    elif family == 'segmentation':
        fixed = _fixed_protocol_values(family)
        for key, value in fixed.items():
            if out.get(key, value) != value:
                raise ValueError('Unsupported segmentation parameter %s: expected %r.' % (key, value))
            out[key] = value
        parameters = normalize_segmentation_parameters(out)
        out.setdefault('geometry_policy', 'shape_spacing')
        if out['geometry_policy'] not in ('shape_spacing', 'physical'):
            raise ValueError('geometry_policy must be shape_spacing or physical.')
        for key in ('dice_smooth', 'topology_smooth', 'branch_coverage_threshold',
                    'topology_component_policy', 'connectivity', 'betti_reference',
                    'label_ids', 'subtree_ignore_labels', 'missing_label_penalty'):
            out[key] = parameters[key]
        for key in ('reference_protocol', 'image_column', 'target_column'):
            if key in out and (not isinstance(out[key], str) or not out[key].strip()):
                raise ValueError(key + ' must be a nonempty string.')
        auxiliary = out.get('auxiliary_columns', {})
        if not isinstance(auxiliary, Mapping) or set(auxiliary) - {'skeleton', 'parsing', 'graph'}:
            raise ValueError('auxiliary_columns must map skeleton/parsing/graph to private columns.')
        if any(not isinstance(value, str) or not value for value in auxiliary.values()) or len(set(auxiliary.values())) != len(auxiliary):
            raise ValueError('Auxiliary column names must be distinct nonempty strings.')
        if out.get('target_column') in auxiliary.values():
            raise ValueError('Original target and auxiliary references require separate columns.')
    elif family == 'detection':
        fixed = _fixed_protocol_values(family)
        for key, value in fixed.items():
            if out.get(key, value) != value:
                raise ValueError('Unsupported detection parameter %s: expected %r.' % (key, value))
            out[key] = value
        threshold_default = _FAMILY_CAPABILITIES[family]['protocol_fields']['recall_confidence_threshold']['default']
        threshold = _finite_number(out.get('recall_confidence_threshold', threshold_default), 'recall_confidence_threshold')
        if not 0 <= threshold <= 1:
            raise ValueError('recall_confidence_threshold must be in [0,1].')
        out['recall_confidence_threshold'] = threshold
    return out


def _numeric_array(value, name, kinds='buif'):
    array = np.asarray(value)
    if array.dtype.kind not in kinds or not array.size or not np.isfinite(array).all():
        raise ValueError(name + ' must be a nonempty finite real numeric array.')
    return array


def restoration_crops(array, crop_size, crop_mode=None):
    """Whole image or four equally weighted corner crops (no center)."""
    if array.ndim != 2:
        raise ValueError('Restoration requires two-dimensional grayscale arrays.')
    if crop_mode not in (None, 'whole_image', 'four_corners', 'four_corner_crops_no_center'):
        raise ValueError('Unsupported restoration crop_mode.')
    if (crop_mode == 'whole_image' and crop_size is not None) or (crop_mode in ('four_corners', 'four_corner_crops_no_center') and crop_size is None):
        raise ValueError('crop_mode conflicts with crop_size.')
    if crop_size is None:
        return [array]
    if isinstance(crop_size, bool) or not isinstance(crop_size, int) or crop_size <= 0:
        raise ValueError('crop_size must be null or a positive integer.')
    h, w = array.shape
    if min(h, w) < crop_size:
        raise ValueError('Image smaller than the declared corner crop.')
    c = crop_size
    return [array[:c, :c], array[:c, w-c:], array[h-c:, :c], array[h-c:, w-c:]]


def restoration_metrics(truth, pred, data_range, ssim_parameters=None, metrics=('psnr', 'ssim')):
    """Score a single already-clipped crop; return only requested metrics."""
    truth = _numeric_array(truth, 'Restoration target', 'uif').astype(np.float64)
    pred = _numeric_array(pred, 'Restoration prediction', 'uif').astype(np.float64)
    if truth.ndim != 2 or truth.shape != pred.shape:
        raise ValueError('Restoration requires identical two-dimensional array shapes.')
    data_range = _finite_number(data_range, 'data_range', positive=True)
    names = set(metrics)
    if names - {'psnr', 'ssim'}:
        raise ValueError('Unsupported restoration metric.')
    result = {}
    if 'psnr' in names:
        mse = float(np.mean((truth - pred) ** 2))
        if mse == 0:
            raise UndefinedMetricError('psnr', 'PSNR is infinite for zero MSE; the finite-score contract cannot represent it.')
        result['psnr'] = float(20 * np.log10(data_range) - 10 * np.log10(mse))
    if 'ssim' in names:
        params = normalize_ssim_parameters(ssim_parameters)
        window = params['win_size']
        if min(truth.shape) < window:
            raise ValueError('SSIM requires image dimensions >= win_size (%d).' % window)
        if params['gaussian_weights']:
            def filt(array):
                return ndi.gaussian_filter(array, sigma=params['sigma'], truncate=3.5, mode='reflect')
        else:
            def filt(array):
                return ndi.uniform_filter(array, size=window, mode='reflect')
        ux, uy = filt(truth), filt(pred)
        n = window ** truth.ndim
        covariance = n / (n - 1) if params['use_sample_covariance'] else 1.0
        vx = (filt(truth * truth) - ux * ux) * covariance
        vy = (filt(pred * pred) - uy * uy) * covariance
        vxy = (filt(truth * pred) - ux * uy) * covariance
        c1, c2 = (params['K1'] * data_range) ** 2, (params['K2'] * data_range) ** 2
        values = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
        pad = (window - 1) // 2
        result['ssim'] = float(values[pad:-pad, pad:-pad].mean())
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Non-finite requested restoration metric.')
    return result


def _spacing(value, ndim, name):
    if value is None:
        return None
    array = _numeric_array(value, name, 'uif').astype(float)
    if array.shape != (ndim,) or np.any(array <= 0):
        raise ValueError(name + ' must have one positive value per array axis.')
    return tuple(array)


def _detection_boxes(value, prediction):
    if not isinstance(value, (list, tuple)):
        raise ValueError('Canonical detections must be a list of (class_id, xyxy_box, confidence).')
    result = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError('Each canonical box requires class_id, xyxy_box, confidence.')
        cls, box, confidence = item
        if isinstance(cls, (bool, np.bool_)) or not isinstance(cls, (int, np.integer)) or cls < 0:
            raise ValueError('Class IDs must be nonnegative integers.')
        box = _numeric_array(box, 'Box coordinates', 'uif').astype(float)
        if box.shape != (4,) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Boxes require four xyxy coordinates with positive width and height.')
        confidence = _finite_number(confidence, 'Prediction confidence') if prediction else 1.0
        if not 0 <= confidence <= 1:
            raise ValueError('Prediction confidence must be in [0,1].')
        result.append((int(cls), tuple(box), confidence))
    return result


def _validate_sample(sample, spec, require_target):
    """Attribute only demonstrable public input violations to the candidate.

    Normal scoring validates once. If validation fails, retry the public-only
    check to distinguish a malformed prediction from private truth/reference
    problems; exception wording is never used to make that classification.
    """
    try:
        return _validate_sample_impl(sample, spec, require_target)
    except ValueError as exc:
        if not require_target:
            raise PredictionValidationError(str(exc)) from exc
        try:
            _validate_sample_impl(sample, spec, require_target=False)
        except ValueError as prediction_exc:
            raise PredictionValidationError(str(prediction_exc)) from prediction_exc
        raise


def _validate_sample_impl(sample, spec, require_target):
    if not isinstance(sample, Mapping) or 'prediction' not in sample:
        raise ValueError('Adapter must return a sample mapping containing prediction.')
    if require_target and ('target' not in sample or sample['target'] is None):
        raise ValueError('Scoring sample requires a target.')
    out = dict(sample)
    family = spec['task_family']
    if family == 'segmentation':
        out = validate_tree_sample(out, [item['name'] for item in normalize_metric_definitions(spec)], spec,
                                   require_target=require_target)
        pred = out['prediction']
        spacing = _spacing(out.get('spacing'), pred.ndim, 'spacing')
        pred_spacing = _spacing(out.get('prediction_spacing'), pred.ndim, 'prediction_spacing')
        if spacing is not None and pred_spacing is not None and not np.allclose(spacing, pred_spacing, rtol=0, atol=1e-4):
            raise ValueError('Target/input and prediction spacing must match.')
        out['spacing'] = spacing
        return out

    if family == 'detection':
        out['prediction'] = _detection_boxes(out['prediction'], True)
        if require_target:
            out['target'] = _detection_boxes(out['target'], False)
        return out
    pred = _numeric_array(out['prediction'], 'Prediction')
    out['prediction'] = pred
    if family == 'regression':
        if pred.ndim > 1 or pred.dtype.kind not in 'uif':
            raise ValueError('Regression prediction must be a finite real numeric scalar or nonempty 1D vector.')
        if 'target_shape' in spec and pred.shape != tuple(spec['target_shape']):
            raise ValueError('Regression prediction must match the declared target_shape exactly.')
        if require_target:
            target = _numeric_array(out['target'], 'Regression target', 'uif')
            if target.shape != pred.shape:
                raise ValueError('Regression prediction and target shapes must match exactly; broadcasting is not allowed.')
            out['target'] = target
        return out
    if family == 'classification':
        if pred.ndim != 1 or np.any(pred < 0) or np.any(pred > 1):
            raise ValueError('Classification predictions must be a vector of probabilities in [0,1].')
        task_type = spec['task_type']
        if task_type != 'multilabel':
            allowed = (1, 2) if task_type == 'binary' else (len(spec['class_labels']),)
            if len(pred) not in allowed or (len(pred) > 1 and not np.isclose(pred.sum(), 1, rtol=1e-5, atol=1e-5)):
                raise ValueError('Classification probability dimensions/order must match class_labels and sum to 1.')
        if require_target:
            if task_type == 'multilabel':
                target = _numeric_array(out['target'], 'Multilabel target')
                if target.shape != pred.shape or not np.isin(target, [0, 1]).all():
                    raise ValueError('Multilabel targets must be a binary vector matching prediction shape.')
                out['target'] = target
            elif np.asarray(out['target']).ndim != 0 or str(out['target']) not in set(map(str, spec['class_labels'])):
                raise ValueError('Classification target must be a scalar declared class label.')
        return out
    if family == 'image_restoration':
        if pred.ndim != 2 or pred.dtype.kind not in 'uif':
            raise ValueError('Restoration prediction must be a two-dimensional grayscale intensity array.')
        protocol = spec['restoration_evaluation']
        out['prediction'] = np.clip(pred.astype(float), 0, protocol['data_range'])
        crops = restoration_crops(pred, protocol['crop_size'], protocol.get('crop_mode'))
        if 'ssim' in [item['name'] for item in normalize_metric_definitions(spec)] and min(crops[0].shape) < spec['ssim_parameters']['win_size']:
            raise ValueError('SSIM requires image dimensions >= win_size.')
        groups = protocol['groups']
        if groups is not None and out.get('group') not in groups:
            raise ValueError('Adapter group must be one of the declared restoration groups.')
        if groups is None and out.get('group') is not None:
            raise ValueError('Adapter returned a group but restoration groups are undeclared.')
    if out.get('input_shape') is not None and tuple(out['input_shape']) != pred.shape:
        raise ValueError('Prediction and declared input_shape must match exactly.')
    if require_target:
        target = _numeric_array(out['target'], 'Target')
        if target.shape != pred.shape:
            raise ValueError('Prediction and target shapes must match exactly.')
        if family == 'image_restoration':
            if target.dtype.kind not in 'uif' or np.any(target < 0) or np.any(target > protocol['data_range']):
                raise ValueError('Restoration target must be intensities within [0,data_range].')
        elif target.dtype.kind not in 'biu' or np.any(target < 0):
            raise ValueError('Segmentation target requires nonnegative integer instance labels.')
        out['target'] = target
    return out


def validate_prediction(sample, spec):
    """Validate one prediction-only sample without reading or requiring answers."""
    _validate_sample(sample, normalize_metric_spec(spec), require_target=False)
    return {'valid': True}


def _dice(truth, prediction, smooth=0.0):
    truth, prediction = truth > 0, prediction > 0
    count = int(truth.sum()) + int(prediction.sum())
    return float((2 * np.logical_and(truth, prediction).sum() + smooth) / (count + smooth)) if count + smooth else 1.0


def _iou(truth, prediction):
    truth, prediction = truth > 0, prediction > 0
    union = int(np.logical_or(truth, prediction).sum())
    return float(np.logical_and(truth, prediction).sum() / union) if union else 1.0


def _aji(truth, prediction):
    # Official MoNuSeg AJI permits the same prediction to match multiple GT IDs.
    # Preserve supplied instance IDs, including disconnected parts of one ID.
    true_labels, true_inverse, true_area = np.unique(truth, return_inverse=True, return_counts=True)
    pred_labels, pred_inverse, pred_area = np.unique(prediction, return_inverse=True, return_counts=True)
    true_ids, pred_ids = np.flatnonzero(true_labels > 0), np.flatnonzero(pred_labels > 0)
    if not len(true_ids) and not len(pred_ids):
        return 1.0
    if not len(true_ids) or not len(pred_ids):
        return 0.0
    stride = len(pred_labels)
    pairs, intersections = np.unique(true_inverse.ravel().astype(np.int64) * stride + pred_inverse.ravel(), return_counts=True)
    best = {}
    for key, intersection in zip(pairs, intersections):
        true_id, pred_id = divmod(int(key), stride)
        if true_labels[true_id] == 0 or pred_labels[pred_id] == 0:
            continue
        intersection = int(intersection)
        union = int(true_area[true_id]) + int(pred_area[pred_id]) - intersection
        iou = intersection / union
        if true_id not in best or iou > best[true_id][0]:
            best[true_id] = (iou, pred_id, intersection, union)
    used = set()
    intersection_sum = union_sum = 0
    for true_id in true_ids:
        match = best.get(int(true_id))
        if match is None:
            union_sum += int(true_area[true_id])
        else:
            _, pred_id, intersection, union = match
            intersection_sum += intersection
            union_sum += union
            used.add(pred_id)
    union_sum += sum(int(pred_area[j]) for j in pred_ids if int(j) not in used)
    return float(intersection_sum / union_sum)


def _hd95(truth, prediction, spacing=None):
    truth, prediction = truth > 0, prediction > 0
    if not truth.any() and not prediction.any():
        return 0.0
    if not truth.any() or not prediction.any():
        raise UndefinedMetricError('hd95', 'HD95 is infinite when exactly one mask is empty.')
    structure = ndi.generate_binary_structure(truth.ndim, 1)
    true_border = truth ^ ndi.binary_erosion(truth, structure=structure, border_value=0)
    pred_border = prediction ^ ndi.binary_erosion(prediction, structure=structure, border_value=0)
    forward = ndi.distance_transform_edt(~pred_border, sampling=spacing)[true_border]
    backward = ndi.distance_transform_edt(~true_border, sampling=spacing)[pred_border]
    return float(np.percentile(np.concatenate([forward, backward]), 95))


def _classification_scores(samples, spec, names):
    scores = np.stack([sample['prediction'] for sample in samples]).astype(float)
    truth = np.asarray([sample['target'] for sample in samples])
    result = {}
    if spec['task_type'] == 'multilabel':
        accuracy = float(np.mean((scores >= spec['threshold']) == truth))
    else:
        labels = np.asarray(list(map(str, spec['class_labels'])))
        truth = truth.astype(str)
        if scores.shape[1] == 1:
            positive = str(spec['positive_class'])
            negative = next(label for label in labels if label != positive)
            predicted = np.where(scores[:, 0] >= spec['threshold'], positive, negative)
        elif spec['task_type'] == 'binary':
            positive = str(spec['positive_class'])
            negative = next(label for label in labels if label != positive)
            index = list(labels).index(positive)
            predicted = np.where(scores[:, index] >= spec['threshold'], positive, negative)
        else:
            predicted = labels[np.argmax(scores, axis=1)]
        accuracy = float(np.mean(predicted == truth))
    for name in names:
        if name in ('accuracy', 'mean_binary_accuracy'):
            result[name] = accuracy
    if any(name in ('auc', 'mean_roc_auc') for name in names):
        from sklearn.metrics import roc_auc_score
        try:
            if spec['task_type'] == 'multilabel':
                value = float(roc_auc_score(truth, scores, average='macro'))
            elif len(spec['class_labels']) == 2:
                positive = str(spec['positive_class'])
                index = list(map(str, spec['class_labels'])).index(positive)
                value = float(roc_auc_score(truth == positive, scores[:, 0 if scores.shape[1] == 1 else index]))
            else:
                indices = {str(label): i for i, label in enumerate(spec['class_labels'])}
                y = np.asarray([indices[label] for label in truth])
                value = float(roc_auc_score(y, scores, average='macro', multi_class='ovr', labels=list(range(len(indices)))))
            if not math.isfinite(value):
                raise UndefinedMetricError('auc', 'ROC AUC is undefined for this label distribution.')
        except UndefinedMetricError:
            raise
        except ValueError as exc:
            raise UndefinedMetricError('auc', 'ROC AUC is undefined for this label distribution.') from exc
        for name in names:
            if name in ('auc', 'mean_roc_auc'):
                result[name] = value
    return result


def _regression_scores(samples, spec, names):
    """Stream equal-weight errors; apply the RMSE root only after reduction."""
    count = 0
    shape = None
    mean_absolute = mean_squared = 0.0
    for raw_sample in samples:
        sample = _validate_sample(raw_sample, spec, require_target=True)
        prediction, target = sample['prediction'], sample['target']
        if shape is None:
            shape = prediction.shape
        elif prediction.shape != shape:
            raise ValueError('Regression output shape must be fixed across all samples.')
        try:
            with np.errstate(over='raise', invalid='raise'):
                error = prediction.astype(np.float64) - target.astype(np.float64)
                absolute = float(np.mean(np.abs(error))) if 'mae' in names else None
                squared = float(np.mean(np.square(error))) if any(name in names for name in ('mse', 'rmse')) else None
        except FloatingPointError as exc:
            raise ValueError('Non-finite requested regression metric.') from exc
        count += 1
        if absolute is not None:
            mean_absolute += (absolute - mean_absolute) / count
        if squared is not None:
            mean_squared += (squared - mean_squared) / count
    if not count:
        raise ValueError('Scoring requires at least one sample.')
    metrics = {}
    if 'mae' in names:
        metrics['mae'] = float(mean_absolute)
    if 'mse' in names:
        metrics['mse'] = float(mean_squared)
    if 'rmse' in names:
        metrics['rmse'] = float(math.sqrt(mean_squared))
    return metrics, count


def _box_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def _detection_scores(samples, spec, names):
    classes = sorted({cls for sample in samples for cls, _, _ in sample['target']})
    if not classes:
        raise ValueError('AP and recall are undefined for a split without ground-truth boxes.')
    per_class = {}
    total_hits = total_gt = 0
    for cls in classes:
        gt = {i: [box for c, box, _ in sample['target'] if c == cls] for i, sample in enumerate(samples)}
        count = sum(map(len, gt.values()))
        detections = [(confidence, i, j, box) for i, sample in enumerate(samples)
                      for j, (c, box, confidence) in enumerate(sample['prediction']) if c == cls]
        detections.sort(key=lambda value: (-value[0], value[1], value[2]))
        used, tp = set(), []
        recall_hits = 0
        for confidence, i, _, box in detections:
            overlaps = [_box_iou(box, other) for other in gt[i]]
            match = int(np.argmax(overlaps)) if overlaps else -1
            hit = match >= 0 and overlaps[match] >= spec['iou_threshold'] and (i, match) not in used
            tp.append(int(hit))
            if hit:
                used.add((i, match))
                recall_hits += int(confidence >= spec['recall_confidence_threshold'])
        metrics = {}
        if 'ap50' in names:
            if tp:
                cumulative = np.cumsum(tp)
                recalls = np.r_[0., cumulative / count, 1.]
                precisions = np.r_[0., cumulative / np.arange(1, len(tp) + 1), 0.]
                precisions = np.maximum.accumulate(precisions[::-1])[::-1]
                metrics['ap50'] = float(np.sum(np.diff(recalls) * precisions[1:]))
            else:
                metrics['ap50'] = 0.0
        if 'recall' in names:
            metrics['recall'] = recall_hits / count
        per_class[str(cls)] = dict(metrics, ground_truth=count, true_positives=recall_hits)
        total_hits += recall_hits
        total_gt += count
    result = {}
    if 'ap50' in names:
        result['ap50'] = float(np.mean([value['ap50'] for value in per_class.values()]))
    if 'recall' in names:
        result['recall'] = total_hits / total_gt
    return result, per_class


def _score_timed_call(callback, stage, operation, metric_names=()):
    if callback is None:
        return operation()
    context = {'stage': stage, 'metric_names': list(metric_names)}
    callback(dict(context, event='started'))
    started = perf_counter()
    try:
        result = operation()
    except Exception as exc:
        callback(dict(context, event='failed', elapsed_seconds=perf_counter() - started,
                      error_type=type(exc).__name__))
        raise
    callback(dict(context, event='finished', elapsed_seconds=perf_counter() - started))
    return result


def _score_image_case(case_index, raw_sample, spec, names, callback):
    """Validate and score one independent image; return only small score data."""
    def emit(event):
        callback(dict(event, case_index=case_index))

    report = emit if callback is not None else None

    def calculate():
        sample = _score_timed_call(report, 'validate',
                                   lambda: _validate_sample(raw_sample, spec, require_target=True))
        truth, pred = sample['target'], sample['prediction']
        if spec['task_family'] == 'image_restoration':
            protocol = spec['restoration_evaluation']
            def restoration():
                values = [restoration_metrics(a, b, protocol['data_range'], spec['ssim_parameters'], names)
                          for a, b in zip(restoration_crops(truth, protocol['crop_size']),
                                          restoration_crops(pred, protocol['crop_size']))]
                return {name: float(np.mean([value[name] for value in values])) for name in names}
            image_scores = _score_timed_call(report, 'metric', restoration, names)
            group = sample['group'] if protocol['groups'] is not None else None
        else:
            kernels = {'binary_dice': lambda a, b: _dice(a, b, spec['dice_smooth']), 'binary_iou': _iou,
                       'aji': _aji, 'hd95': lambda a, b: _hd95(a, b, sample.get('spacing'))}
            image_scores = {name: _score_timed_call(report, 'metric',
                            lambda: kernels[name](truth, pred), (name,)) for name in names if name in kernels}
            extra_names = [name for name in names if name in _SEGMENTATION_TREE_METRICS]
            if extra_names:
                # _validate_sample already validated all requested tree inputs.
                # Never allow a sample/spec flag to opt out of that validation.
                image_scores.update(_seg_score_validated_case(sample, extra_names, spec, report))
            group = None
        return image_scores, group

    return _score_timed_call(report, 'case', calculate, names)


def _image_score_records(samples, spec, names, callback, workers):
    """Bound loaded/in-flight cases and emit results in input order.

    Threads share arrays without pickling full CT volumes. This is deliberately
    a per-invocation bound, not a machine-wide resource scheduler. Harness callers
    must allocate workers with concurrent evaluators and memory limits in mind.
    """
    iterator = iter(samples)
    exhausted = object()

    def load(case_index):
        started = perf_counter()
        if callback is not None:
            callback(dict(stage='load', event='started', case_index=case_index, metric_names=[]))
        try:
            raw_sample = next(iterator)
        except StopIteration:
            raw_sample = exhausted
        except Exception as exc:
            if callback is not None:
                callback(dict(stage='load', event='failed', case_index=case_index, metric_names=[],
                              elapsed_seconds=perf_counter() - started, error_type=type(exc).__name__))
            raise
        if callback is not None:
            callback(dict(stage='load', event='exhausted' if raw_sample is exhausted else 'finished',
                          case_index=case_index, metric_names=[], elapsed_seconds=perf_counter() - started))
        return raw_sample

    case_index = 0
    if workers == 1:
        while True:
            raw_sample = load(case_index)
            if raw_sample is exhausted:
                return
            record = _score_image_case(case_index, raw_sample, spec, names, callback)
            del raw_sample
            yield record
            case_index += 1
    else:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='trusted-score')
        pending = {}
        completed = {}
        next_result = 0
        input_exhausted = False
        try:
            while not input_exhausted or pending:
                # Only live work retains volumes. Completed records are small
                # score dictionaries, so a slow early case cannot idle workers
                # or force later completed arrays to remain resident.
                while not input_exhausted and len(pending) < workers:
                    raw_sample = load(case_index)
                    if raw_sample is exhausted:
                        input_exhausted = True
                        break
                    future = pool.submit(_score_image_case, case_index, raw_sample, spec, names, callback)
                    pending[future] = case_index
                    del raw_sample
                    case_index += 1
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Observe every available failure before loading further cases.
                for future in sorted(done, key=pending.get):
                    index = pending.pop(future)
                    completed[index] = future.result()
                del done, future
                while next_result in completed:
                    yield completed.pop(next_result)
                    next_result += 1
        finally:
            for future in pending:
                future.cancel()
            # Active native calls cannot be killed safely as individual threads.
            # The enclosing evaluator subprocess owns the hard timeout/cleanup.
            pool.shutdown(wait=True, cancel_futures=True)


def score_samples(samples, spec, progress_callback=None, workers=1):
    """Score canonical samples under one normalized and frozen metric protocol.

    Restoration averages corners within images, then images or declared groups.
    Segmentation averages images. Regression averages samples and outputs;
    classification and detection score the entire split. Returned metrics
    contain only requested names, plus restoration metric/group keys; ``n``
    records the exact number of scored samples.

    ``workers`` is a positive integer execution setting, separate from the frozen
    mathematical protocol. Segmentation/restoration may use bounded case threads;
    other families require workers=1. Reduction always follows input order.
    At most ``workers`` cases are loaded/in-flight, never the whole volume split.

    ``progress_callback(event)`` optionally receives dictionaries with ``stage``,
    ``event`` (started/finished/failed; load also exhausted), and ``metric_names``.
    Completed/failed operations include ``elapsed_seconds``; image stages also
    include a zero-based ``case_index``. Metric events may repeat for individual
    classes or share names when computation is shared; their times are inclusive
    inside case/score times and must not be added twice. No paths, arrays, label
    values or scores are emitted. Callbacks are serialized across case threads.
    Callback exceptions propagate; returned score payloads are unchanged.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('Scoring workers must be a positive integer.')
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError('progress_callback must be callable or None.')
    spec = normalize_metric_spec(spec)
    definitions = normalize_metric_definitions(spec)
    names = [item['name'] for item in definitions]
    family = spec['task_family']
    if workers != 1 and family not in ('segmentation', 'image_restoration'):
        raise ValueError('Parallel case scoring supports segmentation and image_restoration only.')
    lock = Lock()

    def emit(event):
        with lock:
            progress_callback(event)

    callback = emit if progress_callback is not None else None

    def score_split():
        nonlocal samples
        extras = {}
        # Whole-split metrics retain their existing mathematical aggregation.
        if family in ('classification', 'detection'):
            samples = [_validate_sample(sample, spec, require_target=True) for sample in samples]
            if not samples:
                raise ValueError('Scoring requires at least one sample.')
            count = len(samples)
        if family == 'classification':
            metrics = _classification_scores(samples, spec, names)
        elif family == 'regression':
            metrics, count = _regression_scores(samples, spec, names)
        elif family == 'detection':
            metrics, extras['per_class'] = _detection_scores(samples, spec, names)
        else:
            per_image = []
            grouped = {}
            for image_scores, group in _image_score_records(samples, spec, names, callback, workers):
                per_image.append(image_scores)
                if family == 'image_restoration' and spec['restoration_evaluation']['groups'] is not None:
                    grouped.setdefault(group, []).append(image_scores)
            count = len(per_image)
            if not count:
                raise ValueError('Scoring requires at least one sample.')
            source = per_image
            if family == 'segmentation':
                extras['per_case'] = per_image
            if family == 'image_restoration':
                groups = spec['restoration_evaluation']['groups']
                if groups is not None and set(grouped) != set(groups):
                    raise ValueError('Scoring samples must contain all and only the declared groups.')
                per_group = {group: {name: float(np.mean([value[name] for value in rows])) for name in names}
                             for group, rows in grouped.items()}
                if spec['restoration_evaluation']['aggregation'] == 'mean_groups':
                    source = list(per_group.values())
                extras['per_group'] = per_group
            metrics = {name: float(np.mean([value[name] for value in source])) for name in names}
            for group, values in extras.get('per_group', {}).items():
                metrics.update({name + '/' + group: value for name, value in values.items()})
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError('Non-finite requested metric.')
        result = dict(metrics=metrics, n=count, **extras)
        if 'metrics' not in spec:
            result.update(metric=metrics[spec['primary_metric']], maximize=spec['maximize'])
        return result

    return _score_timed_call(callback, 'score', score_split, names)
