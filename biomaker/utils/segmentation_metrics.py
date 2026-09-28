"""Trusted topology and multi-class segmentation metrics, independent of BioMaker.

Adapted from EndoluminalSurgicalVision-IMR/Airway-Tree-Modeling-26,
commit 6af4a24529977413a8f5216a0eeb2a8ee8a1e1d3 (2026-09-08),
evaluation/T1_evaluation/metrics.py and evaluation/T2_evaluation/{metrics,
voxel_metrics,projection,graph_utils}.py. The official evaluate.py call chains
determine whether each metric uses raw predictions or their largest component.

Explicit protocol parameters reproduce the cited metric formulas and conventions.
Input validation additionally rejects invalid directed trees rather than allowing the upstream
subtree traversal to loop forever. Ground-truth skeletons, branch parsing and
graphs are supplied references: this module never regenerates them. ATM22-derived
references describe a local reference protocol, not official ATM26 annotations.

The source is embedded in frozen evaluators. Keep it free of package imports and
prefix private names with _seg_ / _SEG_ to avoid collisions with other kernels.
Only requested metric dependencies are validated and computed. Volumes use ZYX
array order; optional geometry dictionaries use SimpleITK XYZ order.

MIT License
Copyright (c) 2026 EndoluminalSurgicalVision@IMR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections.abc import Mapping as _SEG_Mapping
from time import perf_counter as _seg_perf_counter

import numpy as np
from scipy import ndimage as _seg_ndi


_SEG_NAMES = frozenset((
    'cldice', 'tree_length_detected', 'branch_detected', 'betti_error',
    'multiclass_dice', 'multiclass_cldice', 'branch_accuracy', 'branch_f1',
    'subtree_consistency', 'topological_distance',
))
_SEG_SKELETON_NAMES = frozenset((
    'cldice', 'tree_length_detected', 'branch_detected', 'multiclass_cldice',
))
_SEG_BRANCH_NAMES = frozenset((
    'branch_accuracy', 'branch_f1', 'subtree_consistency', 'topological_distance',
))
_SEG_GRAPH_NAMES = frozenset(('subtree_consistency', 'topological_distance'))


def _seg_names(names, allow_traditional=False, allow_empty=False):
    if isinstance(names, (str, bytes)):
        raise ValueError('segmentation metric names must be a nonempty sequence of names')
    names = tuple(names)
    if (not names and not allow_empty) or any(not isinstance(name, str) for name in names):
        raise ValueError('segmentation metric names must be a nonempty sequence of strings')
    if len(set(names)) != len(names):
        raise ValueError('segmentation metric names must be unique')
    supported = _SEG_NAMES
    if allow_traditional:
        supported = supported | {'binary_dice', 'binary_iou', 'aji', 'hd95'}
    unsupported = set(names) - supported
    if unsupported:
        raise ValueError('Unsupported segmentation metrics: ' + ', '.join(sorted(unsupported)))
    return names


def normalize_segmentation_parameters(spec=None):
    """Normalize reusable segmentation parameters, preserving other spec fields.

    The returned parameter values are JSON-compatible. Special dataset label
    ranges, ignored labels, smoothing and topology conventions must be explicit
    in the caller's contract; there are no challenge-specific label constants.
    """
    if spec is None:
        spec = {}
    if not isinstance(spec, _SEG_Mapping):
        raise ValueError('segmentation spec must be a mapping')
    result = dict(spec)
    class_aggregation = spec.get('class_aggregation', 'gt_present')
    if class_aggregation != 'gt_present':
        raise ValueError('class_aggregation must be gt_present')
    result['class_aggregation'] = class_aggregation
    geometry_policy = spec.get('geometry_policy', 'shape_spacing')
    if geometry_policy not in ('shape_spacing', 'physical'):
        raise ValueError('geometry_policy must be shape_spacing or physical')
    result['geometry_policy'] = geometry_policy
    for name, default in (('dice_smooth', 0.0), ('topology_smooth', 0.0),
                          ('branch_coverage_threshold', 0.8), ('missing_label_penalty', 50.0)):
        value = spec.get(name, default)
        if isinstance(value, (bool, str, bytes)):
            raise ValueError('%s must be a finite nonnegative number' % name)
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('%s must be a finite nonnegative number' % name) from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError('%s must be a finite nonnegative number' % name)
        if name == 'branch_coverage_threshold' and value > 1:
            raise ValueError('branch_coverage_threshold must be in [0, 1]')
        result[name] = value
    connectivity = spec.get('connectivity', 3)
    if isinstance(connectivity, bool) or connectivity not in (1, 2, 3):
        raise ValueError('connectivity must be 1, 2, or 3')
    result['connectivity'] = int(connectivity)
    policy = spec.get('topology_component_policy', 'largest')
    if policy not in ('largest', 'raw'):
        raise ValueError('topology_component_policy must be largest or raw')
    result['topology_component_policy'] = policy
    betti = spec.get('betti_reference', 'target')
    if betti not in ('target', 'one'):
        raise ValueError('betti_reference must be target or one')
    result['betti_reference'] = betti
    for name, default in (('label_ids', None), ('subtree_ignore_labels', ())):
        values = spec.get(name, default)
        if values is None and name == 'label_ids':
            result[name] = None
            continue
        if isinstance(values, (str, bytes)):
            raise ValueError('%s must be a sequence of nonnegative integer labels' % name)
        try:
            values = tuple(values)
        except TypeError as exc:
            raise ValueError('%s must be a sequence of nonnegative integer labels' % name) from exc
        for value in values:
            if isinstance(value, (bool, str, bytes)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError('%s must contain nonnegative integer labels' % name)
            if value < 0 or (isinstance(value, (float, np.floating)) and
                             (not np.isfinite(value) or value != int(value))):
                raise ValueError('%s must contain nonnegative integer labels' % name)
        if len(set(values)) != len(values) or (name == 'label_ids' and not values):
            raise ValueError('%s must contain unique labels and label_ids cannot be empty' % name)
        result[name] = [int(value) for value in values]
    return result


def _seg_structure(ndim, connectivity):
    return _seg_ndi.generate_binary_structure(ndim, min(ndim, connectivity))


def _seg_labels(value, name, label_ids=None):
    array = np.asarray(value)
    if array.ndim not in (2, 3) or any(size == 0 for size in array.shape):
        raise ValueError('%s must be a nonempty 2D or 3D label volume' % name)
    if array.dtype.kind not in 'buif':
        raise ValueError('%s must contain real numeric integer labels' % name)
    # Slabs avoid temporary floating/mask arrays as large as an entire CT.
    for start in range(0, array.shape[0], 16):
        block = array[start:start + 16]
        if not np.isfinite(block).all():
            raise ValueError('%s contains non-finite values' % name)
        if block.dtype.kind == 'f' and not np.array_equal(block, np.rint(block)):
            raise ValueError('%s must contain exact integer labels' % name)
        if block.min() < 0:
            raise ValueError('%s cannot contain negative labels' % name)
        if label_ids is not None and not np.isin(block, label_ids).all():
            raise ValueError('%s contains labels outside declared label_ids' % name)
    return array


def _seg_geometry(value, name, array_shape):
    if not isinstance(value, _SEG_Mapping):
        raise ValueError('%s must be a geometry mapping' % name)
    result = {}
    ndim = len(array_shape)
    for field, length in (('size', ndim), ('spacing', ndim), ('origin', ndim), ('direction', ndim * ndim)):
        try:
            entry = np.asarray(value[field], dtype=np.float64)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('%s requires numeric %s' % (name, field)) from exc
        if entry.shape != (length,) or not np.isfinite(entry).all():
            raise ValueError('%s %s must have %d finite entries' % (name, field, length))
        if field in ('size', 'spacing') and np.any(entry <= 0):
            raise ValueError('%s %s must be positive' % (name, field))
        if field == 'size' and not np.array_equal(entry, np.rint(entry)):
            raise ValueError('%s size must contain integers' % name)
        result[field] = tuple(entry.tolist())
    if tuple(result['size'][::-1]) != tuple(array_shape):
        raise ValueError('%s XYZ size does not match the ZYX array shape' % name)
    return result


def _seg_validate_geometry(sample, array_keys, required=False):
    geometry_keys = ['input_geometry'] + [key + '_geometry' for key in array_keys]
    if not required and not any(key in sample for key in geometry_keys):
        return
    missing = [key for key in geometry_keys if key not in sample]
    if missing:
        raise ValueError('Incomplete segmentation geometry metadata; missing ' + ', '.join(missing))
    reference = _seg_geometry(sample['input_geometry'], 'input_geometry', sample['prediction'].shape)
    sample['input_geometry'] = reference
    for key in array_keys:
        geometry_key = key + '_geometry'
        actual = _seg_geometry(sample[geometry_key], geometry_key, sample[key].shape)
        if actual['size'] != reference['size']:
            raise ValueError('%s size does not match input geometry' % geometry_key)
        for field in ('spacing', 'origin', 'direction'):
            if not np.allclose(actual[field], reference[field], rtol=1e-5, atol=1e-6):
                raise ValueError('%s %s does not match input geometry' % (geometry_key, field))
        sample[geometry_key] = actual


def _seg_parsing_nodes(parsing):
    ids = np.unique(parsing)
    ids = ids[ids > 0]
    # Comparing before conversion prevents overflow/wraparound of integer IDs.
    if not np.array_equal(ids, np.arange(1, len(ids) + 1, dtype=np.int64)):
        raise ValueError('parsing branch IDs must be contiguous and start at 1')
    return int(len(ids))


def _seg_validate_graph(value, node_count):
    edges = np.asarray(value)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('graph must have shape (2, E)')
    if edges.dtype.kind not in 'iu':
        raise ValueError('graph must contain integer node indices')
    if np.any(edges < 0) or np.any(edges >= node_count):
        raise ValueError('graph indices must match parsing nodes 0..N-1')
    if edges.shape[1] != node_count - 1:
        raise ValueError('graph must contain N-1 edges for N parsing nodes')
    edges = edges.astype(np.int64, copy=False)
    if np.any(edges[0] == edges[1]):
        raise ValueError('graph cannot contain self edges')
    indegree = np.bincount(edges[1], minlength=node_count)
    roots = np.flatnonzero(indegree == 0)
    if len(roots) != 1 or np.any(indegree > 1):
        raise ValueError('graph must be a directed tree with one root and one parent per other node')
    children = [[] for _ in range(node_count)]
    for parent, child in edges.T:
        children[int(parent)].append(int(child))
    seen = set()
    stack = [int(roots[0])]
    while stack:
        node = stack.pop()
        if node in seen:
            raise ValueError('graph must be acyclic')
        seen.add(node)
        stack.extend(children[node])
    if len(seen) != node_count:
        raise ValueError('graph must be acyclic and all nodes must be reachable from its root')
    return edges


def validate_tree_sample(sample, names, spec=None, require_target=True):
    """Validate requested tree metric inputs and return a shallow canonical copy.

    ``prediction`` and ``target`` are finite, nonnegative integer 2D/3D arrays.
    Optional spec ``label_ids`` restricts their values. Required references:
    skeleton for clDice/TLD/BD; parsing for BD or branch metrics; graph for SC/TD.
    Parsing IDs are contiguous 1..N, graph indices are zero-based parent->child.
    A singleton tree has parsing ID 1 and a graph of shape (2, 0); its subtree
    consistency is 0 and its topological distance is 0 for a matching label.
    Pure arrays may omit geometry; otherwise input and every relevant image must
    provide size/spacing/origin/direction geometry mappings in SimpleITK XYZ order.

    ``require_target=False`` validates only the public prediction and its input
    geometry. It deliberately does not inspect any supplied private target or
    reference fields, so submission validation cannot use hidden ground truth.
    This function does not mutate arrays or construct derived references.
    Supplied references may be derived from a preprocessed target (e.g. hole
    filling), so skeleton containment in the unprocessed target is not required.
    The reference builder validates containment in its own construction mask.
    """
    names = _seg_names(names, allow_traditional=True, allow_empty=True)
    if not isinstance(sample, _SEG_Mapping):
        raise ValueError('segmentation sample must be a mapping')
    if 'prediction' not in sample:
        raise ValueError('segmentation sample requires prediction')
    result = dict(sample)
    parameters = normalize_segmentation_parameters(spec)
    result['prediction'] = _seg_labels(sample['prediction'], 'prediction', parameters['label_ids'])
    array_keys = ['prediction']
    if sample.get('input_shape') is not None and tuple(sample['input_shape']) != result['prediction'].shape:
        raise ValueError('prediction shape does not match input_shape')
    if require_target:
        required = ['target']
        if set(names) & _SEG_SKELETON_NAMES:
            required.append('skeleton')
        if 'branch_detected' in names or set(names) & _SEG_BRANCH_NAMES:
            required.append('parsing')
        for key in required:
            if key not in sample:
                raise ValueError('Requested segmentation metrics require ' + key)
            result[key] = _seg_labels(sample[key], key, parameters['label_ids'] if key == 'target' else None)
            if result[key].shape != result['prediction'].shape:
                raise ValueError('%s and prediction shapes must match' % key)
            array_keys.append(key)
        if 'skeleton' in required and 'parsing' in required:
            for start in range(0, result['skeleton'].shape[0], 16):
                skel = result['skeleton'][start:start + 16] > 0
                if np.any(skel & (result['parsing'][start:start + 16] <= 0)):
                    raise ValueError('every skeleton voxel must have a positive parsing branch ID')
        if 'parsing' in required:
            node_count = _seg_parsing_nodes(result['parsing'])
            if set(names) & _SEG_GRAPH_NAMES:
                if 'graph' not in sample:
                    raise ValueError('Requested segmentation metrics require graph')
                result['graph'] = _seg_validate_graph(sample['graph'], node_count)
    _seg_validate_geometry(result, array_keys, required=parameters['geometry_policy'] == 'physical')
    return result


def _seg_largest_component(seg, connectivity, labeled=None):
    components, count = (labeled if labeled is not None else
                         _seg_ndi.label(seg > 0, structure=_seg_structure(seg.ndim, connectivity)))
    if count <= 1:
        return seg
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return np.where(components == sizes.argmax(), seg, 0)


def _seg_dice(prediction, target, smooth):
    intersection = np.count_nonzero(prediction & target)
    denominator = np.count_nonzero(prediction) + np.count_nonzero(target) + smooth
    return float((2.0 * intersection + smooth) / denominator) if denominator else 1.0


def _seg_skeletonize(mask):
    # Lazy dependency: Dice, graph metrics, TLD/BD and Betti do not skeletonize.
    try:
        from skimage.morphology import skeletonize
    except ImportError:  # pragma: no cover - older scikit-image compatibility
        from skimage.morphology import skeletonize_3d as skeletonize
    return skeletonize(mask > 0) > 0


def _seg_centerline_overlap(volume, skeleton):
    count = np.count_nonzero(skeleton)
    return float(np.count_nonzero(volume & skeleton) / count) if count else 0.0


def _seg_cldice(prediction, target, skeleton):
    prediction_skeleton = _seg_skeletonize(prediction)
    sensitivity = _seg_centerline_overlap(prediction, skeleton)
    precision = _seg_centerline_overlap(target, prediction_skeleton)
    denominator = sensitivity + precision
    return float(2.0 * sensitivity * precision / denominator) if denominator else 0.0


def _seg_bd(prediction, parsing, skeleton, threshold):
    # Upstream counts bins through the maximum skeleton branch ID, including
    # zero-count intermediate bins. Do not change to unique skeleton IDs.
    branch_ids = parsing[skeleton].astype(np.int64, copy=False)
    gt_counts = np.bincount(branch_ids)[1:]
    total = len(gt_counts)
    pred_counts = np.bincount(branch_ids[prediction[skeleton]])[1:]
    if len(pred_counts) < total:
        pred_counts = np.pad(pred_counts, (0, total - len(pred_counts)))
    coverage = pred_counts / np.maximum(gt_counts, 1)
    detected = int(np.count_nonzero(coverage >= threshold))
    return float(detected / total) if total else 0.0


def _seg_project_nodes(seg, parsing, node_count):
    fg = parsing > 0
    nodes = parsing[fg].astype(np.int64, copy=False) - 1
    # Compress class IDs without narrowing them or allocating max(label)+1 bins.
    classes, inverse = np.unique(seg[fg], return_inverse=True)
    if node_count == 0:
        return np.empty(0, dtype=seg.dtype)
    counts = np.zeros((node_count, len(classes)), dtype=np.int64)
    np.add.at(counts, (nodes, inverse), 1)
    # Including background is intentional; sorted classes resolve ties downward.
    return classes[counts.argmax(axis=1)]


def _seg_acc_f1(gt, pred, valid, need_f1):
    gt, pred = gt[valid], pred[valid]
    accuracy = float(np.mean(pred == gt)) if len(gt) else 0.0
    if not need_f1:
        return accuracy, None
    f1s = []
    for label in np.unique(gt):
        if label == 0:
            continue
        tp = np.sum((gt == label) & (pred == label))
        fp = np.sum((gt != label) & (pred == label))
        fn = np.sum((gt == label) & (pred != label))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return accuracy, float(np.mean(f1s)) if f1s else 0.0


def _seg_subtree_members(edges, node_count):
    children = [[] for _ in range(node_count)]
    for parent, child in edges.T:
        children[int(parent)].append(int(child))
    for root in range(node_count):
        members, stack = [], [root]
        while stack:
            node = stack.pop()
            members.append(node)
            stack.extend(children[node])
        yield root, members


def _seg_sc(pred, edges, ignored_labels):
    count, consistent = 0, 0
    for root, members in _seg_subtree_members(edges, len(pred)):
        if pred[root] in ignored_labels or len(members) <= 1:
            continue
        count += 1
        consistent += int(np.all(pred[members] == pred[root]))
    return float(consistent / count) if count else 0.0


def _seg_spd(edges, node_count):
    distances = np.full((node_count, node_count), np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    distances[edges[0], edges[1]] = 1.0
    distances[edges[1], edges[0]] = 1.0
    for intermediate in range(node_count):
        distances = np.minimum(distances, distances[:, intermediate, None] + distances[None, intermediate, :])
    return distances


def _seg_td(gt, pred, valid, edges, missing_label_penalty):
    distances = _seg_spd(edges, len(pred))
    indices = np.flatnonzero(valid)
    scores = np.full(len(indices), missing_label_penalty, dtype=np.float64)
    for k, node in enumerate(indices):
        targets = np.flatnonzero(gt == pred[node])
        if len(targets):
            distance = np.min(distances[node, targets])
            if np.isfinite(distance):
                scores[k] = distance
    return float(np.mean(scores)) if len(scores) else 0.0



def _seg_timed_call(callback, stage, operation, metric_names=()):
    """Emit path-free progress around one operation; leave its result unchanged."""
    if callback is None:
        return operation()
    context = {'stage': stage, 'metric_names': list(metric_names)}
    callback(dict(context, event='started'))
    started = _seg_perf_counter()
    try:
        result = operation()
    except Exception as exc:
        callback(dict(context, event='failed', elapsed_seconds=_seg_perf_counter() - started,
                      error_type=type(exc).__name__))
        raise
    callback(dict(context, event='finished', elapsed_seconds=_seg_perf_counter() - started))
    return result


def segmentation_score_case(sample, names, spec=None, progress_callback=None):
    """Return requested topology/multi-class segmentation scores for one case.

    Component filtering applies only to centerline/branch coverage and branch
    label projection. clDice and multi-class Dice/clDice always score the raw
    prediction. Betti-0 compares raw connected-component counts against the target
    count, or one when explicitly configured. Multi-class means include nonzero
    GT-present labels only; clDice further excludes classes without GT centerline.
    The optional progress callback receives stage/metric timing events, never
    sample arrays, paths, labels or score values. The return value is unchanged.
    """
    names = _seg_names(names)
    parameters = normalize_segmentation_parameters(spec)
    sample = _seg_timed_call(progress_callback, 'validate',
                             lambda: validate_tree_sample(sample, names, spec=parameters))
    return _seg_score_validated_case(sample, names, parameters, progress_callback)


def _seg_score_validated_case(sample, names, parameters, progress_callback=None):
    """Internal kernel: caller must already have validated all requested inputs.

    Keep this separate from the public validating API: a flag in an adapter's
    sample or metric spec must never permit bypassing trusted input validation.
    """
    requested, scores = set(names), {}
    prediction, target = sample['prediction'], sample['target']

    def timed(stage, operation, metric_names=()):
        return _seg_timed_call(progress_callback, stage, operation, metric_names)

    def metric(name, operation):
        return timed('metric', operation, (name,))

    if 'cldice' in requested:
        scores['cldice'] = metric('cldice', lambda: _seg_cldice(
            prediction > 0, target > 0, sample['skeleton'] > 0))
    needs_selected = bool(requested & {'tree_length_detected', 'branch_detected'} or
                          requested & _SEG_BRANCH_NAMES)
    needs_largest = needs_selected and parameters['topology_component_policy'] == 'largest'
    count = None
    if needs_largest or 'betti_error' in requested:
        # The very same raw prediction labeling supplies largest-component
        # filtering and Betti-0. Release its volume as soon as selection is done.
        structure = _seg_structure(prediction.ndim, parameters['connectivity'])
        labeled = timed('components', lambda: _seg_ndi.label(prediction > 0, structure=structure))
        count = labeled[1]
        if needs_largest:
            selected = timed('component_selection', lambda: _seg_largest_component(
                prediction, parameters['connectivity'], labeled=labeled))
        del labeled
    if needs_selected and not needs_largest:
        selected = prediction
    if requested & {'tree_length_detected', 'branch_detected'}:
        selected_foreground, skeleton = selected > 0, sample['skeleton'] > 0
        if 'tree_length_detected' in requested:
            def length_detected():
                smooth = parameters['topology_smooth']
                denominator = np.count_nonzero(skeleton) + smooth
                return float((np.count_nonzero(selected_foreground & skeleton) + smooth) /
                             denominator) if denominator else 0.0
            scores['tree_length_detected'] = metric('tree_length_detected', length_detected)
        if 'branch_detected' in requested:
            scores['branch_detected'] = metric('branch_detected', lambda: _seg_bd(
                selected_foreground, sample['parsing'], skeleton, parameters['branch_coverage_threshold']))
    if 'betti_error' in requested:
        def betti_error():
            expected = 1
            if parameters['betti_reference'] == 'target':
                _, expected = _seg_ndi.label(target > 0, structure=structure)
            return float(abs(count - expected))
        scores['betti_error'] = metric('betti_error', betti_error)
    if requested & {'multiclass_dice', 'multiclass_cldice'}:
        classes = timed('target_classes', lambda: [label for label in np.unique(target) if label != 0])
        dice_scores, cldice_scores = [], []
        skeleton = sample['skeleton'] > 0 if 'multiclass_cldice' in requested else None
        for label in classes:
            target_class, prediction_class = timed('class_masks', lambda: (target == label, prediction == label))
            if 'multiclass_dice' in requested:
                dice_scores.append(metric('multiclass_dice', lambda: _seg_dice(
                    prediction_class, target_class, parameters['dice_smooth'])))
            if 'multiclass_cldice' in requested:
                def class_cldice():
                    class_skeleton = skeleton & target_class
                    if class_skeleton.any():
                        return (_seg_cldice(prediction_class, target_class, class_skeleton)
                                if prediction_class.any() else 0.0)
                    return None
                value = metric('multiclass_cldice', class_cldice)
                if value is not None:
                    cldice_scores.append(value)
        if 'multiclass_dice' in requested:
            scores['multiclass_dice'] = float(np.mean(dice_scores)) if dice_scores else 0.0
        if 'multiclass_cldice' in requested:
            scores['multiclass_cldice'] = float(np.mean(cldice_scores)) if cldice_scores else 0.0
    if requested & _SEG_BRANCH_NAMES:
        parsing = sample['parsing']
        node_count = int(parsing.max())
        actual = timed('target_projection', lambda: _seg_project_nodes(target, parsing, node_count))
        predicted = timed('prediction_projection', lambda: _seg_project_nodes(selected, parsing, node_count))
        valid = actual != 0
        if requested & {'branch_accuracy', 'branch_f1'}:
            shared_names = [name for name in names if name in ('branch_accuracy', 'branch_f1')]
            acc, f1 = timed('metric', lambda: _seg_acc_f1(
                actual, predicted, valid, 'branch_f1' in requested), shared_names)
            if 'branch_accuracy' in requested:
                scores['branch_accuracy'] = acc
            if 'branch_f1' in requested:
                scores['branch_f1'] = f1
        if 'subtree_consistency' in requested:
            scores['subtree_consistency'] = metric('subtree_consistency', lambda: _seg_sc(
                predicted, sample['graph'], parameters['subtree_ignore_labels']))
        if 'topological_distance' in requested:
            scores['topological_distance'] = metric('topological_distance', lambda: _seg_td(
                actual, predicted, valid, sample['graph'], parameters['missing_label_penalty']))
    return {name: scores[name] for name in names}
