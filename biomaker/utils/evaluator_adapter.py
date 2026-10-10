"""Compile data adapters against Harness-owned, standalone metric code.

An adapter interprets cells/files. It never owns sample alignment, scoring,
aggregation, submission validation, or the lifetime of preparation probes.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any


ADAPTER_API = """
Return one Python file implementing exactly these three public functions:

prediction_spec() -> dict
  Declare only data encoding details not already present in the prepared
  contract. Harness injects prepared keys, answer/output columns, task semantics,
  class order, metrics, prediction jobs and artifact paths before validation.
  You may repeat those declarations only with identical values. Never decide
  whether test exists: Harness runs the prepared prediction_jobs only.
  Declare file_reference_columns only for prediction cells containing file paths.
  Original column names and directory layouts are yours to interpret. Sample keys
  must agree with the prepared public splits; [] is allowed only for an explicit
  row-order contract. Predictions may have any declared column names.

load_sample(prediction_row, target_row, expected_row) -> dict
  Inputs are dictionaries for ONE already aligned sample. CSV cells are strings.
  prediction_row includes only declared keys and prediction columns; target_row
  includes only keys and answer columns. expected_row is public split metadata.
  Return {'prediction': ..., 'target': ...} and optional 'group', 'input_shape',
  'spacing', 'prediction_spacing'. When target_row is None, return prediction
  and public metadata only; NEVER read or discover hidden answers in that case.
  Arrays are full-size: do not crop, clamp, aggregate, compute metrics, tune
  thresholds, relabel instances, or change intensity scales to affect scoring.
  Restoration: finite 2D intensity arrays, input_shape from the public input.
  Segmentation: nonnegative integer 2D/3D masks; preserve instance IDs; spacing
  in numpy axis order. Classification: probability vector in class_labels order
  (one positive-class probability for binary allowed), scalar target, or binary
  vectors for multilabel. Detection: lists of (integer class_id, (x1,y1,x2,y2),
  confidence); target confidence can be 1. Convert coordinate representation to
  xyxy consistently; preserve all boxes, class IDs, and confidence values.
  For grouped restoration, declare group_column naming public expected_row
  metadata (any column name); Harness supplies/checks it and samples each group.
  Paths starting input/ are already
  workspace-relative. Resolve other paths explicitly from documented data layout.

write_dummy_predictions(expected_rows, artifact_path) -> None
  expected_rows is a list of public row dictionaries selected by Harness.
  Write exactly those keys, in any order, with valid synthetic predictions to
  artifact_path. Write referenced files under Path(artifact_path).parent, using
  distinct validation/submission directories chosen by Harness. Preserve paths
  in the CSV as workspace-relative paths, including working/.adapter_probe/.
  Use public inputs only, never hidden answers. Do not delete the CSV or files:
  Harness will read them again and owns cleanup after ALL checks finish.
  Do not require the complete split or particular sample counts/filenames.

Only these public functions are allowed. Name helper functions adapter_* and
constants ADAPTER_*. Use allowed installed packages, no network/install/training,
eval/exec, dynamic imports, or importing evaluator/biomaker. Do not implement
score, score_val, validate_submission, self_test, or standard metric functions.
Harness provides those, validates rows, runs small probes including explicit
relocated paths, freezes the accepted spec/code, and reuses it on every split.
""".strip()


def validate_adapter_source(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(m.split(".")[0] in {"evaluator", "biomaker", "importlib"} for m in modules):
                raise ValueError("Adapters cannot import evaluator/biomaker or dynamic module loaders.")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name in {"eval", "exec", "__import__", "import_module"}:
                raise ValueError("Adapters cannot use eval/exec or dynamic imports.")
    required = {"prediction_spec", "load_sample", "write_dummy_predictions"}
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
            if any(n in {"evaluator", "biomaker"} for n in names):
                raise ValueError("Adapters must be standalone data readers, not import evaluator/biomaker.")
            if any((a.asname or a.name).startswith("_h_") for a in node.names):
                raise ValueError("The _h_ namespace belongs to Harness.")
            continue
        if isinstance(node, ast.FunctionDef):
            if node.name not in required and not node.name.startswith("adapter_"):
                raise ValueError("Adapter helpers must use adapter_* names; Harness owns scoring functions.")
            if node.name in found:
                raise ValueError("Duplicate adapter function: " + node.name)
            found.add(node.name)
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if all(isinstance(t, ast.Name) and t.id.startswith("ADAPTER_") for t in targets):
                continue
        raise ValueError("Adapter top level may contain imports, functions, and ADAPTER_* constants only.")
    if not required <= found:
        raise ValueError("Missing adapter functions: " + ", ".join(sorted(required - found)))


def _legacy_adapter_spec(spec):
    """Import the standalone adapter-v1 declaration API at compilation only."""
    spec = dict(spec)
    if 'prediction_jobs' not in spec:
        roles = spec.get('public_splits')
        if spec.get('test_split', True) is None:
            if not isinstance(roles, list) or set(roles) != {'train', 'score_val'} or len(roles) != 2:
                raise ValueError('An explicit prediction_jobs contract or legacy train/validation-only declaration is required.')
        elif roles is not None and (
            not isinstance(roles, list) or len(roles) != 3
            or set(roles) != {'train', 'score_val', 'test'}
        ):
            raise ValueError('Legacy test_split disagrees with public_splits.')
        spec['prediction_jobs'] = {
            'score_val': {'expected': spec.get('score_split', './input/biomaker_splits/score_val.csv'),
                          'artifact': spec.get('validation_artifact', './working/validation_predictions.csv')},
        }
        if spec.get('test_split', True) is not None:
            spec['prediction_jobs']['test'] = {
                'expected': spec.get('test_split', './input/biomaker_splits/agent_test.csv'),
                'artifact': spec.get('test_submission', './working/submission.csv'),
            }
    return spec


def compile_adapter(source: str, frozen_spec: dict[str, Any] | None = None) -> str:
    """Bundle trusted kernels and adapter into one portable, frozen Python file."""
    validate_adapter_source(source)
    # Future statements from generated code must stay before all executable code.
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "prediction_spec":
            node.name = "_h_adapter_spec"
        elif isinstance(node, ast.Name) and node.id == "prediction_spec":
            node.id = "_h_adapter_spec"
    source = ast.unparse(tree)
    from .metric_source import frozen_metric_source

    kernels = frozen_metric_source()
    compatibility = ""
    if frozen_spec is None:
        # Historical standalone callers have no prepared ExperimentContract.
        # Resolve their declaration at this explicit API compatibility boundary.
        compatibility = inspect.getsource(_legacy_adapter_spec).replace(
            'def _legacy_adapter_spec(', 'def _h_legacy_adapter_spec('
        ) + "\n_h_compiled_spec = _h_legacy_adapter_spec(_h_adapter_spec())\n"
        spec_expression = "_h_compiled_spec"
    else:
        spec_expression = repr(_legacy_adapter_spec(frozen_spec))
    return (
        "from __future__ import annotations\n"
        "# Agent-owned data interpretation\n" + source + "\n"
        "_h_load_sample = load_sample\n"
        "_h_write_dummy = write_dummy_predictions\n"
        + compatibility
        + "# Harness-owned metric implementations (frozen source)\n" + kernels + "\n"
        + _RUNTIME.replace("__FROZEN_SPEC__", spec_expression)
    )


_RUNTIME = r'''
# Harness-owned alignment, lifecycle, and split-independent entry points.
import copy as _h_copy
import json as _h_json
import os as _h_os
import shutil as _h_shutil
import tempfile as _h_tempfile
import time as _h_time
import threading as _h_threading
from pathlib import Path as _h_Path
import pandas as _h_pd

_h_declared = __FROZEN_SPEC__
_h_source_spec = _h_declared
_h_source_spec = dict(_h_source_spec, metrics=normalize_metric_definitions(_h_source_spec), evaluator_api_version=3)
for _h_legacy_field in ('primary_metric', 'secondary_metrics', 'maximize'):
    _h_source_spec.pop(_h_legacy_field, None)
_h_spec = normalize_metric_spec(_h_source_spec)
_h_spec['scoring_backend'] = 'trusted_metrics_v1'
_h_spec['adapter_api_version'] = 1
_h_jobs = _h_spec['prediction_jobs']
_h_spec['score_split'] = _h_jobs['score_val']['expected']
_h_spec['validation_artifact'] = _h_jobs['score_val']['artifact']
_h_spec['test_split'] = _h_jobs.get('test', {}).get('expected')
_h_spec['test_submission'] = _h_jobs.get('test', {}).get('artifact')
_h_spec['public_splits'] = ['train', *_h_jobs]
_h_probe = None


def _h_validate_spec():
    if not isinstance(_h_jobs, dict) or 'score_val' not in _h_jobs:
        raise ValueError('prediction_jobs must declare the score_val prediction job.')
    for role, job in _h_jobs.items():
        if not isinstance(job, dict) or not all(isinstance(job.get(key), str) and job[key] for key in ('expected', 'artifact')):
            raise ValueError('Each prediction job requires expected and artifact paths: ' + role)
        path = _h_Path(job['artifact']).resolve()
        if not path.is_relative_to(_h_Path('working').resolve()):
            raise ValueError('Prediction artifacts must be under working/.')
    for name in ('key_columns', 'prediction_columns', 'answer_columns'):
        values = _h_spec.get(name)
        if (not isinstance(values, list) or (name != 'key_columns' and not values)
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('Adapter spec requires distinct column names in ' + name)
    if set(_h_spec['key_columns']) & (set(_h_spec['prediction_columns']) | set(_h_spec['answer_columns'])):
        raise ValueError('Sample keys cannot be predictions or hidden answers.')
    protocol = _h_spec.get('restoration_evaluation', {})
    if protocol.get('groups') is not None and not protocol.get('group_column'):
        raise ValueError('Grouped adapters must declare a public group_column for representative probes.')


_h_validate_spec()


def prediction_spec():
    return _h_copy.deepcopy({k: v for k, v in _h_spec.items() if k != 'answer_columns'})


def _h_adapter_spec():
    return _h_copy.deepcopy(_h_spec)


def _h_read(path):
    return _h_pd.read_csv(path, dtype=str, keep_default_na=False)


def _h_expected(path):
    frame = _h_read(path)
    keys = _h_spec['key_columns']
    if not len(frame) or any(k not in frame for k in keys):
        raise ValueError('Empty expected split or missing sample key columns.')
    if keys and ((frame[keys] == '').any().any() or frame.duplicated(keys).any()):
        raise ValueError('Expected sample keys must be nonempty and unique.')
    # Legacy public validation tables sometimes still include labels. They are
    # never passed to a data adapter as public metadata: targets come exclusively
    # from the separately supplied truth path, including in preparation probes.
    return frame.drop(columns=_h_spec['answer_columns'], errors='ignore')


def _h_align(path, expected, columns, label):
    keys = _h_spec['key_columns']
    frame = _h_read(path)
    error_type = PredictionValidationError if label == 'Predictions' else ValueError
    required = keys + columns
    if any(c not in frame for c in required):
        raise error_type(label + ' missing declared columns.')
    if len(frame) != len(expected):
        raise error_type(label + ' missing rows.')
    if keys and (frame[keys] == '').any().any():
        raise error_type(label + ' contains empty sample keys.')
    if label == 'Predictions' and _h_spec.get('prediction_type') in {
            'probability_columns', 'numeric_columns', 'continuous_scalar', 'continuous_vector'}:
        values = frame[columns].apply(_h_pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise PredictionValidationError('Declared numeric predictions must contain finite numeric values.')
    # Empty encoded values can mean a valid negative image (boxes/RLE), or a
    # dataset-defined missing value. The adapter interprets cells; canonical
    # sample validation determines whether the decoded values are admissible.
    frame = frame[required]
    if not keys:
        return frame.reset_index(drop=True)
    if frame.duplicated(keys).any():
        raise error_type(label + ' contains duplicate sample keys.')
    if set(map(tuple, frame[keys].values)) != set(map(tuple, expected[keys].values)):
        raise error_type(label + ' sample keys differ from expected rows.')
    return expected[keys].merge(frame, on=keys, sort=False, validate='one_to_one')


def _h_samples(predictions, targets, expected):
    p = predictions.to_dict('records')
    t = targets.to_dict('records') if targets is not None else [None] * len(expected)
    group_column = _h_spec.get('restoration_evaluation', {}).get('group_column')
    for index, (pr, tr, row) in enumerate(zip(p, t, expected.to_dict('records'))):
        try:
            sample = _h_load_sample(pr, tr, row)
            if not isinstance(sample, dict) or 'prediction' not in sample:
                raise ValueError('load_sample must return a prediction mapping.')
            if tr is not None and 'target' not in sample:
                raise ValueError('load_sample must return target when target_row is provided.')
            if tr is None and 'target' in sample:
                raise ValueError('Format-only adaptation must not return hidden targets.')
            if _h_spec['task_family'] == 'image_restoration' and sample.get('input_shape') is None:
                raise ValueError('Restoration adapters must return input_shape from public input metadata.')
            if group_column:
                if group_column not in row or not row[group_column]:
                    raise ValueError('Missing declared public group metadata.')
                if 'group' in sample and str(sample['group']) != row[group_column]:
                    raise ValueError('Adapted group disagrees with public metadata.')
                sample['group'] = row[group_column]
        except Exception as exc:
            # Do not forward exception messages containing private rows/values.
            if tr is not None:
                raise ValueError('Data adapter failed for aligned sample index %d (%s); check load_sample and declared columns.' % (index, type(exc).__name__)) from None
            raise
        yield sample
        # Release the generator's reference before decoding the next volume.
        # Completed workers must not leave an extra case live during loading.
        del sample


def score(prediction_path, truth_path, expected_path):
    expected = _h_expected(expected_path)
    predictions = _h_align(prediction_path, expected, _h_spec['prediction_columns'], 'Predictions')
    targets = _h_align(truth_path, expected, _h_spec['answer_columns'], 'Answers')
    # Execution policy is supplied by Harness, separate from the frozen metric
    # protocol. Split-level metrics such as AUC/AP keep their original reduction.
    workers = int(_h_os.environ.get('BIOMAKER_EVALUATOR_WORKERS', '1'))
    if workers < 1:
        raise ValueError('BIOMAKER_EVALUATOR_WORKERS must be positive')
    if _h_spec['task_family'] not in ('segmentation', 'image_restoration'):
        workers = 1
    progress_path = _h_os.environ.get('BIOMAKER_EVALUATOR_PROGRESS_PATH')
    lock = _h_threading.Lock()

    def report(event):
        # Only execution metadata: no private paths, labels or metric values.
        if progress_path:
            with lock, open(progress_path, 'a', encoding='utf-8') as output:
                output.write(_h_json.dumps(dict(event, monotonic_seconds=_h_time.monotonic())) + '\n')

    result = score_samples(_h_samples(predictions, targets, expected), _h_spec,
                           progress_callback=report if progress_path else None,
                           workers=workers)
    result['n'] = len(expected)
    result['score_schema_version'] = 1
    result['status'] = 'ok'
    return result


def score_val():
    job = _h_jobs['score_val']
    return score(job['artifact'], _h_os.environ['BIOMAKER_SCORE_VAL_PATH'], job['expected'])


def _h_validate_submission(prediction_path, expected_path):
    expected = _h_expected(expected_path)
    predictions = _h_align(prediction_path, expected, _h_spec['prediction_columns'], 'Predictions')
    for sample in _h_samples(predictions, None, expected):
        validate_prediction(sample, _h_spec)
    return {'valid': True, 'n': len(expected)}


def validate_submission():
    jobs = {role: job for role, job in _h_jobs.items() if role != 'score_val'}
    if not jobs:
        return {'valid': True, 'skipped': True, 'reason': 'no_test_split', 'n': 0}
    results = [_h_validate_submission(job['artifact'], job['expected']) for job in jobs.values()]
    return {'valid': True, 'n': sum(result['n'] for result in results)}


def _h_detection_probe_target(frame, truth):
    # Negative images are valid, but an all-negative subset cannot exercise AP
    # or recall. Ask the same reader for canonical targets; do not infer any
    # dataset's JSON, file, or column encoding here. Only public rows reach the
    # writer, and each small batch's fixtures are removed after private decoding.
    root = _h_Path('working/.adapter_probe/selection')
    for start in range(0, len(frame), 4):
        batch = frame.iloc[start:start + 4]
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'predictions.csv'
        try:
            _h_write_dummy(batch.to_dict('records'), str(path))
            predictions = _h_align(path, batch, _h_spec['prediction_columns'], 'Predictions')
            for offset, sample in enumerate(_h_samples(predictions, truth.iloc[start:start + 4], batch)):
                if len(sample['target']):
                    return start + offset
        finally:
            _h_shutil.rmtree(root)
    return None


def _h_classification_probe_targets(frame, truth):
    # Raw answer cells may be file paths or serialized records, not labels.
    # Decode privately in small batches, retaining only indices and coverage.
    # The writer sees public rows, never their answers or decoded labels.
    root = _h_Path('working/.adapter_probe/selection')
    multilabel = _h_spec['task_type'] == 'multilabel'
    counts = {} if multilabel else {str(label): 0 for label in _h_spec['class_labels']}
    selected = []
    for start in range(0, len(frame), 4):
        batch = frame.iloc[start:start + 4]
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'predictions.csv'
        try:
            _h_write_dummy(batch.to_dict('records'), str(path))
            predictions = _h_align(path, batch, _h_spec['prediction_columns'], 'Predictions')
            for offset, raw in enumerate(_h_samples(predictions, truth.iloc[start:start + 4], batch)):
                sample = _validate_sample(raw, _h_spec, require_target=True)
                if multilabel:
                    if not counts:
                        counts = {(index, value): 0 for index in range(len(sample['target'])) for value in (0, 1)}
                    categories = [(index, int(value)) for index, value in enumerate(sample['target'])]
                else:
                    categories = [str(sample['target'])]
                if any(counts[category] < 2 for category in categories):
                    selected.append(start + offset)
                    for category in categories:
                        counts[category] += 1
                if counts and all(count >= 2 for count in counts.values()):
                    return selected
        finally:
            _h_shutil.rmtree(root, ignore_errors=True)
    return selected


def _h_select_probe(frame, truth=None):
    # At least two examples of each declared group/class for stable smoke tests.
    selected = list(range(min(4, len(frame))))
    group = _h_spec.get('restoration_evaluation', {}).get('group_column')
    if group and group in frame:
        for indices in frame.groupby(group, sort=False).indices.values():
            selected.extend(list(indices[:2]))
    if truth is not None and _h_spec.get('task_family') == 'classification':
        selected.extend(_h_classification_probe_targets(frame, truth))
    if truth is not None and _h_spec.get('task_family') == 'detection':
        positive = _h_detection_probe_target(frame, truth)
        if positive is not None:
            selected.append(positive)
    return sorted(set(selected))


def self_test():
    global _h_probe
    cleanup_self_test()
    # Private probe answers stay outside the agent-visible workspace.
    _h_probe = _h_tempfile.TemporaryDirectory(prefix='biomaker-adapter-probe-')
    root = _h_Path(_h_probe.name)
    fixtures = _h_Path('working/.adapter_probe')
    fixtures.mkdir(parents=True, exist_ok=True)
    expected = _h_expected(_h_spec['score_split'])
    truth = _h_align(_h_os.environ['BIOMAKER_SCORE_VAL_PATH'], expected, _h_spec['answer_columns'], 'Answers')
    indices = _h_select_probe(expected, truth)
    expected = expected.iloc[indices]
    truth = truth.iloc[indices]
    expected_path, truth_path = root / 'expected.csv', root / 'truth.csv'
    expected.to_csv(expected_path, index=False)
    truth.to_csv(truth_path, index=False)
    prediction_path = fixtures / 'validation/predictions.csv'
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    _h_write_dummy(expected.to_dict('records'), str(prediction_path))
    predictions = _h_align(prediction_path, expected, _h_spec['prediction_columns'], 'Predictions')
    # A valid synthetic prediction can have an undefined metric against real
    # answers (e.g. nonempty dummy versus empty HD95 target). Validate EVERY
    # decoded sample before accepting only that typed mathematical outcome.
    # This is a separate streaming pass so large arrays are never retained.
    for sample in _h_samples(predictions, truth, expected):
        _validate_sample(sample, _h_spec, require_target=True)
    result, score_error = None, None
    try:
        result = score(str(prediction_path), str(truth_path), str(expected_path))
    except UndefinedMetricError as exc:
        score_error = {'metric': exc.metric, 'reason': exc.reason}
    for role, job in _h_jobs.items():
        if role == 'score_val':
            continue
        test = _h_expected(job['expected'])
        test = test.iloc[_h_select_probe(test)]
        test_path = root / (role + '.csv')
        test.to_csv(test_path, index=False)
        submission = fixtures / role / 'predictions.csv'
        submission.parent.mkdir(parents=True, exist_ok=True)
        _h_write_dummy(test.to_dict('records'), str(submission))
        _h_validate_submission(submission, test_path)
    print('EVALUATOR_OK')
    # Caller re-reads these paths for the relocated, environment-free probe.
    return {'prediction_path': str(prediction_path), 'truth_path': str(truth_path),
            'expected_path': str(expected_path), 'score': result,
            'score_error': score_error,
            'structural_probe': {'validated_samples': len(expected)}}


def cleanup_self_test():
    global _h_probe
    if _h_probe is not None:
        _h_probe.cleanup()
        _h_probe = None
'''
