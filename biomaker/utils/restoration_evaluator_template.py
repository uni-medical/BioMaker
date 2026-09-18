"""Standalone scoring: explicit file paths, no dataset-specific path inference."""

from pathlib import Path


_RESTORATION_DATA_ADAPTER = r'''
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

ADAPTER_SPEC = __BIOMAKER_SPEC__

def prediction_spec():
    return ADAPTER_SPEC

def adapter_image(path):
    if Path(path).suffix.lower() == '.npy':
        raw = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            if getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Multi-frame images require a data adapter with explicit sample/axis semantics; the default reader cannot select only the first frame.')
            if image.mode not in ('L', 'I', 'F', 'I;16', 'I;16B', 'I;16L'):
                raise ValueError('Expected a grayscale intensity image.')
            raw = np.asarray(image)
    if raw.dtype.kind not in 'uif' or raw.ndim != 2 or not raw.size or not np.isfinite(raw).all():
        raise ValueError('Expected a finite, nonempty 2D intensity array.')
    return raw

def adapter_input(row):
    protocol = prediction_spec()['restoration_evaluation']
    image = adapter_image(row[protocol['input_column']])
    if image.min() < 0 or image.max() > protocol['data_range']:
        raise ValueError('Public input intensities outside declared data_range.')
    return image

def load_sample(prediction_row, target_row, expected_row):
    spec = prediction_spec()
    sample = {'prediction': adapter_image(prediction_row[spec['prediction_columns'][0]]),
              'input_shape': adapter_input(expected_row).shape}
    if target_row is not None:
        sample['target'] = adapter_image(target_row[spec['answer_columns'][0]])
    return sample

def write_dummy_predictions(expected_rows, artifact_path):
    spec = prediction_spec()
    root = Path(artifact_path).parent
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in enumerate(expected_rows):
        path = root / ('image_%d.npy' % index)
        image = adapter_input(row)
        np.save(path, np.full(image.shape, spec['restoration_evaluation']['data_range'] * .501))
        rows.append({**{key: row[key] for key in spec['key_columns']},
                     spec['prediction_columns'][0]: str(path)})
    pd.DataFrame(rows).to_csv(artifact_path, index=False)
'''


# Embed the same Harness-owned kernel used by generated data adapters. Frozen
# evaluators run without requiring the biomaker package in candidate environments.
_FIXED_RESTORATION_EVALUATOR = Path(__file__).with_name('trusted_metrics.py').read_text(encoding='utf-8') + '\n' + r'''
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

SPEC = __BIOMAKER_SPEC__
P = SPEC['restoration_evaluation']
METRICS = [item['name'] for item in SPEC['metrics']] if 'metrics' in SPEC else list(dict.fromkeys([SPEC['primary_metric'], *SPEC['secondary_metrics']]))


def prediction_spec():
    return {k: v for k, v in SPEC.items() if k != 'answer_columns'}


def _expected(path):
    keys = SPEC['key_columns']
    df = pd.read_csv(path, dtype=str)
    cols = list(dict.fromkeys(keys + [P['input_column']] + ([P['group_column']] if P['group_column'] else [])))
    if any(c not in df for c in cols) or not len(df):
        raise ValueError('Missing expected sample metadata or empty split.')
    if df[cols].isna().any().any() or df.duplicated(keys).any():
        raise ValueError('Expected sample metadata must be complete and unique.')
    if P['group_column'] and set(df[P['group_column']]) != set(P['groups']):
        raise ValueError('Expected rows must contain all and only the declared groups.')
    return df


def _aligned(path, expected, col):
    keys = SPEC['key_columns']
    df = pd.read_csv(path, dtype=str)
    if any(c not in df for c in keys + [col]):
        raise ValueError('Missing sample or image-path columns.')
    if len(df) != len(expected) or df[keys + [col]].isna().any().any() or df.duplicated(keys).any():
        raise ValueError('Missing or duplicate samples/paths.')
    if set(map(tuple, df[keys].values)) != set(map(tuple, expected[keys].values)):
        raise ValueError('Sample keys do not match expected rows.')
    return expected[keys].merge(df[keys + [col]], on=keys, validate='one_to_one')[col]


def _image(path, prediction=False):
    if Path(path).suffix.lower() == '.npy':
        raw = np.load(path, allow_pickle=False)
        if raw.dtype.kind not in 'uif':
            raise ValueError('Expected a real numeric intensity array.')
        a = np.asarray(raw, dtype=np.float64)
    else:
        with Image.open(path) as im:
            if im.mode not in ('L', 'I', 'F', 'I;16', 'I;16B', 'I;16L'):
                raise ValueError('Expected a grayscale intensity image, not RGB/palette/mask conversion.')
            a = np.asarray(im, dtype=np.float64)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError('Image must be finite, nonempty and two-dimensional.')
    if prediction:
        return np.clip(a, 0, P['data_range'])
    if a.min() < 0 or a.max() > P['data_range']:
        raise ValueError('Input/reference intensity outside declared range.')
    return a


def _crops(a):
    crops = restoration_crops(a, P['crop_size'], P.get('crop_mode'))
    if 'ssim' in METRICS:
        window = SPEC['ssim_parameters']['win_size']
        if any(min(crop.shape) < window for crop in crops):
            raise ValueError('SSIM requires image dimensions >= %s.' % window)
    return crops


def _metrics(truth, pred):
    return restoration_metrics(truth, pred, P['data_range'],
                               SPEC['ssim_parameters'], metrics=METRICS)


def score(prediction_path, truth_path, expected_path):
    expected = _expected(expected_path)
    pred = _aligned(prediction_path, expected, SPEC['prediction_columns'][0])
    truth = _aligned(truth_path, expected, SPEC['answer_columns'][0])
    grouped = {}
    samples = []
    for i, row in expected.iterrows():
        inp = _image(row[P['input_column']])
        t, p = _image(truth.iloc[i]), _image(pred.iloc[i], prediction=True)
        if t.shape != inp.shape or p.shape != inp.shape:
            raise ValueError('Input, reference and prediction shapes must match exactly.')
        scores = [_metrics(a,b) for a,b in zip(_crops(t), _crops(p))]
        sample = {m: float(np.mean([v[m] for v in scores])) for m in METRICS}
        samples.append(sample)
        if P['group_column']:
            grouped.setdefault(row[P['group_column']], []).append(sample)
    per_group = {g: {m: float(np.mean([v[m] for v in rows])) for m in METRICS}
                 for g, rows in grouped.items()}
    source = list(per_group.values()) if P['aggregation'] == 'mean_groups' else samples
    metrics = {m: float(np.mean([v[m] for v in source])) for m in METRICS}
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError('Non-finite restoration score.')
    # Flat metric keys survive the shared private-report serialization.
    for g, values in per_group.items():
        metrics.update({m + '/' + g: v for m,v in values.items()})
    return {**({'metric': metrics[SPEC['primary_metric']], 'maximize': True} if 'metrics' not in SPEC else {}),
            'metrics': metrics, 'per_group': per_group, 'n': len(expected)}


def score_val():
    return score(SPEC['validation_artifact'], os.environ['BIOMAKER_SCORE_VAL_PATH'], SPEC['score_split'])


def validate_submission():
    expected = _expected('./input/biomaker_splits/agent_test.csv')
    pred = _aligned(SPEC['test_submission'], expected, SPEC['prediction_columns'][0])
    for i, row in expected.iterrows():
        inp, out = _image(row[P['input_column']]), _image(pred.iloc[i], prediction=True)
        if inp.shape != out.shape:
            raise ValueError('Prediction shape must match input.')
        _crops(out)
    return {'valid': True}


def self_test():
    # The harness probes these artifacts again, then removes working outputs.
    tmp = Path('./working/.restoration_self_test')
    tmp.mkdir(parents=True, exist_ok=True)
    by_shape = {}
    for split, artifact in ((SPEC['score_split'], SPEC['validation_artifact']),
                            ('./input/biomaker_splits/agent_test.csv', SPEC['test_submission'])):
        frame = _expected(split)
        paths = []
        for value in frame[P['input_column']]:
            arr = _image(value)
            if arr.shape not in by_shape:
                out = tmp / ('shape_%s.tif' % len(by_shape))
                Image.fromarray(np.full(arr.shape, P['data_range']*0.501, dtype=np.float32)).save(out)
                by_shape[arr.shape] = str(out)
            paths.append(by_shape[arr.shape])
        result = frame[SPEC['key_columns']].copy()
        result[SPEC['prediction_columns'][0]] = paths
        Path(artifact).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(artifact, index=False)
    score_val()
    validate_submission()
    print('EVALUATOR_OK')
'''
