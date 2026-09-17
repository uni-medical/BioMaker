"""Standalone, frozen detection scorer for explicit JSON box contracts."""

_FIXED_DETECTION_EVALUATOR = r"""
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd

SPEC = __BIOMAKER_SPEC__


def prediction_spec():
    return {k: v for k, v in SPEC.items() if k not in {'answer_columns', 'truth_schema'}}


def _parse_boxes(value, prediction=False):
    items = json.loads(value) if isinstance(value, str) else value
    if not isinstance(items, list):
        raise ValueError('Boxes must be a JSON list; use [] for no detections.')
    schema = SPEC['prediction_schema' if prediction else 'truth_schema']
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Each box must be an object with declared coordinate/class keys.')
        if any(item.get(k, False) for k in ('iscrowd', 'ignore', 'difficult')):
            raise ValueError('Crowd/ignore/difficult boxes require a different evaluation protocol.')
        cls = item[schema['class_key']]
        if isinstance(cls, bool) or not isinstance(cls, int) or cls < 0:
            raise ValueError('Class IDs must be nonnegative integers.')
        keys = schema['coordinate_keys']
        raw = item['bbox'] if keys == ['bbox'] else [item[k] for k in keys]
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError('Each bbox must contain exactly four coordinates.')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in raw):
            raise ValueError('Coordinates must be finite numbers.')
        a, b, c, d = map(float, raw)
        if not all(math.isfinite(x) for x in (a, b, c, d)):
            raise ValueError('Coordinates must be finite numbers.')
        fmt = schema['box_format']
        if fmt == 'cxcywh':
            box = (a-c/2, b-d/2, a+c/2, b+d/2)
        elif fmt == 'xywh':
            box = (a, b, a+c, b+d)
        elif fmt == 'xyxy':
            box = (a, b, c, d)
        else:
            raise ValueError('Unsupported declared box format.')
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Boxes must have positive width and height.')
        score = item['score'] if prediction else 1.0
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Prediction score must be finite and in [0,1].')
        result.append((cls, box, float(score)))
    return result


def _read_aligned(path, expected, columns):
    keys = SPEC['key_columns']
    df = pd.read_csv(path, dtype={k: str for k in keys})
    if any(k not in df for k in keys + columns):
        raise ValueError('Missing required sample/output columns.')
    if len(df) != len(expected) or df[keys + columns].isna().any().any() or df.duplicated(keys).any():
        raise ValueError('Missing/duplicate rows or values in scoring table.')
    if set(map(tuple, df[keys].values)) != set(map(tuple, expected[keys].values)):
        raise ValueError('Scoring sample keys do not match expected rows.')
    return expected[keys].merge(df[keys + columns], on=keys, validate='one_to_one')


def _iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union


def _ap(tp, total):
    if not len(tp):
        return 0.0
    recalls = np.cumsum(tp) / total
    precisions = np.cumsum(tp) / np.arange(1, len(tp)+1)
    recalls = np.r_[0., recalls, 1.]
    precisions = np.r_[0., precisions, 0.]
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    return float(np.sum(np.diff(recalls) * precisions[1:]))


def _evaluate(truth, predictions):
    classes = sorted({c for boxes in truth for c, _, _ in boxes})
    if not classes:
        raise ValueError('AP and Recall are undefined for a split without ground-truth boxes.')
    per_class = {}
    total_hits = total_gt = 0
    for cls in classes:
        gt = {i: [b for c,b,_ in boxes if c == cls] for i,boxes in enumerate(truth)}
        count = sum(map(len, gt.values()))
        detections = [(s,i,j,b) for i,boxes in enumerate(predictions)
                      for j,(c,b,s) in enumerate(boxes) if c == cls]
        # Deterministic ties: expected image order, then submitted box order.
        detections.sort(key=lambda x: (-x[0], x[1], x[2]))
        used = set()
        tp = []
        recall_hits = 0
        for confidence, i, _, box in detections:
            overlaps = [_iou(box, other) for other in gt[i]]
            match = int(np.argmax(overlaps)) if overlaps else -1
            # VOC: highest-IoU GT, then check whether it was already matched.
            hit = match >= 0 and overlaps[match] >= SPEC['iou_threshold'] and (i,match) not in used
            tp.append(int(hit))
            if hit:
                used.add((i,match))
                recall_hits += int(confidence >= SPEC['recall_confidence_threshold'])
        per_class[str(cls)] = {'ap50': _ap(tp, count), 'recall': recall_hits/count,
                               'ground_truth': count, 'true_positives': recall_hits}
        total_hits += recall_hits
        total_gt += count
    return {'ap50': float(np.mean([v['ap50'] for v in per_class.values()])),
            'recall': total_hits/total_gt}, per_class


def score(prediction_path, truth_path, expected_path):
    keys = SPEC['key_columns']
    expected = pd.read_csv(expected_path, dtype={k: str for k in keys})
    if not len(expected) or expected[keys].isna().any().any() or expected.duplicated(keys).any():
        raise ValueError('Expected sample keys must be nonempty and unique.')
    pc, tc = SPEC['prediction_columns'][0], SPEC['answer_columns'][0]
    pred = _read_aligned(prediction_path, expected, [pc])
    true = _read_aligned(truth_path, expected, [tc])
    metrics, per_class = _evaluate([_parse_boxes(v) for v in true[tc]],
                                  [_parse_boxes(v, True) for v in pred[pc]])
    return {'metric': metrics[SPEC['primary_metric']], 'maximize': True,
            'metrics': metrics, 'per_class': per_class, 'n': len(expected)}


def score_val():
    return score(SPEC['validation_artifact'], os.environ['BIOMAKER_SCORE_VAL_PATH'], SPEC['score_split'])


def validate_submission():
    expected = pd.read_csv('./input/biomaker_splits/agent_test.csv', dtype={k: str for k in SPEC['key_columns']})
    col = SPEC['prediction_columns'][0]
    pred = _read_aligned(SPEC['test_submission'], expected, [col])
    for value in pred[col]:
        _parse_boxes(value, True)
    return {'valid': True}


def self_test():
    for split, artifact in [(SPEC['score_split'], SPEC['validation_artifact']),
                            ('./input/biomaker_splits/agent_test.csv', SPEC['test_submission'])]:
        frame = pd.read_csv(split, dtype={k: str for k in SPEC['key_columns']})[SPEC['key_columns']]
        frame[SPEC['prediction_columns'][0]] = '[]'
        Path(artifact).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(artifact, index=False)
    assert score_val()['metric'] == 0.0
    validate_submission()
    print('EVALUATOR_OK')
"""
