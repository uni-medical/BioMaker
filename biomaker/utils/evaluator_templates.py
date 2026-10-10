"""Deterministic evaluator source templates."""

# Harness-owned probes run only on disposable preparation fixtures.
_STRICT_PROBE_HELPERS = r'''
import hashlib
import math
import json
from pathlib import Path
import pandas as pd

def _fingerprint(paths):
    result = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            result[str(p)] = None
        else:
            stat = p.stat()
            result[str(p)] = (stat.st_mtime_ns, stat.st_ctime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
    return result

def _legacy_failure(value):
    if isinstance(value, dict):
        return any(value.get(k) for k in ('error','fallback','fallback_used','used_fallback')) or value.get('valid') is False or value.get('ok') is False or any(_legacy_failure(v) for v in value.values())
    return isinstance(value, list) and any(_legacy_failure(v) for v in value)

def _check_probe_result(value, spec, scoring):
    if not scoring:
        if value is True or (isinstance(value, dict) and not _legacy_failure(value) and (value.get('valid') is True or value.get('ok') is True)):
            return
        raise ValueError('Submission validation must explicitly succeed')
    if not isinstance(value, dict):
        raise ValueError('Scoring must return a dictionary')
    if value.get('score_schema_version') == 1:
        if value.get('status') != 'ok' or value.get('failure') is not None or any(value.get(k) for k in ('error','fallback','fallback_used','used_fallback')):
            raise ValueError('Scoring did not explicitly succeed')
    elif _legacy_failure(value):
        raise ValueError('Scoring returned failure/fallback markers')
    declared = spec.get('metrics')
    if declared:
        wanted = [m['name'] for m in declared]
        metrics = value.get('metrics')
        if not isinstance(metrics, dict) or not set(wanted).issubset(metrics):
            raise ValueError('Scoring must return every frozen metric')
        values = list(metrics.values())
    else:
        values = [value.get('metric')]
        if not isinstance(value.get('maximize', True), bool):
            raise ValueError('maximize must be a boolean')
        if 'maximize' in spec and value.get('maximize', True) != spec['maximize']:
            raise ValueError('maximize disagrees with the frozen metric contract')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('All ranking metrics must be finite numbers')

def _probe_call(call, paths, spec, scoring=True, reject=False, case='valid', allow_undefined=False):
    before = _fingerprint(paths)
    rejected = False
    try:
        _check_probe_result(call(), spec, scoring)
    except Exception as exc:
        undefined = isinstance(exc, getattr(evaluator, 'UndefinedMetricError', ()))
        if undefined and reject:
            raise RuntimeError('Corrupt predictions reached metric computation: ' + case) from exc
        if not reject and not (allow_undefined and undefined):
            raise
        rejected = True
    finally:
        if before != _fingerprint(paths):
            raise RuntimeError('Evaluator modified an artifact during ' + case)
    if reject and not rejected:
        raise RuntimeError('Evaluator accepted corrupted contract case: ' + case)

def _negative_prediction_probes(path, call, watched, spec, scoring=True, allow_undefined=False):
    path = Path(path)
    original = path.read_bytes()
    frame = pd.read_csv(path)
    keys = list(spec.get('key_columns', []))
    predictions = list(spec.get('prediction_columns', []))
    refs = list(spec.get('file_reference_columns', []))
    required = list(dict.fromkeys(keys + predictions + refs))
    _probe_call(call, watched, spec, scoring, allow_undefined=allow_undefined)
    cases = [('missing_column_' + c, frame.drop(columns=[c])) for c in required if c in frame]
    cases.append(('missing_row', frame.iloc[:-1].copy()))
    if len(frame) and keys:
        wrong = frame.copy()
        wrong[keys[0]] = wrong[keys[0]].astype(object)
        wrong.loc[wrong.index[0], keys[0]] = '__biomaker_unknown_key__'
        cases.append(('unexpected_key', wrong))
        if len(frame) > 1:
            duplicate = frame.copy()
            duplicate.loc[duplicate.index[1], keys] = duplicate.iloc[0][keys].to_numpy()
            cases.append(('duplicate_key', duplicate))
    if len(frame) and predictions:
        for label, bad in [('nan', float('nan')), ('infinite', float('inf')), ('nonnumeric', 'not-a-number')]:
            corrupt = frame.copy()
            corrupt[predictions[0]] = corrupt[predictions[0]].astype(object)
            corrupt.loc[corrupt.index[0], predictions[0]] = bad
            cases.append((label, corrupt))
    try:
        for case, corrupt in cases:
            corrupt.to_csv(path, index=False)
            _probe_call(call, watched, spec, scoring, reject=True, case=case)
    finally:
        path.write_bytes(original)
'''

_FIXED_CLASSIFICATION_EVALUATOR = r"""
import math
import os

import numpy as np
import pandas as pd

SPEC = __BIOMAKER_SPEC__
PUBLIC_SPEC_KEYS = [
    "validation_artifact",
    "test_submission",
    "score_split",
    "key_columns",
    "prediction_columns",
    "primary_metric",
    "maximize",
    "task_type",
    "prediction_type",
    "class_labels",
    "positive_class",
    "metrics",
    "secondary_metrics",
    "evaluator_api_version",
]


def prediction_spec():
    return {key: SPEC[key] for key in PUBLIC_SPEC_KEYS if key in SPEC}


def _path(value):
    text = str(value).removeprefix("./")
    return text if os.path.isabs(text) else os.path.join(".", text)


def _read_csv(path):
    try:
        return pd.read_csv(_path(path))
    except Exception as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


def _key_frame(df, key_columns):
    out = df[key_columns].copy()
    for column in key_columns:
        out[column] = out[column].astype(str)
    return out


def _validate_prediction_file(path, expected_split_path):
    spec = prediction_spec()
    key_columns = list(spec["key_columns"])
    prediction_columns = list(spec["prediction_columns"])
    required = key_columns + prediction_columns
    pred = _read_csv(path)
    missing = [column for column in required if column not in pred.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    pred = pred[required].copy()
    if pred[required].isna().any().any():
        raise ValueError(f"{path} contains missing keys or predictions.")
    if key_columns and pred.duplicated(key_columns).any():
        raise ValueError(f"{path} contains duplicate keys.")

    expected = _read_csv(expected_split_path)
    missing_keys = [column for column in key_columns if column not in expected.columns]
    if missing_keys:
        raise ValueError(f"{expected_split_path} missing key columns: {missing_keys}")
    if len(pred) != len(expected):
        raise ValueError(
            f"{path} row count {len(pred)} does not match expected {len(expected)}."
        )
    if key_columns:
        expected_keys = _key_frame(expected, key_columns)
        pred_keys = _key_frame(pred, key_columns)
        merged = expected_keys.merge(pred_keys, on=key_columns, how="outer", indicator=True)
        if not (merged["_merge"] == "both").all():
            missing_count = int((merged["_merge"] == "left_only").sum())
            extra_count = int((merged["_merge"] == "right_only").sum())
            raise ValueError(
                f"{path} key mismatch: missing={missing_count}, extra={extra_count}."
            )

    values = pred[prediction_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{path} predictions must be finite numeric values.")
    if ((values < -1e-8) | (values > 1.0000001)).any().any():
        raise ValueError(f"{path} probability values must be in [0, 1].")
    if spec.get("task_type") in {"binary", "multiclass"} and len(prediction_columns) > 1:
        row_sums = values.sum(axis=1).to_numpy(dtype=float)
        if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError(f"{path} class probability rows must sum to 1.")
    pred[prediction_columns] = values
    return pred


def validate_submission():
    spec = prediction_spec()
    _validate_prediction_file(spec["test_submission"], "./input/biomaker_splits/agent_test.csv")
    return {"valid": True}


def _score_auc(y_true, scores, spec):
    try:
        from sklearn.metrics import roc_auc_score
    except Exception as exc:
        return None, str(exc)
    try:
        if spec.get("task_type") == "multilabel":
            value = float(roc_auc_score(y_true, scores, average="macro"))
            if not math.isfinite(value):
                return None, "ROC AUC is undefined for this validation label distribution."
            return value, None
        labels = list(spec.get("class_labels") or [])
        if scores.shape[1] <= 2 and len(labels) <= 2:
            positive = str(spec.get("positive_class", labels[-1] if labels else "1"))
            binary_true = np.asarray([str(value) == positive for value in y_true], dtype=int)
            score_column = scores[:, -1] if scores.shape[1] == 2 else scores[:, 0]
            value = float(roc_auc_score(binary_true, score_column))
            if not math.isfinite(value):
                return None, "ROC AUC is undefined for this validation label distribution."
            return value, None
        label_index = {str(label): index for index, label in enumerate(labels)}
        y_index = np.asarray([label_index[str(value)] for value in y_true], dtype=int)
        value = float(
            roc_auc_score(
                y_index,
                scores,
                average="macro",
                multi_class="ovr",
                labels=list(range(len(labels))),
            )
        )
        if not math.isfinite(value):
            return None, "ROC AUC is undefined for this validation label distribution."
        return value, None
    except Exception as exc:
        return None, str(exc)


def _wants_auc(spec):
    metric = str(spec.get("primary_metric", "")).lower()
    return "auc" in metric or "roc" in metric


def score_val():
    return score(SPEC["validation_artifact"], os.environ["BIOMAKER_SCORE_VAL_PATH"],
                 SPEC.get("score_split", "./input/biomaker_splits/score_val.csv"))


def score(prediction_path, truth_path, expected_path):
    spec = SPEC
    score_split = expected_path
    key_columns = list(spec["key_columns"])
    prediction_columns = list(spec["prediction_columns"])
    answer_columns = list(spec["answer_columns"])
    pred = _validate_prediction_file(
        prediction_path, score_split
    )
    if not truth_path:
        raise RuntimeError("BIOMAKER_SCORE_VAL_PATH is not set.")
    truth = pd.read_csv(truth_path)
    missing_truth = [column for column in key_columns + answer_columns if column not in truth.columns]
    if missing_truth:
        raise RuntimeError(f"Hidden truth missing columns: {missing_truth}")

    truth_view = truth[key_columns + answer_columns].copy()
    pred_view = pred[key_columns + prediction_columns].copy()
    truth_view = truth_view.rename(columns={column: f"__true_{column}" for column in answer_columns})
    if key_columns:
        for column in key_columns:
            truth_view[column] = truth_view[column].astype(str)
            pred_view[column] = pred_view[column].astype(str)
        merged = truth_view.merge(pred_view, on=key_columns, how="inner")
        if len(merged) != len(truth_view):
            raise RuntimeError(f"Validation predictions matched {len(merged)} rows; expected {len(truth_view)}.")
    else:
        if len(pred_view) != len(truth_view):
            raise RuntimeError(f"Validation predictions have {len(pred_view)} rows; expected {len(truth_view)}.")
        merged = pd.concat(
            [truth_view.reset_index(drop=True), pred_view.reset_index(drop=True)],
            axis=1,
        )

    scores = merged[prediction_columns].to_numpy(dtype=float)
    result = {"maximize": True}
    if spec.get("task_type") == "multilabel":
        true_cols = [f"__true_{column}" for column in answer_columns]
        y_true = merged[true_cols].to_numpy(dtype=int)
        y_pred = (scores >= 0.5).astype(int)
        per_label_accuracy = (y_true == y_pred).mean(axis=0)
        metric = float(per_label_accuracy.mean())
        result.update(
            {
                "metric": metric,
                "mean_binary_accuracy": metric,
                "per_label_accuracy": {
                    column: float(value)
                    for column, value in zip(answer_columns, per_label_accuracy)
                },
            }
        )
        auc, auc_error = _score_auc(y_true, scores, spec)
        if auc is not None:
            result["mean_roc_auc"] = auc
            if _wants_auc(spec):
                result["metric"] = auc
        elif auc_error:
            result["mean_roc_auc_error"] = auc_error
        if _wants_auc(spec) and auc is None:
            raise ValueError("Requested primary AUC is undefined: " + str(auc_error))
        result["metrics"] = {"accuracy": metric}
        if auc is not None:
            result["metrics"]["auc"] = auc
        return result

    labels = list(spec.get("class_labels") or [])
    y_true = merged[f"__true_{answer_columns[0]}"].astype(str).to_numpy()
    if scores.shape[1] == 1:
        positive = str(spec.get("positive_class", labels[-1] if labels else "1"))
        negative = next((str(label) for label in labels if str(label) != positive), "0")
        y_pred = np.where(scores[:, 0] >= 0.5, positive, negative)
    else:
        y_pred = np.asarray([str(labels[index]) for index in np.argmax(scores, axis=1)])
    accuracy = float((y_true == y_pred).mean())
    result.update({"metric": accuracy, "accuracy": accuracy})
    auc, auc_error = _score_auc(y_true, scores, spec)
    if auc is not None:
        result["macro_roc_auc"] = auc
        if _wants_auc(spec):
            result["metric"] = auc
    elif auc_error:
        result["macro_roc_auc_error"] = auc_error
    if _wants_auc(spec) and auc is None:
        raise ValueError("Requested primary AUC is undefined: " + str(auc_error))
    result["metrics"] = {"accuracy": accuracy}
    if auc is not None:
        result["metrics"]["auc"] = auc
    return result


def self_test():
    spec = SPEC
    score_split = spec.get("score_split", "./input/biomaker_splits/score_val.csv")
    key_columns = list(spec["key_columns"])
    prediction_columns = list(spec["prediction_columns"])

    def write_dummy(split_path, artifact_path):
        split = pd.read_csv(split_path)
        out = split[key_columns].copy()
        if spec.get("task_type") == "multiclass" and len(prediction_columns) > 1:
            value = 1.0 / len(prediction_columns)
            for column in prediction_columns:
                out[column] = value
        else:
            for column in prediction_columns:
                out[column] = 0.5
        os.makedirs(os.path.dirname(_path(artifact_path)), exist_ok=True)
        out.to_csv(_path(artifact_path), index=False)

    write_dummy(score_split, spec["validation_artifact"])
    write_dummy("./input/biomaker_splits/agent_test.csv", spec["test_submission"])
    score = score_val()
    if not isinstance(score.get("metric"), (int, float)) or not math.isfinite(float(score["metric"])):
        raise AssertionError(score)
    validate_submission()
    print("EVALUATOR_OK")
"""

_FIXED_SEGMENTATION_EVALUATOR = r"""
import math
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
try:
    import SimpleITK as sitk
except Exception:
    sitk = None

SPEC = __BIOMAKER_SPEC__
PUBLIC_SPEC_KEYS = [
    "validation_artifact",
    "test_submission",
    "score_split",
    "key_columns",
    "prediction_columns",
    "primary_metric",
    "maximize",
    "task_type",
    "prediction_type",
    "image_column",
    "secondary_metrics",
    "evaluator_api_version",
]


def prediction_spec():
    return {key: SPEC[key] for key in PUBLIC_SPEC_KEYS if key in SPEC}


def _path(value):
    text = str(value).strip().removeprefix("./")
    return text if os.path.isabs(text) else os.path.join(".", text)


def _candidate_path(value):
    text = str(value).strip().removeprefix("./")
    if os.path.isabs(text):
        return text
    if text.startswith("working/") or text.startswith("input/"):
        return os.path.join(".", text)
    return os.path.join("./working", text)


def _read_csv(path):
    try:
        return pd.read_csv(_path(path))
    except Exception as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


def _key_frame(df, key_columns):
    out = df[key_columns].copy()
    for column in key_columns:
        out[column] = out[column].astype(str)
    return out


def _resolve_truth_path(value):
    text = str(value).strip().removeprefix("./")
    if os.path.isabs(text):
        return text
    if text.startswith("input/"):
        direct = os.path.join(".", text)
        if os.path.isfile(direct):
            return direct
        text = text[len("input/") :]
    private_root = os.path.dirname(os.environ.get("BIOMAKER_SCORE_VAL_PATH", ""))
    candidate = os.path.join(private_root, text)
    return candidate if os.path.isfile(candidate) else os.path.join(".", text)


def _is_volume_path(path):
    lower = str(path).lower()
    return lower.endswith((".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".npy", ".npz"))


def _load_volume(path):
    lower = str(path).lower()
    if lower.endswith(".npy"):
        return np.load(path), None
    if lower.endswith(".npz"):
        data = np.load(path)
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        return data[key], None
    if sitk is None:
        raise RuntimeError(
            "SimpleITK is required to read medical volume masks such as NIfTI/NRRD/MHA."
        )
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    spacing = tuple(reversed(tuple(float(value) for value in image.GetSpacing())))
    return array, spacing


def _load_mask(path):
    if _is_volume_path(path):
        return _load_volume(path)
    with Image.open(path) as image:
        return np.asarray(image), None


def _load_shape(path):
    array, spacing = _load_mask(path)
    return array.shape, spacing


def _spatial_shape(shape):
    if len(shape) == 3 and shape[-1] in (3, 4):
        return shape[:2]
    return shape


def _write_zero_like(reference_path, output_path):
    os.makedirs(os.path.dirname(_path(output_path)), exist_ok=True)
    ref = _path(reference_path)
    out = _path(output_path)
    if _is_volume_path(ref):
        if ref.lower().endswith(".npy"):
            array = np.load(ref)
            np.save(out, np.zeros_like(array, dtype=np.uint8))
            return
        if ref.lower().endswith(".npz"):
            data = np.load(ref)
            key = "arr_0" if "arr_0" in data.files else data.files[0]
            np.savez_compressed(out, **{key: np.zeros_like(data[key], dtype=np.uint8)})
            return
        if sitk is None:
            raise RuntimeError("SimpleITK is required to write dummy volume masks.")
        image = sitk.ReadImage(ref)
        array = np.zeros_like(sitk.GetArrayFromImage(image), dtype=np.uint8)
        mask = sitk.GetImageFromArray(array)
        mask.CopyInformation(image)
        sitk.WriteImage(mask, out)
        return
    with Image.open(ref) as image:
        Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8)).save(out)


def _binary_mask(mask):
    if mask.ndim == 3 and mask.shape[-1] in (3, 4):
        mask = mask[..., 0]
    return mask > 0


def _dice(mask_true, mask_pred):
    true_bin = _binary_mask(mask_true)
    pred_bin = _binary_mask(mask_pred)
    denom = true_bin.sum() + pred_bin.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(true_bin, pred_bin).sum() / denom)


def _surface_distances(source, target, spacing):
    if not source.any():
        return np.asarray([], dtype=float)
    if not target.any():
        return np.asarray([math.inf], dtype=float)
    structure = ndi.generate_binary_structure(source.ndim, 1)
    source_border = np.logical_xor(source, ndi.binary_erosion(source, structure=structure, border_value=0))
    target_border = np.logical_xor(target, ndi.binary_erosion(target, structure=structure, border_value=0))
    distance = ndi.distance_transform_edt(~target_border, sampling=spacing)
    return distance[source_border]


def _hd95(mask_true, mask_pred, spacing=None):
    true_bin = _binary_mask(mask_true)
    pred_bin = _binary_mask(mask_pred)
    if not true_bin.any() and not pred_bin.any():
        return 0.0
    if not true_bin.any() or not pred_bin.any():
        return math.inf
    if spacing is None:
        spacing = (1.0,) * true_bin.ndim
    forward = _surface_distances(true_bin, pred_bin, spacing)
    backward = _surface_distances(pred_bin, true_bin, spacing)
    distances = np.concatenate([forward, backward])
    return float(np.percentile(distances, 95))


def _aji(mask_true, mask_pred):
    # Official MoNuSeg AJI: independent best-IoU match for each GT instance.
    # https://github.com/ruchikaverma-iitg/MoNuSeg/blob/master/Aggregated_Jaccard_Index_v1_0.m
    # Preserve supplied instance IDs, including disconnected parts of one ID.
    mask_true = np.asarray(mask_true)
    mask_pred = np.asarray(mask_pred)
    if mask_true.shape != mask_pred.shape:
        raise ValueError("AJI requires masks with identical shapes.")
    for mask in (mask_true, mask_pred):
        if mask.dtype.kind not in "biu" or np.any(mask < 0):
            raise ValueError("AJI requires nonnegative integer instance labels.")

    true_labels, true_inverse, true_area = np.unique(
        mask_true, return_inverse=True, return_counts=True
    )
    pred_labels, pred_inverse, pred_area = np.unique(
        mask_pred, return_inverse=True, return_counts=True
    )
    true_ids = np.flatnonzero(true_labels > 0)
    pred_ids = np.flatnonzero(pred_labels > 0)
    if not len(true_ids) and not len(pred_ids):
        # Explicit extension for empty images (official MATLAB divides 0/0).
        return 1.0
    if not len(true_ids) or not len(pred_ids):
        return 0.0

    # Count only observed pairs, avoiding a dense GT-by-prediction matrix.
    stride = len(pred_labels)
    pairs, intersections = np.unique(
        true_inverse.ravel().astype(np.int64) * stride + pred_inverse.ravel(),
        return_counts=True,
    )
    best = {}
    for key, intersection in zip(pairs, intersections):
        true_id, pred_id = divmod(int(key), stride)
        if true_labels[true_id] == 0 or pred_labels[pred_id] == 0:
            continue
        intersection = int(intersection)
        union = int(true_area[true_id]) + int(pred_area[pred_id]) - intersection
        iou = intersection / union
        # Pairs are sorted by ID: strict > retains the lowest prediction ID
        # on ties, as in the official implementation.
        if true_id not in best or iou > best[true_id][0]:
            best[true_id] = (iou, pred_id, intersection, union)

    used_pred = set()
    intersection_sum = 0
    union_sum = 0
    for true_id in true_ids:
        match = best.get(int(true_id))
        if match is None:
            union_sum += int(true_area[true_id])
        else:
            _, pred_id, intersection, union = match
            intersection_sum += intersection
            union_sum += union
            used_pred.add(pred_id)  # Reuse is allowed; track only unused areas.
    union_sum += sum(int(pred_area[j]) for j in pred_ids if int(j) not in used_pred)
    return float(intersection_sum / union_sum)


def _align_predictions(pred, expected, key_columns, path):
    if len(pred) != len(expected):
        raise ValueError(f"{path} row count {len(pred)} does not match expected {len(expected)}.")
    if not key_columns:
        return pred.reset_index(drop=True)
    if pred.duplicated(key_columns).any():
        raise ValueError(f"{path} contains duplicate keys.")
    expected_keys = _key_frame(expected, key_columns)
    pred_keys = _key_frame(pred, key_columns)
    merged = expected_keys.reset_index().merge(
        pred_keys.reset_index().rename(columns={"index": "__pred_index"}),
        on=key_columns,
        how="left",
    )
    if merged["__pred_index"].isna().any():
        raise ValueError(f"{path} is missing expected keys.")
    pred_indices = merged["__pred_index"].astype(int).tolist()
    return pred.iloc[pred_indices].reset_index(drop=True)


def _validate_prediction_file(path, expected_split_path, require_size_match):
    spec = prediction_spec()
    key_columns = list(spec["key_columns"])
    mask_column = spec["prediction_columns"][0]
    required = key_columns + [mask_column]
    pred = _read_csv(path)
    missing = [column for column in required if column not in pred.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    pred = pred[required].copy()
    if pred[required].isna().any().any():
        raise ValueError(f"{path} contains missing keys or mask paths.")
    expected = _read_csv(expected_split_path)
    aligned = _align_predictions(pred, expected, key_columns, path)
    image_column = spec.get("image_column")
    for idx, row in aligned.iterrows():
        mask_path = _candidate_path(row[mask_column])
        if not os.path.isfile(mask_path):
            raise ValueError(f"{path} row {idx} references missing mask file: {row[mask_column]}")
        if require_size_match and image_column and image_column in expected.columns:
            image_path = _path(expected.iloc[idx][image_column])
            if os.path.isfile(image_path):
                image_shape, _ = _load_shape(image_path)
                mask_shape, _ = _load_shape(mask_path)
                if _spatial_shape(image_shape) != _spatial_shape(mask_shape):
                    raise ValueError(
                        f"{path} row {idx} mask shape {mask_shape} does not match image shape {image_shape}."
                    )
    return aligned


def validate_submission():
    spec = prediction_spec()
    _validate_prediction_file(
        spec["test_submission"], "./input/biomaker_splits/agent_test.csv", True
    )
    return {"valid": True}


def score_val():
    return score(SPEC["validation_artifact"], os.environ["BIOMAKER_SCORE_VAL_PATH"],
                 SPEC.get("score_split", "./input/biomaker_splits/score_val.csv"))


def score(prediction_path, truth_path, expected_path):
    spec = SPEC
    score_split = expected_path
    key_columns = list(spec["key_columns"])
    pred_column = spec["prediction_columns"][0]
    answer_column = spec["answer_columns"][0]
    pred = _validate_prediction_file(
        prediction_path, score_split, False
    )
    if not truth_path:
        raise RuntimeError("BIOMAKER_SCORE_VAL_PATH is not set.")
    truth = pd.read_csv(truth_path)
    missing_truth = [column for column in key_columns + [answer_column] if column not in truth.columns]
    if missing_truth:
        raise RuntimeError(f"Hidden truth missing columns: {missing_truth}")
    true_column = "__true_mask_path"

    if key_columns:
        truth_view = truth[key_columns + [answer_column]].copy()
        pred_view = pred[key_columns + [pred_column]].copy()
        truth_view = truth_view.rename(columns={answer_column: true_column})
        for column in key_columns:
            truth_view[column] = truth_view[column].astype(str)
            pred_view[column] = pred_view[column].astype(str)
        rows = truth_view.merge(pred_view, on=key_columns, how="inner")
        if len(rows) != len(truth_view):
            raise RuntimeError(f"Validation predictions matched {len(rows)} rows; expected {len(truth_view)}.")
    else:
        if len(pred) != len(truth):
            raise RuntimeError(f"Validation predictions have {len(pred)} rows; expected {len(truth)}.")
        rows = truth[[answer_column]].copy().rename(columns={answer_column: true_column}).reset_index(drop=True)
        rows[pred_column] = pred[pred_column].reset_index(drop=True)

    dice_scores = []
    aji_scores = []
    hd95_scores = []
    for idx, row in rows.iterrows():
        true_path = _resolve_truth_path(row[true_column])
        pred_path = _candidate_path(row[pred_column])
        if not os.path.isfile(true_path):
            raise RuntimeError(f"Hidden truth mask missing for row {idx}: {row[true_column]}")
        if not os.path.isfile(pred_path):
            raise RuntimeError(f"Prediction mask missing for row {idx}: {row[pred_column]}")
        true_mask, true_spacing = _load_mask(true_path)
        pred_mask, pred_spacing = _load_mask(pred_path)
        if true_mask.shape != pred_mask.shape:
            raise RuntimeError(
                f"Mask shape mismatch for row {idx}: predicted {pred_mask.shape}, true {true_mask.shape}."
            )
        if true_spacing is not None and pred_spacing is not None:
            if len(true_spacing) != len(pred_spacing) or any(
                abs(a - b) > 1e-4 for a, b in zip(true_spacing, pred_spacing)
            ):
                raise RuntimeError(
                    f"Mask spacing mismatch for row {idx}: predicted {pred_spacing}, true {true_spacing}."
                )
        dice_scores.append(_dice(true_mask, pred_mask))
        aji_scores.append(_aji(true_mask, pred_mask))
        hd95_scores.append(_hd95(true_mask, pred_mask, true_spacing))
    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    mean_aji = float(np.mean(aji_scores)) if aji_scores else 0.0
    finite_hd95 = [value for value in hd95_scores if math.isfinite(float(value))]
    mean_hd95 = float(np.mean(finite_hd95)) if len(finite_hd95) == len(hd95_scores) and finite_hd95 else math.inf
    metric_name = str(spec.get("primary_metric", "")).lower()
    if metric_name in {"hd95", "hausdorff95", "hausdorff_95"}:
        metric = mean_hd95
        maximize = False
    elif metric_name in {"binary_dice", "dice", "mean_dice"}:
        metric = mean_dice
        maximize = True
    elif metric_name in {"aji", "mean_aji"}:
        metric = mean_aji
        maximize = True
    else:
        raise RuntimeError(f"Unsupported segmentation metric: {metric_name}")
    if not math.isfinite(float(metric)):
        raise RuntimeError(f"Primary metric {metric_name or 'segmentation'} is not finite.")
    return {
        "metric": metric,
        "maximize": maximize,
        "aji": mean_aji,
        "binary_dice": mean_dice,
        "hd95": mean_hd95,
        "metrics": {name: value for name, value in {
            "aji": mean_aji, "binary_dice": mean_dice, "hd95": mean_hd95
        }.items() if math.isfinite(value)},
    }


def self_test():
    spec = SPEC
    score_split = spec.get("score_split", "./input/biomaker_splits/score_val.csv")
    key_columns = list(spec["key_columns"])
    mask_column = spec["prediction_columns"][0]

    def write_dummy(split_path, artifact_path, subdir):
        split = pd.read_csv(split_path)
        out = split[key_columns].copy()
        image_column = spec.get("image_column")
        mask_refs = []
        os.makedirs(os.path.join("./working", subdir), exist_ok=True)
        for idx, row in split.iterrows():
            if image_column and image_column in split.columns and os.path.isfile(_path(row[image_column])):
                image_ref = row[image_column]
            else:
                image_ref = None
            if image_ref is not None and _is_volume_path(_path(image_ref)):
                suffix = ".nii.gz" if str(image_ref).lower().endswith(".nii.gz") else os.path.splitext(str(image_ref))[1]
                rel = f"working/{subdir}/{idx}{suffix}"
                _write_zero_like(image_ref, rel)
            elif image_ref is not None:
                rel = f"working/{subdir}/{idx}.png"
                _write_zero_like(image_ref, rel)
            else:
                rel = f"working/{subdir}/{idx}.png"
                Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(_path(rel))
            mask_refs.append(rel)
        out[mask_column] = mask_refs
        os.makedirs(os.path.dirname(_path(artifact_path)), exist_ok=True)
        out.to_csv(_path(artifact_path), index=False)

    write_dummy(score_split, spec["validation_artifact"], "val_masks")
    write_dummy("./input/biomaker_splits/agent_test.csv", spec["test_submission"], "test_masks")
    score = score_val()
    if not isinstance(score.get("metric"), (int, float)) or not math.isfinite(float(score["metric"])):
        raise AssertionError(score)
    validate_submission()
    print("EVALUATOR_OK")
"""
