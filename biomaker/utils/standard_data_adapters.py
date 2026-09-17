"""Default data adapters for explicit tabular classification, masks, and boxes.

These source templates only interpret artifacts and make public-input probes.
All alignment, validation, metric definitions, and aggregation are supplied by
``evaluator_adapter.compile_adapter``, exactly as for an agent-written adapter.
"""

import copy


_DATA_ADAPTER_CAPABILITIES = {
    "adapter_api_version": 1,
    "selection": (
        "A built-in reader is selected only when the declared representation and "
        "public data schema satisfy its requirements. Null prediction_type means "
        "the family's documented default. Other representations need a custom "
        "data adapter, with the same trusted metric implementation."
    ),
    "shared_contract": {
        "column_names": (
            "key_columns, prediction_columns, and answer_columns are explicit, "
            "ordered lists of arbitrary column names. CSV cells arrive as strings; "
            "Harness aligns rows by the declared keys before decoding."
        ),
        "file_reference_columns": (
            "Declare the subset of prediction_columns whose cells contain file "
            "paths. Inline numeric probabilities and JSON boxes are not file references."
        ),
        "dummy_predictions": (
            "Use only the supplied public rows and inputs. Write exactly those "
            "keys to the supplied artifact_path; retain the CSV and all referenced "
            "files until Harness finishes its probes."
        ),
        "metric_ownership": (
            "Readers decode data and expose canonical samples. Harness validates "
            "the samples and applies the separately declared metric protocol."
        ),
    },
    "families": {
        "regression": {
            "prediction_types": [None, "columns", "continuous_scalar", "continuous_vector", "numeric_columns"],
            "default_prediction_type": "numeric_columns",
            "required_declarations": ["prediction_columns", "answer_columns"],
            "representation": (
                "Ordered numeric prediction and answer columns, one per output. "
                "One column decodes to a scalar; multiple columns to a vector. "
                "Values preserve the declared target units without rounding or clipping."
            ),
            "additional_requirements": "Prediction and answer column counts must agree.",
            "file_reference_policy": "none",
        },
        "classification": {
            "prediction_types": [
                None, "columns", "probabilities", "class_probabilities", "probability_columns"
            ],
            "default_prediction_type": "probability_columns",
            "required_declarations": [
                "task_type", "prediction_columns", "answer_columns"
            ],
            "representation": (
                "Each prediction column holds one numeric probability. Single-label "
                "probabilities follow class_labels order; a binary one-column "
                "probability refers to positive_class (default final class label). "
                "Single-label targets are scalar class labels. Multilabel target "
                "and probability vectors follow answer_columns/prediction_columns "
                "order respectively. Arbitrary column names are supported."
            ),
            "additional_requirements": (
                "Single-label tasks declare class_labels. Multilabel tasks declare "
                "one probability column for each answer column."
            ),
            "file_reference_policy": "none",
        },
        "segmentation": {
            "prediction_types": [None, "mask_file_path"],
            "default_prediction_type": "mask_file_path",
            "required_declarations": [
                "prediction_columns", "answer_columns", "image_column", "file_reference_columns"
            ],
            "representation": (
                "One arbitrary prediction column references a full-size mask; one "
                "answer column references a target mask. Arrays preserve integer "
                "labels, including instance IDs. Readers support Pillow image "
                "formats, .npy, .npz (arr_0 or first archive array), and SimpleITK "
                "medical volumes (.nii, .nii.gz, .nrrd, .mha, .mhd)."
            ),
            "public_geometry": (
                "image_column identifies a public input file used for shape and "
                "spacing without targets. Color channels are excluded only for "
                "Pillow 2D inputs. NumPy/medical volume axes remain intact. "
                "SimpleITK spacing is reversed into NumPy array-axis order."
            ),
            "path_resolution": (
                "Input and target references must be absolute or workspace-relative. "
                "Prediction paths that are absolute or begin input/ or working/ "
                "are used directly; other prediction paths are relative to working/. "
                "No private-directory discovery or environment-dependent fallback."
            ),
            "file_reference_policy": "all_prediction_columns",
            "prediction_column_count": 1,
            "answer_column_count": 1,
        },
        "image_restoration": {
            "prediction_types": [None, "image_file_path"],
            "default_prediction_type": "image_file_path",
            "required_declarations": [
                "prediction_columns", "answer_columns", "file_reference_columns",
                "restoration_evaluation.input_column"
            ],
            "representation": (
                "One arbitrary prediction column and one answer column reference "
                "finite full-size 2D grayscale intensity arrays. The reader supports "
                ".npy and grayscale Pillow images, preserving intensities. RGB and "
                "palette conversions require a custom data adapter."
            ),
            "public_geometry": (
                "restoration_evaluation.input_column identifies the public image "
                "used to derive input_shape. An optional declared group_column "
                "identifies public group metadata, which Harness reads directly."
            ),
            "path_resolution": (
                "All references are explicit absolute or workspace-relative paths. "
                "Use working/ prefixes for prediction files under working/."
            ),
            "file_reference_policy": "all_prediction_columns",
            "prediction_column_count": 1,
            "answer_column_count": 1,
        },
        "detection": {
            "prediction_types": [None, "columns", "json_boxes"],
            "default_prediction_type": "json_boxes",
            "required_declarations": [
                "prediction_columns", "answer_columns", "prediction_schema", "truth_schema"
            ],
            "representation": (
                "One arbitrary prediction column and one answer column contain "
                "JSON lists of box objects ([] is an empty image). Each schema "
                "declares class_key, coordinate_keys, and box_format; coordinate "
                "keys are four object fields or ['bbox']. Supported formats xyxy, "
                "xywh, and cxcywh convert to canonical xyxy without dropping boxes. "
                "Prediction score_key defaults to score. Class IDs and confidences "
                "are preserved. Input and target coordinate units must agree."
            ),
            "unsupported_annotations": ["iscrowd", "ignore", "difficult"],
            "file_reference_policy": "none",
            "prediction_column_count": 1,
            "answer_column_count": 1,
        },
    },
    "custom_adapter": {
        "prediction_types": "any declared encoding",
        "functions": ["prediction_spec", "load_sample", "write_dummy_predictions"],
        "freedom": (
            "Choose original column names, directory layout, file formats, and "
            "data decoding. Declare actual prediction_columns and any "
            "file_reference_columns. Convert each aligned row to the canonical "
            "sample interface without implementing or changing metric semantics. "
            "Built-in encoding names and paths do not restrict custom adapters."
        ),
    },
}


def data_adapter_capabilities():
    """Return the JSON-ready reader catalog shared by admission and prompts."""
    return copy.deepcopy(_DATA_ADAPTER_CAPABILITIES)


_REGRESSION_DATA_ADAPTER = r'''
import csv
from pathlib import Path
import numpy as np

ADAPTER_SPEC = __BIOMAKER_SPEC__


def prediction_spec():
    return ADAPTER_SPEC


def adapter_numeric(row, columns):
    values = [float(row[column]) for column in columns]
    return values[0] if len(values) == 1 else np.asarray(values, dtype=float)


def load_sample(prediction_row, target_row, expected_row):
    spec = prediction_spec()
    sample = {'prediction': adapter_numeric(prediction_row, spec['prediction_columns'])}
    if target_row is not None:
        sample['target'] = adapter_numeric(target_row, spec['answer_columns'])
    return sample


def write_dummy_predictions(expected_rows, artifact_path):
    spec = prediction_spec()
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=spec['key_columns'] + spec['prediction_columns'])
        writer.writeheader()
        for expected in expected_rows:
            row = {key: expected[key] for key in spec['key_columns']}
            row.update({column: 0.0 for column in spec['prediction_columns']})
            writer.writerow(row)
'''


_CLASSIFICATION_DATA_ADAPTER = r'''
import csv
from pathlib import Path
import numpy as np

ADAPTER_SPEC = __BIOMAKER_SPEC__


def prediction_spec():
    return ADAPTER_SPEC


def load_sample(prediction_row, target_row, expected_row):
    spec = prediction_spec()
    sample = {'prediction': np.asarray(
        [float(prediction_row[column]) for column in spec['prediction_columns']])}
    if target_row is not None:
        if spec['task_type'] == 'multilabel':
            sample['target'] = np.asarray(
                [float(target_row[column]) for column in spec['answer_columns']])
        else:
            sample['target'] = target_row[spec['answer_columns'][0]]
    return sample


def write_dummy_predictions(expected_rows, artifact_path):
    spec = prediction_spec()
    columns = spec['key_columns'] + spec['prediction_columns']
    probability = (0.5 if spec['task_type'] == 'multilabel'
                   or len(spec['prediction_columns']) == 1
                   else 1.0 / len(spec['prediction_columns']))
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for expected in expected_rows:
            row = {key: expected[key] for key in spec['key_columns']}
            row.update({column: probability for column in spec['prediction_columns']})
            writer.writerow(row)
'''


_SEGMENTATION_DATA_ADAPTER = r'''
import csv
from pathlib import Path
import numpy as np
from PIL import Image

ADAPTER_SPEC = __BIOMAKER_SPEC__


def prediction_spec():
    return ADAPTER_SPEC


def adapter_prediction_path(value):
    path = Path(str(value))
    if path.is_absolute() or path.parts[0] in ('working', 'input'):
        return path
    return Path('working') / path


def adapter_is_volume_path(path):
    return str(path).lower().endswith(
        ('.nii', '.nii.gz', '.nrrd', '.mha', '.mhd', '.npy', '.npz'))


def adapter_read_array(path):
    path = Path(path)
    lower = str(path).lower()
    if lower.endswith('.npy'):
        return np.load(path, allow_pickle=False), None
    if lower.endswith('.npz'):
        with np.load(path, allow_pickle=False) as archive:
            key = 'arr_0' if 'arr_0' in archive.files else archive.files[0]
            return archive[key], None
    if adapter_is_volume_path(path):
        import SimpleITK as sitk
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        spacing = tuple(reversed(tuple(float(value) for value in image.GetSpacing())))
        return array, spacing
    with Image.open(path) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Multi-frame images require a data adapter that explicitly preserves the declared sample axes; the default reader cannot select only the first frame.')
        return np.asarray(image).copy(), None


def adapter_public_geometry(expected_row):
    column = prediction_spec().get('image_column')
    if not column or not expected_row.get(column):
        raise ValueError('The default mask adapter requires image_column with a public input path; provide a data adapter for other layouts.')
    path = Path(expected_row[column])
    array, spacing = adapter_read_array(path)
    shape = array.shape
    # Color channels belong to public 2D images, never to volume dimensions.
    if not adapter_is_volume_path(path) and array.ndim == 3 and shape[-1] in (3, 4):
        shape = shape[:2]
    return shape, spacing


def load_sample(prediction_row, target_row, expected_row):
    spec = prediction_spec()
    prediction, prediction_spacing = adapter_read_array(
        adapter_prediction_path(prediction_row[spec['prediction_columns'][0]]))
    shape, spacing = adapter_public_geometry(expected_row)
    sample = {'prediction': prediction, 'input_shape': shape,
              'spacing': spacing, 'prediction_spacing': prediction_spacing}
    if target_row is not None:
        target, target_spacing = adapter_read_array(
            Path(target_row[spec['answer_columns'][0]]))
        sample['target'] = target
        if target_spacing is not None:
            if spacing is not None and (len(spacing) != len(target_spacing)
                    or not np.allclose(spacing, target_spacing, rtol=0, atol=1e-4)):
                raise ValueError('Public input and target spacing disagree.')
            sample['spacing'] = target_spacing
    return sample


def write_dummy_predictions(expected_rows, artifact_path):
    spec = prediction_spec()
    artifact = Path(artifact_path)
    root = artifact.parent / (artifact.stem + '_masks')
    root.mkdir(parents=True, exist_ok=True)
    metrics = [spec['primary_metric']] + spec.get('secondary_metrics', [])
    rows = []
    for index, expected in enumerate(expected_rows):
        shape, _ = adapter_public_geometry(expected)
        mask = np.zeros(shape, dtype=np.uint8)
        if any(str(name).lower() in ('hd95', 'hausdorff95', 'hausdorff_95') for name in metrics):
            # A public synthetic foreground avoids undefined HD95 for common
            # nonempty targets without inspecting or copying any target mask.
            mask[tuple(size // 2 for size in shape)] = 1
        path = root / (str(index) + '.npy')
        np.save(path, mask, allow_pickle=False)
        row = {key: expected[key] for key in spec['key_columns']}
        row[spec['prediction_columns'][0]] = str(path)
        rows.append(row)
    with artifact.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=spec['key_columns'] + spec['prediction_columns'])
        writer.writeheader()
        writer.writerows(rows)
'''


_DETECTION_DATA_ADAPTER = r'''
import csv
import json
import math
from pathlib import Path

ADAPTER_SPEC = __BIOMAKER_SPEC__


def prediction_spec():
    return ADAPTER_SPEC


def adapter_parse_boxes(value, prediction):
    items = json.loads(value) if isinstance(value, str) else value
    if not isinstance(items, list):
        raise ValueError('Boxes must be a JSON list; use [] for no detections.')
    schema = prediction_spec()['prediction_schema' if prediction else 'truth_schema']
    result = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Each box must be an object with declared coordinate/class keys.')
        if any(item.get(key, False) for key in ('iscrowd', 'ignore', 'difficult')):
            raise ValueError('Crowd/ignore/difficult boxes require a different evaluation protocol.')
        keys = schema['coordinate_keys']
        raw = item['bbox'] if keys == ['bbox'] else [item[key] for key in keys]
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError('Each bbox must contain exactly four coordinates.')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in raw):
            raise ValueError('Coordinates must be finite numbers.')
        a, b, c, d = map(float, raw)
        fmt = schema['box_format']
        if fmt == 'cxcywh':
            box = (a-c/2, b-d/2, a+c/2, b+d/2)
        elif fmt == 'xywh':
            box = (a, b, a+c, b+d)
        elif fmt == 'xyxy':
            box = (a, b, c, d)
        else:
            raise ValueError('Unsupported declared box format.')
        result.append((item[schema['class_key']], box,
                       item[schema.get('score_key', 'score')] if prediction else 1.0))
    return result


def load_sample(prediction_row, target_row, expected_row):
    spec = prediction_spec()
    sample = {'prediction': adapter_parse_boxes(
        prediction_row[spec['prediction_columns'][0]], True)}
    if target_row is not None:
        sample['target'] = adapter_parse_boxes(
            target_row[spec['answer_columns'][0]], False)
    return sample


def write_dummy_predictions(expected_rows, artifact_path):
    spec = prediction_spec()
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=spec['key_columns'] + spec['prediction_columns'])
        writer.writeheader()
        for expected in expected_rows:
            row = {key: expected[key] for key in spec['key_columns']}
            row[spec['prediction_columns'][0]] = '[]'
            writer.writerow(row)
'''
