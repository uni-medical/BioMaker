"""Built-in readers use the same frozen metric/runtime boundary as custom readers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from biomaker.utils.evaluator_adapter import compile_adapter
from biomaker.utils.standard_data_adapters import (
    _CLASSIFICATION_DATA_ADAPTER,
    _DETECTION_DATA_ADAPTER,
    _SEGMENTATION_DATA_ADAPTER,
    data_adapter_capabilities,
)


def base_spec(task_type, primary, prediction_columns, answer_columns):
    return dict(
        evaluator_api_version=2,
        task_type=task_type,
        primary_metric=primary,
        secondary_metrics=[],
        maximize=primary != "hd95",
        key_columns=["case"],
        prediction_columns=prediction_columns,
        answer_columns=answer_columns,
        validation_artifact="./working/validation_predictions.csv",
        test_submission="./working/submission.csv",
    )


def load_adapter(template, spec):
    namespace = {}
    exec(compile_adapter(template.replace("__BIOMAKER_SPEC__", repr(spec)), spec), namespace)
    assert namespace["prediction_spec"]()["scoring_backend"] == "trusted_metrics_v1"
    return namespace


def write_tables(root, expected, truth, predictions):
    paths = [root / name for name in ("predictions.csv", "truth.csv", "expected.csv")]
    for path, rows in zip(paths, (predictions, truth, expected)):
        pd.DataFrame(rows).to_csv(path, index=False)
    return list(map(str, paths))


def test_reader_catalog_is_json_ready_and_independent():
    catalog = data_adapter_capabilities()
    assert json.loads(json.dumps(catalog)) == catalog
    catalog["families"]["classification"]["prediction_types"].append("local mutation")
    assert "local mutation" not in data_adapter_capabilities()["families"]["classification"]["prediction_types"]


@pytest.mark.parametrize("family,primary", [
    ("classification", "accuracy"), ("segmentation", "binary_dice"),
    ("image_restoration", "psnr"), ("detection", "ap50"),
])
def test_reader_encoding_catalog_matches_admission_and_custom_freedom(tmp_path, family, primary):
    from biomaker.utils.evaluator_setup import EvaluatorSetupError, _resolve_metric_contract

    spec = base_spec("binary" if family == "classification" else family,
                     primary, ["estimate"], ["reference"])
    if family == "classification":
        spec["class_labels"] = ["negative", "positive"]
    elif family == "image_restoration":
        spec["restoration_evaluation"] = dict(
            data_range=255, crop_size=None, groups=None, aggregation="mean_images")
    capabilities = data_adapter_capabilities()["families"][family]
    for encoding in capabilities["prediction_types"]:
        spec["prediction_type"] = encoding
        resolved = _resolve_metric_contract(spec, tmp_path, "", require_match=False)
        assert resolved["task_family"] == family
    spec["prediction_type"] = "dataset_specific_archive"
    with pytest.raises(EvaluatorSetupError, match="data adapter"):
        _resolve_metric_contract(spec, tmp_path, "", require_match=False)
    # Decoding is agent-owned: custom encodings keep the identical metric spec.
    resolved = _resolve_metric_contract(spec, tmp_path, "", require_match=True)
    assert resolved["prediction_type"] == "dataset_specific_archive"
    assert resolved["primary_metric"] == primary


@pytest.mark.parametrize("multilabel", [False, True])
def test_classification_columns_align_through_shared_harness(tmp_path, monkeypatch, multilabel):
    monkeypatch.chdir(tmp_path)
    spec = base_spec(
        "multilabel" if multilabel else "binary",
        "mean_binary_accuracy" if multilabel else "accuracy",
        ["arbitrary:first", "arbitrary:second"],
        ["a", "b"] if multilabel else ["diagnosis"],
    )
    if not multilabel:
        spec["class_labels"] = ["healthy", "diseased"]
    scorer = load_adapter(_CLASSIFICATION_DATA_ADAPTER, spec)
    expected = [{"case": "001"}, {"case": "002"}]
    truth = ([{"case": "001", "a": 1, "b": 0}, {"case": "002", "a": 0, "b": 1}]
             if multilabel else [{"case": "001", "diagnosis": "healthy"},
                                 {"case": "002", "diagnosis": "diseased"}])
    predictions = [{"case": "002", "arbitrary:first": 0.1, "arbitrary:second": 0.9},
                   {"case": "001", "arbitrary:first": 0.9, "arbitrary:second": 0.1}]
    assert scorer["score"](*write_tables(tmp_path, expected, truth, predictions))["metric"] == 1
    artifact = "working/relocated/probe.csv"
    scorer["write_dummy_predictions"](expected[1:], artifact)
    frame = pd.read_csv(artifact, dtype=str)
    assert frame["case"].tolist() == ["002"]
    sample = scorer["load_sample"](frame.iloc[0].to_dict(), None, expected[1])
    assert "target" not in sample
    scorer["validate_prediction"](sample, spec)


def test_mask_shape_and_instance_ids_preserved_from_public_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "working").mkdir()
    Image.new("RGB", (9, 7)).save("input/source.png")
    mask = np.zeros((7, 9), dtype=np.uint16)
    mask[1:3, 2:4] = 451
    mask[4:6, 5:7] = 999
    np.save("working/output.npy", mask)
    np.save("target.npy", mask)
    spec = base_spec("segmentation", "aji", ["my prediction"], ["my target"])
    spec.update(image_column="source", file_reference_columns=["my prediction"])
    scorer = load_adapter(_SEGMENTATION_DATA_ADAPTER, spec)
    row = {"case": "007", "source": "input/source.png"}
    pred = {"case": "007", "my prediction": "output.npy"}
    truth = {"case": "007", "my target": "target.npy"}
    sample = scorer["load_sample"](pred, truth, row)
    assert sample["input_shape"] == (7, 9)
    assert set(np.unique(sample["target"])) == {0, 451, 999}
    assert scorer["score"](*write_tables(tmp_path, [row], [truth], [pred]))["metric"] == 1
    prediction_only = scorer["load_sample"](pred, None, row)
    assert "target" not in prediction_only
    scorer["validate_prediction"](prediction_only, spec)
    np.save("working/output.npy", mask[:-1])
    with pytest.raises(ValueError, match="input_shape"):
        scorer["validate_prediction"](scorer["load_sample"](pred, None, row), spec)


@pytest.mark.parametrize("extension", ["npy", "npz"])
def test_volume_probes_preserve_shape_and_survive_second_artifact(tmp_path, monkeypatch, extension):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    source = "input/source." + extension
    array = np.zeros((6, 5, 3), dtype=np.int16)
    if extension == "npy":
        np.save(source, array)
    else:
        np.savez(source, volume=array)
    spec = base_spec("segmentation_3d", "hd95", ["prediction"], ["reference"])
    spec["image_column"] = "source"
    scorer = load_adapter(_SEGMENTATION_DATA_ADAPTER, spec)
    expected = [{"case": "abc", "source": source}]
    paths = ["working/relocated/validation/predictions.csv",
             "working/relocated/submission/predictions.csv"]
    refs = []
    for artifact in paths:
        scorer["write_dummy_predictions"](expected, artifact)
        row = pd.read_csv(artifact, dtype=str).iloc[0].to_dict()
        refs.append(row["prediction"])
        sample = scorer["load_sample"](row, None, expected[0])
        assert sample["prediction"].shape == (6, 5, 3)
        assert sample["prediction"].sum() == 1
        scorer["validate_prediction"](sample, spec)
    assert refs[0] != refs[1]
    assert all(np.load(ref).sum() == 1 for ref in refs)


def test_medical_volume_spacing_checked_without_targets(tmp_path, monkeypatch):
    sitk = pytest.importorskip("SimpleITK")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "working").mkdir()
    array = np.zeros((4, 5, 6), dtype=np.uint8)
    array[1:3, 1:4, 1:5] = 9
    source = sitk.GetImageFromArray(array)
    source.SetSpacing((1.1, 1.2, 1.3))
    sitk.WriteImage(source, "input/source.nii.gz")
    sitk.WriteImage(source, "working/output.nii.gz")
    spec = base_spec("segmentation_3d", "binary_dice", ["prediction"], ["reference"])
    spec["image_column"] = "source"
    scorer = load_adapter(_SEGMENTATION_DATA_ADAPTER, spec)
    expected = {"case": "one", "source": "input/source.nii.gz"}
    pred = {"case": "one", "prediction": "working/output.nii.gz"}
    sample = scorer["load_sample"](pred, None, expected)
    assert sample["spacing"] == pytest.approx((1.3, 1.2, 1.1))
    scorer["validate_prediction"](sample, spec)
    source.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(source, "working/output.nii.gz")
    with pytest.raises(ValueError, match="spacing"):
        scorer["validate_prediction"](scorer["load_sample"](pred, None, expected), spec)


@pytest.mark.parametrize("box_format,coords", [
    ("xyxy", [2, 3, 6, 9]), ("xywh", [2, 3, 4, 6]), ("cxcywh", [4, 6, 4, 6]),
])
def test_box_schema_conversion_preserves_confidence_and_class(tmp_path, monkeypatch, box_format, coords):
    monkeypatch.chdir(tmp_path)
    spec = base_spec("detection", "ap50", ["estimate"], ["annotation"])
    spec.update(
        prediction_schema=dict(coordinate_keys=["bbox"], box_format=box_format,
                               class_key="category", score_key="confidence"),
        truth_schema=dict(coordinate_keys=["x1", "y1", "x2", "y2"],
                          box_format="xyxy", class_key="label"),
    )
    scorer = load_adapter(_DETECTION_DATA_ADAPTER, spec)
    expected = {"case": "image"}
    pred = {"case": "image", "estimate": json.dumps([
        {"bbox": coords, "category": 17, "confidence": 0.81}])}
    truth = {"case": "image", "annotation": json.dumps([
        dict(x1=2, y1=3, x2=6, y2=9, label=17)])}
    sample = scorer["load_sample"](pred, truth, expected)
    assert sample["prediction"] == [(17, (2., 3., 6., 9.), 0.81)]
    assert sample["target"] == [(17, (2., 3., 6., 9.), 1.0)]
    result = scorer["score"](*write_tables(tmp_path, [expected], [truth], [pred]))
    assert result["metric"] == 1.0
    artifact = "working/relocated/predictions.csv"
    scorer["write_dummy_predictions"]([expected], artifact)
    row = pd.read_csv(artifact, dtype=str).iloc[0].to_dict()
    sample = scorer["load_sample"](row, None, expected)
    assert sample == {"prediction": []}


@pytest.mark.parametrize("custom_encoding", [False, True])
@pytest.mark.parametrize("positive_index", [1, 9, None])
def test_detection_probe_finds_canonical_targets_without_exposing_answers(
    tmp_path, monkeypatch, custom_encoding, positive_index
):
    monkeypatch.chdir(tmp_path)
    split = tmp_path / "input/biomaker_splits"
    split.mkdir(parents=True)
    expected = [{"case": str(index), "scan": "public-" + str(index)} for index in range(12)]
    pd.DataFrame(expected).to_csv(split / "score_val.csv", index=False)
    pd.DataFrame(expected).to_csv(split / "agent_test.csv", index=False)
    spec = base_spec("detection", "ap50", ["estimate"], ["private reference"])
    schema = dict(coordinate_keys=["bbox"], box_format="xyxy", class_key="category")
    spec.update(prediction_schema=schema, truth_schema=schema)
    source = _DETECTION_DATA_ADAPTER
    if custom_encoding:
        source = source.replace(
            "sample['target'] = adapter_parse_boxes(\n"
            "            target_row[spec['answer_columns'][0]], False)",
            "raw = target_row[spec['answer_columns'][0]]\n"
            "        sample['target'] = [] if raw == 'empty' else [\n"
            "            (int(item.split(':')[0]), tuple(map(float, item.split(':')[1].split(','))), 1.0)\n"
            "            for item in raw.split(';')]",
        )
        present, absent = "17:2,3,6,9", "empty"
    else:
        present = json.dumps([dict(category=17, bbox=[2, 3, 6, 9])])
        absent = "[]"
    scorer = load_adapter(source, spec)
    truth_path = tmp_path / "private.csv"
    pd.DataFrame([{"case": str(index), "private reference": present if index == positive_index else absent}
                  for index in range(12)]).to_csv(truth_path, index=False)
    monkeypatch.setenv("BIOMAKER_SCORE_VAL_PATH", str(truth_path))
    writer = scorer["_h_write_dummy"]
    selection_batches = []

    def checked_writer(rows, artifact_path):
        # The writer cannot inspect answers; it sees the public split only.
        assert all(set(row) == {"case", "scan"} for row in rows)
        if "/selection/" in artifact_path:
            assert len(rows) <= 4
            selection_batches.append([row["case"] for row in rows])
            # A file-valued custom writer may use unique names in each batch.
            # Prior batches must have been cleaned before the next is written.
            previous = list(Path(artifact_path).parent.glob("batch-*.npy"))
            assert not previous
            np.save(Path(artifact_path).parent / ("batch-" + rows[0]["case"] + ".npy"), [1])
        writer(rows, artifact_path)

    scorer["_h_write_dummy"] = checked_writer
    try:
        if positive_index is None:
            with pytest.raises(ValueError, match="without ground-truth boxes"):
                scorer["self_test"]()
        else:
            probe = scorer["self_test"]()
            selected = pd.read_csv(probe["expected_path"], dtype=str)
            assert str(positive_index) in selected["case"].tolist()
            assert len(selected) == (4 if positive_index < 4 else 5)
            assert probe["score"]["metric"] == 0
            monkeypatch.delenv("BIOMAKER_SCORE_VAL_PATH")
            assert scorer["score"](*(probe[key] for key in (
                "prediction_path", "truth_path", "expected_path"))) == probe["score"]
        assert len(selection_batches) == (3 if positive_index is None else positive_index // 4 + 1)
        assert not Path("working/.adapter_probe/selection").exists()
    finally:
        scorer["cleanup_self_test"]()
