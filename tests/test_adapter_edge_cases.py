"""Dataset encodings and degenerate probes must not change scoring semantics."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from biomaker.utils.evaluator_adapter import compile_adapter
from biomaker.utils.restoration_evaluator_template import _RESTORATION_DATA_ADAPTER
from biomaker.utils.standard_data_adapters import (
    _CLASSIFICATION_DATA_ADAPTER,
    _SEGMENTATION_DATA_ADAPTER,
)


def spec(task_type, primary, **extra):
    return dict(
        evaluator_api_version=2, task_type=task_type, primary_metric=primary,
        secondary_metrics=[], maximize=primary != "hd95", key_columns=["id"],
        prediction_columns=["prediction"], answer_columns=["answer"],
        validation_artifact="./working/validation_predictions.csv",
        test_submission="./working/submission.csv", **extra,
    )


def compile_reader(source, declaration):
    namespace = {}
    exec(compile_adapter(source.replace("__BIOMAKER_SPEC__", repr(declaration)), declaration), namespace)
    return namespace


def csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def splits(root, public, truth):
    folder = root / "input/biomaker_splits"
    csv(folder / "score_val.csv", public)
    csv(folder / "agent_test.csv", public)
    return csv(root / "private/answers.csv", truth)


@pytest.mark.parametrize("family", ["segmentation", "restoration"])
def test_multiframe_images_require_explicit_axis_decoding(tmp_path, monkeypatch, family):
    monkeypatch.chdir(tmp_path)
    frames = [Image.fromarray(np.full((4, 4), value, dtype=np.uint8)) for value in (1, 0, 0)]
    path = tmp_path / "stack.tif"
    frames[0].save(path, save_all=True, append_images=frames[1:])
    if family == "segmentation":
        declaration = spec("segmentation_3d", "binary_dice", image_column="image")
        reader = compile_reader(_SEGMENTATION_DATA_ADAPTER, declaration)["adapter_read_array"]
    else:
        declaration = spec("image_restoration", "psnr", restoration_evaluation=dict(
            data_range=255, crop_size=None, group_column=None, groups=None,
            aggregation="mean_images", input_column="image"))
        reader = compile_reader(_RESTORATION_DATA_ADAPTER, declaration)["adapter_image"]
    with pytest.raises(ValueError, match="Multi-frame.*data adapter"):
        reader(str(path))


def test_empty_box_cells_reach_the_declared_decoder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    declaration = spec("detection", "ap50")
    source = '''
import json
ADAPTER_SPEC = __BIOMAKER_SPEC__
def prediction_spec():
    return ADAPTER_SPEC
def adapter_boxes(value):
    return [] if value == '' else json.loads(value)
def load_sample(prediction_row, target_row, expected_row):
    sample = {'prediction': adapter_boxes(prediction_row['prediction'])}
    if target_row is not None:
        sample['target'] = adapter_boxes(target_row['answer'])
    return sample
def write_dummy_predictions(expected_rows, artifact_path):
    pass
'''
    scorer = compile_reader(source, declaration)
    box = json.dumps([[0, [0, 0, 2, 2], 1]])
    public = [{"id": "negative"}, {"id": "positive"}]
    truth = [{"id": "negative", "answer": ""}, {"id": "positive", "answer": box}]
    predictions = [{"id": "negative", "prediction": ""}, {"id": "positive", "prediction": box}]
    truth_path = splits(tmp_path, public, truth)
    artifact = csv(tmp_path / "working/submission.csv", predictions)
    expected = str(tmp_path / "input/biomaker_splits/agent_test.csv")
    assert scorer["validate_submission"]() == {"valid": True, "n": 2}
    assert scorer["score"](artifact, truth_path, expected)["metric"] == 1
    predictions[0]["id"] = ""
    csv(Path(artifact), predictions)
    with pytest.raises(ValueError, match="empty sample keys"):
        scorer["score"](artifact, truth_path, expected)


@pytest.mark.parametrize("multilabel", [False, True])
def test_probe_coverage_uses_decoded_targets_not_unique_answer_files(tmp_path, monkeypatch, multilabel):
    monkeypatch.chdir(tmp_path)
    declaration = spec("multilabel" if multilabel else "binary", "auc")
    if multilabel:
        declaration["prediction_columns"] = ["p0", "p1", "p2", "p3"]
    else:
        declaration["class_labels"] = ["negative", "positive"]
    source = _CLASSIFICATION_DATA_ADAPTER.replace("import csv", "import csv\nimport json")
    start = source.index("    if target_row is not None:\n")
    end = source.index("    return sample", start)
    source = source[:start] + '''    if target_row is not None:
        sample['target'] = json.loads(Path(target_row['answer']).read_text())['label']
''' + source[end:]
    scorer = compile_reader(source, declaration)
    public, truth = [], []
    for index in range(160):
        label = ([int(bool(index & (1 << bit))) for bit in range(4)] if multilabel
                 else "positive" if index % 2 else "negative")
        annotation = tmp_path / f"annotation-{index}.json"
        annotation.write_text(json.dumps({"sample": index, "label": label}))
        public.append({"id": str(index)})
        truth.append({"id": str(index), "answer": str(annotation)})
    writer, batches = scorer["_h_write_dummy"], []

    def public_only_writer(rows, artifact):
        assert all(set(row) == {"id"} for row in rows)
        batches.append(len(rows))
        writer(rows, artifact)

    scorer["_h_write_dummy"] = public_only_writer
    indices = scorer["_h_select_probe"](pd.DataFrame(public), pd.DataFrame(truth))
    assert len(indices) <= (4 + 4 * 4 if multilabel else 8)
    assert max(batches) <= 4
    assert sum(batches) < len(public)
    assert not Path("working/.adapter_probe/selection").exists()
    labels = [json.loads(Path(truth[index]["answer"]).read_text())["label"] for index in indices]
    if multilabel:
        assert all(sum(value == expected for value in column) >= 2
                   for column in zip(*labels) for expected in (0, 1))
    else:
        assert labels.count("negative") >= 2 and labels.count("positive") >= 2


def segmentation_probe(tmp_path, monkeypatch, *, malformed_second=False):
    monkeypatch.chdir(tmp_path)
    public, truth = [], []
    for index in range(2):
        image, answer = tmp_path / f"image-{index}.npy", tmp_path / f"answer-{index}.npy"
        np.save(image, np.zeros((4, 4), dtype=np.uint8))
        np.save(answer, np.zeros((3, 4) if index and malformed_second else (4, 4), dtype=np.uint8))
        public.append({"id": str(index), "image": str(image)})
        truth.append({"id": str(index), "answer": str(answer)})
    truth_path = splits(tmp_path, public, truth)
    monkeypatch.setenv("BIOMAKER_SCORE_VAL_PATH", truth_path)
    declaration = spec("segmentation", "hd95", image_column="image")
    return compile_reader(_SEGMENTATION_DATA_ADAPTER, declaration)


def test_undefined_dummy_hd95_is_structural_success_but_final_score_stays_strict(tmp_path, monkeypatch):
    scorer = segmentation_probe(tmp_path, monkeypatch)
    try:
        probe = scorer["self_test"]()
        assert probe["score"] is None
        assert probe["score_error"]["metric"] == "hd95"
        assert probe["structural_probe"] == {"validated_samples": 2}
        with pytest.raises(scorer["UndefinedMetricError"]):
            scorer["score"](*(probe[key] for key in ("prediction_path", "truth_path", "expected_path")))
        # A real, correct empty-mask prediction has the unchanged defined score.
        predictions = pd.read_csv(probe["prediction_path"], dtype=str)
        for filename in predictions["prediction"]:
            np.save(filename, np.zeros((4, 4), dtype=np.uint8))
        assert scorer["score"](*(probe[key] for key in ("prediction_path", "truth_path", "expected_path")))["metric"] == 0
    finally:
        scorer["cleanup_self_test"]()


def test_undefined_first_metric_cannot_hide_later_invalid_sample(tmp_path, monkeypatch):
    scorer = segmentation_probe(tmp_path, monkeypatch, malformed_second=True)
    try:
        with pytest.raises(ValueError, match="shapes must match"):
            scorer["self_test"]()
    finally:
        scorer["cleanup_self_test"]()


def test_harness_rechecks_undefined_probe_at_relocated_paths(tmp_path, monkeypatch):
    from biomaker.utils.evaluator_setup import _run_self_test

    segmentation_probe(tmp_path, monkeypatch)
    declaration = spec("segmentation", "hd95", image_column="image")
    source = _SEGMENTATION_DATA_ADAPTER.replace("__BIOMAKER_SPEC__", repr(declaration))
    (tmp_path / "evaluator.py").write_text(compile_adapter(source, declaration))
    output = _run_self_test(tmp_path, timeout=30, hidden_score_path=tmp_path / "private/answers.csv")
    report = json.loads(next(line.removeprefix("EVALUATOR_PROBE_JSON=")
                             for line in output.splitlines() if line.startswith("EVALUATOR_PROBE_JSON=")))
    assert report["score"] is None
    assert report["score_error"]["metric"] == "hd95"
    assert report["structural_probe"] == {"validated_samples": 2}
