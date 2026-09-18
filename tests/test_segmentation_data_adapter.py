"""Medical mask readers preserve physical geometry and private references."""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from biomaker.utils.evaluator_adapter import compile_adapter
from biomaker.utils.standard_data_adapters import _SEGMENTATION_DATA_ADAPTER


NAMES = ["binary_dice", "cldice", "tree_length_detected", "branch_detected",
         "betti_error", "multiclass_dice", "multiclass_cldice", "branch_accuracy",
         "branch_f1", "subtree_consistency", "topological_distance"]


def protocol():
    return dict(
        evaluator_api_version=3, task_type="segmentation_3d",
        metrics=[dict(name=name, maximize=name not in ("betti_error", "topological_distance"))
                 for name in NAMES],
        key_columns=["case"], prediction_columns=["predicted mask"],
        answer_columns=["reference mask", "centerline path", "branches path", "edges path"],
        target_column="reference mask", image_column="ct path",
        auxiliary_columns=dict(skeleton="centerline path", parsing="branches path", graph="edges path"),
        geometry_policy="physical", file_reference_columns=["predicted mask"],
        validation_artifact="./working/validation_predictions.csv",
        test_submission="./working/submission.csv",
    )


def load_reader(spec):
    namespace = {}
    source = _SEGMENTATION_DATA_ADAPTER.replace("__BIOMAKER_SPEC__", repr(spec))
    exec(compile_adapter(source, spec), namespace)
    assert namespace["prediction_spec"]()["scoring_backend"] == "trusted_metrics_v1"
    return namespace


def write_volume(path, array, reference=None):
    image = sitk.GetImageFromArray(array)
    if reference is None:
        image.SetSpacing((0.7, 0.8, 1.2))
        image.SetOrigin((10.0, -8.0, 3.5))
        image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    else:
        image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))
    return image


@pytest.fixture
def medical_case(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("input").mkdir()
    Path("private").mkdir()
    Path("working").mkdir()
    target = np.zeros((5, 5, 8), dtype=np.uint8)
    target[2, 2, 1:4] = 1
    target[2, 2, 4:7] = 20
    parsing = (target > 0).astype(np.uint8)
    parsing[target == 20] = 2
    image = write_volume("input/ct.mha", np.full(target.shape, -700, dtype=np.int16))
    write_volume("private/label.mha", target, image)
    write_volume("private/skeleton.mha", (target > 0).astype(np.uint8), image)
    write_volume("private/parsing.mha", parsing, image)
    write_volume("working/prediction.mha", target, image)
    np.save("private/graph.npy", np.array([[0], [1]], dtype=np.int64), allow_pickle=False)
    expected = {"case": "001", "ct path": "input/ct.mha"}
    prediction = {"case": "001", "predicted mask": "prediction.mha"}
    truth = {"case": "001", "reference mask": "private/label.mha",
             "centerline path": "private/skeleton.mha", "branches path": "private/parsing.mha",
             "edges path": "private/graph.npy"}
    return dict(root=tmp_path, target=target, parsing=parsing, image=image,
                expected=expected, prediction=prediction, truth=truth)


def score_tables(case, reader):
    paths = []
    for name, row in [("prediction", case["prediction"]), ("truth", case["truth"]),
                      ("expected", case["expected"])]:
        path = case["root"] / (name + ".csv")
        pd.DataFrame([row]).to_csv(path, index=False)
        paths.append(str(path))
    return reader["score"](*paths)


def test_registered_mask_reader_scores_declared_private_auxiliary_columns(medical_case):
    reader = load_reader(protocol())
    sample = reader["load_sample"](medical_case["prediction"], medical_case["truth"],
                                   medical_case["expected"])
    assert np.array_equal(sample["target"], medical_case["target"])
    assert np.array_equal(sample["parsing"], medical_case["parsing"])
    assert np.array_equal(sample["graph"], np.array([[0], [1]]))
    assert sample["input_shape"] == (5, 5, 8)
    assert sample["input_geometry"]["size"] == [8, 5, 5]
    assert sample["input_geometry"]["spacing"] == pytest.approx([0.7, 0.8, 1.2])
    for field in ("prediction", "target", "skeleton", "parsing"):
        assert sample[field + "_geometry"] == sample["input_geometry"]
    result = score_tables(medical_case, reader)
    expected = dict.fromkeys(NAMES, 1.0)
    expected.update(betti_error=0.0, subtree_consistency=0.0, topological_distance=0.0)
    assert result["metrics"] == pytest.approx(expected)
    assert result["n"] == 1


@pytest.mark.parametrize("field", ["spacing", "origin", "direction"])
@pytest.mark.parametrize("file", ["working/prediction.mha", "private/label.mha",
                                  "private/skeleton.mha", "private/parsing.mha"])
def test_physical_geometry_mismatch_is_rejected_without_resampling(medical_case, file, field):
    image = sitk.ReadImage(file)
    before = sitk.GetArrayFromImage(image).copy()
    if field == "spacing":
        image.SetSpacing((1.7, 0.8, 1.2))
    elif field == "origin":
        image.SetOrigin((11.0, -8.0, 3.5))
    else:
        image.SetDirection(tuple(np.eye(3).ravel()))
    sitk.WriteImage(image, file)
    reader = load_reader(protocol())
    with pytest.raises(ValueError, match=field):
        sample = reader["load_sample"](medical_case["prediction"], medical_case["truth"],
                                       medical_case["expected"])
        reader["score_samples"]([sample], protocol())
    # CSV scoring intentionally redacts adapter exceptions that might contain
    # private target details, but must still reject the entire sample.
    with pytest.raises(ValueError):
        score_tables(medical_case, reader)
    assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(file)), before)


def test_prediction_only_validation_never_reads_private_files(medical_case):
    reader = load_reader(protocol())
    original_reader = reader["adapter_read_array"]
    opened = []

    def public_only(path):
        assert "private" not in Path(path).parts
        opened.append(str(path))
        return original_reader(path)

    reader["adapter_read_array"] = public_only
    # Delete only this test's private fixtures so an accidental direct read,
    # including np.load(graph), fails even if it bypasses adapter_read_array.
    for path in Path("private").iterdir():
        path.unlink()
    sample = reader["load_sample"](medical_case["prediction"], None, medical_case["expected"])
    assert not {"target", "skeleton", "parsing", "graph"}.intersection(sample)
    assert reader["validate_prediction"](sample, protocol()) == {"valid": True}
    assert opened == ["working/prediction.mha"]


def test_dummy_mask_copies_only_public_geometry_and_survives_relocation(medical_case):
    reader = load_reader(protocol())
    for path in Path("private").iterdir():
        path.unlink()
    outputs = []
    for name in ("first", "second"):
        artifact = "working/relocated/" + name + "/predictions.csv"
        reader["write_dummy_predictions"]([medical_case["expected"]], artifact)
        with open(artifact, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["case"] for row in rows] == ["001"]
        sample = reader["load_sample"](rows[0], None, medical_case["expected"])
        assert sample["prediction"].shape == medical_case["target"].shape
        assert sample["prediction_geometry"] == sample["input_geometry"]
        assert reader["validate_prediction"](sample, protocol()) == {"valid": True}
        outputs.append(rows[0]["predicted mask"])
    assert outputs[0] != outputs[1]
    assert all(Path(path).is_file() for path in outputs)


def test_physical_geometry_cannot_be_fabricated_for_npy_prediction(medical_case):
    np.save("working/output.npy", medical_case["target"], allow_pickle=False)
    prediction = dict(medical_case["prediction"], **{"predicted mask": "output.npy"})
    reader = load_reader(protocol())
    with pytest.raises(ValueError, match="physical geometry"):
        reader["load_sample"](prediction, None, medical_case["expected"])


def test_graph_loader_refuses_pickle_payloads(medical_case):
    np.save("private/graph.npy", np.array([{"not": "an integer edge list"}], dtype=object))
    reader = load_reader(protocol())
    with pytest.raises(ValueError, match="allow_pickle=False"):
        reader["load_sample"](medical_case["prediction"], medical_case["truth"], medical_case["expected"])


@pytest.mark.parametrize("change", ["undeclared_auxiliary", "extra_answer", "target_is_auxiliary"])
def test_builtin_reader_admission_requires_exact_answer_mapping(tmp_path, change):
    from biomaker.utils.evaluator_setup import EvaluatorSetupError, _resolve_metric_contract

    spec = protocol()
    spec["prediction_type"] = "mask_file_path"
    if change == "undeclared_auxiliary":
        spec["auxiliary_columns"]["graph"] = "not in answer_columns"
    elif change == "extra_answer":
        spec["answer_columns"].append("an undeclared private file")
    else:
        spec["target_column"] = "centerline path"
    with pytest.raises((ValueError, EvaluatorSetupError)):
        _resolve_metric_contract(spec, tmp_path, "", require_match=False)


def test_custom_reader_cannot_swap_frozen_auxiliary_mapping(tmp_path):
    from biomaker.utils.evaluator_setup import EvaluatorSetupError, _resolve_metric_contract

    spec = protocol()
    contract = {key: spec[key] for key in ("metrics", "target_column", "auxiliary_columns", "geometry_policy")}
    split_dir = tmp_path / "input" / "biomaker_splits"
    split_dir.mkdir(parents=True)
    (split_dir / "split_manifest.json").write_text(json.dumps({"metric_contract": contract}), encoding="utf-8")
    _resolve_metric_contract(spec, tmp_path, "", require_match=True)
    spec["auxiliary_columns"]["skeleton"] = "branches path"
    spec["auxiliary_columns"]["parsing"] = "centerline path"
    with pytest.raises(EvaluatorSetupError, match="auxiliary_columns"):
        _resolve_metric_contract(spec, tmp_path, "", require_match=True)
