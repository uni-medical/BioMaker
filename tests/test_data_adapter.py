"""The adapter interprets samples while the frozen Harness owns evaluation."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biomaker.utils.evaluator_adapter import compile_adapter, validate_adapter_source


def restoration_spec():
    return {
        "evaluator_api_version": 2,
        "task_type": "image_restoration",
        "key_columns": ["source", "condition"],
        "prediction_columns": ["estimate"],
        "answer_columns": ["reference"],
        "validation_artifact": "./working/validation_predictions.csv",
        "test_submission": "./working/submission.csv",
        "primary_metric": "psnr",
        "secondary_metrics": ["ssim"],
        "maximize": True,
        "restoration_evaluation": {
            "data_range": 255,
            "crop_size": None,
            "group_column": "condition",
            "groups": ["low", "medium", "high"],
            "aggregation": "mean_groups",
        },
        "ssim_parameters": {"window_size": 7},
    }


def adapter_source(spec=None, *, sample_body=None):
    spec = restoration_spec() if spec is None else spec
    sample_body = sample_body or """
    assert set(prediction_row) == {'source', 'condition', 'estimate'}
    sample = {'prediction': np.asarray(json.loads(prediction_row['estimate'])),
              'input_shape': tuple(json.loads(expected_row['shape']))}
    if target_row is not None:
        assert set(target_row) == {'source', 'condition', 'reference'}
        sample['target'] = np.asarray(json.loads(target_row['reference']))
    return sample
"""
    return f"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
ADAPTER_SPEC = {spec!r}
def prediction_spec():
    return ADAPTER_SPEC
def load_sample(prediction_row, target_row, expected_row):
{sample_body}
def write_dummy_predictions(expected_rows, artifact_path):
    rows = []
    for row in expected_rows:
        estimate = np.full(tuple(json.loads(row['shape'])), 2.0)
        rows.append({{'source': row['source'], 'condition': row['condition'],
                     'estimate': json.dumps(estimate.tolist())}})
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows[::-1]).to_csv(artifact_path, index=False)
"""


def load_adapter(source=None, frozen_spec=None):
    namespace = {}
    exec(compile_adapter(source or adapter_source(), frozen_spec), namespace)
    return namespace


def write_split(root, *, name="score_val", count=45, first_id=0):
    public, truth, predictions = [], [], []
    groups = ["low", "medium", "high"]
    for index in range(count):
        keys = {"source": f"scan-{first_id + index:04d}", "condition": groups[index % 3]}
        public.append({**keys, "shape": "[8, 8]"})
        truth.append({**keys, "reference": json.dumps(np.zeros((8, 8)).tolist())})
        predictions.append({**keys, "estimate": json.dumps(np.full((8, 8), index % 5 + 1.0).tolist()),
                            # Neither adapter nor trusted kernel should see these.
                            "reference": "candidate-supplied fake answer", "label": -1})
    public_path = root / f"input/biomaker_splits/{name}.csv"
    truth_path = root / f"private/{name}.csv"
    prediction_path = root / "working" / ("submission.csv" if name == "agent_test" else "validation_predictions.csv")
    for path in (public_path, truth_path, prediction_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(public).to_csv(public_path, index=False)
    pd.DataFrame(truth[::-1]).to_csv(truth_path, index=False)
    pd.DataFrame(predictions[::-1]).to_csv(prediction_path, index=False)
    return prediction_path, truth_path, public_path


def test_adapter_aligns_arbitrary_fields_and_uses_identical_math_on_other_split(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scorer = load_adapter()
    paths = write_split(tmp_path)
    result = scorer["score"](*map(str, paths))
    expected = np.mean([20 * np.log10(255 / error) for error in (1, 2, 3, 4, 5)])
    assert result["metric"] == pytest.approx(expected)
    assert result["n"] == 45
    assert set(result["per_group"]) == {"low", "medium", "high"}
    test_paths = write_split(tmp_path, name="agent_test", count=15, first_id=100)
    monkeypatch.delenv("BIOMAKER_SCORE_VAL_PATH", raising=False)
    test_score = scorer["score"](*map(str, test_paths))
    assert test_score["metric"] == pytest.approx(result["metric"])
    assert test_score["n"] == 15
    assert scorer["prediction_spec"]()["ssim_parameters"]["win_size"] == 7


def test_public_validation_works_without_any_truth_and_returns_no_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scorer = load_adapter()
    _, truth, _ = write_split(tmp_path, name="agent_test", count=3)
    truth.unlink()
    monkeypatch.delenv("BIOMAKER_SCORE_VAL_PATH", raising=False)
    assert scorer["validate_submission"]() == {"valid": True, "n": 3}
    assert "answer_columns" not in scorer["prediction_spec"]()


@pytest.mark.parametrize("group_column", [None, ""])
def test_grouped_adapter_requires_a_public_group_column(tmp_path, monkeypatch, group_column):
    monkeypatch.chdir(tmp_path)
    spec = restoration_spec()
    spec["restoration_evaluation"]["group_column"] = group_column
    with pytest.raises(ValueError, match="public group_column"):
        load_adapter(adapter_source(spec))


def test_probe_samples_are_bounded_reusable_and_truth_stays_outside_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scorer = load_adapter()
    _, truth, _ = write_split(tmp_path, count=300)
    write_split(tmp_path, name="agent_test", count=120, first_id=1000)
    monkeypatch.setenv("BIOMAKER_SCORE_VAL_PATH", str(truth))
    probe = scorer["self_test"]()
    private_fixture = Path(probe["truth_path"])
    try:
        expected = pd.read_csv(probe["expected_path"])
        assert 3 <= len(expected) <= 4 + 2 * 3
        assert set(expected["condition"]) == {"low", "medium", "high"}
        assert not private_fixture.is_relative_to(tmp_path)
        assert all(Path(probe[key]).is_file() for key in ("prediction_path", "truth_path", "expected_path"))
        monkeypatch.delenv("BIOMAKER_SCORE_VAL_PATH")
        assert scorer["score"](*(probe[key] for key in ("prediction_path", "truth_path", "expected_path"))) == probe["score"]
    finally:
        scorer["cleanup_self_test"]()
    assert not private_fixture.exists()
    # The outer preparation Harness removes public dummy artifacts after probes.
    assert Path(probe["prediction_path"]).is_file()


def test_frozen_spec_is_independent_of_adapter_declaration_and_public_mutation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = restoration_spec()
    source = adapter_source().replace("return ADAPTER_SPEC", "raise RuntimeError('dynamic spec must not run after freezing')")
    scorer = load_adapter(source, spec)
    public = scorer["prediction_spec"]()
    public["restoration_evaluation"]["data_range"] = 1
    spec["restoration_evaluation"]["data_range"] = 2
    assert scorer["prediction_spec"]()["restoration_evaluation"]["data_range"] == 255
    assert scorer["_h_adapter_spec"]()["answer_columns"] == ["reference"]
    assert scorer["prediction_spec"]()["scoring_backend"] == "trusted_metrics_v1"


def test_referenced_numpy_fixtures_survive_until_harness_finishes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = restoration_spec()
    spec["file_reference_columns"] = ["estimate"]
    source = adapter_source(spec).replace(
        "np.asarray(json.loads(prediction_row['estimate']))",
        "np.load(prediction_row['estimate'], allow_pickle=False)",
    ).replace(
        "'estimate': json.dumps(estimate.tolist())",
        "'estimate': str(Path(artifact_path).with_suffix('')) + '-' + str(len(rows)) + '.npy'",
    ).replace(
        "Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)",
        """Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        np.save(row['estimate'], np.full((8, 8), 2.0))""",
    )
    scorer = load_adapter(source)
    _, truth, _ = write_split(tmp_path, count=30)
    write_split(tmp_path, name="agent_test", count=30)
    monkeypatch.setenv("BIOMAKER_SCORE_VAL_PATH", str(truth))
    probe = scorer["self_test"]()
    try:
        references = pd.read_csv(probe["prediction_path"])["estimate"]
        assert all(Path(path).is_file() for path in references)
        assert all(Path(path).is_relative_to("working/.adapter_probe") for path in references)
        assert scorer["score"](*(probe[key] for key in ("prediction_path", "truth_path", "expected_path"))) == probe["score"]
    finally:
        scorer["cleanup_self_test"]()


def test_compiled_evaluator_is_standalone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = write_split(tmp_path, count=15)
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(compile_adapter(adapter_source(), restoration_spec()))
    script = """
import builtins, json, runpy
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'biomaker' or name.startswith('biomaker.'):
        raise AssertionError('Frozen evaluator must not import BioMaker')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
module = runpy.run_path('evaluator.py')
print(json.dumps(module['score'](*json.loads(__import__('sys').argv[1]))))
"""
    completed = subprocess.run([sys.executable, "-I", "-c", script, json.dumps(list(map(str, paths)))],
                               cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["n"] == 15


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "wrong_key", "empty", "missing_column"])
def test_malformed_prediction_rows_fail_before_sample_interpretation(tmp_path, monkeypatch, mutation):
    monkeypatch.chdir(tmp_path)
    scorer = load_adapter()
    paths = write_split(tmp_path, count=6)
    frame = pd.read_csv(paths[0], dtype=str)
    if mutation == "duplicate":
        frame.iloc[0] = frame.iloc[1]
    elif mutation == "missing":
        frame = frame.iloc[1:]
    elif mutation == "wrong_key":
        frame.loc[0, "source"] = "unknown-sample"
    elif mutation == "empty":
        frame.loc[0, "source"] = ""
    else:
        frame = frame.drop(columns=["estimate"])
    frame.to_csv(paths[0], index=False)
    with pytest.raises(ValueError, match="Predictions"):
        scorer["score"](*map(str, paths))


@pytest.mark.parametrize("body", [
    "    return {'prediction': np.ones((8, 8))}",
    "    return {'prediction': np.ones((8, 8)), 'target': np.zeros((8, 8)), 'group': 'wrong'}",
    "    raise ValueError('secret answer: ' + target_row['reference'])",
])
def test_private_adapter_errors_do_not_expose_targets(tmp_path, monkeypatch, body):
    monkeypatch.chdir(tmp_path)
    scorer = load_adapter(adapter_source(sample_body=body))
    paths = write_split(tmp_path, count=3)
    with pytest.raises(ValueError, match="Data adapter failed") as exc:
        scorer["score"](*map(str, paths))
    assert "secret answer" not in str(exc.value)
    assert "[[0.0" not in str(exc.value)


def test_format_only_adapter_must_not_return_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = adapter_source(sample_body="    return {'prediction': np.ones((8, 8)), 'target': np.zeros((8, 8))}")
    scorer = load_adapter(source)
    write_split(tmp_path, name="agent_test", count=3)
    with pytest.raises(ValueError, match="Format-only"):
        scorer["validate_submission"]()


def test_restoration_format_validation_requires_public_input_shape(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = adapter_source(sample_body="    return {'prediction': np.ones((8, 8))}")
    scorer = load_adapter(source)
    write_split(tmp_path, name="agent_test", count=3)
    with pytest.raises(ValueError, match="input_shape"):
        scorer["validate_submission"]()


@pytest.mark.parametrize("suffix", [
    "\ndef score(*args):\n    return 1\n",
    "\ndef self_test():\n    pass\n",
    "\ndef metric_formula():\n    return 1\n",
    "\nimport biomaker\n",
    "\nimport json as _h_json\n",
])
def test_adapter_cannot_declare_harness_entry_points(suffix):
    with pytest.raises(ValueError):
        validate_adapter_source(adapter_source() + suffix)


def test_forbidden_import_inside_adapter_function_is_rejected():
    source = adapter_source(sample_body="    from biomaker.utils import trusted_metrics\n    return {}")
    with pytest.raises(ValueError, match="cannot import"):
        validate_adapter_source(source)


def test_dynamic_import_inside_adapter_function_is_rejected():
    source = adapter_source(sample_body="    return __import__('evaluator').prediction_spec()")
    with pytest.raises(ValueError, match="dynamic imports"):
        validate_adapter_source(source)


def test_aliased_dynamic_import_loader_is_rejected():
    source = adapter_source() + "\nfrom importlib import import_module as adapter_import\n"
    with pytest.raises(ValueError, match="dynamic module loaders"):
        validate_adapter_source(source)
