# Evaluation contract

The task description and experiment protocol determine the metrics and their
settings. The preparation agent interprets the dataset and adapts predictions
and answers into canonical samples. The Harness owns sample alignment, trusted
metric implementations, aggregation, preparation checks, and private-answer
access. Candidate agents remain free to choose models, losses, training,
preprocessing, and how to produce the declared prediction artifacts.

## Three separate contracts

| Contract | Authority | What it specifies |
| --- | --- | --- |
| Task evaluation protocol | User and experiment protocol | Sample unit, splits, required metrics, metric parameters, aggregation, and which answers remain private. |
| Metric capabilities | Harness | Available implementations, canonical input schemas, accepted parameters and aliases, explicit defaults, and unsupported cases. |
| Data adapter | Preparation agent | How dataset and prediction representations map into canonical samples, using declarations when sufficient and Python decoding when needed. |

The capability catalog describes what the Harness can compute; it does not
override the task protocol. Defaults apply only where the protocol leaves a
choice unspecified. A declared nondefault setting must be preserved or reported
as unsupported. Equivalent parameter spellings are normalized before checking
the frozen protocol.

The capability catalog separates `protocol_fields` (accepted settings, aliases,
defaults and fixed values) from `documentation` (explanations of scoring).
Each family includes a minimal `metric_contract_example`; the agent adapts its
settings to the task rather than copying example dataset assumptions.
`sample_schema` describes the values an adapter returns, not the original
dataset layout. Column names, storage formats, and decoding remain agent-owned.

Preparation validates data and metric declarations separately. When the builder
finishes successfully and the data checks pass, a declaration error is returned
to the preparation agent with the task, prepared data semantics, capability
catalog, and exact diagnostic. The agent returns a corrected `metric_contract`
JSON object. Harness validates it before replacing that field; it does not rerun
the builder or rewrite the split tables, converted files, or private answers.
Rejected proposals leave the manifest unchanged and are included in the next
repair prompt. Builder and declaration repairs share the configured preparation
attempt budget. Data or execution failures still return to the builder.

Known equivalent aliases are normalized without another model call. A named
primary metric such as `{"name": "aji", "aji_matching": "per_gt_best_iou"}`
preserves its settings just like their top-level equivalents; conflicting names
or settings are errors. Custom metric settings are preserved for their separate
implementation and validation. Unknown fields are never automatically deleted
or downgraded to notes: the agent must distinguish redundant explanation from a
real requirement using the task. Unsupported required semantics remain explicit
errors. Cached packages are revalidated before reuse; invalid ones are rejected.

All supported classification, regression, segmentation, detection, and restoration paths
use the same adapter runtime and trusted metric kernels. Built-in adapters
cover familiar declared representations. Generated adapters extend data
interpretation when those declarations are insufficient. The old standalone
scoring templates remain compatibility references; they are not a separate
metric implementation selected for new runs.

This split preserves useful autonomy. A model may resize or normalize its
training inputs, learn a different representation, choose a loss, and use any
training strategy permitted by the experiment. Its final predictions must
represent the declared sample and target. An evaluation adapter may decode a
serialized probability vector, map labels in an explicitly declared order,
resolve paths, or convert a declared box representation into canonical
coordinates. Those are semantic conversions. It may not choose a new threshold,
drop difficult samples, relabel instances, crop evaluation arrays, or rescale
intensities merely to improve the score. Such evaluation operations belong to
the frozen task protocol and trusted kernels.

For a supported metric, a schema or file-format mismatch requires an adapter.
It does not require regenerating metric formulas. If the requested metric or
its settings have no supported implementation, preparation must report that
fact explicitly. A custom metric implementation needs its own validation; it
must preserve the requested definition rather than approximate it silently.

## Public files and frozen protocol

Preparation records data semantics in `input/biomaker_splits/split_manifest.json`:

- `key_columns` identifies each sample using public values. Composite keys and
  input paths are allowed; private answers cannot be keys.
- `prediction_contract` describes prediction columns, values, and referenced
  file formats. Original data column names and directory layouts are not fixed.
- `metric_contract` records the primary and secondary metrics, direction, and
  all task-specific settings, including class order, thresholds, crop rules,
  intensity ranges, grouping, and aggregation when applicable.
- `hidden_answers` declares answer column names, without exposing answer
  values or private filesystem locations to candidate agents.
- `data_semantics`, `row_representation`, and `semantic_assumptions` preserve
  the preparation agent's explanation of units, axes and serialized values.
  `input_file_reference_columns` describes source files separately from
  `prediction_contract.file_reference_columns`, which contains only prediction
  columns. Declared input columns are included in public sample observations
  even when their names contain neither `path` nor `file`.

Private answer tables need the declared sample keys and answer columns; they
need not duplicate public input metadata. The Harness aligns keys and supplies
public metadata separately. Keyless tables require an explicit row-order
contract. If a builder lists known sample keys among `prediction_columns`,
preparation removes those keys deterministically and records the normalization;
the candidate artifact still contains keys followed by predicted values.

The resulting `evaluator.py` and `evaluator_prediction_spec.json` are frozen
before candidate search. Candidates write the declared validation artifact and
`./working/submission.csv`. The Harness applies the same frozen scoring
implementation to validation and private test; private test results are not
used to guide the candidate search. Existing run artifacts are not rewritten
when a new evaluator implementation is installed.

## Data adapter API

An adapter is a small Python module with three functions. It loads and decodes
data; it does not align whole splits, implement metric formulas, select metric
settings from observed answers, or aggregate scores.

```python
def prediction_spec() -> dict:
    ...

def load_sample(prediction_row: dict, target_row: dict | None,
                expected_row: dict) -> dict:
    ...

def write_dummy_predictions(expected_rows: list[dict], artifact_path: str):
    ...
```

Use `adapter_*` names for helper functions and `ADAPTER_*` names for constants.
The module may contain imports, these functions, and constants; the Harness
reserves its scoring entry points and `_h_*` namespace. Use installed packages
for decoding, without importing `evaluator` or `biomaker`, dynamic imports,
`eval`, `exec`, package installation, network access, or training.

`prediction_spec()` returns a JSON-serializable contract. It includes
`evaluator_api_version: 2`, `task_family`, `task_type`, `validation_artifact`,
`test_submission: "./working/submission.csv"`, `key_columns`,
`prediction_columns`, `answer_columns`, `primary_metric`, `secondary_metrics`,
and boolean `maximize`. It declares metric-specific settings explicitly.
`file_reference_columns` identifies prediction columns containing file paths.
Only the private evaluator needs the answer mapping; public prompts and reports
must not include private answer values or paths.

`load_sample()` receives one aligned sample. `prediction_row` contains the
declared prediction values and keys; `expected_row` contains public metadata.
`target_row` contains the matched answer when scoring. CSV cells are strings;
parse the declared encodings explicitly. Return a dictionary with
canonical `prediction` and, when a target row is supplied, `target`. Depending
on the task it may also return `group`, `input_shape`, `spacing`, and
`prediction_spacing`. For example, an image adapter can decode TIFF or NumPy
files into numeric arrays, and a detection adapter can decode serialized boxes
into the declared coordinate representation. It must preserve the experiment's
meaning of intensity, axes, labels, and coordinates.

When `target_row is None`, the call checks prediction format using public
information only. Do not read or infer hidden answers. Restoration adapters
must return `input_shape` from the public input so prediction shape can be
checked without answers. Relative referenced files resolve
from the run workspace; `input/...` is already workspace-relative.

`write_dummy_predictions()` receives a Harness-selected list of public rows and
the output CSV path. Write exactly those keys and the declared prediction
columns, creating referenced dummy files under `./working/.adapter_probe/` if
necessary.
Fixtures must have valid formats and shapes, and must remain available when
the function returns. The Harness owns their cleanup after all preparation
checks. This function does not read private answers or compute scores.

Preparation distinguishes invalid data from a mathematically undefined dummy
score. The Harness validates all selected decoded samples first. For a typed
undefined outcome (for example a nonempty HD95 dummy against a valid empty
mask), it checks that relocated scoring reproduces the same outcome, without
changing the protocol or asking the writer to access private answers. Actual
candidate scoring still rejects undefined requested metrics.

The preparation prompt supplies the supported metric settings and canonical
sample requirements. Declaring an arbitrary `task_type` or metric parameter
does not imply that the backend supports it.

The preparation prompt supplies the capability catalog, full prepared
`metric_contract` and `prediction_contract`, public sample observations, the
three function signatures, and a complete adapter example. Agents therefore
receive the interface constraints before implementing an adapter. The catalog
does not prescribe raw data column names or source directory layouts. The
normalized prediction spec records the accepted settings so every candidate
can inspect the resulting contract.

Source checks reject unsupported adapter entry points and common forbidden
operations. They validate this programming interface; they are not a security
sandbox for arbitrary malicious Python.

For grouped restoration, `restoration_evaluation.group_column` must name an
arbitrary public metadata column containing the declared group values. The
Harness uses it to include every group in bounded probes and supplies or checks
the returned `group`. If groups originate in a directory name or serialized
metadata, expose that interpretation as a public column during preparation.

The trusted backend normalizes the family as `classification`, `regression`, `segmentation`,
`image_restoration`, or `detection`:

| Family | Canonical sample values |
| --- | --- |
| Classification | Probabilities in the declared `class_labels` order and target labels. `task_type` declares binary, multiclass, or multilabel semantics; a single binary probability requires `positive_class`. |
| Regression | Finite scalar or 1D vector predictions and targets with matching shapes. MAE/MSE average samples and output dimensions uniformly; RMSE is the square root of that MSE. Preserve target units and never round or clip. |
| Segmentation | Nonnegative integer 2D or 3D masks, with zero as background and instance identifiers preserved. Target and prediction shapes must agree. Optional per-axis spacing must be positive and agree with prediction spacing. |
| Image restoration | Finite real 2D arrays with matching shapes. Targets lie within the declared intensity range; predictions are clipped to that range by the Harness. |
| Detection | A list of `(class_id, (x1, y1, x2, y2), confidence)` records, with nonnegative integer class IDs, finite coordinates and confidence, and positive box extents. |

Availability of a family does not imply support for every metric variant in
that family. The preparation prompt and normalized prediction spec identify
the supported implementation and its parameters.

Unsupported declared settings are checked before adapter generation, so an
implementation gap does not trigger repeated requests to rewrite a data reader.
Custom metric evaluators must preserve the prepared primary/secondary metrics,
direction, parameters, keys and prediction columns as well. Equivalent supported
parameter spellings are normalized for this comparison.

## Source roles and final held-out data

A task may include a JSON `split-constraints` block with `required_splits`,
`required_answer_splits`, and `source_rules`. Each rule declares an original
file or directory `root`, its `allowed_splits` (`train`, `score_val`, `test`),
and optional `require_complete`. The Harness checks actual source references,
coverage, and cross-split reuse; the agent still chooses columns, grouping and
internal split membership within those constraints. Source directories named
`validation` can be assigned to final private `test` by the experiment protocol.
Cached preparation packages are revalidated against the same constraints.
Coverage checks may use private answer-file references without exposing them as
public input metadata. These are file-membership constraints; a table requiring
row-level source rules needs its own declared row-membership protocol.

## Classification adapter example

Suppose a dataset uses `case_uid` as its public key and `diagnosis` as its
private answer column. A candidate writes one JSON probability vector in
`probabilities_json` per sample. No raw image filename or directory convention
is required by this adapter:

```python
import json
from pathlib import Path
import pandas as pd

ADAPTER_SPEC = {
    "evaluator_api_version": 2,
    "task_family": "classification",
    "task_type": "binary",
    "key_columns": ["case_uid"],
    "prediction_columns": ["probabilities_json"],
    "answer_columns": ["diagnosis"],
    "validation_artifact": "./working/validation_predictions.csv",
    "test_submission": "./working/submission.csv",
    "primary_metric": "auc",
    "secondary_metrics": ["accuracy"],
    "maximize": True,
    "class_labels": ["normal", "lesion"],
    "positive_class": "lesion",
    "threshold": 0.5,
    "auc_average": "macro",
    "multi_class": "ovr",
    "aggregation": "mean_samples",
}

def prediction_spec():
    return ADAPTER_SPEC

def load_sample(prediction_row, target_row, expected_row):
    sample = {
        "prediction": json.loads(prediction_row["probabilities_json"])
    }
    if target_row is not None:
        sample["target"] = target_row["diagnosis"]
    return sample

def write_dummy_predictions(expected_rows, artifact_path):
    rows = [
        {"case_uid": row["case_uid"],
         "probabilities_json": json.dumps([0.5, 0.5])}
        for row in expected_rows
    ]
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(artifact_path, index=False)
```

The adapter handles JSON decoding and the answer-column mapping. The Harness
aligns `case_uid`, checks probabilities and class order, calculates AUC and
accuracy with the declared parameters, and owns the fixture lifetime. A
different storage format changes this small adapter, not the metric code.

## Restoration example

A dataset can choose `sample`, `source`, `reference`, `condition`, and `estimate`
as its columns. Its metric declaration can use:

```json
{
  "primary_metric": "psnr",
  "secondary_metrics": ["ssim"],
  "maximize": true,
  "input_column": "source",
  "target_column": "reference",
  "data_range": 255,
  "crop_size": 256,
  "group_column": "condition",
  "groups": ["raw", "avg2", "avg4", "avg8", "avg16"],
  "aggregation": "mean_groups",
  "ssim_parameters": {
    "win_size": 7,
    "gaussian_weights": false,
    "use_sample_covariance": true,
    "K1": 0.01,
    "K2": 0.03
  }
}
```

The adapter decodes `estimate` and `reference` into arrays and passes the
public `condition` as `group`. The Harness performs the declared crops,
PSNR/SSIM computation, per-image averaging, and group aggregation. Parameter
aliases such as `window_size` are normalized to the public canonical setting;
conflicting declarations must fail. Settings such as a Gaussian SSIM window
must never be silently changed to uniform weights just to match a template.

## Preparation and fixture lifetime

Before freezing, the Harness checks a bounded selection of real public rows,
including declared groups, and uses their matched answers only inside the
scoring process. It checks sample identity, decoded shapes and values,
prediction validation without answers, and scoring with explicitly supplied
paths. It also checks that all requested metric values are returned and agree
with the primary metric and optimization direction.

Fixture creation and cleanup belong to one Harness-controlled lifecycle:

1. Select public rows and create prediction fixtures through the adapter.
2. Validate public artifacts without supplying private answers.
3. Run scoring probes using explicitly supplied prediction, truth, and expected
   row paths, including paths different from the normal validation filenames.
4. Remove fixtures after every probe has completed, including on failure.
5. Freeze the validated protocol and adapter before candidate search.

A protocol or adapter failure is preparation feedback. It must not cause a
supported standard metric to be replaced with a generated scoring formula.

## Custom evaluator compatibility

When no trusted implementation is available, a custom evaluator still follows
the existing split-independent API:

```python
def score(prediction_path, truth_path, expected_path):
    # One implementation for both validation and private test.
    return {"metrics": {"custom_metric": value},
            "metric": value, "maximize": True}
```

It also defines `prediction_spec()`, `score_val()`, `validate_submission()`,
and `self_test()`. `score_val()` only delegates to `score()` using the validation
prediction path, `BIOMAKER_SCORE_VAL_PATH`, and the public validation split.
`score()` consumes its explicit paths and does not depend on that environment
variable or validation-specific row counts. `validate_submission()` checks
format only and never reads private test answers.

This is an escape hatch for unavailable metric definitions or settings, not
for unfamiliar columns or file formats. The Harness supplies
`trusted_score_samples(samples, spec)` in the custom evaluator's standalone
module. For requested metrics already available in the catalog, use that
function with canonical samples and a supported normalized spec; implement
and verify only the unavailable metric definitions. The same protocol and
private-answer boundaries apply to both kinds of metric.

For this legacy custom-evaluator lifecycle, `self_test()` creates dummy
validation and test prediction CSVs plus any referenced files, exercises
scoring and submission validation, and returns with those files intact.
The Harness reads them again for its split-independent probe, then cleans
`./working`. Deleting them inside `self_test()` violates the interface.

Adapters for trusted metrics do not implement `self_test()`. The Harness
generates that lifecycle, keeps temporary answer fixtures outside the public
workspace, and removes them after the relocated-path scoring probe finishes.
The adapter and trusted runtime are bundled into the standalone frozen
`evaluator.py`, so copying or reusing the evaluator preserves its scoring code.

All requested metrics must be present and finite. Undefined metrics must raise
an explicit error; they cannot be replaced with a different metric. API-v1
evaluators require a new run instead of being silently rescored with a new
implementation.
