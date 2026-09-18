# Evaluation contract

The task description and experiment protocol determine the metrics and their
settings. The preparation agent interprets the dataset and adapts predictions
and answers into canonical samples. The Harness owns sample alignment, trusted
metric implementations, aggregation, preparation checks, and private-answer
access. Candidate agents remain free to choose models, losses, training,
preprocessing, and how to produce the declared prediction artifacts.

## Multiple objectives and candidate ranking

New evaluators declare API v3 and a nonempty ordered objective list:

```json
{
  "evaluator_api_version": 3,
  "metrics": [
    {"name": "binary_dice", "maximize": true},
    {"name": "hd95", "maximize": false}
  ]
}
```

Task files may declare the same JSON array in a fenced `metrics` block. Metric
names are canonicalized, and identical duplicate declarations are merged.
Directions are inferred for registered names and must agree with the registry;
custom metrics require an explicit boolean direction. Preserve metric parameters
and the existing single-family sample contract. Arbitrary metric count does not
add mixed task-head support.

The evaluator returns raw `metrics: {name: finite_number}` and diagnostics.
It does not select a winner or return a population-dependent ranking. Every
declared objective must be present and finite. Additional group scores are
diagnostics, not additional objectives.

For each objective, the Journal ranks all committed, completed, valid candidates
in the run, with 1 best and equal values receiving the mean of their occupied
positions. It averages these ranks with equal weights. Selection sorts by lower
average rank, then lower worst individual rank, then earlier step and node ID.
Search first computes this global ranking and only then filters for expandable
parents. One objective is equivalent to ordering by its raw value.

Rankings are dynamic: committing another valid candidate recomputes the ranks
of every earlier candidate and can promote an older node. Each round records its
selection snapshot; already scheduled parents stay fixed. Current displays,
best model artifacts and final retraining consume the current shared snapshot.
Historical rank changes across different candidate populations are not absolute
model improvements. Private test values never enter the ranking.

Legacy `primary_metric` / `secondary_metrics` declarations remain accepted for
new runs, with all declared metrics converted into equal objectives. Old frozen
API-v2 evaluators remain readable and can be rescored without changing formulas.
Historical journals lacking an average-rank policy retain their original scalar
selection. New caches and journals record the v3 protocol and selection policy.

Terminal and HTML summaries show one raw value for one objective, both raw
values for two, and average rank plus the first objective's raw value for three
or more. This first objective is only the display metric; declaration order
does not affect selection. Full values, directions and individual ranks remain
available in the detail tables and exports. Ranking snapshots include their
candidate count and version; raw values are never overwritten by ranks.

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
- `metric_contract` records the `metrics` list, per-metric direction, and
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

`input_file_reference_columns` is the complete list of existing input file
columns; the existing declaration inside `prediction_contract` is also accepted.
Keys may be file references when listed there. Other input columns remain opaque
identifiers or metadata, regardless of slashes or filename extensions in values.
For each `hidden_answers` split, `file_reference_columns` optionally lists the
subset of `answer_columns` containing files. Mirror it in the corresponding
`private_gt/private_gt.json` entry; `[]` explicitly declares inline/scalar answers.
Input and answer roles without file declarations retain legacy path inference.
Declared file references must be nonempty existing files on every row, not
directories. A restoration `target_column` alone does not declare file encoding;
restoration targets must still be nonempty even with an inline encoding.

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
`evaluator_api_version: 3`, `task_family`, `task_type`, `validation_artifact`,
`test_submission: "./working/submission.csv"`, `key_columns`,
`prediction_columns`, `answer_columns`, and a nonempty `metrics` list containing
each name and boolean `maximize`. It declares metric-specific settings explicitly.
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
    "evaluator_api_version": 3,
    "task_family": "classification",
    "task_type": "binary",
    "key_columns": ["case_uid"],
    "prediction_columns": ["probabilities_json"],
    "answer_columns": ["diagnosis"],
    "validation_artifact": "./working/validation_predictions.csv",
    "test_submission": "./working/submission.csv",
    "metrics": [{"name": "auc", "maximize": True},
                {"name": "accuracy", "maximize": True}],
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
  "metrics": [{"name": "psnr", "maximize": true},
              {"name": "ssim", "maximize": true}],
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
paths. It also checks that all declared metric values are returned and that
their identities, settings and optimization directions match the frozen contract.

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
the split-independent API. New `prediction_spec()` declarations use version 3:

```python
def score(prediction_path, truth_path, expected_path):
    # One implementation for both validation and private test.
    return {"metrics": {"custom_metric": value}}
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

## Segmentation overlap, centerlines, and trees

These are generic registered metrics in the existing `segmentation` family.
The names describe the quantity being measured; dataset names and anatomical
class numbers are not part of metric identity. `accuracy`/`acc` continues to
mean classification accuracy; `branch_accuracy` means accuracy after voting
semantic labels on supplied target branches.

| Metric | Direction | Additional per-case references |
|---|---|---|
| `binary_dice` (existing `dice` alias) | maximize | none |
| `multiclass_dice` | maximize | none |
| `cldice`, `multiclass_cldice` | maximize | `skeleton` |
| `tree_length_detected` | maximize | `skeleton` |
| `branch_detected` | maximize | `skeleton`, `parsing` |
| `betti_error` | minimize | none |
| `branch_accuracy`, `branch_f1` | maximize | `parsing` |
| `subtree_consistency`, `topological_distance` | maximize / minimize | `parsing`, `graph` |

Canonical samples use same-shape 2D/3D label arrays, with zero as background.
Labels must be finite, nonnegative exact integers (integer-valued floats are
accepted without truncation). There is no implicit upper class ID. `parsing`
contains contiguous branch IDs 1..N; `graph` is an integer `(2,E)` array of
zero-based parent-to-child node indices (`node = branch ID - 1`). It must be a
rooted tree; a single node uses `(2,0)` edges. `skeleton` is a supplied target
centerline contained in the original target mask. The kernels never derive
target references from predictions.

```python
from biomaker.utils.trusted_metrics import score_samples

spec = {
    "metrics": [
        {"name": "multiclass_dice", "maximize": True},
        {"name": "branch_accuracy", "maximize": True},
        {"name": "topological_distance", "maximize": False},
    ],
    "task_family": "segmentation",
    "dice_smooth": 0.0,
    "topology_component_policy": "raw",
    "missing_label_penalty": 50.0,
}
# Each yielded case: prediction, target, parsing, graph arrays.
result = score_samples(cases, spec)
# result["metrics"]: raw equal-case means; result["per_case"]: raw case scores.
```

The capability catalog documents the frozen defaults and supported values:

- `dice_smooth=0`: added to numerator and denominator of binary/multiclass
  Dice. Existing unsmoothed binary Dice remains unchanged; both empty scores 1.
- `class_aggregation="gt_present"`: multiclass means include nonzero classes
  present in the target. Multiclass clDice further requires target skeleton
  support. An empty included class set scores 0. clDice with no centerline
  overlap scores 0 and has no additive smoothing.
- `topology_component_policy="largest"` (`"raw"` also supported) applies only
  to tree coverage and branch-label projection. Dice and clDice use raw
  predictions. Betti uses raw component counts.
- `connectivity=3`, capped by array dimension: 26-neighbor connectivity in 3D
  and 8-neighbor connectivity in 2D. `betti_reference="target"` compares
  connected-component counts (Betti-0); `"one"` explicitly compares to one.
  This metric does not count tunnels or cavities.
- `topology_smooth=0`: additive numerator/denominator smoothing for skeleton
  voxel coverage. Empty target skeleton scores 0 with zero smoothing.
  Tree length here counts skeleton voxels, not physical millimeters.
- `branch_coverage_threshold=0.8`: a branch is detected at or above this
  fraction of its target skeleton voxels. Branches follow supplied parsing.
- Branch voting includes background; ties select the smallest semantic label.
  Accuracy/F1 use branches with nonzero target votes; F1 is macro over target
  branch classes. `subtree_ignore_labels=[]` excludes no anatomical class.
  Consistency checks nontrivial descendant subtrees; background votes count,
  and no eligible subtree scores 0. Thus consistency alone is not accuracy.
- `missing_label_penalty=50`: topological distance uses the nearest matching
  target label's undirected graph-hop distance; missing labels receive this
  penalty. Finite distances are not clipped. No valid target branches scores 0.
- `label_ids=null`: no range restriction. A dataset can declare a unique list
  of allowed labels. `reference_protocol` is optional provenance metadata.

The mask reader supports a `target_column` and an `auxiliary_columns` mapping
from `skeleton`/`parsing`/`graph` to private CSV columns. `answer_columns` must
contain exactly these columns. References must already exist; scoring only
decodes them. `geometry_policy="physical"` requires medical files preserving
input size, spacing, origin, and direction (numeric tolerance rtol 1e-5,
atol 1e-6). The default `"shape_spacing"` preserves the previous reader contract.
Medical input probes read headers instead of decompressing the CT. Prediction-only
validation never accesses private target or auxiliary files.

The executable registry and aggregation live in `biomaker/utils/trusted_metrics.py`;
the extra pure kernels live in `biomaker/utils/segmentation_metrics.py`.
`metric_source.py` bundles both into frozen evaluators, with source and dependency
versions included in cache identity. Agents select these metrics and decode
artifacts; they do not reimplement formulas. The adapted upstream formula
source, commit and MIT license are recorded in the kernel module; dataset
protocols explicitly select their parameters.

## Reusable target tree preparation

`biomaker.utils.airway_reference.prepare_tree_reference(label_path, output_dir,
include_graph=True)` is a separate, deterministic preparation helper for
airway-shaped 3D labels. It reuses the pinned ATM22 tree-parsing algorithm,
whose unmodified source and license are under `utils/_vendor/atm22/`. It has no
competition task ID or anatomical class limit. Work is performed on a binary
copy; the original semantic label and all physical geometry are preserved.

Run preparation once outside split generation and scoring. Returned paths
point to cached skeleton/parsing volumes, an optional graph, and provenance.
Cache identity includes input, builder, vendor source, geometry and dependency
versions; output hashes are verified on reuse. Derived structures that do not
fit the original target are rejected instead of silently changing the label.
These are local derived references, not organizer-provided hidden annotations.

The helper requires scikit-image 0.22.0 and NumPy below 2. The original parsing
code relies on `skeletonize_3d` returning uint8 values 0/1; version 0.24 returns
bool and changes neighborhood convolution results. This behavior is checked
with a branching synthetic volume. Do not substitute a skeletonizer/version
without defining and validating a new reference protocol.

## Explicit validation-only protocols

A prepared split manifest can explicitly declare `required_splits` or
`public_splits` containing the roles `train` and `score_val` and excluding
`test`. These roles refer to the existing train/validation CSVs, not extra
datasets. The trusted adapter then freezes
`test_split=null`: its self-test creates only validation fixtures and
`validate_submission()` reports a skipped `no_test_split` result. It never
creates a fake test split. Undeclared or contradictory missing-test layouts
remain errors. Metric calculations, validation coverage and subsequent
full-data training semantics do not change.
