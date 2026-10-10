# Saved models and reusable inference

New candidates using common Python training APIs (`fit`, `partial_fit`,
`fit_transform`, `backward`, or native LightGBM/XGBoost training) must save
`working/model_manifest.json`. An evaluator can also explicitly require it with
`model_artifact_required: true`. Existing prediction-only validation callers and
stateless candidates remain compatible. Static detection does not recognize every
possible custom optimizer, native trainer, or dynamically constructed call; the
generation prompt requires the same artifacts for those trained solutions too.

The version 1 manifest contains:

- `framework`, plus `metadata.input_schema`, `output_schema`, `preprocessing`,
  ordered `class_labels` (empty for a nonclassification task), and package versions
  in `dependencies`.
- `files`: relative regular files with `path`, `role` (`checkpoint` or `asset`),
  `sha256`, and checkpoint `format`. Empty files, symlinks, escapes, duplicate
  paths, and mismatched hashes fail validation.
- `inference`: `entrypoint`, `input`, and `expected_output`, all listed among the
  hashed assets. The entrypoint exposes `predict(inputs, artifact_dir)`. The JSON
  input contains 1–8 small public or synthetic examples; the expected numeric
  output comes from the in-memory fitted model, without private labels.

Checkpoint formats are fitted sklearn/joblib objects (including pipelines and
sklearn-compatible LightGBM wrappers), native LightGBM and XGBoost models,
PyTorch state dictionaries, and nonempty numeric NPZ parameter arrays. The loader
rejects malformed serialization, unfitted sklearn objects, empty trees/tensors,
and nonfinite tensor/array state. A PyTorch state dictionary or NPZ alone cannot
prove training occurred; they must also pass the inference comparison.

BioMaker copies declared files into a new temporary location, reloads the
checkpoint, and invokes the saved inference code. It compares finite predictions
with the reference (`rtol=1e-4`, `atol=1e-6`). Missing architecture, preprocessing,
cache weights, or original dataset paths therefore cannot silently satisfy the
ordinary probe. All required backbones, encoders, helper code, and small sample
files must be saved too. Working-best snapshots already copy this complete tree.

The probe uses a separately labelled `Interpreter` execution with the candidate's
environment and containment settings, an independent temporary script, and a
timeout of at most 60 seconds (or the smaller candidate timeout). It preserves the
candidate's stdout, training execution time, and resource record. Probe duration
is recorded separately. Python audit/profile guards reject ordinary network,
subprocess, retraining, and external-data dependencies; these guards are **not an
OS sandbox against adversarial code**.

The resulting receipt records checkpoint/file SHA256 values, probe sample count,
shape, duration, and guard/verification scope. It proves loadability and output
agreement on the small probe only. It does not prove model training, clinical
quality, general inference correctness, or that arbitrary code actually uses
every checkpoint for every prediction.

Generation guidance also separates per-class loss weights from per-sample fit
weights, requires a class-dimension assertion, and uses `drop_last=False` to
preserve required training rows. This is guidance for newly generated code;
unexecuted branches of historical code are not repaired by that prompt. Historical
runs and their missing models stay unchanged. A recovery must be a new recorded
run, never a retroactive claim that the old fitted model was saved.
