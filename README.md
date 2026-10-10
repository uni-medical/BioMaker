<p align="center">
  <img src="assets/BioMaker_logo.png" alt="BioMaker" width="720">
</p>

# BioMaker

BioMaker develops biomedical image analysis models from a task description and
local data. It prepares and verifies data splits, freezes an evaluation contract,
generates and refines candidate programs, and delivers the model selected by
validation performance. CPU, Apple MPS and CUDA are supported.

## Install and configure

```bash
git clone https://github.com/uni-medical/BioMaker.git
cd BioMaker
uv sync --python 3.11
```

Create a local `.env` file with your OpenAI-compatible provider credentials:

```dotenv
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_API_KEY=your-key
```

`BIOMAKER_ENV_FILE` can select another credential file. Existing shell environment
variables take precedence for direct CLI runs. Keep credentials out of Git.
The default model is `deepseek-v4-flash`; override it per task with
`model_for_all_agents=<model>`. Optional `SERPER_API_KEY` and `JINA_API_KEY` enable
web search and page reading.

## Run a task

```bash
uv run biomaker data_dir=/absolute/path/to/data \
  desc_file=/absolute/path/to/task.md \
  log_dir=/absolute/path/to/logs workspace_dir=/absolute/path/to/workspaces \
  model_for_all_agents=your-model
```

A direct run stays in the foreground. `biomaker --queue` submits the same task to
a durable FIFO and follows its console log. Closing the client does not terminate
a queued experiment. Failed runs pause the queue until resumed. CLI, WebUI and
Desktop use the shared application service and queue runtime.

`exec.parallel_workers=auto` assigns one candidate slot per visible CUDA GPU,
or one on MPS/CPU. Set `CUDA_VISIBLE_DEVICES` before submission to select GPUs;
an empty mask disables CUDA. Each queued run retains its selection. Independent
processes and other users remain responsible for sharing the same devices.
Candidate execution and trusted CPU scoring have separate bounded concurrency.
A slot is released only after process cleanup is confirmed; uncertain slots are
quarantined.

### WebUI and Desktop

```bash
cd apps/webui
uv run python -m webui.server --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860> to configure the provider, submit tasks and inspect
candidates, logs and results. See [WebUI documentation](apps/webui/README.md).
The Electron application adds local dataset selection, runtime installation and
native result browsing; see [Desktop documentation](apps/desktop/README.md).

## Evaluation and model delivery

1. The task defines sample identities, input/answer encodings, split roles and
   named metrics. Preparation verifies these declarations and freezes the scorer.
2. Candidates train using visible labels and generate the declared prediction
   artifacts. Supported metrics use shared trusted implementations across
   validation, private evaluation and explicit rescoring.
3. Only complete, finite validation scores participate in selection. Invalid
   predictions are candidate errors; scoring infrastructure failures preserve the
   candidate as `unscored` and stop new work without turning the fault into model
   feedback.
4. The selected candidate can be retrained with the permitted additional labels.
   The private test is evaluated once after selection and successful retraining
   when enabled. Its answers and scores never feed search or selection. Tasks
   without a test split require no test predictions.
5. Model artifacts, a manifest and reload checks are required by default. A
   prediction CSV alone does not establish that a model can be delivered.

`final_retrain.enabled=false` keeps the selected candidate's original model.
`exec.timeout`, `exec.evaluator_timeout`, `exec.evaluator_workers` and
`agent.total_time_limit` control execution budgets. Provider transport defaults
to at most two total attempts, with explicit request and elapsed-time limits.
Authentication and quota failures are not repeated candidate repair loops.

The [evaluation contract](docs/evaluation_contract.md) documents supported
metrics, prediction formats, scoring outcomes and recovery. Existing frozen
runtimes and historical results retain their original protocol.

### Prepared split packages

By default an agent prepares splits. A task can instead provide an external,
verified package using `split_builder.source=verified_package` and
`split_builder.package_dir=/absolute/path/to/package`. Use
`biomaker.utils.verified_split_package.seal_split_package` to validate and record
the package against its source data and exact task text before distribution.

The package contains `biomaker_splits/`, `private_gt/`, `split_manifest.json`,
optional derived `public/` files, and `package_integrity.json`. File columns must
be explicitly declared and use logical `input/...` or `public/...` paths; inline
answers declare an empty file-column list. Admission checks the complete source
and package inventories, SHA-256 hashes and task identity, then reruns the current
split verifier. An integrity inventory is not a cryptographic signature. Keep
packages separate from source data and run outputs.

### Optional MONAI retrieval

```bash
bash scripts/create_rag_db.sh
uv run python -m biomaker.monai_rag.query_rag_db
```

Keyword retrieval is local and requires no embedding endpoint. Opt into embedding
retrieval with `BIOMAKER_RAG_MODE=embedding`; build and query with the same mode.
`BIOMAKER_EMBEDDING_BASE_URL` and `BIOMAKER_EMBEDDING_API_KEY` can select a separate
embedding provider. Missing indexes disable retrieval with a warning.

## Repository boundary

- `biomaker/`: core engine, shared application service, durable runtime and CLI.
- `apps/webui/`, `apps/desktop/`: product interfaces and distribution tooling.
- `tests/`, `docs/`, `scripts/`: generic tests, maintained documentation and tools.
- Dependency locks, package metadata and third-party notices are versioned.

Datasets, experiment recipes, cross-method benchmarks, competition implementations,
weights, run outputs, audit reports and release acceptance receipts belong in
separate workspaces. Machines synchronize this source tree while keeping their
own credentials, data, environments and historical runtime snapshots.

## Acknowledgements

Thanks to medical workers and dataset owners, and to open-source projects
including [MONAI](https://github.com/Project-MONAI) and
[aideml](https://github.com/WecoAI/aideml).
