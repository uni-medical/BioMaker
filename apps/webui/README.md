# BioMaker WebUI

A local launcher and viewer that runs the same `biomaker` CLI and reads its
standard outputs.

## Start

From this directory:

```bash
uv run python -m webui.server --host 127.0.0.1 --port 7860
```

Then open <http://127.0.0.1:7860>.

On WSL, the dataset browse button opens the Windows folder picker and converts
the selected Windows path to a WSL path such as `/mnt/c/Users/name/data`.

The WebUI and BioMaker use the BioMaker repository's `.env` as their single
default configuration file, regardless of the directory used to start them.
Set `BIOMAKER_ENV_FILE` to an absolute path, or to a path relative to the
BioMaker repository, to override it explicitly.

Settings can store a default model for WebUI runs. A task's Advanced model
override takes precedence, and the resolved model is passed to BioMaker as an
explicit `model_for_all_agents` run argument.

The WebUI and source-checkout CLI share one workspace-level durable FIFO in
`.biomaker/runs.sqlite3`. A temporary detached supervisor runs one top-level
experiment at a time, while stdout and stderr append directly to durable files
under `.biomaker/console/`. Closing or restarting the WebUI does not stop a run.
An incomplete, failed, interrupted, stopped, or cancelled run pauses later FIFO
items until **Resume Queue** is explicitly selected.

Experiment artifacts remain in `logs/` and `workspaces/`. Existing log
directories are imported idempotently into the ledger for display without
rewriting their event history. From a complete source checkout,
`uv run biomaker ...` uses this same queue, follows its console log, and returns
the experiment process's exit code.

Run creation, cancellation, stopping, retrying, and settings changes accept
requests only from the local computer, even if the server is bound to another
network interface.

Selecting an existing run loads its name, dataset, task description, and
recorded CLI options into the middle pane as a read-only configuration. The
form becomes editable and blank only after choosing **New**.

## Test

```bash
uv run --group dev pytest
```
