# BioMaker WebUI

A local sidecar interface for BioMaker. It manages projects and queued runs while
leaving the `biomaker` package unchanged.

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
BioMaker repository, to override it explicitly. `BIOMAKER_WEBUI_ENV_FILE`
remains supported for compatibility with older WebUI launches.

Runtime state is stored under `state/` and is ignored by Git. BioMaker outputs
continue to use the workspace-level `logs/` and `workspaces/` directories.

## Test

```bash
uv run pytest
```
