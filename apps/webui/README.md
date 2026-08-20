# BioMaker WebUI

A local sidecar interface for BioMaker. It manages projects and queued runs while
leaving the `biomaker` package unchanged.

## Start

From this directory:

```bash
uv run python -m webui.server --host 127.0.0.1 --port 7860
```

Then open <http://127.0.0.1:7860>.

Runtime state is stored under `state/` and is ignored by Git. BioMaker outputs
continue to use the workspace-level `logs/` and `workspaces/` directories.

## Test

```bash
uv run pytest
```
