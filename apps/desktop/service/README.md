# BioMaker desktop sidecar

This adapter reuses `apps/webui/webui` without introducing another run database
or execution path. It is launched by Electron's main process, never directly by
renderer code.

## Process contract

Run the bundled Python with `service/main.py`. The process binds a preallocated
`127.0.0.1:0` socket and writes one readiness message to standard output after
FastAPI startup succeeds:

```json
{"type":"ready","port":54321}
```

Required environment:

- `BIOMAKER_DESKTOP_TOKEN`: a fresh random session token, at least 32 characters.
- `BIOMAKER_DESKTOP_HOME`: absolute writable application data directory.
- `BIOMAKER_DESKTOP_ENGINE_ROOT`: absolute writable engine snapshot directory.
- `BIOMAKER_WEBUI_UV`: bundled uv executable.

Optional environment:

- `BIOMAKER_DESKTOP_PARENT_PID`: Electron PID. A watchdog stops the HTTP service
  when Electron disappears, including after a hard kill.
- `BIOMAKER_DESKTOP_PYTHON`: Python used to install the engine; defaults to the
  bundled interpreter running the sidecar.
- `BIOMAKER_DESKTOP_ALLOWED_ORIGINS`: exact comma-separated origins. Empty by
  default because the renderer uses Electron IPC, without a browser Origin.

Every HTTP endpoint, including artifacts and static resources, requires
`Authorization: Bearer <session token>`. Host must be loopback and arbitrary
browser origins are rejected. The token is removed from the environment before
any detached experiment starts.

Workspace outputs live under `HOME/workspace/{logs,workspaces}`, the durable
SQLite ledger under `HOME/state`, and settings under `HOME/config.env`. Newly
created settings use mode `0600` through the existing WebUI settings writer.

## Additional API

- `GET /api/desktop/status`: version, installation state, paths, configuration
  completeness, and current platform capabilities.
- `POST /api/desktop/engine/install`: start installation asynchronously. The
  response is the current engine state. Repeated requests are idempotent.
- `POST /api/desktop/engine/cancel`: request installation cancellation.

The engine state is `{ready, installing, error, log}`. The log is a redacted,
bounded string. Installation runs `uv sync --frozen` in the writable snapshot,
then verifies imports and the installed BioMaker CLI entrypoint. Success is
recorded against dependency-file content, OS, and architecture. Cancellation
terminates either the download/install process or the verification process.

Run creation and requeue require a ready engine, an API key, and an explicitly
selected or saved model. Existing run/queue/settings/artifact APIs retain their
WebUI schemas. Closing the desktop stops installation and the HTTP service;
the existing durable supervisor preserves running experiments. Local engine
execution currently supports macOS and Linux; Windows needs additional process
management work before it can be enabled.

## Dependency updates and validation

`requirements.in` describes only the small HTTP sidecar. `requirements.txt`
locks every transitive dependency with hashes for supported Python 3.12 hosts.
The engine's scientific dependencies remain in its own virtual environment.

```bash
uv pip compile apps/desktop/service/requirements.in --upgrade --universal \
  --generate-hashes --python-version 3.12 \
  --output-file apps/desktop/service/requirements.txt
uv run --no-project --python 3.12 \
  --with-requirements apps/desktop/service/requirements.txt \
  --with pytest --with httpx python -m pytest apps/desktop/service/tests -q
uv run --no-project --with pip-audit pip-audit \
  --requirement apps/desktop/service/requirements.txt --disable-pip --no-deps
```

The tests exercise HTTP authorization, origin rejection, preflight, persistent
queue actions, real subprocess failure/retry/cancellation, redaction, service
shutdown, and an actual ephemeral HTTP listener. They do not substitute for
installing and running the scientific engine on a release machine.
