# BioMaker Desktop

A native desktop workspace for BioMaker's biomedical model development engine.
The first supported target is **macOS on Apple silicon**. The desktop shell and
UI are portable; Linux and Windows release targets still require platform acceptance.

中文安装与使用说明见 [快速开始](docs/QUICKSTART.zh-CN.md)。

## Use the macOS app

1. Open the BioMaker DMG and drag **BioMaker** into **Applications**.
2. Open BioMaker. Settings guides you through installing the model development
   environment and configuring an OpenAI-compatible model provider.
3. Choose a local dataset folder, describe your task and validation criteria,
   choose a model, review the compute budget, and start the experiment.
4. Follow activity and logs, then use **Results** to inspect the recorded selection,
   open its candidate, and preview or save reports, solution code and metrics.

The application includes its own Python and package manager. It does not require
Homebrew, Node.js, Terminal, or a system Python installation. The first engine
installation needs internet access and several GB of disk space for the ML packages;
the window remains usable while dependencies install. Provider requests may incur
charges at your configured provider. Experiments generate and execute Python code
with your user account's file access; use trusted task instructions and datasets.

This development build is ad-hoc signed. Public distribution without macOS security
prompts requires an Apple Developer ID certificate and notarization. These credentials
are deliberately not included in the repository. The app does not claim that its
model results have received clinical validation.

## Architecture

| Layer | Technology | Responsibility |
| --- | --- | --- |
| Desktop | Electron / TypeScript | Window lifecycle, native menus, file dialogs, protected IPC, sidecar management |
| UI | React / TypeScript / Vite | Experiments, activity, files, resources, provider configuration, installation UI |
| Service | Python / FastAPI | Authenticated local API, environment setup, existing BioMaker queue and artifacts |
| Engine | Existing BioMaker Python package | Autonomous model development, execution, evaluation, retraining and reports |
| Distribution | electron-builder, bundled CPython + uv | Drag-to-install macOS package, independent of developer tooling |

The renderer has no Node.js integration. Its preload exposes a small typed API;
the Electron main process attaches a random per-launch Bearer token. The service
listens on an ephemeral loopback port and rejects requests without that token.
Secrets stay in the private application data directory and are masked in settings.
Files open through explicit native dialogs or restricted application data paths.

The original CLI and WebUI are not replaced. The service reuses their real task
schema, persistent queue, run ledger, process identity checks, artifact validation,
and change revisions. A run is an experiment execution; a candidate inside a run
is not displayed as an unrelated chat.

Closing the window does not stop an active experiment. Queue supervisors retain
their existing persistent lifecycle, and run history is available on reopening.
Use the experiment's Stop action to stop training. macOS sleep still suspends local
compute: a desktop application cannot make a sleeping Mac continue training.

Completion/failure notifications continue while the macOS app runs in the
background, including with its window closed. Their General setting is separate
from OS notification permission. Clicking a notification opens its experiment.
UI-only upgrades reuse a source- and Python-compatible engine without moving its
virtual environment; engine changes are isolated and preserve older environments.

## Results and files

Completed experiments open on Results; running experiments initially open on
Activity. The overview distinguishes an explicit recorded selection from a unique
match to the saved solution code. It shows the selected candidate's recorded
metric and direction, with a link to that candidate. Missing or conflicting
evidence stays explicit. Generated report prose is a summary to read alongside
the original records, and imported history represents only the copied files.

Final retraining has its own status. A receipt, a completion event without a
receipt, an explicit disabled setting and unavailable records have different
meanings. Recorded candidate metrics and successful retraining do not establish
independent test performance or clinical validity.

Text previews are limited to 500,000 bytes and preserve the original UTF-8 content.
Saving from a preview verifies the downloaded bytes against that preview's
SHA-256 and checks that the local service has not changed. If the file or service
changes, refresh the preview before saving; a rejected export preserves an existing
destination. Save from the file list remains available for direct exports and
larger files. Both actions use the native save dialog.

Only one artifact save can run at a time across the application. Closing the
preview does not cancel an approved save. Wait for that save to finish before
starting another; overlapping requests are refused without opening another
dialog or queuing a later overwrite.

## Development

Use Node.js 22.12+ (or 24+) and a host `uv` for the one-time runtime build. These are
build tools only; they are not requirements for people installing the app.

```sh
cd apps/desktop
npm ci
npm run icons
npm run prepare:runtime
npm run build
npm start
```

`npm run dev` starts the frontend development server and Electron. `npm run
package:mac` creates the arm64 DMG and ZIP under `release/`. Runtime preparation
copies an allowlisted engine snapshot and never includes `.env`, datasets, logs,
credentials, virtual environments, or workspaces.

The desktop engine uses its own [audited dependency overlay](engine/README.md).
It does not modify the original research lock file. Portable Python, uv and the
service are copied atomically to a stable versioned `runtimes/` directory under
Application Support, so moving or replacing the `.app` does not invalidate an
installed environment. Older runtimes remain available to background experiments.

## Validation

```sh
npm run typecheck
npm run test
npm run test:e2e
cd ../webui
uv run --group dev pytest
```

Version 0.1.11 has passed 701 source checks, 32 development checks, 152 packaged
checks, four offline engine checks, package verification and an actual upgrade.
The retained-profile check, installed synthetic CPU training workflow, complete
bundle verification and normal native launch have also passed on this Apple silicon Mac.
See the [0.1.11 acceptance record](docs/ACCEPTANCE-0.1.11.md) for evidence and
[docs/RELEASE.md](docs/RELEASE.md) for release checks, platform boundaries,
and iteration priorities. Keep scientific model evaluation separate from desktop
installation and interaction checks.
