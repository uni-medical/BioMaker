# BioMaker Desktop

A native desktop workspace for BioMaker's biomedical model development engine.
The first supported target is **macOS on Apple silicon**. The desktop shell and
UI are portable; Linux and Windows release targets still require platform acceptance.

中文安装与使用说明见 [快速开始](docs/QUICKSTART.zh-CN.md)。

## Use the macOS app

1. [Download the BioMaker 0.2.0 test build from Google Drive](https://drive.google.com/drive/folders/1W-5JySmOb21Tp4SwtEOetSOQsicKQ1zH)
   (Apple silicon). The folder includes installation instructions and SHA-256
   checksums. Open the DMG and drag **BioMaker** into **Applications**.
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

CLI, WebUI and Desktop share `biomaker.application.RunService` and the
`biomaker.runtime` queue, ledger, process identity checks and artifact readers.
Desktop injects runtime, provider and dataset admission checks at this shared
application boundary. A run is an experiment execution; a candidate inside a run
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

## Experiment drafts

The **Drafts** library stores independent experiment drafts. New Experiment creates an
independent draft with default options; it reuses an already empty, unnamed draft
to avoid clutter. Drafts save automatically and can be searched, reopened,
renamed, duplicated or deleted. Renaming a draft does not change its experiment
output directory name. Deleting a draft does not delete datasets or existing runs.

Reusing an experiment creates a separate draft and preserves the others. Review
its dataset, model and budget before submitting; imported history requires a
local dataset selection. Confirmed submission removes only the submitted draft.
Failed or uncertain submissions retain it for review before retrying.

Drafts persist in the application's native data directory across restarts and
remain editable when the local Python service is unavailable. Existing valid
single-draft data migrates once, with its original content retained for recovery.
Save failures remain visible with retry and recovery actions; closing the app
attempts to save pending edits and protects them if that save fails.

## Activity and console

The log reader provides with explicit following and pause controls,
search and match navigation within the displayed window, earlier windows, line
wrapping, current-window copy and native full-log export. Reading or selecting
older output never stops the experiment. New output waits behind a paused window
until you choose Follow latest. Missing files, read errors and source changes are
reported separately while retained text remains available.

Activity labels match the actual engine. Failed execution/validation and buggy
candidate evaluation are shown as failures. Unknown historical fields remain
inspectable without breaking the workspace. The latest recorded event is evidence
of activity, not an inferred current phase for every parallel candidate.

## Results and files

Completed experiments open on Results; running experiments initially open on
Activity. The overview distinguishes an explicit recorded selection from a unique
match to the saved solution code. It shows the selected candidate's recorded
metric and direction, with a link to that candidate. Missing or conflicting
evidence stays explicit. New reports render the saved journal directly, including
failed attempts, nested validation metrics, recorded execution output and the
source of resource measurements. Markdown tables render in the preview. Imported
history represents only the copied files.

Final retraining has its own status. A receipt, a completion event without a
receipt, an explicit disabled setting and unavailable records have different
meanings. Recorded candidate metrics and successful retraining do not establish
independent test performance or clinical validity.

New runs reject malformed prediction artifacts and undefined or fallback primary
metrics. Private test scoring happens once after validation-based selection and
any successful final retraining. Trained candidates must save a model manifest,
checkpoints and portable inference assets; an independent relocated probe checks
that the saved bundle reloads and reproduces its small example predictions.
See the [evaluation protocol](../../docs/evaluation_contract.md)
and [model artifact contract](docs/MODEL-ARTIFACT-CONTRACT.md) for coverage and
limits. Older experiment files retain their original evidence and protocol.

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
Its generated requirements and lock must match the unified root manifests
byte for byte; the package gate rejects drift. Portable Python, uv and the
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

Validation results apply to the tested source and package only. Follow the
[release policy](docs/RELEASE.md) to verify a new distribution.
