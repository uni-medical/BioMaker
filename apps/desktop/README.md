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

## Experiment drafts

The 0.1.12 source workflow adds a **Drafts** library. New Experiment creates an
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

The 0.1.13 upgrade preserves this draft library and passed the installed workflow
on this Apple silicon Mac; see the versioned acceptance record below.

## Activity and console

Version 0.1.13 adds a log reader with explicit following and pause controls,
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

Version 0.1.14 has passed 857 source checks, 102 development checks, 195 packaged
checks and four offline engine checks. Package verification, the actual upgrade,
independent engine setup, retained drafts/configuration/run records, installed
synthetic CPU training and normal native launch also passed on this Apple silicon Mac.
See the [0.1.14 acceptance record](docs/ACCEPTANCE-0.1.14.md) and
[release policy](docs/RELEASE.md) for scopes, receipts and remaining platform and
signing boundaries. Software acceptance does not establish scientific or clinical
model validity.

Version 0.1.14 reduces repeated experiment reads with indexed run lookup and a
bounded, service-owned observation cache. Fresh ledger records still determine
lifecycle and actions. File changes invalidate summaries and refresh the UI,
including same-size/same-mtime replacements. In the isolated 100-run benchmark,
warm list reads fell from 257 ms to 6.7 ms and warm detail reads from 259 ms to
0.25 ms. These are instrumented endpoint timings; first reads and changed
histories can still scan full files.
