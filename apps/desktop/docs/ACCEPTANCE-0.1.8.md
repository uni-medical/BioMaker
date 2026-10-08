# BioMaker Desktop 0.1.8 acceptance

Date: 2026-10-09. Target: macOS, Apple silicon / arm64.
Release source commit: `7212f921761069f0282c70913bb6838f75f4842d`.

This iteration adds an explicit model-response check to provider settings.
Source validation, development and packaged Electron acceptance, isolated
offline engine repair, artifact verification, installation and the installed
training workflow passed.

## Verified implementation

- Model discovery, saved configuration and a successful short generation request
  have separate UI states. No probe runs automatically on startup, editing,
  saving, discovering models or returning to an experiment.
- A user-triggered check uses the current provider form and one fixed short
  prompt. It does not read a task or dataset, install the engine, save settings,
  enqueue an experiment or retry a paid request automatically.
- Saved credentials are reused only for the same normalized provider URL when
  the user has not chosen to remove the key. Redirects must not receive keys.
- Result bodies contain only bounded status metadata. Provider response text,
  exception details and credentials are not returned to the renderer.
- Edits and service lifecycle changes invalidate older in-flight results;
  unrelated experiment drafts and unsaved settings are preserved.
- A probe has a hard wall-clock limit and concurrency control. Timeout and
  shutdown settle its owned worker before releasing operation ownership.

## Source validation

**407 source tests passed:** 102 WebUI, 226 desktop service/native-runtime and
79 native/package checks, including the real-sidecar integration test. The
service total includes 94 new provider cases. TypeScript checks and the production
build passed. The Vite build reports a nonblocking 518.87 kB minified UI chunk
(162.46 kB gzip); Python tests retain the existing Starlette/httpx warning.

Real localhost fixtures cover successful requests, authentication and upstream
errors, redirect refusal, malformed/oversized responses, base-path preservation,
saved-key reuse boundaries, no configuration/dataset changes, missing training
runtime, concurrent requests and slow-response deadlines. Other cases verify
the worker's independent watchdog, abrupt owner exit, cleanup failures retaining
ownership until exit is confirmed, and continued service cleanup.

The probe sends one fixed `Reply with exactly OK.` message with
`max_completion_tokens=64` and `stream=false`. The response is limited to 64 KiB;
the renderer receives only status, code, elapsed time and a completion timestamp.
The worker has its own 25-second maximum lifetime, in addition to the caller's
deadline, and exits on POSIX parent loss. It uses a direct connection to the
configured base URL without environment proxies or redirect following.

## Development and packaged acceptance

Development Electron acceptance passed **14 of 14 checks** through the real
native bridge and sidecar to a deterministic localhost provider. Twelve short
generation requests and three model-list requests used only synthetic fixture
credentials. There were no forbidden requests, fixture violations or renderer
errors. Edits, including changes reverted to their prior value, invalidated old
responses. A real sidecar restart advanced the service revision from 2 to 4 and
prevented a held older success from being shown.

Tests also covered rapid duplicate clicks, failures and explicit retries,
saved-key removal and changed addresses, unsaved configuration preservation,
exact experiment draft retention, testing before installing the training runtime,
and retaining available models while switching the selected model. Reviewed
screenshots cover light/dark themes, a 960 by 680 window, keyboard focus, result
details and the save controls. The development receipt and images are under
`output/provider-check-0.1.8/`.

All eight packaged suites passed against
`release/mac-arm64/BioMaker.app/Contents/MacOS/BioMaker`, using isolated profiles:

| Packaged acceptance | Passed | Receipt under `output/` |
| --- | ---: | --- |
| Provider response check | 14/14 | `provider-check-packaged-0.1.8/acceptance.json` |
| Runtime health UI | 9/9 | `runtime-health-ui-packaged-0.1.8/acceptance.json` |
| Native close and settings protection | 6/6 | `native-close-packaged-0.1.8/acceptance.json` |
| Experiment organization | 8/8 | `run-organization-packaged-0.1.8/acceptance.json` |
| Experiment reuse | 8/8 | `experiment-reuse-packaged-0.1.8/acceptance.json` |
| History import | 10/10 | `history-import-packaged-0.1.8/acceptance.json` |
| Service recovery and diagnostics | 7/7 | `service-recovery-packaged-0.1.8/acceptance.json` |
| General Electron workflows | 22/22 | `electron-packaged-0.1.8/electron-acceptance.json` |

The packaged provider suite repeated the real native-bridge and sidecar flow:
12 fixed short generation requests and three model-list requests reached only
the localhost fixture with synthetic credentials. It recorded no forbidden
requests, fixture violations or renderer errors, and verified the service
revision change from 2 to 4 after a real isolated sidecar restart. Packaged
screenshots were reviewed for provider settings, runtime states, saved settings,
light/dark themes, compact layout and keyboard access. The runtime UI suite
uses synthetic engine states; it does not itself run engine verification.

## Real offline engine verification and repair

The packaged service passed **4 of 4 checks in 105.041 seconds** on an isolated
copy of the existing engine and dependency cache, with outbound networking
denied. It completed real scientific-package imports and a fixed CPU self-test,
detected a removed MONAI package, refused to report readiness after restarting
with that damaged environment, performed an offline forced reinstall and native
runtime provisioning, and passed a second download-free health check.

The original engine and cache were unchanged: before/after fingerprints matched
for all 34,295 source engine files and 29,507 cache files. This repair exercise
modified only the isolated copies. The result is recorded in
`output/engine-health-packaged-0.1.8.json`; supporting evidence is under
`output/qa/engine-health-0.1.8/`.

## Installation and artifacts

The DMG was mounted read-only and its Applications shortcut resolved correctly.
Deep strict signature verification passed on the unpacked application, mounted
DMG application and extracted ZIP application. Their complete 2,811-file bundle
fingerprints matched. The bundled manifest recorded version 0.1.8 and the clean
source commit above. The verified bundle fingerprint is
`4971b3de300ee6c04879c5d4f9dc36095dcfe4e0e44a73e405227c0b08babdf7`.

| Artifact | SHA-256 |
| --- | --- |
| `BioMaker-0.1.8-mac-arm64.dmg` | `f1c0f43b289a857e1b8835764862a8e5147c9269b730fd84329e9d89f5f403b0` |
| `BioMaker-0.1.8-mac-arm64.zip` | `d4ecc5acadf2d0ee2a57d4b11e2c0d35909f86ac5b2d3dabb8d70bf49e675d09` |

The artifact receipt is `output/package-verification-0.1.8.json`.

The application is installed at `/Applications/BioMaker.app`, and the previous
version is preserved under
`release/installed-backups/BioMaker-0.1.7-20261009-014339.app`.
The installed signature passed strict verification. With a system-only `PATH`,
the actual installed app opened the existing user profile, reported its existing
engine ready and requested no engine installation. Engine selection and the
installed app both confirmed reuse; the engine source SHA-256 remains
`56d274eec3df1d02eefb52dbe5cc894ff4a5eabe066f483046462e4e021e9fa2`.

The actual configuration file and complete run records matched their pre-upgrade
snapshots. The actual profile had no experiments, with the API key and model
still unconfigured; read-only installation acceptance made no provider calls,
configuration edits or fixture imports and observed no renderer errors. This is
a configuration-and-record preservation check, not a claim that every profile
file is unchanged: generated runtime metadata and browser/cache files can change
during ordinary use. Receipts are `output/package-installation-0.1.8.json`,
`output/engine-upgrade-0.1.8.json` and
`output/installed-engine-acceptance-0.1.8.json`.

After final workflow verification, the normal installed app was reopened and
visually confirmed as version 0.1.8, with the local runtime ready, no experiments
in the actual profile and the model still awaiting configuration. Settings showed
the new model-response test control correctly disabled without a key and model.
No values were entered or saved and no provider request was sent. The app was
left open on the model-service settings page.

## Installed training workflow

The installed 0.1.8 executable completed UI submission, durable queue execution,
actual CPU training, frozen evaluation, result inspection and native report
export in **15.65 seconds**. The UI reached `succeeded` with exit code 0 and
cleanup complete. A scikit-learn logistic regression model was fitted on
synthetic data using the reused installed engine, with no installed-source
override. Final retraining was disabled. Five model requests reached only a
deterministic localhost fixture; no external provider calls, renderer errors or
fixture errors were recorded.

A test-only spawn wrapper restricted the Python sidecar and its queue, uv and
training descendants to localhost. The exact `/bin/ps` executable was exempt
for read-only process identity inspection. The candidate confirmed that an
external-address connection was denied with `EPERM` before transmission. The
production source and Chromium sandbox were unchanged.

Installed-file fingerprints remained unchanged during the workflow. Final
verification found no remaining QA processes, confirmed that the actual user
configuration and complete run records were unchanged, and passed strict
signature checks on both the installed and packaged applications. The complete
installed bundle matched the verified 2,811-file package fingerprint above.

Receipts are `output/desktop-workflow-acceptance-0.1.8.json` and
`output/post-workflow-verification-0.1.8.json`; durable synthetic evidence is
under `output/qa/desktop-workflow-0.1.8/`. This verifies the installed software
workflow, without establishing external-provider quality, multi-round biomedical
search or scientific model performance.

## Boundaries

The provider check tests one Chat Completions request with a small token budget.
It does not establish tool-calling support, long-context behavior, complete
experiment success or scientific performance. A provider may reject the request
format or exhaust that budget before producing visible text; those failures do
not prove the model is generally unavailable. Tests use local fixtures, not
real user credentials or external model-provider calls.

Windows, Linux, Intel Mac and independent clean-hardware installation remain
unverified. Apple Developer ID signing, notarization and signed updates require
release credentials and remain pending. The local ad-hoc signature is not
Developer ID distribution readiness.
