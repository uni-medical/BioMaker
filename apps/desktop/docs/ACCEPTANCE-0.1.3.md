# BioMaker Desktop 0.1.3 acceptance

Date: 2026-10-08. Host: macOS, Apple silicon / arm64.
Packaged source: `92418292b65cb7f761848c9f8f0116caffeec5ab`, clean checkout.

## Verified changes

- Native close and quit protect unsaved model settings. Canceling either keeps
  the original form and the real HTTP service available. Explicit discard closes
  the window; reopening restores only the saved configuration. Unsaved keys are
  absent from the configuration file and renderer local storage.
- Configuration fields remain disabled during a save. Closing or quitting during
  the request offers only a wait action. An injected failed response preserves
  edits and close protection; a subsequent real save succeeds and removes the
  prompt. The packaged executable passed all six lifecycle interaction checks
  with a system-only PATH and zero renderer errors.
- The native confirmation decides the current Electron `will-prevent-unload`
  event synchronously. This avoids a measured Electron behavior in which an
  asynchronous retry emits another close event but does not repeat the unload
  event. The independent Python service continues while the native sheet is open;
  main-process polling resumes after the sheet returns.
- Packaging rejects a runtime manifest whose application version is missing or
  differs from the package version, with an actionable preparation command.
- The new runtime has the same engine checksum, platform, Python version and
  standalone build as 0.1.2. Selection reuses the existing engine in place. The
  actual upgrade was installed to `/Applications/BioMaker.app` with the previous
  app backed up. On first launch with a system-only PATH, its real user profile
  was immediately ready without requesting installation. The engine path stayed
  unchanged; provider settings remain unconfigured and the user workspace has
  no seeded test records.

## Validation

102 WebUI tests and 28 desktop service/native-runtime tests passed. The native,
package, notification, engine-upgrade and close-guard suite passed 34 tests; its
separately enabled real-sidecar test also passed, for 35 unique native checks.
TypeScript and the production build passed.

The packaged executable passed the existing Electron acceptance covering setup,
drafts, masked credentials, languages/native menus, themes, historical candidate
views, nonfinite values, file export, compact layout, restart and background
notification dispatch/click restoration. Notification display was intercepted in
QA; actual macOS banner delivery is not established by this check.

The DMG was mounted read-only. Its version, Applications shortcut and application
archive matched the tested package. Deep strict ad-hoc signature verification
passed. The scientific engine source checksum remains
`56d274eec3df1d02eefb52dbe5cc894ff4a5eabe066f483046462e4e021e9fa2`.

## Installed desktop workflow

The installed 0.1.3 executable completed the combined UI → durable queue → installed
engine → result UI workflow in 18.369 seconds. The UI configured a local fixture
provider, selected the synthetic dataset, submitted one candidate with final
retraining disabled, then displayed the real result, code and execution log and
exported the generated report. The run reached `succeeded` with exit code 0 and
zero renderer errors. Its five model requests went to the localhost fixture.

The real engine built the synthetic splits, protected validation/test labels,
fitted a scikit-learn CPU classifier, ran its frozen evaluator and exported the
normal artifacts. No checkout source override or production engine patch was
used. Installed-file fingerprints remained unchanged and the exact QA profile's
processes were confirmed stopped after the run.

The test harness restricts the Python sidecar and its queue, uv and training
descendants to localhost with a temporary macOS sandbox. Only `/bin/ps` is exempt
for read-only process identity inspection; Electron retains its normal Chromium
sandbox. The training candidate independently verifies that an external-address
connection is denied with `EPERM`. These are acceptance-harness controls, not
additional sandboxing supplied by the product.

## Artifacts and evidence

| Artifact | SHA-256 |
| --- | --- |
| BioMaker-0.1.3-mac-arm64.dmg | `38588378fbc71005b72fa0f9f2187c168794de39be5a9f466d1d95a11fd3b6a4` |
| BioMaker-0.1.3-mac-arm64.zip | `c2237b81c08fb02c1535b5e2d9f6b878f53e2d228359a81e9a60b9d7ce3f96cc` |

Local machine-readable evidence is under `apps/desktop/output/`:

- `engine-upgrade-0.1.3.json`
- `installed-engine-acceptance-0.1.3.json`
- `native-close-packaged-0.1.3/acceptance.json`
- `package-installation-0.1.3.json`
- `playwright/electron-acceptance.json`
- `desktop-workflow-acceptance-0.1.3.json`
- `qa/desktop-workflow-0.1.3/` (durable fixture, logs, screenshots and outputs)

## Boundaries

The combined desktop workflow uses synthetic data and deterministic model
responses. It establishes software execution, not biomedical model performance,
external-provider quality, multi-round search or final-retraining acceptance.

Apple Developer ID signing/notarization, signed updates, independent clean-hardware
acceptance and other-platform installation remain pending. The production engine
executes generated code with the user's access; test-only network restrictions
must not be described as a production security sandbox.
