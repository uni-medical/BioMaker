# BioMaker Desktop 0.1.2 acceptance

Date: 2026-10-08. Host: macOS, Apple silicon / arm64.
Packaged source: `07a8d065269d8af397c56d7beec2fb0a6332df4b`, clean checkout.

## Verified behavior

- 102 WebUI, 28 desktop service/native-runtime, 28 native/package/upgrade/notification,
  and 3 model-configuration tests passed. TypeScript and production builds passed.
- UI-only upgrades reuse an existing environment only after matching the complete
  engine source inventory, platform, and exact Python version/build. Tests cover
  modified source, stale locks, injected files, partial copies, wrong interpreters,
  and preserving a process using the old engine. Read-only migration selected the
  installed 0.1.1 engine for an unchanged bundle. This release includes an engine
  fix, so its actual installation correctly created a separate environment.
- A real workflow exposed an OmegaConf startup failure when assigning populated
  compute-slot metadata. The schema annotation fix is covered by CPU/MPS/CUDA
  metadata assignment tests; the installed engine now passes that startup path.
- The 0.1.2 DMG was mounted read-only, its app version and Applications shortcut
  checked, and the app installed to `/Applications/BioMaker.app`. The previous
  app is retained under `release/installed-backups/`. Deep strict code-signature
  verification passed; the signature is ad-hoc, not Developer ID or notarized.
- The installed application completed its real environment setup with a system-only
  PATH. LightGBM's local native provisioning and synthetic CPU predictions, Torch,
  MONAI and the installed entrypoint checks passed. Real provider settings remain
  unconfigured and QA records are isolated from the user's workspace.
- The installed executable passed interaction acceptance: provider guidance,
  preserved drafts, secrets masking, themes/languages/native menus, history and
  candidate views, nonfinite values, native export, compact layout and restart.
- Notification settings persisted across restart. The real main-process monitor
  observed a synthetic ledger transition after the window closed. A native click
  recreated the window, opened the target run, acknowledged the pending target,
  and preserved the original draft. Mute and history replay suppression passed.
  Per-run response ordering prevents an unrelated response from losing a completion.

## Actual installed-engine workflow

`engine-workflow-smoke.py` ran against the installed 0.1.2 snapshot, with no source
checkout override or production engine patches. It completed one candidate in
5.825 seconds and made five requests to a deterministic loopback model fixture.
Eight Python processes had network guards; no external provider requests occurred.

The actual engine built the explicit synthetic splits, separated validation/test
labels, compiled and checked its frozen evaluator, generated the candidate through
its normal API adapter, fitted a CPU logistic regression on 12 training rows,
evaluated the predictions, and exported journal, best solution, prediction files,
results, candidate tree and report. The fixture's authored report is not evidence
of performance; the harness checks the real evaluator and exported artifacts.
Final retraining was disabled. A subsequent engine selection reused the same
installed environment and verified that its source remained unchanged.

## Evidence and artifacts

Local receipts are under `apps/desktop/output/`:

- `model-config-pytest.json`
- `package-installation-0.1.2.json`
- `installed-engine-acceptance-0.1.2.json`
- `engine-workflow-acceptance-0.1.2.json`
- `engine-reuse-after-workflow.json`
- `playwright/electron-acceptance.json`

| Artifact | SHA-256 |
| --- | --- |
| BioMaker-0.1.2-mac-arm64.dmg | `b428a372b9ed8edabef4b42b46025576e89cc603194eeae64e485860e6039afb` |
| BioMaker-0.1.2-mac-arm64.zip | `a52d072996324edbf1a77dc3b8e20a606cca9fb4be26ab3a169133431e12b55e` |

## Boundaries

The model responses and dataset are controlled software fixtures. This proves a
bounded operational workflow, not provider intelligence, general biomedical
performance, multi-round search quality or final retraining. The full engine run
uses its normal entrypoint; desktop queue lifecycle has separate acceptance.

Notification dispatch/click integration was verified with native `show()` intercepted
in QA to avoid test banners. Actual banner delivery remains subject to OS permission.
Apple notarization, signed updates, independent clean-hardware acceptance, and
Windows/Linux/Intel Mac installed-engine acceptance remain pending. Generated code
runs with the user's access, not in an OS security sandbox.
