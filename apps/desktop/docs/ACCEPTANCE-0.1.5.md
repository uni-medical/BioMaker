# BioMaker Desktop 0.1.5 acceptance

Date: 2026-10-08. Host: macOS, Apple silicon / arm64.
Packaged source: `dc02082bc1d2c261f7a35daf205c344b50b91736`, clean checkout.

Source checks, packaged UI workflows, complete distributable verification, the
installed upgrade and the installed synthetic training workflow passed.

## Verified changes

- Display names and archive state are stored in a separate desktop metadata
  database. They do not rewrite original experiment names, run ledger records,
  log directories or artifacts. Imported read-only histories can be organized
  without acquiring execution or process ownership.
- Metadata updates require the expected revision. The service rejects stale
  writes, and the renderer ignores delayed responses with older revisions.
  A conflicting rename preserves the user's input, loads the latest revision
  and requires an explicit retry. Search retains the original experiment name
  alongside the display name; organization survives restart.
- Archiving hides an eligible experiment from the current list and can be
  reversed. Managed active experiments and records awaiting cleanup cannot be
  newly archived. If a previously archived record becomes active or requires
  cleanup, it remains visible in the current list without silently changing its
  saved metadata or revision. Archive and restore do not delete files or alter
  execution state.
- Reusing an experiment creates an editable draft. It copies task text and valid
  options, preserves explicit Boolean `false`, and clears the experiment name
  and template. Missing or unsupported option values use current defaults with
  a visible notice. Reuse does not submit a run, contact a provider or install
  the engine.
- Replacing an existing draft requires an explicit choice. Cancel preserves its
  exact text, dataset path, name and options; confirmed replacement is saved
  immediately and survives restart. Reusing imported or unmanaged read-only
  history clears the dataset path and requires a new dataset selection before
  submission. Original source and imported artifacts, ledger records and
  read-only control restrictions remain unchanged.

## Source validation

102 WebUI tests, 89 desktop service/native-runtime tests and 42 unique native and
package tests passed: **233 tests total**. TypeScript and the production build
passed.

Development Electron passed run organization, experiment reuse and the service
UI lifecycle checks. Organization exercised Unicode display names, stale
response protection, lost-update rejection, conflict recovery, search, archive,
restore, restart, native export and compact layout. Reuse exercised cancellation,
explicit replacement, restart persistence and imported-history dataset reselection.
The harness confirmed unchanged source and imported file hashes and modification
times, unchanged original ledger records, and no run mutations, provider requests
or engine installations.

The service UI harness passed startup gating, rejection of obsolete API errors
and native state snapshots, and genuine service replacement while preserving
drafts and unsaved provider fields without saving secrets. These instrumented
source checks are separate from packaged-executable acceptance.

## Packaged executable acceptance

The packaged 0.1.5 executable passed run organization, experiment reuse,
history import, native-close protection and the general Electron acceptance
suite with isolated synthetic profiles. No renderer errors were observed.

History-import regression checks covered native folder selection, invalid and
empty selection, changed-source rejection and rescan, read-only controls, copy
exclusions, candidate and log viewing, native report export, duplicate skipping,
draft preservation, restart and unchanged source hashes and modification times.
Native-close checks covered cancellation, explicit discard, pending and failed
saves, successful saves and preservation of unsaved settings.

The engine selection check reported `reused=true` for the existing 0.1.4 engine.
The engine source SHA-256 is unchanged:
`56d274eec3df1d02eefb52dbe5cc894ff4a5eabe066f483046462e4e021e9fa2`.
The installed application subsequently confirmed the same engine ready without
requesting installation, and the full workflow below used it in place.

## Installation and final artifact verification

The DMG was mounted read-only and its Applications shortcut resolved correctly.
Deep strict ad-hoc signature verification passed on the unpacked application,
DMG application and extracted ZIP application. All 2,809 files and symlink targets
had the same complete bundle fingerprint. The bundled runtime recorded version
0.1.5, darwin-arm64 and the clean source commit above. Final signatures passed
again after packaged and installed acceptance.

The upgrade is installed at `/Applications/BioMaker.app`. The previous app is
preserved under `release/installed-backups/BioMaker-0.1.4-20261008-235233.app`.
The actual user profile opened with its existing engine ready, without requesting
installation. Configuration file existence/content and complete run records
matched the pre-upgrade snapshot. No synthetic QA data was added to that profile.
The normal installed application was reopened after testing.

## Installed desktop workflow

The installed 0.1.5 executable completed UI submission, durable queue execution,
actual CPU training, frozen evaluation, result inspection and native report
export in **15.073 seconds**. It reached `succeeded` with exit code 0 and zero
renderer errors. The engine fitted a scikit-learn classifier on synthetic data;
five model requests went to a deterministic localhost fixture. Final retraining
was disabled. This verifies software behavior, not external-provider quality,
multi-round biomedical search or scientific model performance.

The harness restricted the Python sidecar and training descendants to localhost;
only the exact `/bin/ps` executable was exempt for process identity inspection.
The candidate confirmed an external-address connection was denied with `EPERM`.
Installed engine/runtime fingerprints remained unchanged, and the QA profile's
processes were confirmed stopped afterward. Durable synthetic evidence is saved
under `output/qa/desktop-workflow-0.1.5/`.

## Artifacts and evidence

| Artifact | SHA-256 |
| --- | --- |
| BioMaker-0.1.5-mac-arm64.dmg | `619bed52796a99f965a73dbe89958958d97e3f9e6cff9b726cc69235fb165c5f` |
| BioMaker-0.1.5-mac-arm64.zip | `bbf3444c68edd5cd1d868d791a58264c06ec73abc7b1ccc7ab2d34984cc7598b` |

Completed machine-readable receipts under `apps/desktop/output/`:

- `engine-upgrade-0.1.5.json`
- `run-organization/acceptance.json`
- `experiment-reuse/acceptance.json`
- `service-ui/acceptance.json`
- `run-organization-packaged-0.1.5/acceptance.json`
- `experiment-reuse-packaged-0.1.5/acceptance.json`
- `history-import-packaged-0.1.5/acceptance.json`
- `native-close-packaged-0.1.5/acceptance.json`
- `playwright/electron-acceptance.json`

- `package-verification-0.1.5.json`
- `package-installation-0.1.5.json`
- `installed-engine-acceptance-0.1.5.json`
- `desktop-workflow-acceptance-0.1.5.json`
- `qa/desktop-workflow-0.1.5/`

## Boundaries

Acceptance in this report is limited to macOS on Apple silicon. Windows, Linux,
Intel Mac packaging and independent clean-hardware installation remain unverified.
Apple Developer ID signing, notarization and signed updates remain pending;
an ad-hoc signature does not establish Developer ID distribution readiness.

History import copies experiment records rather than complete executable
workspaces. Reuse prepares a new draft and does not resume an imported process,
recover its dataset or transfer its runtime environment. Metadata organization
does not change the meaning or provenance of original research results.

Synthetic acceptance establishes software behavior, not biomedical performance
or external-provider quality. The engine runs generated code with the user's
access; test-only network restrictions are not a production security sandbox.
