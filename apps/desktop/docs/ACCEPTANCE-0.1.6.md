# BioMaker Desktop 0.1.6 acceptance

Date: 2026-10-09. Host: macOS, Apple silicon / arm64.
Release source commit: `b3f19fe4b79d52f2f8702ef3b00b761c52a277a1`, recorded clean
in the bundled manifest.

Source validation, expanded development recovery, packaged acceptance, final
artifact verification, installation and the installed synthetic training workflow
passed.

## Verified changes

- A native service construction failure leaves the application window and its
  draft available. Users can retry the local service or export diagnostics
  without a working Python sidecar. Recovery keeps the existing native window
  and renderer instead of restarting the application or clearing unsaved input.
- A native controller owns service construction, startup, retry and shutdown.
  Its public revisions increase across replacement instances; callbacks from
  an older generation cannot replace the current state. Concurrent retries share
  one operation. Replacement waits for the previous service to stop, and a stop
  failure retains ownership of that instance rather than starting another one.
  Stop sends SIGTERM, escalates after four seconds to SIGKILL and waits up to
  two more seconds for confirmed exit. An unconfirmed exit rejects the stop and
  prevents a replacement; signals alone never establish successful shutdown.
- Public controller state contains a fixed status, revision and, when failed,
  an allowlisted failure code. Its bounded event history contains only fixed
  codes, statuses and timestamps. Original exception messages and sidecar
  messages do not enter those structures.
- Native diagnostic export works independently of the service API and saves a
  JSON file through the native save dialog. Cancellation saves nothing; a failed
  write can be retried. The user decides whether to share the local file; export
  does not upload it.
- The diagnostic collector emits an explicit whitelist: validated application
  and system versions, service status, up to 50 structured events, validated
  bundle provenance and required-file availability. It reads at most 64 KiB from
  the fixed bundled runtime manifest. Missing, malformed, oversized, unreadable
  or unsafe manifests produce fixed diagnostic statuses instead of preventing
  export. Symlink and special-file manifests are rejected.
- Storage diagnostics inspect only the main-process-supplied profile root and
  its fixed `logs` and `state` locations using directory metadata and access
  checks. Each reports `exists`, `directory` and `existingWritable`. A regular
  file occupying `logs` is distinguishable from a usable directory, even when
  every bundle check passes. Missing directories remain explicitly absent;
  their status does not imply whether a parent could create them. Symlinks are
  not traversed. No profile file contents or directory listings are read.
- Diagnostic reports exclude raw logs, configuration, ledger contents, task or
  dataset values, paths, usernames, hostnames, provider values, environment
  variables and original exception text. Unknown or malformed values in
  allowlisted fields are reduced to fixed unknown values.
- Modal initialization now preserves focus when a user or an autofocus control
  has already selected a field before the first animation frame. Initial focus
  selection still honors the designated control, and keyboard containment and
  focus restoration remain in place.

## Source validation

102 WebUI tests, 89 desktop service/native-runtime tests and 78 unique native
and package tests passed: **269 tests total**. The native total includes the
separately enabled real-sidecar integration test. TypeScript and production
build passed. Vite reports a nonblocking 503 kB minified UI chunk warning.

Targeted collector tests passed for privacy sentinels, strict field validation,
bounded output, manifest failure handling and storage inspection. Tests verify
that only the fixed runtime manifest is opened, profile inspection uses only
`lstat` and `access` on the three permitted locations, and no file-content reads
or directory enumeration occur for the profile. They cover a regular `logs`
file, missing profile and child directories, unwritable directories and symlinks.

## Development Electron acceptance

The expanded service-recovery workflow passed **7 of 7 checks**:

- A real construction failure preserved the window, draft and native diagnostic
  export controls.
- Concurrent native exports shared one save dialog and one result.
- Cancelling export saved nothing, and a write failure allowed another attempt.
- Manual retry preserved the native window and renderer draft while advancing
  the public service revision.
- A real service failure and retry preserved unsaved provider fields and the
  original saved configuration.
- Native-menu export worked without a sidecar API and excluded private values.
- The harness observed no provider calls, service mutations or renderer errors.

The same workflow verifies that the exported storage facts identify the real
regular-file obstruction at the logs location, and that a 960 by 680 window has
no horizontal overflow. The fault modifies only an isolated synthetic profile,
including its own copied service runtime; application-bundle files are untouched.

Native service-state acceptance passed **6 of 6 checks**, covering authenticated
readiness, transport recovery, stale-response rejection, real sidecar replacement,
folder-picker boundaries and main-frame IPC authorization. Service UI acceptance
passed **3 of 3 checks** for startup gating, obsolete-response rejection and draft
and provider-field preservation during service replacement.

Experiment reuse passed **8 of 8 checks**, including draft cancellation and
explicit replacement, restart persistence, dataset reselection for read-only
history, unchanged source/imported artifacts and ledger records, and no run
mutations, provider requests, engine installation or renderer errors.

These instrumented development checks are separate from packaged-executable and
installed-application acceptance.

## Packaged executable acceptance

The packaged 0.1.6 executable passed all six workflows with isolated synthetic
profiles:

| Workflow | Passed checks |
| --- | --- |
| Service recovery | 7 / 7 |
| Native-close protection | 6 / 6 |
| History import | 10 / 10 |
| Run organization | 8 / 8 |
| Experiment reuse | 8 / 8 |
| General Electron acceptance | 22 / 22 |

Packaged recovery reproduced a real construction failure while keeping the
window, draft, retry and native diagnostic export available. The report identified
the synthetic regular-file obstruction at the logs location without including
private content. Concurrent exports shared one dialog and result; cancellation,
write failure, retry, service replacement and unsaved provider-field preservation
passed. The compact 960 by 680 window had no horizontal overflow.

Regression workflows retained native-close safeguards, history-import exclusions
and source-file preservation, revision-aware organization, and explicit draft
replacement without starting an experiment. General acceptance covered the real
sidecar, renderer sandbox, IPC boundaries, settings, onboarding, theme, language,
restart, artifacts, compact layout and notification interactions. No renderer
errors were observed. Native notification display was intercepted in QA; actual
OS banner delivery is not established by these checks.

## Installation and final artifact verification

The DMG was mounted read-only and its Applications shortcut resolved correctly.
Deep strict signature verification passed on the unpacked application, DMG
application and extracted ZIP application. Their complete 2,809-file bundle
fingerprints, including symlink targets, matched. The bundled manifest recorded
version 0.1.6 and the clean source commit above.

The upgrade is installed at `/Applications/BioMaker.app`. The previous app is
preserved under `release/installed-backups/BioMaker-0.1.5-20261009-002600.app`.
The actual user profile opened with its existing engine ready and requested no
installation. Its configuration file and complete run records matched the
pre-upgrade snapshot, with no synthetic fixtures added. This profile contained
zero run records before and after the upgrade; this particular check does not
establish migration of a populated history.

Engine selection reported `reused=true`, and the installed application confirmed
reuse. The engine source SHA-256 remains
`56d274eec3df1d02eefb52dbe5cc894ff4a5eabe066f483046462e4e021e9fa2`.
After the installed workflow, both installed and packaged signatures passed
again. The installed application's complete bundle fingerprint matched the
verified package:
`8df6b5ab0e72a7f780dcfb6c2b1b8bf3ee31c351a541065bd1082c91edb2fdea`.

## Installed desktop workflow

The installed 0.1.6 executable completed UI submission, durable queue execution,
actual CPU training, frozen evaluation, result inspection and native report
export in **15.234 seconds**. The run reached `succeeded` with exit code 0 and no
renderer errors. A scikit-learn logistic regression model was fitted on synthetic
data using the installed engine, with no installed-source override. Five model
requests went to a deterministic localhost fixture; there were no external
provider calls. Final retraining was disabled.

The test-only spawn wrapper restricted the Python sidecar and its queue, uv and
training descendants to localhost. Only the exact `/bin/ps` executable was exempt
for read-only process identity inspection. The candidate confirmed that an
external-address connection was denied with `EPERM`. The production Chromium
sandbox and source were unchanged.

Installed engine/runtime fingerprints remained unchanged. The actual user
profile was unchanged after QA, and no matching QA processes remained. Durable
synthetic evidence, including preserved symlinks, is saved under
`output/qa/desktop-workflow-0.1.6/`. This verifies the installed software workflow;
it does not measure external-provider quality, multi-round biomedical search or
scientific model performance.

## Artifacts and evidence

| Artifact | SHA-256 |
| --- | --- |
| BioMaker-0.1.6-mac-arm64.dmg | `9d2953f900eba68dbefd8d63fc0068902bab5be68e568e83e782ad8b11a0abbd` |
| BioMaker-0.1.6-mac-arm64.zip | `cd5ba96731764d6749cb86cb806fb4dd253d79f8154b670a78daa347716c5816` |

Machine-readable receipts and durable evidence under `apps/desktop/output/`:

- `service-recovery/acceptance.json`
- `native-service-state/acceptance.json`
- `service-ui/acceptance.json`
- `experiment-reuse/acceptance.json`
- `service-recovery-packaged-0.1.6/acceptance.json`
- `native-close-packaged-0.1.6/acceptance.json`
- `history-import-packaged-0.1.6/acceptance.json`
- `run-organization-packaged-0.1.6/acceptance.json`
- `experiment-reuse-packaged-0.1.6/acceptance.json`
- `electron-packaged-0.1.6/acceptance.json`
- `package-verification-0.1.6.json`
- `package-installation-0.1.6.json`
- `engine-upgrade-0.1.6.json`
- `installed-engine-acceptance-0.1.6.json`
- `desktop-workflow-acceptance-0.1.6.json`
- `post-workflow-verification-0.1.6.json`
- `qa/desktop-workflow-0.1.6/` (durable fixture, logs, screenshots and outputs)

## Boundaries

The checks recorded so far are on macOS with Apple silicon. Windows, Linux,
Intel Mac packaging and independent clean-hardware installation remain
unverified. Apple Developer ID signing, notarization and signed updates remain
pending; an ad-hoc signature does not establish Developer ID distribution
readiness.

Diagnostics report a bounded snapshot of availability and lifecycle state. They
do not include raw logs or establish the full cause of every service failure.
An existing writable directory check is not a guarantee of future writes or
successful service startup.

Synthetic acceptance establishes software behavior, not biomedical performance
or external-provider quality. The engine runs generated code with the user's
access; test-only network restrictions are not a production security sandbox.
