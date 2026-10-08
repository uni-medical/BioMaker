# BioMaker Desktop 0.1.7 acceptance

Date: 2026-10-09. Host: macOS, Apple silicon / arm64.
Release source commit: `9f19afba2686b1919b1e5cb10eb9043ebbd3e1c8`, clean checkout
used for the package.

Source validation, development and packaged UI workflows, real packaged-engine
health and repair acceptance, artifact verification, installation and the
installed synthetic training workflow passed. Post-workflow verification also
passed.

## Verified changes

- Users can check an installed runtime separately from installing or repairing
  it. The check runs the existing verification script with the engine's Python
  interpreter and `-I -B`, including scientific-package imports, CLI entrypoint
  validation and a fixed CPU LightGBM self-test. It does not invoke uv, download
  dependencies or provision native libraries. Verification is limited to
  90 seconds.
- Repair uses `uv sync --frozen --reinstall`, followed by native-runtime
  provisioning and the complete verification script. Forced reinstallation
  restores damaged installed packages instead of accepting their existing
  distribution metadata as sufficient. A successful repair writes a new receipt
  atomically only after verification completes.
- Runtime state distinguishes installation, checking and repair, reports health
  separately from readiness, and records the most recent completed check or
  maintenance result time. A valid older receipt may still permit execution
  while health remains unknown; it is not labeled as a new successful check.
  Readiness is false throughout any active maintenance operation.
- Missing or invalid interpreter/receipt state and genuine verification failures
  invalidate readiness. Failed checks save a durable failure marker through the
  held maintenance-lock descriptor and remove the old installation receipt.
  Restart reports failed health even when permissions prevent receipt removal.
  Marker reads are bounded to 128 bytes, reject symlinks and treat unknown or
  malformed nonempty contents as failed. Empty lock files preserve older valid
  installations. Repair clears and syncs the marker only after verification and
  successful receipt publication. Installation and repair invalidate the receipt
  before modifying dependencies. Cancellation never reports a successful health
  check or publishes a newly trusted receipt; healthy and cancelled checks leave
  marker contents and modification time unchanged.
- Concurrent requests to one installer share its current operation. An advisory
  lock at a fixed engine-root filename prevents another installer from starting
  overlapping maintenance. Every maintenance subprocess inherits the lock file
  descriptor, so abrupt sidecar exit does not release an active worker's lease.
  The lock inode is retained and recognized as generated metadata by engine
  selection.
- Cancellation and timeouts terminate the owned process group and confirm its
  exit, including descendants after their leader exits. An unconfirmed stop
  keeps the operation and lease active. External lock ownership makes the
  runtime unavailable without offering cancellation of another process's work.
  An unusable lock file or disk-access failure is reported separately from
  external ownership. Output buffering and retained logs are bounded, and
  credentials are redacted.
- Authenticated maintenance admission rejects managed queued, running or
  stopping runs and records awaiting process cleanup. Desktop request admission
  serializes maintenance startup with experiment submission, requeue and queue
  resume. Execution requests are rejected while the runtime is unavailable.
  Imported read-only histories do not claim engine ownership or block
  maintenance, and cancellation remains available.
- The UI exposes check and repair actions, requires an explicit repair choice,
  displays progress, results and the latest completion time, and gates experiment
  submission during maintenance. Cancellation waits for worker settlement.
  Drafts and unsaved provider fields remain in the current renderer; failed
  requests and local disk-access problems allow a later retry.

## Source validation

102 WebUI tests, 132 desktop service/native-runtime tests and 79 unique native
and package tests passed: **313 tests total**. The service total includes 43 new
health and maintenance-admission cases. The native total includes the separately
enabled real-sidecar integration test. TypeScript and the production build
passed.

The build reports a nonblocking Vite warning for a **509.41 kB** minified UI
chunk. Python test output retains the existing Starlette/httpx deprecation
warning; these warnings did not fail validation.

Focused health tests cover receipt and interpreter failures, isolated verifier
arguments, no-install checks, real subprocess timeout and cancellation,
duplicate requests, forced reinstall ordering, receipt invalidation, bounded
output, and recovery of cached readiness after external maintenance. A real
worker retained its inherited lease after its owner exited with `os._exit`.
Other tests cover descendants surviving their leader, unsafe lock files,
signal-permission failures and unconfirmed shutdown without releasing the lease.
Failure-marker regressions retain a formerly valid receipt after a simulated
deletion-permission failure and verify that a new installer still refuses
readiness. Successful repair clears the marker in the correct order without
changing its inode; malformed marker contents fail closed.

Admission tests cover authentication, queued and active work, pending cleanup,
imported histories, cancellation availability, concurrent submission and
maintenance, durable admission ordering, requeue and resume. These source tests
do not establish that a real installed dependency can be repaired.

## Development Electron acceptance

The runtime-health UI harness passed **9 of 9 checks**, using real Electron and
the native bridge with explicitly synthetic engine-state responses. It covered
unknown versus verified health, checking and cancellation,
failure presentation, repair confirmation and retry, draft preservation,
restored readiness, external maintenance and local disk-access errors. This
harness does not install, check or repair a real engine and does not establish
scientific-package health. A deliberately delayed older ready response could not
overwrite an accepted check's busy state or enable experiment submission. No
renderer errors or forbidden requests were observed.

## Real-engine health and repair acceptance

The final packaged service at `Contents/Resources/service` passed **4 of 4
checks in 102.394 seconds**, including the failure-marker changes. The offline
harness used isolated APFS clones of the installed engine and populated package
cache. An OS policy denied outbound networking and writes to the original engine
and cache.

The initial healthy check ran real imports and the fixed CPU self-test without
invoking uv. Removing the cloned MONAI package caused a real failed check, removed
the installation receipt and left a restarted installer not ready. Repair then
ran `uv sync --frozen --reinstall` using the offline cloned cache, provisioned the
native runtime and completed full verification. A second check passed without
invoking uv.

Complete before/after fingerprints of the source installed engine and original
cache matched, and the health-QA processes were confirmed gone. The final
acceptance receipt, repaired installation receipt and empty cleared failure
marker are preserved under `output/qa/engine-health-0.1.7/`.

An earlier development run passed the same four scenarios in 104.605 seconds
before marker hardening. The packaged result above validates the final service;
it is separate from synthetic UI acceptance and from the complete installed
training workflow below.

## Packaged executable acceptance

The packaged 0.1.7 executable passed all seven workflows with isolated synthetic
profiles:

| Workflow | Passed checks |
| --- | --- |
| Runtime-health UI | 9 / 9 |
| Native-close protection | 6 / 6 |
| Run organization | 8 / 8 |
| Experiment reuse | 8 / 8 |
| History import | 10 / 10 |
| Service recovery | 7 / 7 |
| General Electron acceptance | 22 / 22 |

The packaged runtime-health workflow verified busy-state gating, cancellation,
repair confirmation, retry after an error, draft preservation, external-worker
and disk-access states, and rejection of a delayed older ready response. Engine
states were synthetic: this UI harness performed no real engine check, install
or repair and sent no provider requests or run mutations.

Regression workflows preserved native-close safeguards, history-import
exclusions and original files, revision-aware organization and explicit draft
replacement. Real service failures retained the window, draft and native
diagnostic export; retry preserved unsaved provider fields. General acceptance
covered the real sidecar, renderer sandbox, IPC boundaries, settings, onboarding,
theme, language, restart, artifacts, compact layout and notification interactions.
No renderer errors were observed. Notification display was intercepted in QA;
actual OS banner delivery is not established by these checks.

## Installation and final artifact verification

The DMG was mounted read-only and its Applications shortcut resolved correctly.
Deep strict signature verification passed on the unpacked application, DMG
application and extracted ZIP application. Their complete 2,809-file bundle
fingerprints, including symlink targets, matched. The bundled manifest recorded
version 0.1.7 and the clean source commit above.

The upgrade is installed at `/Applications/BioMaker.app`. The previous app is
preserved under `release/installed-backups/BioMaker-0.1.6-20261009-010900.app`.
The actual user profile opened with its existing engine ready and requested no
installation. Its configuration file and complete run records matched the
pre-upgrade snapshot, and no synthetic fixtures were imported into that profile.
This is a configuration-and-record preservation check, not a claim that every
profile file is unchanged: 0.1.7 initializes a generated empty maintenance lock
in the reused engine, and browser/cache metadata can change during ordinary use.

Engine selection reported `reused=true`, and the installed application confirmed
reuse. The engine source SHA-256 remains
`56d274eec3df1d02eefb52dbe5cc894ff4a5eabe066f483046462e4e021e9fa2`.
The verified package bundle fingerprint is
`1fe485c5eb24191d2a33ac87d967ff3802ca32249b0cd1f895b33408bbcc4cd9`.

After the complete installed training workflow, installed and packaged
signatures passed again. The installed application's complete bundle fingerprint
matched the verified package. The normal installed application was reopened and
left open; its UI showed version 0.1.7, the local runtime ready, no experiments in
the actual profile and a model still awaiting configuration.

## Installed desktop workflow

The installed 0.1.7 executable completed UI submission, durable queue execution,
actual CPU training, frozen evaluation, result inspection and native report
export in **16.305 seconds**. The run reached `succeeded` with exit code 0 and
cleanup complete. No renderer or fixture errors were observed. A scikit-learn
logistic regression model was fitted on synthetic data using the reused installed
engine, with no installed-source override. Five model requests went to a
deterministic localhost fixture; there were no external provider calls. Final
retraining was disabled.

The test-only spawn wrapper restricted the Python sidecar and its queue, uv and
training descendants to localhost. Only the exact `/bin/ps` executable was exempt
for read-only process identity inspection. The candidate confirmed that an
external-address connection was denied with `EPERM`. The production Chromium
sandbox and source were unchanged.

Installed-file fingerprints remained unchanged during the workflow. The actual
user configuration and complete run records were unchanged after QA, and no
matching QA processes remained. Durable synthetic evidence is saved under
`output/qa/desktop-workflow-0.1.7/`. This verifies the installed software workflow;
it does not measure external-provider quality, multi-round biomedical search or
scientific model performance.

## Artifacts and evidence

| Artifact | SHA-256 |
| --- | --- |
| BioMaker-0.1.7-mac-arm64.dmg | `efc1253d4f1fce5cae170ffdcd110a70c4b10b20997c10e0eb8a1221c578a976` |
| BioMaker-0.1.7-mac-arm64.zip | `552fdabf2a11269051c11fe91d2b2634df87eae74a189a9e89499904a65cde29` |

Completed receipts and durable evidence under `apps/desktop/output/`:

- `runtime-health-ui/acceptance.json` (nine synthetic engine-state UI checks)
- `engine-health-real-0.1.7.json` (initial development run before marker hardening)
- `runtime-health-ui-packaged-0.1.7/acceptance.json`
- `native-close-packaged-0.1.7/acceptance.json`
- `run-organization-packaged-0.1.7/acceptance.json`
- `experiment-reuse-packaged-0.1.7/acceptance.json`
- `history-import-packaged-0.1.7/acceptance.json`
- `service-recovery-packaged-0.1.7/acceptance.json`
- `electron-packaged-0.1.7/acceptance.json`
- `engine-health-packaged-0.1.7.json`
- `qa/engine-health-0.1.7/`
- `package-verification-0.1.7.json`
- `package-installation-0.1.7.json`
- `engine-upgrade-0.1.7.json`
- `installed-engine-acceptance-0.1.7.json`
- `desktop-workflow-acceptance-0.1.7.json`
- `post-workflow-verification-0.1.7.json`
- `qa/desktop-workflow-0.1.7/` (durable fixture, logs, screenshots and outputs)

## Boundaries

The source checks recorded here are on macOS with Apple silicon. Windows,
Linux and Intel Mac packaging and independent clean-hardware installation remain
unverified. Apple Developer ID signing, notarization and signed updates remain
pending; an ad-hoc signature does not establish Developer ID distribution
readiness.

A runtime check validates the listed imports and fixed self-test at that time.
It does not guarantee every dependency path or future workload. Offline repair
depends on an available package cache; it does not establish a fresh download or
cold installation on another machine. Maintenance admission covers the desktop
service and its managed run records, rather than unrelated external processes
manually launched against the same environment.

Synthetic acceptance establishes software behavior, not biomedical performance
or external-provider quality. The engine runs generated code with the user's
access; test-only network restrictions are not a production security sandbox.
