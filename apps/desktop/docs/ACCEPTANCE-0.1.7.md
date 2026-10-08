# BioMaker Desktop 0.1.7 acceptance

Date: 2026-10-09. Host: macOS, Apple silicon / arm64.
Release source commit: **Pending**.

Source validation, the development UI workflow and the initial real-engine
health and repair acceptance passed. Final packaged health acceptance, packaging,
installation and the installed training workflow remain **Pending**.

## Changes under acceptance

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

The initial development offline harness passed **4 of 4 checks in 104.605
seconds**, before the final failure-marker hardening. It used isolated APFS
clones of the installed engine and populated package cache. An OS policy denied
outbound networking and writes to the original engine and cache.

The initial healthy check ran real imports and the fixed CPU self-test without
invoking uv. Removing the cloned MONAI package caused a real failed check, removed
the installation receipt and left a restarted installer not ready. Repair then
ran `uv sync --frozen --reinstall` using the offline cloned cache, provisioned the
native runtime and completed full verification. A second check passed without
invoking uv.

Complete before/after fingerprints of the source installed engine and original
cache matched. These results are recorded separately from the synthetic Electron
UI fixture and do not establish packaged or installed 0.1.7 acceptance.

**Pending.** Run the complete real-engine harness again against the final packaged
0.1.7 service, including the durable failure-marker changes. The initial
development result above does not establish that final packaged result.

## Packaged executable acceptance

**Pending.** Record the final clean source commit and run the packaged
runtime-health UI acceptance plus the existing recovery, close protection,
history import, organization, reuse and general Electron regression workflows.
Preserve the distinction between synthetic UI states and real-engine checks.

## Installation and final artifact verification

**Pending.** Record final DMG and ZIP hashes, read-only DMG mount verification,
Applications shortcut, bundled version and source identity, complete bundle
comparison and final signature verification after packaged QA.

**Pending.** Record installation at `/Applications/BioMaker.app`, the previous
application backup, actual-profile configuration and run-record preservation,
and the installed engine selection result. Earlier releases do not establish
these results for 0.1.7.

## Installed desktop workflow

**Pending.** Exercise the installed 0.1.7 application through UI submission,
durable queue execution, actual CPU training, frozen evaluation, result
inspection and native report export. Record elapsed time, terminal status, exit
code, renderer errors, fixture/provider behavior, installed fingerprints and
final process cleanup. Health checks and dependency repair alone do not
establish this complete workflow.

## Artifacts and evidence

| Artifact | SHA-256 |
| --- | --- |
| BioMaker-0.1.7-mac-arm64.dmg | **Pending** |
| BioMaker-0.1.7-mac-arm64.zip | **Pending** |

Completed development receipts under `apps/desktop/output/`:

- `runtime-health-ui/acceptance.json` (nine synthetic engine-state UI checks)
- `engine-health-real-0.1.7.json` (initial development run before marker hardening)

Final packaged health, packaged UI, installation and installed-training receipts:
**Pending**.
Add the exact completed receipt and durable evidence paths after verification.

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
