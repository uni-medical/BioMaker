# BioMaker Desktop 0.1.6 acceptance

Date: 2026-10-09. Host: macOS, Apple silicon / arm64.
Release source commit: **Pending**.

Source validation and expanded development recovery passed. Packaged acceptance,
installation and the installed training workflow remain **Pending**.

## Changes under acceptance

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

**Pending.** Build from the final recorded source commit and run the packaged
recovery and regression workflows. Record results, renderer errors, profile
isolation and any differences from development behavior.

## Installation and final artifact verification

**Pending.** Record final DMG and ZIP hashes, read-only DMG mount verification,
Applications shortcut, bundled version and source identity, complete bundle
comparison and final signature verification after packaged QA.

**Pending.** Record installation at `/Applications/BioMaker.app`, the previous
application backup, preservation of the actual user's configuration and run
records, and the installed engine selection result. Do not infer preservation
or engine reuse from an earlier release.

## Installed desktop workflow

**Pending.** Exercise the installed 0.1.6 application through UI submission,
durable queue execution, actual CPU training, frozen evaluation, result
inspection and native report export. Record elapsed time, terminal status, exit
code, renderer errors, fixture/provider behavior, installed fingerprints and
final process cleanup. Development recovery checks do not establish this result.

## Artifacts and evidence

| Artifact | SHA-256 |
| --- | --- |
| BioMaker-0.1.6-mac-arm64.dmg | **Pending** |
| BioMaker-0.1.6-mac-arm64.zip | **Pending** |

Current development receipts under `apps/desktop/output/`:

- `service-recovery/acceptance.json`
- `native-service-state/acceptance.json`
- `service-ui/acceptance.json`
- `experiment-reuse/acceptance.json`

Final source, packaged recovery, package verification, installation and installed
training receipts: **Pending**.

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
