# BioMaker Desktop 0.1.9 acceptance

Date: 2026-10-09. Target: macOS, Apple silicon / arm64.
Release source commit: `ef3be5db9c79c115731799f7ec65adb2b77fb01c`.

This iteration adds a local dataset overview and fresh desktop admission checks.
Source checks, development and packaged acceptance, offline repair, artifact
verification, installation and the installed training workflow passed.

## Dataset behavior

- New drafts show ordinary-file counts, logical sizes, bounded file-type counts
  and up to ten top-level entries. Compound extensions such as `.nii.gz` retain
  their meaning. File counts are not sample counts.
- Native code replaces renderer-supplied grants with saved folder grants and the
  application workspace. Invalid saved grants are ignored. Metadata inspection's
  path resolution and directory enumeration run in a separate Python worker.
- The worker reads metadata only, with no file contents, archive extraction,
  image decoding, model calls or engine installation. The renderer holds results
  in memory; full overview results are not saved in the experiment draft.
- Picker grants are canonical paths. Retargeted grants do not authorize new link
  targets. Directory descriptors and no-follow opens protect traversal against
  symbolic-link replacement. Root aliases can resolve to an authorized directory;
  internal links are counted without inspecting their targets.
- Each check has a two-second soft budget, a five-second worker lifetime,
  2,000-entry limit and three descendant-directory levels. Summary limits are
  ten top-level entries and twenty extension categories. Names and numbers are
  bounded, and output is limited to 32 KiB.
- A checkpoint requires successful root authorization, directory opening and a
  real enumeration step. If later I/O reaches the hard deadline, the response can
  identify an accessible root with an incomplete overview. A timeout before that
  checkpoint does not claim root accessibility. A worker watchdog, parent-loss
  check and retained ownership prevent unchecked replacement workers.
- Completely inspected empty trees are blocked. Unknown extensions, zero-byte
  ordinary files, links, special entries and incomplete descendant inspections
  are not treated as empty datasets. Incomplete scans show their limits.
- Desktop create and requeue perform fresh bounded access checks under the
  existing admission lock. Confirmed failures create no run record. Queue resume
  retains its existing contract. A successful check does not freeze the directory
  or guarantee availability when a queued experiment later starts.
- Path changes, refreshes and service revisions invalidate older overview
  results. Submissions use a fixed draft snapshot and reject stale preflight
  results. Known dataset refusals preserve the draft and provide a concrete
  reason; uncertain submission outcomes require checking the experiment list.

These folder grants constrain the overview. They do not sandbox generated
experiment code or change the shared CLI/WebUI scientific evaluation contract.

## Source validation

**455 source tests passed:** 102 WebUI, 268 desktop service/native-runtime and
85 native/package tests, with no skips in the native suite. The service suite
includes 42 new dataset cases; native checks include six authorization-envelope
cases. TypeScript checks and the production build passed. The existing Python
Starlette/httpx deprecation and Vite chunk-size warning remain nonblocking.

Receipts are `output/webui-tests-0.1.9.xml`,
`output/service-tests-0.1.9.xml` and `output/native-tests-0.1.9.tap`.

## Development acceptance

The real Electron/native-bridge/sidecar dataset suite passed **14/14 checks**,
with 25 metadata requests against isolated synthetic directories. It covered
forged grants, workspace aliases, prefix escapes, empty descendants, unfamiliar
formats, zero-byte files, symlink-only directories, cycles, retargeted grants,
missing and replaced roots, FIFO/socket entries, depth limits, picker cancellation,
A-to-B-to-A response races and a real sidecar restart (revision 2 to 4).
An unsupported selected path receives a fixed localized reselect message;
its invalid saved grant cannot block a later valid directory selection.

There were no provider, experiment, installation or configuration-write requests,
and no renderer errors. Reviewed screenshots cover light/dark themes and a
960-by-680 window with expanded metadata and keyboard refresh. Evidence is under
`output/dataset-inspection-0.1.9/`.

Submission UI acceptance passed **7/7 checks** with real metadata scans and two
real sidecar restarts. Engine/provider readiness and four submission responses
were explicitly simulated: no POST reached the real run endpoint, and the real
ledger stayed empty with no configured provider or installed engine. Checks
covered empty-folder blocking, localized admission failures, stale preflight
rejection, snapshot and duplicate-submit protection, a late folder picker, and
uncertain results after a restart. There were no forbidden requests or renderer
errors. This validates UI behavior, not execution; evidence is under
`output/dataset-submission-0.1.9/`.

Existing native service-state checks passed **6/6**, and service UI checks passed
**3/3**, including real sidecar replacement, stale-response rejection and draft /
unsaved-provider-field preservation. These use controlled startup responses and
are development acceptance, separate from packaged acceptance. Receipts are in
`output/native-service-state-0.1.9/` and `output/service-ui-0.1.9/`.

## Packaged acceptance

All **105/105 checks** passed against the actual packaged executable, using
isolated profiles and a system-only PATH:

| Suite | Passed | Receipt directory under `output/` |
| --- | ---: | --- |
| Dataset metadata | 14/14 | `dataset-inspection-packaged-0.1.9/` |
| Submission guards | 7/7 | `dataset-submission-packaged-0.1.9/` |
| Provider response checks | 14/14 | `provider-check-packaged-0.1.9/` |
| Runtime maintenance UI | 9/9 | `runtime-health-ui-packaged-0.1.9/` |
| Native close protection | 6/6 | `native-close-packaged-0.1.9/` |
| Experiment organization | 8/8 | `run-organization-packaged-0.1.9/` |
| Experiment reuse | 8/8 | `experiment-reuse-packaged-0.1.9/` |
| History import | 10/10 | `history-import-packaged-0.1.9/` |
| Service recovery | 7/7 | `service-recovery-packaged-0.1.9/` |
| General Electron workflows | 22/22 | `electron-packaged-0.1.9/` |

Receipts are `acceptance.json`, except the general suite's
`electron-acceptance.json`. The packaged metadata suite performed 25 real local
inspections. Submission tests used nine real metadata requests and two real
sidecar restarts, with four intercepted simulated submissions. The provider suite
used twelve short generation and three model-list requests to a localhost fixture.
Runtime UI states and maintenance responses were simulated; real repair is
verified separately below. No renderer errors or forbidden requests remained
in the completed suites. Reviewed screenshots cover error states, light/dark
themes, compact windows, keyboard controls, settings, history and results.

Two existing QA assumptions initially failed: the runtime UI fixture used an
empty dataset, and the provider harness blocked the new read-only metadata POST.
Only the test fixtures/allowlist were updated, then both suites passed against
the same unmodified package. Initial failure evidence remains in
`output/runtime-health-ui-packaged-0.1.9/initial-fixture-failure.json` and
`output/provider-check-packaged-0.1.9-initial-allowlist-failure.log`.

## Real offline engine verification and repair

The packaged service passed **4/4 checks in 109.824 seconds** under an OS policy
denying outbound networking. An APFS clone passed real dependency imports and
a fixed CPU self-test, detected removal of MONAI in that clone, retained its
not-ready state across restart, completed a real locked offline forced reinstall
and native-library provisioning, and passed another download-free health check.

All 34,295 original engine files and 29,507 original cache files retained matching
before/after fingerprints. The actual engine and cache were not damaged or
repaired. The report is `output/engine-health-packaged-0.1.9.json`; supporting
evidence is under `output/qa/engine-health-0.1.9/`.

## Artifacts and installation

The DMG was mounted read-only and its Applications shortcut resolved correctly.
Deep strict signature checks passed on the unpacked app, DMG app and extracted
ZIP app. Their complete 2,813-file bundle fingerprints matched:
`b6fa9a4fade1cb6ba114b9449c962643fd736beb4133b67a6f78a8776bcdaaf4`.
The manifest records version 0.1.9 and the clean release source commit above.

| Artifact | SHA-256 |
| --- | --- |
| `BioMaker-0.1.9-mac-arm64.dmg` | `c8f29f6100d4598a02882ebf9fe7dda4d66098647fa3b2d0de739dce03c9cd48` |
| `BioMaker-0.1.9-mac-arm64.zip` | `b18882b42b8805048fb0a8a7ac9a9bcba1a44adee694114b9d40a8beb7c9612c` |

Version 0.1.9 is installed at `/Applications/BioMaker.app`. The previous app is
preserved at `release/installed-backups/BioMaker-0.1.8-20261009-023457.app`.
The actual installed app started with a system-only PATH and reused its existing
ready engine without requesting installation. The engine source SHA remains
`56d274eec3df1d02eefb52dbe5cc894ff4a5eabe066f483046462e4e021e9fa2`.

Read-only acceptance confirmed the actual configuration files and complete run
records matched the pre-upgrade snapshot. This profile still had no experiments
or configured API key/model. No synthetic settings or experiments were inserted
into the actual profile. Ordinary runtime/browser metadata may change during
startup; this is not a claim that every profile file was unchanged.

Receipts are `output/package-verification-0.1.9.json`,
`output/package-installation-0.1.9.json`, `output/engine-upgrade-0.1.9.json` and
`output/installed-engine-acceptance-0.1.9.json`.

After the isolated workflow and final verification, the normal installed app was
reopened and visually checked: version 0.1.9, local runtime ready, and the empty
new-experiment screen. It was left open without entering or saving any values.

## Installed training workflow

The installed 0.1.9 executable completed native folder selection, real metadata
inspection, UI submission, durable queue execution, CPU training, frozen
evaluation, result inspection and native report export in **16.161 seconds**.
The run succeeded with exit code 0. It fitted a scikit-learn logistic regression
model on synthetic data using the retained installed engine; final retraining was
disabled. Five model requests reached a deterministic localhost fixture, with
no external provider calls, renderer errors or fixture errors.

A test-only spawn wrapper restricted the sidecar, queue, uv and training
descendants to localhost. The exact `/bin/ps` executable was exempt solely for
read-only process identity inspection. The candidate confirmed external-address
network denial before transmission. Production source and the Chromium sandbox
were unchanged.

Installed engine/runtime file metadata and selected immutable file hashes were
unchanged. Final checks confirmed actual configuration and complete run records
were unchanged, all inspected QA profiles had no remaining processes, installed
and packaged signatures still passed, and the entire installed bundle matched
the verified package fingerprint above.

Receipts are `output/desktop-workflow-acceptance-0.1.9.json` and
`output/post-workflow-verification-0.1.9.json`; durable synthetic evidence is under
`output/qa/desktop-workflow-0.1.9/`. This verifies the software workflow, not
external-provider quality or biomedical model performance.

## Release boundaries

Dataset inspection does not establish file parseability, label correctness,
training/validation separation, scientific suitability or model performance.
Windows, Linux, Intel Mac and independent clean-hardware installation remain
unverified. Developer ID signing, notarization and signed updates still require
release credentials. Local ad-hoc signing is not Developer ID distribution.
