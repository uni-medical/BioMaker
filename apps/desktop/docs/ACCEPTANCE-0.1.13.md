# BioMaker Desktop 0.1.13 acceptance

Target: macOS Apple silicon / arm64. Date: 2026-10-09.

Release source: `7499b13287a492188a43a9f5c96b0a8344ed61f3`.

826 source checks, 89 development checks, 182 packaged checks and four offline
engine checks passed. Package equality/signatures, the actual upgrade, isolated
engine setup, retained-profile verification, installed synthetic workflow and
normal native launch passed on this Apple silicon Mac.

## Activity and log contract

The log reader supports explicit follow/pause, upward-navigation and selection
protection, literal case-insensitive search with match navigation, line wrapping,
current-window copy, earlier pages and native full-log export. Hidden tabs retain
reading state. Paused windows survive new output, read errors, missing sources
and service restart. A late page response cannot overwrite a newer navigation or
follow action. Snapshots carry their run ownership into the first render after a
run switch, including completed experiments whose reader starts paused.

Console windows are at most 400 lines / 256 KiB. Activity windows are at most 200
valid complete JSON objects / 512 KiB. Search highlights at most 1,000 matches and
states that limit. Pages replace rather than concatenate output. Registered
sources are opened through pinned directory descriptors; cursors bind run,
source, offset, size, metadata and small continuity anchors. They are not a
full-history digest. Existing experiment list/detail scans are outside the new
stream-reader IO bounds.

Full exports capture the initial length and original bytes, including invalid
UTF-8. Appends after that length are allowed. Length and SHA-256 must match before
atomic destination replacement. The limit is 256 MiB. Console and artifact saves
share the native in-flight barrier. Source substitution, truncation, service
replacement, cancellation and failures preserve an existing destination.

Activity names match actual engine records; execution exceptions and buggy
candidate evaluations receive failure markers. Historical unknown field types
and prototype-like event names remain inspectable. The latest record is labelled
as an observation, not a uniquely inferred phase of parallel training. Progress
is recorded exploration candidates against the configured limit.

The shared historical catalog now tolerates non-string event names. The legacy
event/detail API represents non-finite numbers as explicit diagnostic strings,
preserving failure fields and source bytes. The new strict activity reader skips
non-standard JSON records and reports skipped/incomplete content.

## Source and development checks

Source receipts are under `output/`: WebUI 106, desktop service 552, native/package
157, close protection 7 and event presentation 4 (826 total). The new stream suite
contributes 61 and native console export contributes 12; subsets are not counted
twice. TypeScript and production build are also required. Existing dependency
and bundle-size warnings are recorded separately from test failures.

Development checks: log reader 14, artifact preview 11, history import 10,
Results 21, general Electron 22, service UI 5 and native service state 6 (89 total).
The new reader uses real files, sidecar reads, native save and service replacement
inside isolated profiles; lifecycle display, failure/timing and clipboard capture
are instrumented. No real provider request or experiment is performed by these
UI checks. Light/dark and compact reader screenshots are retained.

The first two reader attempts failed in the harness: an Electron callback used
its module argument as test configuration, and an exact text locator ignored a
nested retry button. Both attempts are retained under
`output/log-reader-0.1.13-attempt1-failed/` and `...-attempt2-failed/`.
They were corrected before passing acceptance; no product success is inferred
from failed attempts.

## Distribution boundaries

Packages use ad-hoc signatures. Developer ID signing and Apple notarization
remain required for a public release. Only Apple silicon macOS is accepted here;
configured Windows/Linux/x64 build targets are not verified platform support.
Provider credentials remain unconfigured in the actual profile. Synthetic CPU
workflow success is software evidence, not biomedical performance evidence.

This release changes shared engine source. The application selected and prepared
an independent environment through the native settings UI; the previous engine
source, receipts and interpreter remained unchanged. Both environments are kept.


## Packaged acceptance

`output/packaged-acceptance-0.1.13.json` aggregates 182 checks across 15 suites.
The native group has 49, UI has 82 and service group has 51. All passed on the
first frozen-package attempt, with empty renderer-error collections and no
remaining owned profile processes. The UI review checks canonical profile paths
as well. Its 22 screenshots and the native group's 14 screenshots were inspected.
A previous success toast can briefly coexist with a later conflict notice, and
an existing toast retains its language until it expires; these transient states
are recorded in the UI review, not hidden as failures or rewritten evidence.

The reader suite runs through real sidecar/files/native IPC with isolated
synthetic histories. It verifies bounded text, follow/scroll/selection stability,
search shortcuts and copy, hidden tabs, earlier-page races, rotation/truncation,
missing/read-failure retention, long-line endings, original invalid UTF-8 export,
service replacement, held cross-run responses, malformed activity fields,
explicit failure markers and compact themes.

## Package and upgrade

`output/package-verification-0.1.13.json` confirms clean source provenance, deep
strict signatures, the DMG Applications shortcut, and complete equality between
the unpacked application, read-only mounted DMG and extracted ZIP.

The application has 2,817 files and fingerprint
`ff18748f88a0c79692564c729c60598a488b0f6342942f32565267898dd056a3`.

| Artifact | SHA-256 |
| --- | --- |
| `release/BioMaker-0.1.13-mac-arm64.dmg` | `2153546a3b02a010c31011ad5899f64ea4ebdbecd6be72453e45e1ec2414f184` |
| `release/BioMaker-0.1.13-mac-arm64.zip` | `b2c742d2e5db05c3e0e2ff586f653f8334c53b0f2741ee24a55b9e7e71097791` |

The actual `/Applications/BioMaker.app` was upgraded from 0.1.12. The old bundle
is retained at `release/installed-backups/BioMaker-0.1.12-20261009-054920.app`.
Native UI setup installed the locked dependencies in the new isolated engine,
then reported successful dependency and fixed CPU checks.

The new engine is `engine-0e32352cb38aa727873ec64a1550067ded4e99e9075ab1358988d488341f9d4c`
under the actual profile's `engines/` directory. The prior
`engine-26f732a350d23642f2d0f36189451dcee9a5619893c953e123e2aec9ab6ee2c9`
remains available. Bundled engine source SHA-256 is
`65a95b292536f573f833b7abea418d1311c89890491c63bdb1cc13d89c52c18c`.

`output/installed-engine-acceptance-0.1.13.json` and
`output/post-workflow-verification-0.1.13.json` verify unchanged configuration,
preferences, draft bytes and complete SQLite run rows (zero actual user runs).
The original draft file retains SHA-256
`e4b3f05ea91babc4acc1ae81b7093c3bca0fe21af83208dcc675ee0ebe9948f7`.
The installed full bundle still equals the verified package after execution.

The first read-only installed harness attempted its API read before readiness.
Its preserved failure log is `output/installed-readonly-0.1.13-attempt1-failed.log`;
an explicit readiness/status loop fixed the harness before passing. No profile
mutation, source change or package rebuild was used to resolve it.

## Installed workflow and offline recovery

`output/desktop-workflow-acceptance-0.1.13.json` passed in 24.091 seconds. The actual
installed application and newly installed engine performed UI submission,
durable queue execution, fixed CPU training, result/candidate navigation, native
report export and log inspection. The local fixture answered five model stages;
no external provider was called. An inherited test-only network sandbox denies
nonlocal connections. The full QA home is retained with symlinks preserved at
`output/qa/desktop-workflow-0.1.13/`, including its receipt and seven screenshots.
This is synthetic software acceptance; final retraining is disabled.

`output/engine-health-packaged-0.1.13.json` records four passed offline checks in
109.202 seconds. The packaged service detected a dependency removed only from an
APFS clone, retained failure across restart, repaired it with locked offline
reinstallation and passed a second check. Original engine (29,582 files) and
cache (29,520 files) fingerprints match exactly. The clone is retained, its
receipt is copied to `output/qa/engine-health-0.1.13/`, and owned process cleanup
is recorded separately.

Normal CUA launch verified version 0.1.13, the saved original draft and local
runtime ready after raising the window. The app is left open in that draft;
model credentials and model selection remain unconfigured. The goal remains
active for further iteration, including reducing catalog scans for long histories.
