# BioMaker Desktop 0.1.13 acceptance

Target: macOS Apple silicon / arm64. Date: 2026-10-09.

This document accompanies the source freeze. Package, upgrade and installed
workflow acceptance are recorded below once those gates finish; source checks
alone do not establish installable application acceptance.

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

This release changes shared engine source. The new application must select an
independent engine environment and retain the old environment without modifying
its source, receipt or interpreter. Draft/configuration/ledger preservation and
the new environment's real workflow require installed acceptance.
