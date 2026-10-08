# BioMaker Desktop 0.1.11 acceptance — draft

Date: 2026-10-09. Target: macOS, Apple silicon / arm64.
Release source commit: **pending source freeze**.

This draft records implemented behavior, 701 passing source checks and 32 passing
development acceptance checks. It is not a completed release acceptance:
packaged execution, package verification, installation and installed workflow
must still be recorded against the frozen 0.1.11 source and artifacts below.

## Result contract

- Completed experiments initially open Results; active experiments initially open
  Activity. Reports, saved solution code, candidate metrics and retraining receipts
  have direct file actions, with navigation to the identified candidate.
- The desktop result endpoint reads fixed files from an already registered run.
  It uses bounded ordinary-file reads, pinned directories and before/after source
  identity checks. Changing snapshots suppress cross-file conclusions.
- Explicit `step_saved` selection is checked against the journal and saved code.
  A unique exact code match without explicit selection has a separate status.
  The overview does not rank TSV metrics or interpret generated report prose.
- Candidate metrics retain their recorded optimization direction. Missing,
  ambiguous, malformed and conflicting records remain explicit. Candidate
  navigation keeps an identified node accessible beyond the initial list window.
- Final retraining is interpreted independently from candidate selection.
  A receipt, events without a receipt, explicit original configuration and
  unavailable evidence produce different states. A present but unreadable event
  stream cannot confirm that an old receipt describes the current attempt.
- Imported history is a local file snapshot. The current history importer does
  not copy model weights, nested output directories or the final-retraining
  receipt; missing copied evidence is not evidence that the original run failed.

## Preview and save contract

- Native preview requests identify a registered run and top-level artifact name.
  The service returns `{name, content, sha256, size}` from at most 500,000 bytes
  of strictly decoded UTF-8. The native bridge validates the bounded response,
  content length and digest before adding `runId` and `serviceRevision`.
- Raw bytes, including UTF-8 BOM, CRLF and journal `NaN`/`Infinity` text, retain
  their original representation. A preview does not execute or rewrite content.
- The service pins every absolute directory component without following links.
  Platforms missing the required safe-open primitives reject preview access;
  no weaker filesystem fallback is silently substituted.
- Saving a preview supplies its digest and service revision. The authenticated
  artifact download is streamed to a private temporary file; only matching bytes
  and an unchanged ready service may replace the native-selected destination.
  An error preserves an existing target, with no automatic retry as a newer file.
- Direct file-list exports remain available for larger and nontext artifacts.
  Both export paths use the native save dialog and one application-wide in-flight
  guard. A concurrent export returns `[artifact:busy]` before another destination
  prompt or download and is not queued for later execution.
- The UI retains an old preview with an explicit stale notice and refresh action
  when its file or service changes. Closing a preview does not cancel an approved
  export or release the native save guard.

## Source validation — passed

The completed source receipts record **701 passed checks: 102 shared WebUI +
491 desktop service + 108 native/package checks**, with no failures, errors or
skips. The separate result-module receipt covers a subset of the service suite.

| Check | Observed receipt | Scope/status |
| --- | --- | --- |
| Shared WebUI | `output/webui-tests-0.1.11.xml` | 102 passed; no failures/errors/skips |
| Desktop service | `output/service-tests-0.1.11.xml` | 491 passed; no failures/errors/skips |
| Result module | `output/result-overview-tests-0.1.11.xml` | 114 passed; included in the 491 service checks |
| Native/package suite | `output/native-tests-0.1.11.tap` | 108 passed with runtime enabled, no failures/skips |
| Artifact helper and authenticated route | Included in the service receipt | 45 passed, including ancestor-link protection; a subset of the 491 service checks |
| Native artifact module | Included in the native receipt | 17 passed, including export concurrency; a subset of the 108 native checks |
| TypeScript/production build | Completed integration validation | Passed |

Do not sum the targeted subsets into the complete service/native counts. The
existing Starlette/httpx deprecation and Vite chunk-size warnings do not indicate
a failed check. These source checks do not establish packaged execution.

## Development acceptance — passed

The two completed development receipts record **32 passed checks: 21 result
overview + 11 artifact preview/save checks**.

`output/result-overview-0.1.11/acceptance.json` records **21 passed checks**, no
renderer errors and no forbidden requests. It covers real Electron,
sidecar evidence parsing, history import, native preview/save and one isolated
service restart. Source records are synthetic; lifecycle/snapshot states and one
HTTP failure are intercepted. It makes no provider, engine-install or experiment
request. The checked cases include selected-candidate navigation beyond 1,500
nodes, malformed optional fields, conflicting and unavailable evidence, stale
responses, keyboard navigation, and light, dark and compact layouts. The result
screenshots have been reviewed.

`output/artifact-preview-0.1.11/acceptance.json` records **11 passed checks** with
development Electron, no renderer errors and no forbidden native operations. It
covers real native preview/save IPC, exact-byte preservation, size/text/path
refusals, changed bytes with unchanged size/mtime, cross-run export concurrency,
two isolated service restarts, stale responses, large direct exports and temporary
file cleanup. The compact dark screenshot was captured after changing the actual
UI theme and waiting for its applied style; the final screenshot was visually
reviewed.

Earlier harness failures remain in the artifact output directory: the initial
manifest fixture omitted `data_dir`, and two selectors matched multiple elements
in the service and theme flows. Those harness issues were corrected before the
passing run. `pre-dark-ui-acceptance.json` is superseded because its supposed dark
screenshot had not actually switched theme. The `initial-*` captures are retained
as history and are not the final acceptance receipt.

Existing workflow regressions against the packaged application remain **pending**.

## Packaged acceptance — pending

No 0.1.11 packaged-executable acceptance is claimed in this draft. After source
freeze and packaging, repeat the new result/preview suites and the required
existing workflow suites with isolated profiles and a system-only PATH. Record
the executable, suite receipts, renderer errors, forbidden operations, reviewed
screenshots and confirmation that test processes exited.

## Package artifacts and verification — pending

- Frozen-source manifest and application version: pending.
- DMG/ZIP production and SHA-256: pending.
- Read-only DMG mount, Applications shortcut and complete bundle equality: pending.
- Deep strict signatures on unpacked, DMG and ZIP applications: pending.
- `npm run verify:release` receipt: pending.

Do not reuse 0.1.10 hashes or verification receipts as 0.1.11 evidence.

## Installation and retained data — pending

Actual installation/upgrade, preservation of the previous application, configured
profile and complete run rows, compatible-engine reuse, and a normal native launch
are pending for 0.1.11. Record exact before/after checks and paths when performed.
Do not insert synthetic records or settings into the real user profile.

## Engine repair and installed workflow — pending

Repeat the required offline cloned-engine verification/repair and installed
synthetic training workflow after packaging. Record their own receipts, actual
engine/runtime preservation checks and process cleanup. UI fixture results do
not establish that a packaged or installed engine executes training successfully.

## Release boundaries

Deterministic local records and provider fixtures establish software behavior,
not external-provider availability or biomedical model performance. The scientific
engine and shared CLI/WebUI are not redesigned by the result/preview feature.
Windows, Linux, Intel Mac and independent clean-hardware installation remain
unverified. Developer ID signing, notarization and signed updates require release
credentials; local ad-hoc signing does not establish Developer ID distribution.
