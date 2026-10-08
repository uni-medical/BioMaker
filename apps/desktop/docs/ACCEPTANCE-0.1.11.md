# BioMaker Desktop 0.1.11 acceptance

Date: 2026-10-09. Target: macOS, Apple silicon / arm64.
Release source commit: `24a59333cf1c654d12285a591d790a0ddffd8b7d`.

This record confirms 701 passing source checks, 32 passing development checks,
152 passing packaged checks and four passing offline engine checks. Package
verification, the actual upgrade, retained-profile inspection, installed synthetic
training, final file verification and normal native launch have passed on this
Apple silicon Mac. The release boundaries below remain explicit.

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

The packaged regression results are recorded separately below.

## Packaged acceptance — passed

`output/packaged-acceptance-0.1.11.json` records **152 passed checks** against
`release/mac-arm64/BioMaker.app/Contents/MacOS/BioMaker`, built from the frozen
source commit above. The tests use isolated profiles and a system-only PATH.

| Suite | Passed | Receipt |
| --- | ---: | --- |
| Result overview | 21 | `output/result-overview-packaged-0.1.11/acceptance.json` |
| Artifact preview/save | 11 | `output/artifact-preview-packaged-0.1.11/acceptance.json` |
| Provider settings | 15 | `output/provider-settings-packaged-0.1.11/acceptance.json` |
| Provider response check | 14 | `output/provider-check-packaged-0.1.11/acceptance.json` |
| Dataset inspection | 14 | `output/dataset-inspection-packaged-0.1.11/acceptance.json` |
| Dataset submission protection | 7 | `output/dataset-submission-packaged-0.1.11/acceptance.json` |
| Native close lifecycle | 6 | `output/native-close-packaged-0.1.11/acceptance.json` |
| Run organization | 8 | `output/run-organization-packaged-0.1.11/acceptance.json` |
| Experiment reuse | 8 | `output/experiment-reuse-packaged-0.1.11/acceptance.json` |
| History import | 10 | `output/history-import-packaged-0.1.11/acceptance.json` |
| Service recovery | 7 | `output/service-recovery-packaged-0.1.11/acceptance.json` |
| Runtime health UI | 9 | `output/runtime-health-ui-packaged-0.1.11/acceptance.json` |
| General Electron interactions | 22 | `output/electron-packaged-0.1.11/electron-acceptance.json` |

Reported renderer-error and forbidden-request collections are empty. The saved
screenshots were reviewed, including light, dark and compact states, and the
combined receipt records no remaining QA process IDs. These suites use synthetic
records, deterministic loopback providers and explicitly intercepted cases;
dataset-submission fixtures do not establish that a real experiment executed.

## Package artifacts and verification — passed

`output/package-verification-0.1.11.json` records version 0.1.11 and a clean
frozen-source manifest. `output/verify-release-0.1.11.log` records a passing
`npm run verify:release` invocation.

The DMG was mounted read-only and its Applications shortcut verified. Both the
mounted DMG and extracted ZIP application match the complete unpacked bundle:
2,816 files, bundle fingerprint
`81a12cc3337e9199ccb5d3307f1616ba7b3ccf06f31cfb037aa98da9b8ed66c2`.
Deep strict signature verification passed on all three application copies.
This does not establish Developer ID signing or notarization.

| Artifact | SHA-256 |
| --- | --- |
| `release/BioMaker-0.1.11-mac-arm64.dmg` | `c0caa3886f3849055aa1be3bdccdce57d0a4437a57f6012963daec298ad40e2f` |
| `release/BioMaker-0.1.11-mac-arm64.zip` | `c54b2e6a21f7a1912b8da7dcdef0305e7c7950881a6d3414588e974ea5d468d7` |

## Installation and retained data — passed

`output/package-installation-0.1.11.json` records the actual upgrade from 0.1.10
to 0.1.11 at `/Applications/BioMaker.app`. The previous app was preserved at
`release/installed-backups/BioMaker-0.1.10-20261009-041446.app`. The pre-upgrade
profile snapshot is `output/profile-before-upgrade-0.1.11.json`.

`output/installed-engine-acceptance-0.1.11.json` records a read-only check of the
installed application with the existing user profile and a system-only PATH.
Configuration file hashes and complete run records remained identical; the real
profile had zero runs before and after. The compatible engine was reused and
ready, with no installation requested. API key and model configuration were
absent and remained absent. No settings, imported fixtures or provider requests
were introduced, and no renderer errors were recorded.

## Offline engine verification and repair — passed

`output/engine-health-packaged-0.1.11.json` records **four passed checks in
125.938 seconds** using the packaged service, a cloned engine and cloned cache,
with outbound networking denied:

1. A healthy engine completed real scientific-package imports and a fixed CPU
   self-test without invoking the package manager.
2. Removing MONAI from the clone produced a verification failure, and restarting
   did not incorrectly mark that damaged engine ready.
3. A forced offline reinstall restored the locked packages and native libraries,
   then passed the full engine verification.
4. The repaired engine passed a second download-free health check.

The source engine's 34,295 files and source cache's 29,507 files retained their
complete before/after fingerprints. Damage and repair were confined to the clones.
The matching detailed receipt is `output/qa/engine-health-0.1.11/acceptance.json`.

## Installed synthetic training workflow — passed

`output/desktop-workflow-acceptance-0.1.11.json` records a completed workflow in
**15.866 seconds** from `/Applications/BioMaker.app/Contents/MacOS/BioMaker`.
It used an isolated synthetic profile, the compatible installed engine and a
deterministic loopback provider. Production sources were not overridden, and no
new engine installation or external-provider call occurred.

The real queue progressed through queued, running and succeeded. A CPU
`sklearn.linear_model.LogisticRegression` candidate trained on 12 synthetic rows,
completed the fixed evaluator and produced solution code, metrics and a report.
The Results UI confirmed recorded selection from `step_saved`, a matching saved
solution, the fixture's recorded metric of 1.0 and explicitly disabled final
retraining. Candidate navigation, report export and execution-log viewing passed
without renderer errors. This synthetic score is a software fixture outcome,
not a biomedical benchmark result.

The QA Python process tree was restricted to localhost; the candidate confirmed
that an external connection was denied. A test-only exception allowed the exact
`/bin/ps` executable for read-only process identity checks; Electron retained its
production Chromium sandbox. Installed engine metadata and immutable source
hashes remained unchanged. Evidence is retained under
`output/qa/desktop-workflow-0.1.11/`, with symbolic links preserved.

## Final verification and native launch — passed

`output/post-workflow-verification-0.1.11.json` confirms unchanged real-profile
configuration files and complete run records, with zero user runs. Installed and
packaged signatures passed verification, and the entire installed application
matched the verified 2,816-file package fingerprint above. No QA processes remained.

`output/native-launch-0.1.11.json` records a normal native launch and window raise,
checked through the accessibility tree and screenshot. Version 0.1.11 showed the
local runtime ready and an empty new experiment. A subsequent read-only profile
check again confirmed unchanged configuration and run records. The normal user
application was left open.

## Release boundaries

Deterministic local records and provider fixtures establish software behavior,
not external-provider availability or biomedical model performance. The scientific
engine and shared CLI/WebUI are not redesigned by the result/preview feature.
Windows, Linux, Intel Mac and independent clean-hardware installation remain
unverified. Developer ID signing, notarization and signed updates require release
credentials; local ad-hoc signing does not establish Developer ID distribution.
