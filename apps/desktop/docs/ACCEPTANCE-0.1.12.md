# BioMaker Desktop 0.1.12 acceptance

Target: macOS Apple silicon / arm64. Date: 2026-10-09.

Release source commit: `6cd2f2d7f642f1317466ada6a69dfd9237d870ae`.

745 source checks, 115 development checks, 168 packaged checks and four offline
engine checks passed. Package verification, the actual upgrade, exact legacy-draft
migration, retained-profile checks, installed synthetic training, final bundle
verification and normal native launch also passed on this Apple silicon Mac.

## Draft contract

Drafts are independent local documents. New creates a fresh default draft;
repeated New on a pristine untitled default draft reuses that empty editor.
Reusing an experiment creates a separate draft without overwriting other drafts.
The library supports search, Unicode display titles, duplicate, open and confirmed
delete. The display title is separate from the output directory name.

Each draft preserves task, dataset, template, all six options and reuse provenance.
Incomplete numeric inputs persist as null and reopen blank; negative/fractional
edits are preserved until submission validation. Confirmed submission consumes
only its draft. Refusal or an uncertain submission outcome retains it without an
automatic POST retry. Imported history still requires a new local dataset choice.

Native storage is independent of the Python service, uses private atomic JSON,
checks the expected revision and retains 128 mutation receipts for idempotent
retries. Limits are 100 drafts, 256 KiB per record and 8 MiB per document; no draft
is silently pruned or truncated. Existing legacy localStorage is migrated once,
with its exact raw content retained as a backup and its original key unchanged.
Damaged or unsupported files are not treated as an empty library or overwritten.

Saving is acknowledged explicitly. Failed or ambiguous acknowledgments retain
edits. Recovery as a copy reuses its mutation ID on retry; newer input survives.
Both normal and synchronous close saves check the acknowledged draft identity
and content before replacing the local revision. Settings and draft protection
share one beforeunload listener, with saving taking priority over discard.

## Source validation

- Shared WebUI: 102 passed, `output/webui-tests-0.1.12.xml`.
- Desktop service: 491 passed, `output/service-tests-0.1.12.xml`.
- Native/package: 145 passed, zero skipped, `output/native-final-0.1.12.tap`;
  includes 37 new native draft checks.
- Unified close protection: 7 passed, `output/close-protection-0.1.12.tap`.
- TypeScript and production build passed, `output/build-0.1.12.log`.

Total: 745 checks; subsets are not counted twice. Existing dependency deprecation
and bundle-size warnings remain separate from failures.

## Development acceptance

`output/drafts-0.1.12/acceptance.json` records 16 passing checks through real
Electron and the native store in an isolated profile, zero renderer errors and
zero forbidden requests. It covers migration, CRUD, independent reuse, restart,
late picker and same-path A-to-B-to-A inspection, delayed navigation, uncertain
update/recovery acknowledgments, identity conflict recovery, immediate close,
corrupt-store preservation, combined settings protection, incomplete budgets and
oversized-draft recovery. Failure timing is injected; the native file writes are
real. No experiment, external provider request or engine installation is made.

`output/development-regressions-0.1.12.json` records another 99 passing checks
across eight affected suites: reuse (8), submission (7), provider settings (15),
dataset inspection (14), runtime UI (9), provider check (14), general Electron
(22), history import (10). Their scopes retain their synthetic fixtures and
intercepted cases. All eight profile process checks found no remaining process.

The initial draft harness used unsupported dynamic import in Electron evaluation;
a later harness waited for a save indicator intentionally absent on run pages.
These harness issues were corrected before final passing acceptance. They were
not evidence of product failure or successful acceptance.

## Packaged acceptance

`output/packaged-acceptance-0.1.12.json` records 168 passing checks across 14
suites against the final frozen executable with isolated profiles and minimal
PATH. The suites are drafts (16), Results (21), preview/save (11), provider
settings (15), provider check (14), dataset inspection (14), submission protection
(7), native close (6), run organization (8), reuse (8), history import (10),
service recovery (7), runtime UI (9) and general Electron interactions (22).
Renderer-error and forbidden-request collections are empty, and all profile
processes exited. Synthetic readiness, timing faults and provider responses are
identified in their individual receipts. They are not scientific evaluations.

The draft visual receipt covers seven inspected screenshots, light/dark themes,
1360 by 857 and 960 by 680 windows, library, editor and rename dialog. No
horizontal overflow, obscured controls or unusable text was found. Other suites'
representative compact, dark and failure screenshots were inspected separately.

The Results suite had one timing failure: automatic refresh removed the retry
button before the test clicked it. The unchanged frozen script passed its exact
rerun. `output/result-overview-packaged-0.1.12-attempt1-failed/` retains the first
log/screenshot, and the shell aggregate records the prior attempt. Earlier
candidate-package evidence at source cb4728d is retained under
`output/pre-final-0.1.12-cb4728d/`; it does not replace final-source acceptance.
The final source additionally locks draft retry during experiment submission.

## Package verification

`output/package-verification-0.1.12.json` confirms clean-source provenance and
matching applications across the unpacked bundle, read-only mounted DMG and
extracted ZIP. Deep strict signature verification passed on all three copies.
The DMG contains the Applications shortcut. This is ad-hoc signing, not
Developer ID signing or Apple notarization.

The full application has 2,816 files and fingerprint
`f37bf6a5ada53561f82083fe10b5999f0ebd985ef25412417fe6aefb8c6fd068`.

| Artifact | SHA-256 |
| --- | --- |
| `release/BioMaker-0.1.12-mac-arm64.dmg` | `7dbacba4f35e236bac0f487befd8a0b93075042bd4af3f7c22ec83851414cb7d` |
| `release/BioMaker-0.1.12-mac-arm64.zip` | `01a22d7878c2057badf59bc9b159888375e2787039159f5dc90832e69e14df7a` |

## Actual upgrade and migration

`output/package-installation-0.1.12.json` records upgrading the real installed
application at `/Applications/BioMaker.app` from 0.1.11. Its previous copy is
retained at `release/installed-backups/BioMaker-0.1.11-20261009-050809.app`.

Before upgrading, a read-only inspection of the real 0.1.11 application captured
the legacy draft hash without creating test content. After upgrading,
`output/installed-draft-migration-0.1.12.json` confirms that the old localStorage
value, the native raw backup and the migrated content match. The original blank
draft became one native draft; no QA drafts were introduced into the real profile.

`output/installed-engine-acceptance-0.1.12.json` confirms that the existing engine
was reused and ready. Configuration file hashes and complete SQLite run rows
were identical; the real profile still had zero runs. API/model configuration
was absent and remained absent. No real-profile setting writes, engine install,
external-provider requests or imported QA fixtures were requested.

## Offline engine and installed workflow

`output/engine-health-packaged-0.1.12.json` records four passing checks in
108.557 seconds. A network-denied clone passed scientific imports/CPU self-test,
then correctly detected a removed MONAI dependency across restart. A real offline
locked reinstall repaired that clone, and a second download-free check passed.
Original engine and cache complete fingerprints remained identical. This test
used the final packaged service and bundled Python with `-I -B`.

`output/desktop-workflow-acceptance-0.1.12.json` records an installed-app workflow
in 15.591 seconds. The UI submitted a synthetic task through the real durable
queue, which progressed queued to running to succeeded. A CPU logistic-regression
candidate trained on 12 synthetic rows, ran its fixed evaluator, and produced
code, metrics and a report. Results selection, candidate navigation, report
export and execution-log viewing passed with no renderer errors. Final retraining
was explicitly disabled; the fixture metric is not a biomedical result.

This workflow used an isolated profile and deterministic loopback provider. Its
Python process tree was restricted to localhost, with an exact read-only
`/bin/ps` exception for process identity checks; production Chromium protection
remained enabled. There was no installed-source override or external-provider
call. The installed engine's metadata and immutable hashes stayed unchanged.
Durable evidence and screenshots are under `output/qa/desktop-workflow-0.1.12/`,
with symbolic links preserved in the copied QA home.

## Final verification and normal launch

`output/post-workflow-verification-0.1.12.json` confirms the real draft file,
configuration and complete user run records remained unchanged during isolated
QA. Installed and packaged deep strict signatures passed, and the complete
installed bundle matched the verified 2,816-file fingerprint. No QA processes
remained.

`output/native-launch-0.1.12.json` records normal native launch, an explicit
window raise, and accessibility/screenshot inspection. Version 0.1.12 showed the
runtime ready and the original empty draft. Opening the one-entry draft library
and returning to that draft worked without changing its saved bytes. A final
read-only profile/bundle check passed, and the user's application was left open.

## Release boundaries

Developer ID signing, notarization and signed updates require release credentials.
Windows, Linux, Intel Mac and independent hardware remain unverified. The tested
upgrade reuses an existing engine; it is not a separate clean-machine installation.
Synthetic software checks do not establish external-provider availability,
biomedical model performance or clinical validity.
