# BioMaker Desktop 0.1.12 acceptance

Target: macOS Apple silicon / arm64. Date: 2026-10-09.

Source implementation and development acceptance are complete. Packaged and
installed acceptance remain pending until recorded below. The installed 0.1.11
release remains separately documented in ACCEPTANCE-0.1.11.md.

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
- Native/package: 145 passed, zero skipped, `output/native-tests-0.1.12.tap`;
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

## Release boundaries

Package, signature, actual upgrade, original-profile migration, installed workflow
and normal native launch are still pending for this version. Public distribution
requires Developer ID signing and notarization credentials. Windows, Linux,
Intel Mac and independent hardware remain unverified. Synthetic software checks
do not establish medical model performance or clinical validity.
