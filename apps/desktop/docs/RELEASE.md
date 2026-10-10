# Desktop release and iteration policy

## Release gates

- Run `tests/long-history-smoke.mjs` against the built source and packaged app.
  Use isolated large histories to verify cold startup discovery, exact counts,
  last-5,000-valid-object catalog observations, ledger-owned lifecycle/actions,
  event/result appends, bounded activity/console and honest Results read limits.
  Repeat a complete native relaunch with the same QA profile. Record timings as
  fixture-specific evidence, not a guarantee for arbitrary history sizes. Keep
  the native 30-second startup timeout unchanged.
- Run `tests/drafts-smoke.mjs` with the real native draft store and repeat against
  the packaged executable. Verify independent draft identities, autosave, search,
  rename, duplicate, delete, restart, legacy migration and edits without a live
  sidecar. Check new/reused drafts preserve existing entries, confirmed submission
  removes only its draft, and failed or uncertain submissions retain it. Exercise
  persistence failures and recovery, revision conflicts, and native close/quit
  protection for pending edits. Keep intercepted submission states separate from
  the installed experiment workflow and retain every QA profile's cleanup receipt.
- Run `tests/result-overview-smoke.mjs` with the real sidecar and repeat against
  the packaged executable. Verify recorded selection versus code matching,
  candidate identity and metric direction, conflicting/missing evidence,
  imported-snapshot limitations, retraining receipts and partial event streams.
  Include selected candidates beyond the visible-list limit, malformed optional
  fields, keyboard focus and light/dark compact layouts. Keep synthetic lifecycle
  or response states identified separately from real file parsing and service
  replacement.
- Run `tests/artifact-preview-smoke.mjs` through real Electron IPC and repeat
  against the packaged executable. Check original-byte preservation, changed
  content even when size and mtime are retained, native save cancellation,
  service replacement during reads/dialogs/downloads, existing-destination
  preservation and temporary-file cleanup. Verify the app-wide save barrier
  across closed previews and run navigation: overlapping requests must open no
  second dialog, make no download and queue no later overwrite. No model request,
  installation or experiment is needed for these checks.
- The result and preview service tests must reject source/ancestor symbolic links
  and directory replacements, ordinary-file substitutions, unsupported file types
  and bounded-read violations. Platforms without equivalent descriptor-relative
  no-follow reads must fail closed until their safe-open adapter is implemented.
- Run `tests/provider-settings-smoke.mjs` against the real sidecar and repeat
  against the packaged executable. Use only synthetic profiles and loopback
  providers. Verify changed-address key decisions, byte-preserving refusals,
  equivalent-address retention, stale-save conflicts, explicit review/reload,
  draft and unsaved-field preservation, restart recovery and native close
  protection. A successful settings save must make no provider call; any probe
  requires a separate explicit action.
- Run `tests/dataset-inspection-smoke.mjs` through the native bridge and real
  sidecar in an isolated profile, then repeat with the packaged executable.
  Verify granted and ungranted directories, confirmed-empty and partial results,
  symbolic-link boundaries, stale-response protection and responsive themes.
  Metadata inspection must not open dataset file contents, enqueue an experiment,
  install dependencies or send provider requests. Keep submission-time filesystem
  checks separate from scientific data validity and later queue-time availability.
- Run `tests/dataset-submission-smoke.mjs` for localized dataset refusals, stale
  preflight rejection, draft snapshot protection, duplicate submissions and
  uncertain outcomes after service restart. Its readiness and submission
  responses are simulated; all POSTs are intercepted and the real ledger must
  remain empty. Verify actual execution separately with the installed workflow.
- Run `tests/provider-check-smoke.mjs` against the real sidecar and a local model
  fixture, then repeat against the packaged executable. Verify explicit-only
  requests, draft/configuration preservation, credential reuse boundaries,
  editable-field and service-revision invalidation, failure feedback and retry.
  A returned model list, a saved configuration and a successful short generation
  probe are different results. Do not use real user credentials in QA.
- Run `tests/runtime-health-smoke.mjs` for runtime maintenance UI and admission
  feedback, reporting its synthetic backend responses explicitly. Separately run
  `tests/engine-health-smoke.py` with bundled Python `-I -B` inside a network-denying
  macOS sandbox. It clones the installed engine and package cache with APFS,
  removes a real dependency only in the clone, checks failure persistence, then
  performs a real offline locked reinstall and verifies recovery. Original engine
  and cache fingerprints must remain identical. Never damage the actual profile.
- Build all distributions from the same clean, reviewed product commit.
- Pin JS dependencies with `package-lock.json`, record portable runtime provenance,
  and include upstream license notices in the bundle.
- Pass TypeScript, native boundary tests, desktop service tests and the existing
  WebUI suite. Run a real Electron session against the real sidecar in isolated data.
- Check first launch, missing provider/engine, dataset selection, saved settings,
  queue state, reconnect, failure recovery, keyboard navigation, themes and layout.
- Test the packaged `.app`, not only the Vite development window. Verify that it
  starts with a minimal PATH and does not depend on the source checkout or system Python.
- Run `tests/native-close-smoke.mjs` against the packaged executable to verify
  unsaved-settings close/quit cancellation, explicit discard, in-flight saves,
  failed-save recovery, and continued service availability after a canceled quit.
- Run `tests/history-import-smoke.mjs` against the packaged executable for native
  directory selection, preview, changed-source rejection, read-only copies,
  candidates/logs/export, duplicate handling, draft preservation and restart.
  Sources and app profiles must be synthetic and isolated from research records.
- Run `tests/native-service-state-smoke.mjs` and `tests/service-ui-smoke.mjs` for
  authenticated readiness, genuine sidecar crash/recovery and stale-response
  protection. Their instrumented startup checks run from the built source;
  report that scope separately from packaged-executable acceptance.
- Run `tests/service-recovery-smoke.mjs` against the packaged executable. Induce
  real constructor and restart failures only within an isolated QA profile;
  verify the native window survives, manual retry preserves drafts and unsaved
  settings, and diagnostics export works without a live service. Verify cancel,
  save failures, sensitive sentinel exclusion and original configuration bytes.
- Prepare the runtime after updating the application version. Packaging rejects
  a missing or mismatched `manifest.applicationVersion`, an altered engine
  checksum, or bundled source that differs from the current repository.
- On macOS, verify native dependencies with `tests/native-portability-smoke.py`
  against an installed engine. This denies Homebrew/MacPorts library paths and
  tests an isolated copy; it supplements, rather than replaces, separate-hardware
  acceptance. Use `tests/queue-lifecycle-smoke.py` for no-provider quit/reconnect/stop
  verification through the real supervisor. Run `tests/engine-workflow-smoke.py
  --engine <installed-engine>` for a deterministic local-model fixture through
  the actual split/evaluator/candidate/artifact workflow; keep that result distinct
  from medical or model-provider performance.
- On macOS, `tests/desktop-workflow-smoke.py --installed-receipt <receipt.json>
  --expected-version <version> --report <output.json>` adds UI submission, durable
  queue execution, result inspection and native export to the installed-engine
  check. Run it after the actual application profile has initialized the new
  runtime. It uses synthetic data, a local model fixture and a test-only network
  sandbox, with an exact `/bin/ps` exception for process identity inspection.
- Run `tests/run-organization-smoke.mjs` and `tests/experiment-reuse-smoke.mjs`
  against the packaged executable. Verify revision conflicts, delayed refreshes,
  archive/restore, restart, independent draft creation, existing draft preservation
  and original file/ledger preservation. QA uses isolated synthetic profiles and
  makes no provider call.
- After packaged acceptance, run `npm run verify:release`. It verifies deep strict
  signatures, clean-source provenance, version and complete application equality
  across the unpacked bundle, read-only DMG and extracted ZIP, then writes checksums.
  Never execute bundled Python without `-B`; isolated `-I` ignores the environment's
  bytecode setting and can otherwise mutate the signed application during QA.
- For a public release, apply Developer ID signing and Apple notarization. Keep
  signing credentials in the release environment and never in source files.

## Upgrade and evidence contract

Preserve user drafts, provider settings, run ledgers, metadata and artifact files.
Use isolated profiles for fault injection. Runtime maintenance and experiment
admission share one lock; active or queued experiments prevent maintenance.
Compatible UI upgrades may reuse the installed engine. Changed source or locks
require a separate engine environment while older runs retain their original path.

Persisted lifecycle, ownership and actions come from the run ledger. Current
catalog-tail observations are separate evidence. A terminal event absent from the
latest 5,000 valid objects does not rewrite a saved lifecycle assessment. Result
views show the recorded selected candidate and ranking metadata; they must not
select a winner again from the displayed primary metric.

Keep release receipts, screenshots, source hashes, package hashes and environment
provenance outside the product repository. A prior receipt does not establish
acceptance of a later version. Distinguish source tests, native UI checks, packaged
checks, installed engine checks and scientific model evaluation.

## Platform support

The initial distribution target is macOS on Apple silicon. Native dependencies,
process handling, installation, upgrade, packaging and signing must be verified
on each additional platform before publishing support claims. Public macOS
releases require Developer ID signing and notarization.
