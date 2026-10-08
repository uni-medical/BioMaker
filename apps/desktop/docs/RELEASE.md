# Desktop release and iteration policy

## Release gates

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
- Build from the dedicated desktop branch, preserving the research checkout.
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
  a missing or mismatched `manifest.applicationVersion`.
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

## Data and upgrades

The macOS app stores its own workspace, settings, run ledger and writable engine
environment under the user's Application Support directory. It does not take over
the research repository's existing `.env` or outputs. Engine code is versioned
separately from user data. Replacing the `.app` must not delete experiments.

Experiment drafts are stored in `experiment-drafts.json` under the native app data
directory, independently of the Python service. Preserve this library and its
active draft across upgrades. Valid legacy single-draft localStorage content
migrates once; retain the raw original for recovery. Malformed, unsupported or
conflicting persisted state must produce a recoverable error rather than silently
replace user data. Verify migration, restart and failed-save recovery again from
the packaged and installed build; a development pass does not establish upgrade
acceptance.

API credentials are stored in the application's private config file with owner-only
permissions, as required by the existing Python engine. They are not stored in
renderer localStorage, URLs, screenshots, or diagnostics. OS keychain integration
can replace this storage layer in a later compatibility-preserving release.

## Platform acceptance

| Target | Policy |
| --- | --- |
| macOS arm64 | Initial build and native acceptance target |
| macOS x64 | Build portable runtime on x64, then repeat packaged acceptance |
| Linux | Portable UI and shell; repeat dependency, process lifecycle, packaging and GPU acceptance |
| Windows | Shell supports Windows packaging; process identity and supervisor need Windows-specific acceptance before enabling training |

Do not describe configured build targets as tested platform support. CUDA, Apple
MPS, CPU and remote compute are different execution capabilities. This release
does not automatically configure a remote GPU server.

## Iteration priorities

1. Complete the install → provider → dataset → experiment → output workflow.
2. Exercise interrupted installs, corrupt/missing files, backend restarts, queue
   recovery and persisted preferences; fix reproducible defects before features.
3. Refine layout using screenshots of real empty, active, failed and completed
   states at multiple window sizes; verify contrast and keyboard focus.
4. Add migrations, diagnostics and release automation with explicit compatibility
   checks. Add automatic updates only with a configured, signed release feed.
5. Add remote engine connections and additional platforms against versioned
   contracts, without coupling the React UI to a particular process launcher.

Each iteration needs a concrete change, a recorded check and a versioned artifact.
Passing desktop checks is not evidence that a generated model is scientifically
correct or suitable for clinical use.

## Result and file contract in 0.1.11

The Results tab reads fixed evidence files from a registered experiment. Selection
is an explicit saved event or a separately labelled unique exact code match;
metrics are copied from that candidate. Final retraining is a separate conclusion
with its own receipt/event/configuration provenance. An unreadable event stream
cannot establish that an identity-free receipt belongs to the latest attempt.

Artifact-list revisions remain inexpensive refresh signals. Preview-save identity
uses a digest of the actual original bytes and the native service revision.
Downloads go to a private temporary file and replace the chosen destination only
after validation. Direct file exports do not require a preview; all artifact saves
share one app-wide in-flight guard and use the native destination picker.

Track final-source, packaged and installed evidence separately in
[ACCEPTANCE-0.1.11.md](ACCEPTANCE-0.1.11.md). A passing source suite does not complete
the package, signature, installation or retained-engine workflow gates.

For the frozen 0.1.11 build, 701 source checks, 32 development checks, 152 packaged
checks and four offline engine checks have passed. Package verification, the
actual upgrade with retained-profile inspection, installed synthetic CPU training,
final bundle/profile verification, QA process cleanup and normal native launch
have also passed on Apple silicon. The acceptance record retains the distinct
scope of each check and the remaining distribution and platform boundaries.

## Next workflow improvements

- Reduce first-read and actively changing history costs after validating the
  observation cache. Cache hits do not establish bounded IO for changed files.

## Multiple-draft implementation for 0.1.12

The source now includes a searchable native draft library with autosave, independent
new/reused/duplicated entries, rename and confirmed deletion. Draft names are
separate from experiment directory names. Drafts remain editable without the
sidecar; confirmed submission removes only its entry, while failures or uncertain
outcomes preserve it. Persistence errors expose retry and content-recovery actions,
and pending edits participate in native close protection.

The final release source is `6cd2f2d7f642f1317466ada6a69dfd9237d870ae`.
There are 745 passing source checks, 115 development checks, 168 packaged checks
and four offline engine checks. The final package was verified across DMG, ZIP
and installed application. The actual upgrade preserved the legacy draft exactly,
reused the existing engine, and retained configuration and complete run records.
Installed synthetic CPU training, Results/candidate navigation, report export,
final bundle/profile checks and normal native launch passed.

See [ACCEPTANCE-0.1.12.md](ACCEPTANCE-0.1.12.md) for receipts and limitations.
The previous 0.1.11 record remains separate. A synthetic workflow establishes
software behavior, not biomedical model performance or external-provider access.


## Activity and console contract for 0.1.13

Desktop stream routes read only registered sources, independently of the catalog.
Console windows are capped at 400 lines and 256 KiB; activity windows are capped at
200 valid JSON objects and 512 KiB. File identity, byte ranges, observation time,
missing-source state and incomplete/undecodable content accompany each window.
Earlier pages replace the visible window. Authenticated cursors bind the run,
source and byte offset, with bounded continuity anchors; they are not a full-file
integrity proof. A source change or service restart invalidates earlier cursors.
The existing shared list/detail catalog still scans history and is not covered by
these per-window IO bounds.

Log following pauses on upward navigation, selection or search. Background reads
do not replace a paused window. Search is literal, case-insensitive and limited to
the displayed window; highlighting stops at 1,000 matches. Copy uses that displayed
text. Earlier-page reads and latest-follow actions cannot apply stale responses
across navigation or service replacement. Hidden tabs retain reading state.

Full console export resolves the registered path in the service. It captures the
initial file length, permits appends after that boundary, and verifies the exact
bytes against a SHA-256 digest before atomic destination replacement. The export
limit is 256 MiB. It shares the app-wide native export lock with artifact saves;
no renderer path is accepted. Cancellation or validation failure preserves the
existing destination. Truncation, replacement, symbolic links and directory
substitutions are refused.

Activity uses real engine event labels and explicit execution/evaluation failure
fields, handles unknown JSON field types, and keys rows by source and byte offset.
The latest recorded event is not presented as the unique current phase of a
parallel experiment. Progress states recorded exploration candidates versus the
configured limit, without a training percentage or ETA.

Release validation must include tests/log-reader-smoke.mjs against the real
sidecar in an isolated profile, then the final packaged executable. Verify append,
paused selection/search, match navigation and copy scope, earlier pages,
rotation/truncation, same-run and cross-run delayed responses, source replacement,
service recovery, native full export/cancellation and shared save exclusion.
Inspect long-line, compact, keyboard, light/dark and bilingual states. Keep
synthetic lifecycle/timing injection distinct from real filesystem and native IPC
operations. tests/event-presentation.test.mjs covers historical malformed fields
and real execution/evaluation failure semantics.


The 0.1.13 frozen source is `7499b13287a492188a43a9f5c96b0a8344ed61f3`.
All 826 source, 89 development, 182 packaged and four offline engine checks passed.
Package verification, actual installation with independent engine setup, original
profile preservation, installed synthetic workflow and normal launch passed.
See [ACCEPTANCE-0.1.13.md](ACCEPTANCE-0.1.13.md) for exact receipts, source-bound
package checksums, known boundaries and preserved harness failures.

## Observation caching for 0.1.14

Individual run routes use an indexed ledger lookup. List and detail observations
may be reused only after rechecking their filesystem dependencies. Lifecycle
status, ownership and available actions must come from the current ledger record;
the cache must never make an experiment controllable or change its queue state.

The cache belongs to one service instance and retains bounded serialized
summaries. File replacement, in-place rewrite, truncation, deletion, manifest and
result changes, directory substitution and service restart require fresh reads.
Revision signals must also detect replacement or rewrites when size and mtime
are retained, so the renderer knows to request the new detail. Transient read
failures and changing observations must not become persistent cached fallbacks.

Run `tests/catalog-refresh-smoke.mjs` against the built source and packaged
executable in isolated synthetic profiles. Verify file changes reach the real UI,
restart and cross-run navigation preserve reading state, and no model request,
engine setup or experiment submission occurs. Keep performance measurements
separate: report cold versus repeated calls, fixture sizes and logical file bytes,
without interpreting instrumented endpoint timings as physical disk or UI latency.

First reads, changed files, oversized uncached summaries and cache eviction can
still require full history parsing. The cache payload limit is not a bound on
transient parser memory, total service memory or total request IO. Preserve the
existing event-count and last-5,000-valid-event catalog semantics.
