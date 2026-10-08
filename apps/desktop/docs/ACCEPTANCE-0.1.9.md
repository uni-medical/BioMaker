# BioMaker Desktop 0.1.9 acceptance

Date: 2026-10-09. Target: macOS, Apple silicon / arm64.

This iteration adds a local dataset overview and fresh desktop admission checks.
Source and development metadata acceptance have passed. Release artifact and
installation acceptance will be recorded after packaging.

## Dataset behavior

- New drafts show ordinary-file counts, logical sizes, bounded file-type counts
  and up to ten top-level entries. Compound extensions such as `.nii.gz` retain
  their meaning. File counts are not sample counts.
- Native code replaces renderer-supplied grants with saved folder grants and the
  application workspace. Invalid saved grants are ignored. Potentially blocking
  resolution and directory operations run in a separate Python worker.
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

## Release boundaries

Dataset inspection does not establish file parseability, label correctness,
training/validation separation, scientific suitability or model performance.
Windows, Linux, Intel Mac and independent clean-hardware installation remain
unverified. Developer ID signing, notarization and signed updates still require
release credentials. Local ad-hoc signing is not Developer ID distribution.
