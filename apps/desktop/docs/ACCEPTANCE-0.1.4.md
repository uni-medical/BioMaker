# BioMaker Desktop 0.1.4 acceptance

Date: 2026-10-08. Host: macOS, Apple silicon / arm64.

## Verified changes

- Existing experiment folders can be previewed and imported as read-only copies.
  Preview lists the original names, available content, file sizes, exclusions and
  invalid records. Sources are selected using a native folder dialog; the main
  process checks the resolved path against the user's folder grants.
- Imports copy only listed top-level regular record files. Source manifests are
  preserved separately, while the effective copy removes process ownership and
  workspace references. Source bytes and modification times remain unchanged;
  datasets, weights, private ground truth, environment files and subdirectories
  are excluded. Imported runs reject stop, cancel and requeue operations.
- Previewed file identities are checked again before and during copying. Changed
  sources fail with a rescan action. Background jobs record partial successes,
  duplicates and individual errors; restart preserves completed copies and marks
  interrupted work failed. Transient progress-read errors retain the job ID.
- Staged copies are validated before atomic publication. Nonfinite or deeply
  nested event data is rejected before it can poison the shared history reader
  at the next startup. Existing candidate-journal nonfinite handling is preserved.
- Imported experiment titles survive refresh and restart. Candidate code, logs
  and reports use the existing viewer and native export. Unknown completion is
  shown as a historical snapshot rather than proof that the original run stopped.
- Native service state distinguishes starting, ready, reconnecting and failed.
  Ready requires authenticated health. A genuine sidecar crash updates the UI and
  recovers automatically; older responses cannot overwrite the recovered state.
  Drafts and unsaved provider fields remain intact. Ordinary HTTP errors remain
  separate from a lost service connection.

## Source validation

102 WebUI tests, 56 desktop service/native-runtime tests and 42 unique native and
package tests passed, including the separately enabled real-sidecar integration
test. TypeScript and the production build passed.

Real development Electron passed the existing general acceptance, native-close
acceptance and the new history-import workflow. The import workflow verified
native folder selection, invalid/empty selection, changed-source rejection and
rescan, read-only controls, copy exclusions, code/log viewing, native report
export, duplicate skipping, draft preservation, restart, compact layout and
unchanged source hashes/mtimes. It installed no engine and started no experiment.

The native service-state and renderer lifecycle harnesses exercise genuine
sidecar termination and recovery, with controlled delays/failures to expose
startup and stale-response races. These instrumented source checks are distinct
from packaged-executable acceptance.

## Release acceptance

Package, installed-upgrade and full installed-workflow receipts will be recorded
after validating the final 0.1.4 artifacts. The source checks above do not by
themselves establish successful installation or engine execution.

## Boundaries

History import copies experiment records, not complete executable workspaces.
Limits are 100 experiments per preview, 128 top-level entries per experiment,
64 MiB per file and 256 MiB per batch; previews expire after ten minutes. Large
journals may be exported without inline candidate expansion.

Apple Developer ID signing/notarization, signed updates, independent clean-hardware
acceptance and other-platform installation remain pending. Synthetic acceptance
data establishes software behavior, not biomedical performance. The engine runs
generated code with the user's access; test-only network restrictions are not a
production security sandbox.
