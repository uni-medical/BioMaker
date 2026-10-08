# BioMaker Desktop 0.1.4 acceptance

Date: 2026-10-08. Host: macOS, Apple silicon / arm64.
Packaged source: `43b3998ccddfba4e5580d65cad078688ce6cb6b3`, clean checkout.

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

The import recovery harness also passed transient progress-query failure with
preserved job ID, modal reopening, global interruption counts, missing-job
recovery and expired-preview rejection without a repeated submission.

## Release acceptance

The packaged executable passed the history-import workflow, native-close checks
and existing general Electron acceptance, all with isolated synthetic profiles
and a system-only PATH. No renderer errors were observed. Native notification
display was intercepted in QA; actual OS banner delivery is not established.

The final DMG was mounted read-only. Its application archive matched the tested
bundle, its version was 0.1.4 and its Applications shortcut resolved correctly.
Deep strict ad-hoc signature verification passed, including after the corrected
packaged acceptance. The QA ledger helper now uses Python `-B` with `-I`: isolated
mode ignores the environment's bytecode setting and otherwise writes cache files
inside the signed bundle. The initial unpacked QA copy was restored from the
unchanged DMG before retesting; the user's previous application was preserved.

The upgrade is installed at `/Applications/BioMaker.app`. The previous app is
backed up under `release/installed-backups/BioMaker-0.1.3-20261008-231613.app`.
The actual user profile started with its existing engine ready, without requesting
installation. Its configuration file state and complete run records matched the
pre-upgrade snapshot. No research histories or QA records were added to that
profile. The engine source SHA-256 remains
`56d274eec3df1d02eefb52dbe5cc894ff4a5eabe066f483046462e4e021e9fa2`.

## Installed desktop workflow

The installed 0.1.4 executable completed UI submission, durable queue execution,
actual CPU training, frozen evaluation, result inspection and native report
export in **15.888 seconds**. It reached `succeeded` with exit code 0 and zero
renderer errors. The engine fitted a scikit-learn classifier on synthetic data;
five model requests went to a deterministic localhost fixture. Final retraining
was disabled. This verifies software behavior, not external-provider quality,
multi-round biomedical search or scientific model performance.

The harness restricted the Python sidecar and training descendants to localhost;
only the exact `/bin/ps` executable was exempt for process identity inspection.
The candidate confirmed an external-address connection was denied with `EPERM`.
Installed engine/runtime fingerprints remained unchanged, and the QA profile's
processes were confirmed stopped afterward.

## Artifacts and evidence

| Artifact | SHA-256 |
| --- | --- |
| BioMaker-0.1.4-mac-arm64.dmg | `979ecbd43dee8a1c1e766e71913e071dc9356f58372ac8cadd08c0760b5274ae` |
| BioMaker-0.1.4-mac-arm64.zip | `0abd70509ee6878a8f428ac42eb919aac573400db1eb6e8e98b9ad8f2457c9ab` |

Machine-readable receipts under `apps/desktop/output/`:

- `engine-upgrade-0.1.4.json`
- `installed-engine-acceptance-0.1.4.json`
- `history-import-packaged-0.1.4/acceptance.json`
- `native-close-packaged-0.1.4/acceptance.json`
- `native-service-state/acceptance.json`
- `service-ui/acceptance.json`
- `history-import-recovery/acceptance.json`
- `package-installation-0.1.4.json`
- `playwright/electron-acceptance.json`
- `desktop-workflow-acceptance-0.1.4.json`
- `qa/desktop-workflow-0.1.4/` (durable fixture, logs, screenshots and outputs)

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
