# BioMaker Desktop 0.1.10 acceptance

Date: 2026-10-09. Target: macOS, Apple silicon / arm64.
Release source commit: `9379b3cfd55ec9f76658d07d59464592227eaa1a`.

This iteration fixes saving a changed provider address while retaining an omitted
old API key. It also adds conflict-aware desktop settings saves. Source checks,
development and packaged acceptance, offline repair, artifact verification,
installation and the installed training workflow passed.

## Settings contract

- Desktop `GET /api/env` and `PUT /api/env` include a `settings_revision` generated
  with a per-service random HMAC key. Tokens are opaque and change after service
  replacement. The snapshot and token describe the same captured configuration.
- Desktop saves check `expected_revision` when supplied. The renderer sends the
  revision for its reviewed settings baseline. Existing callers that omit it
  retain the provider-switch guard but do not receive stale-edit protection.
- A changed effective provider address cannot retain an omitted saved API key.
  The caller must supply a replacement key or explicitly clear the key. Refused
  saves preserve the whole previous configuration, including optional settings.
- Cooperating desktop writers serialize the read, revision check, provider-key
  decision, atomic replacement and response snapshot. Advisory locks cannot
  prevent an unrelated editor from writing during this transaction.
- Save conflicts preserve form values. Reviewing saved public fields does not
  reveal secrets or save edits. Keeping entered values requires a separate save;
  loading saved settings explicitly replaces the unsaved input.
- Native transport failures and unreadable successful save responses have an
  explicit uncertain outcome. The interface requires reviewing saved settings
  before retrying. Explicit HTTP refusals retain their existing error codes.
- Saving and reviewing settings do not contact providers, install dependencies
  or create experiments. Provider response tests remain explicit actions.

## Source validation

**521 source tests passed:** 102 WebUI, 328 desktop service/native-runtime and
91 native/package tests, with no skips in the native suite. This includes 60 new
service settings cases and six native uncertain-save cases. TypeScript checks
and the production build passed. Existing Starlette/httpx deprecation and Vite
chunk-size warnings remain nonblocking.

The settings cases cover refused switch byte preservation, replacement/removal,
equivalent addresses, stale edits, actual adapter restart, competing threads and
processes, bounded lock contention, atomic write failure, request validation,
unsafe storage, masked responses and prevention of old-key reuse after refusal.
Native cases distinguish explicit HTTP refusals from uncertain transport and
successful-response read failures without automatic retries or raw private errors.

Receipts are `output/webui-tests-0.1.10.xml`,
`output/service-tests-0.1.10.xml`, `output/provider-settings-tests-0.1.10.xml` and
`output/native-tests-0.1.10.tap`.

## Development acceptance

The real Electron/native-bridge/sidecar provider-settings suite passed **15/15
checks**, using an isolated synthetic profile. It made 37 settings/probe bridge
operations and exactly one explicit local-provider generation request. It verified
address/key switching, preserved optional keys, real save conflicts, all five
unsaved fields, explicit keep/load decisions, native close protection, committed
but unconfirmed saves, two real sidecar restarts, and delayed-read focus retention.
No experiment, engine operation or external-provider request occurred. Five
screenshots were reviewed across light/dark themes and a 960-by-680 window.
Evidence is under `output/provider-settings-0.1.10/`.

The existing provider-response suite passed **14/14**, native close protection
passed **6/6**, and submission guards passed **7/7**. The provider suite uses
deterministic loopback responses; the submission suite uses real metadata checks
and sidecar restarts with explicitly simulated readiness/submission responses.
No renderer errors or forbidden requests remained. Receipts are under
`output/provider-check-0.1.10/`, `output/native-close-0.1.10/` and
`output/dataset-submission-0.1.10/`.

Native service-state checks passed **6/6**, and service UI checks passed **5/5**
on the final UI build. Controlled response timing around real service restarts
verified that older background configuration successes and failures cannot undo
explicit review, overwrite loaded settings, or add a stale connection banner.
Those cases modify only a synthetic model identifier in the isolated configuration
to represent an independent edit; they make no non-GET API request. Evidence is
under `output/native-service-state-0.1.10/` and `output/service-ui-0.1.10/`.

The service UI suite first exposed an obsolete expectation that dirty settings
would become saveable immediately after restart. Its updated review flow then
reproduced a real late-background-refresh race, which was fixed and rechecked.
Earlier failure evidence is retained in the service UI output directory.

The settings fixture initially captured the debounced experiment draft too early;
it was corrected to await persistence. The initial captured failure is retained
as `output/provider-settings-0.1.10/initial-draft-fixture-failure.log`. This was a
fixture timing failure, not evidence of a product data-loss defect.

## Packaged acceptance

All **120/120 checks** passed against the frozen packaged executable, using
isolated profiles and a system-only PATH:

| Suite | Passed | Receipt directory under `output/` |
| --- | ---: | --- |
| Provider settings | 15/15 | `provider-settings-packaged-0.1.10/` |
| Dataset metadata | 14/14 | `dataset-inspection-packaged-0.1.10/` |
| Submission guards | 7/7 | `dataset-submission-packaged-0.1.10/` |
| Provider response checks | 14/14 | `provider-check-packaged-0.1.10/` |
| Runtime maintenance UI | 9/9 | `runtime-health-ui-packaged-0.1.10/` |
| Native close protection | 6/6 | `native-close-packaged-0.1.10/` |
| Experiment organization | 8/8 | `run-organization-packaged-0.1.10/` |
| Experiment reuse | 8/8 | `experiment-reuse-packaged-0.1.10/` |
| History import | 10/10 | `history-import-packaged-0.1.10/` |
| Service recovery | 7/7 | `service-recovery-packaged-0.1.10/` |
| General Electron workflows | 22/22 | `electron-packaged-0.1.10/` |

Receipts are `acceptance.json`, except the general suite's
`electron-acceptance.json`. No packaged suite required a source or bundle change.
Reviewed screenshots cover settings conflicts, key switching, environment errors,
light/dark themes, compact windows, keyboard controls, history and candidate views.
No renderer errors or forbidden requests remained in the completed suites.

The settings suite performed 37 native settings/probe operations, two real service
restarts (revisions 4 to 6 to 8), and exactly one explicit deterministic loopback
generation request using provider B's key. It exercised real save conflicts,
preserved unsaved fields, a committed save followed by an injected transport
failure, explicit review/keep/load choices and focus retention during a slow read.
The existing response suite made twelve short generation and three model-list
requests to a local fixture. These are separate from external-provider acceptance.

Metadata checks performed 25 real local inspections. Submission acceptance made
nine real metadata requests and two real service restarts; its four intercepted
submission responses and readiness states were simulated, with no real experiment.
Runtime UI maintenance responses were also simulated. Real repair and execution
were verified separately below; historical records in UI suites were fixtures.

## Real offline engine verification and repair

The packaged service passed **4/4 checks in 108.505 seconds** under an OS policy
denying outbound networking. An APFS clone passed actual dependency imports and a
fixed CPU self-test, detected removal of MONAI in that clone, retained its
not-ready state across restart, completed a locked offline forced reinstall and
native-library provisioning, then passed another download-free health check.

All 34,295 original engine files and 29,507 original cache files retained matching
fresh before/after fingerprints. The actual engine and cache were not damaged or
repaired. Evidence is `output/engine-health-packaged-0.1.10.json` and
`output/qa/engine-health-0.1.10/`.

## Artifacts and installation

The DMG was mounted read-only and its Applications shortcut resolved correctly.
Deep strict signature checks passed on the unpacked app, DMG app and extracted
ZIP app. Their complete 2,814-file bundle fingerprints matched:
`67399b9eca6b6e867f273558a1526b43256a14e869a192b810f6022d1201943a`.
The manifest records version 0.1.10 and the clean release source commit above.

| Artifact | SHA-256 |
| --- | --- |
| `BioMaker-0.1.10-mac-arm64.dmg` | `f46adc2eee61cbf7fab60e37531b7fc5ac9882687747a62e268e1be9515c07b9` |
| `BioMaker-0.1.10-mac-arm64.zip` | `231d9a85622be54744dc3e26e470ab256b4082077e9190a3a9b0150ae1f43a53` |

Version 0.1.10 is installed at `/Applications/BioMaker.app`. The previous app is
preserved at `release/installed-backups/BioMaker-0.1.9-20261009-032212.app`.
The actual installed application started with a system-only PATH and reused its
ready engine without requesting installation. The engine source SHA remains
`56d274eec3df1d02eefb52dbe5cc894ff4a5eabe066f483046462e4e021e9fa2`.

The actual configuration-file hashes and complete run rows matched the snapshot
taken immediately before replacement. The actual profile had no experiments or
configured API key/model. No synthetic settings or experiments were inserted
into it. Ordinary browser/runtime metadata can change during startup; this is
not a claim that every profile file remained unchanged.

Receipts are `output/package-verification-0.1.10.json`,
`output/package-installation-0.1.10.json` and
`output/installed-engine-acceptance-0.1.10.json`.

After the isolated workflow and final verification, the normal installed app was
reopened and its window raised. A fresh native accessibility and screenshot check
showed version 0.1.10, local runtime ready and the empty new-experiment screen.
It was left open without entering or saving values. Configuration hashes and
complete run rows were checked once more after this normal launch and still
matched the pre-upgrade snapshot (`output/normal-launch-0.1.10.json`).

## Installed training workflow

The installed 0.1.10 executable completed native folder selection, real metadata
inspection, UI submission, durable queue execution, CPU training, frozen
evaluation, result inspection and native report export in **16.218 seconds**.
The run succeeded with exit code 0. It fitted a scikit-learn logistic regression
model on synthetic data using the retained installed engine; final retraining was
disabled. Five model requests reached a deterministic localhost fixture, with no
external provider calls, renderer errors or fixture errors.

A test-only spawn wrapper restricted the sidecar, queue, uv and training
descendants to localhost. The exact `/bin/ps` executable was exempt solely for
read-only process identity inspection. The candidate confirmed denial of an
external-address connection before transmission. Production source and the
Chromium sandbox were unchanged.

Installed engine/runtime file metadata and selected immutable file hashes were
unchanged. Final checks confirmed actual configuration and complete run rows
were unchanged, all inspected QA profiles had no remaining processes, installed
and packaged signatures still passed, and the entire installed bundle matched
the verified package fingerprint above. Result and export screenshots were
visually reviewed. Durable QA evidence was copied without following symlinks to
the actual engine/runtime directories.

Receipts are `output/desktop-workflow-acceptance-0.1.10.json` and
`output/post-workflow-verification-0.1.10.json`; supporting synthetic evidence is
under `output/qa/desktop-workflow-0.1.10/`. This verifies the software workflow,
not external-provider quality or biomedical model performance.

## Release boundaries

The shared CLI/WebUI configuration API and scientific engine are unchanged.
Tests with deterministic local providers establish routing and software behavior,
not third-party service availability or biomedical model performance. Windows,
Linux, Intel Mac and independent clean-hardware installation remain unverified.
Developer ID signing, notarization and signed updates require release credentials.
Local ad-hoc signing does not establish Developer ID distribution.
