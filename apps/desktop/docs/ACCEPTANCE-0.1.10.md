# BioMaker Desktop 0.1.10 acceptance

Date: 2026-10-09. Target: macOS, Apple silicon / arm64.

This iteration fixes saving a changed provider address while retaining an omitted
old API key. It also adds conflict-aware desktop settings saves. Validation and
artifact acceptance are in progress; this file does not yet establish a release.

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

## Evidence to complete

Record isolated development and packaged acceptance, package verification,
installation and the installed workflow after their actual receipts are available.

## Release boundaries

The shared CLI/WebUI configuration API and scientific engine are unchanged.
Tests with deterministic local providers establish routing and software behavior,
not third-party service availability or biomedical model performance. Windows,
Linux, Intel Mac and independent clean-hardware installation remain unverified.
Developer ID signing, notarization and signed updates require release credentials.
