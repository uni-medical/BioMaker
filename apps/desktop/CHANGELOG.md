# Changelog

## 0.1.2 — 2026-10-08

Fixes a real first-task startup failure: OmegaConf rejected populated compute-slot
metadata because its structured mapping used `object`. Runtime assignments now
use `Any`, with a focused regression and full synthetic workflow acceptance.

Reuses compatible scientific environments across UI-only upgrades by comparing
all bundled source bytes, platform and exact Python provenance. Existing venvs
remain at their original paths. Changed, partial, or modified engines are preserved
and isolated; dependency/native receipts remain the readiness authority. This
release contains an engine fix, so it correctly creates a separate environment.

Adds optional native completion/failure notifications while BioMaker runs in the
background, including after closing its window. Clicks reopen the matching run
without losing the experiment draft. Notification preferences persist; old history,
manual stops/cancellations, and repeated observations remain quiet. Native banners
omit task and dataset names. Delivery depends on OS permission and stops after
quitting the application.

A deterministic local API fixture now exercises the actual split builder, frozen
evaluator, CPU classifier, scoring, journal, best solution, and report pipeline.
It uses synthetic data and makes no external model-provider call or medical claim.

## 0.1.1 — 2026-10-08

Makes first-use configuration actionable: the primary action opens missing setup,
partial provider settings explain what remains, and a return-to-experiment action
preserves the draft without starting a paid request. Readiness indicators now use
the same state. The exploration time limit is labelled accurately, including work
outside that timer.

Native menus and file dialogs follow the UI language. Language preferences persist
in the native profile, including migration from the original renderer preference.

Repairs macOS LightGBM's OpenMP dependency inside the engine environment so it uses
the copy supplied by PyTorch rather than requiring Homebrew. Runtime verification
includes a direct LightGBM import and computation. See the versioned acceptance
record for the measured checks and remaining release boundaries.

## 0.1.0 — 2026-10-08

First macOS Apple silicon desktop build. Adds a Codex-style experiment workspace
with Chinese/English UI, light/dark/system themes, persisted drafts, keyboard
shortcuts, native folder selection, search, history, activity, candidate lineage,
code/output inspection, artifact preview/export, and run/queue controls.

Includes a portable Python/uv service and one-click scientific environment setup.
Runtime paths survive app moves and upgrades. Local API authentication, a restricted
IPC bridge, versioned data, installation cancellation, service recovery and bounded
diagnostics protect the desktop boundary.

Acceptance iterations fixed cold-start settings recovery, duplicate polling,
stale detail responses, unknown historical budgets, model discovery defaults,
nonfinite metrics/journal records, old-runtime queue isolation, source packaging
guards and selected-state contrast. Desktop dependency upgrades are isolated from
the historical research environment.

This build is ad-hoc signed, not notarized. Windows/Linux packaging configuration
does not establish platform support. UI fixture acceptance and engine import tests
do not establish biomedical model performance.
