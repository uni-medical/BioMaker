# Changelog

## 0.1.11 — 2026-10-09

Adds a Results tab with direct access to reports, saved solution code, candidate
metrics and final-retraining receipts. Recorded selection and exact code matching
have distinct provenance; the overview links to the identified candidate without
ranking unrelated metrics or treating generated report prose as evidence.
Missing, contradictory and changing records remain explicit, including imported
snapshots that do not contain a final-retraining receipt.

Text previews use bounded, stable reads of registered artifacts without following
symbolic links. Saving from a preview verifies the original UTF-8 bytes against
its SHA-256 and the local service generation before replacing the destination.
Changed content, service replacement and failed writes preserve an existing target.
Direct exports and native save dialogs remain available. An application-wide save
barrier refuses overlapping exports without queuing an older overwrite.

Candidate navigation retains the selected result beyond the first 1,500 displayed
nodes, identifies truncated lists, and filters malformed optional display fields.

## 0.1.10 — 2026-10-09

Fixes provider switching: saving a different API address can no longer silently
retain the previous provider's key. Enter a replacement key or explicitly remove
the saved key before saving the new address. Equivalent addresses can still
retain their saved key. Saving settings does not make a provider request.

Desktop configuration updates use a serialized, atomic save and an opaque
revision. Stale settings cannot overwrite a newer configuration without review.
Conflicts preserve entered values, show the latest saved public settings, and
offer loading those settings or continuing to edit the retained inputs before
a separate save. Secrets remain masked and are never included in revision tokens.

## 0.1.9 — 2026-10-09

Adds a local dataset overview to new experiment drafts, with ordinary-file counts,
logical sizes, file types and a limited list of top-level entries. It reads only
filesystem metadata, uses native folder grants, and does not read file contents
or call a model. Large or partly inaccessible directories show partial results;
internal symbolic links are counted without following their targets.

Dataset inspection runs in a bounded worker. Desktop submission and requeue
perform a fresh access check, rejecting confirmed empty or inaccessible roots
before creating an experiment. Draft changes and service replacement invalidate
older overview results. The check describes the filesystem at inspection time,
not scientific validity or availability throughout a queued experiment.

## 0.1.8 — 2026-10-09

Adds an explicit model-response check in provider settings. One short Chat
Completions request tests the current form without saving it, creating an
experiment or installing the training environment. Model lists, saved settings
and successful response checks have distinct feedback. No paid probe runs
automatically or retries without another user action.

Provider checks use a bounded worker, a fixed prompt and output-token budget,
and return status metadata without provider response text. Saved keys are reused
only for the same normalized provider URL; redirects cannot forward them.
Editing provider fields or replacing the local service invalidates stale results.

## 0.1.7 — 2026-10-09

Adds runtime checks and explicit repair from Settings. A check runs the real
dependency imports and a fixed CPU self-test without downloading packages.
Repair reinstalls the locked dependencies, repairs native libraries and verifies
the result before making the runtime available again. Failed checks invalidate
the installation receipt so restarting cannot silently restore a false ready state.

Runtime maintenance supports bounded execution, cancellation, progress logs and
failure retry. An inherited engine lock prevents another service from modifying
the environment while its maintenance worker remains alive. Desktop experiment
submission, requeue and queue resume are serialized with maintenance admission;
queued, running, stopping or cleaning experiments block maintenance.

The interface preserves drafts and unsaved settings, distinguishes checking,
repairing and external maintenance, and ignores status requests that began before
a newer operation result. Repair requires an explicit in-window choice.

## 0.1.6 — 2026-10-09

Keeps the desktop window usable when constructing or starting the local service
fails. Manual retry replaces only the service, preserving the renderer, draft and
unsaved provider settings. Lifecycle revisions remain monotonic across process
replacement; concurrent retries cannot create duplicate services.

Adds native diagnostic export from failure prompts, runtime settings and Help.
Reports use a bounded whitelist of versions, resource availability and lifecycle
codes. They exclude raw logs, tasks, credentials, data and user paths, and can be
saved even when the service has never started. Nothing is uploaded automatically.

Fixes first-frame modal focus stealing from a field the user already selected.

## 0.1.5 — 2026-10-08

Adds Unicode display names and reversible experiment archiving. Organization
metadata is versioned separately from the run ledger and original files;
concurrent edits preserve the user's input and require an explicit retry.
Running or cleaning-up experiments remain visible.

Reusing an experiment now creates an editable draft. It preserves supported
recorded options, confirms replacing an existing draft, and requires dataset
selection for imported histories. A new run is submitted only through Start.

Release verification checks deep strict signatures and complete bundle equality
across the unpacked application, read-only DMG and extracted ZIP.

## 0.1.4 — 2026-10-08

Adds previewed read-only history import with native directory grants, bounded
record copies, changed-source checks, durable jobs, duplicate handling and
restart recovery. Imported results retain their provenance and original files.

Service readiness and reconnect states now follow authenticated health. Stale
responses cannot overwrite recovered UI state; drafts and unsaved settings
survive interruptions.

## 0.1.3 — 2026-10-08

Protects unsaved model settings when closing the native window or quitting the
application. Keep editing preserves the form and local service; discarding must
be explicit. In-flight saves cannot be discarded, configuration fields stay
locked until the response arrives, and failed saves retain edits for retry.
Unsaved credentials remain only in memory.

Packaging now rejects a portable runtime prepared for a different application
version and gives the command needed to refresh it. This desktop-only update
keeps the engine source and scientific dependency contract unchanged.

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
