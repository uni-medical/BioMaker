# BioMaker Desktop 0.1.8 acceptance

Date: 2026-10-09. Target: macOS, Apple silicon / arm64.
Release source commit: **Pending**.

This iteration adds an explicit model-response check to provider settings.
Source validation and development Electron acceptance passed. Packaged
application, artifact and installed workflow validation are **Pending**.

## Verified implementation

- Model discovery, saved configuration and a successful short generation request
  have separate UI states. No probe runs automatically on startup, editing,
  saving, discovering models or returning to an experiment.
- A user-triggered check uses the current provider form and one fixed short
  prompt. It does not read a task or dataset, install the engine, save settings,
  enqueue an experiment or retry a paid request automatically.
- Saved credentials are reused only for the same normalized provider URL when
  the user has not chosen to remove the key. Redirects must not receive keys.
- Result bodies contain only bounded status metadata. Provider response text,
  exception details and credentials are not returned to the renderer.
- Edits and service lifecycle changes invalidate older in-flight results;
  unrelated experiment drafts and unsaved settings are preserved.
- A probe has a hard wall-clock limit and concurrency control. Timeout and
  shutdown settle its owned worker before releasing operation ownership.

## Source validation

**407 source tests passed:** 102 WebUI, 226 desktop service/native-runtime and
79 native/package checks, including the real-sidecar integration test. The
service total includes 94 new provider cases. TypeScript checks and the production
build passed. The Vite build reports a nonblocking 518.87 kB minified UI chunk
(162.46 kB gzip); Python tests retain the existing Starlette/httpx warning.

Real localhost fixtures cover successful requests, authentication and upstream
errors, redirect refusal, malformed/oversized responses, base-path preservation,
saved-key reuse boundaries, no configuration/dataset changes, missing training
runtime, concurrent requests and slow-response deadlines. Other cases verify
the worker's independent watchdog, abrupt owner exit, cleanup failures retaining
ownership until exit is confirmed, and continued service cleanup.

The probe sends one fixed `Reply with exactly OK.` message with
`max_completion_tokens=64` and `stream=false`. The response is limited to 64 KiB;
the renderer receives only status, code, elapsed time and a completion timestamp.
The worker has its own 25-second maximum lifetime, in addition to the caller's
deadline, and exits on POSIX parent loss. It uses a direct connection to the
configured base URL without environment proxies or redirect following.

## Development and packaged acceptance

Development Electron acceptance passed **14 of 14 checks** through the real
native bridge and sidecar to a deterministic localhost provider. Twelve short
generation requests and three model-list requests used only synthetic fixture
credentials. There were no forbidden requests, fixture violations or renderer
errors. Edits, including changes reverted to their prior value, invalidated old
responses. A real sidecar restart advanced the service revision from 2 to 4 and
prevented a held older success from being shown.

Tests also covered rapid duplicate clicks, failures and explicit retries,
saved-key removal and changed addresses, unsaved configuration preservation,
exact experiment draft retention, testing before installing the training runtime,
and retaining available models while switching the selected model. Reviewed
screenshots cover light/dark themes, a 960 by 680 window, keyboard focus, result
details and the save controls. The development receipt and images are under
`output/provider-check-0.1.8/`.

**Pending.** Run required packaged regression workflows and runtime repair gates.

## Installation and artifacts

**Pending.** Record clean source identity, DMG and ZIP hashes, complete bundle
comparison, deep signature checks, backup, installation and engine reuse.
Compare actual configuration files and complete run records before and after
installation; do not claim all browser/profile metadata stays unchanged.

## Installed training workflow

**Pending.** Exercise UI submission, durable queue execution, actual CPU training,
frozen evaluation, results and native export with isolated synthetic data and a
localhost fixture. Verify final cleanup and installed-bundle equality.

## Boundaries

The provider check tests one Chat Completions request with a small token budget.
It does not establish tool-calling support, long-context behavior, complete
experiment success or scientific performance. A provider may reject the request
format or exhaust that budget before producing visible text; those failures do
not prove the model is generally unavailable. Tests use local fixtures, not
real user credentials or external model-provider calls.

Windows, Linux, Intel Mac and independent clean-hardware installation remain
unverified. Apple Developer ID signing, notarization and signed updates require
release credentials and remain pending. The local ad-hoc signature is not
Developer ID distribution readiness.
