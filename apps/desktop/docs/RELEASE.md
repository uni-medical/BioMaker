# Desktop release and iteration policy

## Release gates

- Build from the dedicated desktop branch, preserving the research checkout.
- Pin JS dependencies with `package-lock.json`, record portable runtime provenance,
  and include upstream license notices in the bundle.
- Pass TypeScript, native boundary tests, desktop service tests and the existing
  WebUI suite. Run a real Electron session against the real sidecar in isolated data.
- Check first launch, missing provider/engine, dataset selection, saved settings,
  queue state, reconnect, failure recovery, keyboard navigation, themes and layout.
- Test the packaged `.app`, not only the Vite development window. Verify that it
  starts with a minimal PATH and does not depend on the source checkout or system Python.
- On macOS, verify native dependencies with `tests/native-portability-smoke.py`
  against an installed engine. This denies Homebrew/MacPorts library paths and
  tests an isolated copy; it supplements, rather than replaces, separate-hardware
  acceptance. Use `tests/queue-lifecycle-smoke.py` for no-provider quit/reconnect/stop
  verification through the real supervisor. Run `tests/engine-workflow-smoke.py
  --engine <installed-engine>` for a deterministic local-model fixture through
  the actual split/evaluator/candidate/artifact workflow; keep that result distinct
  from medical or model-provider performance.
- Mount the DMG, check its Applications shortcut and app bundle, then generate
  SHA-256 checksums for the final artifacts.
- For a public release, apply Developer ID signing and Apple notarization. Keep
  signing credentials in the release environment and never in source files.

## Data and upgrades

The macOS app stores its own workspace, settings, run ledger and writable engine
environment under the user's Application Support directory. It does not take over
the research repository's existing `.env` or outputs. Engine code is versioned
separately from user data. Replacing the `.app` must not delete experiments.

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
