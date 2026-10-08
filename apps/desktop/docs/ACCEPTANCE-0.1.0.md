# BioMaker Desktop 0.1.0 acceptance

Date: 2026-10-08. Host: macOS 26.6.2, Apple silicon / arm64.

## Verified

| Check | Evidence |
| --- | --- |
| Existing backend regressions | 102 WebUI tests passed |
| Desktop service | 17 auth, setup, cancellation, subprocess and HTTP tests passed |
| Native boundary and packaging | 11 tests passed, including real service crash/restart and relocated runtime |
| UI build | TypeScript checks and Vite production build passed |
| Runtime installation | Actual locked scientific dependencies installed; `biomaker.run`, Torch and MONAI imported; console entrypoint checked without executing a task |
| Dependency advisories | JavaScript, service lock and desktop engine lock audits found no known vulnerabilities at acceptance time |
| macOS installation | DMG mounted read-only, app and Applications shortcut checked, app copied to `/Applications/BioMaker.app`, volume detached |
| Bundle signature | `codesign --verify --deep --strict` passed; signature is ad-hoc, not Developer ID / notarized |
| Packaged interaction | Installed executable passed the real Electron workflow with PATH restricted to system directories |

The installed-app workflow covers fresh startup, service authentication through IPC,
renderer isolation, unsafe path rejection, missing-environment preflight, provider
settings persistence and masking, themes, language changes, restart, historical run
import, candidate lineage and code, nonfinite journal/metric values, real native
artifact export, console display and a 960px window. No renderer errors occurred.

History and candidate rendering use clearly labelled synthetic records in a
separate temporary application profile. Fake credentials in that profile are not
sent to a provider. Production starts without seeded experiments or benchmark
scores. Installation acceptance is separate from scientific model evaluation.

Machine-readable records and screenshots are retained locally under
`apps/desktop/output/`; distributable artifacts and SHA-256 checksums are under
`apps/desktop/release/`. Generated outputs are excluded from source control.

## Boundaries

- No real biomedical training run or paid model-provider request is claimed by
  this acceptance. The user must configure their chosen model service and data.
- Windows, Linux and Intel macOS have not received equivalent installed training
  acceptance. Their configuration is a migration starting point.
- Developer ID signing, Apple notarization and automatic update delivery require
  a release owner's credentials and a signed release feed.
- The scientific environment executes generated code with the user's access;
  it is not an OS security sandbox. Experiments use trusted tasks and data.
- New desktop dependencies differ from the historical research lock. Scientific
  comparisons require rerunning the relevant evaluation under the recorded lock.
