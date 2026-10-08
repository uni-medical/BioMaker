# BioMaker Desktop 0.1.1 acceptance

Date: 2026-10-08. Host: macOS, Apple silicon / arm64.
Packaged source: `02611bcc22e300bd2f16718bb046e7ecedde64dd`, clean checkout.

## Verified

- 102 existing WebUI tests, 28 desktop service/native-runtime tests, and 11
  native/package tests passed. TypeScript and the production build passed.
- A fresh isolated environment downloaded and installed the 99 locked engine
  packages. Installation repaired LightGBM's native dependency, verified its
  signed library, imported LightGBM before Torch, performed a tiny synthetic CPU
  fit with 40 predictions, imported Torch/MONAI, and wrote the revised receipt.
- An isolated engine copy passed with filesystem reads denied for `/opt/homebrew`,
  `/usr/local`, and `/opt/local`. Loaded OpenMP libraries came from the engine's
  own dependencies. The source engine and package cache were preserved.
- A synthetic NumPy computation launched through the real SQLite queue and detached
  supervisor continued after the HTTP sidecar exited. A new authenticated sidecar
  reconnected, reported it running, and stopped the owned worker successfully.
- The DMG was mounted read-only, its app version and Applications shortcut checked,
  and the app installed to `/Applications/BioMaker.app`. Deep strict code-signature
  verification passed. The signature remains ad-hoc, not notarized.
- The installed executable passed Electron interaction acceptance with PATH limited
  to system directories. Checks include missing-runtime guidance, partially saved
  provider configuration, return to a preserved draft without submitting a run,
  secret masking, Chinese/English native menus, validated preference writes and
  persistence, themes, history/candidates, nonfinite metrics, native output export,
  logs, and compact layout. There were no renderer errors.
- The real installed application's setup action completed in the user's actual
  application profile under the same system-only PATH. Its 0.1.1 engine is ready;
  the user's provider key/model remain unconfigured, without seeded QA records.

## Evidence

Local machine-readable records live in `apps/desktop/output/`:

- `engine-acceptance.json`
- `native-portability-proof.json`
- `queue-lifecycle-acceptance.json`
- `playwright/electron-acceptance.json`
- `installed-engine-acceptance-0.1.1.json`

The UI records are explicitly synthetic and use an isolated application profile.
Provider credentials in that profile are fake and are never sent to a provider.
The native numerical checks do not constitute a biomedical experiment or measure
BioMaker's model-development quality.

Distributables are in `apps/desktop/release/`:

| Artifact | SHA-256 |
| --- | --- |
| BioMaker-0.1.1-mac-arm64.dmg | `73c783caa85bae14398d5f67271c015d6bd3b2fe51d745db07d2dcce3e9f9562` |
| BioMaker-0.1.1-mac-arm64.zip | `243b1184f39b83f6d0bcae5372a81e93304329e1fd535fef4094e3d2453536f2` |

## Remaining release boundaries

No paid model-provider request or full biomedical task is claimed. Apple Developer
ID signing/notarization, signed automatic updates, independent clean-hardware
acceptance, and Windows/Linux/Intel Mac installed-engine acceptance remain pending.
The Homebrew-denial test is a controlled check on this Mac, not another physical
machine. Generated experiment code runs with the user's access. Scientific
comparisons require evaluation under this version's recorded engine lock.
