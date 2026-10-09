# BioMaker Desktop 0.1.16 acceptance

Target: macOS Apple silicon / arm64. Date: 2026-10-09.

932 source checks, 202 development checks across 16 distinct suites, 237
packaged checks across 19 suites and four offline engine checks passed. Package
verification, actual upgrade, independent runtime setup, retained-record checks,
installed synthetic CPU workflow and normal native launch passed on this Mac.

Frozen package source: `5236b37adf626f480c742818b9f93e5f3b155e3e`.
Later acceptance documentation is separate from that frozen package. Earlier
[0.1.15 acceptance](ACCEPTANCE-0.1.15.md) is retained as historical evidence.

## Delivered behavior

The persisted run ledger remains authoritative for lifecycle status, reasons,
ownership and available actions. Current catalog evidence is returned separately
as `event_tail`: the valid-object limit, observed valid-object count, terminal
event and `no_terminal_in_tail` reason code. The tail contains up to the last
5,000 valid JSON objects; the existing event count still counts nonempty lines.

The top-level `terminal_event` now describes the current catalog tail and becomes
`null` if no terminal event is found there, including when a previously observed
terminal moves outside that window. This intentionally removes stale persisted
observations from the response. Unavailable catalog evidence omits `event_tail`,
which differs from an available empty tail. No persistence migration or automatic
lifecycle/action rewrite is introduced.

The UI distinguishes saved status from current history evidence. An imported
record's original missing-terminal reason describes what was read at registration,
not a claim about the current file or original process liveness. Known reasons
are localized; unknown errors remain verbatim and copied-history messages retain
their existing handling. A nonempty current tail without a terminal event can
explain a saved ended status without declaring that status invalid.

Each toast has a fresh identity and its own dismissal timer, including repeated
identical messages. Async feedback belongs to the view, language, settings and
service context that started the action; returning to an earlier context must not
revive old feedback. Settings shows saving, saved and unsaved changes inline;
editing invalidates the prior saved indication.

Explicit experiment-tab changes align the outer scroll before switching content,
and opening another run resets it to the top. Manual reading and refresh retain
their position. Keyboard focus is kept visible below sticky tabs. Ordinary
manual scrolling can still move a heading above the visible content area.

## Recorded checks

| Check group | Result | Receipt under `output/` |
| --- | ---: | --- |
| WebUI | 212 passed | `webui-source-0.1.16.log`, `.xml` |
| Desktop service | 552 passed | `service-source-0.1.16.log`, `.xml` |
| Native/package unit checks | 157 passed | `native-source-0.1.16.log` |
| Close protection | 7 passed | `close-source-0.1.16.log` |
| Event presentation | 4 passed | `activity-source-0.1.16.log` |
| TypeScript / production build | Passed | `typecheck-source-0.1.16.log`, `build-source-0.1.16.log` |
| Broader development regression | 171 passed in 14 suites | `development-regression-0.1.16.json` |
| Navigation | 14 passed | `development-new-interactions-0.1.16.json` |
| Toast feedback, final run | 17 passed | `development-toast-final-0.1.16.json` |

The five source check groups total 932. TypeScript and production build are
separate gates, not additional counted checks. Development checks total
171 + 14 + 17 = 202 across 16 distinct suites. The final toast run includes held
language A-to-B-to-A coverage and supersedes the earlier 16-check toast result.
Superseded attempts and repeated suites are not additive.
The earlier `development-navigation-0.1.16.json` records an unsuccessful attempt;
the later combined receipt records the passing navigation run.

`status-evidence-source-0.1.16.json` records the backend contract and focused
regressions, including terminal events exactly at and outside the tail boundary,
catalog source deletion/restoration and unchanged ledger reasons. Its focused
checks overlap the complete suites and are not added to the 932 total. Build
output retains a bundle-size advisory; Python tests retain the existing
Starlette TestClient deprecation warning.

## Distribution and installed acceptance

| Gate | 0.1.16 result and receipt under `output/` |
| --- | --- |
| Packaged regression | 237 checks, 19 suites; `packaged-acceptance-0.1.16.json` |
| Build identity | Frozen Git tests, runtime envelope and receipt hashes agree; `packaged-provenance-review-0.1.16.json` |
| DMG / ZIP / unpacked app | Complete bundle equality and strict deep signatures passed; `package-verification-0.1.16.json` |
| Actual upgrade | 0.1.15 to 0.1.16; previous bundle backed up and fingerprint retained; `package-installation-0.1.16.json` |
| Native runtime setup | Installed through Settings and observed ready; `native-engine-setup-0.1.16.json` |
| Installed app / retained records | System-only PATH; read-only requests; success after shutdown and preservation checks; `installed-engine-acceptance-0.1.16.json` |
| Installed synthetic CPU workflow | Passed in 23.097 s; `desktop-workflow-acceptance-0.1.16.json` |
| Offline clone repair | Four checks passed in 110.331 s; `engine-health-packaged-0.1.16.json` |
| Final package / profile / process checks | Passed; `post-workflow-verification-0.1.16.json` |
| Normal native launch | Original blank saved draft visible, runtime ready, app left open; `native-launch-0.1.16.json` |

The packaged runner binds the actual clean-source manifest and bundle fingerprint
before and after the group, plus each frozen test, native receipt and runner
record. An incomplete group cannot report aggregate success. The verified bundle
has 2,819 regular files and fingerprint
`cab45d6e1cb685a416a93638a5824301c60b1dd7c7f75ed74c26a94ad2a80b6c`.

Deliverables under `release/`:

| File | SHA256 |
| --- | --- |
| `BioMaker-0.1.16-mac-arm64.dmg` | `3cc9b58a678052555f878c94342feefd5cf7ad70350bf63089c211549a1db992` |
| `BioMaker-0.1.16-mac-arm64.zip` | `48d1dca03269f29447ab624301a62150ecb12e3397133509d10597b1da37e985` |

The pre-upgrade snapshot includes all tables, schema, `user_version` and
`application_id` of both `runs.sqlite3` and `desktop-metadata.sqlite3`, including
queue control and sequence state. After upgrade and workflow, those logical
snapshots are unchanged. Configuration, preferences and original draft bytes
are unchanged; the actual profile still has zero user runs. The previous engine's
selected source/receipt files, interpreter link, resolved executable and bytes
are retained. This is the stated preservation scope, not a whole-profile hash.

The first upgrade-helper attempt stopped before writing the baseline or swapping
the app because Apple's read-only SQLite could not open the metadata database
in WAL format without sidecars after shutdown. The corrected helper uses a
private database copy only after explicit app/profile quiescence, absent journal
files and before/after source identity/content checks; live databases never use
that fallback. The verified staging bundle was explicitly resumed. Twelve
synthetic helper tests passed, separately from the source-check total; the failed
attempt and successful retry logs remain preserved.

The installed workflow exercised the actual queue, a one-step CPU logistic
regression fit on 12 synthetic training rows, evaluator, result/candidate/log UI
and native report export. Its five model-response stages used a deterministic
localhost fixture under an inherited network restriction. It does not establish
external-provider quality or medical model performance. The complete synthetic
home was archived with symlink targets preserved under
`output/qa/desktop-workflow-0.1.16/`; seven current screenshots and their hashes
are retained there. No fixture was added to the actual user profile.

Offline health acceptance removed a real dependency only in an APFS clone,
confirmed failure survived reinitialization, performed a locked offline reinstall
and passed another download-free check. Full original engine and cache hashes
matched before, after and the release-owner recheck. Selected clone receipts,
logs and network policy are archived under `output/qa/engine-health-0.1.16/`.
Installed workflow and offline health ran sequentially. Package signatures and
full installed/unpacked bundle fingerprints still matched after acceptance;
all checked QA profile processes had exited before the final native launch.

## Visual observations and remaining scope

The independent packaged visual review inspected five current screenshots and
recorded exact hashes in `packaged-interaction-visual-review-0.1.16.json`.
One English evidence screenshot after a Settings/language round trip clips the
top of the overall app, including the status badge, despite the inner content
being reset. Its visible evidence text is legible and the automated status check
passes, but this does not establish that the full introduction remains visible.
Ancestor scrolling or compact-layout sizing needs targeted reproduction in the
next iteration. A draft-removal failure also retains the existing broad draft
save-issue banner alongside the precise queued-but-draft-retained toast; clearer
operation-specific wording remains a follow-up.

The current signing boundary remains ad-hoc; Developer ID signing and Apple
notarization are not established. Windows and Linux have no platform acceptance
claim. The actual user's provider and model remain unconfigured. Synthetic
software checks do not establish external-provider reliability, scientific
validity or clinical suitability. See [release policy](RELEASE.md).
