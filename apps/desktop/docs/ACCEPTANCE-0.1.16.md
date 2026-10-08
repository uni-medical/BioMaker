# BioMaker Desktop 0.1.16 candidate acceptance

Target: macOS Apple silicon / arm64. Date: 2026-10-09.

This is a development candidate. 932 source checks and 202 development checks
across 16 distinct suites passed; the release gates below remain open. The release
source is not yet frozen. Earlier
[0.1.15 acceptance](ACCEPTANCE-0.1.15.md) does not establish 0.1.16 acceptance.

## Behavior under review

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

## Open release gates

| Gate | 0.1.16 status |
| --- | --- |
| Release source freeze | Pending |
| Distribution build, package verification and packaged regression | Pending |
| Actual installed upgrade and retained user records | Pending |
| Independent installed engine setup | Pending |
| Installed synthetic CPU training workflow and exported results | Pending |
| Offline engine failure/recovery and original-profile preservation | Pending |
| Normal native launch of the installed application | Pending |

Release checks must run against the frozen 0.1.16 source and its actual artifacts.
The current signing boundary remains ad-hoc; Developer ID signing and Apple
notarization are not established. Windows and Linux have no platform acceptance
claim. Synthetic software checks do not establish external-provider reliability,
scientific validity or clinical suitability. See [release policy](RELEASE.md).
