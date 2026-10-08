# BioMaker Desktop 0.1.14 acceptance

Target: macOS Apple silicon / arm64. Date: 2026-10-09.

Release source: `03fafd895c222ef8e2066e1b74f41efe5fbc1192`.

857 source, 102 development, 195 packaged and four offline engine checks passed.
Package verification, the actual upgrade, independent runtime setup,
retained-profile verification, installed synthetic CPU workflow and normal
native launch passed on this Apple silicon Mac.

## Catalog and detail contract

Individual experiment routes now use indexed SQLite lookup, then enrich only the
requested record. Lifecycle status, ownership and available actions are read
from the current ledger on every request. List ordering and the existing
log-derived enrichment fields are unchanged.

One service instance owns the observation cache. It retains serialized summaries
with at most 512 entries, an 8 MiB accounting limit and a 512 KiB per-payload
limit. Entries include path/metadata accounting; this is not a bound on total
Python memory or transient parser memory. Returned nested observations are
independent copies. The cache does not retain ledger records or artifact lists.

Every hit rechecks file identity, type, size, nanosecond modification/change
times and ancestor identities. Missing files are distinguished from failed
reads. Creation, append, rewrite, truncation, replacement and deletion invalidate
observations; a read failure or a file changing during parsing is not cached.
Links and unavailable metadata bypass the cache using existing reader behavior.
This optimization does not add descriptor-pinned security to legacy readers.

Fixed lock stripes serialize concurrent reads of the same key without a global
filesystem lock. Clearing the cache rejects publication by earlier in-flight
loads. A new service starts with an empty cache. Artifact revisions now include
inode/device/mode/ctime so same-size, same-mtime changes trigger renderer refresh.

Existing event semantics are preserved: count means nonempty lines, including
malformed or non-object JSON; catalog inference uses the last 5,000 valid objects.
Nonfinite numbers remain explicit diagnostic strings. A deque avoids repeatedly
shifting the retained event list. Changed or uncached histories still scan fully.

## Source checks

857 checks passed: WebUI 137, desktop service 552, native/package 157, native close
protection 7 and event presentation 4. TypeScript and production build passed.
Receipts are retained under `output/`, including `webui-source-0.1.14.log`,
`webui-tests-0.1.14.xml`, `service-source-0.1.14.log` and the native logs.
Existing dependency deprecation and bundle-size warnings remain separate from
test failures. The first lookup-test attempt used the nonexistent `/console`
route; correcting the harness to the existing `/log` endpoint passed all six
lookup tests. Its original failure receipt is retained.

Development acceptance passed 102 checks: catalog refresh 13, log reader 14,
artifact preview 11, history import 10, Results 21, general Electron 22, service
UI 5 and native service state 6. The new catalog suite uses real sidecar/file
responses, including same-size/same-mtime rewrites and replacements, missing
and restored files/directories, actual service restart and delayed response
ownership. Only response timing is intercepted. All 13 passed on the first run;
four screenshots were inspected and no owned QA processes or renderer errors
remained. No real provider, engine installation or training was used.

## Isolated performance measurement

The same generated fixture contains 100 registered histories with 1,000 events
each. Measurements call real endpoint functions and SQLite, instrument file
reads and JSON parsing, and run each operation three times. They exclude ASGI
serialization/network/UI latency and include instrumentation overhead and warm
OS caches. Logical returned bytes are not physical disk IO.
The original medians use all three full-scan calls; cached medians use the second
and third calls. Explicit cold samples clear the app cache before every call.

| Repeated request | 0.1.13 median | 0.1.14 median |
| --- | ---: | ---: |
| Experiment list | 257.268 ms | 6.673 ms |
| Selected detail | 259.461 ms | 0.249 ms |
| Selected update check | 255.842 ms | 0.296 ms |
| Missing experiment | 255.580 ms | 0.003 ms |

Previously each list/update/missing request parsed 100,000 event records and read
13,008,600 logical file bytes; detail parsed 101,000 and read 13,138,735 bytes.
Warm repeats now read zero event/manifest/results content bytes and parse zero
event lines; decoding cached summary payloads is counted separately.

With the cache explicitly cleared, selected detail reads one directory and
260,221 bytes (2,000 event parses) in 5.539 ms; selected update reads one directory
and 130,086 bytes (1,000 parses) in 2.911 ms. The first complete list still reads
13,008,600 bytes in 270.655 ms. Those cold costs are not eliminated.

The original and revised profiling scripts and receipts are retained under
`output/profile-catalog-0.1.14.py`, `output/profile-catalog-0.1.14-v2.py`,
`output/catalog-baseline-0.1.14.json`, `output/catalog-after-cache-0.1.14.json`
and `output/catalog-comparison-0.1.14.json`. Source hashes were unchanged during
each measurement. No actual profile, model, experiment or worker was used.

## Distribution boundaries

Developer ID signing and Apple notarization remain required for public release.
Only Apple silicon macOS has installed acceptance; Linux, Windows and
macOS x64 remain separate validation targets. Synthetic software tests do not
establish biomedical performance or external model-provider access.

## Packaged and installed acceptance

`output/packaged-acceptance-0.1.14.json` records 195 checks across 16 suites: native
62, UI 82 and service 51. All frozen-package attempts passed. The native review
inspected 18 screenshots, including four catalog-refresh captures; the UI review
inspected 22. Both rechecked raw and canonical temporary profile paths for process
cleanup. No material layout regression was found. Existing transient toast and
captured scroll-position observations remain recorded in the UI review. Recorded
request traces are not an OS-wide network-capture claim.

`output/package-verification-0.1.14.json` confirms clean source provenance,
deep strict signatures, the Applications shortcut and complete application
equality between the unpacked directory, read-only DMG and extracted ZIP.
Signatures remain ad-hoc; this is not Developer ID or notarization acceptance.

The application has 2,818 files and fingerprint
`3b2c3ed05c0d2c2d7b86c6d0fb2932b21c3680f85f1642511638135549660730`.

| Artifact | SHA-256 |
| --- | --- |
| `release/BioMaker-0.1.14-mac-arm64.dmg` | `510a552fb4fc74cc585e18dde7a09b47d276f8b311aaa53432b52f49b16c8e41` |
| `release/BioMaker-0.1.14-mac-arm64.zip` | `52fdb26c4b0454eff4d4d876afa7785669664a99359958c41e1cc9ce27cb11cf` |

The actual `/Applications/BioMaker.app` was upgraded from 0.1.13. The prior bundle
is preserved at `release/installed-backups/BioMaker-0.1.13-20261009-062106.app`.
The installer initially refused replacement while the old app was still running;
raising its window and quitting normally allowed the upgrade. No files were
replaced by that refused attempt.

Native UI setup prepared the independent engine
`engine-918f7133a2fe716db5d15551948c7fbd9a074da679249d2e8adcb57b73f59775`.
The previous `engine-0e32352cb38aa727873ec64a1550067ded4e99e9075ab1358988d488341f9d4c`
remains unchanged. Bundled engine source SHA-256 is
`245052d1ca1523e58f50018d2c915cb0c9b9a0673e150875f55dcbb0acc42e83`.

Installed read-only verification used system-only PATH and confirmed that
configuration, preferences, exact draft bytes and complete SQLite rows were
preserved (zero actual user runs). The original draft retains SHA-256
`e4b3f05ea91babc4acc1ae81b7093c3bca0fe21af83208dcc675ee0ebe9948f7`.
Provider credentials and model selection remain unconfigured.

`output/desktop-workflow-acceptance-0.1.14.json` passed in 23.092 seconds. The
installed application submitted an isolated synthetic CPU experiment, executed
the durable queue, inspected results/candidates, exported a report natively and
read logs. A local model fixture and inherited network restrictions were used;
this is software acceptance, not biomedical model validation. The full isolated
home and seven screenshots are retained at `output/qa/desktop-workflow-0.1.14/`
with symlinks preserved. The post-workflow verification confirms that the complete
installed bundle still matches the verified package and the actual profile is
unchanged.

## Offline recovery and final launch

`output/engine-health-packaged-0.1.14.json` records four passed checks in 118.050
seconds. In a network-denying macOS sandbox, the packaged service detected a
dependency removed only from an APFS clone, preserved failure across restart,
reinstalled locked packages offline and passed a second fixed CPU check.
Complete fingerprints of the original engine (29,583 files) and package cache
(29,533 files) match before and after. The clone is preserved; its receipt and
zero-owned-process verification are copied to `output/qa/engine-health-0.1.14/`.

Final CUA launch raised the actual application and verified version 0.1.14,
local runtime ready and the saved original blank draft. The app is left open
there. Profile verification after normal launch again confirmed unchanged
configuration, preferences, draft bytes and complete run rows. The goal remains
active for further iteration; cold and actively changing histories still need
work before their entire request IO can be described as bounded.
