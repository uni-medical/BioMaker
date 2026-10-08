# BioMaker Desktop 0.1.14 acceptance

Target: macOS Apple silicon / arm64. Date: 2026-10-09.

Status: source and development validation passed; package and installation
acceptance pending.

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
Only Apple silicon macOS has prior installed acceptance; Linux, Windows and
macOS x64 remain separate validation targets. Synthetic software tests do not
establish biomedical performance or external model-provider access.
