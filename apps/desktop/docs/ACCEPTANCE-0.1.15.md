# BioMaker Desktop 0.1.15 acceptance

Target: macOS Apple silicon / arm64. Date: 2026-10-09.

918 source, 113 development, 206 packaged and four offline engine checks passed.
Package verification, actual upgrade, independent runtime setup, preservation of
the specified user records, installed synthetic CPU workflow and normal native
launch passed on this Apple silicon Mac.

918 source checks passed: WebUI 198, desktop service 552, native/package unit
checks 157, close protection 7 and event presentation 4. TypeScript and the
production build passed. A production build is separate from distribution
packaging and application acceptance.

Release source: `a86f0bf619293e4c171a40b86fdfe4911aff947e`. The performance
baseline is `877f328f96f714396bc67a02528855449668ae87`; final v4 measurements use
the optimized working-tree source, identified by the six SHA256 values in
`output/long-history-after-0.1.15-v4-final.json`. Those six source files stayed
unchanged during measurement and matched the receipt when rechecked afterward.

## Long-history reader contract

Catalog discovery and catalog enrichment now scan backward from the event
file's initial end, stopping after the last 5,000 valid JSON objects. Inference
still uses those objects in chronological order. Malformed lines and valid
non-object JSON do not consume the object limit. A terminal lifecycle event
outside that retained object window does not determine catalog status.

The public event response and observed event count retain their existing
meaning: `count` is the number of nonempty decoded lines, including malformed
JSON and non-object JSON. Counting still scans the complete initial file extent,
but JSON decoding stops after enough valid objects have been retained. A tail
of zero counts without parsing any JSON. Detail observations retain the last
valid object, not necessarily the last physical line.

UTF-8 replacement decoding, CR/LF/CRLF line endings, blank-line handling and
unterminated final records preserve the previous successful-reader behavior.
Nonfinite JSON numbers remain explicit diagnostic strings. Large integer
conversion errors and decoder recursion errors are now counted as nonempty
invalid records and skipped; the legacy parser could raise instead. This is an
intentional robustness change, not an assertion that every former exception is
preserved.

Each event read pins one descriptor and its initial byte length. Appending does
not extend that read indefinitely, and replacing the path does not switch the
open descriptor to another file. A truncation that produces a short read raises
an I/O error. This does not provide a transactional snapshot of a file rewritten
in place. The observation cache separately rechecks dependency metadata before
publishing a result. Missing-file and strict/non-strict I/O policies are retained.

Event-reader memory scales with one 64 KiB block, the longest individual record
and the retained objects. It is not a fixed byte bound for arbitrary records.
A catalog with too few valid objects or a large invalid suffix may still scan
the entire file. A detail count always scans the complete initial event extent
when invalidated. This change does not introduce an incremental event index.

The result-summary reader streams TSV rows, retaining aggregate minimum/maximum
scores and only the latest row. Its count includes every CSV row. Only finite
metrics participate in the best score; the last finite metric row's `maximize`
value chooses the direction over all finite scores. The final row supplies
`latest`, even when its metric is invalid or nonfinite. A late CSV/I/O failure
still discards a partial summary or propagates under strict mode. Parser memory
can grow with an individual field; streaming does not establish a universal
fixed byte ceiling or avoid scanning all rows.

The ledger remains authoritative for lifecycle state, ownership and available
actions. Appended event observations do not silently replace ledger lifecycle
state. Existing observation-cache invalidation, independent returned copies and
fresh artifact-list behavior remain in place.

## Source checks

| Check group | Result | Receipt |
| --- | ---: | --- |
| WebUI | 198 passed; zero failures/errors/skips | `output/webui-tests-0.1.15.xml`, `output/webui-source-0.1.15.log` |
| Desktop service | 552 passed; zero failures/errors/skips | `output/service-source-tests-0.1.15.xml`, `output/service-source-0.1.15.log` |
| Native/package unit checks | 157 passed; zero failed/cancelled/skipped | `output/native-source-0.1.15.log` |
| Native close protection | 7 passed; zero failed/cancelled/skipped | `output/close-source-0.1.15.log` |
| Event presentation | 4 passed; zero failed/cancelled/skipped | `output/activity-source-0.1.15.log` |
| TypeScript | Passed | `output/typecheck-source-0.1.15.log` |
| Production UI/Electron build | Passed | `output/build-source-0.1.15.log` |

The latest-row result-reader refinement is included in the final 198-check WebUI
run and in v4 measurement. The 13 focused result-streaming checks also passed.
Independent Python source review and source/test hashes are retained in
`output/python-source-review-0.1.15.json`.

Event-reader tests compare against the previous text-iterator oracle across
multiple block boundaries and tail sizes, mixed line endings, split UTF-8,
malformed/non-object JSON, nonfinite values, large records, invalid suffixes,
and the last-5,000-valid-object boundary. Descriptor tests cover append,
replacement and short reads. Result tests compare against the materializing
oracle for mixed directions and metrics, missing/duplicate/extra fields,
quoted fields, late CSV failure and retained-memory behavior.

The existing Starlette/httpx deprecation warning and Vite chunk-size warning
remain separate from failures. These source/unit checks do not establish
development UI, packaged application, installed runtime or biomedical validity.

## Final v4 isolated performance measurement

The baseline and final source used the same fixture matrix: 1 and 25 histories,
each with 100,000 events. The selected history in each case starts with 100,000
result rows; other histories have a small result file. One event and then one
result row are appended to the selected history during the invalidation checks.
Each initial event file is 5,288,878 bytes; each large result file is 2,788,912
bytes. The two cases initially contain 143,098,511 total log-content bytes.

Measurements call real GET endpoint functions, SQLite and readers, with separate
standalone `list_log_runs` discovery. They do not run app lifespan or a manager.
Startup discovery precedes the first catalog call and bypasses the application
observation cache. The selected-detail cache is explicitly cleared before its
cold call. Each unchanged scenario immediately follows the corresponding load
or append scenario. Cold/changed operations have one timing sample; unchanged
operations have two, reported as their median.

The following tables show every ordinary timing scenario. Logical bytes are
returned file content, not physical disk traffic. Event JSON parses exclude
defensive-copy decoding of cached payloads.

### One history with 100,000 events

| Operation | Before ms | Final v4 ms | Logical bytes before / after | Event parses before / after |
| --- | ---: | ---: | ---: | ---: |
| Startup discovery | 239.119 | 14.578 | 5,289,193 / 327,995 | 100,000 / 5,000 |
| Cold catalog list | 231.022 | 8.442 | 5,289,193 / 327,995 | 100,000 / 5,000 |
| Unchanged catalog list | 0.113 | 0.103 | 0 / 0 | 0 / 0 |
| Cold selected detail | 613.946 | 192.080 | 13,367,298 / 8,406,100 | 200,000 / 5,001 |
| Unchanged selected detail | 0.277 | 0.277 | 0 / 0 | 0 / 0 |
| Detail after event append | 611.277 | 196.886 | 13,367,458 / 8,406,180 | 200,002 / 5,001 |
| Unchanged after event append | 0.274 | 0.298 | 0 / 0 | 0 / 0 |
| Detail after results append | 376.703 | 192.751 | 8,078,212 / 8,078,212 | 100,001 / 1 |
| Unchanged after results append | 0.286 | 0.288 | 0 / 0 | 0 / 0 |
| Standalone results summary | 159.118 | 166.829 | 2,788,939 / 2,788,939 | 0 / 0 |

### Twenty-five histories with 100,000 events each

| Operation | Before ms | Final v4 ms | Logical bytes before / after | Event parses before / after |
| --- | ---: | ---: | ---: | ---: |
| Startup discovery | 5944.900 | 197.488 | 132,229,850 / 8,199,900 | 2,500,000 / 125,000 |
| Cold catalog list | 5897.705 | 194.744 | 132,229,850 / 8,199,900 | 2,500,000 / 125,000 |
| Unchanged catalog list | 1.842 | 1.718 | 0 / 0 | 0 / 0 |
| Cold selected detail | 624.176 | 194.067 | 13,367,300 / 8,406,102 | 200,000 / 5,001 |
| Unchanged selected detail | 0.263 | 0.271 | 0 / 0 | 0 / 0 |
| Detail after event append | 641.855 | 192.136 | 13,367,460 / 8,406,182 | 200,002 / 5,001 |
| Unchanged after event append | 0.278 | 0.371 | 0 / 0 | 0 / 0 |
| Detail after results append | 391.336 | 190.643 | 8,078,213 / 8,078,213 | 100,001 / 1 |
| Unchanged after results append | 0.267 | 0.268 | 0 / 0 | 0 / 0 |
| Standalone results summary | 162.092 | 158.555 | 2,788,939 / 2,788,939 | 0 / 0 |

### Separate results-memory measurement

The result summary contains 100,001 rows after the declared append. Peak traced
Python allocations are measured in a separate `tracemalloc` pass; these are not
RSS or a total application memory bound. Tracing distorts timing, so those times
are recorded separately and are not latency evidence.

| Fixture case | Peak bytes before | Peak bytes final v4 | Traced-pass ms before / after |
| --- | ---: | ---: | ---: |
| 1 history | 46,794,951 | 49,509 | 987.407 / 1142.035 |
| 25 histories | 46,794,838 | 49,383 | 985.687 / 1108.741 |

Each traced-memory pass reads 2,788,939 logical result bytes and parses zero
event JSON lines. The ordinary standalone result timings vary in both
directions across the two cases, so these single samples do not establish a
CSV throughput improvement. The measured reduction is retained parser memory.

All final unchanged scenarios read zero log-content bytes and parse zero event
lines. Catalog cache hits still decode one defensive JSON copy per history;
detail cache hits decode two. The result-append detail path still scans the full
event file to recover the count and the full TSV to compute its summary, despite
parsing only one event object.

The comparison verifies matching fixture dimensions, operation order, all
recorded semantic assertions and zero content reads/event parses on unchanged
calls. The semantic assertions cover history counts and statuses, selected
identity and ledger state, event counts and last object, terminal observations,
and result count/best/latest. They do not compare every field of every full API
response; the source tests provide the additional edge-case contracts.

Timings include counter overhead and naturally warm OS caches. They exclude
complete application startup, HTTP/ASGI serialization, Electron/UI, queue or
provider work, training, physical disk I/O and RSS. All profiles, logs and SQLite
stores are synthetic and isolated from the actual user's application profile.
Only the two declared appends modify fixture contents.

Reproduction and evidence under `output/`:

- `profile-long-history-0.1.15-v2.py` and `long-history-baseline-0.1.15-v2.json`
  retain the successful baseline; its matching log is preserved.
- `profile-long-history-0.1.15-v4.py` and
  `long-history-after-0.1.15-v4-final.json` retain final source hashes, samples,
  byte/parse counters and assertions; its matching log is preserved.
- `compare-long-history-0.1.15.py` and
  `long-history-comparison-0.1.15-v4-final.json` retain the passed comparison;
  its matching log is preserved.
- `long-history-performance-section-0.1.15-v4-final.md` records the final
  narrative. V3 scripts, receipts, logs and its explicitly intermediate
  narrative remain preserved alongside the final evidence.

The first profiler's lifecycle assertion incorrectly assumed that appending a
terminal event changes ledger status. V2 corrected that harness assumption;
the failed first script and log remain preserved. V4 retains the exact v2
fixture generation, operation order and semantic assertions, with binary read
counters and descriptor passthrough introduced in v3.

## Development application acceptance

Development acceptance passed 113 checks across nine suites: long history 11,
catalog refresh 13, log reader 14, artifact preview 11, history import 10,
Results 21, general Electron 22, service UI 5 and native service state 6.
`output/development-regression-0.1.15.json` records successful first attempts and
empty owned-process lists. The prepared runtime contains the final shared Python
source; these tests do not silently use an older engine copy.

The new native test generated 12 histories with 40,000 ordinary events each and
120,000 selected result rows, totaling 169,700,762 event/result bytes. Cold profile
service readiness took 2618.692 ms; readiness after a complete app relaunch took
981.637 ms on this machine, with zero startup retries. The native timeout remains 30 seconds
per attempt. These are fixture-specific observations, not a size-independent
startup guarantee. The independent Results evidence reader correctly reports
oversized event history as unavailable instead of inferring a selection.

The independent development review inspected 32 screenshots. Eight suites retain
exact profile paths with raw/canonical process checks. The native-service-state
receipt omits its profile path; its harness removes the temporary directory and
the review found no process with its QA prefix. That is a narrower cleanup receipt.

## Packaged and installed acceptance

`output/packaged-acceptance-0.1.15.json` records 206 passed checks across 17 suites:
native 73, UI 82 and service 51. All frozen-package attempts passed. Independent
reviews inspected 20 native and 22 UI screenshots and checked raw/canonical
temporary-profile paths for remaining processes. No blocking visual regression
was found. Existing transient toasts, compact-view heading occlusion and the
catalog status-reason text after terminal events leave the 5,000-object window
remain recorded as follow-up observations. Ledger ownership remains unchanged.

The source-provenance review confirms that the six v4 measured source hashes
match both the frozen Git commit and the packaged files. Package verification
confirms clean source provenance, deep strict signatures, the Applications
shortcut, and complete equality of the unpacked app, read-only DMG and extracted
ZIP. These signatures remain ad-hoc, not Developer ID or notarization acceptance.

The bundle has 2,819 files and fingerprint
`dc35f149df5bd85b740ea50703dbd9feb33ae67bce8835fbac8d65bd442ac720`.

| Artifact | SHA-256 |
| --- | --- |
| `release/BioMaker-0.1.15-mac-arm64.dmg` | `4d2dddc176704865ceac66947b77b76b51dbff89c3034a4e971837c174883024` |
| `release/BioMaker-0.1.15-mac-arm64.zip` | `d5a562f1ad801eb762c7ab9783de9dabc7f61c8248e3725b9bd10902f0686cbc` |

The actual `/Applications/BioMaker.app` was upgraded from 0.1.14 after a normal
native quit. The old bundle is retained at
`release/installed-backups/BioMaker-0.1.14-20261009-065643.app`.
Source and staged fingerprints were verified before replacement.

Native UI setup prepared the independent engine
`engine-7aea2b5edf05dd6c63ac54f08a05f5eb916b1395851b6bf4f9ad9b8f8f22d534`.
The previous engine remains retained at
`engine-918f7133a2fe716db5d15551948c7fbd9a074da679249d2e8adcb57b73f59775`.
Bundled engine source SHA-256 is
`8aeb130bca1d2993950d365a00fbaee4aa74b4c3b6120ddabd4491023879e70a`.
Installed read-only verification used a system-only PATH and confirmed readiness;
provider credentials and model selection remain unconfigured.

Upgrade preservation checks cover exact bytes/existence for `config.env`, `.env`,
`experiment-drafts.json` and `desktop-preferences.json`; complete `runs` table
rows; and the previous engine's two identity/setup receipts, `run_catalog.py`
and interpreter symlink target. Actual user run count remains zero. The original
draft SHA-256 remains
`e4b3f05ea91babc4acc1ae81b7093c3bca0fe21af83208dcc675ee0ebe9948f7`.
These checks do not fingerprint the entire user profile or previous engine.
`state/desktop-metadata.sqlite3` was not snapshotted before this upgrade and is
not included in the preservation claim.

`output/desktop-workflow-acceptance-0.1.15.json` passed in 23.768 seconds. The
installed app submitted an isolated synthetic CPU experiment, ran the durable
queue, inspected results/candidates, exported a report through the native save
dialog and read logs. A local model fixture and inherited network restrictions
were used; this is software acceptance, not biomedical validation or external
model-provider acceptance. The complete isolated home is retained with symlinks
preserved under `output/qa/desktop-workflow-0.1.15/`; its seven screenshots are
copied separately from `output/playwright/desktop-workflow/` into the versioned
archive and verified by hash.

## Offline recovery and final launch

`output/engine-health-packaged-0.1.15.json` records four checks passed in 111.061
seconds. A network-denying macOS sandbox runs the packaged service against an
APFS clone. Removing a real dependency only from that clone produces a visible
failure which persists across restart; offline locked reinstall then restores
the environment and a second fixed CPU check passes. Complete fingerprints of
the original engine (29,584 files) and package cache (29,546 files) are identical
before and after. The clone remains at its recorded path; the receipt and review
are retained under `output/qa/engine-health-0.1.15/`.

Post-workflow verification confirms that the complete installed app still matches
the verified package, strict signatures remain valid, the specified user records
are unchanged and no owned packaged/workflow QA processes remain. Final CUA
launch raised the real app and verified version 0.1.15, local runtime ready and
the original saved blank draft. The app is left open there. Another read-only
check after normal launch confirmed the same preserved records.

Developer ID signing and Apple notarization remain separate public-distribution
requirements. Windows, Linux and macOS x64 remain separate validation targets.
Synthetic software checks do not establish biomedical performance or external
model-provider access. The development goal remains active for further iteration.
