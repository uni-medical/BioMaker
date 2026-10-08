// Real cold-start and long-history acceptance. Synthetic files only; HTTP
// responses and lifecycle state are never replaced. Run after preparing runtime.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { appendFile, mkdir, mkdtemp, open, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-long-history-'));
const profile = path.join(temporary, 'profile');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/long-history-0.1.15'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const HISTORY_COUNT = 12, EVENTS_PER_HISTORY = 40_000, RESULT_ROWS = 120_000;
const runIds = Array.from({ length: HISTORY_COUNT }, (_, index) => `qa-long-history-${String(index).padStart(2, '0')}`);
const selected = runIds[0];
const directories = runIds.map(id => path.join(profile, 'workspace/logs', id));
const dataset = path.join(temporary, 'synthetic-dataset');
const checks = [], rendererErrors = [], launches = [], measurements = [], processChecks = [], requests = [], forbiddenRequests = [];
const fixture = { historyCount: HISTORY_COUNT, ordinaryEventsPerHistory: EVENTS_PER_HISTORY, resultRows: RESULT_ROWS, files: [] };
let application, page, appInfo, failure;
const event = (name, fields = {}) => ({ ts: '2026-10-09T10:00:00Z', event: name, ...fields });
const jsonl = value => JSON.stringify(value) + '\n';
const first = name => path.join(directories[0], name);
const endpoint = (suffix = '') => `/api/runs/${selected}${suffix}`;
const api = route => page.evaluate(route => window.biomaker.request(route), route);
const progress = () => page.locator('.run-summary > div').filter({ hasText: '已记录探索候选 / 配置上限' }).locator('strong');
const candidateCount = () => page.locator('.run-summary > div').filter({ hasText: '已记录的候选方案' }).locator('strong');
const frame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const rounded = value => Math.round(value * 1000) / 1000;

async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 40));
  }
}
async function timed(name, operation) {
  const start = performance.now();
  const result = await operation();
  measurements.push({ name, elapsedMs: rounded(performance.now() - start) });
  return result;
}
async function writeRows(target, count, row, { prefix = '', suffix = '', append = false } = {}) {
  const handle = await open(target, append ? 'a' : 'w');
  try {
    if (prefix) await handle.writeFile(prefix);
    for (let start = 0; start < count; start += 512) {
      let chunk = '';
      for (let index = start; index < Math.min(start + 512, count); index++) chunk += row(index);
      await handle.writeFile(chunk);
    }
    if (suffix) await handle.writeFile(suffix);
  } finally { await handle.close(); }
}
async function generateFixtures() {
  for (const directory of [profile, output, dataset, ...directories]) await mkdir(directory, { recursive: true });
  await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
  for (const [index, directory] of directories.entries()) {
    await writeFile(path.join(directory, 'run_manifest.json'), JSON.stringify({ created_at: '2026-10-09T10:00:00Z', data_dir: dataset, task_description: `Synthetic long history ${index}; no model training.`, config: { agent_steps: 9000, final_retrain_enabled: false, model_for_all_agents: 'qa-never-called' } }));
    const target = path.join(directory, 'events.jsonl');
    await writeRows(target, EVENTS_PER_HISTORY, row => jsonl(event('tool_observation', {
      step: row, node_id: `synthetic-${index}-${row}`, message: `Synthetic history ${index} observation ${row}; ${'recorded diagnostic '.repeat(5)}`,
      tool: 'fixture_only', duration_seconds: 0, usage: { input_tokens: 0, output_tokens: 0 },
    })), { prefix: jsonl(event('run_started')), suffix: jsonl(event('round_finished', { completed_steps: 40 })) + jsonl(event('run_finished', { message: 'LONG_HISTORY_INITIAL_TERMINAL' })) + '  \nnot-json\n[]\n' });
    fixture.files.push({ runId: runIds[index], file: 'events.jsonl', bytes: (await stat(target)).size, nonemptyRecords: EVENTS_PER_HISTORY + 5, validObjects: EVENTS_PER_HISTORY + 3 });
    await writeRows(path.join(directory, `${runIds[index]}.log`), 450, row => `Synthetic history ${index} console ${row}\n`, { suffix: `HISTORY_${index}_CONSOLE_SENTINEL\n` });
  }
  await writeRows(first('results.tsv'), RESULT_ROWS, row => {
    const metric = row === 0 ? '0.02' : row === 1 ? '0.88' : row === RESULT_ROWS - 1 ? 'nan' : '0.4';
    const direction = row === RESULT_ROWS - 2 ? 'false' : 'true';
    const failed = row === RESULT_ROWS - 1;
    return `${row}\tsynthetic-result-${row}\t${metric}\t${direction}\t${failed ? 'failed' : 'succeeded'}\t${failed ? 'invalid' : 'valid'}\t\t${failed ? 'Synthetic nonfinite score' : ''}\n`;
  }, { prefix: 'step\tnode_id\tmetric\tmaximize\tstatus\toutcome\texc_type\terror\n' });
  fixture.files.push({ runId: selected, file: 'results.tsv', bytes: (await stat(first('results.tsv'))).size, rows: RESULT_ROWS });
  fixture.totalGeneratedEventAndResultBytes = fixture.files.reduce((total, file) => total + file.bytes, 0);
  await assert.rejects(stat(path.join(profile, 'state/runs.sqlite3')), { code: 'ENOENT' });
}
async function nativeState() {
  return application.evaluate(() => ({ requests: globalThis.__longHistoryQA.requests, forbidden: globalThis.__longHistoryQA.forbidden, inflight: globalThis.__longHistoryQA.inflight }));
}
async function settled() {
  await until(() => nativeState().then(value => value.inflight === 0), 'Real long-history requests did not settle');
  await frame();
}
async function launch(label) {
  const started = performance.now();
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  // Trace real requests only. No content, lifecycle, delay, or failure mocking.
  await application.evaluate(() => {
    const original = globalThis.fetch;
    const qa = globalThis.__longHistoryQA = { requests: [], forbidden: [], inflight: 0 };
    globalThis.fetch = async (input, options) => {
      const url = new URL(typeof input === 'string' ? input : input.url || String(input));
      const method = String(options?.method || (typeof input === 'object' && input.method) || 'GET').toUpperCase();
      if (url.hostname !== '127.0.0.1' || !['GET', 'HEAD'].includes(method)) {
        qa.forbidden.push({ path: url.pathname, method });
        throw new Error('Long-history QA forbids mutations, provider calls and installation');
      }
      const entry = { path: url.pathname, method };
      const start = performance.now();
      qa.requests.push(entry); qa.inflight++;
      try {
        const response = await original(input, options);
        entry.status = response.status;
        return response;
      } finally { entry.elapsedMs = Math.round((performance.now() - start) * 1000) / 1000; qa.inflight--; }
    };
  });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  const windowMs = rounded(performance.now() - started);
  await until(async () => {
    const state = await page.evaluate(() => window.biomaker.getServiceState()).catch(() => null);
    if (state?.status === 'failed') throw new Error(`The native sidecar failed: ${state.message}`);
    return state?.status === 'ready';
  }, 'The real sidecar did not become ready', 60_000);
  const readyMs = rounded(performance.now() - started);
  appInfo = await page.evaluate(() => window.biomaker.getAppInfo());
  await page.locator('.sidebar-settings').waitFor();
  await settled();
  const readyState = await page.evaluate(() => window.biomaker.getServiceState());
  const diagnostic = await readFile(path.join(profile, 'logs/desktop-service.log'), 'utf8').catch(error => { if (error.code === 'ENOENT') return ''; throw error; });
  assert(!/Startup attempt|本地服务启动超时/.test(diagnostic), 'Native cold startup must succeed without timeout/retry');
  launches.push({ label, firstWindowMs: windowMs, serviceReadyMs: readyMs, initialRequestsSettledMs: rounded(performance.now() - started), readyState, appInfo, nativeStartupRetries: 0 });
}
async function verifyExited(label) {
  const canonicalProfile = await realpath(profile);
  let remaining = [];
  await until(() => {
    remaining = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes(profile) || line.includes(canonicalProfile)).map(line => line.trim());
    return remaining.length === 0;
  }, 'Processes remain for the exact raw/canonical QA profile', 15_000);
  processChecks.push({ label, profile, canonicalProfile, remainingProcesses: remaining });
}
async function close(label) {
  if (!application) return;
  const current = application;
  const native = await nativeState().catch(() => null);
  if (native) { requests.push({ launch: label, requests: native.requests }); forbiddenRequests.push(...native.forbidden); }
  await current.evaluate(({ BrowserWindow }) => { for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
  try { await current.close(); } finally { application = null; page = null; }
  await verifyExited(label);
}
function assertLedgerOwnership(run, expected) {
  for (const key of ['status', 'detailed_status', 'status_source', 'status_reason', 'error', 'managed', 'read_only', 'allowed_actions']) assert.deepEqual(run[key], expected[key], `Live ledger ownership changed for ${key}`);
}
async function registered() {
  const runs = await api('/api/runs');
  assert.deepEqual(runs.map(run => run.id).sort(), [...runIds].sort());
  for (const run of runs) {
    assert.equal(run.managed, false); assert.equal(run.read_only, true);
    assert.equal(run.status, 'complete'); assert.equal(run.detailed_status, 'succeeded');
    assert.deepEqual(run.allowed_actions, []);
  }
  return runs;
}
async function openSelected() {
  await application.evaluate(({ BrowserWindow }, id) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:open-run', id), selected);
  await until(() => page.locator('.run-intro h1').textContent().then(value => value === selected), 'The selected history did not open');
  await page.getByRole('tab', { name: '活动', exact: true }).click();
  await page.locator('.run-activity').waitFor({ state: 'visible' });
  await settled();
}
async function expectSummary(done, count) {
  await until(async () => (await progress().textContent()).replaceAll(/\s/g, '') === `${done}/9000`, 'The selected progress summary did not update');
  await until(() => candidateCount().textContent().then(value => value === String(count)), 'The selected result count did not update');
}
async function refresh() {
  await settled();
  const before = (await nativeState()).requests.length;
  await page.getByRole('button', { name: '刷新', exact: true }).click();
  await until(() => nativeState().then(value => value.requests.slice(before).some(request => request.path === endpoint('/updates'))), 'Manual refresh did not check selected revisions');
  await settled();
}
async function screenshot(name) {
  await page.locator('.run-summary').evaluate(element => {
    const scroller = element.closest('.workspace-content');
    if (scroller) scroller.scrollTop += element.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 85;
  });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({ path: path.join(output, name) });
}

try {
  await timed('generate-fixture', generateFixtures);
  await launch('cold-profile');
  const runs = await timed('cold-first-explicit-list', registered);
  const ledger = runs.find(run => run.id === selected);
  assert.equal(ledger.progress.completed_steps, 40);
  checks.push('cold-native-startup-registers-exactly-twelve-long-histories-with-readonly-ledger-ownership');

  const initial = await timed('first-selected-detail', () => api(endpoint()));
  const initialUpdates = await api(endpoint('/updates'));
  const initialActivity = await api(`/api/desktop/runs/${selected}/activity`);
  assert.equal(initial.observed.event_count, EVENTS_PER_HISTORY + 5);
  assert.equal(initial.observed.last_event.event, 'run_finished');
  assert.deepEqual(initial.event_tail, { valid_event_limit: 5000, valid_event_count: 5000, terminal_event: 'run_finished', reason_code: null });
  assert.equal(initial.status_reason, null);
  assert.equal(initial.observed.results.count, RESULT_ROWS);
  assert.equal(initial.observed.results.best_metric, 0.02);
  assert.equal(initial.observed.results.latest.metric, 'nan');
  assert.equal(initial.observed.results.latest.error, 'Synthetic nonfinite score');
  assert.equal(initial.observed.manifest.config.agent_steps, 9000);
  assert.equal(initialActivity.status, 'ready'); assert.equal(initialActivity.events.length, 200);
  assert.equal(initialActivity.events.at(-1).data.event, 'run_finished');
  assert(initialActivity.has_older && initialActivity.older_cursor);
  assert.equal(initialActivity.skipped_records, 2);
  assert(initialActivity.end - initialActivity.start <= 512 * 1024);
  await openSelected(); await expectSummary(40, RESULT_ROWS);
  await page.locator('.run-activity').getByText('LONG_HISTORY_INITIAL_TERMINAL', { exact: true }).waitFor();
  await screenshot('01-cold-long-history-summary.png');
  checks.push('selected-real-detail-and-ui-show-exact-event-count-last-event-progress-and-large-tsv-summary');

  // At >30,000 events, the independent result-evidence reader must retain its
  // limit. Summary optimization cannot turn an unreadable history into selection.
  const overview = await api(`/api/desktop/runs/${selected}/result`);
  assert.equal(overview.selection.status, 'unavailable');
  assert.equal(overview.selection.metric, null);
  assert.equal(overview.final_retrain.reason, 'events_unavailable');
  assert(overview.warnings.includes('events_too_large'));
  checks.push('independent-result-evidence-limits-do-not-infer-a-selected-candidate-from-long-history');

  await timed('unchanged-manual-refresh', refresh);
  assert.deepEqual(await api(endpoint()), initial);
  assert.deepEqual(await api(endpoint('/updates')), initialUpdates);
  checks.push('unchanged-long-history-refresh-keeps-observation-and-revision-values-stable');

  await appendFile(first('events.jsonl'), jsonl(event('round_finished', { completed_steps: 41, message: 'LONG_HISTORY_APPEND_41' })));
  await timed('one-event-append-refresh', refresh); await expectSummary(41, RESULT_ROWS);
  const appended = await api(endpoint());
  const appendedUpdates = await api(endpoint('/updates'));
  const appendedActivity = await api(`/api/desktop/runs/${selected}/activity`);
  assert.equal(appended.observed.event_count, EVENTS_PER_HISTORY + 6);
  assert.equal(appended.observed.last_event.completed_steps, 41);
  assert.notEqual(appendedUpdates.events.revision, initialUpdates.events.revision);
  assert.notEqual(appendedUpdates.run_revision, initialUpdates.run_revision);
  assert.equal(appendedActivity.source_id, initialActivity.source_id);
  assert.notEqual(appendedActivity.revision, initialActivity.revision);
  assert.equal(appendedActivity.events.at(-1).data.message, 'LONG_HISTORY_APPEND_41');
  assertLedgerOwnership(appended, ledger);
  await page.locator('.run-activity').getByText('LONG_HISTORY_APPEND_41', { exact: true }).waitFor();
  checks.push('tail-append-updates-selected-summary-event-and-activity-revisions-within-the-same-source');

  await appendFile(first('events.jsonl'), jsonl(event('run_failed', { completed_steps: 42, error: 'Synthetic conflicting observation; no process ran.' })));
  await refresh(); await expectSummary(42, RESULT_ROWS);
  const conflicting = await api(endpoint());
  assert.equal(conflicting.terminal_event, 'run_failed');
  assert.deepEqual(conflicting.event_tail, { valid_event_limit: 5000, valid_event_count: 5000, terminal_event: 'run_failed', reason_code: null });
  assert.equal(conflicting.observed.last_event.event, 'run_failed');
  assertLedgerOwnership(conflicting, ledger);
  checks.push('new-terminal-file-observation-never-rewrites-persisted-ledger-lifecycle-or-controls');

  await writeRows(first('events.jsonl'), 5001, row => jsonl(event('tool_observation', { step: row, message: `Tail-boundary observation ${row}` })), { append: true, suffix: jsonl(event('round_finished', { completed_steps: 43, message: 'LONG_HISTORY_AFTER_TAIL_BOUNDARY' })) });
  await timed('append-beyond-catalog-tail-refresh', refresh); await expectSummary(43, RESULT_ROWS);
  const beyondTail = await api(endpoint());
  // Terminal observations describe only the current tail. Missing evidence
  // cannot replace the ledger's successful lifecycle or its explanation.
  assert.equal(beyondTail.terminal_event, null);
  assert.deepEqual(beyondTail.event_tail, { valid_event_limit: 5000, valid_event_count: 5000, terminal_event: null, reason_code: 'no_terminal_in_tail' });
  assert.equal(beyondTail.status_reason, null);
  assert.equal(beyondTail.observed.event_count, EVENTS_PER_HISTORY + 5009);
  assert.equal(beyondTail.observed.last_event.message, 'LONG_HISTORY_AFTER_TAIL_BOUNDARY');
  assertLedgerOwnership(beyondTail, ledger);
  const eventsTail = await api(endpoint('/events?tail=2000'));
  assert.equal(eventsTail.count, EVENTS_PER_HISTORY + 5009);
  assert.equal(eventsTail.events.length, 2000);
  assert.equal(eventsTail.events.at(-1).completed_steps, 43);
  assert(!eventsTail.events.some(item => ['run_finished', 'run_failed', 'run_interrupted'].includes(item.event)));
  await page.locator('.run-activity').getByText('LONG_HISTORY_AFTER_TAIL_BOUNDARY', { exact: true }).waitFor();
  checks.push('catalog-retains-last-five-thousand-valid-object-semantics-after-terminal-records-leave-tail');

  const beforeResultsUpdates = await api(endpoint('/updates'));
  await appendFile(first('results.tsv'), `${RESULT_ROWS}\tappended-finite\t0.95\ttrue\tsucceeded\tvalid\t\t\n`);
  await timed('large-tsv-finite-append-refresh', refresh); await expectSummary(43, RESULT_ROWS + 1);
  const finite = await api(endpoint());
  assert.equal(finite.observed.results.best_metric, 0.95);
  assert.equal(finite.observed.results.latest.metric, '0.95');
  const finiteUpdates = await api(endpoint('/updates'));
  assert.notEqual(finiteUpdates.artifacts.revision, beforeResultsUpdates.artifacts.revision);
  assert.equal(finiteUpdates.events.revision, beforeResultsUpdates.events.revision);
  await appendFile(first('results.tsv'), `${RESULT_ROWS + 1}\tappended-nonfinite\t-inf\tfalse\tfailed\tinvalid\t\tSynthetic trailing invalid score\n`);
  await refresh(); await expectSummary(43, RESULT_ROWS + 2);
  const final = await api(endpoint());
  assert.equal(final.observed.results.count, RESULT_ROWS + 2);
  assert.equal(final.observed.results.best_metric, 0.95, 'Direction comes from the last finite score, not the last table row');
  assert.equal(final.observed.results.latest.metric, '-inf');
  assert.equal(final.observed.results.latest.error, 'Synthetic trailing invalid score');
  assert.deepEqual(final.observed.manifest, initial.observed.manifest);
  checks.push('large-tsv-stream-summary-preserves-count-latest-and-last-finite-metric-direction-after-append');
  await screenshot('02-tail-boundary-and-large-results.png');

  const beforeRelaunch = await api(`/api/desktop/runs/${selected}/activity`);
  await close('cold-profile');
  await launch('fresh-process-existing-profile');
  await registered(); await openSelected(); await expectSummary(43, RESULT_ROWS + 2);
  const recovered = await api(endpoint());
  assert.deepEqual(recovered.observed, final.observed);
  assertLedgerOwnership(recovered, ledger);
  assert.equal(recovered.terminal_event, null);
  assert.deepEqual(recovered.event_tail, beyondTail.event_tail);
  assert.equal(recovered.status_reason, null);
  const recoveredActivity = await api(`/api/desktop/runs/${selected}/activity`);
  assert.notEqual(recoveredActivity.source_id, beforeRelaunch.source_id, 'A new service process must own a new source identity');
  assert.deepEqual(recoveredActivity.events.map(item => item.data), beforeRelaunch.events.map(item => item.data));
  assert.equal(recoveredActivity.events.at(-1).data.message, 'LONG_HISTORY_AFTER_TAIL_BOUNDARY');
  await page.locator('.run-activity').getByText('LONG_HISTORY_AFTER_TAIL_BOUNDARY', { exact: true }).waitFor();
  checks.push('fresh-app-process-rebuilds-long-history-observations-without-duplicates-or-stale-source-identity');

  await page.getByRole('tab', { name: '运行日志', exact: true }).click();
  const consoleWindow = await api(`/api/desktop/runs/${selected}/console`);
  await until(() => page.locator('.run-log-text').textContent().then(value => value === consoleWindow.text), 'The reopened history did not acquire its own console window');
  assert(consoleWindow.text.includes('HISTORY_0_CONSOLE_SENTINEL'));
  assert(consoleWindow.end - consoleWindow.start <= 256 * 1024);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  await settled(); assert.deepEqual((await nativeState()).forbidden, []); assert.deepEqual(rendererErrors, []);
  checks.push('reopened-history-keeps-bounded-console-and-never-installs-engine-or-calls-provider');
} catch (error) {
  failure = error;
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText().catch(() => '')).slice(-6000));
  }
} finally {
  try { await close('final-cleanup'); } catch (error) { failure ||= error; }
}
if (forbiddenRequests.length) failure ||= new Error('Forbidden requests were attempted');
if (!failure) checks.push('both-app-processes-exit-with-no-raw-or-canonical-qa-profile-processes-left');
const report = {
  passed: !failure, mode: 'Real Electron and sidecar cold startup, long-history catalog/detail/stream reads and a fresh app relaunch. Synthetic isolated files; no response or lifecycle mocking. Timings are observations on this host, not universal performance guarantees. No external provider or real experiment acceptance.',
  executablePath: executablePath || 'development Electron', appInfo, profile, output, runIds, fixture, checks, launches, measurements, processChecks, requests, forbiddenRequests, rendererErrors,
  ...(failure ? { error: String(failure), stack: failure.stack } : {}),
};
await mkdir(output, { recursive: true });
await writeFile(path.join(output, failure ? 'failure.json' : 'acceptance.json'), JSON.stringify(report, null, 2) + '\n');
if (failure) throw failure;
console.log(JSON.stringify({ passed: true, checks, profile, output, fixtureBytes: fixture.totalGeneratedEventAndResultBytes, launches, processChecks }, null, 2));
