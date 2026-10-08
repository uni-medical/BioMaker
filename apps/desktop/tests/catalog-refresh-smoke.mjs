// Real Electron/sidecar cache freshness acceptance using synthetic local files.
// HTTP response contents and lifecycle state are never replaced. Only selected
// responses are delayed to exercise ownership guards. No API mutations, model
// calls, engine installation, or real user configuration are permitted.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { appendFile, mkdir, mkdtemp, readFile, realpath, rename, stat, unlink, utimes, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-catalog-refresh-'));
const profile = path.join(temporary, 'profile');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/catalog-refresh-0.1.14'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const runIds = ['qa-catalog-first', 'qa-catalog-second'];
const logs = path.join(profile, 'workspace/logs');
const directories = runIds.map(id => path.join(logs, id));
const dataset = path.join(temporary, 'synthetic-dataset');
const fixedTime = Date.parse('2026-10-09T10:00:00Z') / 1000;
const checks = [], rendererErrors = [], replacements = [], restarts = [];
let application, page;
const event = (name, fields = {}) => ({ ts: '2026-10-09T10:00:00Z', event: name, ...fields });
const jsonl = values => values.map(value => JSON.stringify(value)).join('\n') + '\n';
const chosenCode = '# Synthetic catalog fixture; never executed.\nprint("catalog")\n';
for (const directory of [profile, output, dataset, ...directories]) await mkdir(directory, { recursive: true });
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
for (let index = 0; index < runIds.length; index++) {
  const files = {
    'run_manifest.json': JSON.stringify({ created_at: '2026-10-09T10:00:00Z', data_dir: dataset, task_description: `Synthetic catalog task ${index}; no model training.`, config: { agent_steps: 5, final_retrain_enabled: false, model_for_all_agents: 'qa-never-called' } }),
    'events.jsonl': jsonl([event('run_started'), event('round_finished', { completed_steps: 1 }), event('step_saved', { node_id: 'catalog-choice', best_node_id: 'catalog-choice' }), event('run_finished')]),
    'results.tsv': 'step\tnode_id\tmetric\tmaximize\tstatus\n0\tcatalog-choice\t0.41\ttrue\tsucceeded\n',
    'journal.json': JSON.stringify({ nodes: [{ id: 'catalog-choice', step: 0, is_buggy: false, code: chosenCode, plan: 'Synthetic catalog fixture.', metric: { value: 0.41, maximize: true, name: 'Recorded accuracy' } }] }),
    'best_solution.py': chosenCode,
    'report.md': '# Synthetic catalog report\n',
    [`${runIds[index]}.log`]: Array.from({ length: 500 }, (_, row) => `Run ${index} synthetic line ${row}\n`).join('') + `RUN_${index}_INITIAL_SENTINEL\n`,
  };
  for (const [name, content] of Object.entries(files)) {
    const target = path.join(directories[index], name);
    await writeFile(target, content);
    await utimes(target, fixedTime, fixedTime);
  }
}
const first = name => path.join(directories[0], name);
const endpoint = (suffix = '') => `/api/runs/${runIds[0]}${suffix}`;
const reader = () => page.locator('.run-log-reader');
const logText = () => reader().locator('.run-log-text').textContent();
const progress = () => page.locator('.run-summary > div').filter({ hasText: '已记录探索候选 / 配置上限' }).locator('strong');
const candidateCount = () => page.locator('.run-summary > div').filter({ hasText: '已记录的候选方案' }).locator('strong');
const api = route => page.evaluate(route => window.biomaker.request(route), route);
const frame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const configure = values => application.evaluate((_electron, values) => Object.assign(globalThis.__catalogQA, values), values);
const nativeState = () => application.evaluate(() => {
  const qa = globalThis.__catalogQA;
  return { requests: qa.requests, forbidden: qa.forbidden, inflight: qa.inflight, held: qa.held.length };
});
async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 40));
  }
}
async function settled() {
  await until(() => nativeState().then(value => value.inflight === 0), 'Read-only requests did not settle');
  await frame();
}
async function refreshCatalog() {
  await settled();
  const before = (await nativeState()).requests.length;
  await page.getByRole('button', { name: '刷新', exact: true }).click();
  await until(() => nativeState().then(value => value.requests.slice(before).some(request => request.path === '/api/runs')), 'Catalog refresh did not read the real API');
  await until(() => nativeState().then(value => value.requests.slice(before).some(request => request.path.endsWith('/updates'))), 'Catalog refresh did not reach detail revision checking');
  await settled();
  return (await nativeState()).requests.slice(before);
}
async function openRun(id, tab = '结果') {
  await application.evaluate(({ BrowserWindow }, id) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:open-run', id), id);
  await until(() => page.locator('.run-intro h1').textContent().then(value => value === id), 'Selected run heading did not change');
  await page.getByRole('tab', { name: tab, exact: true }).click();
  await settled();
}
async function expectProgress(value) {
  await until(async () => await progress().count() === 1 && (await progress().textContent()).replace(/\s/g, '') === value, `UI progress did not become ${value}`);
}
async function expectMetric(value) {
  await until(async () => await page.locator('.run-results').getAttribute('aria-busy') === 'false' && (await page.locator('.result-facts').innerText()).includes(value), `Result overview did not show ${value}`);
}
async function screenshot(name, locator = page.locator('.run-summary')) {
  if (await locator.count()) await locator.first().evaluate(element => {
    const scroller = element.closest('.workspace-content');
    if (scroller) scroller.scrollTop += element.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 90;
  });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'The page overflowed horizontally');
  await page.screenshot({ path: path.join(output, name) });
}
async function rewriteSameMetadata(target, transform, replace = false) {
  const original = await readFile(target, 'utf8');
  const next = transform(original);
  assert.notEqual(next, original, 'Freshness fixture must change contents');
  assert.equal(Buffer.byteLength(next), Buffer.byteLength(original), 'Freshness fixture must preserve byte length');
  const before = await stat(target, { bigint: true });
  const destination = replace ? path.join(temporary, `replacement-${replacements.length}`) : target;
  await writeFile(destination, next);
  await utimes(destination, Number(before.atimeNs) / 1e9, Number(before.mtimeNs) / 1e9);
  if (replace) await rename(destination, target);
  const after = await stat(target, { bigint: true });
  assert.equal(after.size, before.size);
  assert.equal(after.mtimeNs, before.mtimeNs, 'The test must preserve exact nanosecond mtime');
  if (replace) assert.notEqual(after.ino, before.ino); else assert.equal(after.ino, before.ino);
  assert.notEqual(after.ctimeNs, before.ctimeNs);
  replacements.push({ file: path.basename(target), replacedInode: replace, size: String(after.size), mtimeNs: String(after.mtimeNs), beforeInode: String(before.ino), afterInode: String(after.ino), beforeCtimeNs: String(before.ctimeNs), afterCtimeNs: String(after.ctimeNs) });
}
async function holdPath(value) { await configure({ holdPath: value }); }
async function held() { await until(() => nativeState().then(value => value.held === 1), 'The real response was not held'); }
async function releaseHeld() {
  await application.evaluate(() => { for (const release of globalThis.__catalogQA.held.splice(0)) release(); });
  await settled();
}
async function restartService() {
  const before = await page.evaluate(() => window.biomaker.getServiceState());
  const canonical = await realpath(profile);
  const rows = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n')
    .filter(line => line.includes('/service/main.py') && line.includes('python') && (line.includes(profile) || line.includes(canonical)));
  assert.equal(rows.length, 1, 'Only the exact isolated QA sidecar may be stopped');
  process.kill(Number(rows[0].trim().split(/\s+/)[0]), 'SIGKILL');
  await until(async () => { const value = await page.evaluate(() => window.biomaker.getServiceState()); return value.status === 'ready' && value.revision > before.revision; }, 'The sidecar did not recover with a newer generation', 60_000);
  restarts.push({ from: before.revision, to: (await page.evaluate(() => window.biomaker.getServiceState())).revision });
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  await application.evaluate(() => {
    const original = globalThis.fetch;
    const qa = globalThis.__catalogQA = { requests: [], forbidden: [], inflight: 0, held: [], holdPath: null };
    globalThis.fetch = async (input, options) => {
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      if (url.hostname !== '127.0.0.1' || !['GET', 'HEAD'].includes(method)) {
        qa.forbidden.push({ path: url.pathname, method });
        throw new Error('Catalog QA forbids API mutations, provider calls and engine installation');
      }
      const entry = { path: url.pathname, query: url.search, method };
      qa.requests.push(entry); qa.inflight++;
      const hold = qa.holdPath === url.pathname;
      if (hold) qa.holdPath = null;
      try {
        const response = await original(input, options);
        entry.status = response.status;
        if (!hold) return response;
        const body = await response.arrayBuffer();
        await new Promise(resolve => qa.held.push(resolve));
        return new Response(body, { status: response.status, headers: response.headers });
      } finally { qa.inflight--; }
    };
  });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  await until(async () => { try { return (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready'; } catch { return false; } }, 'The real sidecar did not start', 60_000);
  await page.locator('.sidebar-settings').waitFor();
  await settled();
  const initialList = await api('/api/runs');
  assert.deepEqual(initialList.map(run => run.id).sort(), [...runIds].sort());
  const initialDetail = await api(endpoint());
  const initialUpdates = await api(endpoint('/updates'));
  for (let repeat = 0; repeat < 4; repeat++) {
    assert.deepEqual(await api('/api/runs'), initialList);
    assert.deepEqual(await api(endpoint()), initialDetail);
    assert.deepEqual(await api(endpoint('/updates')), initialUpdates);
  }
  checks.push('repeated-real-catalog-detail-and-revision-reads-remain-stable');

  await openRun(runIds[0]); await expectProgress('1/5'); await expectMetric('0.41');
  const unchangedReads = await refreshCatalog();
  assert(!unchangedReads.some(request => request.path === endpoint()), 'An unchanged UI refresh unnecessarily reloaded detail');
  assert(!unchangedReads.some(request => request.path.endsWith('/result')), 'An unchanged UI refresh unnecessarily reloaded result evidence');
  checks.push('unchanged-ui-catalog-refresh-retains-detail-and-result-without-reloading');

  await appendFile(first('events.jsonl'), jsonl([event('round_finished', { completed_steps: 2 })]));
  await refreshCatalog(); await expectProgress('2/5');
  assert.equal((await api(endpoint())).observed.event_count, 5);
  assert.equal((await api('/api/runs')).find(run => run.id === runIds[0]).progress.completed_steps, 2);
  assert.notEqual((await api(endpoint('/updates'))).events.revision, initialUpdates.events.revision);
  checks.push('appended-events-refresh-catalog-progress-detail-and-rendered-progress');

  await utimes(first('events.jsonl'), fixedTime, fixedTime);
  await refreshCatalog();
  const eventRevision = (await api(endpoint('/updates'))).events.revision;
  await rewriteSameMetadata(first('events.jsonl'), value => value.replace('"completed_steps":2', '"completed_steps":3'));
  await refreshCatalog(); await expectProgress('3/5');
  const changedDetail = await api(endpoint());
  assert.equal(changedDetail.observed.event_count, 5);
  assert.equal(changedDetail.observed.last_event.completed_steps, 3);
  assert.notEqual((await api(endpoint('/updates'))).events.revision, eventRevision);
  checks.push('same-inode-size-and-exact-mtime-event-rewrite-invalidates-cache-and-ui-revision');

  await appendFile(first('results.tsv'), '1\tcatalog-next\t0.52\ttrue\tsucceeded\n');
  await refreshCatalog();
  await until(() => candidateCount().textContent().then(value => value === '2'), 'Appended results were not rendered');
  assert.equal((await api(endpoint())).observed.results.best_metric, 0.52);
  await utimes(first('results.tsv'), fixedTime, fixedTime); await refreshCatalog();
  const resultsRevision = (await api(endpoint('/updates'))).artifacts.revision;
  await rewriteSameMetadata(first('results.tsv'), value => value.replace('0.52', '0.83'), true);
  await refreshCatalog();
  assert.equal((await api(endpoint())).observed.results.best_metric, 0.83);
  assert.equal((await api(endpoint())).observed.results.latest.metric, '0.83');
  assert.notEqual((await api(endpoint('/updates'))).artifacts.revision, resultsRevision);
  checks.push('result-append-and-same-size-exact-mtime-file-replacement-update-observations');

  await rewriteSameMetadata(first('journal.json'), value => value.replace('0.41', '0.87'), true);
  await refreshCatalog(); await expectMetric('0.87');
  checks.push('same-size-exact-mtime-journal-replacement-refreshes-visible-result-evidence');

  await rewriteSameMetadata(first('run_manifest.json'), value => value.replace('task 0', 'task 9').replace('"agent_steps":5', '"agent_steps":9'), true);
  await refreshCatalog(); await expectProgress('3/9');
  await page.locator('.task-message').getByText('Synthetic catalog task 9; no model training.', { exact: true }).waitFor();
  assert.equal((await api(endpoint())).observed.manifest.config.agent_steps, 9);
  assert.equal((await api('/api/runs')).find(run => run.id === runIds[0]).options.agent_steps, 9);
  await screenshot('01-fresh-progress-and-results.png');
  checks.push('same-size-exact-mtime-manifest-replacement-refreshes-list-detail-and-task-ui');

  const savedEvents = await readFile(first('events.jsonl'));
  await unlink(first('events.jsonl')); await refreshCatalog();
  const noEvents = await api(endpoint());
  assert.equal(noEvents.observed.event_count, 0); assert.equal(noEvents.observed.last_event, null);
  assert.equal(noEvents.progress.completed_steps, null); assert.equal(await progress().count(), 0);
  await writeFile(first('events.jsonl'), savedEvents); await refreshCatalog(); await expectProgress('3/9');
  const savedResults = await readFile(first('results.tsv'));
  await unlink(first('results.tsv')); await refreshCatalog();
  assert.equal((await api(endpoint())).observed.results.count, 0); assert.equal(await candidateCount().count(), 0);
  const resultRow = page.locator('.result-resource').filter({ has: page.locator('.result-resource-meta').getByText('results.tsv', { exact: true }) });
  await resultRow.getByText('未保存此文件', { exact: true }).waitFor();
  await screenshot('02-deleted-result-explicitly-unavailable.png', resultRow);
  await writeFile(first('results.tsv'), savedResults); await refreshCatalog();
  await until(() => candidateCount().textContent().then(value => value === '2'), 'Restored results were not rendered');
  checks.push('deleted-and-restored-events-and-results-never-replay-cached-observations');

  const savedManifest = await readFile(first('run_manifest.json'));
  await unlink(first('run_manifest.json')); await refreshCatalog();
  assert.equal((await api(endpoint())).observed.manifest, null);
  assert.equal((await api('/api/runs')).find(run => run.id === runIds[0]).options.agent_steps, 5, 'Missing manifest must expose the persisted ledger fallback, not the newer cached observation');
  await writeFile(first('run_manifest.json'), savedManifest); await refreshCatalog(); await expectProgress('3/9');
  checks.push('deleted-manifest-uses-ledger-fallback-and-restoration-recovers-current-observations');

  await page.getByRole('tab', { name: '运行日志', exact: true }).click();
  assert.equal(await reader().getAttribute('data-following'), 'false');
  const pausedText = await logText();
  assert(pausedText.includes('RUN_0_INITIAL_SENTINEL'));
  const heldDirectory = path.join(temporary, 'temporarily-unavailable-log-directory');
  await rename(directories[0], heldDirectory); await refreshCatalog();
  const missingDirectory = await api(endpoint());
  assert.deepEqual(missingDirectory.artifacts, []); assert.equal(missingDirectory.observed.event_count, 0);
  assert.equal(missingDirectory.observed.results.count, 0); assert.equal(missingDirectory.observed.manifest, null);
  assert.equal(await logText(), pausedText);
  await rename(heldDirectory, directories[0]); await refreshCatalog();
  assert.equal((await api(endpoint())).observed.results.best_metric, 0.83);
  assert.equal(await logText(), pausedText);
  checks.push('missing-and-restored-log-directory-never-replays-cached-detail-and-keeps-paused-log');

  await appendFile(first(`${runIds[0]}.log`), 'RUN_0_AFTER_CACHE_REFRESH\n');
  await refreshCatalog(); assert.equal(await logText(), pausedText);
  await holdPath(endpoint('/updates'));
  await page.getByRole('button', { name: '刷新', exact: true }).click(); await held();
  await restartService(); await releaseHeld();
  assert.equal(await logText(), pausedText); assert.equal(await reader().getAttribute('data-following'), 'false');
  await reader().getByRole('button', { name: '跟随最新', exact: true }).click();
  await until(async () => (await logText()).includes('RUN_0_AFTER_CACHE_REFRESH'), 'Explicit follow did not adopt the new-generation log');
  await reader().getByRole('button', { name: '暂停跟随', exact: true }).click();
  await screenshot('03-service-restart-preserves-reader.png', reader());
  checks.push('real-service-restart-discards-held-old-generation-detail-and-preserves-paused-window');

  await holdPath(`/api/desktop/runs/${runIds[1]}/console`);
  await application.evaluate(({ BrowserWindow }, id) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:open-run', id), runIds[1]);
  await until(() => reader().getAttribute('data-run-id').then(value => value === runIds[1]), 'Second run did not own its reader');
  await page.getByRole('tab', { name: '运行日志', exact: true }).click(); await held();
  assert(!(await logText()).includes('RUN_0_'), 'Second run exposed first-run text while its response was held');
  await releaseHeld();
  await until(async () => (await logText()).includes('RUN_1_INITIAL_SENTINEL'), 'Second run never acquired its own log');
  assert(!(await logText()).includes('RUN_0_'));
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  await screenshot('04-cross-run-ownership-compact.png', reader());
  await openRun(runIds[0], '运行日志');
  assert((await logText()).includes('RUN_0_AFTER_CACHE_REFRESH')); assert(!(await logText()).includes('RUN_1_'));
  checks.push('cross-run-delayed-response-never-exposes-previous-run-log-after-cache-refresh');

  await openRun(runIds[0]); await expectProgress('3/9'); await expectMetric('0.87');
  assert.equal((await api(endpoint())).observed.results.best_metric, 0.83);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  assert.deepEqual((await api('/api/runs')).map(run => run.id).sort(), [...runIds].sort());
  await settled();
  const native = await nativeState();
  assert.deepEqual(native.forbidden, []); assert.equal(native.held, 0); assert.deepEqual(rendererErrors, []);
  checks.push('no-api-mutations-provider-engine-install-real-experiment-or-renderer-errors');
  const report = { passed: true, mode: 'Real Electron and sidecar catalog/detail/revision/result reads. Synthetic isolated files; only response timing is intercepted. No cache performance or external model acceptance claim.', executablePath: executablePath || 'development Electron', profile, output, runIds, checks, replacements, restarts, requests: native.requests, forbiddenRequests: native.forbidden, rendererErrors };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ passed: true, checks, profile, output }, null, 2));
} catch (error) {
  const native = application ? await nativeState().catch(() => null) : null;
  await writeFile(path.join(output, 'failure.json'), JSON.stringify({ passed: false, profile, checks, replacements, restarts, error: String(error), rendererErrors, native }, null, 2) + '\n');
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-6_000));
  }
  throw error;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => {
      for (const release of globalThis.__catalogQA?.held || []) release();
      for (const window of BrowserWindow.getAllWindows()) window.destroy();
    }).catch(() => {});
    await application.close();
  }
}
