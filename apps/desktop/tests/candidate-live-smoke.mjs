// Real Electron + real sidecar candidate lifecycle acceptance. This script
// writes only synthetic files and the isolated profile's SQLite run ledger.
// All candidate/lifecycle HTTP responses remain untouched. No model request,
// training, engine installation, build, or user's existing profile is involved.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { appendFile, mkdir, mkdtemp, realpath, rename, stat, writeFile } from 'node:fs/promises';
import { DatabaseSync } from 'node:sqlite';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-candidate-live-'));
const profile = path.join(temporary, 'profile');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/candidate-live'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1', NO_PROXY: '127.0.0.1,localhost' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const ids = { live: 'qa-candidate-live', stopped: 'qa-candidate-stopped', failed: 'qa-candidate-failed', old: 'qa-candidate-old', large: 'qa-candidate-large' };
const checks = [], rendererErrors = [], restarts = [], ledgerChanges = [];
let application, page, database, appInfo, nativeReceipt;
const directory = id => path.join(profile, 'workspace/logs', id);
const event = (name, fields = {}) => ({ ts: '2026-10-09T10:00:00Z', event: name, ...fields });
const events = (id, values) => appendFile(path.join(directory(id), 'events.jsonl'), values.map(value => JSON.stringify(value) + '\n').join(''));
const api = endpoint => page.evaluate(endpoint => window.biomaker.request(endpoint), endpoint);
const candidates = () => page.locator('.candidates-view');
const content = () => page.locator('.candidate-content');
const detail = () => page.locator('.candidate-detail');
const row = number => page.locator('.candidate-row').filter({ has: page.locator('.candidate-row-title strong').filter({ hasText: new RegExp(`^#${number}(?:\\s|$)`) }) });
const selected = () => page.locator('.candidate-row[aria-pressed="true"]');
const currentCode = () => content().locator('pre').textContent();
const frame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));

async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 40));
  }
}
async function atomic(id, filename, value) {
  const target = path.join(directory(id), filename);
  const pending = `${target}.qa-pending`;
  await writeFile(pending, typeof value === 'string' ? value : JSON.stringify(value));
  await rename(pending, target);
}
function setStatus(id, status) {
  assert(Object.values(ids).includes(id));
  const record = JSON.parse(database.prepare('SELECT record_json FROM runs WHERE run_id = ?').get(id).record_json);
  assert.equal(record.managed, false);
  assert.equal(record.read_only, true);
  Object.assign(record, { status, detailed_status: status, status_reason: null, error: null });
  database.prepare('UPDATE runs SET record_json = ? WHERE run_id = ?').run(JSON.stringify(record), id);
  ledgerChanges.push({ id, status });
}
async function phase(number, expected) {
  await until(() => row(number).locator(`.candidate-verdict.${expected}`).count().then(count => count === 1), `Candidate #${number} did not become ${expected}`);
}
async function assertSelection(number, section = 'Code') {
  assert.equal(await row(number).getAttribute('aria-pressed'), 'true');
  assert.equal(await selected().count(), 1);
  assert.equal(await detail().locator('h3').textContent(), `Candidate #${number} (generation order)`);
  assert.equal(await detail().getByRole('tab', { name: section, exact: true }).getAttribute('aria-selected'), 'true');
}
async function openRun(id, tab = 'candidates') {
  await application.evaluate(({ BrowserWindow }, id) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:open-run', id), id);
  await until(() => page.locator('.run-intro h1').textContent().then(value => value === id), `Did not open ${id}`);
  await page.locator(`#run-tab-${tab}`).click();
  if (tab === 'candidates') await candidates().waitFor();
  else await until(() => page.locator('.run-results').getAttribute('aria-busy').then(value => value === 'false'), 'Results did not load');
}
async function screenshot(name) {
  await frame();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({ path: path.join(output, name) });
}
async function seed() {
  await mkdir(output, { recursive: true });
  await mkdir(profile, { recursive: true });
  await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'en', theme: 'light', notifications: false, directories: [] }));
  for (const id of Object.values(ids)) {
    await mkdir(directory(id), { recursive: true });
    await atomic(id, 'run_manifest.json', { created_at: '2026-10-09T10:00:00Z', data_dir: path.join(temporary, 'synthetic-dataset'), task_description: 'Synthetic candidate lifecycle UI acceptance. No model training occurred.', config: { agent_steps: 1510, final_retrain_enabled: false, model_for_all_agents: 'qa-never-called' } });
    await events(id, [event('run_started')]);
  }
  await events(ids.live, [event('node_generation_started', { placeholder_id: 'first', scheduled_step: 0, selection_reason: 'FIRST_GENERATION_REASON' })]);
  for (const id of [ids.stopped, ids.failed]) await events(id, [event('node_generation_started', { placeholder_id: 'pending', scheduled_step: 0 })]);
  await atomic(ids.old, 'journal.json', { nodes: [
    { id: 'old-valid', step: 0, is_buggy: false, code: '# OLD_JOURNAL_ONLY', plan: 'Legacy journal only.', metric: { value: 0, maximize: false } },
    { id: 'old-failed', step: 1, is_buggy: true, code: '# OLD_FAILURE', exc_type: 'LegacyValidationError', _term_out: ['OLD_FAILURE_OUTPUT'] },
  ], node2parent: { 'old-failed': 'old-valid' } });
  await events(ids.old, [event('run_finished')]);
  const large = Array.from({ length: 1510 }, (_, step) => ({ id: `large-${step}`, step, is_buggy: false, code: `# LARGE_CODE_${step}\n${'# bounded synthetic program\n'.repeat(110)}`, metric: { value: step / 10000, maximize: true } }));
  await atomic(ids.large, 'journal.json', { nodes: large, node2parent: {} });
  await atomic(ids.large, 'best_solution.py', large.at(-1).code);
  await events(ids.large, [event('step_saved', { node_id: large.at(-1).id, best_node_id: large.at(-1).id }), event('run_finished')]);
  assert((await stat(path.join(directory(ids.large), 'journal.json'))).size > 4 * 1024 * 1024);
  await assert.rejects(stat(path.join(directory(ids.live), 'journal.json')), { code: 'ENOENT' });
}
async function restartWhileSelected() {
  const before = await page.evaluate(() => window.biomaker.getServiceState());
  const canonical = await realpath(profile);
  const matches = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes('/service/main.py') && line.includes('python') && (line.includes(profile) || line.includes(canonical)));
  assert.equal(matches.length, 1, 'Only the exact isolated QA sidecar may be stopped');
  await application.evaluate(() => { globalThis.__candidateLiveQA.holdHealth = true; });
  process.kill(Number(matches[0].trim().split(/\s+/)[0]), 'SIGKILL');
  await until(() => page.evaluate(() => window.biomaker.getServiceState()).then(value => value.status === 'reconnecting'), 'Service did not enter reconnecting');
  await until(() => candidates().innerText().then(value => value.includes('Updates paused.')), 'Candidate records were not retained during disconnect');
  await assertSelection(2);
  assert.match(await currentCode(), /SECOND_LIVE_CODE/);
  await atomic(ids.live, 'candidates_live.json', { nodes: [first, { ...second, phase: 'running', code: '# RECOVERED_SECOND_CODE' }] });
  await until(() => application.evaluate(() => globalThis.__candidateLiveQA.held.length > 0), 'Restart did not reach the authenticated health check', 60_000);
  await screenshot('03-disconnected-records-preserved.png');
  await application.evaluate(() => { const qa = globalThis.__candidateLiveQA; qa.holdHealth = false; for (const release of qa.held.splice(0)) release(); });
  await until(() => page.evaluate(() => window.biomaker.getServiceState()).then(value => value.status === 'ready' && value.revision > before.revision), 'Service did not recover', 60_000);
  await until(() => currentCode().then(value => value.includes('RECOVERED_SECOND_CODE')), 'Recovery did not automatically read the changed real snapshot');
  await assertSelection(2);
  restarts.push({ from: before.revision, to: (await page.evaluate(() => window.biomaker.getServiceState())).revision });
}
const first = { id: 'first', scheduled_step: 0, phase: 'queued', code: '# FIRST_LIVE_CODE', plan: 'First generated plan.', agent_name: 'Draft' };
const second = { id: 'second', scheduled_step: 1, phase: 'queued', code: '# SECOND_LIVE_CODE', plan: 'Second generated plan.', agent_name: 'Draft' };

try {
  await seed();
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  await application.evaluate(({ BrowserWindow }) => {
    const original = globalThis.fetch;
    const qa = globalThis.__candidateLiveQA = { requests: [], forbidden: [], holdHealth: false, held: [] };
    globalThis.fetch = async (input, options) => {
      const url = new URL(typeof input === 'string' ? input : input.url || String(input));
      const method = String(options?.method || (typeof input === 'object' && input.method) || 'GET').toUpperCase();
      if (url.hostname !== '127.0.0.1' || !['GET', 'HEAD'].includes(method)) {
        qa.forbidden.push({ path: url.pathname, method });
        throw new Error('Candidate QA forbids model requests, installation and API mutations');
      }
      const entry = { path: url.pathname, query: url.search, method };
      qa.requests.push(entry);
      const response = await original(input, options);
      entry.status = response.status;
      if (qa.holdHealth && url.pathname === '/api/health') await new Promise(resolve => qa.held.push(resolve));
      return response;
    };
  });
  page = await application.firstWindow();
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1280, 820));
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  await until(async () => {
    const state = await page.evaluate(() => window.biomaker.getServiceState()).catch(() => null);
    if (state?.status === 'failed') throw new Error(`The real sidecar failed: ${state.message}`);
    return state?.status === 'ready';
  }, 'Sidecar did not start', 60_000);
  await page.locator('.sidebar-settings').waitFor();
  appInfo = await page.evaluate(() => window.biomaker.getAppInfo());
  database = new DatabaseSync(path.join(profile, 'state/runs.sqlite3'));
  database.exec('PRAGMA busy_timeout = 5000');
  assert.deepEqual((await api('/api/runs')).map(run => run.id).sort(), Object.values(ids).sort());
  for (const id of [ids.live, ids.stopped, ids.failed]) setStatus(id, 'running');
  assert.equal((await api(`/api/runs/${ids.live}`)).status, 'running');
  checks.push('isolated-readonly-synthetic-ledger-with-real-unmodified-lifecycle-and-candidate-responses');

  await openRun(ids.live);
  await phase(1, 'generating');
  assert.match(await content().innerText(), /FIRST_GENERATION_REASON/);
  await detail().getByRole('tab', { name: 'Code', exact: true }).click();
  assert.match(await currentCode(), /Code will appear here after generation/);
  assert.equal((await api(`/api/desktop/runs/${ids.live}/candidates`)).nodes[0].id, 'first');
  await screenshot('01-generation-before-journal.png');
  await atomic(ids.live, 'candidates_live.json', { nodes: [first] });
  // The real engine assigns node.id = placeholder.id before this event.
  await events(ids.live, [event('node_generation_finished', { node_id: 'first', scheduled_step: 0 }), event('node_queued_for_execution', { node_id: 'first' })]);
  await phase(1, 'queued');
  await until(() => currentCode().then(value => value.includes('FIRST_LIVE_CODE')), 'Generated code required a manual refresh');
  await assertSelection(1);
  assert.equal(await page.locator('.candidate-row').count(), 1);
  await assert.rejects(stat(path.join(directory(ids.live), 'journal.json')), { code: 'ENOENT' });
  checks.push('event-only-placeholder-becomes-same-selected-generated-id-and-code-before-first-journal-without-refresh');

  await events(ids.live, [event('node_generation_started', { placeholder_id: 'second', scheduled_step: 1 })]);
  await phase(2, 'generating');
  await assertSelection(1);
  await row(2).click();
  await detail().getByRole('tab', { name: 'Code', exact: true }).click();
  await atomic(ids.live, 'candidates_live.json', { nodes: [first, second] });
  await events(ids.live, [event('node_generation_finished', { node_id: 'second', scheduled_step: 1 })]);
  await phase(2, 'queued');
  await until(() => currentCode().then(value => value.includes('SECOND_LIVE_CODE')), 'Second placeholder selection did not follow its generated code');
  await assertSelection(2);
  await events(ids.live, [event('execution_started', { node_id: 'second' }), event('node_submitted', { node_id: 'second' })]);
  await phase(2, 'running');
  await assertSelection(2);
  assert.equal(await row(2).locator('.candidate-verdict small').count(), 0);
  await screenshot('02-running-code-without-score.png');
  checks.push('selected-candidate-and-code-tab-survive-new-rows-and-out-of-order-running-submitted-events-without-invented-metric');

  await atomic(ids.live, 'candidates_live.json', '{"nodes":');
  await until(() => candidates().innerText().then(value => value.includes('Updates paused.')), 'Unreadable snapshot did not display a paused update notice');
  await assertSelection(2);
  assert.match(await currentCode(), /SECOND_LIVE_CODE/);
  await atomic(ids.live, 'candidates_live.json', { nodes: [first, second] });
  await until(() => candidates().innerText().then(value => !value.includes('Updates paused.')), 'Snapshot repair did not recover automatically');
  await restartWhileSelected();
  checks.push('malformed-live-file-and-real-sidecar-restart-preserve-selected-code-and-recover-automatically');

  await events(ids.live, [event('execution_finished', { node_id: 'second' }), event('candidate_validation_started', { node_id: 'second' }), event('candidate_evaluation_started', { node_id: 'second' }), event('node_evaluated', { node_id: 'second', is_buggy: false, metric: 0.99 })]);
  await phase(2, 'evaluating');
  await assertSelection(2);
  assert.equal(await detail().locator('.candidate-facts strong').first().textContent(), '—');
  await atomic(ids.live, 'journal.json', { nodes: [{ id: 'second', step: 0, is_buggy: false, code: '# SAVED_AUTHORITATIVE_CODE', metric: { value: 0.81234, maximize: true }, exec_time: 3, _term_out: ['SAVED_EXECUTION_OUTPUT'] }], node2parent: {} });
  // Saved completion step 0 differs from scheduled step 1. Match Results #1.
  await until(() => currentCode().then(value => value.includes('SAVED_AUTHORITATIVE_CODE')), 'Saved journal did not replace transient code');
  assert.equal(await detail().locator('h3').textContent(), 'Candidate #1');
  assert.equal(await detail().locator('.candidate-status.valid').count(), 1);
  assert.equal(await detail().locator('.candidate-facts strong').first().textContent(), '0.81234');
  assert.equal(await selected().count(), 1);
  await events(ids.live, [event('node_generation_failed', { placeholder_id: 'first', exc_type: 'SyntheticGenerationError' })]);
  await until(() => page.locator('.candidate-verdict.failed').count().then(value => value === 1), 'Generation failure did not appear');
  assert.match(await currentCode(), /SAVED_AUTHORITATIVE_CODE/);
  await events(ids.live, [event('run_finished', { completed_steps: 1 })]);
  setStatus(ids.live, 'complete');
  await screenshot('04-journal-authoritative-number-and-metric.png');
  checks.push('evaluation-remains-unscored-until-journal-and-saved-number-score-code-are-authoritative-with-selection-preserved');

  await openRun(ids.stopped);
  await phase(1, 'generating');
  await events(ids.stopped, [event('run_interrupted')]);
  setStatus(ids.stopped, 'stopped');
  await phase(1, 'interrupted');
  assert.equal(await page.locator('.candidate-verdict.valid').count(), 0);
  await openRun(ids.failed);
  await phase(1, 'generating');
  await events(ids.failed, [event('node_generation_failed', { placeholder_id: 'pending', exc_type: 'SyntheticProviderError' }), event('run_failed', { error: 'Synthetic failure' })]);
  setStatus(ids.failed, 'failed');
  await phase(1, 'failed');
  assert.match(await content().innerText(), /SyntheticProviderError/);
  assert.equal(await detail().locator('.candidate-facts strong').first().textContent(), '—');
  checks.push('stop-and-failure-end-transient-states-without-fabricating-validity-or-metrics');

  await openRun(ids.old);
  await phase(1, 'valid');
  await row(1).click();
  assert.equal(await detail().locator('.candidate-facts strong').first().textContent(), '0');
  await row(2).click();
  await detail().getByRole('tab', { name: 'Output', exact: true }).click();
  assert.match(await currentCode(), /OLD_FAILURE_OUTPUT/);
  await assert.rejects(stat(path.join(directory(ids.old), 'candidates_live.json')), { code: 'ENOENT' });
  // An oversized activity source must not hide an independently valid journal.
  await atomic(ids.old, 'events.jsonl', 'x'.repeat(16 * 1024 * 1024 + 1));
  await openRun(ids.large);
  await openRun(ids.old);
  await phase(1, 'valid');
  await row(1).click();
  assert.match(await candidates().innerText(), /Some activity could not be read/);
  assert(!String(await candidates().innerText()).includes('Updates paused.'));
  checks.push('journal-only-legacy-history-and-oversized-events-still-show-valid-saved-records-with-a-source-warning');

  await openRun(ids.large, 'results');
  await page.locator('.result-selection').getByRole('button', { name: 'View candidate', exact: true }).click();
  await candidates().waitFor();
  await until(() => detail().locator('h3').textContent().then(value => value === 'Candidate #1510'), 'Results-selected ID outside the first 1500 did not open');
  assert.equal(await row(1510).getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('.candidate-row').count(), 1501);
  await detail().getByRole('tab', { name: 'Code', exact: true }).click();
  assert.match(await currentCode(), /LARGE_CODE_1509/);
  const largeRecords = await api(`/api/desktop/runs/${ids.large}/candidates?requested_id=large-1509`);
  assert.equal(largeRecords.total, 1510);
  assert(largeRecords.nodes.some(node => node.id === 'large-1509'));
  await screenshot('05-large-history-results-selection.png');
  checks.push('multi-megabyte-journal-window-includes-and-selects-result-id-beyond-first-1500-with-real-requested-id-api');

  nativeReceipt = await application.evaluate(() => ({ requests: globalThis.__candidateLiveQA.requests, forbidden: globalThis.__candidateLiveQA.forbidden }));
  assert(nativeReceipt.requests.filter(request => request.path === `/api/desktop/runs/${ids.live}/candidates`).length >= 10);
  assert(nativeReceipt.requests.some(request => request.path === `/api/desktop/runs/${ids.large}/candidates` && request.query === '?requested_id=large-1509'));
  assert.deepEqual(nativeReceipt.forbidden, []);
  assert.deepEqual(rendererErrors, []);
  const receipt = { passed: true, mode: 'Real Electron and sidecar, unchanged HTTP responses, synthetic filesystem and read-only ledger fixtures, one exact-profile real restart; no training or model calls.', executablePath: executablePath || 'development Electron', appInfo, profile, checks, ledgerChanges, restarts, ...nativeReceipt, rendererErrors };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(receipt, null, 2));
  console.log(JSON.stringify({ passed: true, checks, receipt: path.join(output, 'acceptance.json') }, null, 2));
} catch (error) {
  await writeFile(path.join(output, 'failure.json'), JSON.stringify({ passed: false, profile, checks, ledgerChanges, restarts, error: String(error), rendererErrors }, null, 2));
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-5000));
  }
  throw error;
} finally {
  database?.close();
  if (application) {
    await application.evaluate(({ BrowserWindow }) => { const qa = globalThis.__candidateLiveQA; if (qa) { qa.holdHealth = false; for (const release of qa.held.splice(0)) release(); } for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await application.close();
    const canonical = await realpath(profile);
    await until(() => !execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').some(line => line.includes(profile) || line.includes(canonical)), 'Exact isolated profile process remained after exit', 15_000);
  }
}
