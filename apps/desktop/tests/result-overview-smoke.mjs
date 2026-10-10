// Real Electron, portable sidecar result parsing, history import, and native
// artifact preview/export against isolated synthetic logs. Only lifecycle UI
// states, snapshot availability, one HTTP failure, and response delays are
// intercepted. No engine installation, provider request, or actual experiment.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-result-overview-'));
const profile = path.join(temporary, 'profile');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/result-overview'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const logs = path.join(profile, 'workspace/logs');
const source = path.join(temporary, 'imported-result-snapshot');
const names = ['recorded-selection', 'code-match-only', 'selection-conflict', 'missing-report', 'failed-retrain', 'lifecycle-fixture', 'large-candidate-record', 'malformed-candidate-record'];
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1', NO_PROXY: '127.0.0.1,localhost' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
for (const directory of [profile, output, logs]) await mkdir(directory, { recursive: true });
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
const chosenCode = '# Synthetic selected candidate; never executed.\nprint("chosen")\n';
const laterCode = '# Synthetic later candidate; never executed.\nprint("later")\n';
const reportText = '# Synthetic generated summary\n\nThis fixture deliberately claims score 12345. It must not determine the result overview.\n';
const nodes = [
  { id: 'chosen-original', step: 0, is_buggy: false, agent_name: 'Draft', code: chosenCode, plan: 'Synthetic chosen candidate plan.', metric: { value: 0.73, maximize: true, name: 'Recorded accuracy' }, metrics: { accuracy: 0.73, log_loss: 0.38 }, metric_specs: [{ name: 'accuracy', maximize: true }, { name: 'log_loss', maximize: false }], ranking: { average_rank: 1.5, worst_rank: 2, candidate_count: 2, metric_ranks: { accuracy: 1, log_loss: 2 }, snapshot_version: 'synthetic-recorded-ranking' }, exec_time: 3 },
  { id: 'failed-high-score', step: 1, is_buggy: true, agent_name: 'Debug', code: '# failed fixture\n', metric: { value: 999, maximize: true }, exec_time: 1 },
  { id: 'later-candidate', step: 2, is_buggy: false, agent_name: 'Improve', code: laterCode, plan: 'Synthetic later candidate plan.', metric: { value: 0.41, maximize: true }, exec_time: 4 },
];
const event = (name, fields = {}) => ({ ts: '2026-10-09T10:00:00Z', event: name, ...fields });
function fixtureEvents({ selection = true, retrain = 'success', selected = nodes[0].id } = {}) {
  return [event('run_started'), ...(selection ? [event('step_saved', { node_id: nodes.at(-1).id, best_node_id: selected })] : []),
    ...(retrain === 'disabled' ? [] : [event('final_retrain_started', { node_id: selected }), event('final_retrain_finished', { ok: retrain !== 'failed', ...(retrain === 'failed' ? { error: 'Synthetic retraining failure.' } : {}) })]),
    event(retrain === 'failed' ? 'run_failed' : 'run_finished', { completed_steps: 3 })];
}
async function writeEvents(directory, values) { await writeFile(path.join(directory, 'events.jsonl'), values.map(value => JSON.stringify(value)).join('\n') + '\n'); }
async function seed(directory, { selection = true, mismatch = false, report = true, retrain = 'success', receipt = true } = {}) {
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, 'run_manifest.json'), JSON.stringify({ created_at: '2026-10-09T10:00:00Z', data_dir: path.join(temporary, 'synthetic-dataset'), task_description: 'Synthetic result overview acceptance only; no model training.', config: { agent_steps: 3, final_retrain_enabled: retrain !== 'disabled', model_for_all_agents: 'qa-never-called' } }));
  await writeFile(path.join(directory, 'journal.json'), JSON.stringify({ nodes, node2parent: { 'later-candidate': 'chosen-original' } }));
  await writeFile(path.join(directory, 'best_solution.py'), mismatch ? laterCode : chosenCode);
  await writeFile(path.join(directory, 'results.tsv'), 'step\tnode_id\tmetric\tmaximize\tstatus\n0\tchosen-original\t0.73\ttrue\tsucceeded\n1\tfailed-high-score\t999\ttrue\tfailed\n2\tlater-candidate\t0.41\ttrue\tsucceeded\n');
  if (report) await writeFile(path.join(directory, 'report.md'), reportText);
  await writeEvents(directory, fixtureEvents({ selection, retrain }));
  if (receipt && retrain !== 'disabled') await writeFile(path.join(directory, 'final_retrain_result.json'), JSON.stringify({ ok: retrain !== 'failed', train_rows_before: 12, score_rows_added: 3, train_rows_after: 15, error: retrain === 'failed' ? 'Synthetic retraining failure.' : null }));
}
await seed(path.join(logs, names[0]));
await seed(path.join(logs, names[1]), { selection: false, retrain: 'disabled' });
await seed(path.join(logs, names[2]), { mismatch: true, retrain: 'disabled' });
await seed(path.join(logs, names[3]), { report: false, retrain: 'disabled' });
await seed(path.join(logs, names[4]), { retrain: 'failed' });
await seed(path.join(logs, names[5]), { retrain: 'disabled' });
await seed(path.join(logs, names[6]), { retrain: 'disabled' });
const largeNodes = Array.from({ length: 1503 }, (_, step) => ({ id: `large-candidate-${step}`, step, is_buggy: false, code: `# Synthetic large candidate ${step}; never executed.\n`, plan: `Synthetic large candidate plan ${step}.`, metric: { value: step / 10000, maximize: true } }));
const largeSelected = largeNodes[1501];
await writeFile(path.join(logs, names[6], 'journal.json'), JSON.stringify({ nodes: largeNodes, node2parent: { [largeSelected.id]: largeNodes[1500].id } }));
await writeFile(path.join(logs, names[6], 'best_solution.py'), largeSelected.code);
await writeFile(path.join(logs, names[6], 'results.tsv'), 'step\tnode_id\tmetric\tmaximize\tstatus\n' + largeNodes.map(node => `${node.step}\t${node.id}\t${node.metric.value}\ttrue\tsucceeded\n`).join(''));
await writeEvents(path.join(logs, names[6]), [event('run_started'), event('step_saved', { node_id: largeNodes.at(-1).id, best_node_id: largeSelected.id }), event('run_finished', { completed_steps: largeNodes.length })]);
await seed(path.join(logs, names[7]));
await writeFile(path.join(logs, names[7], 'journal.json'), JSON.stringify({ nodes: [{ ...nodes[0], agent_name: { malformed: true }, plan: { malformed: true }, analysis: ['malformed'], tag: { malformed: true }, outcome: { malformed: true }, exec_time: { malformed: true }, exc_type: { malformed: true }, _term_out: 'not-an-array' }, ...nodes.slice(1)], node2parent: { 'later-candidate': { malformed: true } } }));
await writeFile(path.join(logs, names[7], 'events.jsonl'), fixtureEvents().map(value => JSON.stringify(value)).join('\n') + '\n{"event":"step_saved",');
await seed(source, { receipt: false });

async function snapshot(directory) {
  const result = {};
  for (const name of (await readdir(directory)).sort()) {
    const target = path.join(directory, name);
    const info = await stat(target);
    result[name] = info.isDirectory() ? await snapshot(target) : { sha256: createHash('sha256').update(await readFile(target)).digest('hex'), mtimeMs: info.mtimeMs };
  }
  return result;
}
let expectedLogs = await snapshot(logs);
const sourceBefore = await snapshot(source);
const rendererErrors = [];
const checks = [];
const restarts = [];
let application;
let page;
let imported;
let appInfo;
async function until(predicate, reason, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(reason);
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}
const api = (route, method = 'GET', body) => page.evaluate(({ route, method, body }) => window.biomaker.request(route, { method, ...(body === undefined ? {} : { body }) }), { route, method, body });
const frame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const results = () => page.locator('.run-results');
const selection = () => page.locator('.result-selection');
const resources = name => page.locator('.result-resource').filter({ has: page.locator('.result-resource-meta').getByText(name, { exact: true }) });
async function openRun(name) {
  await page.locator('.primary-nav').getByRole('button', { name: /全部实验|All experiments/ }).click();
  await page.getByLabel(/^(搜索全部实验|Search all experiments)$/).fill(name);
  await page.locator('.history-list').getByText(name, { exact: true }).click();
  await until(async () => (await page.locator('.run-intro h1').innerText()) === name, `Did not open ${name}`);
}
async function loaded() { await until(async () => await results().count() > 0 && await results().getAttribute('aria-busy') === 'false' && await selection().count() > 0, 'Result overview did not finish loading'); }
async function configure(values) { await application.evaluate((_electron, values) => Object.assign(globalThis.__resultQA, values), values); }
async function holdNext(id) { await configure({ holdRun: id }); }
async function held() { await until(() => application.evaluate(() => typeof globalThis.__resultQA.release === 'function'), 'Real result response was not held'); }
async function beginHeldRead(id) {
  await holdNext(id);
  let clicked = false;
  await until(async () => {
    if (await application.evaluate(() => typeof globalThis.__resultQA.release === 'function')) return true;
    if (!clicked) clicked = await page.getByRole('button', { name: '刷新结果', exact: true }).evaluate(element => {
      if (element.disabled) return false;
      element.click();
      return true;
    });
    return false;
  }, 'Neither automatic nor manual real result read was held');
}
async function release() {
  await application.evaluate(() => globalThis.__resultQA.release());
  await until(() => application.evaluate(() => globalThis.__resultQA.heldDone), 'Held result response did not settle');
  await frame();
}
async function restart() {
  const before = await page.evaluate(() => window.biomaker.getServiceState());
  const runtimes = (await readdir(path.join(profile, 'runtimes'))).filter(name => !name.includes('.staging-'));
  assert.equal(runtimes.length, 1);
  const serviceFile = path.join(profile, 'runtimes', runtimes[0], 'service/main.py');
  const processes = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes(serviceFile) && line.includes('python'));
  assert.equal(processes.length, 1, 'Only the exact isolated QA sidecar may be stopped');
  process.kill(Number(processes[0].trim().split(/\s+/)[0]), 'SIGKILL');
  await until(async () => { const state = await page.evaluate(() => window.biomaker.getServiceState()); return state.status === 'ready' && state.revision > before.revision; }, 'Isolated service did not recover', 60_000);
  restarts.push({ from: before.revision, to: (await page.evaluate(() => window.biomaker.getServiceState())).revision });
}
async function screenshot(name, locator = results()) {
  if (await locator.count()) await locator.evaluate(element => {
    const scroller = element.closest('.workspace-content');
    if (!scroller) return;
    const tabs = document.querySelector('.run-tabs');
    scroller.scrollTop += element.getBoundingClientRect().top - scroller.getBoundingClientRect().top - (tabs?.getBoundingClientRect().height || 0) - 14;
  });
  await frame();
  await page.screenshot({ path: path.join(output, name) });
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  await application.evaluate(() => {
    const originalFetch = globalThis.fetch;
    globalThis.__resultQA = { requests: [], forbidden: [], allowImport: false, holdRun: null, heldDone: false, failRun: null, snapshotState: null, lifecycleRun: 'lifecycle-fixture', activeLifecycle: false };
    globalThis.fetch = async (input, options) => {
      const qa = globalThis.__resultQA;
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      const allowedImport = qa.allowImport && method === 'POST' && ['/api/desktop/history/preview', '/api/desktop/history/import'].includes(url.pathname);
      if (url.hostname !== '127.0.0.1' || (!['GET', 'HEAD'].includes(method) && !allowedImport)) {
        qa.forbidden.push({ path: url.pathname, method });
        throw new Error('Result QA forbids provider calls, engine operations, and run mutations');
      }
      const resultMatch = /^\/api\/desktop\/runs\/([^/]+)\/result$/.exec(url.pathname);
      const runId = resultMatch ? decodeURIComponent(resultMatch[1]) : null;
      qa.requests.push({ path: url.pathname, method });
      if (runId && qa.failRun === runId) { qa.failRun = null; return new Response('{"detail":"Synthetic result read failure"}', { status: 503, headers: { 'Content-Type': 'application/json' } }); }
      const shouldHold = runId && qa.holdRun === runId;
      if (shouldHold) { qa.holdRun = null; qa.heldDone = false; }
      const response = await originalFetch(input, options);
      const lifecyclePath = url.pathname === '/api/runs' || url.pathname === `/api/runs/${qa.lifecycleRun}` || url.pathname === `/api/runs/${qa.lifecycleRun}/updates`;
      if (shouldHold || (runId && qa.snapshotState) || lifecyclePath) {
        const body = await response.text();
        let value = JSON.parse(body);
        if (lifecyclePath) {
          const patch = run => run?.id === qa.lifecycleRun && qa.activeLifecycle ? { ...run, status: 'running', detailed_status: 'running', allowed_actions: [], cleanup_pending: false } : run;
          value = Array.isArray(value) ? value.map(patch) : patch(value);
          if (url.pathname.endsWith('/updates')) value.run_revision = `${value.run_revision}:synthetic-lifecycle-${qa.activeLifecycle}`;
        }
        if (runId && qa.snapshotState) value = { ...value, snapshot: { ...value.snapshot, status: qa.snapshotState } };
        const buffered = new Response(JSON.stringify(value), { status: response.status, headers: response.headers });
        if (shouldHold) await new Promise(resolve => { qa.release = () => { delete qa.release; qa.heldDone = true; resolve(); }; });
        return buffered;
      }
      return response;
    };
  });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  await until(async () => { try { return (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready'; } catch { return false; } }, 'Sidecar did not become ready', 60_000);
  await page.locator('.sidebar-settings').waitFor();
  appInfo = await page.evaluate(() => window.biomaker.getAppInfo());
  const initialRuns = await api('/api/runs');
  assert.equal(initialRuns.length, names.length);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  const overview = await api(`/api/desktop/runs/${names[0]}/result`);
  assert.equal(overview.selection.node_id, nodes[0].id);
  assert.equal(overview.selection.metric.value, 0.73);
  assert.deepEqual(overview.selection.metrics, nodes[0].metrics);
  assert.deepEqual(overview.selection.metric_specs, nodes[0].metric_specs);
  assert.deepEqual(overview.selection.ranking, nodes[0].ranking);
  assert.equal(overview.final_retrain.state, 'succeeded');
  await openRun(names[0]); await loaded();
  assert.equal(await page.getByRole('tab', { name: '结果', exact: true }).getAttribute('aria-selected'), 'true');
  assert.match(await selection().innerText(), /运行记录中的选择/);
  assert.match(await selection().innerText(), /0\.73/);
  const metricFacts = selection().getByLabel('全部验证指标', { exact: true });
  assert.match(await metricFacts.innerText(), /accuracy ↑\s+0\.73/);
  assert.match(await metricFacts.innerText(), /log_loss ↓\s+0\.38/);
  assert.match(await selection().innerText(), /记录的平均排名\s+1\.5/);
  assert.match(await selection().innerText(), /参与排名的候选数\s+2/);
  assert(!(await selection().innerText()).includes('999'));
  assert(!(await results().innerText()).includes('12345'));
  assert.equal(await page.getByText('最佳记录指标', { exact: true }).count(), 0);
  assert.match(await page.locator('.result-retrain').innerText(), /最终重训已完成/);
  await screenshot('01-recorded-selection-light.png');
  checks.push('terminal-opens-results-and-recorded-choice-excludes-failed-extreme-last-node-and-report-claim', 'successful-retrain-shown-from-real-receipt', 'recorded-multi-metric-directions-average-rank-and-cohort-render-without-reranking');

  await page.getByRole('button', { name: '查看候选方案', exact: true }).click();
  await until(async () => await page.locator('.candidate-detail-heading h3').innerText() === '候选方案 #1', 'Exact selected candidate was not opened');
  assert.match(await page.locator('.candidate-content').innerText(), /Synthetic chosen candidate plan/);
  await until(() => page.locator('.candidate-detail-heading h3').evaluate(element => element === document.activeElement), 'Candidate detail heading did not receive focus');
  await page.getByRole('tab', { name: '结果', exact: true }).click(); await loaded();
  await page.getByRole('button', { name: '查看全部文件', exact: true }).click();
  assert.equal(await page.getByRole('tab', { name: /^文件/ }).getAttribute('aria-selected'), 'true');
  assert.equal(await page.getByRole('tab', { name: /^文件/ }).evaluate(element => element === document.activeElement), true);
  await page.keyboard.press('Home'); await loaded();
  assert.equal(await page.getByRole('tab', { name: '结果', exact: true }).evaluate(element => element === document.activeElement), true);
  checks.push('selected-candidate-navigation-targets-exact-id-and-focus', 'all-files-navigation-and-roving-tab-keyboard-focus');

  await page.getByRole('button', { name: '预览实验报告', exact: true }).click();
  const preview = page.getByRole('dialog', { name: 'report.md', exact: true });
  await preview.waitFor();
  assert.match(await preview.innerText(), /Synthetic generated summary/);
  await page.keyboard.press('Escape');
  await preview.waitFor({ state: 'hidden' });
  assert.equal(await page.getByRole('button', { name: '预览实验报告', exact: true }).evaluate(element => element === document.activeElement), true);
  const destination = path.join(temporary, 'saved-result-report.md');
  await application.evaluate(({ dialog }, filePath) => { dialog.showSaveDialog = async () => ({ canceled: false, filePath }); }, destination);
  await results().getByRole('button', { name: '保存 report.md', exact: true }).click();
  await until(async () => { try { return (await readFile(destination, 'utf8')) === reportText; } catch { return false; } }, 'Real native save from Results did not write report');
  checks.push('report-preview-and-native-save-dispatch-from-results-with-focus-restoration');

  await openRun(names[1]); await loaded();
  assert.match(await selection().innerText(), /与保存代码匹配的候选/);
  assert.match(await selection().innerText(), /无法确认运行中的选择记录/);
  assert.match(await page.locator('.result-retrain').innerText(), /未启用最终重训/);
  checks.push('exact-code-match-is-explicitly-distinct-from-recorded-selection-and-disabled-retrain');
  await openRun(names[2]); await loaded();
  assert.match(await selection().innerText(), /选择记录不一致/);
  assert.equal(await selection().locator('.result-facts').count(), 0);
  assert.equal(await page.getByRole('button', { name: '查看候选方案', exact: true }).count(), 0);
  await screenshot('02-selection-conflict.png');
  checks.push('conflicting-code-and-selection-hide-candidate-conclusion-and-metric');
  await openRun(names[3]); await loaded();
  assert.match(await resources('report.md').innerText(), /未保存此文件/);
  assert.equal(await resources('report.md').getByRole('button', { name: '预览实验报告', exact: true }).isDisabled(), true);
  assert.equal(await resources('report.md').getByRole('button', { name: '保存 report.md', exact: true }).isDisabled(), true);
  checks.push('missing-report-remains-explicit-with-disabled-file-actions');
  await openRun(names[4]); await loaded();
  assert.match(await page.locator('.result-retrain').innerText(), /最终重训未成功/);
  assert.equal(await resources('final_retrain_result.json').getByRole('button', { name: '预览最终重训记录', exact: true }).isEnabled(), true);
  checks.push('failed-retrain-preserves-access-to-recorded-result-files');

  await openRun(names[6]); await loaded();
  assert.match(await selection().innerText(), /#1502/);
  await page.getByRole('button', { name: '查看候选方案', exact: true }).click();
  await until(async () => await page.locator('.candidate-detail-heading h3').innerText() === '候选方案 #1502', 'Selected candidate beyond the list limit was not opened');
  assert.match(await page.locator('.candidate-window-note').innerText(), /前 1500 个候选/);
  assert.match(await page.locator('.candidate-row.selected').innerText(), /上级候选未显示/);
  assert(!(await page.locator('.candidate-row.selected').innerText()).includes('独立候选方案'));
  assert.equal(await page.getByText('此候选方案已不在当前记录中。请返回结果页刷新后再查看。', { exact: true }).count(), 0);
  assert.equal(await page.locator('.candidate-row').count(), 1501);
  await until(() => page.locator('.candidate-detail-heading h3').evaluate(element => element === document.activeElement), 'Large selected candidate did not receive focus');
  await screenshot('03a-large-selected-candidate.png', page.locator('.candidate-detail'));
  checks.push('real-journal-selected-candidate-beyond-1500-remains-reachable-with-focus-and-honest-parent-label');

  await openRun(names[7]); await loaded();
  assert.match(await selection().innerText(), /与保存代码匹配的候选/);
  assert.match(await selection().innerText(), /无法确认运行中的选择记录/);
  assert.match(await page.locator('.result-retrain').innerText(), /事件记录暂不可用/);
  assert(!(await page.locator('.result-retrain').innerText()).includes('最终重训已完成'));
  await page.getByRole('button', { name: '查看候选方案', exact: true }).click();
  await until(async () => await page.locator('.candidate-detail-heading h3').innerText() === '候选方案 #1', 'Malformed optional fields prevented candidate navigation');
  assert.match(await page.locator('.candidate-content').innerText(), /没有记录说明/);
  await page.getByRole('tab', { name: '执行输出', exact: true }).click();
  assert.match(await page.locator('.candidate-content').innerText(), /尚无执行输出/);
  await page.getByRole('tab', { name: '代码', exact: true }).click();
  assert.match(await page.locator('.candidate-content').innerText(), /Synthetic selected candidate/);
  assert.equal(await page.locator('.run-intro h1').innerText(), names[7]);
  await screenshot('03b-malformed-fields-safe-candidate.png', page.locator('.candidate-detail'));
  checks.push('malformed-optional-candidate-fields-stay-readable-and-corrupt-events-cannot-confirm-retrain-attempt');

  await configure({ allowImport: true });
  await application.evaluate(({ dialog }, folder) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [folder] }); }, source);
  const selectedSource = await page.evaluate(() => window.biomaker.chooseDirectory({ purpose: 'history' }));
  const importPreview = await api('/api/desktop/history/preview', 'POST', { path: selectedSource });
  const importJob = await api('/api/desktop/history/import', 'POST', { preview_id: importPreview.preview_id, source_ids: [importPreview.runs[0].source_id] });
  let settled;
  await until(async () => { settled = await api(`/api/desktop/history/imports/${importJob.job_id}`); return ['completed', 'failed'].includes(settled.status); }, 'Real history import did not settle');
  assert.equal(settled.status, 'completed');
  await configure({ allowImport: false });
  imported = (await api('/api/runs')).find(run => run.id === settled.imported[0].run_id);
  assert(imported?.history_import);
  expectedLogs = await snapshot(logs);
  await page.locator('.app-header').getByRole('button', { name: '刷新', exact: true }).click();
  await openRun(imported.history_import.name); await loaded();
  assert.match(await results().innerText(), /导入时保存的历史快照/);
  assert.match(await page.locator('.result-retrain').innerText(), /结果记录缺失/);
  assert.equal(await resources('final_retrain_result.json').getByRole('button', { name: '保存 final_retrain_result.json', exact: true }).isDisabled(), true);
  await screenshot('03-imported-snapshot-missing-receipt.png');
  checks.push('real-history-import-labels-snapshot-and-missing-retrain-receipt');

  await configure({ activeLifecycle: true });
  await page.locator('.app-header').getByRole('button', { name: '刷新', exact: true }).click();
  await openRun(names[5]);
  assert.equal(await page.getByRole('tab', { name: '活动', exact: true }).getAttribute('aria-selected'), 'true');
  assert.equal(await results().count(), 0);
  await configure({ activeLifecycle: false });
  await page.locator('.app-header').getByRole('button', { name: '刷新', exact: true }).click();
  await until(async () => (await page.locator('.status-label').innerText()).includes('已完成'), 'Simulated lifecycle did not return to real terminal status');
  assert.equal(await page.getByRole('tab', { name: '活动', exact: true }).getAttribute('aria-selected'), 'true');
  checks.push('synthetic-active-lifecycle-opens-activity-and-terminal-refresh-does-not-steal-tab');

  await openRun(names[0]); await loaded();
  await beginHeldRead(names[0]);
  await openRun(names[1]); await loaded();
  await release();
  assert.match(await selection().innerText(), /与保存代码匹配的候选/);
  assert.equal(await page.locator('.run-intro h1').innerText(), names[1]);
  checks.push('late-real-result-for-previous-run-cannot-replace-current-overview');

  await openRun(names[0]); await loaded();
  await beginHeldRead(names[0]);
  await writeFile(path.join(logs, names[0], 'best_solution.py'), laterCode);
  await writeEvents(path.join(logs, names[0]), fixtureEvents({ selected: nodes[2].id }));
  expectedLogs = await snapshot(logs);
  await restart();
  await until(async () => (await selection().innerText()).includes('0.41'), 'New service generation did not load current real selection');
  await release();
  assert.match(await selection().innerText(), /0\.41/);
  assert(!(await selection().innerText()).includes('0.73'));
  checks.push('late-real-result-across-real-sidecar-restart-cannot-overwrite-current-records');

  await holdNext(names[0]);
  await writeFile(path.join(logs, names[0], 'best_solution.py'), chosenCode);
  await writeEvents(path.join(logs, names[0]), fixtureEvents());
  expectedLogs = await snapshot(logs);
  await page.locator('.app-header').getByRole('button', { name: '刷新', exact: true }).click();
  await held();
  assert.equal(await selection().count(), 0, 'Old selection remained visible after artifact identity changed');
  assert.equal(await resources('report.md').getByRole('button', { name: '保存 report.md', exact: true }).isDisabled(), true);
  await release(); await loaded();
  assert.match(await selection().innerText(), /0\.73/);
  checks.push('changed-artifact-signature-hides-old-conclusions-and-actions-until-current-read-finishes');

  await configure({ failRun: names[0] });
  await page.getByRole('button', { name: '刷新结果', exact: true }).click();
  await results().getByText(/暂时无法读取结果/).waitFor();
  assert.equal(await selection().count(), 0);
  await results().getByRole('button', { name: '重试', exact: true }).click(); await loaded();
  assert.match(await selection().innerText(), /0\.73/);
  checks.push('explicit-read-error-hides-conclusions-and-manual-retry-recovers');
  for (const state of ['changing', 'unavailable']) {
    await configure({ snapshotState: state });
    await page.getByRole('button', { name: '刷新结果', exact: true }).click();
    const statePattern = state === 'changing' ? /实验文件正在更新/ : /目前无法读取足够的结果记录/;
    await until(async () => await results().getAttribute('aria-busy') === 'false' && await selection().count() === 0 && statePattern.test(await results().innerText()), `Synthetic ${state} snapshot did not hide conclusions`);
    assert.equal(await page.locator('.result-retrain').count(), 0);
    assert.match(await results().innerText(), statePattern);
  }
  await configure({ snapshotState: null });
  await page.getByRole('button', { name: '刷新结果', exact: true }).click(); await loaded();
  checks.push('synthetic-changing-and-unavailable-snapshots-never-display-selection-or-retrain-conclusions');

  await page.locator('.sidebar-settings').click();
  await page.getByRole('button', { name: '通用', exact: true }).click();
  await page.getByRole('button', { name: '深色', exact: true }).click();
  await page.getByLabel('界面语言', { exact: true }).selectOption('en');
  await page.getByText('Make BioMaker your own', { exact: true }).waitFor();
  await page.keyboard.press('Escape');
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  await results().getByRole('heading', { name: 'Results overview', exact: true }).waitFor();
  assert.match(await selection().innerText(), /Recorded selection/);
  assert.match(await selection().innerText(), /Recorded average rank\s+1\.5/);
  assert.match(await selection().getByLabel('All validation metrics', { exact: true }).innerText(), /log_loss ↓\s+0\.38/);
  await screenshot('04-english-dark-results.png');
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  await frame();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  const reportRow = resources('report.md');
  await reportRow.getByRole('button', { name: 'Preview experiment report', exact: true }).scrollIntoViewIfNeeded();
  assert(await reportRow.getByRole('button', { name: 'Save report.md', exact: true }).isVisible());
  const box = await reportRow.boundingBox();
  assert(box && box.x >= 0 && box.x + box.width <= 960);
  await screenshot('05-compact-result-actions.png', reportRow);
  checks.push('english-dark-and-960x680-layout-retain-readable-result-file-actions');

  assert.deepEqual(await snapshot(logs), expectedLogs, 'Viewing results changed synthetic original files');
  assert.deepEqual(await snapshot(source), sourceBefore, 'Viewing/importing changed the source snapshot');
  const finalRuns = await api('/api/runs');
  assert.equal(finalRuns.length, names.length + 1);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  const native = await application.evaluate(() => ({ requests: globalThis.__resultQA.requests, forbidden: globalThis.__resultQA.forbidden }));
  assert.deepEqual(native.forbidden, []);
  assert.deepEqual(rendererErrors, []);
  checks.push('original-files-and-source-snapshot-preserved-with-zero-provider-engine-or-run-mutations', 'no-renderer-errors');
  const receipt = { passed: true, mode: 'Real Electron, sidecar result parsing, history import, native preview/save, and one real isolated sidecar restart. Synthetic log fixtures; lifecycle/snapshot states and one HTTP error intercepted; no model, engine installation, or experiment.', executablePath: executablePath || 'development Electron', appInfo, profile, source, checks, restarts, forbiddenRequests: native.forbidden, rendererErrors };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(receipt, null, 2));
  console.log(JSON.stringify(receipt, null, 2));
} catch (error) {
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-8000));
  }
  throw error;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => { globalThis.__resultQA?.release?.(); for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await application.close();
  }
}
