// Real Electron, sidecar, native draft storage, settings persistence and artifact
// preview in an isolated profile. Clipboard completions, submission/lifecycle
// responses and draft-delete faults are controlled UI fixtures; no clipboard,
// provider, engine installation, training or real experiment is invoked.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, realpath, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-toast-feedback-'));
const profile = path.join(temporary, 'profile');
const dataset = path.join(temporary, 'synthetic-dataset');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/toast-feedback-0.1.16'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
for (const directory of [profile, dataset, output]) await mkdir(directory, { recursive: true });
await writeFile(path.join(dataset, 'metadata.csv'), 'id,feature\nsynthetic,1\n');
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
for (const name of ['toast-history-a', 'toast-history-b']) {
  const directory = path.join(profile, 'workspace/logs', name);
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, 'run_manifest.json'), JSON.stringify({ data_dir: dataset, created_at: '2026-10-09T04:00:00Z', task_description: 'Synthetic toast UI acceptance only; no biomedical claim or training.', config: { agent_steps: 1, final_retrain_enabled: false } }));
  await writeFile(path.join(directory, 'events.jsonl'), JSON.stringify({ event: 'run_finished', completed_steps: 0 }) + '\n');
  await writeFile(path.join(directory, 'report.md'), `# ${name}\n\nSynthetic UI fixture; never executed.\n`);
}
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_HOME: profile, BIOMAKER_DESKTOP_USER_DATA: profile, PYTHONDONTWRITEBYTECODE: '1', NO_PROXY: '127.0.0.1,localhost' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const checks = [], rendererErrors = [], screenshots = [], restarts = [], observations = {};
let application, page, appInfo, trace, failure;
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!await predicate()) { if (Date.now() >= deadline) throw new Error(message); await delay(40); }
}
const frame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const api = route => page.evaluate(route => window.biomaker.request(route), route);
const state = () => page.evaluate(() => window.biomaker.getServiceState());
const drafts = () => page.evaluate(() => window.biomaker.getDrafts());
const toast = () => page.locator('.toast');
const modal = () => page.getByRole('dialog', { name: 'report.md', exact: true });
const configure = values => application.evaluate((_electron, values) => Object.assign(globalThis.__toastQA, values), values);
const snapshot = async name => { const filename = path.join(output, name); await page.screenshot({ path: filename }); screenshots.push(filename); };
async function noToast(reason) { await frame(); assert.equal(await toast().count(), 0, reason); }
async function expectToast(pattern) { await until(async () => await toast().count() === 1 && pattern.test(await toast().innerText()), `Missing toast: ${pattern}`); }
async function clearToast() { if (await toast().count()) await toast().getByRole('button').click(); await noToast('Dismissed toast remained'); }
async function closePreview() { if (await modal().count()) { await page.keyboard.press('Escape'); await modal().waitFor({ state: 'hidden' }); } }
async function openRun(name) {
  await closePreview();
  await page.locator('.primary-nav').getByRole('button', { name: /全部实验|All experiments/ }).click();
  await page.getByLabel(/^(搜索全部实验|Search all experiments)$/).fill(name);
  await page.locator('.history-list').getByText(name, { exact: true }).click();
  await until(async () => await page.locator('.run-intro h1').innerText() === name, `Run ${name} not selected`);
}
async function showReport() {
  await page.getByRole('tab', { name: /^(文件|Files)/ }).click();
  const row = page.locator('.artifact-row').filter({ has: page.getByText('report.md', { exact: true }) });
  await row.getByRole('button', { name: /^(预览|Preview)$/ }).click();
  await modal().waitFor();
}
async function beginCopy(hold = false) {
  await page.evaluate(hold => { window.__toastClipboard.hold = hold; window.__toastClipboard.release = null; }, hold);
  await modal().getByRole('button', { name: /^(复制内容|Copy contents)$/ }).click();
  if (hold) await until(() => page.evaluate(() => typeof window.__toastClipboard.release === 'function'), 'Clipboard completion was not held');
  else await expectToast(/已复制|Copied/);
}
async function releaseCopy() { await page.evaluate(() => window.__toastClipboard.release()); await frame(); }
async function openSettings() { await closePreview(); await page.locator('.sidebar-settings').click(); await page.locator('.settings-modal').waitFor(); }
async function language(value) {
  await page.locator('.settings-nav').getByRole('button', { name: /^(通用|General)$/ }).click();
  await page.getByLabel(/^(界面语言|Interface language)$/).selectOption(value);
  await until(() => page.evaluate(value => document.documentElement.lang === (value === 'zh' ? 'zh-CN' : 'en'), value), 'Language did not change');
}
async function closeSettings() { await page.keyboard.press('Escape'); await page.locator('.settings-modal').waitFor({ state: 'hidden' }); }
async function restartSidecar() {
  const before = await state(), canonical = await realpath(profile);
  const matches = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n')
    .filter(line => line.includes('/service/main.py') && line.includes('python') && (line.includes(profile) || line.includes(canonical)));
  assert.equal(matches.length, 1, 'Only this isolated sidecar may be stopped');
  process.kill(Number(matches[0].trim().split(/\s+/)[0]), 'SIGKILL');
  await until(async () => { const next = await state(); return next.status === 'ready' && next.revision > before.revision; }, 'Sidecar did not recover', 60_000);
  restarts.push({ before: before.revision, after: (await state()).revision });
  await frame();
}
async function refresh() { await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:menu-action', 'refresh')); await frame(); }
async function prepareSubmission(name, deleteMode) {
  await page.locator('.primary-nav').getByRole('button', { name: /新建实验|New experiment/ }).click();
  await page.getByLabel(/^(实验任务|Experiment task)$/).fill(`Synthetic ${name} toast acceptance only. No actual run, model, or biomedical result.`);
  await application.evaluate(({ dialog }, directory) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [directory] }); }, dataset);
  await page.locator('.dataset-button').click();
  await until(async () => await page.locator('.dataset-overview').getAttribute('data-code') === 'ok' && await page.locator('.dataset-overview').getAttribute('data-state') !== 'checking', 'Real dataset inspection did not settle');
  const before = await drafts();
  assert(before.activeId);
  await configure({ nextSubmission: { name }, deleteMode });
  await until(() => page.locator('.submit-button').isEnabled(), 'Synthetic submission UI did not become ready');
  await page.locator('.submit-button').click();
  await until(async () => await page.locator('.run-intro h1').count() === 1 && await page.locator('.run-intro h1').innerText() === name, 'Submission did not render its destination run');
  return { id: before.activeId, before };
}
async function action(kind, plan = {}) {
  await configure({ actionPlan: { kind, ...plan } });
  await page.locator('.run-actions').getByRole('button', { name: kind === 'cancel' ? /^(取消排队|Cancel queued run)$/ : /^(停止|Stop)$/ }).click();
  const confirmation = page.locator('.confirm-modal');
  await confirmation.waitFor();
  await confirmation.locator('.danger-button').click();
  await confirmation.waitFor({ state: 'hidden' });
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  await until(async () => { try { return (await state()).status === 'ready'; } catch { return false; } }, 'Isolated sidecar did not become ready');
  appInfo = await page.evaluate(() => window.biomaker.getAppInfo());
  const initial = { status: await api('/api/desktop/status'), runs: await api('/api/runs') };
  assert.equal(initial.status.engine.ready, false);
  assert.equal(initial.runs.length, 2);
  const sample = initial.runs[0];
  initial.activity = await api(`/api/desktop/runs/${sample.id}/activity`);
  initial.console = await api(`/api/desktop/runs/${sample.id}/console`);
  initial.result = await api(`/api/desktop/runs/${sample.id}/result`);
  await application.evaluate(({ ipcMain }, initial) => {
    const fetchOriginal = globalThis.fetch;
    const f = globalThis.__toastQA = { initial, forbidden: [], operations: [], submissions: [], actions: [], deletes: [], syntheticRuns: [], syntheticSubmission: false, allowSave: false, holdSave: false, deleteMode: 'fast' };
    const json = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
    const emptyStream = (source, extra) => ({ ...source, source_id: 'synthetic-toast-stream', status: 'ready', start: 0, end: 0, source_size: 0, has_older: false, older_cursor: null, ...extra });
    globalThis.fetch = async (input, options) => {
      const url = new URL(String(input)), method = String(options?.method || 'GET').toUpperCase();
      const forbidden = () => { f.forbidden.push({ method, path: url.pathname }); throw new Error('Toast QA forbids external calls and unplanned mutations'); };
      if (url.hostname !== '127.0.0.1') return forbidden();
      f.operations.push({ method, path: url.pathname });
      if (url.pathname === '/api/env' && method === 'PUT') {
        if (!f.allowSave) return forbidden();
        const response = await fetchOriginal(input, options);
        if (!f.holdSave) return response;
        f.holdSave = false;
        const body = await response.text();
        await new Promise(resolve => { f.releaseSave = () => { f.releaseSave = null; resolve(); }; });
        return new Response(body, { status: response.status, headers: response.headers });
      }
      if (url.pathname === '/api/desktop/dataset/inspect' && method === 'POST') return fetchOriginal(input, options);
      if (f.syntheticSubmission && url.pathname === '/api/desktop/status' && method === 'GET') return json({ ...initial.status, engine: { ...initial.status.engine, ready: true, installing: false, checking: false, operation: null, busy_external: false, maintenance_unavailable: false, health: 'healthy' } });
      if (url.pathname === '/api/runs' && method === 'POST') {
        if (!f.syntheticSubmission || !f.nextSubmission) return forbidden();
        const body = JSON.parse(options.body), plan = f.nextSubmission;
        f.nextSubmission = null;
        const run = { id: `qa-toast-${f.submissions.length + 1}`, actual_exp_name: plan.name, requested_exp_name: plan.name,
          status: 'queued', detailed_status: 'queued', managed: true, read_only: false, allowed_actions: ['cancel'],
          dataset_path: body.experiment.dataset_path, project_snapshot: body.experiment, options: body.experiment.advanced_defaults,
          artifacts: [], timestamps: { created_at: new Date().toISOString() } };
        f.submissions.push({ id: run.id, name: plan.name, synthetic: true });
        f.syntheticRuns.push(run);
        return json(run, 202);
      }
      const match = /^\/api\/runs\/(qa-toast-\d+)(?:\/(.*))?$/.exec(url.pathname);
      if (match) {
        const run = f.syntheticRuns.find(run => run.id === match[1]);
        if (!run) return json({ detail: 'Unknown synthetic run' }, 404);
        if (method === 'POST' && ['cancel', 'stop'].includes(match[2])) {
          const plan = f.actionPlan;
          if (!plan || plan.kind !== match[2]) return forbidden();
          f.actionPlan = null;
          f.actions.push({ id: run.id, kind: match[2], error: !!plan.error, held: !!plan.hold });
          if (plan.hold) await new Promise(resolve => { f.releaseAction = () => { f.releaseAction = null; resolve(); }; });
          if (plan.error) return json({ detail: 'Synthetic action failure for toast acceptance' }, 409);
          Object.assign(run, { status: 'incomplete', detailed_status: match[2] === 'cancel' ? 'cancelled' : 'stopping', allowed_actions: [] });
          return json(run);
        }
        if (method !== 'GET') return forbidden();
        if (!match[2]) return json(run);
        if (match[2] === 'updates') return json({ run_revision: `synthetic-${run.detailed_status}`, events: { revision: 'synthetic', available: true }, console: { revision: 'synthetic', available: true }, artifacts: { revision: 'synthetic' }, final_barrier: false });
      }
      const detail = /^\/api\/desktop\/runs\/(qa-toast-\d+)\/(activity|console|result)$/.exec(url.pathname);
      if (detail && method === 'GET') {
        if (detail[2] === 'activity') return json(emptyStream(initial.activity, { events: [], skipped_records: 0 }));
        if (detail[2] === 'console') return json(emptyStream(initial.console, { text: '' }));
        return json({ ...initial.result, run_id: detail[1], origin: 'managed', artifacts: [], warnings: [] });
      }
      if (!['GET', 'HEAD'].includes(method)) return forbidden();
      const response = await fetchOriginal(input, options);
      if (url.pathname === '/api/runs' && f.syntheticRuns.length) return json([...(await response.json()), ...f.syntheticRuns]);
      return response;
    };
    const channel = 'biomaker:mutate-drafts';
    const original = ipcMain._invokeHandlers.get(channel);
    if (typeof original !== 'function') throw new Error('Native draft IPC handler unavailable; no fallback to fake storage');
    ipcMain.removeHandler(channel);
    ipcMain.handle(channel, async (event, input) => {
      if (input?.operation?.type !== 'delete') return original(event, input);
      const mode = f.deleteMode;
      f.deleteMode = 'fast';
      f.deletes.push({ id: input.operation.id, mode });
      if (mode === 'delayed') await new Promise(resolve => { f.releaseDelete = () => { f.releaseDelete = null; resolve(); }; });
      if (mode === 'failure') throw new Error('[drafts:save_failed] Synthetic delete fault; source draft retained');
      return original(event, input);
    });
  }, initial);
  await page.evaluate(() => {
    const f = window.__toastClipboard = { hold: false, copies: 0, completions: 0, release: null };
    Object.defineProperty(navigator.clipboard, 'writeText', { configurable: true, value: async () => {
      f.copies++;
      if (f.hold) { f.hold = false; await new Promise(resolve => { f.release = () => { f.release = null; resolve(); }; }); }
      f.completions++;
    } });
  });
  checks.push('isolated-real-sidecar-and-native-storage-with-uninstalled-engine');

  // Repeat the real preview action with a controlled clipboard adapter. Each
  // completion must own a fresh timer even when the visible text is identical.
  await openRun('toast-history-a'); await showReport(); await beginCopy();
  const firstAt = Date.now();
  await delay(4800);
  await beginCopy();
  await expectToast(/已复制/);
  const secondAt = Date.now();
  await delay(Math.max(0, firstAt + 6350 - Date.now()));
  await expectToast(/已复制/);
  observations.sameMessage = { repeatedAfterMs: secondAt - firstAt, stillVisibleAfterOriginalDeadlineMs: Date.now() - firstAt };
  await snapshot('01-repeat-identical-toast.png');
  await clearToast();
  await beginCopy();
  await expectToast(/已复制/);
  const thirdAt = Date.now();
  await delay(Math.max(0, secondAt + 6350 - Date.now()));
  await expectToast(/已复制/);
  await delay(Math.max(0, thirdAt + 6350 - Date.now()));
  await noToast('The newest toast did not expire after its own six-second window');
  checks.push('same-message-restarts-six-second-lifetime', 'dismiss-and-old-deadlines-cannot-clear-new-toast');
  await closePreview();
  await openSettings();
  await page.getByLabel('API Key', { exact: false }).first().fill('qa-synthetic-toast-key-never-external');
  await page.getByLabel('默认模型', { exact: true }).fill('qa-toast-never-called-model');
  await configure({ allowSave: true, holdSave: true });
  await page.getByRole('button', { name: '保存配置', exact: true }).click();
  await until(() => application.evaluate(() => typeof globalThis.__toastQA.releaseSave === 'function'), 'Real settings save response was not held');
  await page.keyboard.press('Escape');
  await page.locator('.settings-footer [role=status]').getByText('正在保存…', { exact: true }).waitFor();
  await noToast('Pending save feedback must stay inline and cannot outlive its operation');
  await application.evaluate(() => globalThis.__toastQA.releaseSave());
  await page.locator('.settings-footer [role=status]').getByText('所有更改已保存', { exact: true }).waitFor();
  await noToast('Saving settings should use its inline status without a success toast');
  await page.getByLabel('默认模型', { exact: true }).fill('qa-toast-edited-model');
  await page.locator('.settings-footer [role=status]').getByText('有未保存的更改', { exact: true }).waitFor();
  await noToast('Editing settings must not retain a saved toast');
  await snapshot('02-edited-settings-inline-status.png');
  await page.getByRole('button', { name: '保存配置', exact: true }).click();
  await page.locator('.settings-footer [role=status]').getByText('所有更改已保存', { exact: true }).waitFor();
  await language('en');
  await page.locator('.settings-nav').getByRole('button', { name: 'Model provider', exact: true }).click();
  await page.locator('.settings-footer [role=status]').getByText('All changes saved', { exact: true }).waitFor();
  await noToast('Settings section/locale navigation retained a Chinese toast');
  await closeSettings();
  await configure({ allowSave: false });
  checks.push('real-settings-save-edit-section-and-language-use-current-inline-status');

  for (const destination of ['history', 'drafts', 'new', 'settings']) {
    await openRun('toast-history-a'); await showReport(); await beginCopy(); await closePreview();
    if (destination === 'settings') await openSettings();
    else await page.locator('.primary-nav').getByRole('button', { name: destination === 'history' ? /All experiments/ : destination === 'drafts' ? /Drafts/ : /New experiment/ }).click();
    await noToast(`Visible toast survived navigation to ${destination}`);
    if (destination === 'settings') await closeSettings();
  }
  checks.push('existing-toast-clears-on-history-drafts-new-experiment-and-settings');
  await openRun('toast-history-a'); await showReport(); await beginCopy(true);
  await closePreview(); await openRun('toast-history-b'); await openRun('toast-history-a'); await showReport();
  await releaseCopy();
  await noToast('A -> B -> A revived the old clipboard completion');
  await beginCopy(); await expectToast(/Copied/); await clearToast();
  checks.push('late-copy-does-not-revive-after-A-B-A-but-new-action-notifies');
  await beginCopy(true); await closePreview(); await openSettings(); await language('zh'); await closeSettings();
  await releaseCopy(); await noToast('Late English clipboard completion crossed language and settings scope');
  await showReport(); await beginCopy(); await expectToast(/已复制/); await clearToast();
  checks.push('late-copy-cannot-reintroduce-old-language-and-new-copy-localizes');
  await beginCopy(true); await restartSidecar(); await releaseCopy();
  await noToast('Late clipboard completion crossed the real service generation');
  await modal().getByText(/本地服务已变化/).waitFor();
  await snapshot('03-service-change-no-stale-toast.png');
  await beginCopy(); await expectToast(/已复制/); await clearToast(); await closePreview();
  checks.push('real-service-restart-invalidates-old-toast-scope-without-reloading-renderer');

  await configure({ syntheticSubmission: true }); await refresh();
  for (const mode of ['fast', 'delayed', 'failure']) {
    const name = `synthetic-toast-submit-${mode}`;
    const before = await prepareSubmission(name, mode);
    if (mode === 'delayed') {
      await until(() => application.evaluate(() => typeof globalThis.__toastQA.releaseDelete === 'function'), 'Draft delete was not held');
      await noToast('Submission success must wait for the held source-draft deletion');
      assert((await drafts()).drafts.some(entry => entry.id === before.id));
      await delay(500);
      await application.evaluate(() => globalThis.__toastQA.releaseDelete());
    }
    await expectToast(mode === 'failure' ? /实验已加入队列，但未能移除已提交草稿/ : /^实验已加入队列/);
    const after = await drafts();
    assert.equal(after.drafts.some(entry => entry.id === before.id), mode === 'failure', 'Native source-draft retention does not match the delete outcome');
    assert.deepEqual(after.drafts.filter(entry => entry.id !== before.id), before.before.drafts.filter(entry => entry.id !== before.id), 'Submission changed unrelated native drafts');
    await refresh(); await frame();
    await expectToast(mode === 'failure' ? /未能移除已提交草稿/ : /^实验已加入队列/);
    await snapshot(`04-submit-${mode}.png`);
    checks.push(`synthetic-submit-${mode}-publishes-at-destination-after-real-draft-outcome`);
    await clearToast();
    if (mode === 'fast') {
      await action('cancel'); await expectToast(/已取消排队/);
      checks.push('confirmed-cancel-success-survives-intentional-modal-close');
      await clearToast();
    }
    if (mode === 'delayed') {
      // The target run and final language are unchanged, so equality checks
      // alone would revive this error after an intervening settings/locale
      // round trip. The action's captured notice token must also stay valid.
      await action('cancel', { hold: true, error: true });
      await until(() => application.evaluate(() => typeof globalThis.__toastQA.releaseAction === 'function'), 'Cancel error response was not held for the locale round trip');
      await openSettings();
      await language('en');
      await language('zh');
      await closeSettings();
      assert.equal(await page.locator('.run-intro h1').innerText(), name);
      await application.evaluate(() => globalThis.__toastQA.releaseAction());
      await until(() => page.locator('.run-actions').getByRole('button', { name: '取消排队', exact: true }).isEnabled(), 'Held cancel error did not finish and release the action lock');
      await noToast('A held cancel error revived after settings and language A -> B -> A');
      await snapshot('05-held-cancel-error-locale-roundtrip.png');
      checks.push('held-confirmed-action-error-cannot-revive-after-settings-and-language-A-B-A');
      await action('cancel', { error: true }); await expectToast(/Synthetic action failure for toast acceptance/);
      checks.push('confirmed-action-error-survives-intentional-modal-close');
      await clearToast();
      await configure({ actionPlan: null });
      await application.evaluate(() => { const run = globalThis.__toastQA.syntheticRuns.find(run => run.actual_exp_name === 'synthetic-toast-submit-delayed'); Object.assign(run, { status: 'running', detailed_status: 'running', allowed_actions: ['stop'] }); });
      await refresh();
      await until(() => page.locator('.run-actions').getByRole('button', { name: '停止', exact: true }).count(), 'Synthetic running control did not appear');
      await action('stop'); await expectToast(/正在停止实验/);
      checks.push('confirmed-stop-success-survives-intentional-modal-close');
      await clearToast();
    }
    if (mode === 'failure') {
      await action('cancel', { hold: true });
      await until(() => application.evaluate(() => typeof globalThis.__toastQA.releaseAction === 'function'), 'Synthetic cancel response was not held');
      await openRun('toast-history-b');
      await application.evaluate(() => globalThis.__toastQA.releaseAction());
      await frame(); await noToast('A late confirmed action notified an unrelated run');
      checks.push('held-confirmed-action-cannot-notify-an-unrelated-run');
    }
  }
  await configure({ syntheticSubmission: false });
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  const realRuns = await application.evaluate(async () => {
    const f = globalThis.__toastQA; const saved = f.syntheticRuns; f.syntheticRuns = [];
    return { count: f.initial.runs.length, simulatedCount: saved.length };
  });
  const persisted = await api('/api/runs');
  assert.equal(persisted.length, 2, 'A synthetic submission was accidentally forwarded into the real ledger');
  assert.equal(realRuns.simulatedCount, 3);
  trace = await application.evaluate(() => { const { forbidden, operations, submissions, actions, deletes } = globalThis.__toastQA; return { forbidden, operations, submissions, actions, deletes }; });
  assert.deepEqual(trace.forbidden, []);
  assert.deepEqual(rendererErrors, []);
  assert.equal(trace.submissions.length, 3);
  assert.deepEqual(trace.deletes.map(row => row.mode), ['fast', 'delayed', 'failure']);
  observations.clipboard = await page.evaluate(() => ({ copies: window.__toastClipboard.copies, completions: window.__toastClipboard.completions, systemClipboardTouched: false }));
  checks.push('no-renderer-errors-external-provider-runtime-or-real-experiment-writes');
} catch (error) {
  failure = { message: String(error?.message || error), stack: String(error?.stack || error) };
  if (page && !page.isClosed()) await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
  if (application) trace = await application.evaluate(() => { const f = globalThis.__toastQA; return f ? { forbidden: f.forbidden, operations: f.operations, submissions: f.submissions, actions: f.actions, deletes: f.deletes } : null; }).catch(() => null);
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => {
      const f = globalThis.__toastQA;
      for (const name of ['releaseSave', 'releaseDelete', 'releaseAction']) if (typeof f?.[name] === 'function') f[name]();
      for (const window of BrowserWindow.getAllWindows()) window.destroy();
    }).catch(() => {});
    await application.close().catch(() => {});
  }
  const canonicalProfile = await realpath(profile);
  let remainingProcesses = [];
  await until(() => {
    remainingProcesses = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes(profile) || line.includes(canonicalProfile));
    return remainingProcesses.length === 0;
  }, 'QA processes did not exit', 15_000).catch(error => { failure ||= { message: error.message }; });
  const receipt = { appInfo, mode: executablePath ? 'packaged' : 'development', executablePath: executablePath || null, profile, canonicalProfile,
    checks, passed: !failure, allChecksPassed: !failure, rendererErrors, screenshots, observations, restarts, trace, remainingProcesses, failure,
    scope: 'Real isolated Electron, sidecar, settings and native draft persistence. Buffered settings response; controlled clipboard completion without OS clipboard access; synthetic engine readiness/submission/stop/cancel responses; held or failed native draft deletion. No external provider or real experiment acceptance.' };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(receipt, null, 2) + '\n');
  if (failure) { console.error(failure.message); process.exitCode = 1; }
  else console.log(JSON.stringify({ passed: checks.length, profile, receipt: path.join(output, 'acceptance.json'), screenshots: screenshots.length }));
}
