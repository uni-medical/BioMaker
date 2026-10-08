// Real Electron UI/native bridge acceptance with explicitly synthetic engine
// states. No engine check, installation, repair, provider request, or user run is
// executed. Real-engine validation belongs to a separate isolated acceptance.
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-runtime-health-ui-'));
const profile = path.join(temporary, 'profile');
const dataset = path.join(temporary, 'synthetic-dataset');
const output = process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/runtime-health-ui');
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
for (const directory of [profile, dataset, output]) await mkdir(directory, { recursive: true });
await writeFile(path.join(dataset, 'synthetic-metadata-fixture.txt'), 'Synthetic runtime UI fixture; never used for training.\n');
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
const env = { ...process.env, BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1' };
for (const key of ['ELECTRON_RUN_AS_NODE', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'SERPER_API_KEY', 'JINA_API_KEY']) delete env[key];
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const initialEngine = { ready: true, installing: false, checking: false, operation: null, health: 'unknown', checked_at: null, busy_external: false, error: null, log: '' };
const checkedAt = '2026-10-09T10:00:00.000Z';
const repairedAt = '2026-10-09T10:03:00.000Z';
const task = '  Synthetic runtime health acceptance.\nKeep this exact draft; never submit.  ';
const checks = [];
const errors = [];
let application;
let page;
let draftRaw;
async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 30));
  }
}
const storedDraft = () => page.evaluate(() => localStorage.getItem('biomaker.draft'));
async function refresh() {
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:menu-action', 'refresh'));
}
async function setEngine(state) {
  await application.evaluate((_electron, engine) => { globalThis.__runtimeHealthFixture.engine = engine; }, state);
  await refresh();
}
async function openRuntime() {
  if (!(await page.getByRole('dialog', { name: '设置', exact: true }).count())) await page.locator('.sidebar-settings').click();
  await page.getByRole('button', { name: '运行环境', exact: true }).click();
  await page.getByRole('heading', { name: '本地运行环境', exact: true }).waitFor();
}
async function closeSettings() {
  await page.keyboard.press('Escape');
  await page.getByRole('dialog', { name: '设置', exact: true }).waitFor({ state: 'hidden' });
}
async function maintenanceRequests() { return application.evaluate(() => globalThis.__runtimeHealthFixture.requests); }
async function assertDraft() {
  assert.equal(await storedDraft(), draftRaw, 'Runtime operations must preserve the exact persisted draft');
  assert.equal(await page.getByLabel('实验任务', { exact: true }).inputValue(), task);
}
async function confirmRepair() {
  await page.getByRole('button', { name: '修复运行环境', exact: true }).click();
  await page.getByRole('group', { name: '修复运行环境？', exact: true }).waitFor();
  await page.getByRole('button', { name: '开始修复', exact: true }).click();
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => errors.push(error.message));
  await until(async () => {
    try { return (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready'; }
    catch { return false; }
  }, 'The real desktop service did not become ready');
  await page.locator('.sidebar-settings').waitFor();
  const base = await page.evaluate(async () => ({ status: await window.biomaker.request('/api/desktop/status'), env: await window.biomaker.request('/api/env') }));
  assert.equal(base.status.engine.ready, false, 'The isolated profile must have no actual installed engine');
  assert.equal(base.env.values.OPENAI_API_KEY.configured, false);

  await application.evaluate((_electron, fixture) => {
    const originalFetch = globalThis.fetch;
    const provider = structuredClone(fixture.env);
    provider.values.OPENAI_API_KEY.configured = true;
    provider.values.BIOMAKER_MAIN_MODEL.value = 'qa-runtime-health-model';
    provider.values.BIOMAKER_MAIN_MODEL.configured = true;
    globalThis.__runtimeHealthFixture = { engine: fixture.engine, requests: [], forbidden: [], nextFailure: null, originalFetch, status: fixture.status };
    const json = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
    globalThis.fetch = async (input, options) => {
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      const f = globalThis.__runtimeHealthFixture;
      const maintenance = /^\/api\/desktop\/engine\/(install|check|repair|cancel)$/.exec(url.pathname);
      if (url.hostname !== '127.0.0.1' || (method !== 'GET' && /^\/api\/(?:runs(?:\/|$)|env(?:\/|$)|models\/discover)/.test(url.pathname))) {
        f.forbidden.push({ path: url.pathname, method });
        throw new Error('Synthetic runtime QA blocked a user run, provider request, or non-local request');
      }
      if (url.pathname === '/api/desktop/status' && method === 'GET') {
        const snapshot = structuredClone({ ...f.status, engine: f.engine });
        if (f.holdNextStatus) {
          f.holdNextStatus = false;
          f.heldStatus = snapshot;
          return new Promise(resolve => {
            f.releaseHeldStatus = () => {
              delete f.releaseHeldStatus;
              f.heldStatusReleased = true;
              resolve(json(snapshot));
            };
          });
        }
        return json(snapshot);
      }
      if (url.pathname === '/api/env' && method === 'GET') return json(provider);
      if (maintenance) {
        f.requests.push({ action: maintenance[1], method });
        if (method !== 'POST' || maintenance[1] === 'install') {
          f.forbidden.push({ path: url.pathname, method });
          return json({ detail: 'Unexpected synthetic maintenance operation' }, 409);
        }
        if (f.nextFailure?.action === maintenance[1]) {
          const message = f.nextFailure.message; f.nextFailure = null;
          return json({ detail: message }, 409);
        }
        if (maintenance[1] !== 'cancel') {
          f.engine = { ...f.engine, ready: false, installing: maintenance[1] === 'repair', checking: maintenance[1] === 'check', operation: maintenance[1], busy_external: false, health: 'unknown', error: null, log: 'Synthetic UI fixture; no engine command is running.' };
        }
        // Cancellation remains pending until the fixture reports that its fake
        // worker exited, matching the backend's asynchronous cancellation.
        return json(f.engine);
      }
      return originalFetch(input, options);
    };
  }, { ...base, engine: initialEngine });
  // Start the renderer fixture with a fresh configuration read; normal refresh
  // intentionally keeps the already loaded provider configuration in memory.
  await page.reload();
  await page.locator('.sidebar-settings').waitFor();
  await refresh();
  await page.getByLabel('实验任务', { exact: true }).fill(task);
  await application.evaluate(({ dialog }, directory) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [directory] }); }, dataset);
  await page.locator('.dataset-button').click();
  await page.getByRole('button', { name: '实验设置', exact: true }).click();
  await page.getByLabel('实验名称', { exact: false }).fill('qa-runtime-health-draft');
  await page.getByLabel('最大探索步数', { exact: true }).fill('9');
  await until(async () => {
    const draft = JSON.parse(await storedDraft() || '{}');
    return draft.task === task && draft.name === 'qa-runtime-health-draft' && draft.options?.agent_steps === 9;
  }, 'The synthetic draft was not persisted');
  draftRaw = await storedDraft();
  await until(async () => await page.locator('.dataset-overview').getAttribute('data-code') === 'ok' &&
    await page.locator('.dataset-overview').getAttribute('data-state') === 'ready', 'The real dataset metadata fixture did not become ready');
  await until(async () => await page.locator('.submit-button').isEnabled() && await page.locator('.submit-button').getAttribute('aria-label') === '开始实验', 'Fixture setup did not enable the baseline experiment control');

  await openRuntime();
  await page.getByText('运行环境已安装', { exact: true }).waitFor();
  assert.equal(await page.getByText('运行环境检查通过', { exact: true }).count(), 0);
  assert.equal(await page.locator('.runtime-checked-at').count(), 0);
  await page.getByRole('button', { name: '修复运行环境', exact: true }).click();
  await page.getByRole('group', { name: '修复运行环境？', exact: true }).waitFor();
  await until(() => page.getByRole('button', { name: '取消返回', exact: true }).evaluate(element => document.activeElement === element), 'Repair confirmation must initially focus cancellation');
  await page.getByRole('button', { name: '取消返回', exact: true }).click();
  assert.deepEqual(await maintenanceRequests(), []);
  checks.push('installed-receipt-is-not-labelled-verified-and-repair-cancel-sends-no-request');

  // Hold a ready status snapshot across the check POST. The POST updates the
  // renderer immediately; its following refresh waits for this older request.
  await application.evaluate(() => { globalThis.__runtimeHealthFixture.holdNextStatus = true; });
  await refresh();
  await until(() => application.evaluate(() => typeof globalThis.__runtimeHealthFixture.releaseHeldStatus === 'function'), 'An older status response was not held');
  assert.equal(await application.evaluate(() => globalThis.__runtimeHealthFixture.heldStatus.engine.ready), true);
  await page.getByRole('button', { name: '检查运行环境', exact: true }).click();
  await page.getByRole('button', { name: '取消检查', exact: true }).waitFor();
  await application.evaluate(() => globalThis.__runtimeHealthFixture.releaseHeldStatus());
  // The action only becomes available after onRefresh settles, proving that
  // the old ready response reached the renderer before these assertions.
  await until(() => page.getByRole('button', { name: '取消检查', exact: true }).isEnabled(), 'The check action did not settle after releasing the older ready response');
  assert.match(await page.locator('.local-status').innerText(), /正在检查运行环境/);
  assert.equal(await page.getByRole('button', { name: '修复运行环境', exact: true }).count(), 0);
  await closeSettings();
  assert.notEqual(await page.locator('.submit-button').getAttribute('aria-label'), '开始实验');
  await page.getByLabel('实验任务', { exact: true }).focus();
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Enter' : 'Control+Enter');
  await page.getByRole('dialog', { name: '设置', exact: true }).waitFor();
  await closeSettings();
  await assertDraft();
  assert.deepEqual(await application.evaluate(() => globalThis.__runtimeHealthFixture.forbidden), []);
  checks.push('older-ready-status-response-cannot-overwrite-check-busy-or-enable-run');
  await openRuntime();
  await page.screenshot({ path: path.join(output, '01-checking.png') });
  await page.getByRole('button', { name: '取消检查', exact: true }).click();
  await until(() => page.getByRole('button', { name: '取消检查', exact: true }).isDisabled(), 'Pending cancellation must disable another cancellation');
  await setEngine({ ...initialEngine, ready: false, error: 'Engine check cancelled.' });
  await page.getByText('操作已取消', { exact: true }).waitFor();
  await page.getByRole('button', { name: '检查运行环境', exact: true }).waitFor();
  checks.push('check-busy-blocks-experiment-submit-and-cancel-waits-for-worker-settlement');

  await page.getByRole('button', { name: '检查运行环境', exact: true }).click();
  await page.getByRole('button', { name: '取消检查', exact: true }).waitFor();
  const failedEngine = { ...initialEngine, ready: false, health: 'failed', checked_at: checkedAt, error: 'Engine verification exited with code 1. See the engine log.', log: 'Synthetic missing dependency for UI acceptance.' };
  await setEngine(failedEngine);
  await page.getByText('运行环境需要修复', { exact: true }).waitFor();
  assert.equal(await page.locator('.runtime-checked-at time').getAttribute('datetime'), checkedAt);
  assert.equal(await page.getByRole('button', { name: '检查运行环境', exact: true }).isEnabled(), true);
  await page.screenshot({ path: path.join(output, '02-check-failed.png') });
  checks.push('failed-check-shows-failure-and-last-completed-verification-with-retry-actions');

  await application.evaluate(() => { globalThis.__runtimeHealthFixture.nextFailure = { action: 'repair', message: 'Synthetic repair could not start; try again.' }; });
  await confirmRepair();
  await page.getByText('Synthetic repair could not start; try again.', { exact: false }).waitFor();
  assert.equal(await page.getByRole('button', { name: '修复运行环境', exact: true }).isEnabled(), true);
  await confirmRepair();
  await page.getByRole('button', { name: '取消修复', exact: true }).waitFor();
  assert.equal((await maintenanceRequests()).filter(request => request.action === 'repair').length, 2);
  await closeSettings();
  assert.notEqual(await page.locator('.submit-button').getAttribute('aria-label'), '开始实验');
  await page.locator('.submit-button').click();
  await page.getByRole('dialog', { name: '设置', exact: true }).waitFor();
  await closeSettings();
  await assertDraft();
  await openRuntime();
  await page.getByRole('button', { name: '取消修复', exact: true }).click();
  await until(() => page.getByRole('button', { name: '取消修复', exact: true }).isDisabled(), 'Repair cancellation must wait for worker settlement');
  await setEngine({ ...initialEngine, ready: false, checked_at: checkedAt, error: 'Engine repair cancelled.' });
  await page.getByText('操作已取消', { exact: true }).waitFor();
  checks.push('repair-requires-confirmed-post-errors-stay-retryable-and-cancellation-preserves-draft');

  await confirmRepair();
  await page.getByRole('button', { name: '取消修复', exact: true }).waitFor();
  await setEngine({ ...initialEngine, health: 'healthy', checked_at: repairedAt });
  await page.getByText('运行环境检查通过', { exact: true }).waitFor();
  assert.equal(await page.locator('.runtime-checked-at time').getAttribute('datetime'), repairedAt);
  await page.screenshot({ path: path.join(output, '03-repair-verified.png') });
  await closeSettings();
  await until(async () => await page.locator('.submit-button').isEnabled() && await page.locator('.submit-button').getAttribute('aria-label') === '开始实验', 'Verified fixture did not restore experiment availability');
  await assertDraft();
  checks.push('verified-repair-restores-readiness-with-new-verification-time-and-unchanged-draft');

  await setEngine({ ...initialEngine, ready: false, busy_external: true, error: 'Another BioMaker engine operation is active. Wait for it to finish.' });
  await until(async () => await page.locator('.submit-button').getAttribute('aria-label') !== '开始实验', 'External maintenance must remove experiment submission');
  await page.locator('.submit-button').click();
  await page.getByRole('dialog', { name: '设置', exact: true }).waitFor();
  await openRuntime();
  assert.equal(await page.getByRole('button', { name: '检查运行环境', exact: true }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '修复运行环境', exact: true }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: /^(取消检查|取消修复|取消安装)$/ }).count(), 0);
  await page.screenshot({ path: path.join(output, '04-external-maintenance.png') });
  await setEngine({ ...initialEngine, health: 'healthy', checked_at: repairedAt });
  await page.getByText('运行环境检查通过', { exact: true }).waitFor();
  await closeSettings();
  await assertDraft();
  checks.push('external-worker-lock-blocks-local-maintenance-and-run-without-offering-cancel');

  await setEngine({ ...initialEngine, ready: false, health: 'failed', maintenance_unavailable: true, error: 'The engine operation lock is unavailable. Check local disk access.' });
  await openRuntime();
  await page.getByText('运行环境暂时无法维护', { exact: true }).waitFor();
  assert.equal(await page.getByRole('button', { name: '检查运行环境', exact: true }).isEnabled(), true);
  assert.equal(await page.getByRole('button', { name: '修复运行环境', exact: true }).isEnabled(), true);
  assert.equal(await page.getByRole('button', { name: /^(取消检查|取消修复|取消安装)$/ }).count(), 0);
  assert.equal(await page.getByText('另一进程正在维护运行环境', { exact: true }).count(), 0);
  await page.screenshot({ path: path.join(output, '05-maintenance-unavailable.png') });
  await setEngine({ ...initialEngine, health: 'healthy', checked_at: repairedAt });
  await page.getByText('运行环境检查通过', { exact: true }).waitFor();
  await closeSettings();
  await assertDraft();
  checks.push('disk-access-failure-is-distinct-from-another-worker-and-maintenance-remains-retryable');

  const observed = await application.evaluate(async () => {
    const f = globalThis.__runtimeHealthFixture;
    globalThis.fetch = f.originalFetch;
    return { requests: f.requests, forbidden: f.forbidden };
  });
  const actual = await page.evaluate(async () => ({ status: await window.biomaker.request('/api/desktop/status'), env: await window.biomaker.request('/api/env'), runs: await window.biomaker.request('/api/runs') }));
  assert.equal(actual.status.engine.ready, false);
  assert.equal(actual.env.values.OPENAI_API_KEY.configured, false);
  assert.deepEqual(actual.runs, []);
  assert.equal(await readFile(path.join(profile, 'config.env'), 'utf8').catch(error => error.code === 'ENOENT' ? '' : Promise.reject(error)), '');
  assert.deepEqual(observed.forbidden, []);
  assert.deepEqual(observed.requests.map(request => request.action), ['check', 'cancel', 'check', 'repair', 'repair', 'cancel', 'repair']);
  assert.equal(observed.requests.every(request => request.method === 'POST'), true);
  assert.deepEqual(errors, []);
  checks.push('synthetic-maintenance-only-no-actual-engine-install-provider-call-run-or-renderer-error');
  const report = { passed: true, mode: 'synthetic-engine-state Electron UI acceptance; not real engine verification', executablePath: executablePath || 'development Electron', profile, checks, maintenanceRequests: observed.requests, forbiddenRequests: observed.forbidden, rendererErrors: errors };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-6000));
  }
  throw error;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => { for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await application.close();
  }
}
