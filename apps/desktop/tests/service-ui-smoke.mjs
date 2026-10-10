// Renderer acceptance against a real Electron shell and Python sidecar.
// Response timing and obsolete failures are controlled. Provider settings file
// changes use only synthetic model names in an isolated profile; no PUT, model
// call, engine installation, or experiment is performed.
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-service-ui-'));
const profile = path.join(temporary, 'profile');
const output = process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/service-ui');
await mkdir(profile, { recursive: true }); await mkdir(output, { recursive: true });
const harness = path.join(temporary, 'entry.cjs');
await writeFile(harness, `
const { app, ipcMain } = require('electron');
app.setAppPath(${JSON.stringify(root)});
globalThis.__qaRequests = [];
const realHandle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, callback) => realHandle(channel, async (event, ...args) => {
  if (channel !== 'biomaker:request') return callback(event, ...args);
  const record = { path: args[0], method: args[1]?.method || 'GET', done: false }; globalThis.__qaRequests.push(record);
  const startingPID = globalThis.__qaChild?.pid;
  try {
    if (globalThis.__delayRuns && args[0] === '/api/runs') {
      globalThis.__delayRuns = false;
      await new Promise((_resolve, reject) => {
        globalThis.__releaseOldRuns = () => reject(new Error('Synthetic obsolete response'));
      });
    }
    const result = await callback(event, ...args);
    if (globalThis.__holdEnvironment && args[0] === '/api/env' && record.method === 'GET' && startingPID !== globalThis.__holdEnvAfterPID) {
      globalThis.__holdEnvironment = false;
      return await new Promise((resolve, reject) => {
        globalThis.__releaseEnvironment = (fail = false) => {
          delete globalThis.__releaseEnvironment;
          if (fail) reject(new Error('Synthetic obsolete environment response'));
          else resolve(result);
        };
      });
    }
    return result;
  } finally { record.done = true; }
});
const childProcess = require('node:child_process');
const realSpawn = childProcess.spawn;
childProcess.spawn = (...args) => {
  const child = realSpawn(...args);
  if (args[2]?.env?.BIOMAKER_DESKTOP_TOKEN) globalThis.__qaChild = child;
  return child;
};
const realFetch = globalThis.fetch;
globalThis.__holdHealth = true;
globalThis.fetch = async (input, options) => {
  const pathname = new URL(String(input)).pathname;
  if (globalThis.__failHealth && pathname === '/api/health') {
    globalThis.__failHealth = false; throw new TypeError('Synthetic recoverable transport failure');
  }
  if (globalThis.__holdHealth && pathname === '/api/health') {
    globalThis.__holdHealth = false;
    await new Promise(resolve => { globalThis.__releaseHealth = resolve; });
  }
  return realFetch(input, options);
};
require(${JSON.stringify(path.join(root, 'dist-electron/main.js'))});
`);
const env = { ...process.env, BIOMAKER_DESKTOP_HOME: profile, BIOMAKER_DESKTOP_USER_DATA: profile };
delete env.ELECTRON_RUN_AS_NODE;
let application;
let page;
const checks = [];
const errors = [];
async function until(predicate, message) {
  const deadline = Date.now() + 30_000;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 25));
  }
}
async function renderFrame() { await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))); }
async function replaceServiceWithHeldEnvironment() {
  const previousPID = await application.evaluate(() => {
    delete globalThis.__releaseHealth;
    globalThis.__holdHealth = true;
    globalThis.__holdEnvironment = true;
    const child = globalThis.__qaChild;
    globalThis.__holdEnvAfterPID = child.pid;
    child.kill('SIGKILL');
    return child.pid;
  });
  await page.locator('.service-banner').filter({ hasText: '正在恢复本地服务' }).waitFor();
  await until(() => application.evaluate(() => typeof globalThis.__releaseHealth === 'function'), 'Replacement service did not reach its held health check');
  await application.evaluate(() => { globalThis.__releaseHealth(); });
  await page.locator('.service-banner').waitFor({ state: 'hidden' });
  await until(() => application.evaluate(() => typeof globalThis.__releaseEnvironment === 'function'), 'The automatic App environment response was not held');
  assert.notEqual(await application.evaluate(() => globalThis.__qaChild.pid), previousPID);
}
async function readReview() {
  await until(async () => await page.locator('.provider-settings-review').getAttribute('data-state') === 'required', 'Dirty settings did not require review after recovery');
  assert.equal(await page.getByRole('button', { name: '保存配置', exact: true }).isDisabled(), true);
  await page.getByRole('button', { name: '读取最新已保存配置', exact: true }).click();
  await until(async () => await page.locator('.provider-settings-review').getAttribute('data-state') === 'loaded', 'Latest saved settings did not load for review');
}
async function releaseEnvironment(fail = false) {
  const pending = await application.evaluate(() => globalThis.__qaRequests.findIndex(request => request.path === '/api/env' && !request.done));
  assert(pending >= 0);
  await application.evaluate((_electron, fail) => globalThis.__releaseEnvironment(fail), fail);
  await until(() => application.evaluate((_electron, index) => globalThis.__qaRequests[index].done, pending), 'The held App environment response did not settle');
  await renderFrame();
  assert.equal(await page.locator('.connection-banner').count(), 0, 'An obsolete environment response must not create a connection error');
  assert.equal(await page.locator('.provider-settings-review').count(), 0, 'An obsolete environment response must not reopen review');
}
try {
  application = await electron.launch({ args: [harness], env, timeout: 60_000 });
  page = await application.firstWindow();
  page.on('pageerror', error => errors.push(error.message));
  await page.locator('.service-banner').filter({ hasText: '正在启动本地服务' }).waitFor();
  assert.equal(await page.locator('.connection-banner').count(), 0);
  await page.getByLabel('实验任务', { exact: true }).fill('Synthetic lifecycle draft. Never submit.');
  await page.locator('.sidebar-settings').click();
  await page.getByRole('button', { name: '运行环境', exact: true }).click();
  assert.equal(await page.getByRole('button', { name: '等待本地服务', exact: true }).isDisabled(), true);
  assert.deepEqual(await application.evaluate(() => globalThis.__qaRequests), [], 'Renderer must not call business APIs while the service is starting');
  await page.screenshot({ path: path.join(output, 'starting.png') });
  await page.keyboard.press('Escape');
  await until(() => application.evaluate(() => typeof globalThis.__releaseHealth === 'function'), 'Startup health was not held');
  await application.evaluate(() => { globalThis.__releaseHealth(); });
  await page.locator('.service-banner').waitFor({ state: 'hidden' });
  await page.locator('.onboarding').waitFor();
  assert.equal(await page.locator('.connection-banner').count(), 0);
  checks.push('starting-is-neutral-and-pauses-api-and-runtime-installation-until-native-ready');

  await application.evaluate(({ BrowserWindow }) => {
    globalThis.__delayRuns = true;
    BrowserWindow.getAllWindows()[0].webContents.send('biomaker:menu-action', 'refresh');
  });
  await until(() => application.evaluate(() => typeof globalThis.__releaseOldRuns === 'function'), 'Old renderer refresh was not held');
  const oldRequest = await application.evaluate(() => globalThis.__qaRequests.findIndex(record => record.path === '/api/runs' && !record.done));
  const revision = (await page.evaluate(() => window.biomaker.getServiceState())).revision;
  await application.evaluate(() => { globalThis.__failHealth = true; });
  await assert.rejects(page.evaluate(() => window.biomaker.request('/api/health')), /Synthetic recoverable transport failure/);
  await until(async () => (await page.evaluate(() => window.biomaker.getServiceState())).revision > revision + 1, 'New native ready generation was not delivered');
  await until(() => application.evaluate((_electron, index) => globalThis.__qaRequests.slice(index + 1).some(record => record.path === '/api/runs' && record.done), oldRequest), 'New ready generation did not refresh while the old request was pending');
  await application.evaluate(() => { globalThis.__releaseOldRuns(); });
  await until(() => application.evaluate((_electron, index) => globalThis.__qaRequests[index].done, oldRequest), 'Old response did not settle');
  await renderFrame();
  assert.equal(await page.locator('.connection-banner').count(), 0, 'An obsolete error must not overwrite recovered workspace state');
  assert.equal(await page.locator('.service-banner').count(), 0);
  await application.evaluate(({ BrowserWindow }, oldRevision) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:service-state', { status: 'failed', revision: oldRevision, message: 'Synthetic obsolete state' }), revision);
  await renderFrame();
  assert.equal(await page.locator('.service-banner').count(), 0, 'An older lifecycle snapshot must be ignored');
  checks.push('ready-push-starts-a-new-refresh-and-rejects-obsolete-api-errors-and-state-snapshots');

  await page.locator('.sidebar-settings').click();
  await page.getByLabel('默认模型', { exact: true }).fill('unsaved-lifecycle-model');
  await page.getByLabel('API Key', { exact: false }).first().fill('synthetic-unsaved-key-never-sent');
  const formerPID = await application.evaluate(() => {
    delete globalThis.__releaseHealth;
    globalThis.__holdHealth = true;
    globalThis.__holdEnvironment = true;
    const child = globalThis.__qaChild;
    globalThis.__holdEnvAfterPID = child.pid;
    child.kill('SIGKILL'); return child.pid;
  });
  await page.locator('.service-banner').filter({ hasText: '正在恢复本地服务' }).waitFor();
  assert.equal(await page.locator('.connection-banner').count(), 0);
  assert.equal(await page.getByRole('button', { name: '保存配置', exact: true }).isDisabled(), true);
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'unsaved-lifecycle-model');
  await page.screenshot({ path: path.join(output, 'reconnecting.png') });
  await until(() => application.evaluate(() => typeof globalThis.__releaseHealth === 'function'), 'Replacement service did not reach its held health check');
  await application.evaluate(() => { globalThis.__releaseHealth(); });
  await page.locator('.service-banner').waitFor({ state: 'hidden' });
  assert.notEqual(await application.evaluate(() => globalThis.__qaChild.pid), formerPID);
  await until(() => application.evaluate(() => typeof globalThis.__releaseEnvironment === 'function'), 'The restored App environment response was not held');
  await writeFile(path.join(profile, 'config.env'), 'BIOMAKER_MAIN_MODEL=qa-independently-saved-one\n', { mode: 0o600 });
  const review = page.locator('.provider-settings-review');
  await until(async () => await review.getAttribute('data-state') === 'required', 'Dirty settings did not require review after recovery');
  assert.equal(await page.getByRole('button', { name: '保存配置', exact: true }).isDisabled(), true);
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'unsaved-lifecycle-model');
  assert.equal(await page.getByLabel('API Key', { exact: false }).first().inputValue(), 'synthetic-unsaved-key-never-sent');
  await page.getByRole('button', { name: '读取最新已保存配置', exact: true }).click();
  await until(async () => await review.getAttribute('data-state') === 'loaded', 'Latest saved settings did not load for review');
  assert.equal(await review.locator('[data-field="model"]').innerText(), 'qa-independently-saved-one');
  await review.scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, 'recovered-settings-review.png') });
  await page.getByRole('button', { name: '保留输入并继续编辑', exact: true }).click();
  await releaseEnvironment();
  await until(() => page.getByRole('button', { name: '保存配置', exact: true }).isEnabled(), 'Settings were not editable after explicitly reviewing the latest saved configuration');
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'unsaved-lifecycle-model');
  assert.equal(await page.getByLabel('API Key', { exact: false }).first().inputValue(), 'synthetic-unsaved-key-never-sent');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: '放弃更改', exact: true }).click();
  assert.equal(await page.getByLabel('实验任务', { exact: true }).inputValue(), 'Synthetic lifecycle draft. Never submit.');
  assert.equal((await page.evaluate(() => window.biomaker.request('/api/env'))).values.OPENAI_API_KEY.configured, false);
  checks.push('real-service-replacement-preserves-draft-and-unsaved-provider-fields-and-requires-explicit-settings-review-without-saving-secrets');

  // Hold a real old App GET across an independent local-file update and explicit
  // Load. The accepted public settings must remain in the now-clean form.
  await page.locator('.sidebar-settings').click();
  await page.getByLabel('默认模型', { exact: true }).fill('qa-unsaved-before-load');
  await replaceServiceWithHeldEnvironment();
  await writeFile(path.join(profile, 'config.env'), 'BIOMAKER_MAIN_MODEL=qa-independently-saved-two\n', { mode: 0o600 });
  await readReview();
  await page.getByRole('button', { name: '载入已保存配置并替换输入', exact: true }).click();
  await releaseEnvironment();
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'qa-independently-saved-two');
  assert.equal(await page.getByRole('button', { name: '保存配置', exact: true }).isDisabled(), true);
  checks.push('held-old-real-App-environment-success-cannot-rearm-kept-input-review-or-reset-explicitly-loaded-clean-settings');

  // The same epoch rule must suppress an obsolete error after explicit Keep.
  await page.getByLabel('默认模型', { exact: true }).fill('qa-unsaved-after-review');
  await replaceServiceWithHeldEnvironment();
  await writeFile(path.join(profile, 'config.env'), 'BIOMAKER_MAIN_MODEL=qa-independently-saved-three\n', { mode: 0o600 });
  await readReview();
  await page.getByRole('button', { name: '保留输入并继续编辑', exact: true }).click();
  await releaseEnvironment(true);
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'qa-unsaved-after-review');
  assert.equal(await page.getByRole('button', { name: '保存配置', exact: true }).isEnabled(), true);
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: '放弃更改', exact: true }).click();
  assert.equal(await page.getByLabel('实验任务', { exact: true }).inputValue(), 'Synthetic lifecycle draft. Never submit.');
  assert.equal(await readFile(path.join(profile, 'config.env'), 'utf8'), 'BIOMAKER_MAIN_MODEL=qa-independently-saved-three\n');
  const operations = await application.evaluate(() => globalThis.__qaRequests);
  assert.equal(operations.some(operation => operation.method !== 'GET'), false, 'This acceptance must perform no PUT, provider request, installation, or run');
  checks.push('held-obsolete-App-environment-error-cannot-overwrite-explicit-review-or-show-connection-failure-and-no-api-mutations-occur');
  assert.deepEqual(errors, []);
  const report = { passed: true, profile, checks, rendererErrors: errors };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  if (page && !page.isClosed()) await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
  throw error;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => { for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await application.close();
  }
}
