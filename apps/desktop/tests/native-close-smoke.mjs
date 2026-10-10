// Real Electron lifecycle checks with an isolated profile and synthetic credentials.
// Native dialog responses are automated; the settings API and sidecar remain real.
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'biomaker-native-close-'));
const output = process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output', 'native-close');
await mkdir(output, { recursive: true });
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = { ...process.env, BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile };
delete env.ELECTRON_RUN_AS_NODE;
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
let application;
let page;
const errors = [];
const checks = [];

async function waitUntil(callback, message) {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    if (await callback()) return;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error(message);
}
async function ready(nextPage) {
  page = nextPage || await application.firstWindow();
  page.on('pageerror', error => errors.push(error.message));
  await waitUntil(async () => {
    try { return (await page.evaluate(() => window.biomaker.request('/api/health'))).status === 'ok'; }
    catch { return false; }
  }, 'Real sidecar did not become healthy');
  await page.locator('.sidebar-settings').waitFor();
}
async function settings() {
  await page.locator('.sidebar-settings').click();
  await page.getByLabel('默认模型', { exact: true }).waitFor();
}
async function triggerClose(quit = false) {
  const previous = await application.evaluate(() => globalThis.__closeDialogs.length);
  await application.evaluate(({ app, BrowserWindow }, quitting) => {
    setTimeout(() => quitting ? app.quit() : BrowserWindow.getAllWindows()[0].close(), 0);
  }, quit);
  await waitUntil(async () => (await application.evaluate(() => globalThis.__closeDialogs.length)) > previous,
    'Native close attempt did not present a settings guard');
  await page.getByLabel('默认模型', { exact: true }).waitFor();
  // A canceled exit must leave the service alive, not just its visible window.
  assert.equal((await page.evaluate(() => window.biomaker.request('/api/health'))).status, 'ok');
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  await ready();
  await application.evaluate(({ dialog }) => {
    globalThis.__closeDialogs = [];
    globalThis.__discardNext = false;
    const answer = (...args) => {
      const options = args.at(-1);
      globalThis.__closeDialogs.push({ message: options.message, buttons: options.buttons });
      const discard = globalThis.__discardNext;
      globalThis.__discardNext = false;
      const index = options.buttons.findIndex(label => /放弃|丢弃|Discard/i.test(label));
      return discard && index >= 0 ? index : options.cancelId ?? 0;
    };
    dialog.showMessageBoxSync = (...args) => answer(...args);
    dialog.showMessageBox = async (...args) => ({ response: answer(...args), checkboxChecked: false });
  });
  await settings();
  await page.getByLabel('API 地址', { exact: false }).fill('http://127.0.0.1:9/v1');
  await page.getByLabel('API Key', { exact: false }).first().fill('qa-saved-key-never-sent');
  await page.getByLabel('默认模型', { exact: true }).fill('qa-saved-model');
  await page.getByRole('button', { name: '保存配置', exact: true }).click();
  await page.getByText('所有更改已保存', { exact: true }).waitFor();

  await page.getByLabel('默认模型', { exact: true }).fill('qa-unsaved-model');
  await page.getByLabel('API Key', { exact: false }).first().fill('qa-unsaved-key-never-persisted');
  await triggerClose();
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'qa-unsaved-model');
  assert.equal(await page.getByLabel('API Key', { exact: false }).first().inputValue(), 'qa-unsaved-key-never-persisted');
  checks.push('cancel-native-window-close-preserves-form-and-service');
  await triggerClose(true);
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'qa-unsaved-model');
  checks.push('cancel-native-app-quit-preserves-form-and-service');

  await application.evaluate(() => { globalThis.__discardNext = true; });
  const closed = page.waitForEvent('close');
  await application.evaluate(({ BrowserWindow }) => { setTimeout(() => BrowserWindow.getAllWindows()[0].close(), 0); });
  await closed;
  if (process.platform !== 'darwin') throw new Error('Window recreation acceptance currently targets macOS');
  const reopened = application.waitForEvent('window');
  await application.evaluate(({ app }) => { app.emit('activate'); });
  await ready(await reopened);
  await settings();
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'qa-saved-model');
  assert.equal(await page.getByLabel('API Key', { exact: false }).first().inputValue(), '');
  assert(!(await readFile(path.join(profile, 'config.env'), 'utf8')).includes('qa-unsaved-key-never-persisted'));
  assert(!(await page.evaluate(() => JSON.stringify(localStorage))).includes('qa-unsaved-key-never-persisted'));
  checks.push('explicit-discard-reopens-saved-settings-without-persisting-unsaved-secret');

  // Delay exactly one model-settings response to exercise a real pending request.
  await application.evaluate(() => {
    const original = globalThis.fetch;
    globalThis.__originalFetch = original;
    globalThis.fetch = async (input, options) => {
      if (String(input).endsWith('/api/env') && options?.method === 'PUT') {
        await new Promise(resolve => { globalThis.__releaseSave = resolve; });
        return new Response(JSON.stringify({ detail: 'Synthetic settings save failure' }), {
          status: 503, headers: { 'Content-Type': 'application/json' },
        });
      }
      return original(input, options);
    };
  });
  await page.getByLabel('默认模型', { exact: true }).fill('qa-retried-model');
  await page.getByRole('button', { name: '保存配置', exact: true }).click();
  await waitUntil(() => application.evaluate(() => typeof globalThis.__releaseSave === 'function'), 'Save request did not start');
  assert.equal(await page.getByLabel('默认模型', { exact: true }).isDisabled(), true);
  assert.equal(await page.getByLabel('API Key', { exact: false }).first().isDisabled(), true);
  assert.equal(await page.getByLabel('API 地址', { exact: false }).isDisabled(), true);
  await triggerClose(true);
  const savingDialog = await application.evaluate(() => globalThis.__closeDialogs.at(-1));
  assert(!savingDialog.buttons.some(label => /放弃|丢弃|Discard/i.test(label)), 'In-flight saves cannot be discarded');
  checks.push('pending-save-locks-fields-and-keeps-service-alive-on-quit');
  await application.evaluate(() => { globalThis.fetch = globalThis.__originalFetch; globalThis.__releaseSave(); });
  await page.getByText('Synthetic settings save failure', { exact: false }).waitFor();
  assert.equal(await page.getByLabel('默认模型', { exact: true }).isDisabled(), false);
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'qa-retried-model');
  await triggerClose();
  checks.push('failed-save-retains-edits-and-native-close-protection');
  await page.getByRole('button', { name: '保存配置', exact: true }).click();
  await page.getByText('所有更改已保存', { exact: true }).waitFor();
  assert.equal((await page.evaluate(() => window.biomaker.request('/api/env'))).values.BIOMAKER_MAIN_MODEL.value, 'qa-retried-model');
  await page.screenshot({ path: path.join(output, 'saved-settings.png') });
  const promptCount = await application.evaluate(() => globalThis.__closeDialogs.length);
  const lastClose = page.waitForEvent('close');
  await application.evaluate(({ BrowserWindow }) => { setTimeout(() => BrowserWindow.getAllWindows()[0].close(), 0); });
  await lastClose;
  assert.equal(await application.evaluate(() => globalThis.__closeDialogs.length), promptCount);
  checks.push('successful-save-closes-without-unnecessary-prompt');
  assert.deepEqual(errors, []);
  const report = { passed: true, executablePath: executablePath || 'development Electron', profile, checks, rendererErrors: errors };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-3000));
  }
  throw error;
} finally {
  if (application) {
    // Only the isolated QA process is forcibly closed after recording a failure.
    await application.evaluate(({ BrowserWindow }) => { for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await application.close();
  }
}
