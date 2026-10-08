// Real packaged/development Electron startup failure, native diagnostics and
// manual recovery. All faults are confined to this synthetic profile.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { access, mkdir, mkdtemp, readFile, readdir, rename, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'biomaker-recovery-private-path-'));
const output = process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/service-recovery');
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const secrets = ['qa-private-config-token', 'qa-private-log-content', 'qa-private-task-content', 'qa-private-unsaved-model', 'qa-private-unsaved-key'];
await mkdir(output, { recursive: true });
await writeFile(path.join(profile, 'config.env'), `OPENAI_API_KEY=${secrets[0]}\n`);
// The existing Sidecar constructor tries to mkdir logs. A regular file here
// produces a real EEXIST before it can construct or spawn any Python service.
await writeFile(path.join(profile, 'logs'), secrets[1]);
const configBefore = await readFile(path.join(profile, 'config.env'));
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
let application;
let page;
const errors = [];
const checks = [];
async function until(predicate, message, timeout = 60000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 75));
  }
  throw new Error(message);
}
const state = () => page.evaluate(() => window.biomaker.getServiceState());
async function saveDialog(destination, canceled = false) {
  await application.evaluate(({ dialog }, value) => {
    dialog.showSaveDialog = async () => { globalThis.__qaSaveDialogs++; return { canceled: value.canceled, filePath: value.destination }; };
  }, { destination, canceled });
}
async function assertReport(filename, status) {
  await until(async () => { try { await access(filename); return true; } catch { return false; } }, 'Native diagnostics were not written');
  const contents = await readFile(filename, 'utf8');
  const report = JSON.parse(contents);
  assert.equal(report.schemaVersion, 1);
  assert.equal(report.service.status, status);
  assert(contents.length < 65536);
  for (const value of [...secrets, profile, os.homedir(), 'OPENAI_API_KEY', 'record_json', 'config.env']) assert(!contents.includes(value), `Diagnostics leaked a forbidden value: ${value}`);
  assert.equal((await stat(filename)).mode & 0o777, 0o600);
  return report;
}
async function settingsProvider() {
  await page.locator('.sidebar-settings').click();
  await page.getByRole('button', { name: '模型服务', exact: true }).click();
}
try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60000 });
  page = await application.firstWindow();
  page.setDefaultTimeout(20000);
  page.on('pageerror', error => errors.push(error.message));
  await until(async () => { try { return (await state()).status === 'failed'; } catch { return false; } }, 'Constructor failure did not leave a usable failed window');
  assert.equal((await state()).code, 'SERVICE_CONSTRUCTION_FAILED');
  await page.locator('.service-banner').getByRole('button', { name: '重试本地服务', exact: true }).waitFor();
  const identity = await application.evaluate(({ BrowserWindow }) => ({ pid: process.pid, windowId: BrowserWindow.getAllWindows()[0].id }));
  await page.evaluate(() => { window.__qaRecoveryMarker = 'renderer-preserved'; });
  await page.getByLabel('实验任务', { exact: true }).fill(secrets[2]);
  await application.evaluate(() => {
    globalThis.__qaSaveDialogs = 0;
    globalThis.__qaForbiddenRequests = [];
    const original = globalThis.fetch;
    globalThis.fetch = (input, options) => {
      const url = new URL(String(input));
      if (url.hostname !== '127.0.0.1' || !['GET', 'HEAD'].includes(options?.method || 'GET')) {
        globalThis.__qaForbiddenRequests.push({ method: options?.method, path: url.pathname });
        throw new Error('QA forbids provider calls and all service mutations');
      }
      return original(input, options);
    };
  });
  const failedReport = path.join(profile, 'constructor-failed.json');
  await saveDialog(failedReport);
  await page.locator('.service-banner').getByRole('button', { name: '导出诊断', exact: true }).click();
  const constructorReport = await assertReport(failedReport, 'failed');
  assert.equal(constructorReport.storage.root.directory, true);
  assert.deepEqual(constructorReport.storage.logs, { exists: true, directory: false, existingWritable: false });
  await page.screenshot({ path: path.join(output, '01-constructor-failure.png') });
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.screenshot({ path: path.join(output, '01b-compact-failure.png') });
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1360, 900));
  checks.push('real-constructor-failure-keeps-window-draft-and-native-diagnostics-available');

  const concurrentReport = path.join(profile, 'concurrent-export.json');
  await application.evaluate(({ dialog }, destination) => {
    globalThis.__qaConcurrentDialogs = 0;
    dialog.showSaveDialog = () => {
      globalThis.__qaConcurrentDialogs++;
      return new Promise(resolve => { globalThis.__qaReleaseExport = () => resolve({ canceled: false, filePath: destination }); });
    };
  }, concurrentReport);
  const concurrentExport = page.evaluate(() => Promise.all([window.biomaker.exportDiagnostics(), window.biomaker.exportDiagnostics()]));
  await until(() => application.evaluate(() => typeof globalThis.__qaReleaseExport === 'function'), 'Concurrent native export dialog did not open');
  assert.equal(await application.evaluate(() => globalThis.__qaConcurrentDialogs), 1);
  await application.evaluate(() => globalThis.__qaReleaseExport());
  assert.deepEqual(await concurrentExport, [concurrentReport, concurrentReport]);
  await assertReport(concurrentReport, 'failed');
  checks.push('concurrent-native-exports-share-one-save-dialog-and-one-result');

  await saveDialog(path.join(profile, 'cancelled.json'), true);
  const dialogCount = await application.evaluate(() => globalThis.__qaSaveDialogs);
  await page.locator('.service-banner').getByRole('button', { name: '导出诊断', exact: true }).click();
  await until(() => application.evaluate((_electron, count) => globalThis.__qaSaveDialogs > count, dialogCount), 'Cancel dialog did not open');
  await assert.rejects(access(path.join(profile, 'cancelled.json')));
  await saveDialog(path.join(profile, 'missing-parent', 'report.json'));
  await page.locator('.service-banner').getByRole('button', { name: '导出诊断', exact: true }).click();
  await page.locator('.service-banner').getByText('未能导出诊断，请再试一次。', { exact: true }).waitFor();
  await page.locator('.service-banner').getByText('本地错误详情', { exact: true }).click();
  await page.locator('.service-banner').getByText(/无法保存诊断文件/).waitFor();
  assert.equal((await state()).status, 'failed');
  await saveDialog(path.join(profile, 'retry-export.json'));
  await page.locator('.service-banner').getByRole('button', { name: '导出诊断', exact: true }).click();
  await assertReport(path.join(profile, 'retry-export.json'), 'failed');
  checks.push('cancel-does-not-save-and-export-write-failure-can-be-retried');

  await rename(path.join(profile, 'logs'), path.join(profile, 'preserved-qa-log-blocker'));
  const failedRevision = (await state()).revision;
  await page.locator('.service-banner').getByRole('button', { name: '重试本地服务', exact: true }).click();
  await until(async () => (await state()).status === 'ready', 'Manual retry did not recover after removing the profile fault');
  assert((await state()).revision > failedRevision);
  assert.equal(await page.evaluate(() => window.__qaRecoveryMarker), 'renderer-preserved');
  assert.equal(await page.getByLabel('实验任务', { exact: true }).inputValue(), secrets[2]);
  assert.deepEqual(await application.evaluate(({ BrowserWindow }) => ({ pid: process.pid, windowId: BrowserWindow.getAllWindows()[0].id })), identity);
  const readyState = await state();
  assert.deepEqual(await page.evaluate(() => window.biomaker.retryService()), readyState, 'Retry while healthy must not replace a service');
  checks.push('manual-retry-preserves-native-window-renderer-draft-and-monotonic-revision');

  await settingsProvider();
  await page.getByLabel('默认模型', { exact: true }).fill(secrets[3]);
  await page.getByLabel('API Key', { exact: false }).first().fill(secrets[4]);
  const runtimes = (await readdir(path.join(profile, 'runtimes'))).filter(name => !name.includes('.staging-'));
  assert.equal(runtimes.length, 1);
  const serviceFile = path.join(profile, 'runtimes', runtimes[0], 'service/main.py');
  const heldService = `${serviceFile}.qa-held`;
  const commandLines = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n');
  const children = commandLines.filter(line => line.includes(serviceFile) && line.includes('python'));
  assert.equal(children.length, 1, 'Exactly one QA sidecar must own this private runtime');
  const previousPID = Number(children[0].trim().split(/\s+/)[0]);
  await rename(serviceFile, heldService);
  try {
    process.kill(previousPID, 'SIGKILL');
    await until(async () => (await state()).status === 'failed', 'Real service restart failure did not become actionable');
    assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), secrets[3]);
    assert.equal(await page.getByLabel('API Key', { exact: false }).first().inputValue(), secrets[4]);
    const settings = page.getByRole('dialog');
    await saveDialog(path.join(profile, 'startup-failed.json'));
    await settings.getByRole('button', { name: '导出诊断', exact: true }).click();
    await assertReport(path.join(profile, 'startup-failed.json'), 'failed');
    await page.screenshot({ path: path.join(output, '02-service-failure-preserved-settings.png') });
  } finally { await rename(heldService, serviceFile); }
  const stoppedRevision = (await state()).revision;
  await page.getByRole('dialog').getByRole('button', { name: '重试本地服务', exact: true }).click();
  await until(async () => (await state()).status === 'ready', 'Retry did not recover the restored private service');
  assert((await state()).revision > stoppedRevision);
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), secrets[3]);
  assert.equal(await page.getByLabel('API Key', { exact: false }).first().inputValue(), secrets[4]);
  assert.deepEqual(await readFile(path.join(profile, 'config.env')), configBefore);
  assert.equal(await page.evaluate(() => window.__qaRecoveryMarker), 'renderer-preserved');
  assert.deepEqual(await application.evaluate(({ BrowserWindow }) => ({ pid: process.pid, windowId: BrowserWindow.getAllWindows()[0].id })), identity);
  checks.push('real-service-failure-and-manual-retry-preserve-unsaved-provider-fields-and-original-config');

  await saveDialog(path.join(profile, 'ready-native-menu.json'));
  await application.evaluate(({ Menu }) => {
    const item = Menu.getApplicationMenu().items.flatMap(item => item.submenu?.items ?? []).find(item => item.label === '导出诊断…');
    if (!item) throw new Error('Native diagnostics menu is absent');
    item.click();
  });
  await assertReport(path.join(profile, 'ready-native-menu.json'), 'ready');
  assert.deepEqual(await application.evaluate(() => globalThis.__qaForbiddenRequests), []);
  assert.deepEqual(errors, []);
  checks.push('native-menu-export-needs-no-sidecar-api-and-excludes-private-values', 'no-provider-calls-service-mutations-or-renderer-errors');
  const report = { passed: true, executablePath: executablePath || 'development Electron', profile, checks, rendererErrors: errors, configSHA256: createHash('sha256').update(configBefore).digest('hex') };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-5000));
  }
  throw error;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => { for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await application.close();
  }
}
