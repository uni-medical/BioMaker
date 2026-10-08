// Real Electron IPC and HTTP service; hold only the first native health response
// to make the startup boundary observable independently of machine speed.
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, realpath, rename, rm, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-native-state-'));
const profile = path.join(temporary, 'profile');
const allowed = path.join(temporary, 'allowed');
const nested = path.join(allowed, 'nested');
const outside = path.join(temporary, 'outside');
for (const directory of [profile, nested, outside]) await mkdir(directory, { recursive: true });
await symlink(outside, path.join(allowed, 'escape'), 'dir');
const output = process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/native-service-state');
await mkdir(output, { recursive: true });
const checks = [];
const harness = path.join(temporary, 'entry.cjs');
await writeFile(harness, `
const { app } = require('electron');
app.setAppPath(${JSON.stringify(root)});
const childProcess = require('node:child_process');
const realSpawn = childProcess.spawn;
childProcess.spawn = (...args) => {
  const child = realSpawn(...args);
  if (args[2]?.env?.BIOMAKER_DESKTOP_TOKEN) globalThis.__qaSidecarChild = child;
  return child;
};
globalThis.__realFetch = globalThis.fetch;
let holdStartup = true;
globalThis.fetch = async (input, options) => {
  if (holdStartup && String(input).endsWith('/api/health')) {
    holdStartup = false;
    await new Promise(resolve => { globalThis.__releaseStartupHealth = resolve; });
  }
  return globalThis.__realFetch(input, options);
};
require(${JSON.stringify(path.join(root, 'dist-electron/main.js'))});
`);
const env = { ...process.env, BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile };
delete env.ELECTRON_RUN_AS_NODE;
let application;
let page;
async function waitUntil(predicate, message) {
  const deadline = Date.now() + 30_000;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 20));
  }
}
async function preview(directory, endpoint = '/api/desktop/history/preview') {
  return page.evaluate(({ directory, endpoint }) => window.biomaker.request(endpoint, { method: 'POST', body: { path: directory } }), { directory, endpoint });
}
try {
  application = await electron.launch({ args: [harness], env, timeout: 60_000 });
  page = await application.firstWindow();
  await page.waitForFunction(() => !!window.biomaker);
  await page.evaluate(() => {
    window.__nativeStates = [];
    window.__unsubscribeNativeStates = window.biomaker.onServiceState(state => window.__nativeStates.push(state));
  });
  assert.deepEqual(await page.evaluate(() => window.biomaker.getServiceState()), { status: 'starting', revision: 0 });
  await assert.rejects(page.evaluate(() => window.biomaker.request('/api/health')), /正在启动|starting/);
  await waitUntil(() => application.evaluate(() => typeof globalThis.__releaseStartupHealth === 'function'), 'Native startup did not reach its health check');
  await application.evaluate(() => { globalThis.__releaseStartupHealth(); });
  await waitUntil(async () => (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready', 'Native service did not become ready');
  assert.equal((await page.evaluate(() => window.biomaker.request('/api/health'))).status, 'ok');
  assert.deepEqual(await page.evaluate(() => window.__nativeStates.map(state => state.status)), ['ready']);
  assert.deepEqual(Object.keys(await page.evaluate(() => window.biomaker.getServiceState())).sort(), ['revision', 'status']);
  checks.push('initial-starting-query-authenticated-ready-push-and-no-private-service-fields');

  // An HTTP response with an error status is a data/API failure, not a lost
  // connection. A rejected fetch causes a health probe and state restoration.
  const readyRevision = (await page.evaluate(() => window.biomaker.getServiceState())).revision;
  await application.evaluate(() => {
    globalThis.fetch = async (input, options) => String(input).endsWith('/api/health')
      ? new Response(JSON.stringify({ detail: 'Synthetic HTTP failure' }), { status: 503, headers: { 'Content-Type': 'application/json' } })
      : globalThis.__realFetch(input, options);
  });
  await assert.rejects(page.evaluate(() => window.biomaker.request('/api/health')), /Synthetic HTTP failure/);
  assert.deepEqual(await page.evaluate(() => window.biomaker.getServiceState()), { status: 'ready', revision: readyRevision });
  await application.evaluate(() => {
    let failRequest = true;
    globalThis.fetch = async (input, options) => {
      if (failRequest && String(input).endsWith('/api/health')) { failRequest = false; throw new TypeError('Synthetic transport failure'); }
      return globalThis.__realFetch(input, options);
    };
  });
  await assert.rejects(page.evaluate(() => window.biomaker.request('/api/health')), /Synthetic transport failure/);
  await waitUntil(async () => (await page.evaluate(() => window.biomaker.getServiceState())).revision === readyRevision + 2, 'Transport health probe did not restore the ready state');
  assert.deepEqual(await page.evaluate(() => window.__nativeStates.slice(-2).map(state => state.status)), ['reconnecting', 'ready']);
  checks.push('http-errors-preserve-ready-transport-errors-probe-and-push-reconnection');

  // A delayed rejection from an older request must not destabilize a service
  // generation that another health probe has already confirmed ready.
  await application.evaluate(() => {
    let requests = 0;
    globalThis.fetch = async (input, options) => {
      if (String(input).endsWith('/api/health')) {
        requests++;
        if (requests === 1) return new Promise((_resolve, reject) => { globalThis.__rejectOldRequest = () => reject(new TypeError('Old transport failure')); });
        if (requests === 2) throw new TypeError('Current transport failure');
      }
      return globalThis.__realFetch(input, options);
    };
  });
  const priorRevision = (await page.evaluate(() => window.biomaker.getServiceState())).revision;
  await page.evaluate(() => { window.__oldRequest = window.biomaker.request('/api/health').catch(error => error.message); });
  await waitUntil(() => application.evaluate(() => typeof globalThis.__rejectOldRequest === 'function'), 'Older request did not reach the native proxy');
  await assert.rejects(page.evaluate(() => window.biomaker.request('/api/health')), /Current transport failure/);
  await waitUntil(async () => (await page.evaluate(() => window.biomaker.getServiceState())).revision === priorRevision + 2, 'Newer request did not finish its health probe');
  await application.evaluate(() => { globalThis.__rejectOldRequest(); });
  assert.match(await page.evaluate(() => window.__oldRequest), /Old transport failure/);
  assert.deepEqual(await page.evaluate(() => window.biomaker.getServiceState()), { status: 'ready', revision: priorRevision + 2 });
  checks.push('late-request-failure-cannot-reopen-an-already-recovered-connection');

  const crashRevision = (await page.evaluate(() => window.biomaker.getServiceState())).revision;
  const crashedPID = await application.evaluate(() => {
    const child = globalThis.__qaSidecarChild;
    if (!child?.pid) throw new Error('Real sidecar process was not captured');
    child.kill('SIGKILL');
    return child.pid;
  });
  await waitUntil(async () => {
    const state = await page.evaluate(() => window.biomaker.getServiceState());
    return state.status === 'ready' && state.revision > crashRevision;
  }, 'Real sidecar did not recover after SIGKILL');
  assert.notEqual(await application.evaluate(() => globalThis.__qaSidecarChild.pid), crashedPID);
  assert.deepEqual(await page.evaluate(revision => window.__nativeStates.filter(state => state.revision > revision).map(state => state.status), crashRevision), ['reconnecting', 'ready']);
  assert.equal((await page.evaluate(() => window.biomaker.request('/api/health'))).status, 'ok');
  checks.push('real-sidecar-sigkill-pushes-reconnecting-and-authenticated-ready-after-replacement');

  await application.evaluate(() => {
    globalThis.__previewRequests = [];
    globalThis.fetch = async (input, options) => {
      if (new URL(String(input)).pathname.replace(/\/+$/, '') === '/api/desktop/history/preview') {
        globalThis.__previewRequests.push({ path: JSON.parse(options.body).path, authenticated: /^Bearer [a-f0-9]{64}$/.test(options.headers.Authorization) });
        return new Response(JSON.stringify({ path: JSON.parse(options.body).path }), { headers: { 'Content-Type': 'application/json' } });
      }
      return globalThis.__realFetch(input, options);
    };
  });
  await assert.rejects(preview(outside), /grant access|目录选择器/);
  await assert.rejects(preview(profile), /grant access|目录选择器/);
  await assert.rejects(preview(outside, '/api/desktop/history/%70review'), /grant access|目录选择器/);
  await assert.rejects(preview(outside, '/api/desktop/history/preview/'), /grant access|目录选择器/);
  await application.evaluate(({ dialog }, directory) => {
    dialog.showOpenDialog = async (...args) => { globalThis.__pickerOptions = args.at(-1); return { canceled: false, filePaths: [directory] }; };
  }, allowed);
  await assert.rejects(page.evaluate(() => window.biomaker.chooseDirectory({ purpose: 'arbitrary' })), /Invalid directory purpose/);
  const chosen = await page.evaluate(() => window.biomaker.chooseDirectory({ purpose: 'history' }));
  assert.equal(chosen, await realpath(allowed));
  assert.deepEqual(await application.evaluate(() => globalThis.__pickerOptions.properties), ['openDirectory']);
  assert.match(await application.evaluate(() => globalThis.__pickerOptions.title), /已有实验|existing experiment/);
  assert.equal((await preview(allowed)).path, await realpath(allowed));
  assert.equal((await preview(nested)).path, await realpath(nested));
  await assert.rejects(preview(path.join(allowed, 'escape')), /grant access|目录选择器/);
  assert.deepEqual(await application.evaluate(() => globalThis.__previewRequests.map(request => request.authenticated)), [true, true]);
  await rename(allowed, `${allowed}-original`);
  await symlink(outside, allowed, 'dir');
  await assert.rejects(preview(allowed), /grant access|目录选择器/);
  checks.push('history-picker-authorizes-only-real-selected-tree-and-rejects-symlink-or-encoded-path-bypasses');

  const unauthorized = await application.evaluate(async ({ BrowserWindow }, preload) => {
    const foreign = new BrowserWindow({ show: false, webPreferences: { preload, nodeIntegration: false, contextIsolation: true, sandbox: true } });
    try {
      await foreign.loadURL('about:blank');
      return await foreign.webContents.executeJavaScript("window.biomaker.getServiceState().then(() => 'unexpectedly allowed', error => error.message)");
    } finally { foreign.destroy(); }
  }, path.join(root, 'dist-electron/preload.js'));
  assert.match(unauthorized, /Untrusted application frame/);
  checks.push('lifecycle-query-keeps-main-frame-ipc-authorization');

  await page.evaluate(() => window.__unsubscribeNativeStates());
  const report = { passed: true, executable: 'development Electron with controlled startup health response', checks };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally {
  await application?.close();
  await rm(temporary, { recursive: true, force: true });
}
