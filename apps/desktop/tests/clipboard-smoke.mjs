// Production native clipboard IPC, real Electron and sidecar, isolated history.
// No clipboard, permission, IPC, provider, or lifecycle mocking. Previous native
// clipboard formats stay only in main-process memory and are restored on exit.
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, realpath, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'biomaker-native-clipboard-'));
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/native-clipboard'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1', NO_PROXY: '127.0.0.1,localhost' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const runId = 'qa-native-clipboard';
const history = path.join(profile, 'workspace/logs', runId);
const report = '\ufeff# 真实原生复制测试 α\r\n\r\n| 指标 | 分数 |\r\n| :--- | ---: |\r\n| AUROC | 0.875 |\r\n| 中文 ✓ | 1.0 |\r\n\r\n<script>window.__unsafeReport = true</script>\r\n<img src=x onerror="window.__unsafeReport = true">\r\n[unsafe](javascript:alert(1))\r\n';
const log = 'Synthetic log only — no training executed.\nUnicode: 皮肤病理 α\n';
await mkdir(history, { recursive: true });
await mkdir(output, { recursive: true });
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
await writeFile(path.join(history, 'run_manifest.json'), JSON.stringify({ created_at: '2026-10-09T10:00:00Z', data_dir: path.join(profile, 'synthetic-dataset'), task_description: 'Synthetic native clipboard acceptance; never execute.', config: { agent_steps: 1, final_retrain_enabled: false } }));
await writeFile(path.join(history, 'events.jsonl'), JSON.stringify({ event: 'run_finished', completed_steps: 0 }) + '\n');
await writeFile(path.join(history, 'report.md'), report);
await writeFile(path.join(history, `${runId}.log`), log);
const checks = [], rendererErrors = [], requests = [], forbiddenRequests = [];
let application, page, failure, clipboardRestored = false;
async function until(test, message, timeout = 30_000) {
  const end = Date.now() + timeout;
  while (!(await test())) { if (Date.now() > end) throw new Error(message); await new Promise(resolve => setTimeout(resolve, 50)); }
}
const nativeText = () => application.evaluate(({ clipboard }) => clipboard.readText());
const writeNative = text => page.evaluate(text => window.biomaker.writeClipboardText(text), text);
try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  await application.evaluate(async ({ clipboard, ClipboardItem }) => {
    const saved = [];
    for (const item of await clipboard.read()) {
      const types = {};
      for (const type of item.types) types[type] = await item.getType(type);
      saved.push(new ClipboardItem(types));
    }
    globalThis.__nativeClipboardQA = { saved, requests: [], forbidden: [] };
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (input, options) => {
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      if (url.hostname !== '127.0.0.1' || method !== 'GET') {
        globalThis.__nativeClipboardQA.forbidden.push({ path: url.pathname, method });
        throw new Error('Clipboard QA forbids engine, provider and run mutations');
      }
      globalThis.__nativeClipboardQA.requests.push(url.pathname);
      return originalFetch(input, options);
    };
  });
  await until(async () => { try { return (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready'; } catch { return false; } }, 'Real sidecar did not start', 60_000);
  await page.locator('.primary-nav').getByRole('button', { name: /全部实验/ }).click();
  await page.locator('.history-list').getByText(runId, { exact: true }).click();
  await page.locator('.run-intro h1').filter({ hasText: runId }).waitFor();
  await page.getByRole('tab', { name: /^文件/ }).click();
  await page.locator('.artifact-row').filter({ has: page.getByText('report.md', { exact: true }) }).getByRole('button', { name: '预览', exact: true }).click();
  const modal = page.getByRole('dialog', { name: 'report.md', exact: true });
  await modal.waitFor();
  assert.deepEqual(await modal.locator('table th').allTextContents(), ['指标', '分数']);
  assert.deepEqual(await modal.locator('table td').allTextContents(), ['AUROC', '0.875', '中文 ✓', '1.0']);
  assert.equal(await modal.locator('script,img').count(), 0);
  assert.equal(await page.evaluate(() => Boolean(window.__unsafeReport)), false);
  assert.equal(await modal.locator('a[href^="javascript:"]').count(), 0);
  await writeNative('Synthetic pre-copy sentinel');
  assert.equal(await nativeText(), 'Synthetic pre-copy sentinel');
  await modal.getByRole('button', { name: '复制内容', exact: true }).click();
  await until(async () => await nativeText() !== 'Synthetic pre-copy sentinel', 'Native clipboard write did not complete');
  // The native text clipboard omits the leading encoding BOM; file export
  // tests separately cover byte-exact BOM preservation. All actual text and
  // CRLF line endings must remain unchanged.
  assert.equal(await nativeText(), report.replace(/^\ufeff/, ''));
  checks.push('real-preview-click-writes-unicode-crlf-report-text-to-system-clipboard', 'gfm-table-renders-real-semantic-cells-and-unsafe-html-does-not-execute');
  await page.screenshot({ path: path.join(output, '01-table-preview-real-copy.png') });
  await page.keyboard.press('Escape');
  await page.locator('#run-tab-log').click();
  const reader = page.locator('.run-log-reader');
  await reader.getByRole('button', { name: '复制当前片段', exact: true }).waitFor();
  await until(() => reader.locator('.run-log-text').innerText().then(value => value.includes('皮肤病理')), 'Real log did not load');
  const displayedLog = await reader.locator('.run-log-text').innerText();
  await reader.getByRole('button', { name: '复制当前片段', exact: true }).click();
  await until(async () => await nativeText() === displayedLog, 'Log copy differs from displayed window');
  checks.push('real-log-copy-click-writes-exact-displayed-window');

  const sentinel = 'Synthetic clipboard validation sentinel';
  await writeNative(sentinel);
  for (const invalid of [null, 42, { text: 'not text' }, '界'.repeat(1_398_102)]) {
    await assert.rejects(writeNative(invalid), /Clipboard text must be a string no larger than 4 MiB/);
    assert.equal(await nativeText(), sentinel);
  }
  const denied = await page.evaluate(async () => {
    try { await navigator.clipboard.writeText('must remain denied'); return false; } catch { return true; }
  });
  assert.equal(denied, true, 'Browser clipboard permission was unexpectedly granted');
  assert.equal(await nativeText(), sentinel);
  assert.equal(await page.evaluate(() => Object.keys(window.biomaker).some(key => /read.*clipboard|clipboard.*read/i.test(key))), false);
  const foreign = await application.evaluate(async ({ app, BrowserWindow }) => {
    const nodePath = process.getBuiltinModule('node:path');
    const window = new BrowserWindow({ show: false, webPreferences: { preload: nodePath.join(app.getAppPath(), 'dist-electron/preload.js'), nodeIntegration: false, contextIsolation: true, sandbox: true } });
    try { await window.loadURL('about:blank'); return await window.webContents.executeJavaScript("window.biomaker.writeClipboardText('foreign must not write').then(() => 'unexpected success', error => error.message)"); }
    finally { window.destroy(); }
  });
  assert.match(foreign, /Untrusted application frame/);
  assert.equal(await nativeText(), sentinel);
  checks.push('invalid-and-oversized-ipc-writes-preserve-clipboard', 'renderer-permission-and-foreign-frame-remain-denied-with-write-only-bridge');

  await page.locator('.primary-nav').getByRole('button', { name: /新建实验/ }).click();
  const task = page.getByLabel('实验任务', { exact: true });
  await task.waitFor();
  const snapshot = await page.evaluate(() => window.biomaker.getDrafts());
  assert(snapshot.activeId);
  await page.evaluate(input => window.biomaker.mutateDrafts(input), { expectedRevision: snapshot.revision, mutationId: randomUUID(), operation: { type: 'rename', id: snapshot.activeId, title: 'Synthetic concurrent writer' } });
  const unsaved = '仅用于草稿恢复复制的未保存内容 α';
  await task.fill(unsaved);
  await page.getByText('本机保存的草稿已变化，当前编辑仍保留。可以将当前内容另存为新草稿。', { exact: true }).waitFor();
  assert.equal(await page.getByTestId('draft-save-state').innerText(), '草稿保存异常');
  await page.getByRole('button', { name: '复制草稿内容', exact: true }).click();
  await until(async () => { try { return JSON.parse(await nativeText()).task === unsaved; } catch { return false; } }, 'Draft recovery copy did not reach the native clipboard');
  assert.equal(await task.inputValue(), unsaved);
  await page.screenshot({ path: path.join(output, '02-unsaved-draft-real-copy.png') });
  checks.push('real-draft-conflict-retains-edits-and-recovery-copy-writes-valid-json');
  assert.equal((await page.evaluate(() => window.biomaker.request('/api/desktop/status'))).engine.ready, false);
  assert.deepEqual(rendererErrors, []);
} catch (error) {
  failure = error;
  if (page && !page.isClosed()) await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
} finally {
  if (application) {
    const trace = await application.evaluate(async ({ clipboard }) => {
      const qa = globalThis.__nativeClipboardQA;
      if (!qa) return null;
      if (qa.saved.length) await clipboard.write(qa.saved); else clipboard.clear();
      const after = await clipboard.read();
      let restored = after.length === qa.saved.length;
      for (let index = 0; index < qa.saved.length; index++) {
        const old = qa.saved[index], current = after[index];
        for (const type of old.types) {
          if (!current?.types.includes(type)) { restored = false; continue; }
          const beforeData = await old.getType(type), afterData = await current.getType(type);
          restored &&= type === 'electron application/bookmark' ? JSON.stringify(beforeData) === JSON.stringify(afterData) : Buffer.from(await beforeData.arrayBuffer()).equals(Buffer.from(await afterData.arrayBuffer()));
        }
      }
      const result = { restored, requests: qa.requests, forbidden: qa.forbidden };
      delete globalThis.__nativeClipboardQA;
      return result;
    }).catch(error => { failure ||= error; return null; });
    if (trace) { clipboardRestored = trace.restored; requests.push(...trace.requests); forbiddenRequests.push(...trace.forbidden); }
    await application.evaluate(({ BrowserWindow }) => { for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await application.close().catch(error => { failure ||= error; });
  }
}
if (!clipboardRestored) failure ||= new Error('Original system clipboard could not be confirmed restored');
if (forbiddenRequests.length) failure ||= new Error('Forbidden request attempted');
const canonicalProfile = await realpath(profile);
let remainingProcesses = [];
await until(() => { remainingProcesses = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes(profile) || line.includes(canonicalProfile)); return remainingProcesses.length === 0; }, 'Isolated QA process remains', 15_000).catch(error => { failure ||= error; });
await writeFile(path.join(output, failure ? 'failure.json' : 'acceptance.json'), JSON.stringify({ passed: !failure, scope: 'Actual native clipboard writes and production IPC; isolated synthetic records and real sidecar only, no engine or provider execution.', executablePath: executablePath || 'development Electron', profile, checks, clipboardRestored, clipboardTextNormalization: 'Electron native text clipboard omits the leading UTF-8 encoding BOM; Unicode content and CRLF line endings are unchanged. File export verifies original bytes separately.', requests, forbiddenRequests, rendererErrors, remainingProcesses, ...(failure ? { error: String(failure), stack: failure.stack } : {}) }, null, 2) + '\n');
if (failure) throw failure;
console.log(JSON.stringify({ passed: true, checks, clipboardRestored, output }, null, 2));
