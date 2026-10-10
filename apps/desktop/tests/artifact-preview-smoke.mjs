// Real native preview/export with isolated synthetic files; no model or engine.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from 'playwright';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'biomaker-artifact-preview-'));
const output = process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/artifact-preview');
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = { ...process.env, BIOMAKER_DESKTOP_HOME: profile, BIOMAKER_DESKTOP_USER_DATA: profile };
delete env.ELECTRON_RUN_AS_NODE;
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
await mkdir(output, { recursive: true });
const original = Buffer.from('\ufeff# 原始报告 α\r\n\r\nSynthetic acceptance only.\r\n');
const changed = Buffer.from(original.toString('utf8').replace('α', 'β'));
const logs = path.join(profile, 'workspace/logs');
for (const name of ['preview-a', 'preview-b']) {
  const directory = path.join(logs, name);
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, 'run_manifest.json'), JSON.stringify({ data_dir: path.join(profile, 'synthetic-dataset'), created_at: '2026-10-09T04:00:00Z', task_description: 'Synthetic artifact fixture, no training.', config: { agent_steps: 1, final_retrain_enabled: false } }));
  await writeFile(path.join(directory, 'events.jsonl'), JSON.stringify({ event: 'run_finished', completed_steps: 0 }) + '\n');
  await writeFile(path.join(directory, 'report.md'), name === 'preview-a' ? original : '# Synthetic second experiment\n');
  await writeFile(path.join(directory, 'raw.json'), '{"value": NaN, "other": Infinity}\r\n');
  await writeFile(path.join(directory, 'large.txt'), Buffer.alloc(500_001, 65));
  await writeFile(path.join(directory, 'invalid.txt'), Buffer.from([255, 254, 65]));
}
const checks = [], rendererErrors = [], restarts = [];
let app, page;
async function until(test, message, timeout = 60_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) { if (await test()) return; await new Promise(resolve => setTimeout(resolve, 75)); }
  throw new Error(message);
}
const state = () => page.evaluate(() => window.biomaker.getServiceState());
const api = route => page.evaluate(route => window.biomaker.request(route), route);
const preview = (runId, name) => page.evaluate(input => window.biomaker.previewArtifact(input), { runId, name });
async function openRun(name) {
  await page.locator('.primary-nav').getByRole('button', { name: /全部实验/ }).click();
  await page.locator('.history-list').getByText(name, { exact: true }).click();
  await page.locator('.run-intro h1').filter({ hasText: name }).waitFor();
  await page.getByRole('tab', { name: /^文件/ }).click();
}
const reportRow = () => page.locator('.artifact-row').filter({ has: page.getByText('report.md', { exact: true }) });
const modal = () => page.getByRole('dialog', { name: 'report.md', exact: true });
async function showReport() { await reportRow().getByRole('button', { name: '预览', exact: true }).click(); await modal().waitFor(); }
async function destination(file, hold = false) {
  await app.evaluate(({ dialog }, { file, hold }) => {
    dialog.showSaveDialog = async () => {
      globalThis.__artifactQA.dialogs++;
      if (hold) await new Promise(resolve => { globalThis.__artifactQA.releaseDialog = resolve; });
      return { canceled: false, filePath: file };
    };
  }, { file, hold });
}
async function nativeRefresh() {
  await app.evaluate(({ Menu }) => {
    const item = Menu.getApplicationMenu().items.flatMap(item => item.submenu?.items || []).find(item => item.label === '刷新数据');
    if (!item) throw new Error('Refresh menu not found');
    item.click();
  });
}
async function holdPreview() { await app.evaluate(() => { globalThis.__artifactQA.holdNext = true; globalThis.__artifactQA.releasePreview = null; }); }
async function heldPreview() { await until(() => app.evaluate(() => typeof globalThis.__artifactQA.releasePreview === 'function'), 'Preview response not held'); }
async function releasePreview() { await app.evaluate(() => globalThis.__artifactQA.releasePreview()); }
async function restart() {
  const before = await state();
  const runtime = (await readdir(path.join(profile, 'runtimes'))).filter(name => !name.includes('.staging-'));
  assert.equal(runtime.length, 1);
  const service = path.join(profile, 'runtimes', runtime[0], 'service/main.py');
  const processes = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes(service) && line.includes('python'));
  assert.equal(processes.length, 1);
  process.kill(Number(processes[0].trim().split(/\s+/)[0]), 'SIGKILL');
  await until(async () => { const next = await state(); return next.status === 'ready' && next.revision > before.revision; }, 'Service did not recover');
  restarts.push({ before: before.revision, after: (await state()).revision });
}

try {
  app = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await app.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  await until(async () => { try { return (await state()).status === 'ready'; } catch { return false; } }, 'Service did not start');
  await app.evaluate(() => {
    const originalFetch = globalThis.fetch;
    globalThis.__artifactQA = { requests: [], forbidden: [], dialogs: 0, holdNext: false };
    globalThis.fetch = async (input, options) => {
      const url = new URL(String(input));
      if (url.hostname !== '127.0.0.1' || (options?.method || 'GET') !== 'GET') {
        globalThis.__artifactQA.forbidden.push(url.pathname);
        throw new Error('QA forbids nonlocal or mutating requests');
      }
      globalThis.__artifactQA.requests.push(url.pathname);
      const response = await originalFetch(input, options);
      if (url.pathname.endsWith('/preview') && globalThis.__artifactQA.holdNext) {
        globalThis.__artifactQA.holdNext = false;
        const body = await response.text();
        await new Promise(resolve => { globalThis.__artifactQA.releasePreview = resolve; });
        return new Response(body, { status: response.status, headers: response.headers });
      }
      if (url.pathname.startsWith('/artifacts/runs/') && globalThis.__artifactQA.holdDownload) {
        globalThis.__artifactQA.holdDownload = false;
        const body = await response.arrayBuffer();
        await new Promise(resolve => { globalThis.__artifactQA.releaseDownload = resolve; });
        return new Response(body, { status: response.status, headers: response.headers });
      }
      return response;
    };
  });
  const records = await api('/api/runs');
  assert.equal(records.length, 2);
  const a = records.find(run => run.actual_exp_name === 'preview-a');
  assert(a);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  const first = await preview(a.id, 'report.md');
  assert.equal(first.content, original.toString('utf8'));
  assert.equal(first.sha256, createHash('sha256').update(original).digest('hex'));
  assert.equal(first.size, original.length);
  assert.equal((await preview(a.id, 'raw.json')).content, '{"value": NaN, "other": Infinity}\r\n');
  await assert.rejects(preview(a.id, 'large.txt'), /\[artifact:too_large\]/);
  await assert.rejects(preview(a.id, 'invalid.txt'), /\[artifact:invalid_text\]/);
  await assert.rejects(preview(a.id, '../config.env'), /\[artifact:invalid_request\]/);
  checks.push('real-native-preview-preserves-bom-crlf-unicode-and-nonfinite-json', 'preview-size-text-and-path-refusals');

  await openRun('preview-a');
  await showReport();
  await modal().getByText('原始报告 α', { exact: true }).waitFor();
  const saved = path.join(profile, 'saved-report.md');
  await destination(saved);
  await modal().getByRole('button', { name: '保存文件', exact: true }).click();
  await until(async () => { try { return (await readFile(saved)).equals(original); } catch { return false; } }, 'Preview save bytes differ');
  checks.push('preview-save-writes-exact-displayed-bytes');

  // Hold a real download after its bytes arrive, then navigate while it is
  // still saving. A second component must not open a competing save dialog.
  await writeFile(saved, 'Existing destination before overlapping exports');
  await app.evaluate(() => { globalThis.__artifactQA.holdDownload = true; });
  await modal().getByRole('button', { name: '保存文件', exact: true }).click();
  await until(() => app.evaluate(() => typeof globalThis.__artifactQA.releaseDownload === 'function'), 'Artifact download not held');
  await page.keyboard.press('Escape');
  assert.equal(await reportRow().getByRole('button', { name: '保存 report.md', exact: true }).isEnabled(), false);
  await openRun('preview-b');
  const beforeOverlap = await app.evaluate(() => ({ dialogs: globalThis.__artifactQA.dialogs, downloads: globalThis.__artifactQA.requests.filter(route => route.startsWith('/artifacts/runs/')).length }));
  await reportRow().getByRole('button', { name: '保存 report.md', exact: true }).click();
  await page.getByText('已有文件正在导出，请完成后再保存。', { exact: true }).waitFor();
  const afterOverlap = await app.evaluate(() => ({ dialogs: globalThis.__artifactQA.dialogs, downloads: globalThis.__artifactQA.requests.filter(route => route.startsWith('/artifacts/runs/')).length }));
  assert.deepEqual(afterOverlap, beforeOverlap);
  await app.evaluate(() => globalThis.__artifactQA.releaseDownload());
  await until(async () => (await readFile(saved)).equals(original), 'First export changed bytes');
  await openRun('preview-a');
  await showReport();
  checks.push('closing-preview-and-cross-run-save-cannot-overlap-or-open-second-dialog');

  // Keep size and mtime identical to defeat the list's inexpensive revision.
  execFileSync('/usr/bin/python3', ['-B', '-c', 'import os,pathlib,sys;p=pathlib.Path(sys.argv[1]);s=p.stat();p.write_bytes(p.read_bytes().replace("α".encode(),"β".encode()));os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns))', path.join(logs, 'preview-a/report.md')]);
  await modal().getByRole('button', { name: '保存文件', exact: true }).click();
  await modal().getByText(/文件已更新/).waitFor();
  assert((await readFile(saved)).equals(original));
  await page.screenshot({ path: path.join(output, '01-same-stat-changed.png') });
  await modal().getByRole('button', { name: '重新预览', exact: true }).click();
  await modal().getByText('原始报告 β', { exact: true }).waitFor();
  await modal().getByRole('button', { name: '保存文件', exact: true }).click();
  await until(async () => (await readFile(saved)).equals(changed), 'Updated preview did not save new exact bytes');
  checks.push('same-size-same-mtime-change-preserves-destination-and-explicit-repreview');

  await writeFile(path.join(logs, 'preview-a/report.md'), '# New file revision\n');
  await nativeRefresh();
  await modal().getByText(/当前仍显示上次预览/).waitFor();
  assert.equal(await modal().getByRole('button', { name: '保存文件', exact: true }).isEnabled(), false);
  assert.equal(await modal().getByText('原始报告 β', { exact: true }).count(), 1);
  await modal().getByRole('button', { name: '重新预览', exact: true }).click();
  await modal().getByText('New file revision', { exact: true }).waitFor();
  checks.push('listed-update-marks-preview-stale-with-content-retained');

  await page.keyboard.press('Escape');
  await holdPreview();
  await reportRow().getByRole('button', { name: '预览', exact: true }).click();
  await heldPreview();
  await openRun('preview-b');
  await openRun('preview-a');
  await releasePreview();
  await page.waitForTimeout(150);
  assert.equal(await page.getByRole('dialog').count(), 0);
  checks.push('late-a-to-b-to-a-preview-cannot-open-in-new-view');

  await showReport();
  await destination(saved, true);
  await modal().getByRole('button', { name: '保存文件', exact: true }).click();
  await until(() => app.evaluate(() => typeof globalThis.__artifactQA.releaseDialog === 'function'), 'Save dialog was not held');
  await restart();
  await app.evaluate(() => globalThis.__artifactQA.releaseDialog());
  await modal().getByText('本地服务已变化。请重新预览后再保存；目标文件未被替换。', { exact: true }).waitFor();
  assert((await readFile(saved)).equals(changed));
  assert.equal(await modal().getByRole('button', { name: '保存文件', exact: true }).isEnabled(), false);
  await page.screenshot({ path: path.join(output, '02-service-change.png') });
  checks.push('real-service-restart-during-save-dialog-preserves-destination');

  await modal().getByRole('button', { name: '重新预览', exact: true }).click();
  await modal().getByText('New file revision', { exact: true }).waitFor();
  await page.keyboard.press('Escape');
  await holdPreview();
  await reportRow().getByRole('button', { name: '预览', exact: true }).click();
  await heldPreview();
  await restart();
  await releasePreview();
  await page.waitForTimeout(150);
  assert.equal(await page.getByRole('dialog').count(), 0);
  checks.push('late-preview-from-replaced-service-is-discarded');

  await page.locator('.sidebar-settings').click();
  await page.getByRole('button', { name: '通用', exact: true }).click();
  await page.getByRole('button', { name: '深色', exact: true }).click();
  await page.waitForFunction(() => document.documentElement.dataset.theme === 'dark');
  await page.keyboard.press('Escape');
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  await openRun('preview-a');
  await showReport();
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  await page.waitForFunction(() => innerWidth <= 960 && innerHeight <= 680);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({ path: path.join(output, '03-compact-dark-preview.png') });
  await page.keyboard.press('Escape');
  const largeSaved = path.join(profile, 'large-export.txt');
  await destination(largeSaved);
  await page.getByRole('button', { name: '保存 large.txt', exact: true }).click();
  await until(async () => { try { return (await readFile(largeSaved)).length === 500_001; } catch { return false; } }, 'Direct export failed');
  assert.equal((await readdir(profile)).filter(name => name.endsWith('.part')).length, 0);
  const native = await app.evaluate(() => ({ requests: globalThis.__artifactQA.requests, forbidden: globalThis.__artifactQA.forbidden, dialogs: globalThis.__artifactQA.dialogs }));
  assert.deepEqual(native.forbidden, []);
  assert.deepEqual(rendererErrors, []);
  assert.equal((await api('/api/runs')).length, 2);
  assert.deepEqual((await api('/api/desktop/status')).configuration, { api_key_configured: false, model_configured: false });
  checks.push('compact-dark-preview-and-large-direct-export', 'no-forbidden-requests-renderer-errors-or-temp-files');
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify({ passed: true, executablePath: executablePath || 'development Electron', profile, checks, restarts, native, rendererErrors }, null, 2) + '\n');
  console.log(JSON.stringify({ passed: true, checks, profile, output }, null, 2));
} catch (error) {
  if (page) await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
  await writeFile(path.join(output, 'failure.json'), JSON.stringify({ passed: false, profile, checks, error: String(error), rendererErrors }, null, 2) + '\n');
  throw error;
} finally { if (app) await app.close(); }
