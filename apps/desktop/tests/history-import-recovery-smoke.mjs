// Isolated renderer recovery checks with synthetic job receipts. Real copying,
// candidate viewing, and source integrity are covered by history-import-smoke.
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'biomaker-import-recovery-'));
const output = path.join(root, 'output/history-import-recovery');
await mkdir(output, { recursive: true });
const env = { ...process.env, BIOMAKER_DESKTOP_HOME: profile, BIOMAKER_DESKTOP_USER_DATA: profile };
delete env.ELECTRON_RUN_AS_NODE;
const jobId = 'e'.repeat(32);
let application;
let page;
const errors = [];
const checks = [];
async function until(predicate, message) {
  const deadline = Date.now() + 20_000;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}
async function open() { await page.locator('.sidebar-empty').getByRole('button', { name: '导入已有实验', exact: true }).click(); }
try {
  application = await electron.launch({ args: [root], env, timeout: 60_000 });
  page = await application.firstWindow();
  page.on('pageerror', error => errors.push(error.message));
  await until(async () => (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready', 'Service did not become ready');
  await page.locator('.onboarding').waitFor();
  await application.evaluate(({ dialog }, directory) => {
    const original = globalThis.fetch;
    globalThis.__receiptMode = 'running';
    globalThis.__submitCount = 0;
    globalThis.fetch = async (input, options) => {
      const pathname = new URL(String(input)).pathname;
      if (pathname.startsWith('/api/desktop/history/imports/')) {
        if (globalThis.__receiptMode === 'failure') return new Response(JSON.stringify({ detail: 'Synthetic unknown receipt failure' }), { status: 503, headers: { 'Content-Type': 'application/json' } });
        if (globalThis.__receiptMode === 'missing') return new Response(JSON.stringify({ detail: 'Import job was not found. Preview the source again.' }), { status: 404, headers: { 'Content-Type': 'application/json' } });
        const interrupted = globalThis.__receiptMode === 'interrupted';
        return new Response(JSON.stringify({ job_id: 'e'.repeat(32), status: interrupted ? 'failed' : 'running', completed_runs: 1, total_runs: 3, current_name: null, imported: [{ source_id: 'source-one', run_id: 'receipt-fixture', name: '已保存的合成记录' }], skipped: [], errors: interrupted ? [{ message: 'Import was interrupted when the desktop service stopped. Completed copies are preserved; preview again to continue.' }] : [] }), { headers: { 'Content-Type': 'application/json' } });
      }
      if (pathname === '/api/desktop/history/preview') return new Response(JSON.stringify({ preview_id: 'expired-preview', source_path: directory, source_kind: 'run', expires_at: '2000-01-01T00:00:00Z', limits: { max_runs: 100, max_files_per_run: 128, max_file_bytes: 67108864, max_batch_bytes: 268435456 }, warnings: [], runs: [{ source_id: 'expired-source', name: '预览过期样例', created_at: null, status: 'snapshot', total_bytes: 100, files: [{ name: 'run_manifest.json', size: 100 }], capabilities: { candidates: false, code: false, report: true, results: false, logs: false }, already_imported: false, existing_run_id: null, eligible: true, warnings: ['No completion event was observed. This is a static snapshot; the original experiment may still be running.'] }] }), { headers: { 'Content-Type': 'application/json' } });
      if (pathname === '/api/desktop/history/import') globalThis.__submitCount++;
      return original(input, options);
    };
    dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [directory] });
  }, profile);
  await page.evaluate(id => localStorage.setItem('biomaker.history-import-job', JSON.stringify(id)), jobId);
  await open();
  await page.getByText('正在导入实验记录…', { exact: true }).waitFor();
  await application.evaluate(() => { globalThis.__receiptMode = 'failure'; });
  await page.getByText('Synthetic unknown receipt failure', { exact: false }).waitFor();
  assert.equal(await page.getByRole('button', { name: '重新扫描', exact: true }).count(), 0, 'Temporary failures must not offer a reset that loses the running job');
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('biomaker.history-import-job'))), jobId);
  await page.getByRole('button', { name: '稍后查看', exact: true }).click();
  await application.evaluate(() => { globalThis.__receiptMode = 'interrupted'; });
  await open();
  await page.getByText('导入未完成', { exact: true }).waitFor();
  assert.match(await page.locator('.import-result-summary').innerText(), /已导入 1 个 · 跳过 0 个 · 未完成 2 个/);
  assert.match(await page.locator('.history-import-body').innerText(), /本地服务停止导致导入中断/);
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('biomaker.history-import-job'))), null);
  await page.screenshot({ path: path.join(output, 'interrupted.png') });
  checks.push('temporary-poll-failure-keeps-job-for-reopen-and-unknown-detail-is-preserved');
  checks.push('global-interruption-counts-all-unimported-runs-and-localizes-known-diagnostic');
  await page.getByRole('button', { name: '完成', exact: true }).click();
  await application.evaluate(() => { globalThis.__receiptMode = 'missing'; });
  await page.evaluate(id => localStorage.setItem('biomaker.history-import-job', JSON.stringify(id)), jobId);
  await open();
  await page.getByRole('button', { name: '重新扫描', exact: true }).waitFor();
  await page.getByRole('button', { name: '重新扫描', exact: true }).click();
  assert.equal(await page.evaluate(() => JSON.parse(localStorage.getItem('biomaker.history-import-job'))), null);
  await page.getByRole('button', { name: '选择实验目录', exact: true }).click();
  await page.getByLabel('选择 预览过期样例', { exact: true }).waitFor();
  assert.match(await page.locator('.import-warning').innerText(), /未找到完成事件/);
  await page.getByRole('button', { name: '导入所选实验', exact: true }).click();
  await page.getByText('预览已失效，或源文件发生了变化。请重新扫描后再导入。', { exact: true }).waitFor();
  assert.equal(await application.evaluate(() => globalThis.__submitCount), 0, 'Expired previews must not be submitted');
  assert.equal(await page.getByRole('button', { name: '导入所选实验', exact: true }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '重新扫描', exact: true }).isEnabled(), true);
  checks.push('confirmed-missing-job-can-reset-and-expired-preview-requires-rescan-without-post');
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
