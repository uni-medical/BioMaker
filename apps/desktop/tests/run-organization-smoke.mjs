// Real desktop metadata operations, with synthetic immutable history records.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'biomaker-organization-'));
const output = process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/run-organization');
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = { ...process.env, BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile };
delete env.ELECTRON_RUN_AS_NODE;
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
await mkdir(output, { recursive: true });
const logs = path.join(profile, 'workspace/logs');
for (const name of ['organization-a', 'organization-b']) {
  const directory = path.join(logs, name);
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, 'run_manifest.json'), JSON.stringify({
    data_dir: path.join(profile, 'dataset'), created_at: '2026-10-08T10:00:00Z',
    task_description: `Synthetic organization fixture ${name}; no model training.`,
    config: { agent_steps: 2, model_for_all_agents: 'qa-never-requested' },
  }));
  await writeFile(path.join(directory, 'events.jsonl'), JSON.stringify({ ts: '2026-10-08T10:01:00Z', event: 'run_finished', completed_steps: 1 }) + '\n');
  await writeFile(path.join(directory, 'report.md'), '# Synthetic organization report\n\nOriginal artifact, retained through display edits.\n');
}
async function artifactSnapshot() {
  const result = {};
  for (const name of ['organization-a', 'organization-b']) {
    for (const file of (await readdir(path.join(logs, name))).sort()) {
      const target = path.join(logs, name, file);
      result[`${name}/${file}`] = { sha256: createHash('sha256').update(await readFile(target)).digest('hex'), modifiedAt: (await stat(target)).mtimeMs };
    }
  }
  return result;
}
const originalArtifacts = await artifactSnapshot();
const checks = [];
const rendererErrors = [];
let app;
let page;
async function until(callback, reason, timeout = 60_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await callback()) return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(reason);
}
async function launch() {
  app = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await app.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  await until(async () => {
    try { return (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready'; }
    catch { return false; }
  }, 'Service did not become ready');
  await page.locator('.primary-nav').waitFor();
}
const api = (route, method = 'GET', body) => page.evaluate(({ route, method, body }) => window.biomaker.request(route, { method, body }), { route, method, body });
const metadata = id => api(`/api/desktop/run-metadata/${encodeURIComponent(id)}`);
async function allExperiments() { await page.locator('.primary-nav').getByRole('button', { name: /全部实验/ }).click(); }
async function openRun(name) {
  await allExperiments();
  await page.getByLabel('实验范围', { exact: true }).selectOption('current');
  await page.getByLabel('搜索全部实验', { exact: true }).fill('');
  await page.locator('.history-list').getByText(name, { exact: true }).click();
  await page.locator('.run-intro h1').waitFor();
}

try {
  await launch();
  const originalRuns = await api('/api/runs');
  assert.equal(originalRuns.length, 2);
  const record = originalRuns.find(run => run.actual_exp_name === 'organization-a');
  assert(record);
  assert.equal(record.read_only, true);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  assert.deepEqual(await metadata(record.id), { run_id: record.id, display_name: null, archived: false, revision: 0, updated_at: null });
  await openRun('organization-a');

  // Hold an old list response while a newer native-bridge mutation completes.
  await app.evaluate(() => {
    const original = globalThis.fetch;
    globalThis.__organizationOriginalFetch = original;
    let held = false;
    globalThis.fetch = async (input, options) => {
      if (!held && String(input).endsWith('/api/desktop/run-metadata') && (!options?.method || options.method === 'GET')) {
        held = true;
        const response = await original(input, options);
        const text = await response.text();
        await new Promise(resolve => { globalThis.__releaseOrganizationList = resolve; });
        globalThis.__organizationListReleased = true;
        return new Response(text, { status: response.status, headers: response.headers });
      }
      return original(input, options);
    };
  });
  await page.locator('.app-header').getByRole('button', { name: '刷新', exact: true }).click();
  await until(() => app.evaluate(() => typeof globalThis.__releaseOrganizationList === 'function'), 'Old metadata list was not held');
  await page.getByRole('button', { name: '重命名', exact: true }).click();
  await page.getByLabel('显示名称', { exact: true }).fill('中文基线 · 第一轮');
  await page.getByRole('button', { name: '保存名称', exact: true }).click();
  await page.getByRole('dialog', { name: '重命名实验' }).waitFor({ state: 'hidden' });
  assert.equal(await page.locator('.run-intro h1').innerText(), '中文基线 · 第一轮');
  await app.evaluate(() => { globalThis.fetch = globalThis.__organizationOriginalFetch; globalThis.__releaseOrganizationList(); });
  await until(() => app.evaluate(() => globalThis.__organizationListReleased === true), 'Delayed metadata list was not released');
  await page.waitForTimeout(250);
  assert.equal(await page.locator('.run-intro h1').innerText(), '中文基线 · 第一轮');
  assert.equal((await metadata(record.id)).revision, 1);
  await assert.rejects(api(`/api/desktop/run-metadata/${record.id}`, 'PUT', { expected_revision: 0, display_name: 'stale overwrite' }));
  assert.equal((await metadata(record.id)).display_name, '中文基线 · 第一轮');
  checks.push('unicode-display-name-and-stale-response-protection', 'server-rejects-lost-update');

  // A genuine concurrent metadata edit must preserve the user's unsaved input.
  await page.getByRole('button', { name: '重命名', exact: true }).click();
  await page.getByLabel('显示名称', { exact: true }).fill('我的实验名称');
  await api(`/api/desktop/run-metadata/${record.id}`, 'PUT', { expected_revision: 1, display_name: '另一处更新的名称' });
  await page.getByRole('button', { name: '保存名称', exact: true }).click();
  await page.getByText(/此实验已在其他位置更新/).waitFor();
  assert.equal(await page.getByLabel('显示名称', { exact: true }).inputValue(), '我的实验名称');
  assert.equal((await metadata(record.id)).display_name, '另一处更新的名称');
  await page.screenshot({ path: path.join(output, '01-conflict-recovery.png') });
  await page.getByRole('button', { name: '保存名称', exact: true }).click();
  await page.getByRole('dialog', { name: '重命名实验' }).waitFor({ state: 'hidden' });
  assert.equal((await metadata(record.id)).display_name, '我的实验名称');
  checks.push('edit-conflict-preserves-input-and-explicit-retry');

  await allExperiments();
  await page.getByLabel('搜索全部实验', { exact: true }).fill('organization-a');
  await page.locator('.history-list').getByText('我的实验名称', { exact: true }).waitFor();
  await page.locator('.history-list').getByText('我的实验名称', { exact: true }).click();
  await page.getByRole('button', { name: '归档实验', exact: true }).click();
  await until(async () => (await metadata(record.id)).archived, 'Archive did not persist');
  await page.getByRole('button', { name: '恢复实验', exact: true }).waitFor();
  assert.equal(await page.locator('.sidebar-runs').getByText('我的实验名称', { exact: true }).count(), 0);
  await allExperiments();
  await page.getByLabel('搜索全部实验', { exact: true }).fill('');
  assert.equal(await page.locator('.history-list').getByText('我的实验名称', { exact: true }).count(), 0);
  await page.getByLabel('实验范围', { exact: true }).selectOption('archived');
  await page.locator('.history-list').getByText('我的实验名称', { exact: false }).waitFor();
  await page.screenshot({ path: path.join(output, '02-archived-history.png') });
  assert.deepEqual(await api('/api/runs'), originalRuns, 'Organization must not change the experiment ledger/API');
  assert.deepEqual(await artifactSnapshot(), originalArtifacts);
  checks.push('search-keeps-original-name', 'archive-hides-only-current-view-with-original-records-intact');

  await app.close(); app = undefined;
  await launch();
  const persisted = await metadata(record.id);
  assert.equal(persisted.display_name, '我的实验名称'); assert.equal(persisted.archived, true);
  await allExperiments();
  await page.getByLabel('实验范围', { exact: true }).selectOption('archived');
  await page.locator('.history-list').getByText('我的实验名称', { exact: false }).click();
  await page.getByRole('button', { name: '恢复实验', exact: true }).click();
  await until(async () => !(await metadata(record.id)).archived, 'Restore did not persist');
  await page.locator('.sidebar-runs').getByText('我的实验名称', { exact: true }).waitFor();
  await page.getByRole('tab', { name: /^文件/ }).click();
  const destination = path.join(profile, 'exported-original-report.md');
  await app.evaluate(({ dialog }, filePath) => { dialog.showSaveDialog = async () => ({ canceled: false, filePath }); }, destination);
  await page.getByRole('button', { name: '保存 report.md', exact: true }).click();
  await until(async () => { try { return (await readFile(destination, 'utf8')).includes('Original artifact'); } catch { return false; } }, 'Original report export failed after restoring');
  assert.equal(await readFile(destination, 'utf8'), await readFile(path.join(logs, 'organization-a/report.md'), 'utf8'));
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.screenshot({ path: path.join(output, '03-restored-compact.png') });
  assert.deepEqual(await artifactSnapshot(), originalArtifacts);
  assert.deepEqual(await api('/api/runs'), originalRuns);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  assert.deepEqual(rendererErrors, []);
  checks.push('restart-retains-organization', 'restore-and-native-export-preserve-original-files', 'compact-layout-no-engine-install-no-renderer-errors');
  const report = { passed: true, executablePath: executablePath || 'development Electron', profile, checks, rendererErrors };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-6500));
  }
  throw error;
} finally {
  if (app) {
    await app.evaluate(({ BrowserWindow }) => { for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await app.close();
  }
}
