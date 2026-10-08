// Real Electron / sidecar acceptance. All source histories and profiles are synthetic.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { lstat, mkdir, mkdtemp, readFile, readdir, readlink, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-history-import-'));
const profile = path.join(temporary, 'profile');
const sources = path.join(temporary, 'original-logs');
const output = process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output', 'history-import');
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = { ...process.env, BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile };
delete env.ELECTRON_RUN_AS_NODE;
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
await mkdir(output, { recursive: true });
await mkdir(profile, { recursive: true });
await mkdir(sources, { recursive: true });
const alpha = '中文实验 · baseline';
const beta = 'second-snapshot';
const invalid = 'invalid-record';
const draft = '尚未提交的实验草稿，应在导入前后保留。';
const reportContent = '# Imported synthetic report\n\nThis is a UI fixture, not a medical result.\n';
const errors = [];
const checks = [];
let application;
let page;

async function fixture(name, complete) {
  const directory = path.join(sources, name);
  await mkdir(path.join(directory, 'working_best'), { recursive: true });
  await writeFile(path.join(directory, 'working_best', 'private-data.csv'), 'should never be copied');
  await writeFile(path.join(directory, 'weights.pt'), 'synthetic checkpoint exclusion');
  await writeFile(path.join(directory, '.env'), 'SYNTHETIC_SECRET=should-never-be-copied');
  await writeFile(path.join(directory, 'run_manifest.json'), JSON.stringify({
    data_dir: path.join(temporary, 'original-dataset'), created_at: '2026-10-08T12:00:00Z',
    task_description: 'Synthetic import fixture; no model training.',
    pid: 2147483647, pgid: 2147483647, launch_token: 'synthetic-original-ownership',
    workspace_dir: path.join(directory, 'working_best'),
    config: { agent_steps: 2, model_for_all_agents: 'qa-no-provider', per_node_timeout_seconds: 600 },
  }));
  await writeFile(path.join(directory, 'journal.json'), JSON.stringify({ __version: '2', node2parent: {}, nodes: [
    { id: 'candidate-a', step: 0, agent_name: 'Draft', is_buggy: false, metric: { value: 0.7, maximize: true },
      plan: 'Imported synthetic candidate plan.', code: '# Imported synthetic candidate code.', exec_time: 1 },
  ] }));
  await writeFile(path.join(directory, 'best_solution.py'), '# Imported synthetic best solution.\n');
  await writeFile(path.join(directory, 'report.md'), reportContent);
  await writeFile(path.join(directory, 'results.tsv'), 'step\tmetric\tmaximize\tstatus\n0\t0.7\ttrue\tsucceeded\n');
  const events = [{ ts: '2026-10-08T12:00:00Z', event: 'run_started' }];
  if (complete) events.push({ ts: '2026-10-08T12:01:00Z', event: 'run_finished', completed_steps: 1 });
  await writeFile(path.join(directory, 'events.jsonl'), events.map(event => JSON.stringify(event)).join('\n') + '\n');
  await writeFile(path.join(directory, `${name}.log`), 'Imported fixture log; no experiment was launched.\n');
  await symlink(path.join(directory, 'weights.pt'), path.join(directory, 'evaluator.py'));
}
await fixture(alpha, true);
await fixture(beta, false);
await mkdir(path.join(sources, invalid));
await writeFile(path.join(sources, invalid, 'run_manifest.json'), '{"data_dir":""}');

async function snapshot(directory) {
  const result = {};
  async function visit(parent, prefix = '') {
    for (const name of (await readdir(parent)).sort()) {
      const file = path.join(parent, name);
      const key = path.join(prefix, name);
      const stat = await lstat(file);
      if (stat.isSymbolicLink()) result[key] = { link: await readlink(file), mtimeMs: stat.mtimeMs };
      else if (stat.isDirectory()) await visit(file, key);
      else result[key] = { sha256: createHash('sha256').update(await readFile(file)).digest('hex'), mtimeMs: stat.mtimeMs };
    }
  }
  await visit(directory);
  return result;
}
async function until(callback, message, timeout = 60_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await callback()) return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(message);
}
async function launch() {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await application.firstWindow();
  page.on('pageerror', error => errors.push(error.message));
  page.setDefaultTimeout(20_000);
  await until(async () => {
    try { return (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready'; }
    catch { return false; }
  }, 'Desktop service did not become ready');
  await page.locator('.sidebar-settings').waitFor();
}
const api = (route, method = 'GET', body) => page.evaluate(({ route, method, body }) => window.biomaker.request(route, { method, body }), { route, method, body });
async function allExperiments() { await page.locator('.primary-nav').getByRole('button', { name: /全部实验/ }).click(); }
async function openImport() {
  await allExperiments();
  await page.getByRole('button', { name: '导入已有实验', exact: true }).first().click();
  await page.getByRole('dialog', { name: '导入已有实验' }).waitFor();
}
async function chooseSources() {
  await application.evaluate(({ dialog }, source) => {
    dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [source] });
  }, sources);
  await page.getByRole('button', { name: /^(选择实验目录|更换)$/ }).click();
  await page.getByLabel(`选择 ${alpha}`, { exact: true }).waitFor();
}
async function settled(jobId) {
  let job;
  await until(async () => {
    job = await api(`/api/desktop/history/imports/${jobId}`);
    return ['completed', 'failed'].includes(job.status);
  }, 'Import job did not settle');
  return job;
}

try {
  await launch();
  assert.deepEqual(await api('/api/runs'), []);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  await assert.rejects(api('/api/desktop/history/preview', 'POST', { path: sources }), /choose|selected|authorized|选择/i);
  checks.push('unpicked-source-rejected');
  await page.getByLabel('实验任务', { exact: true }).fill(draft);
  await openImport();
  await chooseSources();
  assert.equal(await page.getByLabel(`选择 ${invalid}`, { exact: true }).isDisabled(), true);
  await page.getByRole('button', { name: '清空选择', exact: true }).click();
  assert.equal(await page.getByRole('button', { name: '导入所选实验', exact: true }).isDisabled(), true);
  await page.getByLabel(`选择 ${alpha}`, { exact: true }).check();
  await page.screenshot({ path: path.join(output, '01-preview.png') });
  checks.push('native-folder-picker-preview-invalid-and-empty-selection');

  // A deliberate fixture mutation verifies stale preview rejection before any import.
  await writeFile(path.join(sources, alpha, 'report.md'), reportContent + '\nUpdated before importing.\n');
  const sourceBefore = await snapshot(sources);
  await page.getByRole('button', { name: '导入所选实验', exact: true }).click();
  await page.getByText('导入未完成', { exact: true }).waitFor();
  assert.deepEqual(await api('/api/runs'), []);
  await page.getByRole('button', { name: '导入其它实验', exact: true }).click();
  await chooseSources();
  await page.getByRole('button', { name: '导入所选实验', exact: true }).click();
  await page.getByText('导入完成', { exact: true }).waitFor();
  await page.screenshot({ path: path.join(output, '02-complete.png') });
  checks.push('changed-source-rejected-and-rescan-recovers');
  const records = await api('/api/runs');
  assert.equal(records.length, 2);
  const imported = records.find(run => run.history_import?.name === alpha);
  const snapshotRun = records.find(run => run.history_import?.name === beta);
  assert(imported && snapshotRun);
  assert.equal(imported.read_only, true);
  assert.equal(imported.managed, false);
  assert.deepEqual(imported.allowed_actions, []);
  assert.equal(snapshotRun.terminal_event, null);
  assert.match(snapshotRun.status_reason, /snapshot|completion|running/i);
  for (const action of ['stop', 'cancel', 'requeue']) {
    await assert.rejects(api(`/api/runs/${imported.id}/${action}`, 'POST', {}), /view-only|cannot be controlled/i);
  }
  const copied = path.join(profile, 'workspace', 'logs', imported.id);
  const copiedFiles = await readdir(copied);
  for (const name of ['weights.pt', '.env', 'working_best', 'evaluator.py']) assert(!copiedFiles.includes(name), name);
  const effective = JSON.parse(await readFile(path.join(copied, 'run_manifest.json'), 'utf8'));
  for (const name of ['pid', 'pgid', 'launch_token', 'workspace_dir']) assert.equal(effective[name], null);
  assert.deepEqual(await readFile(path.join(copied, 'source_run_manifest.json')), await readFile(path.join(sources, alpha, 'run_manifest.json')));
  checks.push('read-only-actions-blocked-ownership-cleared-and-exclusions-respected');

  await page.getByRole('button', { name: `打开 ${alpha}`, exact: true }).click();
  await page.locator('.run-intro').waitFor();
  assert((await page.locator('.run-intro').innerText()).includes(alpha));
  await page.getByText('从历史日志导入，仅供查看。', { exact: true }).waitFor();
  await page.getByRole('tab', { name: '候选方案', exact: true }).click();
  await page.getByText('Imported synthetic candidate plan.', { exact: true }).waitFor();
  await page.getByRole('tab', { name: '代码', exact: true }).click();
  await page.getByText('# Imported synthetic candidate code.', { exact: true }).waitFor();
  await page.screenshot({ path: path.join(output, '03-candidate-code.png') });
  await page.getByRole('tab', { name: /^文件/ }).click();
  const destination = path.join(temporary, 'exported-report.md');
  await application.evaluate(({ dialog }, filePath) => { dialog.showSaveDialog = async () => ({ canceled: false, filePath }); }, destination);
  await page.getByRole('button', { name: '保存 report.md', exact: true }).click();
  await until(async () => {
    try { return (await readFile(destination, 'utf8')).includes('Imported synthetic report'); }
    catch { return false; }
  }, 'Native report export was not written');
  assert.equal(await readFile(destination, 'utf8'), await readFile(path.join(sources, alpha, 'report.md'), 'utf8'));
  await page.getByRole('tab', { name: '运行日志', exact: true }).click();
  await page.getByText('Imported fixture log; no experiment was launched.', { exact: true }).waitFor();
  checks.push('candidate-code-log-viewing-and-native-report-export');

  await openImport();
  await chooseSources();
  assert.equal(await page.getByLabel(`选择 ${alpha}`, { exact: true }).isDisabled(), true);
  assert.equal(await page.getByLabel(`选择 ${beta}`, { exact: true }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '导入所选实验', exact: true }).isDisabled(), true);
  const preview = await api('/api/desktop/history/preview', 'POST', { path: sources });
  const duplicate = await api('/api/desktop/history/import', 'POST', { preview_id: preview.preview_id, source_ids: [imported.history_import.source_id] });
  const duplicateResult = await settled(duplicate.job_id);
  assert.equal(duplicateResult.imported.length, 0);
  assert.equal(duplicateResult.skipped[0].run_id, imported.id);
  await page.getByRole('button', { name: '取消', exact: true }).click();
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:menu-action', 'new-experiment'));
  assert.equal(await page.getByLabel('实验任务', { exact: true }).inputValue(), draft);
  checks.push('duplicates-skip-without-overwrite-and-draft-preserved');

  await application.close(); application = undefined;
  await launch();
  const restarted = await api('/api/runs');
  assert.equal(restarted.length, 2);
  assert.equal(restarted.find(run => run.id === imported.id).history_import.name, alpha);
  assert.equal((await api(`/api/desktop/history/imports/${duplicate.job_id}`)).status, 'completed');
  await allExperiments();
  await page.getByText(alpha, { exact: true }).last().click();
  await page.locator('.run-intro').waitFor();
  assert((await page.locator('.run-intro').innerText()).includes(alpha));
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.screenshot({ path: path.join(output, '04-restart-compact.png') });
  assert.deepEqual(await snapshot(sources), sourceBefore);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  assert.deepEqual(errors, []);
  checks.push('original-titles-and-job-receipts-survive-restart', 'source-hashes-and-mtimes-unchanged', 'no-engine-install-or-training', 'compact-layout-no-renderer-errors');
  const report = { passed: true, executablePath: executablePath || 'development Electron', profile, sources, checks, rendererErrors: errors };
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
