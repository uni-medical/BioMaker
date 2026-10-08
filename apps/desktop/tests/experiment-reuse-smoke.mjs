// Real Electron / sidecar acceptance of draft reuse. All records and directories
// are synthetic and isolated; no engine installation or provider call is allowed.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { lstat, mkdir, mkdtemp, readFile, readdir, realpath, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-experiment-reuse-'));
const profile = path.join(temporary, 'profile');
const sources = path.join(temporary, 'original-history');
const output = process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/experiment-reuse');
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = { ...process.env, BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1' };
for (const key of ['ELECTRON_RUN_AS_NODE', 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'SERPER_API_KEY', 'JINA_API_KEY']) delete env[key];
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const ownId = 'qa-reuse-owned-complete';
const ownTask = 'Synthetic owned experiment.\nReuse this exact task; never launch training.';
const importedTask = 'Synthetic imported experiment.\nThe old dataset location must not be reused.';
const ownOptions = { model_for_all_agents: 'qa-reuse-model', agent_steps: 7, exec_timeout: 123, exec_parallel_workers: 2, agent_total_time_limit: 5400, final_retrain_enabled: false };
const importedOptions = { model_for_all_agents: 'qa-imported-model', agent_steps: 4, exec_timeout: 321, exec_parallel_workers: 4, agent_total_time_limit: 9000, final_retrain_enabled: true };
const ownDataset = path.join(profile, 'workspace/datasets/owned-dataset');
const originalDataset = path.join(temporary, 'original-dataset');
const dirtyDataset = path.join(temporary, 'draft-dataset');
const selectedDataset = path.join(temporary, 'selected-for-new-experiment');
const ownLogs = path.join(profile, 'workspace/logs', ownId);
for (const directory of [profile, output, ownDataset, originalDataset, dirtyDataset, selectedDataset]) await mkdir(directory, { recursive: true });
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));

async function seedHistory(directory, dataset, task, options) {
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, 'run_manifest.json'), JSON.stringify({
    data_dir: dataset, task_description: task, created_at: '2026-10-08T10:00:00Z',
    config: { model_for_all_agents: options.model_for_all_agents, agent_steps: options.agent_steps, per_node_timeout_seconds: options.exec_timeout, parallel_workers: options.exec_parallel_workers, total_time_limit_seconds: options.agent_total_time_limit, final_retrain_enabled: options.final_retrain_enabled },
  }));
  await writeFile(path.join(directory, 'events.jsonl'), [
    { ts: '2026-10-08T10:00:00Z', event: 'run_started' },
    { ts: '2026-10-08T10:02:00Z', event: 'run_finished', completed_steps: options.agent_steps },
  ].map(event => JSON.stringify(event)).join('\n') + '\n');
  await writeFile(path.join(directory, 'report.md'), '# Synthetic reuse acceptance\nNo model was trained.\n');
  await writeFile(path.join(directory, 'results.tsv'), 'step\tmetric\tmaximize\tstatus\n0\t0.7\ttrue\tsucceeded\n');
  await writeFile(path.join(directory, `${path.basename(directory)}.log`), 'Synthetic original log. Reuse must not change this file.\n');
}
await seedHistory(ownLogs, ownDataset, ownTask, ownOptions);
await seedHistory(sources, originalDataset, importedTask, importedOptions);

async function snapshot(directory) {
  const result = {};
  async function visit(parent, prefix = '') {
    for (const name of (await readdir(parent)).sort()) {
      const file = path.join(parent, name);
      const stat = await lstat(file);
      const relative = path.join(prefix, name);
      if (stat.isDirectory()) await visit(file, relative);
      else result[relative] = { sha256: createHash('sha256').update(await readFile(file)).digest('hex'), mtimeMs: stat.mtimeMs };
    }
  }
  await visit(directory);
  return result;
}
const sourceBefore = await snapshot(sources);
const ownBefore = await snapshot(ownLogs);
let application;
let page;
const errors = [];
const forbiddenRequests = [];
const checks = [];
async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 50));
  }
}
async function launch() {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  await application.evaluate(() => {
    const original = globalThis.fetch;
    globalThis.__reuseForbiddenRequests = [];
    globalThis.fetch = async (input, options) => {
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      const mutatingRun = method !== 'GET' && /^\/api\/runs(?:\/|$)/.test(url.pathname);
      const installOrProvider = /^\/api\/(?:desktop\/engine\/(?:install|cancel|check|repair)|models\/discover|env(?:\/|$))/.test(url.pathname) && method !== 'GET';
      if (url.hostname !== '127.0.0.1' || mutatingRun || installOrProvider) {
        globalThis.__reuseForbiddenRequests.push({ method, path: url.pathname });
        throw new Error('Reuse QA blocked a run mutation, installation, or provider operation');
      }
      return original(input, options);
    };
  });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => errors.push(error.message));
  await until(async () => {
    try { return (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready'; }
    catch { return false; }
  }, 'The real desktop service did not become ready', 60_000);
  await page.locator('.sidebar-settings').waitFor();
}
async function close() {
  if (!application) return;
  forbiddenRequests.push(...await application.evaluate(() => globalThis.__reuseForbiddenRequests));
  await application.close(); application = undefined;
}
const api = (endpoint, method = 'GET', body) => page.evaluate(({ endpoint, method, body }) => window.biomaker.request(endpoint, { method, ...(body === undefined ? {} : { body }) }), { endpoint, method, body });
const storedDraft = () => page.evaluate(() => localStorage.getItem('biomaker.draft'));
async function draftObject() { return JSON.parse(await storedDraft()); }
async function openRun(id) {
  // Native notification navigation selects the fixture; the reuse button itself
  // is exercised through its accessible UI label below.
  await application.evaluate(({ BrowserWindow }, runId) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:open-run', runId), id);
  await page.locator('.run-intro').waitFor();
  await page.getByRole('button', { name: '复用为新实验', exact: true }).waitFor();
}
async function newDraft() {
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:menu-action', 'new-experiment'));
  await page.getByLabel('实验任务', { exact: true }).waitFor();
}
async function advanced() {
  const button = page.getByRole('button', { name: '实验设置', exact: true });
  if (await button.getAttribute('aria-expanded') !== 'true') await button.click();
}
async function assertForm(expected) {
  assert.equal(await page.getByLabel('实验任务', { exact: true }).inputValue(), expected.task);
  assert.equal(await page.locator('.dataset-button').getAttribute('title'), expected.dataset || '选择数据集文件夹');
  await advanced();
  assert.equal(await page.getByLabel('实验名称', { exact: false }).inputValue(), expected.name);
  assert.equal(await page.getByLabel('最大探索步数', { exact: true }).inputValue(), String(expected.options.agent_steps));
  assert.equal(await page.getByLabel('探索时长上限（小时）', { exact: false }).inputValue(), String(Number((expected.options.agent_total_time_limit / 3600).toFixed(4))));
  assert.equal(await page.getByLabel('单次执行时限（秒）', { exact: true }).inputValue(), String(expected.options.exec_timeout));
  assert.equal(await page.getByLabel('并发工作进程', { exact: false }).inputValue(), String(expected.options.exec_parallel_workers));
  assert.equal(await page.getByLabel('此次实验使用的模型', { exact: true }).inputValue(), expected.options.model_for_all_agents || '');
  assert.equal(await page.getByRole('checkbox', { name: /^最终重训/ }).isChecked(), expected.options.final_retrain_enabled);
}
async function chooseDataset(directory) {
  await application.evaluate(({ dialog }, folder) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [folder] }); }, directory);
  await page.locator('.dataset-button').click();
  const resolved = await realpath(directory);
  await until(async () => (await draftObject())?.dataset === resolved, 'Chosen dataset was not persisted');
  return resolved;
}
function invariantRecord(run) {
  return Object.fromEntries(['id', 'managed', 'read_only', 'status', 'detailed_status', 'timestamps', 'allowed_actions', 'project_snapshot', 'options', 'history_import'].map(key => [key, run[key]]));
}

try {
  await launch();
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  await until(async () => (await api('/api/runs')).some(run => run.id === ownId), 'The local synthetic history was not cataloged');

  // Turn exactly one synthetic completed record into an owned-ledger fixture.
  // -I -B prevents the immutable bundle interpreter from creating bytecode.
  const runtime = executablePath ? path.resolve(path.dirname(executablePath), process.platform === 'darwin' ? '../Resources/runtime' : 'resources/runtime') : path.join(root, 'build/runtime');
  const python = path.join(runtime, 'python', process.platform === 'win32' ? 'python.exe' : 'bin/python3');
  const seeded = spawnSync(python, ['-I', '-B', '-c', `import json,sqlite3,sys\nc=sqlite3.connect(sys.argv[1])\nwith c:\n r=json.loads(c.execute('SELECT record_json FROM runs WHERE run_id=?',(sys.argv[2],)).fetchone()[0])\n r.update(managed=True,read_only=False,source='run_ledger',status='succeeded',detailed_status='succeeded',command=[],pid=None,pgid=None,launch_token=None)\n c.execute('UPDATE runs SET record_json=? WHERE run_id=?',(json.dumps(r),sys.argv[2]))`, path.join(profile, 'state/runs.sqlite3'), ownId], { encoding: 'utf8' });
  assert.equal(seeded.status, 0, seeded.stderr);

  await application.evaluate(({ dialog }, folder) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [folder] }); }, sources);
  const selectedSource = await page.evaluate(() => window.biomaker.chooseDirectory({ purpose: 'history' }));
  const preview = await api('/api/desktop/history/preview', 'POST', { path: selectedSource });
  const job = await api('/api/desktop/history/import', 'POST', { preview_id: preview.preview_id, source_ids: [preview.runs[0].source_id] });
  let settled;
  await until(async () => { settled = await api(`/api/desktop/history/imports/${job.job_id}`); return ['completed', 'failed'].includes(settled.status); }, 'Synthetic history import did not finish');
  assert.equal(settled.status, 'completed', JSON.stringify(settled));
  const importedId = settled.imported[0].run_id;
  const importedLogs = path.join(profile, 'workspace/logs', importedId);
  const importedBefore = await snapshot(importedLogs);
  const beforeRecords = await api('/api/runs');
  assert.equal(beforeRecords.length, 2);
  const ownRecord = beforeRecords.find(run => run.id === ownId);
  const importedRecord = beforeRecords.find(run => run.id === importedId);
  assert.equal(ownRecord.read_only, false);
  assert.equal(importedRecord.read_only, true);
  assert.deepEqual(importedRecord.allowed_actions, []);
  await page.getByRole('button', { name: '刷新', exact: true }).click();

  await openRun(ownId);
  await page.getByRole('button', { name: '复用为新实验', exact: true }).click();
  await page.getByLabel('实验任务', { exact: true }).waitFor();
  const ownExpected = { task: ownTask, dataset: ownDataset, name: '', template: null, options: ownOptions };
  await assertForm(ownExpected);
  let reused = await draftObject();
  for (const [key, value] of Object.entries(ownExpected)) assert.deepEqual(reused[key], value);
  assert.equal(reused.reusedFrom.runId, ownId);
  checks.push('owned-run-reuse-prefills-task-dataset-and-all-known-options-without-launch');

  const dirtyTask = '  保留这份草稿的所有空格。\nSecond line: do not replace me.  ';
  const dirtyOptions = { model_for_all_agents: 'qa-unsaved-model', agent_steps: 11, exec_timeout: 222, exec_parallel_workers: 4, agent_total_time_limit: 8100, final_retrain_enabled: true };
  await page.getByLabel('实验任务', { exact: true }).fill(dirtyTask);
  const dirtyPath = await chooseDataset(dirtyDataset);
  await page.getByLabel('实验名称', { exact: false }).fill('qa-unsaved-draft');
  await page.getByLabel('最大探索步数', { exact: true }).fill('11');
  await page.getByLabel('探索时长上限（小时）', { exact: false }).fill('2.25');
  await page.getByLabel('单次执行时限（秒）', { exact: true }).fill('222');
  await page.getByLabel('并发工作进程', { exact: false }).selectOption('4');
  await page.getByLabel('此次实验使用的模型', { exact: true }).fill('qa-unsaved-model');
  await page.getByRole('checkbox', { name: /^最终重训/ }).check();
  await until(async () => {
    const options = (await draftObject())?.options;
    return options && Object.entries(dirtyOptions).every(([key, value]) => options[key] === value);
  }, 'Edited options were not persisted');
  const dirtyRaw = await storedDraft();
  const dirtyExpected = { task: dirtyTask, dataset: dirtyPath, name: 'qa-unsaved-draft', options: dirtyOptions };
  await assertForm(dirtyExpected);
  await openRun(importedId);
  await page.getByRole('button', { name: '复用为新实验', exact: true }).click();
  await page.getByRole('dialog', { name: '替换当前草稿？', exact: true }).waitFor();
  await until(async () => await page.getByRole('button', { name: '保留当前草稿', exact: true }).evaluate(element => document.activeElement === element), 'Cancel must receive the initial focus in the replacement dialog');
  await page.getByRole('button', { name: '保留当前草稿', exact: true }).click();
  assert.equal(await storedDraft(), dirtyRaw);
  await newDraft();
  await assertForm(dirtyExpected);
  assert.equal(await storedDraft(), dirtyRaw);
  checks.push('cancel-reuse-preserves-exact-draft-text-path-name-and-options');

  await openRun(ownId);
  await page.getByRole('button', { name: '复用为新实验', exact: true }).click();
  await page.getByRole('button', { name: '替换并复用', exact: true }).click();
  await page.getByLabel('实验任务', { exact: true }).waitFor();
  const replacedRaw = await storedDraft();
  for (const [key, value] of Object.entries(ownExpected)) assert.deepEqual(JSON.parse(replacedRaw)[key], value);
  await close();
  await launch();
  await newDraft();
  assert.deepEqual(await draftObject(), JSON.parse(replacedRaw));
  await assertForm(ownExpected);
  checks.push('explicit-replacement-persists-immediately-and-survives-app-restart');

  await openRun(importedId);
  await page.getByRole('button', { name: '复用为新实验', exact: true }).click();
  await page.getByRole('button', { name: '替换并复用', exact: true }).click();
  await page.getByLabel('实验任务', { exact: true }).waitFor();
  const importExpected = { task: importedTask, dataset: '', name: '', template: null, options: importedOptions };
  await assertForm(importExpected);
  reused = await draftObject();
  for (const [key, value] of Object.entries(importExpected)) assert.deepEqual(reused[key], value);
  assert.equal(reused.reusedFrom.runId, importedId);
  assert.equal(reused.reusedFrom.requiresDataset, true);
  assert.equal(await page.locator('.submit-button').isDisabled(), true);
  assert.match(await page.locator('.dataset-button').innerText(), /选择数据集/);
  await page.screenshot({ path: path.join(output, '01-imported-reuse-needs-dataset.png') });
  await chooseDataset(selectedDataset);
  assert.notEqual((await draftObject()).dataset, originalDataset);
  assert.equal((await draftObject()).dataset, await realpath(selectedDataset));
  assert.equal(await page.locator('.submit-button').isDisabled(), false, 'Selecting a dataset enables review of the missing runtime setup; it does not start a run');
  checks.push('imported-readonly-reuse-clears-existing-dataset-and-requires-a-new-selection');

  const afterRecords = await api('/api/runs');
  assert.deepEqual(afterRecords.map(run => run.id).sort(), beforeRecords.map(run => run.id).sort());
  for (const before of beforeRecords) assert.deepEqual(invariantRecord(afterRecords.find(run => run.id === before.id)), invariantRecord(before));
  assert.deepEqual(await snapshot(sources), sourceBefore);
  assert.deepEqual(await snapshot(ownLogs), ownBefore);
  assert.deepEqual(await snapshot(importedLogs), importedBefore);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  assert.equal((await api('/api/env')).values.OPENAI_API_KEY.configured, false);
  await page.screenshot({ path: path.join(output, '02-reuse-ready-for-review.png') });
  await close();
  assert.deepEqual(forbiddenRequests, []);
  assert.deepEqual(errors, []);
  checks.push('source-and-imported-log-hashes-and-mtimes-unchanged', 'original-ledger-records-and-readonly-controls-unchanged', 'zero-run-mutations-provider-requests-or-engine-installs', 'no-renderer-errors');
  const report = { passed: true, executablePath: executablePath || 'development Electron', profile, source: sources, checks, forbiddenRequests, rendererErrors: errors };
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
    await close();
  }
}
