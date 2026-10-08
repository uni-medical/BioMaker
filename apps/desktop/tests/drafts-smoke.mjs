// Real Electron/native draft persistence and sidecar metadata, isolated synthetic
// profile only. No experiment, engine installation, or external provider call.
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, mkdtemp, readFile, realpath, unlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-drafts-'));
const profile = path.join(temporary, 'profile');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/drafts-0.1.12'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1', NO_PROXY: '127.0.0.1,localhost' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const datasetA = path.join(profile, 'workspace/datasets/draft-a');
const datasetB = path.join(profile, 'workspace/datasets/draft-b');
const historyId = 'qa-drafts-history';
const history = path.join(profile, 'workspace/logs', historyId);
for (const directory of [profile, output, datasetA, datasetB, history]) await mkdir(directory, { recursive: true });
for (const directory of [datasetA, datasetB]) await writeFile(path.join(directory, 'synthetic.csv'), 'feature,label\n1,0\n2,1\n');
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
const optionsA = { model_for_all_agents: 'qa-drafts-model-a', agent_steps: 7, exec_timeout: 123, exec_parallel_workers: 2, agent_total_time_limit: 5400, final_retrain_enabled: false };
const optionsB = { model_for_all_agents: 'qa-drafts-model-b', agent_steps: 11, exec_timeout: 456, exec_parallel_workers: 4, agent_total_time_limit: 7200, final_retrain_enabled: true };
const legacy = {
  task: '  Legacy draft A.\nKeep the exact whitespace and settings.  ', dataset: await realpath(datasetA), name: 'qa-legacy-a', options: optionsA,
  template: null, reusedFrom: { runId: 'legacy-source', title: '旧实验来源', requiresDataset: false, usedDefaults: false },
};
const historyTask = 'Synthetic history reused as an independent draft. Never execute this task.';
const historyFiles = {
  'run_manifest.json': JSON.stringify({ data_dir: datasetA, task_description: historyTask, created_at: '2026-10-08T10:00:00Z', config: {
    model_for_all_agents: optionsB.model_for_all_agents, agent_steps: optionsB.agent_steps, per_node_timeout_seconds: optionsB.exec_timeout,
    parallel_workers: optionsB.exec_parallel_workers, total_time_limit_seconds: optionsB.agent_total_time_limit, final_retrain_enabled: optionsB.final_retrain_enabled,
  } }),
  'events.jsonl': JSON.stringify({ ts: '2026-10-08T10:02:00Z', event: 'run_finished', completed_steps: 1 }) + '\n',
  'report.md': '# Synthetic draft reuse fixture\nNo model was trained.\n',
};
for (const [name, contents] of Object.entries(historyFiles)) await writeFile(path.join(history, name), contents);

let application;
let page;
const checks = [];
const rendererErrors = [];
const forbiddenRequests = [];
async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 40));
  }
}
const api = (pathname, method = 'GET', body) => page.evaluate(([pathname, method, body]) => window.biomaker.request(pathname, { method, ...(body === undefined ? {} : { body }) }), [pathname, method, body]);
const taskInput = () => page.getByLabel('实验任务', { exact: true });
const draftRow = id => page.locator(`.draft-library-row[data-draft-id="${id}"]`);
async function ready(nextPage) {
  page = nextPage || await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  await until(async () => {
    try { return (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready'; }
    catch { return false; }
  }, 'The real sidecar did not become ready', 60_000);
  await page.locator('.primary-nav').waitFor();
}
async function launch() {
  console.log(`[drafts-smoke] Launching isolated profile ${profile}`);
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  const firstPage = await application.firstWindow();
  await application.evaluate(({ app, dialog }) => {
    const originalFetch = globalThis.fetch;
    const state = globalThis.__draftsQA = { forbidden: [], dialogs: [], mutations: [], discardNext: false, holdSettings: false };
    const { createRequire } = process.getBuiltinModule('node:module');
    const nativeRequire = createRequire(process.execPath);
    const nativePath = nativeRequire('node:path');
    const { DraftStore } = nativeRequire(nativePath.join(app.getAppPath(), 'dist-electron/drafts.js'));
    const originalMutation = DraftStore.prototype.mutateDrafts;
    DraftStore.prototype.mutateDrafts = function(input) {
      // The real native store commits first. Only its IPC acknowledgment is
      // delayed or failed, so retry checks exercise durable mutation receipts.
      const result = originalMutation.call(this, input);
      const operation = { input: structuredClone(input), revision: result.revision, done: true };
      state.mutations.push(operation);
      if (state.nextMutation?.type !== input.operation.type) return result;
      state.nextMutation = null;
      operation.done = false;
      return new Promise((resolve, reject) => {
        state.releaseMutation = (fail = false) => {
          delete state.releaseMutation;
          operation.done = true;
          if (fail) reject(new Error('[drafts:save_failed] Synthetic lost native acknowledgment'));
          else resolve(result);
        };
      });
    };
    globalThis.fetch = async (input, options) => {
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      if (url.pathname === '/api/desktop/dataset/inspect' && method === 'POST' && state.holdInspectionPath && JSON.parse(options.body).path === state.holdInspectionPath) {
        state.holdInspectionPath = null;
        const response = await originalFetch(input, options);
        const contents = await response.text();
        state.heldInspection = JSON.parse(contents);
        await new Promise(resolve => { state.releaseInspection = resolve; });
        state.inspectionReleased = true;
        return new Response(contents, { status: response.status, headers: response.headers });
      }
      if (url.hostname === '127.0.0.1' && url.pathname === '/api/env' && method === 'PUT' && state.holdSettings) {
        state.holdSettings = false;
        await new Promise(resolve => { state.releaseSettings = resolve; });
        return new Response(JSON.stringify({ detail: 'Synthetic held settings failure' }), { status: 503, headers: { 'Content-Type': 'application/json' } });
      }
      if (url.hostname !== '127.0.0.1' || (method !== 'GET' && url.pathname !== '/api/desktop/dataset/inspect')) {
        state.forbidden.push({ method, path: url.pathname });
        throw new Error('Draft QA blocked a run, install, or provider mutation');
      }
      return originalFetch(input, options);
    };
    const answer = (...args) => {
      const options = args.at(-1);
      state.dialogs.push({ title: options.title, message: options.message, buttons: options.buttons });
      const discard = state.discardNext;
      state.discardNext = false;
      const index = options.buttons.findIndex(label => /放弃|丢弃|Discard/i.test(label));
      return discard && index >= 0 ? index : options.cancelId ?? 0;
    };
    dialog.showMessageBoxSync = answer;
    dialog.showMessageBox = async (...args) => ({ response: answer(...args), checkboxChecked: false });
  });
  await ready(firstPage);
}
async function closeApp() {
  if (!application) return;
  forbiddenRequests.push(...await application.evaluate(() => globalThis.__draftsQA.forbidden));
  await application.close();
  application = undefined;
  page = undefined;
}
async function newDraft() {
  const previous = (await snapshot()).activeId;
  await page.locator('.primary-nav').getByRole('button', { name: /新建实验/ }).click();
  await persisted(state => state.activeId !== previous, 'New did not create an independent draft');
  await taskInput().waitFor();
}
async function library() {
  await page.locator('.primary-nav').getByRole('button', { name: /草稿/ }).click();
  await page.getByRole('region', { name: '草稿', exact: true }).waitFor();
}
async function openDraft(id) {
  await library();
  await draftRow(id).locator('.draft-library-open').click();
  await persisted(state => state.activeId === id, 'Opening a draft did not activate its identity');
  await taskInput().waitFor();
}
async function advanced() {
  const button = page.getByRole('button', { name: '实验设置', exact: true });
  if (await button.getAttribute('aria-expanded') !== 'true') await button.click();
}
async function setOptions(options, name) {
  await advanced();
  await page.getByLabel('实验名称', { exact: false }).fill(name);
  await page.getByLabel('最大探索步数', { exact: true }).fill(String(options.agent_steps));
  await page.getByLabel('探索时长上限（小时）', { exact: false }).fill(String(options.agent_total_time_limit / 3600));
  await page.getByLabel('单次执行时限（秒）', { exact: true }).fill(String(options.exec_timeout));
  await page.getByLabel('并发工作进程', { exact: false }).selectOption(String(options.exec_parallel_workers));
  await page.getByLabel('此次实验使用的模型', { exact: true }).fill(options.model_for_all_agents || '');
  await page.getByRole('checkbox', { name: /^最终重训/ }).setChecked(options.final_retrain_enabled);
}
async function assertForm(expected) {
  assert.equal(await taskInput().inputValue(), expected.task);
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
  await application.evaluate(({ dialog }, selected) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [selected] }); }, directory);
  await page.locator('.dataset-button').click();
  const canonical = await realpath(directory);
  await until(async () => await page.locator('.dataset-button').getAttribute('title') === canonical, 'Dataset selection did not reach the current draft');
  return canonical;
}
async function guardedClose(quit = false) {
  const before = await application.evaluate(() => globalThis.__draftsQA.dialogs.length);
  await application.evaluate(({ app, BrowserWindow }, quitting) => {
    setTimeout(() => quitting ? app.quit() : BrowserWindow.getAllWindows()[0].close(), 0);
  }, quit);
  await until(async () => (await application.evaluate(() => globalThis.__draftsQA.dialogs.length)) === before + 1, 'Native close did not present exactly one protection dialog');
  assert.equal(page.isClosed(), false);
  assert.equal((await api('/api/health')).status, 'ok');
  return application.evaluate(() => globalThis.__draftsQA.dialogs.at(-1));
}
async function reopenWindow() {
  assert.equal(process.platform, 'darwin', 'Window recreation acceptance currently targets macOS');
  const next = application.waitForEvent('window');
  await application.evaluate(({ app }) => app.emit('activate'));
  await ready(await next);
}

const draftFile = path.join(profile, 'experiment-drafts.json');
const snapshot = () => page.evaluate(() => window.biomaker.getDrafts());
const savedDraft = (state, id = state.activeId) => state.drafts.find(draft => draft.id === id);
const mutate = input => page.evaluate(value => window.biomaker.mutateDrafts(value), input);
const flush = input => page.evaluate(value => window.biomaker.flushDrafts(value), input);
const nativeInput = (state, operation) => ({ expectedRevision: state.revision, mutationId: randomUUID(), operation });
const diskDocument = async () => JSON.parse(await readFile(draftFile, 'utf8'));
async function persisted(predicate, message) {
  let value;
  await until(async () => { value = await snapshot(); return predicate(value); }, message);
  return value;
}
async function renameDraft(id, title) {
  await library();
  await draftRow(id).getByRole('button', { name: /^重命名草稿：/ }).click();
  await page.getByRole('dialog', { name: '重命名草稿', exact: true }).waitFor();
  await page.getByLabel('草稿标题', { exact: true }).fill(title);
  await page.getByRole('button', { name: '保存标题', exact: true }).click();
  await page.getByRole('dialog', { name: '重命名草稿', exact: true }).waitFor({ state: 'hidden' });
}
async function openHistory() {
  await application.evaluate(({ BrowserWindow }, id) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:open-run', id), historyId);
  await page.getByRole('button', { name: '复用为新实验', exact: true }).waitFor();
}

try {
  // Prime Chromium's real persistent origin, then remove only the verified-empty
  // native QA store with the application stopped to simulate a 0.1.11 upgrade.
  await launch();
  await persisted(state => state.legacyMigration !== null && state.drafts.length === 1, 'Initial native workspace was not ready');
  const bootstrap = await snapshot();
  assert(bootstrap.drafts.every(item => item.content.task === '' && item.content.dataset === '' && item.content.name === ''));
  const legacyRaw = JSON.stringify(legacy);
  await page.evaluate(raw => localStorage.setItem('biomaker.draft', raw), legacyRaw);
  await closeApp();
  const empty = await diskDocument();
  assert.equal(empty.drafts.length, 1);
  assert.equal(empty.drafts[0].content.task, '');
  assert.equal(empty.legacyMigration.raw, null);
  await unlink(draftFile);
  await launch();
  const migrated = await persisted(state => !!state.legacyMigration?.draftId, 'Legacy draft was not migrated');
  assert.equal(migrated.drafts.length, 1);
  const aId = migrated.activeId;
  assert.deepEqual(savedDraft(migrated).content, legacy);
  assert.equal(migrated.legacyMigration.sha256, createHash('sha256').update(legacyRaw).digest('hex'));
  assert.equal((await diskDocument()).legacyMigration.raw, legacyRaw);
  await assertForm(legacy);
  await renameDraft(aId, '旧草稿 A · 完整保留');
  checks.push('legacy-single-draft-migrates-exact-content-options-and-source-once');

  await newDraft();
  const bId = (await snapshot()).activeId;
  assert.notEqual(bId, aId);
  const expectedB = { task: 'Draft B.\nA second independent experiment.', dataset: await realpath(datasetB), name: 'qa-second-b', options: optionsB, template: null };
  await taskInput().fill(expectedB.task);
  await chooseDataset(datasetB);
  await setOptions(optionsB, expectedB.name);
  await persisted(state => isDeepStrictEqual(savedDraft(state, bId)?.content, expectedB), 'Second draft content was not saved');
  await renameDraft(bId, '第二份草稿 B');
  await openDraft(aId);
  await assertForm(legacy);
  await openDraft(bId);
  await assertForm(expectedB);
  const twoDrafts = (await snapshot()).drafts;
  assert.deepEqual(savedDraft({ drafts: twoDrafts }, aId).content, legacy);
  checks.push('new-and-switch-preserve-independent-task-dataset-name-and-all-options');

  await library();
  await page.getByLabel('搜索草稿', { exact: true }).fill('不存在的草稿');
  await page.getByRole('heading', { name: '没有匹配的草稿', exact: true }).waitFor();
  await page.getByRole('button', { name: '清除搜索', exact: true }).click();
  assert.equal(await page.locator('.draft-library-row').count(), 2);
  await draftRow(aId).getByRole('button', { name: /^复制草稿：/ }).click();
  const copied = await persisted(state => state.drafts.length === 3, 'Copy did not create a third draft');
  const copyId = copied.drafts.find(item => item.id !== aId && item.id !== bId).id;
  assert.deepEqual(savedDraft(copied, copyId).content, legacy);
  assert.deepEqual(savedDraft(copied, aId).content, legacy);
  await openDraft(aId);
  await until(async () => await page.locator('.dataset-overview').getAttribute('data-code') === 'ok' && await page.locator('.dataset-overview').getAttribute('data-state') !== 'checking', 'Source draft inspection did not finish');
  await application.evaluate((_electron, directory) => { globalThis.__draftsQA.holdInspectionPath = directory; }, legacy.dataset);
  await page.getByRole('button', { name: '重新检查数据集', exact: true }).click();
  await until(() => application.evaluate(() => typeof globalThis.__draftsQA.releaseInspection === 'function'), 'Same-path inspection response was not held');
  assert.equal(await application.evaluate(() => globalThis.__draftsQA.heldInspection.files), 1);
  await openDraft(copyId);
  assert.equal(await page.locator('.dataset-overview').getAttribute('data-state'), 'checking');
  assert.equal(await page.locator('.dataset-overview-counts').count(), 0);
  await openDraft(aId);
  assert.equal(await page.locator('.dataset-overview').getAttribute('data-state'), 'checking');
  const secondMetadataFile = path.join(datasetA, 'appeared-after-inspection.csv');
  await writeFile(secondMetadataFile, 'Synthetic file added after the held inspection.\n');
  const beforeInspectionReturn = await snapshot();
  await application.evaluate(() => globalThis.__draftsQA.releaseInspection());
  await until(async () => await page.locator('.dataset-overview').getAttribute('data-code') === 'ok' && /2 个普通文件/.test(await page.locator('.dataset-overview-counts').innerText()), 'Switching same-path drafts accepted the obsolete one-file inspection');
  assert.deepEqual(await snapshot(), beforeInspectionReturn);
  await unlink(secondMetadataFile);
  await openDraft(copyId);
  checks.push('same-path-a-b-a-draft-switch-rejects-an-old-inspection-and-reads-fresh-directory-metadata');
  await library();
  await draftRow(copyId).getByRole('button', { name: /^删除草稿：/ }).click();
  const deleteDialog = page.getByRole('dialog', { name: '删除这份草稿？', exact: true });
  await deleteDialog.waitFor();
  assert.equal(await deleteDialog.getByRole('button', { name: '保留草稿', exact: true }).evaluate(button => document.activeElement === button), true);
  const beforeCancel = await snapshot();
  await deleteDialog.getByRole('button', { name: '保留草稿', exact: true }).click();
  assert.deepEqual(await snapshot(), beforeCancel);
  await draftRow(copyId).getByRole('button', { name: /^删除草稿：/ }).click();
  await deleteDialog.getByRole('button', { name: '删除草稿', exact: true }).click();
  await persisted(state => state.drafts.length === 2 && !state.drafts.some(item => item.id === copyId), 'Delete did not remove only the copied draft');
  assert.equal((await snapshot()).activeId, null);
  checks.push('rename-search-copy-and-cancel-or-confirm-delete-use-independent-draft-identities');

  const beforeReuse = (await snapshot()).drafts;
  await openHistory();
  await page.getByRole('button', { name: '复用为新实验', exact: true }).click();
  await taskInput().waitFor();
  const reused = await persisted(state => state.drafts.length === 3 && savedDraft(state)?.content.task === historyTask, 'History reuse did not create an independent draft');
  const reusedId = reused.activeId;
  assert.equal(savedDraft(reused).content.dataset, '');
  assert.equal(savedDraft(reused).content.reusedFrom.requiresDataset, true);
  assert.deepEqual(reused.drafts.filter(item => item.id !== reusedId), beforeReuse);
  assert.equal(await page.getByRole('dialog', { name: '替换当前草稿？', exact: true }).count(), 0);
  checks.push('readonly-history-reuse-adds-a-draft-preserves-other-drafts-and-requires-dataset-selection');

  await openDraft(bId);
  const beforeRestart = await snapshot();
  await closeApp();
  await launch();
  await persisted(state => state.activeId === bId, 'Active draft was not restored');
  assert.deepEqual(await snapshot(), beforeRestart);
  await assertForm(expectedB);
  assert.equal(await page.evaluate(() => localStorage.getItem('biomaker.draft')), legacyRaw);
  checks.push('native-restart-restores-exact-library-active-draft-and-does-not-remigrate-legacy-key');

  await advanced();
  await page.getByLabel('最大探索步数', { exact: true }).fill('');
  await page.getByLabel('探索时长上限（小时）', { exact: false }).fill('');
  await page.getByLabel('单次执行时限（秒）', { exact: true }).fill('-3.5');
  const incomplete = await persisted(state => {
    const options = savedDraft(state, bId)?.content.options;
    return options?.agent_steps === null && options.agent_total_time_limit === null && options.exec_timeout === -3.5;
  }, 'Incomplete numeric draft values were not preserved');
  await closeApp();
  await launch();
  await taskInput().waitFor();
  assert.deepEqual(await snapshot(), incomplete);
  await advanced();
  assert.equal(await page.getByLabel('最大探索步数', { exact: true }).inputValue(), '');
  assert.equal(await page.getByLabel('探索时长上限（小时）', { exact: false }).inputValue(), '');
  assert.equal(await page.getByLabel('单次执行时限（秒）', { exact: true }).inputValue(), '-3.5');
  await page.getByLabel('最大探索步数', { exact: true }).fill('2.5');
  await persisted(state => savedDraft(state, bId)?.content.options.agent_steps === 2.5, 'Fractional draft input was truncated');
  await openDraft(aId);
  await openDraft(bId);
  await advanced();
  assert.equal(await page.getByLabel('最大探索步数', { exact: true }).inputValue(), '2.5');
  assert.equal(await page.getByLabel('单次执行时限（秒）', { exact: true }).inputValue(), '-3.5');
  await setOptions(optionsB, expectedB.name);
  await persisted(state => isDeepStrictEqual(savedDraft(state, bId)?.content, expectedB), 'Valid B settings were not restored');
  checks.push('incomplete-numeric-budgets-persist-as-null-reopen-blank-and-preserve-negative-or-fractional-edits');

  const beforeOversize = await snapshot();
  const oversizedTask = 'Synthetic oversized editable draft. ' + 'x'.repeat(270_000);
  await taskInput().fill(oversizedTask);
  await page.getByText('草稿数量或内容超过本机保存上限，请缩短内容或整理已有草稿。', { exact: true }).waitFor();
  assert.deepEqual(await snapshot(), beforeOversize);
  await library();
  await openDraft(bId);
  assert.equal(await taskInput().inputValue(), oversizedTask);
  assert.deepEqual(await snapshot(), beforeOversize);
  await taskInput().fill(expectedB.task);
  await persisted(state => isDeepStrictEqual(savedDraft(state, bId)?.content, expectedB), 'Shortening the oversized draft did not restore persistence');
  await until(async () => await page.locator('.draft-save-error').count() === 0, 'Oversize save error did not clear after shortening');
  checks.push('oversized-dirty-draft-can-return-from-library-to-the-same-editor-and-save-after-shortening');

  // Delay a real draft activation acknowledgment, then navigate through the
  // native notification channel. The old acknowledgment must not steal focus.
  await library();
  await application.evaluate(() => { globalThis.__draftsQA.nextMutation = { type: 'activate' }; });
  await draftRow(aId).locator('.draft-library-open').click();
  await until(() => application.evaluate(() => typeof globalThis.__draftsQA.releaseMutation === 'function'), 'Native draft activation acknowledgment was not held');
  assert.equal((await snapshot()).activeId, aId);
  await openHistory();
  await application.evaluate(() => globalThis.__draftsQA.releaseMutation());
  await until(() => page.locator('.primary-nav').getByRole('button', { name: /新建实验/ }).isEnabled(), 'Draft activation did not settle after acknowledgment');
  assert.equal(await taskInput().count(), 0);
  await page.getByRole('button', { name: '复用为新实验', exact: true }).waitFor();
  checks.push('held-native-draft-activation-acknowledgment-does-not-steal-newer-run-navigation');

  await openDraft(aId);
  const mutationStart = await application.evaluate(() => globalThis.__draftsQA.mutations.length);
  const firstCommittedTask = 'A first committed update with a deliberately lost IPC acknowledgment.';
  const newerTask = 'A newer input typed while its previous save acknowledgment is held.';
  await application.evaluate(() => { globalThis.__draftsQA.nextMutation = { type: 'update' }; });
  await taskInput().fill(firstCommittedTask);
  await until(() => application.evaluate(() => typeof globalThis.__draftsQA.releaseMutation === 'function'), 'Committed update acknowledgment was not held');
  const firstCommitted = await snapshot();
  assert.equal(savedDraft(firstCommitted, aId).content.task, firstCommittedTask);
  await taskInput().fill(newerTask);
  await application.evaluate(() => globalThis.__draftsQA.releaseMutation(true));
  await page.locator('.draft-save-error').waitFor();
  assert.equal(await taskInput().inputValue(), newerTask);
  await page.getByRole('button', { name: '重试草稿保存', exact: true }).click();
  await persisted(state => savedDraft(state, aId)?.content.task === newerTask, 'Retry did not persist the newer editor input');
  await until(async () => await page.locator('.draft-save-error').count() === 0, 'Successful retry did not clear the storage error');
  const retryMutations = await application.evaluate((_electron, offset) => globalThis.__draftsQA.mutations.slice(offset), mutationStart);
  assert.equal(retryMutations.length, 3);
  assert.deepEqual(retryMutations[1].input, retryMutations[0].input);
  assert.equal(retryMutations[1].revision, retryMutations[0].revision, 'Replaying a saved receipt must not rewrite the document');
  assert.notEqual(retryMutations[2].input.mutationId, retryMutations[0].input.mutationId);
  assert.equal(retryMutations[2].input.operation.content.task, newerTask);
  assert.equal((await snapshot()).drafts.length, beforeRestart.drafts.length);
  assert.deepEqual(savedDraft(await snapshot(), bId).content, expectedB);
  checks.push('lost-save-acknowledgment-retries-the-same-mutation-once-before-saving-newer-input');

  // A replay returns the CURRENT disk snapshot. If another writer activated B
  // after A committed, that receipt cannot move A's newer text into B.
  const ownerFirstTask = 'A committed before another native writer activates draft B.';
  const ownerNewerTask = 'Newer input still belongs to draft A, never to draft B.';
  await application.evaluate(() => { globalThis.__draftsQA.nextMutation = { type: 'update' }; });
  await taskInput().fill(ownerFirstTask);
  await until(() => application.evaluate(() => typeof globalThis.__draftsQA.releaseMutation === 'function'), 'Owner-change acknowledgment was not held');
  const ownerBase = await snapshot();
  const ownerChanged = await mutate(nativeInput(ownerBase, { type: 'activate', id: bId }));
  await taskInput().fill(ownerNewerTask);
  await application.evaluate(() => globalThis.__draftsQA.releaseMutation(true));
  await page.locator('.draft-save-error').waitFor();
  await page.getByRole('button', { name: '重试草稿保存', exact: true }).click();
  await page.getByText('本机保存的草稿已变化，当前编辑仍保留。可以将当前内容另存为新草稿。', { exact: true }).waitFor();
  assert.equal(await taskInput().inputValue(), ownerNewerTask);
  assert.deepEqual(await snapshot(), ownerChanged);
  assert.deepEqual(savedDraft(await snapshot(), bId).content, expectedB);
  assert.equal(savedDraft(await snapshot(), aId).content.task, ownerFirstTask);
  await mutate(nativeInput(await snapshot(), { type: 'activate', id: aId }));
  await page.getByRole('button', { name: '重试草稿保存', exact: true }).click();
  await persisted(state => savedDraft(state, aId)?.content.task === ownerNewerTask, 'Returning the matching owner did not allow safe retry');
  assert.deepEqual(savedDraft(await snapshot(), bId).content, expectedB);
  await taskInput().fill(legacy.task);
  await persisted(state => isDeepStrictEqual(savedDraft(state, aId).content, legacy), 'Original A fixture was not restored after acknowledgment races');
  checks.push('replayed-save-receipt-with-another-active-draft-never-transfers-editor-ownership-or-input');

  await openDraft(bId);

  // Retain the old native picker, then switch ownership before it can resolve.
  await application.evaluate(({ dialog }, directory) => {
    dialog.showOpenDialog = () => new Promise(resolve => {
      globalThis.__draftsQA.releasePicker = () => { globalThis.__draftsQA.pickerReleased = true; resolve({ canceled: false, filePaths: [directory] }); };
    });
  }, datasetA);
  await page.locator('.dataset-button').click();
  await until(() => application.evaluate(() => typeof globalThis.__draftsQA.releasePicker === 'function'), 'Native picker was not held');
  await openDraft(aId);
  const beforeLatePicker = await snapshot();
  await application.evaluate(() => globalThis.__draftsQA.releasePicker());
  await until(() => application.evaluate(() => globalThis.__draftsQA.pickerReleased === true), 'Held picker did not resolve');
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  assert.deepEqual(await snapshot(), beforeLatePicker);
  await assertForm(legacy);
  checks.push('late-native-picker-cannot-change-another-draft-after-switch');

  // A native close within the debounce window must synchronously flush, without
  // prompting users who have no unsaved settings or storage failure.
  const immediateTask = `${legacy.task}\nLast edit immediately before native close.`;
  const dialogsBeforeClose = await application.evaluate(() => globalThis.__draftsQA.dialogs.length);
  await taskInput().fill(immediateTask);
  assert.notEqual((await diskDocument()).drafts.find(item => item.id === aId).content.task, immediateTask, 'The immediate-close fixture missed the debounce interval');
  const closed = page.waitForEvent('close');
  await application.evaluate(({ BrowserWindow }) => { setTimeout(() => BrowserWindow.getAllWindows()[0].close(), 0); });
  await closed;
  assert.equal((await diskDocument()).drafts.find(item => item.id === aId).content.task, immediateTask);
  assert.equal(await application.evaluate(() => globalThis.__draftsQA.dialogs.length), dialogsBeforeClose);
  await reopenWindow();
  await taskInput().waitFor();
  await assertForm({ ...legacy, task: immediateTask });
  checks.push('immediate-native-close-flushes-last-edit-and-reopens-without-an-unnecessary-prompt');

  // Real native compare-and-swap conflict: a second writer renames the saved
  // record, then the visible editor tries to persist its older snapshot.
  const concurrentBase = await snapshot();
  const concurrent = await mutate(nativeInput(concurrentBase, { type: 'rename', id: aId, title: '另一处保存的标题' }));
  const conflictTask = 'Unsaved conflict input; must survive save refusal and close cancellation.';
  await taskInput().fill(conflictTask);
  await page.getByText('本机保存的草稿已变化，当前编辑仍保留。可以将当前内容另存为新草稿。', { exact: true }).waitFor();
  assert.equal(await taskInput().inputValue(), conflictTask);
  assert.deepEqual(await snapshot(), concurrent);
  const stale = nativeInput(concurrentBase, { type: 'update', id: aId, content: { ...legacy, task: 'stale native overwrite' } });
  assert.deepEqual(await flush(stale), { ok: false, code: 'conflict' });
  assert.deepEqual(await snapshot(), concurrent);
  const unsavedDialog = await guardedClose();
  assert(unsavedDialog.buttons.some(label => /放弃|Discard/.test(label)));
  assert.equal(await taskInput().inputValue(), conflictTask);
  const recoveryStart = await application.evaluate(() => globalThis.__draftsQA.mutations.length);
  await application.evaluate(() => { globalThis.__draftsQA.nextMutation = { type: 'create' }; });
  await page.getByRole('button', { name: /另存为新草稿/ }).click();
  await until(() => application.evaluate(() => typeof globalThis.__draftsQA.releaseMutation === 'function'), 'Recovery creation acknowledgment was not held');
  const committedRecovery = await snapshot();
  assert.equal(committedRecovery.drafts.length, concurrent.drafts.length + 1);
  assert.equal(savedDraft(committedRecovery).content.task, conflictTask);
  await application.evaluate(() => globalThis.__draftsQA.releaseMutation(true));
  await until(async () => await page.getByRole('button', { name: '另存为新草稿', exact: true }).isEnabled(), 'Lost recovery acknowledgment did not release the editor');
  assert.equal(await taskInput().inputValue(), conflictTask);
  const newerRecoveryTask = `${conflictTask}\nA newer edit after the recovery acknowledgment was lost.`;
  await taskInput().fill(newerRecoveryTask);
  await page.getByRole('button', { name: '另存为新草稿', exact: true }).click();
  const recovered = await persisted(state => state.drafts.length === concurrent.drafts.length + 1 && savedDraft(state)?.content.task === newerRecoveryTask, 'Recovery retry did not preserve newer edits in exactly one copy');
  const recoveredId = recovered.activeId;
  assert.equal(recoveredId, committedRecovery.activeId);
  assert.deepEqual(savedDraft(recovered, aId), savedDraft(concurrent, aId));
  assert.equal(savedDraft(recovered).title, '恢复的草稿');
  const recoveryMutations = await application.evaluate((_electron, offset) => globalThis.__draftsQA.mutations.slice(offset).filter(item => item.input.operation.type === 'create'), recoveryStart);
  assert.equal(recoveryMutations.length, 2);
  assert.deepEqual(recoveryMutations[0].input, recoveryMutations[1].input);
  assert.equal(recoveryMutations[0].revision, recoveryMutations[1].revision);
  assert.equal(await taskInput().inputValue(), newerRecoveryTask);
  checks.push('native-revision-conflict-preserves-saved-record-and-unsaved-input-with-explicit-save-as-copy');
  checks.push('lost-recovery-create-acknowledgment-replays-one-copy-and-preserves-newer-editor-input');

  const validBytes = await readFile(draftFile);
  const damaged = Buffer.from('{"schema":1,"drafts":[');
  await writeFile(draftFile, damaged);
  await assert.rejects(snapshot(), /\[drafts:corrupt\]/);
  await assert.rejects(mutate(nativeInput(recovered, { type: 'rename', id: recoveredId, title: 'must not overwrite damage' })), /\[drafts:corrupt\]/);
  const corruptTask = 'Current text while the native draft file is damaged. Preserve this input.';
  await taskInput().fill(corruptTask);
  await page.getByText('草稿文件无法读取，原文件未被覆盖。请保留当前内容后检查本机存储。', { exact: true }).waitFor();
  assert.deepEqual(await readFile(draftFile), damaged);
  assert.equal(await taskInput().inputValue(), corruptTask);
  await guardedClose();
  assert.deepEqual(await readFile(draftFile), damaged);

  // A pending model-settings save must dominate the failed draft flush. The
  // fixture intercepts only this synthetic save, without contacting a provider.
  await page.locator('.sidebar-settings').click();
  await page.getByLabel('API 地址', { exact: false }).fill('http://127.0.0.1:9/v1');
  await page.getByLabel('API Key', { exact: false }).first().fill('qa-drafts-synthetic-never-sent');
  await page.getByLabel('默认模型', { exact: true }).fill('qa-held-settings-model');
  await application.evaluate(() => { globalThis.__draftsQA.holdSettings = true; });
  await page.getByRole('button', { name: '保存配置', exact: true }).click();
  await until(() => application.evaluate(() => typeof globalThis.__draftsQA.releaseSettings === 'function'), 'Settings save fixture was not held');
  const combinedDialog = await guardedClose(true);
  assert(!combinedDialog.buttons.some(label => /放弃|丢弃|Discard/.test(label)));
  assert.deepEqual(await readFile(draftFile), damaged);
  await application.evaluate(() => globalThis.__draftsQA.releaseSettings());
  await page.getByText('Synthetic held settings failure', { exact: false }).waitFor();
  const combinedFailed = await guardedClose();
  assert(combinedFailed.buttons.some(label => /放弃|Discard/.test(label)));
  assert.equal(await page.getByLabel('默认模型', { exact: true }).inputValue(), 'qa-held-settings-model');
  assert.deepEqual(await readFile(draftFile), damaged);
  checks.push('damaged-native-store-refuses-overwrite-retains-input-and-combined-settings-save-keeps-close-protection');

  // Restore only this QA fixture, then discard the unsaved synthetic settings.
  // The draft callback now flushes its retained text before the window closes.
  await writeFile(draftFile, validBytes);
  await application.evaluate(() => { globalThis.__draftsQA.discardNext = true; });
  const dismissed = page.waitForEvent('close');
  await application.evaluate(({ BrowserWindow }) => { setTimeout(() => BrowserWindow.getAllWindows()[0].close(), 0); });
  await dismissed;
  assert.equal((await diskDocument()).drafts.find(item => item.id === recoveredId).content.task, corruptTask);
  await reopenWindow();
  await taskInput().waitFor();
  assert.equal(await taskInput().inputValue(), corruptTask);
  assert.equal((await api('/api/env')).values.OPENAI_API_KEY.configured, false);
  const final = await snapshot();
  assert.equal(final.drafts.length, 4);
  assert.deepEqual(savedDraft(final, bId).content, expectedB);
  assert.equal(final.legacyMigration.sha256, migrated.legacyMigration.sha256);
  for (const [name, contents] of Object.entries(historyFiles)) assert.equal(await readFile(path.join(history, name), 'utf8'), contents);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  assert.deepEqual((await api('/api/runs')).map(run => run.id), [historyId]);
  await library();
  await page.screenshot({ path: path.join(output, '01-native-draft-library.png') });
  await openDraft(recoveredId);
  await page.screenshot({ path: path.join(output, '02-recovered-draft.png') });
  await closeApp();
  assert.deepEqual(forbiddenRequests, []);
  assert.deepEqual(rendererErrors, []);
  const report = { passed: true, mode: 'Real Electron, native draft IPC and native close guard; synthetic isolated profile and one intercepted settings failure; no experiment or provider calls', executablePath: executablePath || 'development Electron', profile, checks, draftCount: final.drafts.length, forbiddenRequests, rendererErrors };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-3500));
  }
  throw error;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => {
      globalThis.__draftsQA?.releaseSettings?.();
      globalThis.__draftsQA?.releasePicker?.();
      globalThis.__draftsQA?.releaseInspection?.();
      globalThis.__draftsQA?.releaseMutation?.();
      for (const window of BrowserWindow.getAllWindows()) window.destroy();
    }).catch(() => {});
    await application.close();
  }
}
