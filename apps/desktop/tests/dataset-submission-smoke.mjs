// Real Electron, directory metadata inspection, and sidecar restarts. Engine
// readiness, provider configuration, and ALL submission responses are synthetic.
// POST /api/runs is never forwarded; no experiment, model, or installation runs.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, realpath, rename, unlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-dataset-submission-'));
const profile = path.join(temporary, 'profile');
const dataset = path.join(temporary, 'dataset');
const empty = path.join(temporary, 'empty');
const alternate = path.join(temporary, 'late-picker-target');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/dataset-submission-0.1.9'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
for (const directory of [profile, dataset, empty, alternate, output]) await mkdir(directory, { recursive: true });
await writeFile(path.join(dataset, 'sample.csv'), 'synthetic metadata fixture,not scientific data\n');
await writeFile(path.join(alternate, 'sample.csv'), 'synthetic alternate folder\n');
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const checks = [];
const errors = [];
const restarts = [];
const task = '  Synthetic submission race test.\nPreserve this exact draft; no actual experiment may run.  ';
const nextTask = '  A new draft after a simulated submission.\nDo not let the older picker replace this dataset.  ';
let application;
let page;
const api = (route) => page.evaluate(route => window.biomaker.request(route), route);
const state = () => page.evaluate(() => window.biomaker.getServiceState());
const storedDraft = () => page.evaluate(() => localStorage.getItem('biomaker.draft'));
const submitButton = () => page.locator('.submit-button');
const taskInput = () => page.getByLabel('实验任务', { exact: true });
const renderFrame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const postCount = () => application.evaluate(() => globalThis.__submissionQA.posts.length);
async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 40));
  }
}
async function ready(code = 'ok') {
  await until(async () => await page.locator('.dataset-overview').getAttribute('data-code') === code &&
    await page.locator('.dataset-overview').getAttribute('data-state') !== 'checking', `Dataset inspection did not settle as ${code}`);
}
async function choose(directory, code = 'ok') {
  await application.evaluate(({ dialog }, directory) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [directory] }); }, directory);
  await page.locator('.dataset-button').click();
  const canonical = await realpath(directory);
  await until(async () => JSON.parse(await storedDraft() || '{}').dataset === canonical, 'The picker selection was not persisted');
  await ready(code);
}
async function refreshInspection() {
  await page.getByRole('button', { name: '重新检查数据集', exact: true }).click();
  await ready();
}
async function assertDraft(raw) {
  await renderFrame();
  assert.equal(await storedDraft(), raw, 'The exact persisted draft must be preserved');
  assert.equal(await taskInput().inputValue(), JSON.parse(raw).task);
}
async function restartSidecar() {
  const before = await state();
  const canonicalProfile = await realpath(profile);
  const children = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n')
    .filter(line => line.includes('/service/main.py') && (line.includes(profile) || line.includes(canonicalProfile)) && line.includes('python'));
  assert.equal(children.length, 1, 'Exactly one real sidecar must belong to this isolated profile');
  process.kill(Number(children[0].trim().split(/\s+/)[0]), 'SIGKILL');
  await until(async () => { const value = await state(); return value.status === 'ready' && value.revision > before.revision; }, 'The real sidecar did not recover with a newer generation', 60_000);
  await ready();
  restarts.push({ from: before.revision, to: (await state()).revision });
}
async function screenshot(name) {
  await page.locator('.toast').waitFor({ state: 'hidden' });
  await page.locator('.experiment-form .notice').last().scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, name) });
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => errors.push(error.message));
  await until(async () => { try { return (await state()).status === 'ready'; } catch { return false; } }, 'The isolated service did not become ready');
  await page.locator('.sidebar-settings').waitFor();
  const actual = { status: await api('/api/desktop/status'), env: await api('/api/env'), runs: await api('/api/runs') };
  assert.equal(actual.status.engine.ready, false);
  assert.equal(actual.env.values.OPENAI_API_KEY.configured, false);
  assert.deepEqual(actual.runs, []);
  await application.evaluate((_electron, actual) => {
    const originalFetch = globalThis.fetch;
    const provider = structuredClone(actual.env);
    provider.values.OPENAI_API_KEY.configured = true;
    provider.values.BIOMAKER_MAIN_MODEL = { configured: true, value: 'qa-synthetic-never-called-model' };
    const engine = { ...actual.status.engine, ready: true, installing: false, checking: false, operation: null, busy_external: false, maintenance_unavailable: false };
    const f = globalThis.__submissionQA = { originalFetch, posts: [], inspections: [], forbidden: [], syntheticRuns: [], nextPost: null, holdEnv: false };
    const json = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
    globalThis.fetch = async (input, options) => {
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      if (url.hostname !== '127.0.0.1') { f.forbidden.push({ path: url.pathname, method }); throw new Error('External requests are forbidden in submission QA'); }
      if (url.pathname === '/api/desktop/status' && method === 'GET') return json({ ...actual.status, engine });
      if (url.pathname === '/api/env' && method === 'GET') {
        if (f.holdEnv) {
          f.holdEnv = false;
          await new Promise(resolve => { f.releaseEnv = () => { delete f.releaseEnv; resolve(); }; });
        }
        return json(provider);
      }
      if (url.pathname === '/api/desktop/dataset/inspect' && method === 'POST') {
        const operation = { path: JSON.parse(options.body).path, done: false };
        f.inspections.push(operation);
        const response = await originalFetch(input, options);
        operation.done = true;
        return response;
      }
      if (url.pathname === '/api/runs' && method === 'POST') {
        const plan = f.nextPost;
        f.nextPost = null;
        const operation = { body: JSON.parse(options.body), simulated: true, done: false };
        f.posts.push(operation);
        if (!plan) { f.forbidden.push({ path: url.pathname, method }); throw new Error('Unplanned simulated submission'); }
        if (plan.hold) await new Promise(resolve => { f.releasePost = () => { delete f.releasePost; resolve(); }; });
        operation.done = true;
        if (plan.code) return json({ detail: `[dataset:${plan.code}] Synthetic pre-enqueue directory refusal` }, 422);
        const run = { id: `qa-simulated-${f.posts.length}`, status: 'queued', detailed_status: 'queued', managed: true,
          requested_exp_name: operation.body.requested_exp_name || 'Synthetic submission response',
          dataset_path: operation.body.experiment.dataset_path, project_snapshot: operation.body.experiment,
          options: operation.body.experiment.advanced_defaults, timestamps: { created_at: new Date().toISOString() } };
        f.syntheticRuns.push(run);
        return json(run, 202);
      }
      if (!['GET', 'HEAD'].includes(method)) { f.forbidden.push({ path: url.pathname, method }); throw new Error('No provider, runtime, settings, or real experiment writes are allowed'); }
      if (url.pathname === '/api/runs') return json(f.syntheticRuns);
      const simulated = /^\/api\/runs\/(qa-simulated-\d+)(?:\/(.*))?$/.exec(url.pathname);
      if (simulated) {
        const run = f.syntheticRuns.find(run => run.id === simulated[1]);
        if (!simulated[2]) return json(run);
        if (simulated[2] === 'revisions') return json({ run_revision: 'synthetic', events: { revision: 'synthetic' }, artifacts: { revision: 'synthetic' }, console: { revision: 'synthetic' } });
        if (simulated[2] === 'console') return json({ text: '' });
        return json([]);
      }
      return originalFetch(input, options);
    };
  }, actual);
  await page.reload();
  await taskInput().fill(task);
  await page.getByRole('button', { name: '实验设置', exact: true }).click();
  await page.getByLabel('实验名称', { exact: false }).fill('qa-simulated-submit');
  await page.getByLabel('最大探索步数', { exact: true }).fill('7');

  await choose(empty, 'empty');
  assert.equal(await submitButton().isDisabled(), true);
  await page.locator('.experiment-form').evaluate(form => form.requestSubmit());
  await renderFrame();
  assert.equal(await postCount(), 0);
  assert.match(await page.locator('.experiment-form .notice').last().innerText(), /未发现普通文件/);
  checks.push('real-confirmed-empty-directory-blocks-submit-even-through-form-requestSubmit');

  await choose(dataset);
  let draftRaw = await storedDraft();
  for (const code of ['empty', 'not_found']) {
    const beforeScans = await application.evaluate(() => globalThis.__submissionQA.inspections.length);
    if (code === 'empty') await unlink(path.join(dataset, 'sample.csv'));
    else await rename(dataset, `${dataset}-moved`);
    await application.evaluate((_electron, code) => { globalThis.__submissionQA.nextPost = { code }; }, code);
    await submitButton().click();
    await page.locator('.experiment-form .notice').filter({ hasText: '实验尚未提交，草稿已保留' }).waitFor();
    await ready(code);
    assert.match(await page.locator('.experiment-form .notice').last().innerText(), code === 'empty' ? /未发现普通文件/ : /目录已不存在/);
    await assertDraft(draftRaw);
    assert((await application.evaluate(() => globalThis.__submissionQA.inspections.length)) > beforeScans);
    if (code === 'empty') await writeFile(path.join(dataset, 'sample.csv'), 'restored synthetic metadata fixture\n');
    else await rename(`${dataset}-moved`, dataset);
    await refreshInspection();
  }
  checks.push('simulated-admission-refusals-localize-empty-and-missing-keep-draft-and-reinspect-real-folder');

  const beforePreflight = await postCount();
  await application.evaluate(() => { globalThis.__submissionQA.holdEnv = true; });
  await submitButton().click();
  await until(() => application.evaluate(() => typeof globalThis.__submissionQA.releaseEnv === 'function'), 'The synthetic preflight response was not held');
  await restartSidecar();
  await application.evaluate(() => globalThis.__submissionQA.releaseEnv());
  await page.locator('.experiment-form .notice').filter({ hasText: '尚未提交实验' }).waitFor();
  await until(() => submitButton().isEnabled(), 'The submit lock did not release');
  assert.equal(await postCount(), beforePreflight);
  await assertDraft(draftRaw);
  checks.push('held-preflight-across-real-service-restart-never-dispatches-submission');
  await screenshot('01-preflight-generation-change.png');

  // Hold a native picker before starting. It must not overwrite a cleared or
  // subsequently edited draft when its old selection finally comes back.
  await application.evaluate(({ dialog }, directory) => {
    dialog.showOpenDialog = () => new Promise(resolve => {
      globalThis.__submissionQA.releasePicker = () => { delete globalThis.__submissionQA.releasePicker; globalThis.__submissionQA.pickerReleased = true; resolve({ canceled: false, filePaths: [directory] }); };
    });
  }, alternate);
  await page.locator('.dataset-button').click();
  await until(() => application.evaluate(() => typeof globalThis.__submissionQA.releasePicker === 'function'), 'The native picker was not held');
  const beforeLocked = await postCount();
  await application.evaluate(() => { globalThis.__submissionQA.nextPost = { hold: true }; });
  await submitButton().evaluate(button => { button.click(); button.click(); button.form.requestSubmit(); });
  await until(() => application.evaluate(() => typeof globalThis.__submissionQA.releasePost === 'function'), 'The simulated POST response was not held');
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Enter' : 'Control+Enter');
  await application.evaluate(({ BrowserWindow }) => {
    BrowserWindow.getAllWindows()[0].webContents.send('biomaker:menu-action', 'new-experiment');
    BrowserWindow.getAllWindows()[0].webContents.send('biomaker:menu-action', 'settings');
  });
  await renderFrame();
  assert.equal(await postCount(), beforeLocked + 1);
  for (const control of [taskInput(), page.locator('.dataset-button'), page.getByLabel('实验名称', { exact: false }), page.getByLabel('最大探索步数', { exact: true }), page.locator('.sidebar-settings')]) assert.equal(await control.isDisabled(), true);
  assert.equal(await page.getByRole('dialog', { name: '设置', exact: true }).count(), 0);
  const sent = await application.evaluate(() => globalThis.__submissionQA.posts.at(-1).body);
  const snapshot = JSON.parse(draftRaw);
  assert.deepEqual(sent, { experiment: { dataset_path: snapshot.dataset, task_template_path: snapshot.template, task_body: snapshot.task.trim(), advanced_defaults: snapshot.options }, requested_exp_name: snapshot.name });
  await assertDraft(draftRaw);
  await application.evaluate(() => globalThis.__submissionQA.releasePost());
  await until(async () => JSON.parse(await storedDraft() || '{}').task === '', 'A successful simulated submission did not clear its own draft');
  await page.locator('.primary-nav').getByRole('button', { name: /新建实验/ }).click();
  await taskInput().fill(nextTask);
  await ready();
  await until(async () => JSON.parse(await storedDraft() || '{}').task === nextTask, 'The next draft was not saved');
  draftRaw = await storedDraft();
  await application.evaluate(() => globalThis.__submissionQA.releasePicker());
  await renderFrame();
  await assertDraft(draftRaw);
  assert.equal(JSON.parse(draftRaw).dataset, await realpath(dataset));
  assert.equal(await page.locator('.dataset-button').getAttribute('title'), await realpath(dataset));
  checks.push('submission-lock-captures-exact-snapshot-blocks-controls-menus-and-duplicate-submit');
  checks.push('late-native-picker-cannot-overwrite-the-new-draft-after-simulated-success');

  const beforeUncertain = await postCount();
  await application.evaluate(() => { globalThis.__submissionQA.nextPost = { hold: true }; });
  await submitButton().click();
  await until(() => application.evaluate(() => typeof globalThis.__submissionQA.releasePost === 'function'), 'The second simulated POST was not held');
  await restartSidecar();
  await application.evaluate(() => globalThis.__submissionQA.releasePost());
  await page.locator('.experiment-form .notice').filter({ hasText: '实验可能已加入队列' }).waitFor();
  await until(() => submitButton().isEnabled(), 'The ambiguous POST did not release the submit lock');
  await assertDraft(draftRaw);
  await renderFrame();
  assert.equal(await postCount(), beforeUncertain + 1);
  await screenshot('02-post-generation-change-preserves-draft.png');
  checks.push('held-simulated-post-across-real-service-restart-preserves-draft-and-reports-uncertainty-without-retry');

  const fixture = await application.evaluate(() => {
    const f = globalThis.__submissionQA;
    globalThis.fetch = f.originalFetch;
    return { posts: f.posts, inspections: f.inspections, forbidden: f.forbidden };
  });
  assert.deepEqual(fixture.forbidden, []);
  assert(fixture.posts.every(post => post.simulated && post.done));
  assert.deepEqual(await api('/api/runs'), []);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  assert.equal((await api('/api/env')).values.OPENAI_API_KEY.configured, false);
  const config = await readFile(path.join(profile, 'config.env'), 'utf8').catch(error => error.code === 'ENOENT' ? '' : Promise.reject(error));
  assert.equal(config, '');
  assert.deepEqual(errors, []);
  checks.push('real-ledger-remains-empty-with-no-runtime-provider-or-settings-write-and-no-renderer-errors');
  const report = { passed: true, mode: 'Real Electron, metadata scans and sidecar restarts; explicitly synthetic readiness, provider configuration and intercepted submission responses. No actual experiment acceptance.', executablePath: executablePath || 'development Electron', profile, checks, realServiceRestarts: restarts, simulatedSubmissions: fixture.posts, realInspectionCount: fixture.inspections.length, forbiddenRequests: fixture.forbidden, rendererErrors: errors };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-6500));
  }
  throw error;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => { for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await application.close();
  }
}
