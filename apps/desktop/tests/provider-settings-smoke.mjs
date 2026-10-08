// Real Electron/native IPC/portable sidecar acceptance for provider settings.
// Every save and CAS conflict reaches the real backend. Only fully received
// responses are delayed for stale-generation checks. Synthetic profile and
// loopback provider only: no user configuration, external API, install or run.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdir, mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-provider-settings-'));
const profile = path.join(temporary, 'profile');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/provider-settings-0.1.10'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
for (const folder of [profile, output]) await mkdir(folder, { recursive: true });
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1', NO_PROXY: '127.0.0.1,localhost' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const secrets = Object.fromEntries(['keyA', 'keyB', 'serperA', 'serperB', 'jinaA', 'jinaB'].map(name => [name, `qa-private-settings-${name}-never-external`]));
const privateTask = '  Private provider-settings draft; never send this to a provider.  ';
const checks = [];
const rendererErrors = [];
const fixtureViolations = [];
const providerRequests = [];
const restarts = [];
let application;
let page;
let draft;
const sanitize = value => Object.values(secrets).reduce((text, secret) => text.split(secret).join('[synthetic-secret-redacted]'), String(value));
const equalPrivate = (actual, expected, message) => assert(actual === expected, message);
async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 30));
  }
}
const server = createServer(async (request, response) => {
  try {
    let raw = '';
    for await (const chunk of request) raw += chunk;
    const body = raw ? JSON.parse(raw) : {};
    const correct = request.url === '/b/v1/chat/completions' && request.method === 'POST' && request.headers.authorization === `Bearer ${secrets.keyB}`;
    providerRequests.push({ path: request.url, method: request.method, expectedProviderAndKey: correct, fixedPrompt: JSON.stringify(body.messages) === JSON.stringify([{ role: 'user', content: 'Reply with exactly OK.' }]), completionBudget: body.max_completion_tokens, model: body.model });
    if (!correct) fixtureViolations.push('Unexpected provider or authorization');
    if ([privateTask, profile, ...Object.values(secrets)].some(value => raw.includes(value))) fixtureViolations.push('Provider body included private data');
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ id: 'qa-settings-probe', object: 'chat.completion', created: 1, model: body.model, choices: [{ index: 0, message: { role: 'assistant', content: 'OK' }, finish_reason: 'stop' }], usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 } }));
  } catch {
    fixtureViolations.push('Loopback fixture failed');
    if (!response.headersSent) response.writeHead(500, { 'Content-Type': 'application/json' });
    response.end('{"error":{"message":"Fixture failed"}}');
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const baseA = `${origin}/a/v1`;
const baseB = `${origin}/b/v1`;
const api = (pathname, method = 'GET', body) => page.evaluate(([pathname, method, body]) => window.biomaker.request(pathname, { method, ...(body === undefined ? {} : { body }) }), [pathname, method, body]);
const config = () => readFile(path.join(profile, 'config.env'), 'utf8').catch(error => error.code === 'ENOENT' ? '' : Promise.reject(error));
const state = () => page.evaluate(() => window.biomaker.getServiceState());
const baseInput = () => page.getByRole('textbox', { name: /^API 地址/ });
const keyInput = () => page.getByLabel('API Key', { exact: false }).first();
const modelInput = () => page.getByLabel('默认模型', { exact: true });
const serperInput = () => page.getByLabel('Serper API Key', { exact: false });
const jinaInput = () => page.getByLabel('Jina API Key', { exact: false });
const saveButton = () => page.getByRole('button', { name: '保存配置', exact: true });
const reviewPanel = () => page.locator('.provider-settings-review');
const readLatest = () => page.getByRole('button', { name: '读取最新已保存配置', exact: true });
const keepInputs = () => page.getByRole('button', { name: '保留输入并继续编辑', exact: true });
const loadSaved = () => page.getByRole('button', { name: '载入已保存配置并替换输入', exact: true });
const renderFrame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const operationCount = () => application.evaluate(() => globalThis.__settingsQA.operations.length);
const putCount = () => application.evaluate(() => globalThis.__settingsQA.operations.filter(item => item.method === 'PUT').length);
async function waitOperation(index) {
  await until(() => application.evaluate((_electron, index) => globalThis.__settingsQA.operations[index]?.done, index), 'The real settings response did not settle');
  await renderFrame();
}
async function allowSave(allowed) { await application.evaluate((_electron, value) => { globalThis.__settingsQA.allowSave = value; }, allowed); }
async function mutate(values) {
  await allowSave(true);
  try { return await api('/api/env', 'PUT', values); }
  finally { await allowSave(false); }
}
async function externalModel(model) {
  const latest = await api('/api/env');
  assert.match(latest.settings_revision, /^[a-f0-9]{64}$/);
  return mutate({ BIOMAKER_MAIN_MODEL: model, expected_revision: latest.settings_revision });
}
async function optionalFields() {
  const toggle = page.getByRole('button', { name: '可选的搜索服务', exact: true });
  if (await toggle.getAttribute('aria-expanded') !== 'true') await toggle.click();
}
async function fillInputs(values) {
  if ('base' in values) await baseInput().fill(values.base);
  if ('key' in values) await keyInput().fill(values.key);
  if ('model' in values) await modelInput().fill(values.model);
  if ('serper' in values || 'jina' in values) await optionalFields();
  if ('serper' in values) await serperInput().fill(values.serper);
  if ('jina' in values) await jinaInput().fill(values.jina);
}
async function assertInputs(expected) {
  for (const [name, input] of [['base', baseInput], ['key', keyInput], ['model', modelInput], ['serper', serperInput], ['jina', jinaInput]]) {
    if (name in expected) equalPrivate(await input().inputValue(), expected[name], `${name} input was not preserved`);
  }
}
async function saveUI(expectConflict = false) {
  const before = await putCount();
  await allowSave(true);
  try {
    await saveButton().click();
    await until(async () => await putCount() === before + 1, 'Save did not reach the real backend');
    if (expectConflict) await until(async () => await reviewPanel().getAttribute('data-state') === 'required', 'Conflict did not require an explicit review');
    else await page.getByText('所有更改已保存', { exact: true }).waitFor();
  } finally { await allowSave(false); }
}
async function review() {
  await readLatest().click();
  await until(async () => await reviewPanel().getAttribute('data-state') === 'loaded', 'The latest saved settings were not presented for review');
}
async function noSecretVisible() {
  const text = await page.locator('body').innerText();
  assert(!Object.values(secrets).some(secret => text.includes(secret)), 'A synthetic secret appeared in visible UI');
  for (const input of [keyInput(), serperInput(), jinaInput()]) if (await input.count()) assert.equal(await input.getAttribute('type'), 'password');
  const stored = await page.evaluate(() => JSON.stringify(localStorage));
  assert(!Object.values(secrets).some(secret => stored.includes(secret)), 'A synthetic secret entered browser storage');
  equalPrivate(await page.evaluate(() => localStorage.getItem('biomaker.draft')), draft, 'The experiment draft changed');
}
async function snapshot(filename, locator = reviewPanel()) {
  await page.locator('.toast').waitFor({ state: 'hidden' });
  if (await locator.count()) await locator.scrollIntoViewIfNeeded();
  await noSecretVisible();
  await page.screenshot({ path: path.join(output, filename) });
}
async function holdNext(method) {
  await application.evaluate((_electron, method) => { globalThis.__settingsQA.holdNext = method; }, method);
}
async function held() {
  await until(() => application.evaluate(() => typeof globalThis.__settingsQA.releaseHeld === 'function'), 'A fully buffered real settings response was not held');
}
async function releaseHeld() {
  const index = await application.evaluate(() => globalThis.__settingsQA.heldIndex);
  await application.evaluate(() => globalThis.__settingsQA.releaseHeld());
  await waitOperation(index);
}
async function restartSidecar() {
  const before = await state();
  const runtimes = (await readdir(path.join(profile, 'runtimes'))).filter(name => !name.includes('.staging-'));
  assert.equal(runtimes.length, 1);
  const serviceFile = path.join(profile, 'runtimes', runtimes[0], 'service/main.py');
  const children = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes(serviceFile) && line.includes('python'));
  assert.equal(children.length, 1, 'Only the exact isolated sidecar may be stopped');
  process.kill(Number(children[0].trim().split(/\s+/)[0]), 'SIGKILL');
  await until(async () => { const next = await state(); return next.status === 'ready' && next.revision > before.revision; }, 'The isolated sidecar did not recover with a newer revision', 60_000);
  restarts.push({ previousRevision: before.revision, recoveredRevision: (await state()).revision });
}
async function nativeCloseCancel() {
  const before = await application.evaluate(() => globalThis.__settingsQA.closeDialogs);
  await application.evaluate(({ BrowserWindow }) => { setTimeout(() => BrowserWindow.getAllWindows()[0].close(), 0); });
  await until(() => application.evaluate((_electron, before) => globalThis.__settingsQA.closeDialogs > before, before), 'Dirty settings did not guard native window close');
  assert.equal((await api('/api/health')).status, 'ok');
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(sanitize(error.message)));
  await until(async () => { try { return (await state()).status === 'ready'; } catch { return false; } }, 'Portable sidecar did not become ready');
  await page.locator('.sidebar-settings').waitFor();
  await application.evaluate(({ dialog }, bases) => {
    const originalFetch = globalThis.fetch;
    globalThis.__settingsQA = { operations: [], forbidden: [], allowSave: false, allowProbe: false, bases, closeDialogs: 0 };
    globalThis.fetch = async (input, options) => {
      const qa = globalThis.__settingsQA;
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      const settings = url.pathname === '/api/env';
      const provider = url.pathname === '/api/desktop/provider/check' && method === 'POST';
      const body = options?.body ? JSON.parse(options.body) : null;
      if (url.hostname !== '127.0.0.1' || (!['GET', 'HEAD'].includes(method) && !(settings && method === 'PUT' && qa.allowSave) && !(provider && qa.allowProbe && qa.bases.includes(body?.base_url)))) {
        qa.forbidden.push({ path: url.pathname, method });
        throw new Error('Settings QA forbids unplanned mutations and provider calls');
      }
      const operation = settings || provider ? { method, path: url.pathname, status: null, done: false } : null;
      const index = qa.operations.length;
      if (operation) qa.operations.push(operation);
      const shouldHold = settings && qa.holdNext === method;
      if (shouldHold) qa.holdNext = null;
      try {
        const response = await originalFetch(input, options);
        if (operation) operation.status = response.status;
        if (settings && method === 'PUT' && response.ok && qa.failNextCommittedSave) {
          qa.failNextCommittedSave = false;
          await response.arrayBuffer();
          throw new Error('Synthetic transport failure after a real committed settings save');
        }
        if (shouldHold) {
          const content = await response.arrayBuffer();
          const buffered = new Response(content, { status: response.status, headers: response.headers });
          qa.heldIndex = index;
          await new Promise(resolve => { qa.releaseHeld = () => { delete qa.releaseHeld; resolve(); }; });
          return buffered;
        }
        return response;
      } finally { if (operation) operation.done = true; }
    };
    const cancel = (...args) => { globalThis.__settingsQA.closeDialogs++; return args.at(-1).cancelId ?? 0; };
    dialog.showMessageBoxSync = (...args) => cancel(...args);
    dialog.showMessageBox = async (...args) => ({ response: cancel(...args), checkboxChecked: false });
  }, [baseA, baseB]);
  const initial = await api('/api/env');
  assert.match(initial.settings_revision, /^[a-f0-9]{64}$/);
  await page.getByLabel('实验任务', { exact: true }).fill(privateTask);
  await until(async () => JSON.parse(await page.evaluate(() => localStorage.getItem('biomaker.draft')) || '{}').task === privateTask, 'The synthetic task draft was not persisted');
  draft = await page.evaluate(() => localStorage.getItem('biomaker.draft'));
  await page.locator('.sidebar-settings').click();
  await until(() => baseInput().isEnabled(), 'The initial saved settings did not load');
  await fillInputs({ base: baseA, key: secrets.keyA, model: 'qa-model-a', serper: secrets.serperA, jina: secrets.jinaA });
  await saveUI();

  const seedBytes = await config();
  const beforeBlocked = await putCount();
  await baseInput().fill(baseB);
  assert.equal(await saveButton().isDisabled(), true);
  await saveButton().evaluate(element => element.click());
  await renderFrame();
  assert.equal(await putCount(), beforeBlocked);
  equalPrivate(await config(), seedBytes, 'Changing the address without a key mutated persisted settings');
  assert.equal(await keyInput().inputValue(), '');
  await snapshot('01-address-change-blocked-light.png', keyInput());
  checks.push('new-address-with-blank-key-blocks-ui-save-and-preserves-exact-file-bytes');

  await fillInputs({ key: secrets.keyB, model: 'qa-model-b' });
  await saveUI();
  let bytes = await config();
  assert(bytes.includes(secrets.keyB) && !bytes.includes(secrets.keyA), 'Explicit replacement was not stored exclusively');
  assert(bytes.includes(secrets.serperA) && bytes.includes(secrets.jinaA), 'Blank optional search keys were not preserved');
  assert.equal(providerRequests.length, 0, 'Saving settings must not contact a provider');
  await application.evaluate(() => { globalThis.__settingsQA.allowProbe = true; });
  await page.getByRole('button', { name: '测试模型响应', exact: true }).click();
  await page.locator('.provider-check-result[role="status"]').getByText(/^模型响应测试通过[。]?$/).waitFor();
  await application.evaluate(() => { globalThis.__settingsQA.allowProbe = false; });
  assert.equal(providerRequests.length, 1);
  assert.equal(providerRequests[0].expectedProviderAndKey, true);
  checks.push('explicit-replacement-saves-new-key-and-only-explicit-probe-sends-it-to-provider-b');

  await baseInput().fill(`${baseB}/`);
  assert.equal(await keyInput().inputValue(), '');
  await saveUI();
  bytes = await config();
  assert(bytes.includes(secrets.keyB), 'Equivalent normalized provider address did not retain its saved key');
  assert(bytes.includes(secrets.serperA) && bytes.includes(secrets.jinaA), 'Optional search keys changed during normalized-address save');
  checks.push('same-normalized-address-retains-key-and-optional-search-settings');

  await baseInput().fill(baseA);
  await page.getByRole('button', { name: '移除已保存的密钥', exact: true }).click();
  await saveUI();
  assert.equal((await api('/api/env')).values.OPENAI_API_KEY.configured, false);
  bytes = await config();
  assert(!bytes.includes(secrets.keyA) && !bytes.includes(secrets.keyB), 'Explicit key removal left a saved provider key');
  assert(bytes.includes(secrets.serperA) && bytes.includes(secrets.jinaA), 'Removing the provider key changed optional search settings');
  checks.push('explicit-removal-allows-address-switch-and-clears-only-provider-key');

  await fillInputs({ key: secrets.keyA, model: 'qa-model-a' });
  await saveUI();
  const typed = { base: baseB, key: secrets.keyB, model: 'qa-unsaved-model', serper: secrets.serperB, jina: secrets.jinaB };
  await fillInputs(typed);
  await externalModel('qa-concurrent-model');
  const conflictBytes = await config();
  await saveUI(true);
  await assertInputs(typed);
  equalPrivate(await config(), conflictBytes, 'A stale save changed the real config file');
  assert.equal(await saveButton().isDisabled(), true);
  await nativeCloseCancel();
  await assertInputs(typed);
  checks.push('real-backend-cas-conflict-preserves-all-five-inputs-file-bytes-and-native-close-guard');

  await holdNext('GET');
  await readLatest().click();
  await held();
  typed.model = 'qa-edited-during-settings-read';
  await modelInput().fill(typed.model);
  await modelInput().focus();
  await releaseHeld();
  await until(async () => await reviewPanel().getAttribute('data-state') === 'loaded', 'The current-service review did not load');
  await assertInputs(typed);
  assert(await modelInput().evaluate(element => document.activeElement === element), 'A completed settings read stole focus from an actively edited model field');
  checks.push('slow-current-settings-read-preserves-active-model-edit-and-input-focus');
  assert((await reviewPanel().innerText()).includes('qa-concurrent-model'));
  for (const field of ['base', 'model', 'api-key', 'serper-key', 'jina-key']) assert.equal(await reviewPanel().locator(`[data-field="${field}"]`).count(), 1);
  await snapshot('02-conflict-review-light.png');
  const beforeKeep = await putCount();
  await keepInputs().click();
  await assertInputs(typed);
  assert.equal(await putCount(), beforeKeep, 'Keeping inputs must not implicitly save');
  equalPrivate(await config(), conflictBytes, 'Reviewing and keeping inputs must not modify config');
  await saveUI();
  bytes = await config();
  assert(bytes.includes(secrets.keyB) && bytes.includes(secrets.serperB) && bytes.includes(secrets.jinaB), 'Separate save did not store the reviewed inputs');
  await assertInputs({ key: '', serper: '', jina: '' });
  checks.push('review-shows-redacted-latest-state-keep-adopts-baseline-and-requires-separate-save');

  await fillInputs({ model: 'qa-discarded-model', key: secrets.keyA, serper: secrets.serperA, jina: secrets.jinaA });
  await externalModel('qa-latest-discard-baseline');
  const discardBytes = await config();
  await saveUI(true);
  await review();
  const beforeDiscard = await putCount();
  await loadSaved().click();
  await assertInputs({ base: baseB, model: 'qa-latest-discard-baseline', key: '', serper: '', jina: '' });
  assert.equal(await putCount(), beforeDiscard);
  equalPrivate(await config(), discardBytes, 'Loading the saved baseline unexpectedly wrote settings');
  checks.push('explicit-load-discards-unsaved-inputs-without-another-save');

  const uncertainInputs = { base: baseB, key: secrets.keyB, model: 'qa-committed-uncertain-model', serper: secrets.serperA, jina: secrets.jinaA };
  await fillInputs(uncertainInputs);
  const beforeUncertain = await putCount();
  await application.evaluate(() => { globalThis.__settingsQA.failNextCommittedSave = true; });
  await saveUI(true);
  await assertInputs(uncertainInputs);
  assert.equal(await saveButton().isDisabled(), true);
  const uncertainBytes = await config();
  assert(uncertainBytes.includes('qa-committed-uncertain-model'), 'The uncertain-response fixture must follow an actual committed save');
  assert.equal(await putCount(), beforeUncertain + 1, 'An uncertain save must not be replayed automatically');
  await review();
  assert((await reviewPanel().innerText()).includes('qa-committed-uncertain-model'), 'Explicit review did not show the actual committed settings');
  await loadSaved().click();
  await assertInputs({ base: baseB, model: 'qa-committed-uncertain-model', key: '', serper: '', jina: '' });
  equalPrivate(await config(), uncertainBytes, 'Resolving an uncertain result by loading settings must not write again');
  checks.push('real-committed-save-with-transport-failure-requires-review-and-explicit-resolution');

  const removalInputs = { base: baseA, key: secrets.keyA, model: 'qa-removal-conflict-model', serper: secrets.serperB, jina: secrets.jinaB };
  await fillInputs(removalInputs);
  const removeButton = page.getByRole('button', { name: '移除已保存的密钥', exact: true });
  await removeButton.click();
  await assertInputs(removalInputs);
  assert.match(await removeButton.getAttribute('class'), /danger-text/);
  assert.equal(await page.getByRole('button', { name: '测试模型响应', exact: true }).isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '连接并获取模型', exact: true }).isDisabled(), true);
  await externalModel('qa-concurrent-removal-baseline');
  const removalConflictBytes = await config();
  await saveUI(true);
  await assertInputs(removalInputs);
  assert.match(await removeButton.getAttribute('class'), /danger-text/);
  equalPrivate(await config(), removalConflictBytes, 'Conflicting key removal must not change config');
  await review();
  await keepInputs().click();
  await assertInputs(removalInputs);
  assert.match(await removeButton.getAttribute('class'), /danger-text/);
  await nativeCloseCancel();
  await saveUI();
  assert.equal((await api('/api/env')).values.OPENAI_API_KEY.configured, false);
  bytes = await config();
  assert(!bytes.includes(secrets.keyA) && !bytes.includes(secrets.keyB), 'Explicit removal must override a retained typed key');
  checks.push('removal-intent-overrides-typed-key-and-survives-conflict-review-keep-and-close-guard');

  const staleReadInputs = { base: baseA, key: secrets.keyA, model: 'qa-stale-read-model', serper: secrets.serperA, jina: secrets.jinaA };
  await fillInputs(staleReadInputs);
  await externalModel('qa-before-review-restart');
  await saveUI(true);
  await holdNext('GET');
  await readLatest().click();
  await held();
  await restartSidecar();
  await releaseHeld();
  await until(async () => await reviewPanel().getAttribute('data-state') === 'required', 'An old-service read was incorrectly accepted as current');
  await assertInputs(staleReadInputs);
  assert.equal(await saveButton().isDisabled(), true);
  await review();
  await keepInputs().click();
  await saveUI();
  checks.push('old-service-review-response-is-discarded-and-fresh-review-recovers-with-inputs-intact');

  const staleSaveInputs = { base: baseB, key: secrets.keyB, model: 'qa-stale-save-model', serper: secrets.serperB, jina: secrets.jinaB };
  await fillInputs(staleSaveInputs);
  await holdNext('PUT');
  const beforeStaleSave = await putCount();
  await allowSave(true);
  await saveButton().click();
  await held();
  await allowSave(false);
  assert.equal(await putCount(), beforeStaleSave + 1);
  const committedBytes = await config();
  assert(committedBytes.includes('qa-stale-save-model'), 'The delayed save response must follow a real completed write');
  await restartSidecar();
  await releaseHeld();
  await until(async () => await reviewPanel().getAttribute('data-state') === 'required', 'An old-service save response was incorrectly accepted as current');
  await assertInputs(staleSaveInputs);
  assert.equal(await saveButton().isDisabled(), true);
  assert.equal(await putCount(), beforeStaleSave + 1, 'Service recovery must not retry a save automatically');
  equalPrivate(await config(), committedBytes, 'Recovery unexpectedly changed already committed settings');
  await nativeCloseCancel();
  await assertInputs(staleSaveInputs);
  checks.push('old-service-save-success-needs-review-retains-inputs-and-never-replays-write');

  await page.getByRole('button', { name: '通用', exact: true }).click();
  await page.getByRole('button', { name: '深色', exact: true }).click();
  await page.getByRole('button', { name: '模型服务', exact: true }).click();
  await review();
  await snapshot('03-review-dark.png');
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  await snapshot('04-review-compact.png');
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'Compact settings overflow horizontally');
  await keepInputs().focus();
  await page.keyboard.press('Enter');
  await assertInputs(staleSaveInputs);
  await nativeCloseCancel();
  await saveButton().scrollIntoViewIfNeeded();
  const bounds = await saveButton().boundingBox();
  const viewport = await page.evaluate(() => ({ width: window.innerWidth, height: window.innerHeight }));
  assert(bounds && bounds.x >= 0 && bounds.y >= 0 && bounds.x + bounds.width <= viewport.width && bounds.y + bounds.height <= viewport.height, 'Compact save action is not reachable');
  await snapshot('05-compact-save.png', saveButton());
  checks.push('light-dark-compact-conflict-review-and-keyboard-actions-remain-usable');
  await saveUI();
  await assertInputs({ key: '', serper: '', jina: '' });
  checks.push('fresh-review-and-explicit-save-recover-after-service-restart');

  await noSecretVisible();
  const final = await api('/api/desktop/status');
  assert.equal(final.engine.ready, false);
  assert.deepEqual(await api('/api/runs'), []);
  const native = await application.evaluate(() => ({ operations: globalThis.__settingsQA.operations, forbidden: globalThis.__settingsQA.forbidden, closeDialogs: globalThis.__settingsQA.closeDialogs }));
  assert(native.operations.every(operation => operation.done), 'A native request was left pending');
  assert.deepEqual(native.forbidden, []);
  assert.deepEqual(fixtureViolations, []);
  assert.deepEqual(rendererErrors, []);
  assert.equal(providerRequests.length, 1);
  assert(providerRequests[0].fixedPrompt && providerRequests[0].completionBudget === 64, 'Provider probe was not the bounded fixed contract');
  checks.push('no-secret-display-storage-external-provider-runtime-install-run-or-renderer-error');
  const report = { passed: true, mode: 'Real Electron and sidecar settings writes/CAS; synthetic profile and one explicit deterministic loopback probe; not external model acceptance', executablePath: executablePath || 'development Electron', profile, checks, providerRequests, nativeOperations: native.operations, forbiddenRequests: native.forbidden, closeDialogs: native.closeDialogs, rendererErrors, restarts, configSHA256: createHash('sha256').update(await config()).digest('hex') };
  const serialized = JSON.stringify(report, null, 2);
  assert(!Object.values(secrets).some(secret => serialized.includes(secret)), 'Receipt contained a synthetic secret');
  await writeFile(path.join(output, 'acceptance.json'), serialized);
  console.log(serialized);
} catch (error) {
  if (page && !page.isClosed()) {
    try { await noSecretVisible(); await page.screenshot({ path: path.join(output, 'failure.png') }); } catch { /* Never capture UI that exposes synthetic secrets. */ }
  }
  console.error(sanitize(error?.stack || error));
  process.exitCode = 1;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => {
      globalThis.__settingsQA?.releaseHeld?.();
      for (const window of BrowserWindow.getAllWindows()) window.destroy();
    }).catch(() => {});
    await application.close();
  }
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
