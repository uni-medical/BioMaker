// Real Electron -> native bridge -> Python sidecar -> deterministic loopback
// provider acceptance. Synthetic credentials/profile only; no external model,
// runtime installation, training, or user experiment is contacted or executed.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdir, mkdtemp, readFile, readdir, realpath, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-provider-check-'));
const profile = path.join(temporary, 'profile');
const dataset = path.join(temporary, 'private-synthetic-dataset');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/provider-check'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
for (const directory of [profile, dataset, output]) await mkdir(directory, { recursive: true });
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1', NO_PROXY: '127.0.0.1,localhost' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const fixtureKey = 'qa-provider-fixture-key-never-external';
const replacementKey = 'qa-provider-replacement-key-never-external';
const fixtureModel = 'qa-provider-model';
const alternativeModel = 'qa-provider-alternative';
const privateTask = '  Private synthetic task; must never reach model connection probes.\nPreserve this exact draft.  ';
const rawProviderError = 'QA_RAW_PROVIDER_ERROR_MUST_NOT_REACH_UI';
const checks = [];
const errors = [];
const upstream = [];
const fixtureViolations = [];
const plans = { check: [], discover: [] };
const heldUpstream = [];
let application;
let page;
let draftRaw;
let draftId;
let savedConfig;
let restartedService;

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
    const body = raw ? JSON.parse(raw) : null;
    const kind = /\/chat\/completions$/.test(request.url) ? 'check' : /\/models$/.test(request.url) ? 'discover' : 'unknown';
    const entry = { kind, method: request.method, path: request.url, body, authorization: request.headers.authorization, completed: false };
    upstream.push(entry);
    if (kind === 'unknown' || (kind === 'check' ? request.method !== 'POST' : request.method !== 'GET')) fixtureViolations.push('Unexpected provider path or method');
    if (![`Bearer ${fixtureKey}`, `Bearer ${replacementKey}`].includes(entry.authorization)) fixtureViolations.push('Unexpected provider authorization');
    for (const value of [privateTask, dataset, profile, fixtureKey, replacementKey]) if (raw.includes(value)) fixtureViolations.push('Probe body contained private content or credentials');
    const plan = plans[kind]?.shift();
    if (!plan) fixtureViolations.push(`No planned ${kind} request`);
    if (plan?.hold) await new Promise(resolve => heldUpstream.push({ kind, entry, release: resolve }));
    const status = plan?.status || 200;
    const result = status !== 200
      ? { error: { message: `${rawProviderError} ${fixtureKey}`, type: 'fixture_error' } }
      : kind === 'discover'
        ? { object: 'list', data: (plan?.models || [fixtureModel, alternativeModel]).map(id => ({ id, object: 'model' })) }
        : { id: 'qa-fixed-completion', object: 'chat.completion', created: 1, model: body?.model, choices: [{ index: 0, message: { role: 'assistant', content: 'OK' }, finish_reason: 'stop' }], usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 } };
    response.writeHead(status, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(result));
    entry.completed = true;
  } catch (error) {
    fixtureViolations.push(error.message);
    if (!response.headersSent) response.writeHead(500, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ error: { message: 'Fixture failure' } }));
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const providerOrigin = `http://127.0.0.1:${server.address().port}`;
const providerBase = `${providerOrigin}/v1`;
const alternativeBase = `${providerOrigin}/alternate/v1`;
const config = () => readFile(path.join(profile, 'config.env'), 'utf8').catch(error => error.code === 'ENOENT' ? '' : Promise.reject(error));
const state = () => page.evaluate(() => window.biomaker.getServiceState());
const draftState = () => page.evaluate(() => window.biomaker.getDrafts());
const storedDraft = () => page.evaluate(async () => {
  const snapshot = await window.biomaker.getDrafts();
  const active = snapshot.drafts.find(entry => entry.id === snapshot.activeId);
  return active ? JSON.stringify(active.content) : null;
});
const probeButton = () => page.getByRole('button', { name: '测试模型响应', exact: true });
const discoverButton = () => page.getByRole('button', { name: '连接并获取模型', exact: true });
const baseInput = () => page.getByRole('textbox', { name: /^API 地址/ });
const keyInput = () => page.getByLabel('API Key', { exact: false }).first();
const modelInput = () => page.getByLabel('默认模型', { exact: true });
const renderFrame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
async function noResult() {
  await renderFrame();
  assert.equal(await page.locator('.provider-check-result').count(), 0, 'Stale provider results must not be displayed');
}
async function noModels() {
  await renderFrame();
  assert.equal(await page.getByLabel('选择可用模型', { exact: true }).count(), 0, 'Stale model lists must not be displayed');
}
async function assertPrivateState(expectedConfig = '') {
  assert.equal(await config(), expectedConfig, 'Testing a provider must not persist unsaved settings');
  assert.equal(await storedDraft(), draftRaw, 'Provider operations must retain the exact experiment draft');
  assert.equal((await draftState()).activeId, draftId, 'Provider operations must retain the active draft identity');
  const browserStorage = await page.evaluate(() => JSON.stringify(localStorage));
  assert(!browserStorage.includes(fixtureKey));
  assert(!browserStorage.includes(replacementKey));
}
async function operationCount() { return application.evaluate(() => globalThis.__providerQA.operations.length); }
async function waitOperation(index) {
  await until(() => application.evaluate((_electron, i) => globalThis.__providerQA.operations[i]?.done === true, index), 'Provider response did not settle');
  await renderFrame();
}
async function start(kind, plan = {}, duplicate = false) {
  const index = await operationCount();
  const before = upstream.length;
  plans[kind].push(plan);
  const button = kind === 'check' ? probeButton() : discoverButton();
  if (duplicate) await button.evaluate(element => { element.click(); element.click(); });
  else await button.click();
  await until(() => upstream.length > before, `The real sidecar did not call the ${kind} fixture`);
  assert.equal(upstream.length, before + 1, 'One UI action must send only one provider request');
  return { index, entry: upstream.at(-1) };
}
async function release(kind, operation) {
  const held = heldUpstream.find(item => item.kind === kind && item.entry === operation.entry);
  assert(held, 'The expected provider request was not held');
  held.release();
  await waitOperation(operation.index);
}
async function success() {
  await page.locator('.provider-check-result[role="status"]').getByText(/^模型响应测试通过[。]?$/).waitFor();
}
async function screenshotResult(filename) {
  await page.locator('.toast').waitFor({ state: 'hidden' });
  await page.locator('.provider-check-result').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, filename) });
}
async function save() {
  await application.evaluate(() => { globalThis.__providerQA.allowSave = true; });
  await page.getByRole('button', { name: '保存配置', exact: true }).click();
  await page.getByText('所有更改已保存', { exact: true }).waitFor();
  await application.evaluate(() => { globalThis.__providerQA.allowSave = false; });
  savedConfig = await config();
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => errors.push(error.message));
  await until(async () => { try { return (await state()).status === 'ready'; } catch { return false; } }, 'Real sidecar did not become ready');
  await page.locator('.sidebar-settings').waitFor();
  const actual = await page.evaluate(async () => ({ status: await window.biomaker.request('/api/desktop/status'), env: await window.biomaker.request('/api/env') }));
  assert.equal(actual.status.engine.ready, false);
  assert.equal(actual.env.values.OPENAI_API_KEY.configured, false);
  await application.evaluate((_electron, fixture) => {
    const originalFetch = globalThis.fetch;
    globalThis.__providerQA = { originalFetch, operations: [], forbidden: [], allowSave: false, bases: fixture.bases, dataset: fixture.dataset };
    globalThis.fetch = async (input, options) => {
      const f = globalThis.__providerQA;
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      const providerAction = method === 'POST' && ['/api/desktop/provider/check', '/api/models/discover'].includes(url.pathname);
      const body = options?.body ? JSON.parse(options.body) : null;
      const safeProvider = providerAction && f.bases.includes(body?.base_url);
      const metadataRead = method === 'POST' && url.pathname === '/api/desktop/dataset/inspect' && body?.path === f.dataset;
      if (url.hostname !== '127.0.0.1' || (!['GET', 'HEAD'].includes(method) && !safeProvider && !metadataRead && !(f.allowSave && method === 'PUT' && url.pathname === '/api/env'))) {
        f.forbidden.push({ path: url.pathname, method });
        throw new Error('Provider QA blocked an unplanned destination or mutation');
      }
      const operation = providerAction ? { path: url.pathname, done: false } : null;
      if (operation) f.operations.push(operation);
      try {
        const response = await originalFetch(input, options);
        if (url.pathname === '/api/desktop/provider/check' && f.holdNextCheckResult) {
          f.holdNextCheckResult = false;
          const contents = await response.arrayBuffer();
          f.heldCheckCode = JSON.parse(new TextDecoder().decode(contents)).code;
          const buffered = new Response(contents, { status: response.status, headers: response.headers });
          await new Promise(resolve => { f.releaseCheckResult = () => { delete f.releaseCheckResult; resolve(); }; });
          return buffered;
        }
        return response;
      } finally { if (operation) operation.done = true; }
    };
  }, { bases: [providerBase, alternativeBase], dataset: await realpath(dataset) });

  await page.getByLabel('实验任务', { exact: true }).fill(privateTask);
  await application.evaluate(({ dialog }, directory) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [directory] }); }, dataset);
  await page.locator('.dataset-button').click();
  await page.getByRole('button', { name: '实验设置', exact: true }).click();
  await page.getByLabel('实验名称', { exact: false }).fill('private-provider-check-draft');
  await page.getByLabel('最大探索步数', { exact: true }).fill('7');
  await until(async () => JSON.parse(await storedDraft() || '{}').options?.agent_steps === 7, 'Draft was not persisted');
  draftRaw = await storedDraft();
  draftId = (await draftState()).activeId;
  assert(draftId, 'The synthetic task must have a durable draft identity');
  await page.locator('.sidebar-settings').click();
  await baseInput().fill(providerBase);
  await keyInput().fill(fixtureKey);
  await modelInput().fill(fixtureModel);
  await probeButton().waitFor();
  assert.equal(await probeButton().isEnabled(), true, 'A missing runtime must not prevent checking the provider');
  assert.match(await page.locator('.provider-readiness').innerText(), /配置已填写，尚未保存/);
  checks.push('unsaved-provider-can-be-tested-before-installing-runtime');

  const duplicate = await start('check', { hold: true }, true);
  assert.equal(await probeButton().isDisabled(), true);
  assert.equal(await page.getByRole('button', { name: '保存配置', exact: true }).isDisabled(), true);
  assert.equal(await modelInput().isEnabled(), true);
  assert.equal(await baseInput().isEnabled(), true);
  assert.equal(await keyInput().isEnabled(), true);
  assert.equal(await operationCount(), duplicate.index + 1);
  await release('check', duplicate);
  await success();
  await assertPrivateState();
  assert.equal(await page.getByRole('button', { name: '保存配置', exact: true }).isEnabled(), true);
  assert.equal(await page.getByText('所有更改已保存', { exact: true }).count(), 0);
  await screenshotResult('01-verified-unsaved.png');
  checks.push('real-provider-success-is-not-saving-and-double-click-sends-one-bounded-request');

  for (const field of ['model', 'key', 'base']) {
    const operation = await start('check', { hold: true });
    const input = field === 'model' ? modelInput() : field === 'key' ? keyInput() : baseInput();
    const original = await input.inputValue();
    await input.fill(field === 'model' ? alternativeModel : field === 'key' ? replacementKey : alternativeBase);
    await input.fill(original);
    await noResult();
    await release('check', operation);
    await noResult();
    await assertPrivateState();
    checks.push(`editing-${field}-and-reverting-discards-older-check-success`);
  }

  const oldModels = await start('discover', { hold: true, models: ['qa-stale-list-model'] });
  await modelInput().fill(alternativeModel);
  await release('discover', oldModels);
  await noResult();
  await noModels();
  await modelInput().fill(fixtureModel);
  const discovery = await start('discover');
  await waitOperation(discovery.index);
  await page.getByLabel('选择可用模型', { exact: true }).waitFor();
  assert.equal(await page.locator('.provider-check-result[role="status"]').count(), 0);
  await page.getByLabel('选择可用模型', { exact: true }).selectOption(alternativeModel);
  assert.equal(await modelInput().inputValue(), alternativeModel);
  assert.equal(await page.getByLabel('选择可用模型', { exact: true }).count(), 1, 'Selecting another model must retain this provider\'s model list');
  await modelInput().fill(fixtureModel);
  assert.equal(await page.getByLabel('选择可用模型', { exact: true }).count(), 1);
  await keyInput().fill(replacementKey);
  await noModels();
  await keyInput().fill(fixtureKey);
  const rediscovery = await start('discover');
  await waitOperation(rediscovery.index);
  await page.getByLabel('选择可用模型', { exact: true }).waitFor();
  await baseInput().fill(alternativeBase);
  await noResult();
  await noModels();
  await baseInput().fill(providerBase);
  await assertPrivateState();
  checks.push('model-discovery-is-not-response-verification-and-edits-discard-old-lists');

  const failure = await start('check', { status: 401 });
  await waitOperation(failure.index);
  await page.locator('.provider-check-result.failed').waitFor();
  assert(!(await page.getByRole('dialog', { name: '设置', exact: true }).innerText()).includes(rawProviderError));
  assert(!(await page.getByRole('dialog', { name: '设置', exact: true }).innerText()).includes(fixtureKey));
  assert.equal(await probeButton().isEnabled(), true);
  await screenshotResult('02-auth-failure.png');
  const retry = await start('check');
  await waitOperation(retry.index);
  await success();
  await assertPrivateState();
  checks.push('provider-errors-are-safe-actionable-and-retryable-without-persisting-settings');

  const incompatible = await start('check', { status: 400 });
  await waitOperation(incompatible.index);
  await page.locator('.provider-check-result.failed').waitFor();
  assert.match(await page.locator('.provider-check-result').innerText(), /Chat Completions/);
  assert.match(await page.locator('.provider-check-result').innerText(), /max_completion_tokens/);
  assert(!(await page.locator('.provider-check-result').innerText()).includes(rawProviderError));
  const compatibleRetry = await start('check');
  await waitOperation(compatibleRetry.index);
  await success();
  checks.push('incompatible-probe-request-explains-format-without-misreporting-model-or-key');

  await save();
  assert(savedConfig.includes(fixtureKey));
  assert.equal(await keyInput().inputValue(), '');
  // Saving may retain this exact verification or require rechecking. It must
  // never fabricate a successful result after the provider/model changes.
  await modelInput().fill(alternativeModel);
  await save();
  await noResult();
  assert.match(await page.locator('.provider-readiness').innerText(), /尚未测试模型响应/);
  const savedKeyCheck = await start('check');
  await waitOperation(savedKeyCheck.index);
  await success();
  assert.equal(savedKeyCheck.entry.authorization, `Bearer ${fixtureKey}`);
  await assertPrivateState(savedConfig);
  checks.push('saving-alone-is-not-verification-and-check-can-use-the-saved-fixture-key');

  const beforeAddressChange = await operationCount();
  await baseInput().fill(alternativeBase);
  await noResult();
  assert.equal(await probeButton().isDisabled(), true);
  assert.equal(await discoverButton().isDisabled(), true);
  await probeButton().evaluate(element => element.click());
  await renderFrame();
  assert.equal(await operationCount(), beforeAddressChange, 'The saved key must not be forwarded to an edited provider address');
  await baseInput().fill(providerBase);
  assert.equal(await probeButton().isEnabled(), true);
  await assertPrivateState(savedConfig);
  checks.push('saved-key-is-bound-to-the-saved-base-and-not-forwarded-to-an-edited-address');

  const removedKey = await start('check', { hold: true });
  await page.getByRole('button', { name: '移除已保存的密钥', exact: true }).click();
  await noResult();
  await release('check', removedKey);
  await noResult();
  const countBeforeRemovalCheck = upstream.length;
  if (await probeButton().isEnabled()) {
    const removalOperation = await operationCount();
    await probeButton().click();
    await waitOperation(removalOperation);
  }
  assert.equal(upstream.length, countBeforeRemovalCheck, 'A key marked for removal must not silently use the saved key');
  await assertPrivateState(savedConfig);
  await page.getByRole('button', { name: '移除已保存的密钥', exact: true }).click();
  await noResult();
  checks.push('removing-saved-key-invalidates-inflight-check-and-prevents-key-fallback');

  await application.evaluate(() => { globalThis.__providerQA.holdNextCheckResult = true; });
  const oldService = await start('check');
  await until(() => application.evaluate(() => typeof globalThis.__providerQA.releaseCheckResult === 'function'), 'Real successful provider response was not held');
  assert.equal(await application.evaluate(() => globalThis.__providerQA.heldCheckCode), 'ok');
  const beforeRestart = await state();
  const runtimes = (await readdir(path.join(profile, 'runtimes'))).filter(name => !name.includes('.staging-'));
  assert.equal(runtimes.length, 1);
  const serviceFile = path.join(profile, 'runtimes', runtimes[0], 'service/main.py');
  const children = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes(serviceFile) && line.includes('python'));
  assert.equal(children.length, 1, 'Exactly one sidecar must own this isolated profile');
  const pid = Number(children[0].trim().split(/\s+/)[0]);
  process.kill(pid, 'SIGKILL');
  await until(async () => { const next = await state(); return next.status === 'ready' && next.revision > beforeRestart.revision; }, 'Isolated sidecar did not restart with a newer revision', 60_000);
  await application.evaluate(() => globalThis.__providerQA.releaseCheckResult());
  await waitOperation(oldService.index);
  await noResult();
  assert.equal(await modelInput().inputValue(), alternativeModel);
  assert.equal(await baseInput().inputValue(), providerBase);
  await assertPrivateState(savedConfig);
  restartedService = { previousRevision: beforeRestart.revision, recoveredRevision: (await state()).revision };
  assert.equal(await probeButton().isDisabled(), true, 'An unreviewed saved key cannot be reused after service replacement');
  await page.getByRole('button', { name: '读取最新已保存配置', exact: true }).click();
  await page.locator('.provider-settings-review[data-state="loaded"]').waitFor();
  await assertPrivateState(savedConfig);
  await page.getByRole('button', { name: '保留输入并继续编辑', exact: true }).click();
  await assertPrivateState(savedConfig);
  const afterRestart = await start('check');
  await waitOperation(afterRestart.index);
  await success();
  await screenshotResult('03-recovered-service.png');
  checks.push('real-sidecar-restart-discards-old-success-preserves-draft-and-reviews-saved-key-before-new-check');

  await page.getByRole('button', { name: '通用', exact: true }).click();
  await page.getByRole('button', { name: '深色', exact: true }).click();
  await page.getByRole('button', { name: '模型服务', exact: true }).click();
  await screenshotResult('04-dark-provider-check.png');
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  await probeButton().scrollIntoViewIfNeeded();
  await renderFrame();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await baseInput().focus();
  let probeReached = false;
  for (let i = 0; i < 24; i++) {
    await page.keyboard.press('Tab');
    probeReached = await probeButton().evaluate(element => document.activeElement === element);
    if (probeReached) break;
  }
  assert(probeReached, 'Provider response testing must be reachable with the keyboard');
  await probeButton().scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, '05-compact-keyboard-check.png') });
  await screenshotResult('05b-compact-check-result.png');
  const saveButton = page.getByRole('button', { name: '保存配置', exact: true });
  await saveButton.scrollIntoViewIfNeeded();
  const saveBounds = await saveButton.boundingBox();
  const viewport = await page.evaluate(() => ({ width: window.innerWidth, height: window.innerHeight }));
  assert(saveBounds && saveBounds.x >= 0 && saveBounds.y >= 0 && saveBounds.x + saveBounds.width <= viewport.width && saveBounds.y + saveBounds.height <= viewport.height, 'Save control must remain reachable inside a compact window');
  await page.screenshot({ path: path.join(output, '06-compact-save-controls.png') });
  checks.push('light-dark-and-compact-settings-retain-visible-controls-and-keyboard-access');

  const final = await page.evaluate(async () => ({ status: await window.biomaker.request('/api/desktop/status'), runs: await window.biomaker.request('/api/runs') }));
  assert.equal(final.status.engine.ready, false);
  assert.deepEqual(final.runs, []);
  await assertPrivateState(savedConfig);
  const native = await application.evaluate(() => ({ operations: globalThis.__providerQA.operations, forbidden: globalThis.__providerQA.forbidden }));
  assert.deepEqual(native.forbidden, []);
  assert.deepEqual(fixtureViolations, []);
  assert.deepEqual(errors, []);
  assert.equal(native.operations.every(operation => operation.done), true);
  // The contract intentionally contains no arbitrary task input. The backend
  // owns the fixed prompt and a small bounded completion budget.
  const probes = upstream.filter(entry => entry.kind === 'check');
  assert(probes.length > 0);
  const reference = probes[0].body;
  assert.deepEqual(reference.messages, [{ role: 'user', content: 'Reply with exactly OK.' }]);
  const budget = reference.max_completion_tokens;
  assert.equal(budget, 64, 'Connection probes must use the fixed 64-token completion budget');
  assert.deepEqual(Object.keys(reference).sort(), ['max_completion_tokens', 'messages', 'model', 'stream']);
  for (const entry of probes) {
    assert([fixtureModel, alternativeModel].includes(entry.body.model));
    assert.deepEqual(entry.body.messages, reference.messages);
    assert.equal(entry.body.max_tokens ?? entry.body.max_completion_tokens, budget);
    assert.equal(entry.body.stream, false);
    assert.deepEqual(Object.keys(entry.body).sort(), Object.keys(reference).sort());
  }
  checks.push('only-fixed-bounded-loopback-probes-no-user-task-runtime-operation-run-or-renderer-error');
  const report = {
    passed: true,
    mode: 'real Electron and sidecar with deterministic loopback provider; not an external model acceptance',
    executablePath: executablePath || 'development Electron', profile, providerBase, checks,
    requestContract: { messages: reference.messages, completionBudget: budget, fields: Object.keys(reference).sort() },
    providerRequests: upstream.map(entry => ({ kind: entry.kind, method: entry.method, path: entry.path, model: entry.body?.model || null, fixtureAuthorization: [`Bearer ${fixtureKey}`, `Bearer ${replacementKey}`].includes(entry.authorization), completed: entry.completed })),
    nativeOperations: native.operations, forbiddenRequests: native.forbidden, fixtureViolations, rendererErrors: errors, restartedService,
  };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-6000));
  }
  throw error;
} finally {
  for (const held of heldUpstream) held.release();
  if (application) {
    await application.evaluate(({ BrowserWindow }) => {
      globalThis.__providerQA?.releaseCheckResult?.();
      for (const window of BrowserWindow.getAllWindows()) window.destroy();
    }).catch(() => {});
    await application.close();
  }
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
