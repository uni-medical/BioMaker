// Real Electron/native bridge/portable sidecar metadata inspection against
// synthetic temporary directories. No experiment runtime or provider is used.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createServer } from 'node:net';
import { mkdir, mkdtemp, readFile, readdir, realpath, rename, rmdir, symlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// macOS AF_UNIX socket names have a short path limit; a private /tmp directory
// also exercises the system alias independently of the canonical picker paths.
const temporary = await mkdtemp(path.join(process.platform === 'darwin' ? '/tmp' : os.tmpdir(), 'biomaker-dataset-inspection-'));
const profile = path.join(temporary, 'profile');
const fixtures = path.join(temporary, 'fixtures');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/dataset-inspection-0.1.9'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
for (const directory of [profile, fixtures, output]) await mkdir(directory, { recursive: true });
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const endpoint = '/api/desktop/dataset/inspect';
const contentSentinel = 'PRIVATE_SYNTHETIC_CONTENT_MUST_NOT_BE_RETURNED_OR_SENT';
const task = '  Dataset metadata acceptance. Preserve this exact private task; never submit.  ';
const checks = [];
const errors = [];
const results = [];
const fixturePaths = {};
let application;
let page;
let draftRaw;
let draftId;
let specialServer;
let restart;

async function directory(name) {
  const target = path.join(fixtures, name);
  await mkdir(target, { recursive: true });
  fixturePaths[name] = target;
  return target;
}
const normal = await directory('normal');
await mkdir(path.join(normal, 'images'));
await writeFile(path.join(normal, 'volume.nii.gz'), contentSentinel);
await writeFile(path.join(normal, 'labels.csv'), 'class\n1\n');
await writeFile(path.join(normal, 'images', 'image.png'), 'synthetic image metadata only');
const normalHash = createHash('sha256').update(await readFile(path.join(normal, 'volume.nii.gz'))).digest('hex');
const second = await directory('second');
await writeFile(path.join(second, 'sample.txt'), 'synthetic second dataset');
const empty = await directory('empty');
const invalidName = await directory('unsupported\nfolder-name');
const emptyDescendants = await directory('empty-descendants');
await mkdir(path.join(emptyDescendants, 'one', 'two'), { recursive: true });
const unknown = await directory('unknown-format');
await writeFile(path.join(unknown, 'sample.unknownfixture'), contentSentinel);
const zero = await directory('zero-byte');
await writeFile(path.join(zero, 'placeholder.bin'), '');
const outsider = await directory('ungranted');
await writeFile(path.join(outsider, 'private-external.txt'), contentSentinel);
const prefixEscape = await directory('normal-private');
await writeFile(path.join(prefixEscape, 'private.txt'), contentSentinel);
const onlyLinks = await directory('only-symlinks');
await symlink(outsider, path.join(onlyLinks, 'external-directory'), 'dir');
const internalLinks = await directory('internal-links');
await writeFile(path.join(internalLinks, 'ordinary.csv'), '1,2\n');
await symlink(path.join(outsider, 'private-external.txt'), path.join(internalLinks, 'external-file'));
await symlink(internalLinks, path.join(internalLinks, 'cycle'), 'dir');
const alias = path.join(fixtures, 'normal-alias');
await symlink(normal, alias, 'dir');
const partial = await directory('depth-limited');
await mkdir(path.join(partial, 'one', 'two', 'three', 'four'), { recursive: true });
await writeFile(path.join(partial, 'one', 'two', 'three', 'four', 'beyond-budget.nii.gz'), contentSentinel);
const special = await directory('special-files');
execFileSync('mkfifo', [path.join(special, 'never-open.fifo')]);
specialServer = createServer(socket => socket.destroy());
await new Promise((resolve, reject) => { specialServer.once('error', reject); specialServer.listen(path.join(special, 'never-connect.sock'), resolve); });

async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 30));
  }
}
const api = (pathname, method = 'GET', body) => page.evaluate(([pathname, method, body]) => window.biomaker.request(pathname, { method, ...(body === undefined ? {} : { body }) }), [pathname, method, body]);
const state = () => page.evaluate(() => window.biomaker.getServiceState());
const draftState = () => page.evaluate(() => window.biomaker.getDrafts());
const storedDraft = () => page.evaluate(async () => {
  const snapshot = await window.biomaker.getDrafts();
  const active = snapshot.drafts.find(entry => entry.id === snapshot.activeId);
  return active ? JSON.stringify(active.content) : null;
});
const overview = () => page.locator('.dataset-overview');
const refreshButton = () => page.getByRole('button', { name: '重新检查数据集', exact: true });
const renderFrame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
async function picker(target, canceled = false) {
  await application.evaluate(({ dialog }, selection) => {
    dialog.showOpenDialog = async () => ({ canceled: selection.canceled, filePaths: selection.canceled ? [] : [selection.target] });
  }, { target, canceled });
}
async function grant(target) {
  await picker(target);
  const selected = await page.evaluate(() => window.biomaker.chooseDirectory({ purpose: 'dataset' }));
  assert.equal(selected, await realpath(target));
  return selected;
}
async function inspect(target, expected = 'ok', suppliedRoots) {
  const result = await api(endpoint, 'POST', { path: target, ...(suppliedRoots ? { authorized_roots: suppliedRoots } : {}) });
  assert.equal(result.code, expected, `${path.basename(target)} returned ${JSON.stringify(result)}`);
  assert.equal(result.status, expected === 'ok' ? 'ok' : 'error');
  for (const key of ['files', 'directories', 'symlinks', 'other', 'bytes']) assert(Number.isFinite(result[key]) && result[key] >= 0, `Invalid ${key} in metadata`);
  assert.equal(typeof result.complete, 'boolean');
  assert(Array.isArray(result.entries));
  assert(Array.isArray(result.extensions));
  assert(Array.isArray(result.warnings));
  assert(!JSON.stringify(result).includes(contentSentinel), 'Inspection must never return file contents');
  for (const entry of result.entries) {
    assert.equal(typeof entry.name, 'string');
    assert(!path.isAbsolute(entry.name) && !entry.name.split(/[\\/]/).includes('..'), 'Samples must be relative paths');
  }
  results.push({ fixture: path.basename(target), ...result });
  return result;
}
async function choose(target, code = 'ok', waitForInspection = true) {
  const canonical = await realpath(target);
  const previousOperations = await application.evaluate(() => globalThis.__datasetQA.operations.length);
  await picker(target);
  await page.locator('.dataset-button').click();
  await until(async () => (JSON.parse(await storedDraft() || '{}').dataset) === canonical, 'Selected dataset was not persisted');
  if (!waitForInspection) { await renderFrame(); return; }
  await until(() => application.evaluate((_electron, expected) => globalThis.__datasetQA.operations.slice(expected.previousOperations).some(operation => operation.path === expected.canonical && operation.done), { previousOperations, canonical }), 'The selected dataset was not inspected');
  await until(async () => await overview().getAttribute('data-code') === code && await overview().getAttribute('data-state') !== 'checking', `Overview did not settle as ${code}`);
  await renderFrame();
}
async function holdNext(target) {
  await application.evaluate((_electron, target) => { globalThis.__datasetQA.holdPath = target; }, await realpath(target));
}
async function held() {
  await until(() => application.evaluate(() => typeof globalThis.__datasetQA.releaseHeld === 'function'), 'The real dataset metadata response was not held');
}
async function releaseHeld() {
  await application.evaluate(() => globalThis.__datasetQA.releaseHeld());
  await until(() => application.evaluate(() => globalThis.__datasetQA.heldOperation?.done === true), 'Held response did not settle');
  await renderFrame();
}
async function snapshot(name) {
  await page.locator('.toast').waitFor({ state: 'hidden' });
  await overview().scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, name) });
}
async function assertNoMutation() {
  const contents = await readFile(path.join(profile, 'config.env'), 'utf8').catch(error => error.code === 'ENOENT' ? '' : Promise.reject(error));
  assert.equal(contents, '');
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  assert.deepEqual(await api('/api/runs'), []);
  assert.equal(createHash('sha256').update(await readFile(path.join(normal, 'volume.nii.gz'))).digest('hex'), normalHash);
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => errors.push(error.message));
  await until(async () => { try { return (await state()).status === 'ready'; } catch { return false; } }, 'Portable sidecar did not become ready');
  await page.locator('.sidebar-settings').waitFor();
  await application.evaluate((_electron, endpoint) => {
    const originalFetch = globalThis.fetch;
    globalThis.__datasetQA = { originalFetch, operations: [], forbidden: [], holdPath: null };
    globalThis.fetch = async (input, options) => {
      const f = globalThis.__datasetQA;
      const url = new URL(String(input));
      const method = String(options?.method || 'GET').toUpperCase();
      const isInspection = url.pathname === endpoint && method === 'POST';
      if (url.hostname !== '127.0.0.1' || (!['GET', 'HEAD'].includes(method) && !isInspection)) {
        f.forbidden.push({ path: url.pathname, method });
        throw new Error('Dataset QA forbids providers, experiments, runtime operations and settings writes');
      }
      const body = isInspection ? JSON.parse(options.body) : null;
      const operation = isInspection ? { path: body.path, roots: body.authorized_roots, done: false } : null;
      if (operation) f.operations.push(operation);
      try {
        const response = await originalFetch(input, options);
        if (operation && body.path === f.holdPath) {
          f.holdPath = null;
          const contents = await response.arrayBuffer();
          f.heldResult = JSON.parse(new TextDecoder().decode(contents));
          f.heldOperation = operation;
          const buffered = new Response(contents, { status: response.status, headers: response.headers });
          await new Promise(resolve => { f.releaseHeld = () => { delete f.releaseHeld; resolve(); }; });
          return buffered;
        }
        return response;
      } finally { if (operation) operation.done = true; }
    };
  }, endpoint);
  await assertNoMutation();

  const denied = await inspect(outsider, 'unauthorized', [fixtures, path.parse(fixtures).root]);
  assert.deepEqual(denied.entries, []);
  assert.equal(await application.evaluate(() => globalThis.__datasetQA.operations.at(-1).roots.length), 1, 'Only the application workspace grant should exist before a picker selection');
  checks.push('renderer-grants-are-overwritten-and-unselected-directory-is-not-enumerated');

  const workspaceData = path.join(profile, 'workspace', 'dataset-qa');
  await mkdir(workspaceData, { recursive: true });
  await writeFile(path.join(workspaceData, 'sample.csv'), '1,2\n');
  const builtIn = await inspect(workspaceData);
  assert.equal(builtIn.files, 1);
  checks.push('application-workspace-is-authorized-without-picker-even-through-platform-temp-alias');

  await grant(normal);
  const normalResult = await inspect(normal);
  assert.equal(normalResult.files, 3);
  assert.equal(normalResult.directories, 1);
  assert.equal(normalResult.complete, true);
  assert(normalResult.extensions.some(item => item.extension === '.nii.gz' && item.count === 1));
  await inspect(prefixEscape, 'unauthorized', [prefixEscape]);
  checks.push('picked-directory-shows-compound-extensions-and-rejects-prefix-sibling-escape');

  await grant(empty);
  await inspect(empty, 'empty');
  await grant(emptyDescendants);
  const emptyTree = await inspect(emptyDescendants, 'empty');
  assert.equal(emptyTree.complete, true);
  assert.equal(emptyTree.files, 0);
  checks.push('empty-root-and-completely-scanned-empty-descendants-are-distinguished-from-data');

  await grant(unknown);
  const unknownResult = await inspect(unknown);
  assert.equal(unknownResult.files, 1);
  await grant(zero);
  const zeroResult = await inspect(zero);
  assert.equal(zeroResult.files, 1);
  assert.equal(zeroResult.bytes, 0);
  checks.push('unknown-extensions-and-zero-byte-files-are-not-misclassified-as-empty');

  await grant(onlyLinks);
  const linksOnly = await inspect(onlyLinks);
  assert.equal(linksOnly.symlinks, 1);
  assert.equal(linksOnly.files, 0);
  assert(linksOnly.warnings.includes('symlinks_not_followed'));
  await grant(internalLinks);
  const links = await inspect(internalLinks);
  assert.equal(links.symlinks, 2);
  assert.equal(links.files, 1);
  assert(!links.entries.some(entry => entry.name.includes('private-external')));
  const aliasResult = await inspect(alias);
  assert.equal(aliasResult.files, normalResult.files);
  assert(aliasResult.warnings.includes('root_symlink_resolved'));
  checks.push('root-alias-is-canonicalized-while-internal-and-cyclic-symlinks-are-not-followed');

  const retarget = await directory('retargeted-grant');
  await grant(retarget);
  await rename(retarget, `${retarget}-original`);
  await symlink(outsider, retarget, 'dir');
  await inspect(retarget, 'unauthorized');
  const missing = await directory('disappeared');
  await grant(missing);
  await rmdir(missing);
  await inspect(missing, 'not_found');
  const replaced = await directory('replaced-by-file');
  await grant(replaced);
  await rmdir(replaced);
  await writeFile(replaced, contentSentinel);
  await inspect(replaced, 'not_directory');
  checks.push('retargeted-grants-missing-folders-and-file-replacements-fail-without-reading-content');

  await grant(special);
  const specialResult = await inspect(special);
  assert.equal(specialResult.other, 2);
  assert.equal(specialResult.files, 0);
  await grant(partial);
  const partialResult = await inspect(partial);
  assert.equal(partialResult.complete, false);
  assert(partialResult.warnings.includes('depth_limit'));
  checks.push('special-files-are-not-opened-and-depth-budget-produces-an-explicit-partial-overview');

  await page.getByLabel('实验任务', { exact: true }).fill(task);
  const beforeInvalidPath = await application.evaluate(() => globalThis.__datasetQA.operations.length);
  await picker(invalidName);
  await page.locator('.dataset-button').click();
  await until(async () => await overview().getAttribute('data-code') === 'invalid_request', 'A selected unsupported path must ask for reselection rather than generic retry');
  assert.match(await page.locator('.dataset-inspection-status').innerText(), /重新选择/);
  assert.equal(await application.evaluate(() => globalThis.__datasetQA.operations.length), beforeInvalidPath, 'Invalid path metadata must not be forwarded to the sidecar');
  await snapshot('00-invalid-path.png');
  await choose(normal);
  assert.equal(await application.evaluate(() => globalThis.__datasetQA.operations.at(-1).roots.some(root => root.includes('\n'))), false);
  checks.push('unsupported-picked-name-prompts-reselection-without-scan-and-cannot-poison-valid-grants');
  draftRaw = await storedDraft();
  draftId = (await draftState()).activeId;
  assert(draftId, 'The synthetic task must have a durable draft identity');
  const originalCard = await overview().innerText();
  await picker(normal, true);
  await page.locator('.dataset-button').click();
  await renderFrame();
  assert.equal(await storedDraft(), draftRaw);
  assert.equal(await overview().innerText(), originalCard);
  await snapshot('01-normal-light.png');
  checks.push('picker-cancellation-preserves-selected-dataset-overview-and-exact-draft');

  await choose(empty, 'empty');
  assert.equal(await page.locator('.submit-button').isDisabled(), true);
  await page.getByLabel('实验任务', { exact: true }).focus();
  await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Enter' : 'Control+Enter');
  assert.equal(await page.getByRole('dialog', { name: '设置', exact: true }).count(), 0, 'An empty dataset must be resolved before runtime setup or submission');
  await snapshot('02-empty-directory.png');
  await choose(partial);
  await snapshot('03-partial-directory.png');
  await choose(normal);
  await holdNext(normal);
  await refreshButton().click();
  await held();
  assert.equal(await page.locator('.submit-button').isDisabled(), true, 'Submitting must remain unavailable while metadata is pending');
  assert.equal((await application.evaluate(() => globalThis.__datasetQA.heldResult)).files, 3);
  await choose(second, 'ok', false);
  await writeFile(path.join(normal, 'new-after-preview.tsv'), 'new synthetic file');
  await choose(normal, 'ok', false);
  assert.equal(await overview().getAttribute('data-state'), 'checking');
  assert.equal(await page.locator('.dataset-overview-counts').count(), 0);
  // The latest A response differs from the earlier held A result, proving
  // that a path comparison alone cannot pass this A -> B -> A race.
  await releaseHeld();
  await until(async () => await overview().getAttribute('data-code') === 'ok' && /4 个普通文件/.test(await page.locator('.dataset-overview-counts').innerText()), 'A -> B -> A did not replace the old three-file result with a fresh four-file result');
  assert.equal(await storedDraft(), draftRaw);
  checks.push('refresh-and-a-b-a-selection-ignore-an-older-response-for-the-same-path');

  await holdNext(normal);
  await refreshButton().click();
  await held();
  assert.equal((await application.evaluate(() => globalThis.__datasetQA.heldResult)).files, 4);
  await writeFile(path.join(normal, 'new-after-restart.json'), '{}');
  const beforeRestart = await state();
  const runtimes = (await readdir(path.join(profile, 'runtimes'))).filter(name => !name.includes('.staging-'));
  assert.equal(runtimes.length, 1);
  const serviceFile = path.join(profile, 'runtimes', runtimes[0], 'service/main.py');
  const children = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes(serviceFile) && line.includes('python'));
  assert.equal(children.length, 1, 'Only the isolated sidecar may be restarted');
  process.kill(Number(children[0].trim().split(/\s+/)[0]), 'SIGKILL');
  await until(async () => { const next = await state(); return next.status === 'ready' && next.revision > beforeRestart.revision; }, 'Sidecar did not restart', 60_000);
  await until(async () => await overview().getAttribute('data-state') === 'checking', 'Recovery did not invalidate the older overview');
  assert.equal(await page.locator('.dataset-overview-counts').count(), 0);
  await releaseHeld();
  await until(async () => await overview().getAttribute('data-code') === 'ok' && /5 个普通文件/.test(await page.locator('.dataset-overview-counts').innerText()), 'Recovered service did not inspect the current five-file directory contents');
  assert.equal(await storedDraft(), draftRaw);
  restart = { previousRevision: beforeRestart.revision, recoveredRevision: (await state()).revision };
  checks.push('service-revision-invalidates-old-overview-and-preserves-task-and-selection');

  await page.locator('.sidebar-settings').click();
  await page.getByRole('button', { name: '通用', exact: true }).click();
  await page.getByRole('button', { name: '深色', exact: true }).click();
  await page.keyboard.press('Escape');
  await snapshot('04-normal-dark.png');
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  await overview().scrollIntoViewIfNeeded();
  await renderFrame();
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.getByText('查看目录概览', { exact: true }).click();
  await snapshot('05-compact-details.png');
  await refreshButton().focus();
  assert(await refreshButton().evaluate(element => document.activeElement === element));
  const beforeKeyboardRefresh = await application.evaluate(() => globalThis.__datasetQA.operations.length);
  await page.keyboard.press('Enter');
  await until(() => application.evaluate((_electron, previous) => globalThis.__datasetQA.operations.slice(previous).some(operation => operation.done), beforeKeyboardRefresh), 'Keyboard refresh did not send an inspection request');
  await until(async () => await overview().getAttribute('data-code') === 'ok' && await overview().getAttribute('data-state') !== 'checking', 'Keyboard refresh did not settle');
  await snapshot('06-compact-keyboard.png');
  checks.push('light-dark-and-compact-overview-keeps-details-and-keyboard-refresh-usable');

  await assertNoMutation();
  assert.equal(await page.getByLabel('实验任务', { exact: true }).inputValue(), task);
  assert.equal(await storedDraft(), draftRaw);
  assert.equal((await draftState()).activeId, draftId, 'Dataset inspection must preserve the active draft identity');
  const native = await application.evaluate(() => ({ operations: globalThis.__datasetQA.operations, forbidden: globalThis.__datasetQA.forbidden }));
  assert.deepEqual(native.forbidden, []);
  assert.deepEqual(errors, []);
  assert(native.operations.every(operation => operation.done));
  checks.push('metadata-only-no-provider-run-install-secret-setting-change-or-renderer-error');
  const report = { passed: true, mode: 'real Electron and sidecar metadata inspection of synthetic directories; no experimental runtime', executablePath: executablePath || 'development Electron', profile, checks, results, inspections: native.operations.map(operation => ({ path: operation.path, authorizedRootCount: operation.roots.length, done: operation.done })), forbiddenRequests: native.forbidden, rendererErrors: errors, restart };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ ...report, results: results.map(result => ({ fixture: result.fixture, code: result.code, complete: result.complete, files: result.files, directories: result.directories, symlinks: result.symlinks, other: result.other, warnings: result.warnings })), inspections: native.operations.length }, null, 2));
} catch (error) {
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-6500));
  }
  throw error;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => {
      globalThis.__datasetQA?.releaseHeld?.();
      for (const window of BrowserWindow.getAllWindows()) window.destroy();
    }).catch(() => {});
    await application.close();
  }
  if (specialServer) await new Promise(resolve => specialServer.close(resolve));
}
