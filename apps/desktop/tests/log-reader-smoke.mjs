// Real Electron/sidecar bounded log reading and native export in a synthetic
// profile. Only lifecycle display, response timing/failure and clipboard are
// intercepted. No experiment, provider request, engine installation or user data.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { appendFile, mkdir, mkdtemp, readFile, readdir, realpath, rename, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-log-reader-'));
const profile = path.join(temporary, 'profile');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/log-reader-0.1.13'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const names = ['qa-log-reader-first', 'qa-log-reader-second'];
const directories = names.map(name => path.join(profile, 'workspace/logs', name));
const logFiles = directories.map((directory, index) => path.join(directory, `${names[index]}.log`));
const dataset = path.join(temporary, 'synthetic-dataset');
const checks = [];
const errors = [];
const restarts = [];
let application;
let page;
let runIds;
for (const directory of [profile, output, dataset, ...directories]) await mkdir(directory, { recursive: true });
await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
const event = (name, fields = {}) => ({ ts: '2026-10-09T10:00:00Z', event: name, ...fields });
const activity = [
  ...Array.from({ length: 230 }, (_, index) => event('step_saved', { step: index, message: `Synthetic activity ${index}` })),
  event({ malformed: true }, { step: { unsafeStep: true }, message: { nested: 'not a JSX child' } }),
  event('constructor', { message: 'Prototype-name fixture, not executable code.' }),
  event('execution_finished', { exc_type: 'SyntheticValidationError' }),
  event('node_evaluated', { is_buggy: true }),
  event('run_finished', { completed_steps: 3 }),
];
for (let index = 0; index < directories.length; index++) {
  await writeFile(path.join(directories[index], 'run_manifest.json'), JSON.stringify({ created_at: '2026-10-09T10:00:00Z', data_dir: dataset, task_description: `Synthetic log reader ${index}; no model training.`, config: { agent_steps: 3, model_for_all_agents: 'qa-never-called' } }));
  await writeFile(path.join(directories[index], 'events.jsonl'), activity.map(value => JSON.stringify(value)).join('\n') + '\nnot valid json\n{"unfinished":');
}
const initialText = Array.from({ length: 6_000 }, (_, index) => `[${String(index).padStart(5, '0')}] Synthetic output ${index % 57 === 0 ? 'needle' : 'plain'} · 仅供软件验收 ${'x'.repeat(30)}\n`).join('') + 'INITIAL_LATEST_SENTINEL\n';
await writeFile(logFiles[0], initialText);
await writeFile(logFiles[1], 'SECOND_RUN_PRIVATE_SENTINEL\n');

async function until(predicate, message, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(message);
    await new Promise(resolve => setTimeout(resolve, 40));
  }
}
const reader = () => page.locator('.run-log-reader');
const viewport = () => reader().locator('.run-log-viewport');
const content = () => reader().locator('.run-log-text');
const text = () => content().textContent();
const api = endpoint => page.evaluate(endpoint => window.biomaker.request(endpoint), endpoint);
const configure = values => application.evaluate((_electron, values) => { Object.assign(globalThis.__logReaderQA, values); }, values);
async function openRun(id) {
  await application.evaluate(({ BrowserWindow }, id) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:open-run', id), id);
  await until(() => reader().getAttribute('data-run-id').then(value => value === id), 'Run identity did not change');
  await page.getByRole('tab', { name: '运行日志', exact: true }).click();
  await reader().waitFor({ state: 'visible' });
  await until(() => viewport().getAttribute('data-source-id').then(Boolean), 'No log snapshot was displayed');
}
async function refresh() {
  const button = reader().getByRole('button', { name: '刷新日志', exact: true });
  await until(() => button.isEnabled(), 'Refresh log remained disabled');
  await button.click();
  await until(() => button.isEnabled(), 'Log refresh did not finish');
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}
async function follow() {
  if (await reader().getAttribute('data-following') !== 'true') await reader().getByRole('button', { name: '跟随最新', exact: true }).click();
  await until(() => reader().getAttribute('data-following').then(value => value === 'true'), 'Following did not resume');
}
async function pause() {
  if (await reader().getAttribute('data-following') === 'true') await reader().getByRole('button', { name: '暂停跟随', exact: true }).click();
  assert.equal(await reader().getAttribute('data-following'), 'false');
}
async function screenshot(name) {
  await reader().evaluate(element => {
    const scroller = element.closest('.workspace-content');
    if (scroller) scroller.scrollTop += element.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 65;
  });
  await page.screenshot({ path: path.join(output, name) });
}
async function releaseHeld() {
  await application.evaluate(() => { const qa = globalThis.__logReaderQA; for (const held of qa.held.splice(0)) held.release(); });
  await until(() => application.evaluate(() => globalThis.__logReaderQA.pending === 0), 'Held responses did not settle');
  await page.waitForTimeout(120);
}
async function restartService() {
  const before = await page.evaluate(() => window.biomaker.getServiceState());
  const canonical = await realpath(profile);
  const processes = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n')
    .filter(line => line.includes('/service/main.py') && line.includes('python') && (line.includes(profile) || line.includes(canonical)));
  assert.equal(processes.length, 1, 'Only the exact isolated QA sidecar may be stopped');
  process.kill(Number(processes[0].trim().split(/\s+/)[0]), 'SIGKILL');
  await until(async () => { const next = await page.evaluate(() => window.biomaker.getServiceState()); return next.status === 'ready' && next.revision > before.revision; }, 'The isolated service did not recover', 60_000);
  restarts.push({ from: before.revision, to: (await page.evaluate(() => window.biomaker.getServiceState())).revision });
}

try {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  await application.evaluate(() => {
    const originalFetch = globalThis.fetch;
    const qa = globalThis.__logReaderQA = { activeRun: 'qa-log-reader-first', active: true, failLatest: false, holdOlder: false, holdLatest: false, held: [], pending: 0, requests: [], forbidden: [] };
    globalThis.fetch = async (input, options) => {
      const url = new URL(String(input)); const method = String(options?.method || 'GET').toUpperCase();
      if (url.hostname !== '127.0.0.1' || !['GET', 'HEAD'].includes(method)) {
        qa.forbidden.push({ path: url.pathname, method });
        throw new Error('Log reader QA forbids provider calls, installations and run/settings mutations');
      }
      qa.requests.push({ path: url.pathname, query: url.search, method });
      const log = /\/api\/desktop\/runs\/[^/]+\/console$/.test(url.pathname);
      const older = log && url.searchParams.has('before');
      if (log && !older && qa.failLatest) return new Response('{"detail":"[stream:unavailable] Synthetic log read failure"}', { status: 404, headers: { 'Content-Type': 'application/json' } });
      const held = older && qa.holdOlder || log && !older && qa.holdLatest;
      if (held) { if (older) qa.holdOlder = false; else qa.holdLatest = false; qa.pending++; }
      const response = await originalFetch(input, options);
      const lifecycle = url.pathname === '/api/runs' || url.pathname === `/api/runs/${qa.activeRun}` || url.pathname === `/api/runs/${qa.activeRun}/updates`;
      if (held || lifecycle) {
        const raw = await response.text();
        let value = JSON.parse(raw);
        if (lifecycle) {
          const patch = run => run?.id === qa.activeRun && qa.active ? { ...run, status: 'running', detailed_status: 'running', allowed_actions: [], cleanup_pending: false } : run;
          value = Array.isArray(value) ? value.map(patch) : patch(value);
          if (url.pathname.endsWith('/updates')) value.run_revision = `${value.run_revision}:synthetic-active-${qa.active}`;
        }
        if (held) await new Promise(resolve => qa.held.push({ older, release: () => { qa.pending--; resolve(); } }));
        return new Response(JSON.stringify(value), { status: response.status, headers: response.headers });
      }
      return response;
    };
  });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => errors.push(error.message));
  await until(async () => { try { return (await page.evaluate(() => window.biomaker.getServiceState())).status === 'ready'; } catch { return false; } }, 'The real sidecar did not start', 60_000);
  await page.locator('.sidebar-settings').waitFor();
  await page.evaluate(() => { window.__logReaderCopied = []; navigator.clipboard.writeText = async value => { window.__logReaderCopied.push(value); }; });
  const runs = await api('/api/runs');
  runIds = names.map(name => runs.find(run => run.id === name || run.name === name)?.id);
  assert(runIds.every(Boolean), 'Both synthetic histories must be cataloged');
  await configure({ activeRun: runIds[0] });
  await openRun(runIds[0]);
  await until(async () => (await text()).includes('INITIAL_LATEST_SENTINEL'), 'Latest initial text was missing');
  assert.equal(await reader().getAttribute('data-following'), 'true');
  await until(() => viewport().evaluate(element => element.scrollHeight - element.clientHeight - element.scrollTop < 4), 'Visible following reader did not start at the bottom');
  assert.equal(await text(), (await api(`/api/desktop/runs/${runIds[0]}/console`)).text);
  checks.push('real-bounded-tail-renders-exact-text-and-initially-follows-when-revealed');

  await appendFile(logFiles[0], 'FOLLOWED_APPEND_SENTINEL\n'); await refresh();
  await until(async () => (await text()).includes('FOLLOWED_APPEND_SENTINEL'), 'Appended text was not followed');
  assert(await viewport().evaluate(element => element.scrollHeight - element.clientHeight - element.scrollTop < 4));
  await viewport().hover(); await page.mouse.wheel(0, -700);
  await until(() => reader().getAttribute('data-following').then(value => value === 'false'), 'Scrolling up did not pause');
  const pausedText = await text(); const pausedTop = await viewport().evaluate(element => element.scrollTop);
  await appendFile(logFiles[0], 'PAUSED_APPEND_SENTINEL\n'); await refresh();
  await reader().getByRole('button', { name: '有新日志 · 跟随最新', exact: true }).waitFor();
  assert.equal(await text(), pausedText);
  assert(Math.abs(await viewport().evaluate(element => element.scrollTop) - pausedTop) < 3);
  await follow(); assert((await text()).includes('PAUSED_APPEND_SENTINEL'));
  checks.push('append-follows-bottom-scroll-up-freezes-text-position-and-explicit-resume-adopts-latest');

  await viewport().evaluate(element => { const node = element.querySelector('pre').firstChild; const range = document.createRange(); range.setStart(node, 0); range.setEnd(node, 20); const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range); });
  await until(() => reader().getAttribute('data-following').then(value => value === 'false'), 'Text selection did not pause');
  const selectedText = await page.evaluate(() => window.getSelection().toString());
  const selectedWindow = await text();
  await appendFile(logFiles[0], 'SELECTED_APPEND_SENTINEL\n'); await refresh();
  assert.equal(await text(), selectedWindow); assert.equal(await page.evaluate(() => window.getSelection().toString()), selectedText);
  await page.evaluate(() => window.getSelection().removeAllRanges());
  checks.push('selected-text-and-displayed-window-survive-background-append');

  await follow();
  const modifier = process.platform === 'darwin' ? 'Meta' : 'Control';
  await page.keyboard.press(`${modifier}+f`);
  const search = reader().getByRole('searchbox', { name: '在当前片段中搜索', exact: true });
  await search.fill('needle');
  assert.equal(await reader().getAttribute('data-following'), 'false');
  await until(() => content().locator('mark').count().then(count => count > 2), 'Literal search did not mark expected matches');
  assert.equal(await content().locator('mark.is-current').getAttribute('data-log-match'), '0');
  await search.press('Enter'); assert.equal(await content().locator('mark.is-current').getAttribute('data-log-match'), '1');
  await search.press('Shift+Enter'); assert.equal(await content().locator('mark.is-current').getAttribute('data-log-match'), '0');
  await search.fill('[99999].*'); assert.equal(await content().locator('mark').count(), 0);
  await search.fill('needle');
  const copiedWindow = await text();
  await reader().getByRole('button', { name: '复制当前片段', exact: true }).click();
  assert.equal(await page.evaluate(() => window.__logReaderCopied.at(-1)), copiedWindow);
  assert.equal(await text(), copiedWindow);
  await page.getByRole('tab', { name: '活动', exact: true }).click();
  await page.keyboard.press(`${modifier}+f`);
  assert.equal(await page.evaluate(() => !!document.activeElement?.closest('.run-log-reader')), false, 'Hidden reader stole search focus');
  await page.getByRole('tab', { name: '运行日志', exact: true }).click();
  assert.equal(await search.inputValue(), 'needle'); assert.equal(await reader().getAttribute('data-following'), 'false');
  await search.press('Escape'); assert.equal(await search.count(), 0);
  assert.equal(await viewport().evaluate(element => document.activeElement === element), true);
  assert.equal(await reader().getAttribute('data-following'), 'false');
  await screenshot('01-search-copy-and-paused-light.png');
  checks.push('literal-search-match-navigation-shortcuts-exact-window-copy-and-hidden-tab-state');

  const latest = await api(`/api/desktop/runs/${runIds[0]}/console`);
  assert(latest.has_older && latest.older_cursor);
  await reader().getByRole('button', { name: '查看此前片段', exact: true }).click();
  await until(() => reader().getAttribute('data-older-page').then(value => value === 'true'), 'Earlier window did not appear');
  assert(Number(await viewport().getAttribute('data-end')) <= latest.start);
  const earlierText = await text(); assert(!earlierText.includes('SELECTED_APPEND_SENTINEL'));
  const earlier = await api(`/api/desktop/runs/${runIds[0]}/console?before=${encodeURIComponent(latest.older_cursor)}`);
  assert.equal(earlierText, earlier.text);
  await configure({ holdOlder: true });
  await reader().getByRole('button', { name: '查看此前片段', exact: true }).click();
  await until(() => application.evaluate(() => globalThis.__logReaderQA.held.length === 1), 'Older response was not held');
  await follow(); const resumedText = await text();
  await releaseHeld(); assert.equal(await text(), resumedText); assert.equal(await reader().getAttribute('data-older-page'), 'false');
  checks.push('earlier-page-replaces-never-merges-and-late-response-cannot-override-follow-latest');

  await pause();
  const beforeRotation = await text();
  const replacement = `${logFiles[0]}.replacement`;
  await writeFile(replacement, 'ROTATED_FILE_SENTINEL\n'); await rename(replacement, logFiles[0]);
  await refresh();
  await reader().getByText(/日志来源已更新|日志文件已更换/).waitFor();
  assert.equal(await text(), beforeRotation);
  assert.equal(await reader().getByRole('button', { name: '查看此前片段', exact: true }).isDisabled(), true);
  await follow(); assert.equal(await text(), 'ROTATED_FILE_SENTINEL\n');
  await writeFile(logFiles[0], 'TRUNCATED_SENTINEL\n'); await refresh();
  await until(async () => await text() === 'TRUNCATED_SENTINEL\n', 'Truncated latest log did not replace text');
  checks.push('real-file-replacement-keeps-paused-window-reports-source-change-and-truncation-never-merges');

  await configure({ failLatest: true }); await refresh();
  await reader().locator('.run-log-notice.is-error').filter({ hasText: '日志暂时无法读取。已显示的内容仍然保留，可以稍后重试。' }).waitFor();
  assert.equal(await text(), 'TRUNCATED_SENTINEL\n');
  await configure({ failLatest: false }); await refresh();
  assert.equal(await reader().locator('.run-log-notice.is-error').count(), 0);
  const missing = `${logFiles[0]}.missing`; await rename(logFiles[0], missing); await refresh();
  await reader().getByText('当前日志文件不可用，保留上次读取的内容。', { exact: true }).waitFor();
  assert.equal(await text(), 'TRUNCATED_SENTINEL\n');
  assert.equal(await reader().getByRole('button', { name: '保存完整日志', exact: true }).isDisabled(), true);
  await rename(missing, logFiles[0]); await refresh();
  checks.push('failed-read-and-missing-file-preserve-last-text-with-explicit-state-and-recovery');

  await writeFile(logFiles[0], 'A'.repeat(320_000) + 'LONG_LINE_FINAL_SENTINEL'); await refresh();
  await until(async () => (await text()).endsWith('LONG_LINE_FINAL_SENTINEL'), 'Valid maximum-byte tail lost its final text');
  const longWindow = await api(`/api/desktop/runs/${runIds[0]}/console`);
  assert.equal(await text(), longWindow.text); assert.equal((await text()).length, 262_144);
  await reader().getByText('片段从一行中间开始，首行内容不完整。', { exact: true }).waitFor();
  await page.keyboard.press(`${modifier}+f`); await search.fill('A');
  assert.equal(await content().locator('mark').count(), 1_000);
  await reader().getByText(/仅高亮前 1,000 处匹配/).waitFor();
  await search.press('Escape'); await follow();
  await writeFile(logFiles[0], Buffer.from([0x61, 0xff, 0xfe, 0x62, 0x0a])); await refresh();
  await until(async () => await text() === 'a\ufffd\ufffdb\n', 'Invalid UTF-8 display did not match bounded decoder');
  await reader().getByText(/部分字节无法解码为文本/).waitFor();
  checks.push('maximum-byte-long-line-retains-latest-text-bounds-highlights-and-labels-invalid-utf8');

  const destination = path.join(temporary, 'exported-full-log.bin');
  await application.evaluate(({ dialog }) => { dialog.showSaveDialog = async () => ({ canceled: true, filePath: undefined }); });
  await reader().getByRole('button', { name: '保存完整日志', exact: true }).click();
  await until(() => reader().getByRole('button', { name: '保存完整日志', exact: true }).isEnabled(), 'Canceled export did not release controls');
  await application.evaluate(({ dialog }, destination) => { dialog.showSaveDialog = async () => ({ canceled: false, filePath: destination }); }, destination);
  await reader().getByRole('button', { name: '保存完整日志', exact: true }).click();
  await until(async () => { try { return (await readFile(destination)).equals(await readFile(logFiles[0])); } catch { return false; } }, 'Native full log export did not preserve original bytes');
  assert.deepEqual(await readFile(destination), Buffer.from([0x61, 0xff, 0xfe, 0x62, 0x0a]));
  checks.push('native-full-log-export-cancellation-and-invalid-utf8-original-byte-preservation');

  await writeFile(logFiles[0], initialText); await refresh(); await pause();
  const priorServiceText = await text();
  await configure({ holdOlder: true }); await reader().getByRole('button', { name: '查看此前片段', exact: true }).click();
  await until(() => application.evaluate(() => globalThis.__logReaderQA.held.length === 1), 'Pre-restart older response was not held');
  await restartService(); await releaseHeld();
  assert.equal(await text(), priorServiceText); assert.equal(await reader().getAttribute('data-older-page'), 'false');
  await follow(); await refresh();
  checks.push('real-sidecar-restart-invalidates-held-older-response-and-preserves-paused-text');

  await configure({ active: false }); await refresh();
  await pause();
  await configure({ holdLatest: true });
  await application.evaluate(({ BrowserWindow }, id) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:open-run', id), runIds[1]);
  await until(() => reader().getAttribute('data-run-id').then(value => value === runIds[1]), 'Terminal B reader identity did not change');
  await page.getByRole('tab', { name: '运行日志', exact: true }).click();
  await until(() => application.evaluate(() => globalThis.__logReaderQA.held.length === 1), 'Terminal B snapshot was not held');
  assert(!(await text()).includes('INITIAL_LATEST_SENTINEL'), 'Terminal B briefly exposed A while its own snapshot was pending');
  await releaseHeld();
  await until(async () => await text() === 'SECOND_RUN_PRIVATE_SENTINEL\n', 'Terminal B displayed the previous run snapshot');
  assert(!(await text()).includes('INITIAL_LATEST_SENTINEL'));
  await openRun(runIds[0]);
  await until(async () => (await text()).includes('INITIAL_LATEST_SENTINEL'), 'Terminal A failed to restore its own snapshot');
  assert(!(await text()).includes('SECOND_RUN_PRIVATE_SENTINEL'));
  checks.push('terminal-run-switch-with-held-response-never-initializes-paused-reader-from-previous-run-snapshot');

  await page.getByRole('tab', { name: '活动', exact: true }).click();
  const activityView = page.locator('.run-activity');
  await activityView.locator('.event-item.event-error').getByRole('button', { name: /候选方案执行或校验失败/ }).waitFor();
  await activityView.locator('.event-item.event-error').getByRole('button', { name: /候选方案评估未通过/ }).waitFor();
  await activityView.getByRole('button', { name: /^constructor/ }).waitFor();
  const malformed = activityView.getByRole('button', { name: /^event / });
  await malformed.click();
  await activityView.getByText(/"unsafeStep": true/).waitFor();
  assert.equal(await malformed.locator('.event-step').count(), 0);
  await activityView.getByText(/不完整或无法解析的记录/).waitFor();
  await activityView.getByRole('button', { name: '查看更早活动', exact: true }).click();
  await activityView.getByRole('button', { name: '返回最新活动', exact: true }).waitFor();
  await activityView.getByRole('button', { name: '返回最新活动', exact: true }).click();
  await activityView.getByText('最近记录的活动', { exact: true }).waitFor();
  checks.push('activity-malformed-fields-prototype-name-and-explicit-execution-evaluation-failures-render-safely');

  await page.getByRole('tab', { name: '运行日志', exact: true }).click();
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  await page.locator('.sidebar-settings').click();
  await page.getByRole('button', { name: '通用', exact: true }).click();
  await page.getByRole('button', { name: '深色', exact: true }).click();
  await page.keyboard.press('Escape');
  await reader().waitFor({ state: 'visible' });
  await page.keyboard.press(`${modifier}+f`); await search.fill('needle');
  await search.press('Enter');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  assert.equal(await reader().getByRole('button', { name: '自动换行', exact: true }).getAttribute('aria-pressed'), 'true');
  await reader().getByRole('button', { name: '自动换行', exact: true }).click();
  assert.equal(await reader().getByRole('button', { name: '自动换行', exact: true }).getAttribute('aria-pressed'), 'false');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await screenshot('02-compact-dark-search.png');
  await reader().getByRole('button', { name: '自动换行', exact: true }).click();
  await screenshot('03-compact-dark-wrap.png');
  checks.push('compact-dark-and-light-reader-search-wrap-keyboard-and-controls-fit-without-page-overflow');

  const native = await application.evaluate(() => ({ requests: globalThis.__logReaderQA.requests, forbidden: globalThis.__logReaderQA.forbidden, pending: globalThis.__logReaderQA.pending }));
  assert.deepEqual(native.forbidden, []); assert.equal(native.pending, 0); assert.deepEqual(errors, []);
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  const reports = (await api('/api/runs')).map(run => run.id).sort(); assert.deepEqual(reports, [...runIds].sort());
  checks.push('no-provider-engine-install-real-experiment-api-mutation-or-renderer-errors');
  const report = { passed: true, mode: 'Real Electron and sidecar bounded file reading, paging, rotation, native export and sidecar restart; synthetic records and intercepted lifecycle/response timing/failure/clipboard. No experiment or provider acceptance.', executablePath: executablePath || 'development Electron', profile, output, runIds, checks, requests: native.requests, forbiddenRequests: native.forbidden, rendererErrors: errors, restarts };
  await writeFile(path.join(output, 'acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  const native = application ? await application.evaluate(() => {
    const qa = globalThis.__logReaderQA;
    return qa ? { active: qa.active, failLatest: qa.failLatest, holdOlder: qa.holdOlder, holdLatest: qa.holdLatest, pending: qa.pending, held: qa.held.map(item => ({ older: item.older })), requests: qa.requests, forbidden: qa.forbidden } : null;
  }).catch(() => null) : null;
  await writeFile(path.join(output, 'failure.json'), JSON.stringify({ passed: false, profile, checks, error: String(error), rendererErrors: errors, native }, null, 2));
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-6_000));
  }
  throw error;
} finally {
  if (application) {
    await application.evaluate(({ BrowserWindow }) => { for (const held of globalThis.__logReaderQA?.held || []) held.release(); for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
    await application.close();
  }
}
