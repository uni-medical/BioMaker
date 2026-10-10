// Real Electron and sidecar acceptance for detail navigation and status evidence.
// Isolated synthetic histories only. No response/lifecycle mocking, engine
// installation, provider request, training, or mutation of the user's profile.
// Run only after the release owner prepares and authorizes the runtime.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { appendFile, mkdir, mkdtemp, realpath, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron as electron } from '@playwright/test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const temporary = await mkdtemp(path.join(os.tmpdir(), 'biomaker-run-navigation-'));
const profile = path.join(temporary, 'profile');
const output = path.resolve(process.env.BIOMAKER_QA_OUTPUT || path.join(root, 'output/run-navigation'));
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
const env = Object.fromEntries(['HOME', 'USER', 'LOGNAME', 'LANG', 'TMPDIR', 'PATH'].filter(key => process.env[key]).map(key => [key, process.env[key]]));
Object.assign(env, { BIOMAKER_DESKTOP_USER_DATA: profile, BIOMAKER_DESKTOP_HOME: profile, PYTHONDONTWRITEBYTECODE: '1', NO_PROXY: '127.0.0.1,localhost' });
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const ids = { complete: 'qa-navigation-complete', unknown: 'qa-navigation-unknown', failed: 'qa-navigation-failed' };
const unknownError = 'SyntheticError[QUUX-2718]: unrecognized diagnosis detail / 未知原始错误';
const checks = [], geometry = [], rendererErrors = [], requests = [], forbiddenRequests = [], processChecks = [];
let application, page, appInfo, failure, selectedId;
const directory = id => path.join(profile, 'workspace/logs', id);
const event = (name, fields = {}) => ({ ts: '2026-10-09T10:00:00Z', event: name, ...fields });
const jsonl = value => JSON.stringify(value) + '\n';
const api = route => page.evaluate(route => window.biomaker.request(route), route);
const frame = () => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const tabs = () => page.locator('.run-tabs');
const panel = () => page.locator('#run-detail-panel');
const tab = id => page.locator(`#run-tab-${id}`);
const scroller = () => page.locator('.workspace-content');

async function until(predicate, reason, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
    if (Date.now() > deadline) throw new Error(reason);
    await new Promise(resolve => setTimeout(resolve, 40));
  }
}
async function seed() {
  await mkdir(output, { recursive: true });
  await mkdir(profile, { recursive: true });
  await writeFile(path.join(profile, 'desktop-preferences.json'), JSON.stringify({ language: 'zh', theme: 'light', notifications: false, directories: [] }));
  const nodes = Array.from({ length: 32 }, (_, index) => ({ id: `navigation-candidate-${index}`, step: index, is_buggy: false, agent_name: 'Draft', plan: `Synthetic candidate ${index}; no execution.`, code: `# Synthetic candidate ${index}; never executed.\n`, metric: { value: index / 100, maximize: true } }));
  for (const [kind, id] of Object.entries(ids)) {
    const target = directory(id);
    await mkdir(target, { recursive: true });
    await writeFile(path.join(target, 'run_manifest.json'), JSON.stringify({ created_at: '2026-10-09T10:00:00Z', data_dir: path.join(temporary, 'long-synthetic-dataset-directory-for-layout-regression'.repeat(4)), task_description: 'Synthetic navigation and evidence acceptance. These records were not produced by model training.', config: { agent_steps: 32, final_retrain_enabled: false, model_for_all_agents: 'qa-never-called' } }));
    const values = [event('run_started')];
    if (kind === 'complete') values.push(...Array.from({ length: 220 }, (_, step) => event('tool_observation', { step, message: `Synthetic recorded observation ${step}` })), event('step_saved', { node_id: nodes.at(-1).id, best_node_id: nodes.at(-1).id }), event('run_finished', { completed_steps: 32 }));
    if (kind === 'failed') values.push(event('run_failed', { error: unknownError }));
    await writeFile(path.join(target, 'events.jsonl'), values.map(jsonl).join(''));
    await writeFile(path.join(target, `${id}.log`), Array.from({ length: 240 }, (_, row) => `Synthetic console line ${row}; no process ran.\n`).join(''));
    if (kind === 'complete') {
      await writeFile(path.join(target, 'journal.json'), JSON.stringify({ nodes, node2parent: {} }));
      await writeFile(path.join(target, 'best_solution.py'), nodes.at(-1).code);
      await writeFile(path.join(target, 'report.md'), '# Synthetic report\n\nNavigation fixture only.\n');
      await writeFile(path.join(target, 'results.tsv'), 'step\tnode_id\tmetric\tmaximize\tstatus\n' + nodes.map(node => `${node.step}\t${node.id}\t${node.metric.value}\ttrue\tsucceeded\n`).join(''));
      // Files is intentionally tall; tab changes must not inherit its offset.
      for (let index = 0; index < 36; index++) await writeFile(path.join(target, `navigation-${String(index).padStart(2, '0')}.txt`), `Synthetic local artifact ${index}.\n`);
    }
  }
  await assert.rejects(stat(path.join(profile, 'state/runs.sqlite3')), { code: 'ENOENT' });
}
async function nativeState() {
  return application.evaluate(() => ({ requests: globalThis.__runNavigationQA.requests, forbidden: globalThis.__runNavigationQA.forbidden, inflight: globalThis.__runNavigationQA.inflight }));
}
async function settled() {
  await until(() => nativeState().then(value => value.inflight === 0), 'Real sidecar reads did not settle');
  await frame();
}
async function launch() {
  application = await electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
  page = await application.firstWindow();
  page.setDefaultTimeout(20_000);
  page.on('pageerror', error => rendererErrors.push(error.message));
  await application.evaluate(({ BrowserWindow }) => {
    const original = globalThis.fetch;
    const qa = globalThis.__runNavigationQA = { requests: [], forbidden: [], inflight: 0 };
    globalThis.fetch = async (input, options) => {
      const url = new URL(typeof input === 'string' ? input : input.url || String(input));
      const method = String(options?.method || (typeof input === 'object' && input.method) || 'GET').toUpperCase();
      if (url.hostname !== '127.0.0.1' || !['GET', 'HEAD'].includes(method)) {
        qa.forbidden.push({ path: url.pathname, method });
        throw new Error('Navigation QA forbids mutations, provider requests and installation');
      }
      const entry = { path: url.pathname, method };
      qa.requests.push(entry); qa.inflight++;
      try { const response = await original(input, options); entry.status = response.status; return response; }
      finally { qa.inflight--; }
    };
    BrowserWindow.getAllWindows()[0].setSize(1180, 780);
  });
  await until(async () => {
    const state = await page.evaluate(() => window.biomaker.getServiceState()).catch(() => null);
    if (state?.status === 'failed') throw new Error(`The sidecar failed: ${state.message}`);
    return state?.status === 'ready';
  }, 'Real sidecar did not become ready', 60_000);
  await page.locator('.sidebar-settings').waitFor();
  appInfo = await page.evaluate(() => window.biomaker.getAppInfo());
  await settled();
}
async function openRun(id) {
  await application.evaluate(({ BrowserWindow }, id) => BrowserWindow.getAllWindows()[0].webContents.send('biomaker:open-run', id), id);
  await until(() => page.locator('.run-intro h1').textContent().then(value => value === id), `Did not open ${id}`);
  selectedId = id;
  await panel().waitFor();
  await settled();
}
async function selectTab(id) {
  await tab(id).click();
  await until(() => panel().getAttribute('aria-labelledby').then(value => value === `run-tab-${id}`), `Did not select ${id}`);
  if (id === 'results') await until(() => page.locator('.run-results').getAttribute('aria-busy').then(value => value === 'false'), 'Results did not load');
  if (id === 'candidates') await page.locator('.candidates-view').waitFor();
  if (id === 'artifacts') await page.locator('.artifact-row').first().waitFor();
  if (id === 'activity') await page.locator('.run-activity').waitFor();
  await settled();
}
async function captureGeometry(label) {
  const value = await page.evaluate(() => {
    const box = element => { const value = element.getBoundingClientRect(); return { top: value.top, bottom: value.bottom, left: value.left, right: value.right, height: value.height }; };
    const scroller = document.querySelector('.workspace-content');
    return { panel: box(document.querySelector('#run-detail-panel')), tabs: box(document.querySelector('.run-tabs')), scroller: box(scroller), scrollTop: scroller.scrollTop, scrollHeight: scroller.scrollHeight, selected: document.querySelector('.run-tabs [aria-selected="true"]').id, focused: document.activeElement?.id, horizontalOverflow: document.documentElement.scrollWidth > innerWidth };
  });
  geometry.push({ label, ...value });
  return value;
}
async function assertPanelStart(label, id, focus = false) {
  await until(() => tab(id).getAttribute('aria-selected').then(value => value === 'true'), `${label}: selection did not change`);
  await settled();
  const value = await captureGeometry(label);
  assert(value.panel.top >= value.tabs.bottom - 2, `${label}: panel starts under sticky tabs`);
  assert(value.panel.top < value.scroller.bottom - 40, `${label}: panel start is outside the viewport`);
  assert.equal(value.horizontalOverflow, false);
  assert.equal(await panel().getAttribute('aria-labelledby'), `run-tab-${id}`);
  if (focus) assert.equal(value.focused, `run-tab-${id}`, `${label}: keyboard focus did not follow the selected tab`);
}
async function deepFiles() {
  await selectTab('artifacts');
  await page.evaluate(() => {
    const scroller = document.querySelector('.workspace-content');
    const panel = document.querySelector('#run-detail-panel');
    const tabs = document.querySelector('.run-tabs');
    const stickyBottom = scroller.getBoundingClientRect().top + tabs.getBoundingClientRect().height;
    scroller.scrollTop += panel.getBoundingClientRect().top - stickyBottom + 850;
  });
  await frame();
  const value = await captureGeometry('deep-files-source');
  assert(value.panel.top < value.tabs.bottom - 600, 'Fixture did not create a genuinely deep inherited offset');
  assert(Math.abs(value.tabs.top - value.scroller.top) < 3, 'Tabs did not become sticky in the source panel');
}
async function refresh() {
  await settled();
  const before = (await nativeState()).requests.length;
  await page.getByRole('button', { name: /^(刷新|Refresh)$/ }).click();
  await until(() => nativeState().then(value => value.requests.slice(before).some(request => request.path === `/api/runs/${selectedId}/updates`)), 'Manual refresh did not read selected revisions');
  await settled();
}
async function language(value) {
  await assertShellFits(`before-settings-${value}`);
  await page.locator('.sidebar-settings').click();
  await page.getByRole('button', { name: /^(通用|General)$/ }).click();
  await page.getByLabel(/^(界面语言|Interface language)$/).selectOption(value);
  await until(() => page.evaluate(() => window.biomaker.getPreferences()).then(preferences => preferences.language === value), 'Language preference did not persist');
  await frame();
  await assertShellFits(`settings-language-${value}`);
  await page.keyboard.press('Escape');
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  await settled();
  await assertShellFits(`returned-from-settings-${value}`);
}
async function assertShellFits(label) {
  const value = await page.evaluate(() => {
    const box = element => { const r = element.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, height: r.height }; };
    const shell = document.querySelector('.app-shell');
    const inspector = document.querySelector('.inspector');
    const reader = document.querySelector('.inspector-scroll');
    return { height: innerHeight, width: innerWidth, shell: box(shell), shellScroll: shell.scrollTop,
      shellOverflow: shell.scrollHeight - shell.clientHeight, header: box(document.querySelector('.app-header')),
      statusbar: box(document.querySelector('.workspace-statusbar')),
      inspector: inspector && getComputedStyle(inspector).display !== 'none' ? box(inspector) : null,
      reader: reader && inspector && getComputedStyle(inspector).display !== 'none' ? { ...box(reader), clientHeight: reader.clientHeight, scrollHeight: reader.scrollHeight } : null,
      horizontalOverflow: document.documentElement.scrollWidth > innerWidth };
  });
  geometry.push({ label, ...value });
  assert.equal(value.shellScroll, 0, label);
  assert(value.shellOverflow <= 2, `${label}: app shell overflows`);
  assert(value.header.top >= 0 && value.header.bottom <= value.height, `${label}: topbar is outside window`);
  assert(value.statusbar.top >= 0 && value.statusbar.bottom <= value.height + 1, `${label}: statusbar is outside window`);
  assert.equal(value.horizontalOverflow, false, label);
  if (value.inspector) {
    assert(value.inspector.top >= 0 && value.inspector.bottom <= value.height + 1, `${label}: inspector escapes window`);
    assert(value.reader.top >= value.inspector.top && value.reader.bottom <= value.inspector.bottom + 1, `${label}: inspector scroller escapes panel`);
    assert(value.reader.clientHeight > 100, `${label}: inspector has no usable scroll viewport`);
  }
}
async function screenshot(name, introduction = false) {
  if (introduction) await scroller().evaluate(element => { element.scrollTop = 0; });
  await frame();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({ path: path.join(output, name) });
}
function assertLedger(run, expected) {
  for (const key of ['status', 'detailed_status', 'status_source', 'status_reason', 'error', 'read_only', 'managed', 'allowed_actions']) assert.deepEqual(run[key], expected[key], `Tail changed saved lifecycle field ${key}`);
}
async function assertNeutralNotice(text) {
  const notice = page.locator('.run-intro .notice').filter({ hasText: text });
  await notice.waitFor();
  assert.equal(await notice.count(), 1);
  assert.equal(await notice.getAttribute('role'), 'status');
  assert(!String(await notice.getAttribute('class')).split(/\s+/).includes('danger'));
}
async function assertEvidenceInBothLanguages(kind) {
  const localized = {
    missing: ['当前读取的 5,000 条历史事件中未包含结束事件。上方状态来自已保存的实验记录。', 'The 5,000 historical events currently read contain no terminal event. The status above comes from the saved experiment record.'],
    conflict: ['当前历史片段记录了“失败”事件，与已保存状态不同。上方仍显示已保存的实验状态，可在活动中核对事件。', 'The current history records a failure event that differs from the saved status. The status above is preserved; inspect the events in Activity.'],
    unknown: ['登记这条实验记录时，读取的历史片段未包含结束事件。状态保留登记时的判断，无法据此确认原进程是否仍在运行。', 'The history read when this experiment was registered contained no terminal event. Its saved status preserves that assessment and does not establish whether the original process is still running.'],
  };
  for (const [index, lang] of ['zh', 'en'].entries()) {
    if ((await page.evaluate(() => window.biomaker.getPreferences())).language !== lang) await language(lang);
    await assertNeutralNotice(localized[kind][index]);
    const intro = await page.locator('.run-intro').innerText();
    assert(!intro.includes('The process is not known to be alive'));
    assert(!intro.includes('No terminal lifecycle event was found in the last up to'));
    if (kind !== 'unknown') assert.equal(await page.locator('.status-label.status-complete').count(), 1);
    else assert.equal(await page.locator('.status-label.status-incomplete').count(), 1);
    await screenshot(`evidence-${kind}-${lang}.png`, true);
  }
  checks.push(`${kind}-evidence-localizes-in-chinese-and-english-with-neutral-tone-and-preserved-status`);
}
async function close() {
  if (!application) return;
  const native = await nativeState().catch(() => null);
  if (native) { requests.push(...native.requests); forbiddenRequests.push(...native.forbidden); }
  const current = application;
  // The window has finished loading before any fixture interaction; destroy it
  // to release renderer unload guards, then let normal quit stop the sidecar.
  await current.evaluate(({ BrowserWindow }) => { for (const window of BrowserWindow.getAllWindows()) window.destroy(); }).catch(() => {});
  try { await current.close(); } finally { application = null; page = null; }
  const canonicalProfile = await realpath(profile);
  let remaining = [];
  await until(() => {
    remaining = execFileSync('/bin/ps', ['-axo', 'pid=,command='], { encoding: 'utf8' }).split('\n').filter(line => line.includes(profile) || line.includes(canonicalProfile)).map(line => line.trim());
    return remaining.length === 0;
  }, 'An exact isolated QA profile process remains after exit', 15_000);
  processChecks.push({ profile, canonicalProfile, remainingProcesses: remaining });
}

try {
  await seed();
  await launch();
  const runs = await api('/api/runs');
  assert.deepEqual(runs.map(run => run.id).sort(), Object.values(ids).sort());
  for (const run of runs) { assert.equal(run.read_only, true); assert.equal(run.managed, false); assert.deepEqual(run.allowed_actions, []); }
  const saved = Object.fromEntries(runs.map(run => [run.id, run]));
  assert.equal(saved[ids.complete].status, 'complete');
  assert.equal(saved[ids.unknown].detailed_status, 'orphaned');
  assert.equal(saved[ids.failed].status_reason, unknownError);
  checks.push('real-startup-discovers-three-readonly-fixtures-without-engine-or-provider');
  await openRun(ids.complete);
  for (const [width, height] of [[1180, 780], [900, 620]]) {
    await application.evaluate(({ BrowserWindow }, size) => BrowserWindow.getAllWindows()[0].setContentSize(...size), [width, height]);
    await frame();
    if (!await page.locator('.inspector').count()) await page.getByRole('button', { name: '显示实验信息', exact: true }).click();
    await page.locator('.inspector-scroll').waitFor();
    assert(await page.locator('.inspector-scroll').evaluate(element => element.scrollHeight > element.clientHeight + 100), 'Long-path inspector must actually overflow internally');
    await page.locator('.inspector-scroll').evaluate(element => { element.scrollTop = element.scrollHeight; });
    await language('en');
    await page.locator('.inspector-scroll').waitFor();
    await language('zh');
    await page.locator('.inspector-scroll').waitFor();
    await screenshot(`layout-settings-roundtrip-${width}-${height}.png`, true);
  }
  await application.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setContentSize(1360, 900));
  await frame();
  checks.push('small-and-minimum-windows-retain-topbar-statusbar-and-bounded-inspector-after-settings-language-roundtrips');
  await selectTab('results');
  await scroller().evaluate(element => { element.scrollTop = 0; });
  await selectTab('artifacts');
  assert.equal(await scroller().evaluate(element => element.scrollTop), 0, 'Changing tabs at the top must retain the visible introduction');
  checks.push('tab-change-at-top-keeps-introduction-visible');

  for (const target of ['results', 'activity', 'candidates', 'log']) {
    await deepFiles();
    await selectTab(target);
    await assertPanelStart(`click-${target}`, target);
  }
  checks.push('clicking-each-other-tab-from-deep-files-exposes-new-panel-start');
  for (const [key, target] of [['ArrowRight', 'log'], ['ArrowLeft', 'candidates'], ['Home', 'results'], ['End', 'log']]) {
    await deepFiles();
    await tab('artifacts').evaluate(element => element.focus({ preventScroll: true }));
    await page.keyboard.press(key);
    await assertPanelStart(`keyboard-${key}`, target, true);
  }
  await screenshot('01-keyboard-tab-start.png');
  checks.push('arrow-home-end-navigation-preserves-roving-focus-and-exposes-panel-start');

  await selectTab('artifacts');
  const control = page.locator('.artifact-row').nth(6).getByRole('button', { name: '预览', exact: true });
  await control.evaluate(element => {
    const scroller = element.closest('.workspace-content');
    const tabs = document.querySelector('.run-tabs');
    const stickyBottom = scroller.getBoundingClientRect().top + tabs.getBoundingClientRect().height;
    scroller.scrollTop += element.getBoundingClientRect().top - stickyBottom + tabs.getBoundingClientRect().height / 2;
  });
  await frame();
  const hidden = await control.evaluate(element => ({ target: element.getBoundingClientRect().top, tabsTop: document.querySelector('.run-tabs').getBoundingClientRect().top, tabsBottom: document.querySelector('.run-tabs').getBoundingClientRect().bottom }));
  assert(hidden.target >= hidden.tabsTop && hidden.target < hidden.tabsBottom, 'Fixture control is not under sticky tabs');
  await control.evaluate(element => element.focus({ preventScroll: true }));
  await frame();
  const revealed = await control.evaluate(element => ({ top: element.getBoundingClientRect().top, bottom: element.getBoundingClientRect().bottom, tabsBottom: document.querySelector('.run-tabs').getBoundingClientRect().bottom, viewportBottom: element.closest('.workspace-content').getBoundingClientRect().bottom, focused: document.activeElement === element }));
  assert(revealed.focused && revealed.top >= revealed.tabsBottom + 6 && revealed.bottom <= revealed.viewportBottom, 'Focused control remained hidden under sticky tabs');
  geometry.push({ label: 'focus-reveal', before: hidden, after: revealed });
  await screenshot('02-focus-revealed-below-tabs.png');
  checks.push('focus-under-sticky-tabs-reveals-the-real-file-control');

  await tab('artifacts').evaluate(element => element.focus({ preventScroll: true }));
  await scroller().evaluate(element => { element.scrollTop += 350; });
  await control.evaluate(element => element.focus({ preventScroll: true }));
  await frame();
  assert(await control.evaluate(element => element.getBoundingClientRect().top >= document.querySelector('.run-tabs').getBoundingClientRect().bottom + 6), 'A focused control entirely above sticky tabs was not exposed');
  checks.push('focus-entirely-above-sticky-tabs-is-exposed');

  await selectTab('results'); await selectTab('artifacts');
  const box = await scroller().boundingBox();
  const beforeWheel = await scroller().evaluate(element => element.scrollTop);
  await page.mouse.move(box.x + box.width * 0.6, box.y + box.height * 0.7);
  await page.mouse.wheel(0, 620);
  await until(() => scroller().evaluate(element => element.scrollTop).then(value => value > beforeWheel + 300), 'Manual wheel did not move the reading position');
  await frame();
  const reading = await captureGeometry('manual-reading-before-refresh');
  await refresh();
  const refreshed = await captureGeometry('manual-reading-after-refresh');
  assert(Math.abs(refreshed.scrollTop - reading.scrollTop) <= 2, 'Ordinary refresh reset the manual reading position');
  assert(refreshed.panel.top < refreshed.tabs.bottom - 100, 'Refresh unexpectedly jumped to the panel start');
  await screenshot('03-refresh-preserves-reading.png');
  checks.push('real-wheel-scroll-and-unchanged-refresh-preserve-reading-position');

  await openRun(ids.unknown);
  assert.equal(await scroller().evaluate(element => element.scrollTop), 0, 'A different experiment inherited the previous reading position');
  await openRun(ids.complete);
  assert.equal(await scroller().evaluate(element => element.scrollTop), 0, 'Returning to an experiment must show its introduction');
  checks.push('different-experiment-navigation-starts-at-the-introduction');

  await appendFile(path.join(directory(ids.complete), 'events.jsonl'), Array.from({ length: 5001 }, (_, step) => jsonl(event('tool_observation', { step, message: `Synthetic tail after completed record ${step}` }))).join(''));
  await refresh();
  const missing = await api(`/api/runs/${ids.complete}`);
  assertLedger(missing, saved[ids.complete]);
  assert.deepEqual(missing.event_tail, { valid_event_limit: 5000, valid_event_count: 5000, terminal_event: null, reason_code: 'no_terminal_in_tail' });
  assert.equal(missing.terminal_event, null);
  assert.equal(missing.status_reason, null);
  await assertEvidenceInBothLanguages('missing');

  await appendFile(path.join(directory(ids.complete), 'events.jsonl'), jsonl(event('run_failed', { error: 'Synthetic contradictory observation; not a ledger failure.' })));
  await refresh();
  const conflict = await api(`/api/runs/${ids.complete}`);
  assertLedger(conflict, saved[ids.complete]);
  assert.equal(conflict.event_tail.terminal_event, 'run_failed');
  assert.equal(conflict.status_reason, null);
  await assertEvidenceInBothLanguages('conflict');
  assert(!(await page.locator('.run-intro').innerText()).includes('Synthetic contradictory observation'));

  await openRun(ids.unknown);
  assertLedger(await api(`/api/runs/${ids.unknown}`), saved[ids.unknown]);
  await assertEvidenceInBothLanguages('unknown');
  assert.equal(await page.locator('.run-evidence-note').count(), 0, 'Unknown initial lifecycle reason should not be duplicated as a tail notice');
  await openRun(ids.failed);
  for (const lang of ['zh', 'en']) {
    await language(lang);
    const notice = page.locator('.run-intro .notice').filter({ hasText: unknownError });
    assert.equal(await notice.innerText(), unknownError);
    assert.equal(await notice.getAttribute('role'), 'alert');
    assertLedger(await api(`/api/runs/${ids.failed}`), saved[ids.failed]);
    await screenshot(`evidence-unknown-error-${lang}.png`, true);
  }
  checks.push('unrecognized-real-ledger-error-remains-verbatim-in-both-languages');
  assert.equal((await api('/api/desktop/status')).engine.ready, false);
  assert.deepEqual((await nativeState()).forbidden, []);
  assert.deepEqual(rendererErrors, []);
  checks.push('all-requests-stay-local-readonly-with-no-renderer-errors-or-runtime-installation');
} catch (error) {
  failure = error;
  if (page && !page.isClosed()) {
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText().catch(() => '')).slice(-6000));
  }
} finally {
  try { await close(); } catch (error) { failure ||= error; }
}
if (forbiddenRequests.length) failure ||= new Error('Forbidden requests were attempted');
if (!failure) checks.push('native-app-and-sidecar-exit-with-no-qa-profile-processes-left');
await mkdir(output, { recursive: true });
await writeFile(path.join(output, failure ? 'failure.json' : 'acceptance.json'), JSON.stringify({ passed: !failure, mode: 'Real Electron and sidecar. Synthetic isolated histories and intentional local fixture appends. No response or lifecycle mocking, provider calls, installation or training.', executablePath: executablePath || 'development Electron', appInfo, profile, output, runIds: ids, checks, geometry, requests, forbiddenRequests, rendererErrors, processChecks, ...(failure ? { error: String(failure), stack: failure.stack } : {}) }, null, 2) + '\n');
if (failure) throw failure;
console.log(JSON.stringify({ passed: true, checks, profile, output, processChecks }, null, 2));
