/** UI acceptance companion to desktop-workflow-smoke.py. Uses real engine data. */
import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, readFile, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';

const config = JSON.parse(await readFile(process.argv[2], 'utf8'));
const output = path.resolve('output/playwright/desktop-workflow');
await mkdir(output, { recursive: true });
const errors = [];
const proof = { passed: false, screenshots: [], renderer_errors: errors, queue_transitions: [] };
let app;
let page;
let runId;
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
async function snapshot(name) {
  const file = path.join(output, `${name}.png`);
  await page.screenshot({ path: file });
  proof.screenshots.push(file);
  await writeFile(path.join(config.home, `${name}.txt`), await page.locator('body').innerText());
}
async function getRuns() { return page.evaluate(() => window.biomaker.request('/api/runs')); }
try {
  app = await electron.launch({ executablePath: config.executable, args: [], env: process.env, timeout: 60_000 });
  app.process().stderr?.on('data', data => process.stderr.write(data));
  app.process().stdout?.on('data', data => process.stdout.write(data));
  app.process().on('exit', (code, signal) => console.log(JSON.stringify({ electronExit: code, signal })));
  page = await app.firstWindow({ timeout: 20000 });
  page.setDefaultTimeout(15000);
  page.on('pageerror', error => errors.push(error.message));
  let state;
  const startupDeadline = Date.now() + 60_000;
  while (Date.now() < startupDeadline) {
    try { state = await page.evaluate(() => window.biomaker.request('/api/desktop/status')); } catch {}
    if (state) break;
    await wait(100);
  }
  assert(state, 'Desktop sidecar must become healthy');
  assert.equal(state.version, config.expectedVersion);
  assert.equal(state.engine.ready, true, 'Reuse the installed engine without installing packages');
  assert.equal(await realpath(state.paths.engine), config.engine);
  assert.equal(await realpath(state.paths.workspace), await realpath(path.join(config.profile, 'workspace')));
  assert.deepEqual(await getRuns(), []);
  assert.equal(await page.evaluate(() => typeof window.require), 'undefined');
  proof.startup = state;
  // Install test instrumentation after normal renderer startup. Only this empty
  // QA profile's sidecar is restarted; its real recovery path launches the
  // service under OS network restrictions before any model configuration exists.
  proof.sandbox_restart = await app.evaluate((_electron, qa) => {
    const cp = process.getBuiltinModule('node:child_process');
    const fs = process.getBuiltinModule('node:fs');
    const spawn = cp.spawn;
    cp.spawn = function(command, args, options) {
      if (typeof command === 'string' && /\/runtimes\/[^/]+\/python\/bin\/python3$/.test(command) && args?.some(x => /\/service\/main\.py$/.test(x)) && options?.env?.BIOMAKER_DESKTOP_HOME === qa.profile) {
        fs.appendFileSync(qa.home + '/sandbox-spawns.jsonl', JSON.stringify({ command, args, policy: qa.home + '/network.sb' }) + '\n');
        return spawn.call(this, '/usr/bin/sandbox-exec', ['-f', qa.home + '/network.sb', command, ...args], options);
      }
      return spawn.call(this, command, args, options);
    };
    const own = cp.execFileSync('/bin/ps', ['-axo', 'pid=,ppid=,command='], { encoding: 'utf8' }).split('\n').map(line => line.trim().match(/^(\d+)\s+(\d+)\s+(.+)$/)).filter(match => match && Number(match[2]) === process.pid && match[3].includes(qa.profile + '/runtimes/') && match[3].includes('/service/main.py'));
    if (own.length !== 1) throw new Error('Expected exactly one QA-owned sidecar, got ' + own.length);
    const previousPid = Number(own[0][1]);
    process.kill(previousPid, 'SIGTERM');
    return { main_pid: process.pid, previous_sidecar_pid: previousPid, reason: 'Apply QA localhost-only sandbox through real sidecar recovery' };
  }, { home: config.home, profile: config.profile });
  let sandboxReady = false;
  const recoveryDeadline = Date.now() + 20000;
  while (Date.now() < recoveryDeadline) {
    try {
      const launches = (await readFile(path.join(config.home, 'sandbox-spawns.jsonl'), 'utf8')).trim();
      const healthy = await page.evaluate(() => window.biomaker.request('/api/health'));
      if (launches && healthy.status === 'ok') { sandboxReady = true; break; }
    } catch {}
    await wait(100);
  }
  assert(sandboxReady, 'Real sidecar recovery must launch inside the network sandbox');
  await page.getByLabel('实验任务', { exact: true }).waitFor();
  await snapshot('01-isolated-ready');

  await page.locator('.composer-meta button').click();
  await page.getByRole('dialog').waitFor();
  await page.getByLabel('API 地址', { exact: false }).fill(config.baseURL);
  await page.getByLabel('API Key', { exact: false }).first().fill(config.fakeKey);
  await page.getByLabel('默认模型', { exact: true }).fill(config.model);
  await page.getByRole('button', { name: '保存配置', exact: true }).click();
  await page.getByText('所有更改已保存', { exact: true }).waitFor();
  const saved = await page.evaluate(() => window.biomaker.request('/api/env'));
  assert.equal(saved.values.OPENAI_API_KEY.configured, true);
  assert.equal(saved.values.OPENAI_BASE_URL.value, config.baseURL);
  assert.equal(saved.values.BIOMAKER_MAIN_MODEL.value, config.model);
  assert(!JSON.stringify(saved).includes(config.fakeKey));
  await page.keyboard.press('Escape');
  await page.getByRole('dialog').waitFor({ state: 'hidden' });

  // Only the native directory dialog is supplied by QA; task/options/submit use UI.
  await app.evaluate(({ dialog }, dataset) => { dialog.showOpenDialog = async () => ({ canceled: false, filePaths: [dataset] }); }, config.dataset);
  await page.getByLabel('实验任务', { exact: true }).fill(config.task);
  await page.locator('.dataset-button').click();
  await page.locator('.dataset-overview[data-code="ok"]').waitFor();
  await page.getByRole('button', { name: '实验设置', exact: true }).click();
  await page.getByLabel('实验名称', { exact: false }).fill('synthetic-desktop-ui-workflow');
  await page.getByLabel('最大探索步数', { exact: true }).fill('1');
  await page.getByLabel('探索时长上限（小时）', { exact: false }).fill('0.02');
  await page.getByLabel('单次执行时限（秒）', { exact: true }).fill('30');
  await page.getByLabel('并发工作进程', { exact: false }).selectOption('1');
  await page.getByRole('checkbox', { name: /^最终重训/ }).uncheck();
  await snapshot('02-configured-synthetic-task');
  await page.getByRole('button', { name: '开始实验', exact: true }).click();

  const deadline = Date.now() + 90_000;
  let terminal;
  while (Date.now() < deadline) {
    const records = await getRuns();
    if (records.length) {
      assert.equal(records.length, 1);
      const record = records[0]; runId = record.id;
      const state = record.detailed_status || record.status;
      if (proof.queue_transitions.at(-1)?.status !== state) proof.queue_transitions.push({ status: state, at: new Date().toISOString() });
      if (!proof.submitted_record) {
        proof.submitted_record = record;
        assert.equal(record.source, 'run_ledger');
        assert.equal(record.read_only, false);
        assert.equal(record.options.agent_steps, 1);
        assert.equal(record.options.final_retrain_enabled, false);
        await snapshot('03-submitted-real-queue');
      }
      if (['succeeded', 'failed', 'interrupted', 'stopped', 'cancelled'].includes(state)) { terminal = record; break; }
    }
    await wait(200);
  }
  assert(terminal, 'Real queue run must reach a terminal state');
  assert.equal(terminal.detailed_status, 'succeeded', JSON.stringify(terminal));
  proof.terminal_record = terminal;
  const detail = await page.evaluate(id => window.biomaker.request(`/api/runs/${id}`), runId);
  assert.equal(detail.observed.results.best_metric, 1);
  assert(detail.artifacts.some(file => file.name === 'report.md'));
  assert(detail.artifacts.some(file => file.name === 'best_solution.py'));
  proof.detail = detail;
  await page.locator('.run-intro .status-label').filter({ hasText: '已完成' }).waitFor({ timeout: 30000 });
  await page.getByRole('tab', { name: '结果', exact: true }).click();
  await page.locator('.result-selection').getByRole('heading', { name: '运行记录中的选择', exact: true }).waitFor();
  const overview = await page.evaluate(id => window.biomaker.request(`/api/desktop/runs/${id}/result`), runId);
  assert.equal(overview.snapshot.status, 'stable');
  assert.equal(overview.selection.status, 'recorded');
  assert.equal(overview.selection.code_match, true);
  assert.equal(overview.selection.metric.value, 1);
  assert.equal(overview.final_retrain.state, 'disabled');
  proof.result_overview = overview;
  await snapshot('04-real-result-overview');
  await page.getByRole('button', { name: '查看候选方案', exact: true }).click();
  await page.locator('.candidate-row').waitFor();
  assert.equal(await page.locator('.candidate-row').count(), 1);
  await page.getByRole('tab', { name: '代码', exact: true }).click();
  await page.getByText('LogisticRegression', { exact: false }).last().waitFor();
  await snapshot('05-real-candidate-code');
  await page.getByRole('tab', { name: /^文件/ }).click();
  const destination = path.join(config.home, 'ui-export-report.md');
  await app.evaluate(({ dialog }, filePath) => { dialog.showSaveDialog = async () => ({ canceled: false, filePath }); }, destination);
  await page.getByRole('button', { name: '保存 report.md', exact: true }).click();
  await page.getByText('文件已保存', { exact: true }).waitFor();
  const report = await readFile(destination, 'utf8');
  assert(report.includes('Synthetic workflow acceptance') && report.includes('software integration test'));
  await snapshot('06-exported-real-report');
  await page.getByRole('tab', { name: '运行日志', exact: true }).click();
  await page.getByText('Starting BioMaker workflow', { exact: false }).last().waitFor();
  await snapshot('07-real-execution-log');
  assert.deepEqual(errors, []);
  proof.passed = true;
  proof.run_id = runId;
  proof.exported_report = destination;
} catch (error) {
  proof.error = String(error);
  proof.stack = error.stack;
  if (page && !page.isClosed()) { try { await snapshot('failure'); } catch {} }
  throw error;
} finally {
  if (app) {
    // Failure cleanup may only stop the single real run from this isolated profile.
    if (!proof.passed && runId && page && !page.isClosed()) {
      try { await page.evaluate(id => window.biomaker.request(`/api/runs/${id}/stop`, { method: 'POST' }), runId); } catch {}
    }
    const forceClose = setTimeout(() => app.process().kill('SIGTERM'), 10000);
    forceClose.unref();
    try { await app.close(); } finally { clearTimeout(forceClose); }
  }
  await writeFile(path.join(config.home, 'ui-proof.json'), JSON.stringify(proof, null, 2));
  console.log(JSON.stringify({ passed: proof.passed, runId, output }));
}
