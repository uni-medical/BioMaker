import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const profile = await mkdtemp(path.join(os.tmpdir(), 'biomaker-desktop-qa-'));
const output = path.join(root, 'output', 'playwright');
await mkdir(output, { recursive: true });
const env = { ...process.env, BIOMAKER_DESKTOP_HOME: profile, BIOMAKER_DESKTOP_USER_DATA: profile };
delete env.ELECTRON_RUN_AS_NODE;
// CI/release review can explicitly validate the packaged executable.
const executablePath = process.env.BIOMAKER_QA_EXECUTABLE;
if (executablePath && process.platform === 'darwin') env.PATH = '/usr/bin:/bin:/usr/sbin:/sbin';
const launch = () => electron.launch({ ...(executablePath ? { executablePath, args: [] } : { args: [root] }), env, timeout: 60_000 });
let app;
const errors = [];
console.log(JSON.stringify({ profile, output, executablePath: executablePath || 'development Electron' }));
async function ready() {
  const page = await app.firstWindow();
  page.on('pageerror', error => errors.push(error.message));
  let healthy = false;
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    try { healthy = (await page.evaluate(() => window.biomaker.request('/api/health'))).status === 'ok'; } catch {}
    if (healthy) break;
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  assert(healthy, 'The real desktop service must become healthy within 60 seconds');
  await page.waitForFunction(() => document.querySelector('textarea'));
  return page;
}
try {
  app = await launch();
  let page = await ready();
  const status = await page.evaluate(() => window.biomaker.request('/api/desktop/status'));
  assert.equal(status.engine.ready, false);
  assert.equal(await realpath(status.paths.workspace), await realpath(path.join(profile, 'workspace')));
  assert.deepEqual(await page.evaluate(() => window.biomaker.request('/api/runs')), []);
  assert.equal(await page.evaluate(() => typeof window.require), 'undefined');
  assert.equal(await page.evaluate(() => typeof window.process), 'undefined');
  await assert.rejects(page.evaluate(() => window.biomaker.request('/api/../env')), /Unsafe|Invalid|allowed/);
  await assert.rejects(page.evaluate(() => window.biomaker.openPath('/Applications')), /授权|allowed/);
  await assert.rejects(page.evaluate(() => window.biomaker.request('/api/runs', { method: 'POST', body: { experiment: { dataset_path: '/tmp', task_body: 'QA must not start without setup' } } })), /environment|runtime|install|环境|engine/i);
  await page.screenshot({ path: path.join(output, '01-first-launch.png') });

  await page.locator('.sidebar-settings').click();
  await page.getByLabel('API 地址', { exact: false }).fill('http://127.0.0.1:9/v1');
  await page.getByLabel('API Key', { exact: false }).first().fill('desktop-qa-fake-key-never-sent');
  await page.getByLabel('默认模型', { exact: true }).fill('desktop-qa-model');
  await page.getByRole('button', { name: '保存配置', exact: true }).click();
  await page.getByText('所有更改已保存', { exact: true }).waitFor();
  const saved = await page.evaluate(() => window.biomaker.request('/api/env'));
  assert.equal(saved.values.OPENAI_API_KEY.configured, true);
  assert(!JSON.stringify(saved).includes('desktop-qa-fake-key-never-sent'));
  assert(!(await page.evaluate(() => JSON.stringify(localStorage))).includes('desktop-qa-fake-key-never-sent'));
  await page.screenshot({ path: path.join(output, '02-provider-settings.png') });
  await page.getByRole('button', { name: '通用', exact: true }).click();
  await page.getByRole('button', { name: '深色', exact: true }).click();
  assert.equal((await page.evaluate(() => window.biomaker.getPreferences())).theme, 'dark');
  await page.getByLabel('界面语言', { exact: true }).selectOption('en');
  await page.getByText('Make BioMaker your own', { exact: true }).waitFor();
  await page.screenshot({ path: path.join(output, '03-dark-english.png') });
  await page.getByLabel('Interface language', { exact: true }).selectOption('zh');
  await page.getByRole('button', { name: '浅色', exact: true }).click();
  await page.keyboard.press('Escape');
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  await app.close(); app = undefined;

  // These clearly named synthetic records live only in an isolated QA profile.
  // They exercise the real history importer and artifact API, never production data.
  const logdir = path.join(profile, 'workspace', 'logs', 'desktop-acceptance-fixture');
  const dataset = path.join(profile, 'workspace', 'datasets', 'desktop-qa-fixture');
  await mkdir(logdir, { recursive: true }); await mkdir(dataset, { recursive: true });
  await writeFile(path.join(logdir, 'run_manifest.json'), JSON.stringify({ data_dir: dataset, task_description: '桌面验收样例：用于验证历史记录、日志和文件展示。这里不是实际模型实验。', created_at: '2026-10-08T10:00:00Z', config: { agent_steps: 3, model_for_all_agents: 'desktop-qa-model', per_node_timeout_seconds: 600 } }));
  const events = [{ ts: '2026-10-08T10:00:00Z', event: 'run_started' }, { ts: '2026-10-08T10:01:00Z', event: 'round_finished', completed_steps: 3 }, { ts: '2026-10-08T10:02:00Z', event: 'run_finished', completed_steps: 3 }];
  await writeFile(path.join(logdir, 'events.jsonl'), events.map(e => JSON.stringify(e)).join('\n') + '\n');
  await writeFile(path.join(logdir, 'acceptance-report.md'), '# Desktop acceptance fixture\n\nSynthetic UI fixture, not a medical model result.\n');
  await writeFile(path.join(logdir, 'journal.json'), JSON.stringify({ __version: '2', node2parent: { 'candidate-b': 'candidate-a' }, nodes: [
    { id: 'candidate-a', step: 0, agent_name: 'Draft', is_buggy: true, metric: { value: '_nonfinite_', maximize: true }, plan: 'Synthetic baseline for desktop acceptance only.', code: '# Synthetic fixture; not executable training code.', exec_time: 12 },
    { id: 'candidate-b', step: 1, agent_name: 'Improve', is_buggy: false, metric: { value: 0.7, maximize: true }, plan: 'Synthetic refinement for lineage display.', analysis: 'This is a UI fixture, not a model result.', code: '# Synthetic refinement fixture.', exec_time: 10 },
  ] }).replace('"_nonfinite_"', 'NaN'));
  await writeFile(path.join(logdir, 'results.tsv'), 'step\tmetric\tmaximize\tstatus\n0\tnan\ttrue\tfailed\n1\t0.7\ttrue\tsucceeded\n');
  await writeFile(path.join(logdir, 'desktop-acceptance-fixture.log'), 'Desktop acceptance fixture log. No model was trained.\n');
  app = await launch(); page = await ready();
  const records = await page.evaluate(() => window.biomaker.request('/api/runs'));
  assert.equal(records.length, 1); assert.equal(records[0].status, 'complete');
  const detail = await page.evaluate(id => window.biomaker.request(`/api/runs/${id}`), records[0].id);
  assert(detail.artifacts.some(file => file.name === 'acceptance-report.md'));
  assert.equal(detail.observed.results.best_metric, 0.7);
  await page.getByText('desktop-acceptance-fixture', { exact: true }).first().click();
  await page.locator('.run-intro .status-label').filter({ hasText: '已完成' }).waitFor();
  await page.screenshot({ path: path.join(output, '04-history.png') });
  await page.getByRole('tab', { name: '候选方案', exact: true }).click();
  await page.getByText('Synthetic refinement for lineage display.', { exact: true }).waitFor();
  assert((await page.locator('.candidate-row').first().innerText()).includes('—'));
  await page.getByRole('tab', { name: '代码', exact: true }).click();
  await page.getByText('# Synthetic refinement fixture.', { exact: true }).waitFor();
  await page.screenshot({ path: path.join(output, '05-candidates.png') });
  await page.getByRole('tab', { name: /^文件/ }).click();
  const destination = path.join(profile, 'saved-acceptance-report.md');
  await app.evaluate(({ dialog }, filePath) => { dialog.showSaveDialog = async () => ({ canceled: false, filePath }); }, destination);
  await page.getByRole('button', { name: '保存 acceptance-report.md', exact: true }).click();
  await page.waitForFunction(() => document.body.innerText.includes('已保存') || document.body.innerText.includes('已导出'));
  assert.equal(await readFile(destination, 'utf8'), '# Desktop acceptance fixture\n\nSynthetic UI fixture, not a medical model result.\n');
  await page.getByRole('tab', { name: '运行日志', exact: true }).click();
  await page.getByText('Desktop acceptance fixture log. No model was trained.', { exact: true }).waitFor();
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(960, 680));
  await page.waitForTimeout(200);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.screenshot({ path: path.join(output, '06-compact.png') });
  assert.deepEqual(errors, []);
  const report = { passed: true, executablePath: executablePath || 'development Electron', platform: process.platform, arch: process.arch, profile, output, checks: ['real-sidecar', 'isolated-profile', 'renderer-sandbox', 'IPC-path-boundaries', 'missing-engine-preflight', 'settings-persistence', 'secret-masking', 'theme', 'language', 'restart', 'history-import', 'candidates-and-nonfinite-metrics', 'native-artifact-export', 'console', 'compact-layout', 'no-renderer-errors'] };
  await writeFile(path.join(output, 'electron-acceptance.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  if (app) {
    const page = await app.firstWindow();
    await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
    console.error((await page.locator('body').innerText()).slice(-4500));
    console.error('Renderer errors:', errors);
  }
  throw error;
} finally { if (app) await app.close(); }
