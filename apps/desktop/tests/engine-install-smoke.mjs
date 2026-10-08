// Real dependency installation, imports and synthetic CPU LightGBM acceptance.
// Does not call a model provider or run a biomedical experiment.
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { cp, mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = path.join(root, 'build', 'runtime');
const home = await mkdtemp(path.join(os.tmpdir(), 'biomaker-engine-acceptance-'));
const engine = path.join(home, 'engine');
await cp(path.join(runtime, 'engine'), engine, { recursive: true });
const token = randomBytes(32).toString('hex');
const python = path.join(runtime, 'python', 'bin', 'python3');
const child = spawn(python, ['-u', path.join(root, 'service', 'main.py')], {
  cwd: engine,
  env: { ...process.env, PYTHONPATH: path.join(runtime, 'site-packages'), PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', BIOMAKER_DESKTOP_TOKEN: token, BIOMAKER_DESKTOP_HOME: home, BIOMAKER_DESKTOP_ENGINE_ROOT: engine, BIOMAKER_WEBUI_UV: path.join(runtime, 'bin', 'uv'), BIOMAKER_DESKTOP_PARENT_PID: String(process.pid) },
  stdio: ['ignore', 'pipe', 'pipe'],
});
let errorLog = '';
child.stderr.on('data', data => { errorLog = (errorLog + data.toString()).slice(-8000); });
try {
  const port = await new Promise((resolve, reject) => {
    let output = '';
    const timeout = setTimeout(() => reject(new Error('Sidecar startup timeout')), 30000);
    child.on('exit', code => { clearTimeout(timeout); reject(new Error(`Sidecar exited ${code}: ${errorLog}`)); });
    child.stdout.on('data', data => {
      output += data.toString();
      for (const line of output.split('\n')) { try { const ready = JSON.parse(line); if (ready.type === 'ready') { clearTimeout(timeout); resolve(ready.port); } } catch {} }
    });
  });
  const call = async (endpoint, method = 'GET') => {
    const response = await fetch(`http://127.0.0.1:${port}${endpoint}`, { method, headers: { Authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(30000) });
    const result = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(result));
    return result;
  };
  console.log(JSON.stringify({ phase: 'installing', home }));
  await call('/api/desktop/engine/install', 'POST');
  const deadline = Date.now() + 15 * 60 * 1000;
  let lastLog = '';
  while (Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 3000));
    const status = await call('/api/desktop/status');
    if (status.engine.log !== lastLog) { lastLog = status.engine.log; console.log(lastLog.split('\n').slice(-3).join('\n')); }
    if (!status.engine.installing) {
      const report = { passed: status.engine.ready, home, engine, status, verified: 'Real bundled uv installation, isolated engine imports, and synthetic CPU LightGBM calculation; no model provider calls or biomedical experiment.' };
      const output = path.join(root, 'output', 'engine-acceptance.json');
      await mkdir(path.dirname(output), { recursive: true }); await writeFile(output, JSON.stringify(report, null, 2));
      if (!report.passed) throw new Error(status.engine.error || 'Runtime did not become ready');
      console.log(JSON.stringify({ passed: true, output, home })); break;
    }
  }
  if (Date.now() >= deadline) { await call('/api/desktop/engine/cancel', 'POST'); throw new Error('Engine setup exceeded acceptance deadline'); }
} finally {
  child.kill('SIGTERM');
  const timeout = setTimeout(() => child.kill('SIGKILL'), 5000); timeout.unref();
  await new Promise(resolve => { if (child.exitCode !== null) resolve(); else child.once('exit', resolve); });
  clearTimeout(timeout);
}
