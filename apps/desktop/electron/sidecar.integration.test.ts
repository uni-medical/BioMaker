import test from 'node:test';
import assert from 'node:assert/strict';
import { cpSync, mkdtempSync, renameSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import type { ChildProcess } from 'node:child_process';
import { Sidecar } from './sidecar';

test('portable sidecar authenticates, restarts after a crash, and exits without an orphan', { skip: process.env.BIOMAKER_TEST_RUNTIME !== '1', timeout: 30_000 }, async () => {
  const desktop = path.resolve(__dirname, '..');
  const directory = mkdtempSync(path.join(os.tmpdir(), 'BioMaker native test '));
  const failures: string[] = [];
  const bundle = path.join(directory, 'Application bundle');
  cpSync(path.join(desktop, 'build/runtime'), path.join(bundle, 'runtime'), { recursive: true, verbatimSymlinks: true });
  cpSync(path.join(desktop, 'service'), path.join(bundle, 'service'), { recursive: true });
  const service = new Sidecar({ runtime: path.join(bundle, 'runtime'), service: path.join(bundle, 'service/main.py'), data: path.join(directory, 'profile'), version: 'test', onFailure: message => failures.push(message) });
  const getChild = () => (service as unknown as { child: ChildProcess | null }).child;
  let latest: ChildProcess | null = null;
  try {
    await service.start();
    const first = getChild();
    assert.ok(first?.pid);
    assert.equal((await fetch(`${service.origin}/api/health`)).status, 401);
    const health = await fetch(`${service.origin}/api/health`, { headers: { Authorization: `Bearer ${service.token}` } });
    assert.equal(health.status, 200);
    const python = path.join(service.runtimeDirectory, 'python', process.platform === 'win32' ? 'python.exe' : 'bin/python3');
    const uv = path.join(service.runtimeDirectory, 'bin', process.platform === 'win32' ? 'uv.exe' : 'uv');
    const venv = path.join(directory, 'test engine venv');
    execFileSync(uv, ['venv', '--python', python, venv], { stdio: 'pipe' });
    const venvPython = path.join(venv, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    const base = execFileSync(venvPython, ['-c', 'import sys;print(sys._base_executable)'], { encoding: 'utf8' }).trim();
    assert.ok(base.startsWith(service.runtimeDirectory), 'engine venv uses the persistent application-support runtime');
    renameSync(bundle, `${bundle} moved`);
    assert.equal(execFileSync(venvPython, ['-c', 'print("still available")'], { encoding: 'utf8' }).trim(), 'still available');
    first.kill('SIGKILL');
    for (let i = 0; i < 100; i++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      latest = getChild();
      if (latest?.pid && latest.pid !== first.pid) {
        try { if ((await fetch(`${service.origin}/api/health`, { headers: { Authorization: `Bearer ${service.token}` } })).ok) break; } catch { /* Startup pending. */ }
      }
    }
    assert.ok(latest?.pid && latest.pid !== first.pid, 'a replacement service started after the original application bundle moved');
    const origin = service.origin;
    assert.equal((await fetch(`${origin}/api/health`, { headers: { Authorization: `Bearer ${service.token}` } })).status, 200);
    await service.stop();
    assert.equal(getChild(), null);
    assert.throws(() => process.kill(latest!.pid!, 0), { code: 'ESRCH' });
    await assert.rejects(fetch(`${origin}/api/health`, { signal: AbortSignal.timeout(1000) }));
    assert.deepEqual(failures, []);
  } finally {
    await service.stop();
    rmSync(directory, { recursive: true, force: true });
  }
});
