import test, { type TestContext } from 'node:test';
import assert from 'node:assert/strict';
import childProcess, { type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { PassThrough } from 'node:stream';
import { Sidecar, type ServiceState } from './sidecar';

class ControlledChild extends EventEmitter {
  stdout = new PassThrough();
  stderr = new PassThrough();
  exitCode: number | null = null;
  private exited = false;
  announce(port = 43210) { this.stdout.write(`${JSON.stringify({ type: 'ready', port })}\n`); }
  exit(code: number | null = 1, signal: NodeJS.Signals | null = null) {
    if (this.exited) return;
    this.exited = true; this.exitCode = code;
    this.emit('exit', code, signal);
  }
  kill(signal: NodeJS.Signals = 'SIGTERM') { queueMicrotask(() => this.exit(null, signal)); return true; }
}
async function waitFor(predicate: () => boolean) {
  const deadline = Date.now() + 5000;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error('Timed out waiting for sidecar transition');
    await new Promise(resolve => setTimeout(resolve, 5));
  }
}
function fixture(t: TestContext, options: { onSpawn?: (child: ControlledChild) => void; fetch?: typeof fetch } = {}) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'biomaker-service-state-'));
  const runtime = path.join(root, 'runtime');
  const serviceFile = path.join(root, 'service/main.py');
  for (const item of ['runtime/python/bin', 'runtime/bin', 'runtime/site-packages', 'runtime/licenses', 'runtime/engine', 'service']) mkdirSync(path.join(root, item), { recursive: true });
  for (const item of [process.platform === 'win32' ? 'python/python.exe' : 'python/bin/python3', process.platform === 'win32' ? 'bin/uv.exe' : 'bin/uv', 'THIRD_PARTY_NOTICES.md', 'service-requirements.txt']) writeFileSync(path.join(runtime, item), 'test fixture');
  writeFileSync(serviceFile, '# controlled subprocess fixture');
  writeFileSync(path.join(runtime, 'engine/pyproject.toml'), 'engine fixture');
  writeFileSync(path.join(runtime, 'manifest.json'), JSON.stringify({ target: `${process.platform}-${process.arch}`, pythonVersion: '3.12.12', pythonBuild: '20260211', applicationVersion: 'test', engineSha256: createHash('sha256').update('pyproject.toml').update('engine fixture').digest('hex') }));
  const children: ControlledChild[] = [];
  t.mock.method(childProcess, 'spawn', () => {
    const child = new ControlledChild(); children.push(child);
    queueMicrotask(() => options.onSpawn?.(child));
    return child as unknown as ChildProcess;
  });
  const fetchMock = t.mock.method(globalThis, 'fetch', options.fetch ?? (async () => new Response('{}', { status: 200 })));
  const states: ServiceState[] = [];
  const failures: string[] = [];
  const service = new Sidecar({ runtime, service: serviceFile, data: path.join(root, 'profile'), version: 'test', onState: state => states.push(state), onFailure: message => failures.push(message) });
  t.after(async () => { await service.stop(); rmSync(root, { recursive: true, force: true }); });
  return { service, runtime, children, states, failures, fetchMock };
}

test('the service stays starting until authenticated health succeeds', async t => {
  let release!: (response: Response) => void;
  const f = fixture(t, { fetch: async () => new Promise<Response>(resolve => { release = resolve; }) });
  assert.deepEqual(f.service.getState(), { status: 'starting', revision: 0 });
  const started = f.service.start();
  f.children[0].announce();
  assert.equal(f.service.getState().status, 'starting');
  assert.throws(() => f.service.origin, /正在启动/);
  assert.equal(f.fetchMock.mock.calls[0].arguments[0], 'http://127.0.0.1:43210/api/health');
  assert.deepEqual(f.fetchMock.mock.calls[0].arguments[1]?.headers, { Authorization: `Bearer ${f.service.token}` });
  release(new Response('{}', { status: 200 }));
  await started;
  assert.deepEqual(f.states, [{ status: 'ready', revision: 1 }]);
  const snapshot = f.service.getState(); snapshot.status = 'failed';
  assert.equal(f.service.getState().status, 'ready');
  assert.equal(f.service.origin, 'http://127.0.0.1:43210');
});

test('startup retries remain starting and only exhausted retries publish failure', async t => {
  const f = fixture(t, { onSpawn: child => child.exit(1) });
  await assert.rejects(f.service.start(), /服务退出/);
  assert.equal(f.children.length, 3);
  assert.equal(f.states.length, 1);
  assert.equal(f.states[0].status, 'failed');
  assert.match(f.states[0].message!, /服务退出/);
  assert.deepEqual(f.failures, [], 'initial failure remains owned by the startup dialog');
  assert.throws(() => f.service.origin);
});

test('runtime preparation errors publish a terminal failure without spawning', async t => {
  const f = fixture(t);
  rmSync(path.join(f.runtime, 'licenses'), { recursive: true });
  await assert.rejects(f.service.start(), /ENOENT/);
  assert.equal(f.service.getState().status, 'failed');
  assert.equal(f.children.length, 0);
});

test('crashes immediately report reconnecting, recover, and respect the restart limit', async t => {
  const f = fixture(t, { onSpawn: child => child.announce() });
  await f.service.start();
  for (let i = 0; i < 3; i++) {
    f.children[i].exit(1);
    assert.equal(f.service.getState().status, 'reconnecting');
    assert.throws(() => f.service.origin);
    await waitFor(() => f.children.length === i + 2 && f.service.getState().status === 'ready');
  }
  f.children[3].exit(1);
  assert.equal(f.service.getState().status, 'failed');
  assert.equal(f.failures.length, 1);
  assert.equal(f.failures[0], f.service.getState().message);
  assert.equal(f.children.length, 4);
  assert.deepEqual(f.states.map(state => state.revision), f.states.map((_, index) => index + 1));
});

test('transport recovery deduplicates health probes and retains a healthy child', async t => {
  let release!: (response: Response) => void;
  let probing = false;
  const f = fixture(t, { onSpawn: child => child.announce(), fetch: async () => probing ? new Promise<Response>(resolve => { release = resolve; }) : new Response('{}', { status: 200 }) });
  await f.service.start();
  probing = true;
  const first = f.service.checkConnection();
  const second = f.service.checkConnection();
  assert.equal(first, second);
  assert.equal(f.service.getState().status, 'reconnecting');
  assert.equal(f.fetchMock.mock.calls.length, 2);
  release(new Response('{}', { status: 200 }));
  await first;
  assert.equal(f.service.getState().status, 'ready');
  assert.equal(f.children.length, 1);
  assert.deepEqual(f.failures, []);
});

test('failed health after a transport error replaces the child through bounded recovery', async t => {
  let healthy = true;
  const f = fixture(t, { onSpawn: child => child.announce(), fetch: async () => new Response('{}', { status: healthy ? 200 : 503 }) });
  await f.service.start();
  healthy = false;
  await f.service.checkConnection();
  assert.equal(f.service.getState().status, 'reconnecting');
  healthy = true;
  await waitFor(() => f.children.length === 2 && f.service.getState().status === 'ready');
  assert.deepEqual(f.failures, []);
});

test('stopping during a pending health check never publishes ready or retries', async t => {
  let release!: (response: Response) => void;
  const f = fixture(t, { fetch: async () => new Promise<Response>(resolve => { release = resolve; }) });
  const started = f.service.start();
  f.children[0].announce();
  await f.service.stop();
  release(new Response('{}', { status: 200 }));
  await started;
  assert.deepEqual(f.states, []);
  assert.equal(f.children.length, 1);
  assert.throws(() => f.service.origin);
});
