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
  signalCode: NodeJS.Signals | null = null;
  private exited = false;
  announce(port = 43210) { this.stdout.write(`${JSON.stringify({ type: 'ready', port })}\n`); }
  exit(code: number | null = 1, signal: NodeJS.Signals | null = null) {
    if (this.exited) return;
    this.exited = true; this.exitCode = code; this.signalCode = signal;
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
  assert.deepEqual(f.failures, [], 'initial failure remains owned by the calling controller');
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

test('stop observes synchronous exit because listeners precede the termination signal', async t => {
  const f = fixture(t, { onSpawn: child => child.announce() });
  await f.service.start();
  const child = f.children[0];
  const signals: NodeJS.Signals[] = [];
  t.mock.timers.enable({ apis: ['setTimeout'] });
  t.mock.method(child, 'kill', (signal: NodeJS.Signals) => { signals.push(signal); child.exit(null, signal); return true; });
  await f.service.stop();
  t.mock.timers.tick(10_000);
  assert.deepEqual(signals, ['SIGTERM']);
  assert.equal(child.signalCode, 'SIGTERM');
  assert.equal(f.children.length, 1);
});

test('stop escalates after four seconds and waits for the forced process exit', async t => {
  const f = fixture(t, { onSpawn: child => child.announce() });
  await f.service.start();
  const child = f.children[0];
  const signals: NodeJS.Signals[] = [];
  t.mock.timers.enable({ apis: ['setTimeout'] });
  t.mock.method(child, 'kill', (signal: NodeJS.Signals) => { signals.push(signal); return true; });
  let finished = false;
  const stopped = f.service.stop().then(() => { finished = true; });
  assert.deepEqual(signals, ['SIGTERM']);
  t.mock.timers.tick(3999);
  assert.deepEqual(signals, ['SIGTERM']);
  t.mock.timers.tick(1);
  await Promise.resolve();
  assert.deepEqual(signals, ['SIGTERM', 'SIGKILL']);
  assert.equal(finished, false, 'sending SIGKILL alone does not release process ownership');
  t.mock.timers.tick(1999);
  assert.equal(finished, false);
  child.exit(null, 'SIGKILL');
  await stopped;
  t.mock.timers.tick(10_000);
  assert.equal(finished, true);
  assert.equal(f.children.length, 1);
});

test('unconfirmed forced exit rejects within six seconds and retains the child for another stop', async t => {
  const f = fixture(t, { onSpawn: child => child.announce() });
  await f.service.start();
  const child = f.children[0];
  const signals: NodeJS.Signals[] = [];
  const priorExitListeners = child.listenerCount('exit');
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const kill = t.mock.method(child, 'kill', (signal: NodeJS.Signals) => { signals.push(signal); return true; });
  const rejection = assert.rejects(f.service.stop(), /无法确认.*已退出/);
  t.mock.timers.tick(4000);
  t.mock.timers.tick(2000);
  await rejection;
  assert.deepEqual(signals, ['SIGTERM', 'SIGKILL']);
  assert.equal(child.listenerCount('exit'), priorExitListeners);
  assert.throws(() => f.service.origin);
  kill.mock.mockImplementation((signal: NodeJS.Signals) => { signals.push(signal); child.exit(null, signal); return true; });
  await f.service.stop();
  assert.deepEqual(signals, ['SIGTERM', 'SIGKILL', 'SIGTERM']);
  assert.equal(f.children.length, 1);
});

test('kill throws, false returns, and error events remain bounded failures without unhandled errors', async t => {
  const f = fixture(t, { onSpawn: child => child.announce() });
  await f.service.start();
  const child = f.children[0];
  const signals: NodeJS.Signals[] = [];
  t.mock.timers.enable({ apis: ['setTimeout'] });
  t.mock.method(child, 'kill', (signal: NodeJS.Signals) => {
    signals.push(signal);
    if (signal === 'SIGTERM') throw new Error('synthetic signal failure');
    child.emit('error', new Error('synthetic kill error'));
    return false;
  });
  const rejection = assert.rejects(f.service.stop(), /未能终止/);
  t.mock.timers.tick(4000);
  t.mock.timers.tick(2000);
  await rejection;
  assert.deepEqual(signals, ['SIGTERM', 'SIGKILL']);
  child.exit(null, 'SIGKILL');
  await f.service.stop();
  t.mock.timers.tick(10_000);
  assert.equal(f.children.length, 1);
});

test('an already signaled exit is confirmed even when exitCode is null', async t => {
  const f = fixture(t, { onSpawn: child => child.announce() });
  await f.service.start();
  const child = f.children[0];
  child.signalCode = 'SIGKILL';
  const kill = t.mock.method(child, 'kill', () => { throw new Error('must not signal an exited process'); });
  await f.service.stop();
  assert.equal(kill.mock.callCount(), 0);
  assert.throws(() => f.service.origin);
  child.exit(null, 'SIGKILL');
});
