import test from 'node:test';
import assert from 'node:assert/strict';
import { ServiceController, type ServiceCallbacks, type ServiceControllerState, type ServiceFailureCode, type ServiceProcess } from './service-controller';
import type { ServiceState } from './sidecar';

function deferred() {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<void>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
class ControlledService implements ServiceProcess {
  state: ServiceState = { status: 'starting', revision: 0 };
  starts = 0;
  stops = 0;
  startAction = async () => { this.emit('ready'); };
  stopAction = async () => {};
  constructor(readonly callbacks: ServiceCallbacks) {}
  start() { this.starts++; return this.startAction(); }
  stop() { this.stops++; return this.stopAction(); }
  getState() { return { ...this.state }; }
  emit(status: ServiceState['status'], revision = this.state.revision + 1) {
    this.state = { status, revision, message: 'SYNTHETIC_SECRET /private/test/path' };
    this.callbacks.onState(this.getState());
  }
}
async function settle() { for (let i = 0; i < 10; i++) await Promise.resolve(); }
function fixture(configure?: (service: ControlledService, index: number) => void) {
  const services: ControlledService[] = [];
  const states: ServiceControllerState[] = [];
  const failures: ServiceFailureCode[] = [];
  const controller = new ServiceController({
    factory: callbacks => {
      const service = new ControlledService(callbacks);
      configure?.(service, services.length);
      services.push(service);
      return service;
    },
    onState: state => states.push(state),
    onFailure: code => failures.push(code),
    now: () => new Date('2026-10-09T00:00:00.000Z'),
  });
  return { controller, services, states, failures };
}

test('constructor failure publishes a recoverable fixed code without a raw exception', async () => {
  let attempts = 0;
  const states: ServiceControllerState[] = [];
  const controller = new ServiceController({
    factory: callbacks => {
      if (++attempts === 1) throw new Error('SYNTHETIC_SECRET /private/test/path');
      return new ControlledService(callbacks);
    }, onState: state => states.push(state),
  });
  assert.deepEqual(await controller.start(), { status: 'failed', revision: 2, code: 'SERVICE_CONSTRUCTION_FAILED' });
  assert.equal(controller.getService(), null);
  assert.equal((await controller.retry()).status, 'ready');
  assert.equal(attempts, 2);
  assert.deepEqual(states.map(state => state.revision), [1, 2, 3, 4]);
  assert.doesNotMatch(JSON.stringify([states, controller.getDiagnostics()]), /SYNTHETIC_SECRET|private|message/);
  await controller.stop();
});

test('failed startup retries only after the previous service has finished stopping', async () => {
  const stopGate = deferred();
  const f = fixture((service, index) => {
    if (index === 0) {
      service.startAction = async () => { service.emit('failed'); throw new Error('start failed'); };
      service.stopAction = () => stopGate.promise;
    }
  });
  assert.equal((await f.controller.start()).code, 'SERVICE_START_FAILED');
  const retry = f.controller.retry();
  await settle();
  assert.equal(f.services.length, 1);
  assert.equal(f.services[0].stops, 1);
  assert.equal(f.controller.getState().status, 'starting');
  stopGate.resolve();
  assert.equal((await retry).status, 'ready');
  assert.equal(f.services.length, 2);
  assert.equal(f.services[1].getState().revision, 1);
  assert.deepEqual(f.states.map(state => state.revision), [1, 2, 3, 4]);
  assert.equal('code' in f.controller.getState(), false);
  await f.controller.stop();
});

test('concurrent explicit retries share one operation and construct one replacement', async () => {
  const startGate = deferred();
  const f = fixture((service, index) => {
    if (!index) service.startAction = async () => { throw new Error('failed'); };
    else service.startAction = async () => { await startGate.promise; service.emit('ready'); };
  });
  await f.controller.start();
  const retries = Array.from({ length: 20 }, () => f.controller.retry());
  for (const retry of retries) assert.equal(retry, retries[0]);
  await settle();
  assert.equal(f.services.length, 2);
  assert.equal(f.services[0].stops, 1);
  assert.equal(f.services[1].starts, 1);
  startGate.resolve();
  assert.equal((await Promise.all(retries)).every(state => state.status === 'ready'), true);
  await f.controller.stop();
});

test('starting, reconnecting, and ready cannot be replaced by a retry', async () => {
  const startGate = deferred();
  const f = fixture(service => { service.startAction = async () => { await startGate.promise; service.emit('ready'); }; });
  assert.deepEqual(await f.controller.retry(), { status: 'starting', revision: 0 });
  assert.equal(f.services.length, 0);
  const start = f.controller.start();
  assert.equal(f.controller.start(), start);
  assert.deepEqual(await f.controller.retry(), { status: 'starting', revision: 1 });
  await settle();
  assert.equal(f.services.length, 1);
  startGate.resolve();
  await start;
  assert.deepEqual(await f.controller.retry(), f.controller.getState());
  f.services[0].emit('reconnecting');
  assert.equal((await f.controller.retry()).status, 'reconnecting');
  assert.equal(f.services.length, 1);
  assert.equal(f.services[0].stops, 0);
  await f.controller.stop();
});

test('quit invalidates pending startup and recovery callbacks before waiting for shutdown', async () => {
  const startGate = deferred();
  const stopGate = deferred();
  const f = fixture(service => {
    service.startAction = async () => { await startGate.promise; service.emit('ready'); };
    service.stopAction = () => stopGate.promise;
  });
  const start = f.controller.start();
  await settle();
  const before = f.controller.getState();
  const stop = f.controller.stop();
  assert.equal(f.controller.stop(), stop);
  assert.equal(f.controller.getService(), null);
  f.services[0].emit('reconnecting');
  f.services[0].callbacks.onFailure('SYNTHETIC_SECRET');
  startGate.resolve();
  stopGate.resolve();
  await Promise.all([start, stop]);
  assert.deepEqual(f.controller.getState(), before);
  assert.equal(f.services[0].stops, 1);
  assert.equal(f.states.length, 1);
  assert.equal(f.failures.length, 0);
  await f.controller.retry();
  await f.controller.start();
  assert.equal(f.services.length, 1);
  assert.equal(f.controller.getDiagnostics().at(-1)?.code, 'SERVICE_STOPPED');
});

test('stale instance and out-of-order source callbacks cannot roll back current state', async () => {
  const f = fixture();
  await f.controller.start();
  const original = f.services[0];
  original.emit('failed');
  assert.equal(f.controller.getState().code, 'SERVICE_FAILED');
  await f.controller.retry();
  const before = f.controller.getState();
  original.emit('ready', 1000);
  original.callbacks.onFailure('late old failure');
  f.services[1].emit('failed', 0);
  assert.deepEqual(f.controller.getState(), before);
  assert.equal(f.controller.getService(), f.services[1]);
  assert.deepEqual(f.states.map(state => state.revision), f.states.map((_, i) => i + 1));
  await f.controller.stop();
});

test('quit during replacement waits for old stop and never constructs another service', async () => {
  const stopGate = deferred();
  const f = fixture(service => {
    service.startAction = async () => { throw new Error('failed'); };
    service.stopAction = () => stopGate.promise;
  });
  await f.controller.start();
  const retry = f.controller.retry();
  await settle();
  const stop = f.controller.stop();
  stopGate.resolve();
  await Promise.all([retry, stop]);
  assert.equal(f.services.length, 1);
  assert.equal(f.services[0].stops, 1);
  assert.equal(f.controller.getService(), null);
});

test('a rejected old stop blocks construction until a later retry stops it successfully', async () => {
  let rejectStop = true;
  const f = fixture((service, index) => {
    if (!index) {
      service.startAction = async () => { throw new Error('failed'); };
      service.stopAction = async () => { if (rejectStop) throw new Error('SYNTHETIC_SECRET stop'); };
    }
  });
  await f.controller.start();
  assert.equal((await f.controller.retry()).code, 'SERVICE_STOP_FAILED');
  assert.equal(f.services.length, 1);
  assert.equal(f.controller.getService(), f.services[0]);
  rejectStop = false;
  assert.equal((await f.controller.retry()).status, 'ready');
  assert.equal(f.services.length, 2);
  assert.equal(f.services[0].stops, 2);
  assert.doesNotMatch(JSON.stringify(f.controller.getDiagnostics()), /SYNTHETIC_SECRET/);
  await f.controller.stop();
});

test('diagnostics are bounded snapshots with only fixed codes, status, and ISO timestamps', async () => {
  const f = fixture();
  await f.controller.start();
  for (let i = 0; i < 60; i++) f.services[0].emit(i % 2 ? 'ready' : 'reconnecting');
  const history = f.controller.getDiagnostics();
  assert.equal(history.length, 50);
  for (const event of history) {
    assert.deepEqual(Object.keys(event).sort(), ['code', 'status', 'timestamp']);
    assert.equal(event.timestamp, '2026-10-09T00:00:00.000Z');
    assert.match(event.code, /^SERVICE_(READY|RECONNECTING)$/);
  }
  history[0].code = 'SERVICE_FAILED';
  history.pop();
  assert.equal(f.controller.getDiagnostics().length, 50);
  assert.notEqual(f.controller.getDiagnostics()[0].code, 'SERVICE_FAILED');
  const state = f.controller.getState(); state.status = 'failed';
  assert.equal(f.controller.getState().status, 'ready');
  assert.deepEqual(f.states.map(snapshot => snapshot.revision), f.states.map((_, i) => i + 1));
  assert.doesNotMatch(JSON.stringify(f.controller.getDiagnostics()), /SYNTHETIC_SECRET|private|message/);
  await f.controller.stop();
});

test('synchronous factory callback requesting quit cleans up the late instance without starting it', async () => {
  let service!: ControlledService;
  let stop!: Promise<void>;
  const controller: ServiceController<ControlledService> = new ServiceController({
    factory: callbacks => {
      service = new ControlledService(callbacks);
      service.emit('ready');
      return service;
    },
    onState: state => { if (state.status === 'ready') stop = controller.stop(); },
  });
  await controller.start();
  await stop;
  assert.equal(service.starts, 0);
  assert.equal(service.stops, 1);
  assert.equal(controller.getService(), null);
});
