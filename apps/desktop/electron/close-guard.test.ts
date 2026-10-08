import test from 'node:test';
import assert from 'node:assert/strict';
import { CloseGuard, type CloseBlockReason } from './close-guard';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(yes => { resolve = yes; });
  return { promise, resolve };
}
const settle = () => new Promise<void>(resolve => setImmediate(resolve));
function fixture() {
  let window: number | null = 1;
  let answer = () => false;
  const shutdown = deferred<void>();
  const calls: string[] = [];
  const reasons: CloseBlockReason[] = [];
  const guard = new CloseGuard({
    windowId: () => window,
    closeWindow: () => { calls.push('close'); },
    confirmClose: reason => { reasons.push(reason); return answer(); },
    stop: () => { calls.push('stop'); return shutdown.promise; },
    quit: () => { calls.push('quit'); },
  });
  return { guard, shutdown, calls, reasons, setWindow: (id: number | null) => { window = id; }, answerWith: (callback: () => boolean) => { answer = callback; } };
}

test('cancelled quit preserves the window and service; later window-only close does not quit', () => {
  const f = fixture();
  f.guard.requestQuit();
  assert.deepEqual(f.calls, ['close']);
  assert.equal(f.guard.blockedUnload('unsaved'), false);
  assert.deepEqual(f.calls, ['close']);
  f.setWindow(null); f.guard.windowClosed();
  assert.deepEqual(f.calls, ['close']);
  assert.equal(f.guard.readyToQuit, false);
});

test('discard allows the current unload; service stops only after the window closes', async () => {
  const f = fixture(); f.answerWith(() => true);
  f.guard.requestQuit();
  assert.equal(f.guard.blockedUnload('unsaved'), true);
  assert.deepEqual(f.calls, ['close']);
  f.setWindow(null); f.guard.windowClosed();
  assert.deepEqual(f.calls, ['close', 'stop']);
  assert.equal(f.guard.readyToQuit, false);
  f.guard.requestQuit();
  assert.deepEqual(f.calls, ['close', 'stop']);
  f.shutdown.resolve(); await settle();
  assert.equal(f.guard.readyToQuit, true);
  assert.deepEqual(f.calls, ['close', 'stop', 'quit']);
});

test('reentrant close and quit attempts cannot open a second confirmation', () => {
  const f = fixture();
  f.answerWith(() => {
    f.guard.requestQuit();
    assert.equal(f.guard.blockedUnload('unsaved'), false);
    return false;
  });
  assert.equal(f.guard.blockedUnload('unsaved'), false);
  assert.deepEqual(f.reasons, ['unsaved']);
  f.setWindow(null); f.guard.windowClosed();
  assert(!f.calls.includes('stop'));
});

test('saving never grants permission to discard or stop the service', () => {
  const f = fixture(); f.answerWith(() => true);
  f.guard.requestQuit();
  assert.equal(f.guard.blockedUnload('saving'), false);
  f.setWindow(null); f.guard.windowClosed();
  assert.deepEqual(f.calls, ['close']);
  assert.deepEqual(f.reasons, ['saving']);
});

test('failure to show a confirmation keeps edits and cancels quit safely', () => {
  const f = fixture(); f.answerWith(() => { throw new Error('Native dialog unavailable'); });
  f.guard.requestQuit();
  assert.equal(f.guard.blockedUnload('unsaved'), false);
  f.setWindow(null); f.guard.windowClosed();
  assert.deepEqual(f.calls, ['close']);
});

test('quit with no window stops once and waits before allowing the final quit', async () => {
  const f = fixture(); f.setWindow(null);
  f.guard.requestQuit(); f.guard.requestQuit();
  assert.deepEqual(f.calls, ['stop']);
  assert.equal(f.guard.readyToQuit, false);
  f.shutdown.resolve(); await settle();
  assert.deepEqual(f.calls, ['stop', 'quit']);
  assert.equal(f.guard.readyToQuit, true);
});
