import test from 'node:test';
import assert from 'node:assert/strict';
import { RunNotifications } from './run-notifications';

test('historical results stay silent; active runs notify once including canonical incomplete failures', () => {
  const tracker = new RunNotifications();
  assert.deepEqual(tracker.observe([{ id: 'old', status: 'complete' }, { id: 'a', status: 'running' }, { id: 'b', status: 'queued' }], tracker.issue()), []);
  assert.deepEqual(tracker.observe([{ id: 'a', status: 'complete', detailed_status: 'succeeded' }, { id: 'b', status: 'incomplete', detailed_status: 'failed' }], tracker.issue()), [{ id: 'a', outcome: 'complete' }, { id: 'b', outcome: 'failed' }]);
  assert.deepEqual(tracker.observe([{ id: 'a', status: 'complete' }, { id: 'b', status: 'incomplete' }], tracker.issue()), []);
  assert.equal(tracker.activeCount, 0);
});

test('out-of-order concurrent requests cannot revive completed runs or duplicate notifications', () => {
  const tracker = new RunNotifications();
  tracker.observe({ id: 'a', status: 'running' }, tracker.issue());
  const older = tracker.issue(), newer = tracker.issue();
  assert.equal(tracker.observe({ id: 'a', status: 'complete' }, newer).length, 1);
  assert.deepEqual(tracker.observe({ id: 'a', status: 'running' }, older), []);
  assert.deepEqual(tracker.observe({ id: 'a', status: 'complete' }, tracker.issue()), []);
});

test('user stops and cancellations stay quiet, while retries and unexpected interruptions remain observable', () => {
  const tracker = new RunNotifications();
  for (const status of ['queued', 'stopping', 'failed']) tracker.observe({ id: 'a', status }, tracker.issue());
  assert.equal(tracker.activeCount, 0);
  tracker.observe({ id: 'a', status: 'queued' }, tracker.issue());
  assert.deepEqual(tracker.observe({ id: 'a', status: 'cancelled' }, tracker.issue()), []);
  tracker.observe({ id: 'a', status: 'running' }, tracker.issue());
  assert.deepEqual(tracker.observe({ id: 'a', status: 'incomplete', detailed_status: 'interrupted' }, tracker.issue()), [{ id: 'a', outcome: 'interrupted' }]);
  tracker.observe({ id: 'a', status: 'queued' }, tracker.issue());
  assert.deepEqual(tracker.observe({ id: 'a', status: 'complete' }, tracker.issue()), [{ id: 'a', outcome: 'complete' }]);
});

test('a newer response for another run cannot discard an active observation', () => {
  const tracker = new RunNotifications();
  const earlier = tracker.issue(), later = tracker.issue();
  tracker.observe({ id: 'b', status: 'complete' }, later);
  tracker.observe({ id: 'a', status: 'running' }, earlier);
  assert.deepEqual(tracker.observe({ id: 'a', status: 'complete' }, tracker.issue()), [{ id: 'a', outcome: 'complete' }]);
});

test('malformed status responses do not erase a known active transition', () => {
  const tracker = new RunNotifications();
  tracker.observe({ id: 'a', status: 'running' }, tracker.issue());
  assert.deepEqual(tracker.observe([null, { id: '../a', status: 'running' }, { id: 'a', status: 3 }], tracker.issue()), []);
  assert.deepEqual(tracker.observe({ id: 'a', status: 'complete' }, tracker.issue()), [{ id: 'a', outcome: 'complete' }]);
});
