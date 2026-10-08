import assert from 'node:assert/strict';
import { test } from 'node:test';
import { eventPresentation } from '../src/eventPresentation.ts';

const english = (_zh, en) => en;
test('actual candidate execution and evaluation failures cannot receive completion markers', () => {
  for (const value of [
    { event: 'execution_finished', exc_type: 'EvaluatorValidationError' },
    { event: 'node_evaluated', is_buggy: true },
    { event: 'node_evaluated', exc_type: 'TimeoutError' },
    { event: 'final_retrain_finished', ok: false },
  ]) {
    const result = eventPresentation(value, english);
    assert.equal(result.failure, true);
    assert.equal(result.complete, false);
    assert.match(result.title, /failed/);
  }
});
test('unknown and malformed historical fields stay renderable and cannot use map prototypes', () => {
  for (const event of ['constructor', '__proto__', 'toString', { bad: true }, null, 17]) {
    const result = eventPresentation({ event, ts: { bad: true }, step: { bad: true }, message: ['bad'] }, english);
    assert.equal(typeof result.title, 'string');
    assert.equal(result.timestamp, undefined);
    assert.equal(result.step, '');
    assert.equal(result.summary, '');
    assert.equal(result.failure, false);
  }
});
test('actual engine generation/tool/execution records receive direct labels without invented progress', () => {
  assert.equal(eventPresentation({ event: 'node_generation_started' }, english).title, 'Candidate generation started');
  assert.equal(eventPresentation({ event: 'tool_observation' }, english).title, 'Tool result recorded');
  const execution = eventPresentation({ event: 'execution_finished', exc_type: null, step: 0 }, english);
  assert.equal(execution.failure, false);
  assert.equal(execution.complete, true);
  assert.equal(execution.step, '0');
  assert.equal(eventPresentation({ event: 'node_evaluated', is_buggy: false }, english).failure, false);
});
test('non-finite and structured field values are not rendered as text', () => {
  for (const step of [NaN, Infinity, -Infinity, {}, [], true]) {
    const result = eventPresentation({ event: 'node_started', step }, english);
    assert.equal(result.step, '');
  }
});
