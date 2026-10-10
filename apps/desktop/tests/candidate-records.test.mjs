import assert from 'node:assert/strict';
import { test } from 'node:test';
import { candidateNumber, candidatePhaseLabel, keepCandidateSelection, readCandidateRecords } from '../src/candidateRecords.ts';
const english = (_zh, en) => en;
const records = nodes => readCandidateRecords({ state: 'ready', nodes, total: nodes.length });

test('live generation and queue remain visible without an authoritative journal or score', () => {
  const value = records([{ id: 'one', phase: 'generating', recorded: false, metric: { value: 10 }, is_buggy: false }, { id: 'two', phase: 'queued', code: '# generated' }]);
  assert.equal(value.nodes[0].phase, 'generating');
  assert.equal(value.nodes[0].metric, undefined);
  assert.equal(value.nodes[1].code, '# generated');
  assert.equal(candidatePhaseLabel(value.nodes[1].phase, english), 'Queued');
});
test('transient completed-looking fields cannot fabricate valid results', () => {
  const node = records([{ id: 'one', phase: 'valid', recorded: false, is_buggy: false, metric: { value: 0.9 } }]).nodes[0];
  assert.equal(node.phase, 'unconfirmed');
  assert.equal(node.metric, undefined);
  const valid = records([{ id: 'one', phase: 'valid', recorded: true, is_buggy: false, metric: { value: 0, maximize: false } }]).nodes[0];
  assert.equal(valid.metric.value, 0);
  assert.equal(valid.metric.maximize, false);
});
test('poll updates keep the selected candidate as other candidates arrive', () => {
  const first = records([{ id: 'one', phase: 'running' }]);
  const next = records([{ id: 'one', phase: 'evaluating' }, { id: 'two', phase: 'running' }]);
  assert.equal(keepCandidateSelection(first.nodes, null), 'one');
  assert.equal(keepCandidateSelection(next.nodes, 'one'), 'one');
  assert.equal(keepCandidateSelection(next.nodes, 'missing'), 'two');
});
test('saved candidate numbers agree with Results when completion and scheduling order differ', () => {
  const saved = records([{ id: 'best', step: 1, scheduled_step: 2, recorded: true, is_buggy: false, phase: 'valid' }]).nodes[0];
  assert.equal(candidateNumber(saved), '#2');
  assert.equal(candidateNumber({ ...saved, recorded: false, phase: 'running' }), '#3');
  assert.equal(candidateNumber({ ...saved, step: null }), '#3');
});
test('a disconnected or changing source rejects replacement so callers retain the prior view', () => {
  for (const state of ['updating', 'unavailable']) assert.throws(() => readCandidateRecords({ state, nodes: [] }));
  assert.equal(candidatePhaseLabel('interrupted', english), 'Interrupted');
  assert.equal(candidatePhaseLabel('unconfirmed', english), 'Result unconfirmed');
  assert.equal(candidatePhaseLabel('failed', english), 'Failed');
});
test('malformed history and duplicate identifiers cannot become renderable valid metrics', () => {
  assert.throws(() => records([{ id: 'one' }, { id: 'one' }]));
  const node = records([{ id: 'old', phase: 'failed', recorded: true, is_buggy: true, metric: { value: NaN }, _term_out: ['failure', {}] }]).nodes[0];
  assert.equal(node.metric.value, null);
  assert.deepEqual(node._term_out, ['failure']);
});
