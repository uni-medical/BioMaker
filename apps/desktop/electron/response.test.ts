import test from 'node:test';
import assert from 'node:assert/strict';
import { readResponse } from './response';
import { MAX_RESPONSE_BYTES } from './security';

test('journal artifacts preserve nonfinite Python numbers and original formatting', async () => {
  const journal = '{\n  "nodes": [{"metric": NaN}, {"metric": Infinity}, {"metric": -Infinity}]\n}\n';
  const artifact = new Response(journal, { headers: { 'Content-Type': 'application/json' } });
  assert.equal(await readResponse(artifact, false), journal);
  const ordinaryJSON = ' { "experiment": "实验" }\n';
  assert.equal(await readResponse(new Response(ordinaryJSON, { headers: { 'Content-Type': 'application/json' } }), false), ordinaryJSON);
});
test('API JSON stays strict and API errors stay actionable for artifact requests', async () => {
  assert.deepEqual(await readResponse(new Response('{"ready":true}', { headers: { 'Content-Type': 'application/json' } })), { ready: true });
  await assert.rejects(readResponse(new Response('{"metric":NaN}', { headers: { 'Content-Type': 'application/json' } })), /无效数据/);
  await assert.rejects(readResponse(new Response('{"detail":"Artifact does not exist"}', { status: 404, headers: { 'Content-Type': 'application/json' } }), false), /Artifact does not exist/);
});
test('artifact text is bounded before reading an oversized response', async () => {
  await assert.rejects(readResponse(new Response('x', { headers: { 'Content-Length': String(MAX_RESPONSE_BYTES + 1) } }), false), /导出文件/);
});
