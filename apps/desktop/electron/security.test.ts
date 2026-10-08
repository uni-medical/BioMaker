import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { externalURL, isWithin, validateArtifact, validateRequest } from './security';

test('request bridge rejects origins, traversal and encoded separators before normalization', () => {
  for (const value of ['https://evil.test/api/status', '//evil.test/api/status', '/api/../private', '/api/%2e%2e/private', '/api/%252e%252e/private', '/api/foo%2fbar', '/api/a\\b', '/api/a\n']) {
    assert.throws(() => validateRequest(value), { name: 'Error' }, value);
  }
  assert.deepEqual(validateRequest('/api/runs?limit=10'), { path: '/api/runs?limit=10', method: 'GET', body: undefined });
  assert.equal(validateRequest('/api/runs', { method: 'post', body: { name: '研究' } }).method, 'POST');
});
test('renderer cannot supply headers, alter credentials, or mutate artifacts', () => {
  assert.throws(() => validateRequest('/api/runs', { headers: { Authorization: 'x' } }));
  assert.throws(() => validateRequest('/api/runs', { method: 'TRACE' }));
  assert.throws(() => validateRequest('/artifacts/runs/abc/report.html', { method: 'POST' }));
  assert.throws(() => validateRequest('/api/runs', { method: 'POST', body: 'x'.repeat(2_100_000) }));
});
test('engine checks and repairs use the same bounded native request contract as installation', () => {
  for (const action of ['install', 'cancel', 'check', 'repair']) {
    const endpoint = `/api/desktop/engine/${action}`;
    assert.deepEqual(validateRequest(endpoint, { method: 'post' }), { path: endpoint, method: 'POST', body: undefined });
    assert.throws(() => validateRequest(endpoint, { method: 'POST', headers: { Authorization: 'renderer-token' } }));
    assert.throws(() => validateRequest(endpoint, { method: 'POST', credentials: 'include' }));
    assert.throws(() => validateRequest(endpoint, { method: 'TRACE' }));
    assert.throws(() => validateRequest(endpoint, { method: 'POST', body: 'x'.repeat(2_100_000) }));
    assert.throws(() => validateRequest(`https://example.test${endpoint}`, { method: 'POST' }));
    assert.throws(() => validateRequest(`/api/desktop/engine/%2e%2e/${action}`, { method: 'POST' }));
    assert.throws(() => validateRequest(`/api/desktop/engine%2f${action}`, { method: 'POST' }));
  }
});
test('artifact names cannot escape a run directory', () => {
  for (const name of ['../key', '/etc/passwd', '..', '.env', 'a\\b', 'a\0b']) assert.throws(() => validateArtifact('abc', name));
  assert.throws(() => validateArtifact('../abc', 'report.html'));
  assert.deepEqual(validateArtifact('run_123-a', '实验报告.html'), { runId: 'run_123-a', name: '实验报告.html' });
  assert.equal(validateArtifact('derm.v2-35step', 'results.tsv').runId, 'derm.v2-35step');
  assert.equal(validateArtifact('历史实验 1', 'results.tsv').runId, '历史实验 1');
});
test('path containment includes separator boundary', () => {
  const root = path.resolve('workspace');
  assert.ok(isWithin(path.join(root, 'logs', 'run'), root));
  assert.ok(!isWithin(`${root}-private/key`, root));
  assert.ok(!isWithin(path.resolve(root, '..'), root));
});
test('external links are HTTPS on explicit hosts and do not carry userinfo', () => {
  assert.equal(externalURL('https://github.com/uni-medical/BioMaker'), 'https://github.com/uni-medical/BioMaker');
  for (const url of ['file:///etc/passwd', 'javascript:alert(1)', 'https://github.com.evil.test', 'https://x@github.com', 'https://github.com:8443']) assert.throws(() => externalURL(url));
});
