import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { artifactService, consoleRequest, createArtifactSaver, MAX_CONSOLE_DOWNLOAD_BYTES, saveArtifact } from './artifacts';

const target = { runId: 'registered-console-run', name: 'console.log' };
const digest = (value: Uint8Array | string) => createHash('sha256').update(value).digest('hex');
const noChange = () => {};
const fixed = (code: string) => (error: unknown) => {
  assert(error instanceof Error);
  assert(error.message.startsWith(`[artifact:${code}]`), error.message);
  assert(!error.message.includes('private-sentinel'));
  return true;
};
const headers = (bytes: Uint8Array) => ({
  'content-type': 'application/octet-stream', 'content-length': String(bytes.byteLength),
  'x-biomaker-console-sha256': digest(bytes),
});
const response = (bytes: Uint8Array = Buffer.from('complete log'), overrides: Record<string, string> = {}) =>
  new Response(Buffer.from(bytes), { headers: { ...headers(bytes), ...overrides } });
function destination(t: { after(fn: () => void): void }) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'biomaker-console-export-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = path.join(root, 'saved.log');
  writeFileSync(file, 'existing destination');
  return { root, file, preserved() {
    assert.equal(readFileSync(file, 'utf8'), 'existing destination');
    assert.deepEqual(readdirSync(root), ['saved.log']);
  } };
}
const save = (choose: () => Promise<string | null>, send: () => Promise<Response>, current = noChange) =>
  saveArtifact(target, choose, send, current, 'console');

test('console requests accept only a registered run identity and exact renderer service generation', () => {
  assert.deepEqual(consoleRequest({ runId: '实验-1', serviceRevision: 0 }), { runId: '实验-1', serviceRevision: 0 });
  const valid = { runId: target.runId, serviceRevision: 3 };
  for (const input of [null, [], {}, { runId: target.runId }, { serviceRevision: 3 },
    ...['', '.', '..', '../private-sentinel', 'a/b', 'a\\b', '\0', 'x'.repeat(181)].map(runId => ({ ...valid, runId })),
    ...[-1, 1.5, Number.NaN, Number.POSITIVE_INFINITY, Number.MAX_SAFE_INTEGER + 1, '3', null].map(serviceRevision => ({ ...valid, serviceRevision })),
    ...['path', 'name', 'filename', 'content', 'url', 'expectedSha256'].map(key => ({ ...valid, [key]: 'private-sentinel' })),
    Object.create(valid)]) assert.throws(() => consoleRequest(input), fixed('invalid_request'));
});

test('full console export preserves exact raw bytes beyond preview size and publishes private permissions', async t => {
  const output = destination(t);
  const raw = Buffer.concat([Buffer.from('\ufeffUnicode 实验\r\n'), Buffer.from([255, 0, 13, 10]), Buffer.alloc(600_000, 65)]);
  assert.equal(await save(async () => output.file, async () => response(raw)), output.file);
  assert.deepEqual(readFileSync(output.file), raw);
  assert.deepEqual(readdirSync(output.root), ['saved.log']);
  if (process.platform !== 'win32') assert.equal(statSync(output.file).mode & 0o777, 0o600);
  assert.equal(await save(async () => output.file, async () => response(Buffer.alloc(0))), output.file);
  assert.equal(readFileSync(output.file).length, 0);
});

test('console download refuses missing or malformed snapshot evidence before replacing a destination', async t => {
  const output = destination(t);
  const raw = Buffer.from('complete log');
  for (const key of ['content-length', 'x-biomaker-console-sha256', 'content-type']) {
    const bad = new Headers(headers(raw)); bad.delete(key);
    await assert.rejects(save(async () => output.file, async () => new Response(raw, { headers: bad })), fixed('unavailable'));
    output.preserved();
  }
  const invalidHeaders: Record<string, string>[] = [
    { 'content-length': '-1' }, { 'content-length': '1.2' }, { 'content-length': '01' }, { 'content-length': '9007199254740992' },
    { 'x-biomaker-console-sha256': 'not-a-digest' }, { 'x-biomaker-console-sha256': 'A'.repeat(64) },
    { 'content-type': 'text/html' }, { 'content-encoding': 'gzip' },
  ];
  for (const bad of invalidHeaders) {
    await assert.rejects(save(async () => output.file, async () => response(raw, bad)), fixed('unavailable'));
    output.preserved();
  }
  await assert.rejects(save(async () => output.file, async () => new Response(raw, { status: 206, headers: headers(raw) })), fixed('unavailable'));
  output.preserved();
  await assert.rejects(save(async () => output.file, async () => response(raw, { 'content-length': String(MAX_CONSOLE_DOWNLOAD_BYTES + 1) })), fixed('too_large'));
  output.preserved();
});

test('underdeclared, truncated and equal-length altered console bodies never publish partial or wrong files', async t => {
  const output = destination(t);
  for (const [raw, evidence] of [
    [Buffer.from('longer'), Buffer.from('short')],
    [Buffer.from('short'), Buffer.from('longer')],
    [Buffer.from('altered!'), Buffer.from('original')],
  ]) {
    await assert.rejects(save(async () => output.file, async () => new Response(raw, { headers: headers(evidence) })), fixed('changed'));
    output.preserved();
  }
});

test('console route status codes map to fixed errors without reading or disclosing response bodies', async t => {
  const output = destination(t);
  for (const [status, code] of [[413, 'too_large'], [409, 'changed'], [404, 'unavailable'], [403, 'unavailable'], [500, 'unavailable']] as const) {
    await assert.rejects(save(async () => output.file, async () => new Response('private-sentinel', { status })), fixed(code));
    output.preserved();
  }
});

test('canceling console export opens no download and leaves the destination intact', async t => {
  const output = destination(t);
  assert.equal(await save(async () => null, async () => { throw new Error('must not fetch'); }), null);
  output.preserved();
});

test('console export rejects obsolete renderer generation and service replacement while choosing a destination', async t => {
  const output = destination(t);
  let service = {};
  assert.throws(() => artifactService(() => service, () => ({ status: 'ready', revision: 3 }), 2), fixed('service_changed'));
  const guard = artifactService(() => service, () => ({ status: 'ready', revision: 3 }), 3);
  await assert.rejects(save(async () => { service = {}; return output.file; }, async () => { throw new Error('must not fetch'); }, guard.assertCurrent), fixed('service_changed'));
  output.preserved();
});

test('console bytes from an old service generation cannot be published after reconnection', async t => {
  const output = destination(t);
  const service = {};
  let revision = 1;
  const guard = artifactService(() => service, () => ({ status: 'ready', revision }), 1);
  const raw = Buffer.from('old generation full console');
  const body = new ReadableStream({ pull(controller) { controller.enqueue(raw); revision = 3; controller.close(); } });
  await assert.rejects(save(async () => output.file, async () => new Response(body, { headers: headers(raw) }), guard.assertCurrent), fixed('service_changed'));
  output.preserved();
});

test('interrupted console streams and native filesystem failures preserve existing targets and remove temporary files', async t => {
  const output = destination(t);
  const raw = Buffer.from('complete log');
  let chunk = 0;
  const body = new ReadableStream({ pull(controller) {
    if (chunk++ === 0) controller.enqueue(Buffer.from('part'));
    else controller.error(new Error('private-sentinel'));
  } });
  await assert.rejects(save(async () => output.file, async () => new Response(body, { headers: headers(raw) })), fixed('save_failed'));
  output.preserved();
  await assert.rejects(save(async () => path.join(output.root, 'missing-private-sentinel', 'log'), async () => response(raw)), fixed('save_failed'));
  output.preserved();
  const directory = path.join(output.root, 'existing-directory');
  mkdirSync(directory); writeFileSync(path.join(directory, 'keep'), 'keep');
  await assert.rejects(save(async () => directory, async () => response(raw)), fixed('save_failed'));
  assert.equal(readFileSync(path.join(directory, 'keep'), 'utf8'), 'keep');
  assert.deepEqual(readdirSync(output.root).sort(), ['existing-directory', 'saved.log']);
});

for (const firstSource of ['console', 'artifact'] as const) test(`the app-wide export guard blocks ${firstSource === 'console' ? 'artifact' : 'console'} saves during a ${firstSource} chooser`, async t => {
  const output = destination(t);
  const sharedSave = createArtifactSaver();
  let release: (choice: string | null) => void = () => {};
  const choice = new Promise<string | null>(resolve => { release = resolve; });
  const first = sharedSave(target, () => choice, async () => { throw new Error('must not fetch'); }, noChange, firstSource);
  const secondSource = firstSource === 'console' ? 'artifact' : 'console';
  await assert.rejects(sharedSave({ runId: 'other-run', name: 'report.md' }, async () => { throw new Error('must not show another dialog'); }, async () => { throw new Error('must not fetch'); }, noChange, secondSource), fixed('busy'));
  release(null); assert.equal(await first, null);
  output.preserved();
  assert.equal(await sharedSave(target, async () => output.file, async () => response(), noChange, 'console'), output.file);
});

test('a failed console integrity check releases the same global guard for a later explicit artifact save', async t => {
  const output = destination(t);
  const sharedSave = createArtifactSaver();
  await assert.rejects(sharedSave(target, async () => output.file, async () => response(Buffer.from('changed'), { 'x-biomaker-console-sha256': digest('original') }), noChange, 'console'), fixed('changed'));
  output.preserved();
  assert.equal(await sharedSave(target, async () => output.file, async () => new Response('explicit later artifact'), noChange), output.file);
  assert.equal(readFileSync(output.file, 'utf8'), 'explicit later artifact');
});
