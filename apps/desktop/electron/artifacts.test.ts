import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { artifactRequest, artifactService, previewArtifact, saveArtifact, createArtifactSaver, MAX_PREVIEW_BYTES, MAX_PREVIEW_RESPONSE_BYTES } from './artifacts';

const target = { runId: 'registered-run', name: '实验 report.md' };
const digest = (bytes: Buffer | string) => createHash('sha256').update(bytes).digest('hex');
const snapshot = (bytes: Buffer | string, changes = {}) => {
  const raw = Buffer.from(bytes);
  return { name: target.name, content: raw.toString('utf8'), sha256: digest(raw), size: raw.length, ...changes };
};
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const fixed = (code: string) => (error: unknown) => {
  assert(error instanceof Error);
  assert(error.message.startsWith(`[artifact:${code}]`), error.message);
  assert(!error.message.includes('private-sentinel'));
  return true;
};
const noChange = () => {};
function destination(t: { after(fn: () => void): void }) {
  const root = mkdtempSync(path.join(os.tmpdir(), 'biomaker-artifact-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const file = path.join(root, 'saved.md');
  writeFileSync(file, 'existing destination');
  return { root, file, assertPreserved() {
    assert.equal(readFileSync(file, 'utf8'), 'existing destination');
    assert.deepEqual(readdirSync(root), ['saved.md']);
  } };
}

test('native artifact requests validate paths and paired snapshot evidence before opening a dialog', () => {
  assert.deepEqual(artifactRequest(target), target);
  assert.deepEqual(artifactRequest(target, true), target);
  const evidence = { ...target, expectedSha256: digest('x'), serviceRevision: 2 };
  assert.deepEqual(artifactRequest(evidence, true), evidence);
  for (const input of [null, [], { ...target, name: '../private-sentinel' }, { ...target, runId: '..' },
    { ...target, expectedSha256: digest('x') }, { ...target, serviceRevision: 2 },
    { ...evidence, expectedSha256: 'bad' }, { ...evidence, serviceRevision: -1 },
    { ...evidence, serviceRevision: 1.2 }, { ...evidence, serviceRevision: Number.MAX_SAFE_INTEGER + 1 },
    { ...target, content: 'private-sentinel' }]) assert.throws(() => artifactRequest(input, true), fixed('invalid_request'));
  assert.throws(() => artifactRequest(evidence), fixed('invalid_request'));
});

test('service guard binds both object identity and revision, including same-object reconnection', () => {
  let service: object | null = {};
  let state = { status: 'ready', revision: 2 };
  const guard = artifactService(() => service, () => state, 2);
  guard.assertCurrent();
  state = { status: 'ready', revision: 4 };
  assert.throws(guard.assertCurrent, fixed('service_changed'));
  state = { status: 'ready', revision: 2 };
  service = {};
  assert.throws(guard.assertCurrent, fixed('service_changed'));
  assert.throws(() => artifactService(() => service, () => state, 1), fixed('service_changed'));
  service = null;
  assert.throws(() => artifactService(() => service, () => state), fixed('unavailable'));
});

test('preview validates exact UTF-8 bytes while preserving BOM, CRLF, Unicode and nonfinite JSON text', async () => {
  const raw = Buffer.from('\ufeff{ "value": NaN, "message": "实验 😀" }\r\n');
  const result = await previewArtifact(target, 7, async () => json(snapshot(raw)), noChange);
  assert.deepEqual(result, { ...target, ...snapshot(raw), serviceRevision: 7 });
  assert.deepEqual(Buffer.from(result.content), raw);
  const empty = await previewArtifact(target, 7, async () => json(snapshot('')), noChange);
  assert.equal(empty.size, 0);
});

test('malformed preview responses and false content digests cannot create snapshot evidence', async () => {
  for (const value of [[], null, snapshot('hello', { name: 'other.md' }), snapshot('hello', { size: 4 }),
    snapshot('hello', { sha256: digest('world') }), snapshot('hello', { size: -1 }),
    snapshot('hello', { content: 123 }), snapshot('hello', { sha256: 'A'.repeat(64) })]) {
    await assert.rejects(previewArtifact(target, 1, async () => json(value), noChange), fixed('unavailable'));
  }
  await assert.rejects(previewArtifact(target, 1, async () => json(snapshot('x', { content: '\ud800' })), noChange), fixed('invalid_text'));
  await assert.rejects(previewArtifact(target, 1, async () => new Response(Buffer.from([255])), noChange), fixed('unavailable'));
});

test('preview enforces declared, actual envelope and decoded UTF-8 size bounds', async () => {
  await assert.rejects(previewArtifact(target, 1, async () => new Response('x', { headers: { 'Content-Length': String(MAX_PREVIEW_RESPONSE_BYTES + 1) } }), noChange), fixed('too_large'));
  await assert.rejects(previewArtifact(target, 1, async () => new Response('x'.repeat(MAX_PREVIEW_RESPONSE_BYTES + 1)), noChange), fixed('too_large'));
  await assert.rejects(previewArtifact(target, 1, async () => json(snapshot('字'.repeat(170_000))), noChange), fixed('too_large'));
  const atLimit = '\0'.repeat(MAX_PREVIEW_BYTES);
  assert.equal((await previewArtifact(target, 1, async () => json(snapshot(atLimit)), noChange)).size, MAX_PREVIEW_BYTES);
});

test('preview errors use allowlisted codes and never echo service, transport or parse details', async () => {
  for (const code of ['changed', 'too_large', 'invalid_text', 'unavailable']) {
    await assert.rejects(previewArtifact(target, 1, async () => json({ code, detail: '/private-sentinel' }, 409), noChange), fixed(code));
  }
  await assert.rejects(previewArtifact(target, 1, async () => json({ code: 'private-sentinel', detail: 'private-sentinel' }, 500), noChange), fixed('unavailable'));
  await assert.rejects(previewArtifact(target, 1, async () => { throw new Error('private-sentinel'); }, noChange), fixed('unavailable'));
});

test('a buffered old-service preview is rejected after the service revision changes', async () => {
  const service = {};
  let revision = 1;
  const guard = artifactService(() => service, () => ({ status: 'ready', revision }));
  const body = Buffer.from(JSON.stringify(snapshot('original preview')));
  const response = new Response(new ReadableStream({ pull(controller) { controller.enqueue(body); revision = 2; controller.close(); } }));
  await assert.rejects(previewArtifact(target, 1, async () => response, guard.assertCurrent), fixed('service_changed'));
});

test('preview Save uses exact original bytes and atomically replaces an existing destination', async t => {
  const { file, root } = destination(t);
  const raw = Buffer.from('\ufeffUnicode 实验\r\n{ "value": NaN }\r\n');
  const result = await saveArtifact({ ...target, expectedSha256: digest(raw), serviceRevision: 1 }, async () => file, async () => new Response(raw), noChange);
  assert.equal(result, file);
  assert.deepEqual(readFileSync(file), raw);
  assert.deepEqual(readdirSync(root), ['saved.md']);
  if (process.platform !== 'win32') assert.equal(statSync(file).mode & 0o777, 0o600);
});

test('content changes, including equal-length changes, preserve the existing destination and remove temporary files', async t => {
  const output = destination(t);
  for (const value of ['changed!', 'x'.repeat(MAX_PREVIEW_BYTES + 1)]) {
    await assert.rejects(saveArtifact({ ...target, expectedSha256: digest('original'), serviceRevision: 1 }, async () => output.file, async () => new Response(value), noChange), fixed('changed'));
    output.assertPreserved();
  }
});

test('canceling the native save dialog never fetches or changes the destination', async t => {
  const output = destination(t);
  const result = await saveArtifact(target, async () => null, async () => { throw new Error('must not fetch'); }, noChange);
  assert.equal(result, null);
  output.assertPreserved();
});

test('service replacement during native dialog prevents a fetch and preserves the destination', async t => {
  const output = destination(t);
  let service = {};
  const guard = artifactService(() => service, () => ({ status: 'ready', revision: 1 }));
  await assert.rejects(saveArtifact(target, async () => { service = {}; return output.file; }, async () => { throw new Error('must not fetch'); }, guard.assertCurrent), fixed('service_changed'));
  output.assertPreserved();
});

test('same-object service reconnection during a download prevents final publication', async t => {
  const output = destination(t);
  const service = {};
  let revision = 1;
  const guard = artifactService(() => service, () => ({ status: 'ready', revision }));
  let release: () => void = () => {};
  const wait = new Promise<void>(resolve => { release = resolve; });
  let started: () => void = () => {};
  const downloading = new Promise<void>(resolve => { started = resolve; });
  const response = new Response(new ReadableStream({ async pull(controller) {
    started(); await wait; controller.enqueue(Buffer.from('original')); controller.close();
  } }));
  const pending = saveArtifact({ ...target, expectedSha256: digest('original'), serviceRevision: 1 }, async () => output.file, async () => response, guard.assertCurrent);
  await downloading; revision = 3; release();
  await assert.rejects(pending, fixed('service_changed'));
  output.assertPreserved();
});

test('direct export retains streaming support for large binary files without preview constraints', async t => {
  const { file } = destination(t);
  const raw = Buffer.alloc(MAX_PREVIEW_BYTES * 3, 255);
  assert.equal(await saveArtifact(target, async () => file, async () => new Response(raw), noChange), file);
  assert.deepEqual(readFileSync(file), raw);
});

test('HTTP, transport, interrupted stream and filesystem failures never publish partial files or raw errors', async t => {
  const output = destination(t);
  await assert.rejects(saveArtifact(target, async () => output.file, async () => json({ detail: 'private-sentinel' }, 404), noChange), fixed('unavailable'));
  output.assertPreserved();
  await assert.rejects(saveArtifact(target, async () => output.file, async () => { throw new Error('private-sentinel'); }, noChange), fixed('unavailable'));
  output.assertPreserved();
  let chunks = 0;
  const broken = () => new Response(new ReadableStream({ pull(controller) {
    if (chunks++ === 0) controller.enqueue(Buffer.from('partial'));
    else controller.error(new Error('private-sentinel'));
  } }));
  await assert.rejects(saveArtifact(target, async () => output.file, async () => broken(), noChange), fixed('save_failed'));
  output.assertPreserved();
  await assert.rejects(saveArtifact(target, async () => path.join(output.root, 'missing-private-sentinel', 'out'), async () => new Response('x'), noChange), fixed('save_failed'));
  output.assertPreserved();
});

for (const sameDestination of [true, false]) test(`an in-flight export rejects a second ${sameDestination ? 'same' : 'different'}-destination export before dialog or network`, async t => {
  const output = destination(t);
  const save = createArtifactSaver();
  let release: () => void = () => {};
  const wait = new Promise<void>(resolve => { release = resolve; });
  let started: () => void = () => {};
  const downloading = new Promise<void>(resolve => { started = resolve; });
  let dialogs = 0;
  let requests = 0;
  const oldBytes = Buffer.from('first approved bytes');
  const first = save({ ...target, expectedSha256: digest(oldBytes), serviceRevision: 1 }, async () => { dialogs++; return output.file; }, async () => {
    requests++;
    return new Response(new ReadableStream({ async pull(controller) { started(); await wait; controller.enqueue(oldBytes); controller.close(); } }));
  }, noChange);
  await downloading;
  await assert.rejects(save(target, async () => { dialogs++; return sameDestination ? output.file : path.join(output.root, 'different.md'); }, async () => { requests++; return new Response('second bytes'); }, noChange), fixed('busy'));
  assert.equal(dialogs, 1);
  assert.equal(requests, 1);
  assert.equal(readFileSync(output.file, 'utf8'), 'existing destination');
  assert(!readdirSync(output.root).includes('different.md'));
  release();
  assert.equal(await first, output.file);
  assert.deepEqual(readFileSync(output.file), oldBytes);
  assert.deepEqual(readdirSync(output.root), ['saved.md']);
  // No queue or replay: the rejected request cannot publish after the first.
  assert.equal(requests, 1);
  await save(target, async () => output.file, async () => new Response('new explicit export'), noChange);
  assert.equal(readFileSync(output.file, 'utf8'), 'new explicit export');
});

test('exclusive export starts before the native chooser and unlocks after cancel and failures', async t => {
  const output = destination(t);
  const save = createArtifactSaver();
  let finishDialog: (value: string | null) => void = () => {};
  const choice = new Promise<string | null>(resolve => { finishDialog = resolve; });
  const pending = save(target, () => choice, async () => { throw new Error('must not fetch'); }, noChange);
  await assert.rejects(save(target, async () => { throw new Error('must not show another dialog'); }, async () => { throw new Error('must not fetch'); }, noChange), fixed('busy'));
  finishDialog(null);
  assert.equal(await pending, null);
  output.assertPreserved();
  await assert.rejects(save(target, async () => output.file, async () => { throw new Error('private-sentinel'); }, noChange), fixed('unavailable'));
  output.assertPreserved();
  await assert.rejects(save({ ...target, expectedSha256: digest('original'), serviceRevision: 1 }, async () => output.file, async () => new Response('changed'), noChange), fixed('changed'));
  output.assertPreserved();
  assert.equal(await save(target, async () => output.file, async () => new Response('after failures'), noChange), output.file);
  assert.equal(readFileSync(output.file, 'utf8'), 'after failures');
});
