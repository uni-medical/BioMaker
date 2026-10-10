import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { DraftStore, DraftError, DRAFT_FILENAME, MAX_DRAFT_BYTES, MAX_DRAFT_FILE_BYTES, type DraftMutation, type DraftPayload, type DraftSnapshot } from './drafts';

const content = (task = '  Task\r\n中文 🧬\n'): DraftPayload => ({ task, dataset: '/private/data with spaces', name: '', template: null, options: { model_for_all_agents: null, agent_steps: 35, exec_timeout: 600, exec_parallel_workers: 'auto', agent_total_time_limit: 86400, final_retrain_enabled: true } });
const request = (revision: string, operation: DraftMutation['operation'], mutationId = randomUUID()): DraftMutation => ({ expectedRevision: revision, mutationId, operation });
function fixture(t: { after: (fn: () => void) => void }, io?: ConstructorParameters<typeof DraftStore>[1]) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'biomaker-draft-store-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return { directory, filename: path.join(directory, DRAFT_FILENAME), store: new DraftStore(directory, io) };
}
function error(code: string) { return (value: unknown) => value instanceof DraftError && value.code === code && value.message.startsWith(`[drafts:${code}]`) && !value.message.includes('/private/'); }
function create(store: DraftStore, revision = 'missing', task?: string): DraftSnapshot { return store.mutateDrafts(request(revision, { type: 'create', content: content(task) })); }
function files(directory: string) { return fs.readdirSync(directory).sort(); }

test('missing reads are side-effect free and creating returns detached native identities', t => {
  const f = fixture(t);
  assert.deepEqual(f.store.getDrafts(), { revision: 'missing', activeId: null, drafts: [], legacyMigration: null });
  assert.deepEqual(files(f.directory), []);
  const saved = create(f.store);
  assert.match(saved.revision, /^[a-f0-9]{64}$/);
  assert.match(saved.activeId!, /^[a-f0-9-]{36}$/);
  assert.equal(saved.drafts[0].id, saved.activeId);
  assert.equal(new Date(saved.drafts[0].createdAt).toISOString(), saved.drafts[0].createdAt);
  assert.deepEqual(saved.drafts[0].content, content());
  assert.equal(fs.statSync(f.filename).mode & 0o777, 0o600);
  saved.drafts[0].content.task = 'mutated renderer snapshot';
  assert.deepEqual(new DraftStore(f.directory).getDrafts().drafts[0].content, content());
  assert.equal(f.store.getDrafts().revision, createHash('sha256').update(fs.readFileSync(f.filename)).digest('hex'));
});

test('create update rename activate and delete preserve independent drafts and immutable createdAt', t => {
  const { store } = fixture(t);
  const first = create(store, 'missing', 'one'); const firstId = first.activeId!;
  const second = create(store, first.revision, 'two'); const secondId = second.activeId!;
  const revised = store.mutateDrafts(request(second.revision, { type: 'update', id: firstId, content: content('changed one') }));
  assert.equal(revised.activeId, secondId); assert.equal(revised.drafts[0].createdAt, first.drafts[0].createdAt);
  assert.equal(revised.drafts[1].content.task, 'two');
  const named = store.mutateDrafts(request(revised.revision, { type: 'rename', id: firstId, title: '  标題 🧬 ' }));
  assert.equal(named.drafts[0].title, '  标題 🧬 ');
  const active = store.mutateDrafts(request(named.revision, { type: 'activate', id: firstId }));
  const removed = store.mutateDrafts(request(active.revision, { type: 'delete', id: firstId }));
  assert.equal(removed.activeId, null); assert.equal(removed.drafts.length, 1); assert.equal(removed.drafts[0].id, secondId);
  const clear = store.mutateDrafts(request(removed.revision, { type: 'activate', id: null }));
  assert.equal(clear.activeId, null);
  assert.throws(() => store.mutateDrafts(request(clear.revision, { type: 'update', id: firstId, content: content() })), error('conflict'));
});

test('finite partial numeric inputs and all existing fields survive exact round trips', t => {
  const { store } = fixture(t);
  const draft = content('\uFEFF\r\n\u0000 and \\ literals');
  draft.template = '/templates/一.yaml';
  draft.reusedFrom = { runId: 'existing-run', title: 'historical title', requiresDataset: true, usedDefaults: false };
  draft.options = { model_for_all_agents: '  provider/model  ', agent_steps: null, exec_timeout: -2.5, exec_parallel_workers: null, agent_total_time_limit: 0.25, final_retrain_enabled: false };
  const saved = store.mutateDrafts(request('missing', { type: 'create', content: draft }));
  assert.deepEqual(saved.drafts[0].content, draft);
  assert.deepEqual(store.getDrafts().drafts[0].content, draft);
});

test('stale independent callers and late autosaves cannot overwrite a newer snapshot', t => {
  const f = fixture(t); const first = create(f.store);
  const secondStore = new DraftStore(f.directory);
  const second = f.store.mutateDrafts(request(first.revision, { type: 'update', id: first.activeId!, content: content('newer') }));
  assert.throws(() => secondStore.mutateDrafts(request(first.revision, { type: 'update', id: first.activeId!, content: content('late') })), error('conflict'));
  assert.equal(f.store.getDrafts().revision, second.revision);
  assert.equal(f.store.getDrafts().drafts[0].content.task, 'newer');
});

test('successful mutation retries never replay and the same ID with changed payload is refused', t => {
  const f = fixture(t); const initial = request('missing', { type: 'create', content: content('original') });
  const first = f.store.mutateDrafts(initial);
  const next = f.store.mutateDrafts(request(first.revision, { type: 'rename', id: first.activeId!, title: 'new title' }));
  assert.deepEqual(f.store.mutateDrafts(initial), next);
  assert.deepEqual(new DraftStore(f.directory).mutateDrafts(initial), next);
  assert.throws(() => f.store.mutateDrafts({ ...initial, operation: { type: 'create', content: content('different') } }), error('invalid_request'));
  assert.throws(() => f.store.mutateDrafts({ ...initial, expectedRevision: next.revision }), error('invalid_request'));
  assert.equal(f.store.getDrafts().drafts.length, 1);
});

test('synchronous beforeunload flush updates through the same CAS and supports only committed non-update retries', t => {
  const { store } = fixture(t); const initial = request('missing', { type: 'create', content: content() });
  assert.deepEqual(store.flushDrafts(initial), { ok: false, code: 'invalid_request' });
  const saved = store.mutateDrafts(initial);
  assert.deepEqual(store.flushDrafts(initial), { ok: true, snapshot: saved });
  const lastKeystroke = request(saved.revision, { type: 'update', id: saved.activeId!, content: content('last keystroke') });
  const flushed = store.flushDrafts(lastKeystroke); assert.equal(flushed.ok, true);
  assert.equal(store.getDrafts().drafts[0].content.task, 'last keystroke');
  assert.deepEqual(store.flushDrafts(request(saved.revision, { type: 'update', id: saved.activeId!, content: content('stale') })), { ok: false, code: 'conflict' });
  assert.deepEqual(store.flushDrafts(request(store.getDrafts().revision, { type: 'delete', id: saved.activeId! })), { ok: false, code: 'invalid_request' });
  assert.deepEqual(store.flushDrafts(null), { ok: false, code: 'invalid_request' });
});

test('legacy migration preserves exact raw text privately and fills only missing fields', t => {
  const f = fixture(t);
  const raw = ' { "task" : "  old\\r\\n任务 ", "options": { "agent_steps": null, "exec_timeout": -1.5 } }\n';
  const migration = request('missing', { type: 'migrate', raw });
  const saved = f.store.mutateDrafts(migration);
  assert.equal(saved.drafts.length, 1); assert.equal(saved.drafts[0].content.task, '  old\r\n任务 ');
  assert.deepEqual(saved.drafts[0].content.options, { ...content().options, agent_steps: null, exec_timeout: -1.5 });
  assert.deepEqual(saved.legacyMigration, { sha256: createHash('sha256').update(raw).digest('hex'), draftId: saved.activeId });
  assert.equal(JSON.parse(fs.readFileSync(f.filename, 'utf8')).legacyMigration.raw, raw);
  assert(!Object.hasOwn(saved.legacyMigration!, 'raw'));
  assert.deepEqual(f.store.mutateDrafts(migration), saved);
  const later = f.store.mutateDrafts(request(saved.revision, { type: 'migrate', raw }));
  assert.equal(later.drafts.length, 1);
  const removed = f.store.mutateDrafts(request(later.revision, { type: 'delete', id: later.activeId! }));
  assert.equal(f.store.mutateDrafts(request(removed.revision, { type: 'migrate', raw })).drafts.length, 0);
});

test('legacy migration appends to existing native drafts without replacing them', t => {
  const { store } = fixture(t); const existing = create(store, 'missing', 'native');
  const saved = store.mutateDrafts(request(existing.revision, { type: 'migrate', raw: JSON.stringify({ task: 'legacy' }) }));
  assert.deepEqual(saved.drafts.map(record => record.content.task), ['native', 'legacy']);
  assert.equal(saved.activeId, saved.drafts[1].id);
  assert.throws(() => store.mutateDrafts(request(saved.revision, { type: 'migrate', raw: JSON.stringify({ task: 'a different legacy value' }) })), error('conflict'));
});

test('null legacy initializes explicitly with no artificial draft and records migration once', t => {
  const f = fixture(t); const saved = f.store.mutateDrafts(request('missing', { type: 'migrate', raw: null }));
  assert.equal(saved.drafts.length, 0); assert.equal(saved.activeId, null);
  assert.deepEqual(saved.legacyMigration, { sha256: createHash('sha256').update('').digest('hex'), draftId: null });
  assert.deepEqual(new DraftStore(f.directory).getDrafts(), saved);
  assert.throws(() => f.store.mutateDrafts(request(saved.revision, { type: 'migrate', raw: '{}' })), error('conflict'));
});

for (const raw of ['{broken', 'null', '[]', '"text"', '{"unknown":"do not drop"}', '{"task":null}', '{"options":{"unknown":1}}', '{"options":{"agent_steps":"35"}}']) {
  test(`unrecognizable legacy data is refused without creating a native file (${raw})`, t => {
    const f = fixture(t);
    assert.throws(() => f.store.mutateDrafts(request('missing', { type: 'migrate', raw })), error('invalid_request'));
    assert.deepEqual(files(f.directory), []);
  });
}

test('request shapes are strict and credentials or renderer file paths are never accepted as extra fields', t => {
  const f = fixture(t);
  const original = request('missing', { type: 'create', content: content() });
  const cases: unknown[] = [null, [], { ...original, filename: '/private/secret' }, { ...original, operation: { ...original.operation, id: randomUUID() } }, { ...original, operation: { type: 'create', content: { ...content(), apiKey: 'secret' } } }, { ...original, operation: { type: 'create', content: { ...content(), options: { ...content().options, model_for_all_agents: 3 } } } }, { ...original, operation: { type: 'create', content: { ...content(), options: { ...content().options, agent_steps: NaN } } } }, { ...original, operation: { type: 'create', content: { ...content(), options: { ...content().options, agent_steps: Infinity } } } }, { ...original, operation: { type: 'create', title: 'x'.repeat(201), content: content() } }, { ...original, mutationId: '' }];
  for (const candidate of cases) assert.throws(() => f.store.mutateDrafts(candidate), error('invalid_request'));
  assert.deepEqual(files(f.directory), []);
});

for (const [name, bytes, code] of [['broken JSON', Buffer.from('{not-json'), 'corrupt'], ['invalid UTF8', Buffer.from([0xff, 0xfe]), 'corrupt'], ['future format', Buffer.from('{"schema":2,"private":"preserve"}'), 'unsupported'], ['empty file', Buffer.alloc(0), 'corrupt']] as const) {
  test(`${name} is never treated as an empty draft library`, t => {
    const f = fixture(t); fs.writeFileSync(f.filename, bytes);
    assert.throws(() => f.store.getDrafts(), error(code));
    assert.throws(() => create(f.store), error(code));
    assert.deepEqual(fs.readFileSync(f.filename), bytes); assert.deepEqual(files(f.directory), [DRAFT_FILENAME]);
  });
}

test('strict on-disk validation rejects duplicate IDs, dangling selection and changed migration records', t => {
  const f = fixture(t); create(f.store);
  const valid = JSON.parse(fs.readFileSync(f.filename, 'utf8'));
  const variants = [
    { ...valid, drafts: [valid.drafts[0], valid.drafts[0]] },
    { ...valid, activeId: randomUUID() },
    { ...valid, extra: 'do not silently discard' },
    { ...valid, drafts: [{ ...valid.drafts[0], createdAt: 'yesterday' }] },
    { ...valid, legacyMigration: { raw: '{}', sha256: '0'.repeat(64), draftId: valid.drafts[0].id } },
  ];
  for (const malformed of variants) {
    const bytes = JSON.stringify(malformed); fs.writeFileSync(f.filename, bytes);
    assert.throws(() => f.store.getDrafts(), error('corrupt'));
    assert.throws(() => create(f.store), error('corrupt'));
    assert.equal(fs.readFileSync(f.filename, 'utf8'), bytes);
  }
  fs.writeFileSync(f.filename, JSON.stringify(valid));
  assert.equal(f.store.getDrafts().drafts.length, 1, 'Explicitly repairing the file allows a read; no auto reset occurred');
});

test('draft quota refuses oversized content and total-file overflow without truncation', t => {
  const f = fixture(t);
  assert.throws(() => create(f.store, 'missing', 'x'.repeat(MAX_DRAFT_BYTES)), error('too_large'));
  let saved = create(f.store, 'missing', 'x'.repeat(230 * 1024));
  while (fs.statSync(f.filename).size + 240 * 1024 < MAX_DRAFT_FILE_BYTES) saved = create(f.store, saved.revision, 'x'.repeat(230 * 1024));
  const before = fs.readFileSync(f.filename);
  assert.throws(() => create(f.store, saved.revision, 'x'.repeat(255 * 1024)), error('too_large'));
  assert.deepEqual(fs.readFileSync(f.filename), before);
  const reduced = f.store.mutateDrafts(request(saved.revision, { type: 'delete', id: saved.drafts[0].id }));
  assert.equal(create(f.store, reduced.revision, 'small replacement').drafts.length, saved.drafts.length);
});

test('100-draft quota requires explicit deletion and never evicts older drafts', t => {
  const f = fixture(t); let saved = f.store.getDrafts();
  for (let index = 0; index < 100; index++) saved = create(f.store, saved.revision, `draft ${index}`);
  const before = fs.readFileSync(f.filename);
  assert.throws(() => create(f.store, saved.revision, '101'), error('too_large'));
  assert.deepEqual(fs.readFileSync(f.filename), before);
  assert.equal(saved.drafts[0].content.task, 'draft 0');
});

test('oversized existing files and nonregular paths are refused without modification', t => {
  const f = fixture(t); fs.writeFileSync(f.filename, 'x'.repeat(MAX_DRAFT_FILE_BYTES + 1));
  assert.throws(() => f.store.getDrafts(), error('too_large'));
  assert.throws(() => create(f.store), error('too_large'));
  fs.unlinkSync(f.filename); fs.mkdirSync(f.filename);
  assert.throws(() => f.store.getDrafts(), error('unavailable'));
  assert.throws(() => create(f.store), error('unavailable'));
  assert(fs.statSync(f.filename).isDirectory());
});

test('symbolic links and hard links cannot redirect draft reads or writes', t => {
  const f = fixture(t); const outside = path.join(f.directory, 'private-original'); fs.writeFileSync(outside, 'private contents');
  fs.symlinkSync(outside, f.filename);
  assert.throws(() => f.store.getDrafts(), error('unavailable'));
  assert.throws(() => create(f.store), error('unavailable'));
  assert.equal(fs.readFileSync(outside, 'utf8'), 'private contents');
  fs.unlinkSync(f.filename); fs.linkSync(outside, f.filename);
  assert.throws(() => f.store.getDrafts(), error('unavailable'));
  assert.throws(() => create(f.store), error('unavailable'));
  assert.equal(fs.readFileSync(outside, 'utf8'), 'private contents');
});

test('replacing the registered profile directory is not silently followed', t => {
  const f = fixture(t); const moved = `${f.directory}-original`; t.after(() => fs.rmSync(moved, { recursive: true, force: true }));
  fs.renameSync(f.directory, moved); fs.mkdirSync(f.directory);
  assert.throws(() => f.store.getDrafts(), error('unavailable'));
  assert.throws(() => create(f.store), error('unavailable'));
  assert.deepEqual(files(f.directory), []);
});

for (const stage of ['write', 'fsync', 'rename'] as const) {
  test(`failed ${stage} preserves the previous file and removes only its temporary file`, t => {
    const f = fixture(t); const saved = create(f.store); const before = fs.readFileSync(f.filename);
    const injected = new DraftStore(f.directory, { io: { ...fs,
      ...(stage === 'write' ? { writeSync: () => { throw new Error('private path /private/secret write failed'); } } : {}),
      ...(stage === 'fsync' ? { fsyncSync: () => { throw new Error('private path /private/secret fsync failed'); } } : {}),
      ...(stage === 'rename' ? { renameSync: () => { throw new Error('private path /private/secret rename failed'); } } : {}),
    } });
    assert.throws(() => injected.mutateDrafts(request(saved.revision, { type: 'update', id: saved.activeId!, content: content('not committed') })), error('save_failed'));
    assert.deepEqual(fs.readFileSync(f.filename), before); assert.deepEqual(files(f.directory), [DRAFT_FILENAME]);
  });
}

test('a write failure during the first transaction does not initialize an empty library', t => {
  const f = fixture(t); const injected = new DraftStore(f.directory, { io: { ...fs, writeSync: () => { throw new Error('disk full'); } } });
  assert.throws(() => create(injected), error('save_failed'));
  assert.deepEqual(files(f.directory), []); assert.equal(f.store.getDrafts().revision, 'missing');
});

test('external replacement during file preparation fails CAS and preserves the replacement', t => {
  const f = fixture(t); const saved = create(f.store); let external = ''; let changed = false;
  const injected = new DraftStore(f.directory, { io: { ...fs, fsyncSync: descriptor => {
    fs.fsyncSync(descriptor);
    if (!changed) { changed = true; const document = JSON.parse(fs.readFileSync(f.filename, 'utf8')); document.drafts[0].content.task = 'external newer'; external = JSON.stringify(document); fs.writeFileSync(f.filename, external); }
  } } });
  assert.throws(() => injected.mutateDrafts(request(saved.revision, { type: 'update', id: saved.activeId!, content: content('late') })), error('conflict'));
  assert.equal(fs.readFileSync(f.filename, 'utf8'), external); assert.deepEqual(files(f.directory), [DRAFT_FILENAME]);
});

test('post-rename confirmation failure is recoverable with the identical request without duplicate creation', t => {
  const f = fixture(t); const initial = request('missing', { type: 'create', content: content('exactly once') }); let syncs = 0;
  const injected = new DraftStore(f.directory, { io: { ...fs, fsyncSync: descriptor => { if (++syncs === 2) throw new Error('directory sync failed'); fs.fsyncSync(descriptor); } } });
  assert.throws(() => injected.mutateDrafts(initial), error('save_failed'));
  assert.equal(f.store.getDrafts().drafts.length, 1);
  assert.deepEqual(injected.mutateDrafts(initial), f.store.getDrafts());
  assert.equal(f.store.getDrafts().drafts[0].content.task, 'exactly once');
  assert.deepEqual(files(f.directory), [DRAFT_FILENAME]);
});

test('a replaced temporary file is rejected and cleanup does not delete someone else\'s file', t => {
  const f = fixture(t); const saved = create(f.store); const before = fs.readFileSync(f.filename);
  let temporary = ''; let changed = false;
  const injected = new DraftStore(f.directory, { io: { ...fs, closeSync: descriptor => {
    fs.closeSync(descriptor);
    if (!changed) {
      const name = files(f.directory).find(name => name.startsWith('.experiment-drafts-'));
      if (name) { changed = true; temporary = path.join(f.directory, name); fs.renameSync(temporary, `${temporary}.original`); fs.writeFileSync(temporary, 'other contents'); }
    }
  } } });
  assert.throws(() => injected.mutateDrafts(request(saved.revision, { type: 'update', id: saved.activeId!, content: content('rejected') })), error('conflict'));
  assert.deepEqual(fs.readFileSync(f.filename), before);
  assert.equal(fs.readFileSync(temporary, 'utf8'), 'other contents');
});

test('reentrant access cannot interleave with a synchronous transaction', t => {
  const f = fixture(t); let store: DraftStore; let attempted = false;
  store = new DraftStore(f.directory, { io: { ...fs, writeSync: ((...args: Parameters<typeof fs.writeSync>) => {
    if (!attempted) { attempted = true; assert.throws(() => store.getDrafts(), error('conflict')); }
    return (fs.writeSync as (...parameters: Parameters<typeof fs.writeSync>) => number)(...args);
  }) as typeof fs.writeSync } });
  assert.equal(create(store).drafts.length, 1); assert(attempted);
});

test('expired idempotency records still cannot replay a stale CAS request', t => {
  const f = fixture(t); const initial = request('missing', { type: 'create', content: content() }); let saved = f.store.mutateDrafts(initial);
  for (let index = 0; index < 130; index++) saved = f.store.mutateDrafts(request(saved.revision, { type: 'update', id: saved.activeId!, content: content(`autosave ${index}`) }));
  assert.equal(JSON.parse(fs.readFileSync(f.filename, 'utf8')).mutations.length, 128);
  assert.throws(() => f.store.mutateDrafts(initial), error('conflict'));
  assert.equal(f.store.getDrafts().drafts[0].content.task, 'autosave 129');
});
