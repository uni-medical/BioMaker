import * as fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';

export interface DraftOptions {
  model_for_all_agents: string | null;
  agent_steps: number | null;
  exec_timeout: number | null;
  exec_parallel_workers: number | 'auto' | null;
  agent_total_time_limit: number | null;
  final_retrain_enabled: boolean;
}
export interface DraftPayload {
  task: string;
  dataset: string;
  name: string;
  options: DraftOptions;
  template: string | null;
  reusedFrom?: { runId: string; title: string; requiresDataset: boolean; usedDefaults: boolean };
}
export interface DraftRecord { id: string; title: string; createdAt: string; updatedAt: string; content: DraftPayload }
export interface DraftSnapshot {
  revision: string;
  activeId: string | null;
  drafts: DraftRecord[];
  legacyMigration: { sha256: string; draftId: string | null } | null;
}
export type DraftOperation =
  | { type: 'create'; content: DraftPayload; title?: string }
  | { type: 'update'; id: string; content: DraftPayload }
  | { type: 'rename'; id: string; title: string }
  | { type: 'delete'; id: string }
  | { type: 'activate'; id: string | null }
  | { type: 'migrate'; raw: string | null };
export interface DraftMutation { expectedRevision: string; mutationId: string; operation: DraftOperation }
export type DraftErrorCode = 'invalid_request' | 'conflict' | 'corrupt' | 'unsupported' | 'unavailable' | 'too_large' | 'save_failed';
export type DraftFlushResult = { ok: true; snapshot: DraftSnapshot } | { ok: false; code: DraftErrorCode };
const messages: Record<DraftErrorCode, string> = {
  invalid_request: 'The draft request is invalid. Your edits have not been discarded.',
  conflict: 'Saved drafts changed. Reload and review them before saving your edits.',
  corrupt: 'The saved draft file is damaged. It has not been replaced.',
  unsupported: 'The saved drafts use an unsupported format. They have not been replaced.',
  unavailable: 'Saved drafts could not be read. Your edits have not been discarded.',
  too_large: 'The draft storage limit was reached. No drafts have been removed.',
  save_failed: 'The draft save could not be confirmed. Reload before retrying with a new request.',
};
export class DraftError extends Error {
  constructor(readonly code: DraftErrorCode) { super(`[drafts:${code}] ${messages[code]}`); }
}
export function draftErrorCode(error: unknown): DraftErrorCode { return error instanceof DraftError ? error.code : 'unavailable'; }
export const MAX_DRAFTS = 100;
export const MAX_DRAFT_BYTES = 256 * 1024;
export const MAX_DRAFT_FILE_BYTES = 8 * 1024 * 1024;
export const DRAFT_FILENAME = 'experiment-drafts.json';
const MAX_MUTATIONS = 128;
const defaultOptions: DraftOptions = { model_for_all_agents: null, agent_steps: 35, exec_timeout: 600, exec_parallel_workers: 'auto', agent_total_time_limit: 86400, final_retrain_enabled: true };
interface DraftDocument {
  schema: 1;
  generation: string;
  activeId: string | null;
  drafts: DraftRecord[];
  legacyMigration: { sha256: string; draftId: string | null; raw: string | null } | null;
  mutations: { id: string; sha256: string }[];
}
const sha256 = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
const fail = (code: DraftErrorCode): never => { throw new DraftError(code); };
const hash = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const identifier = (value: unknown): value is string => typeof value === 'string' && /^[a-zA-Z0-9_-]{1,100}$/.test(value);
const nativeId = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/.test(value);
function object(value: unknown, allowed: string[], required = allowed): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return fail('invalid_request');
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) return fail('invalid_request');
  if (Object.keys(value).some(key => !allowed.includes(key)) || required.some(key => !Object.hasOwn(value, key))) return fail('invalid_request');
  return value as Record<string, unknown>;
}
function text(value: unknown): string { if (typeof value !== 'string') return fail('invalid_request'); return value; }
function title(value: unknown): string { const result = text(value); if (Array.from(result).length > 200) return fail('invalid_request'); return result; }
function id(value: unknown): string { if (!nativeId(value)) return fail('invalid_request'); return value; }
function number(value: unknown): number | null { if (value === null || typeof value === 'number' && Number.isFinite(value)) return value; return fail('invalid_request'); }
function boolean(value: unknown): boolean { if (typeof value !== 'boolean') return fail('invalid_request'); return value; }
function date(value: unknown): string { const result = text(value); if (!Number.isFinite(Date.parse(result)) || new Date(result).toISOString() !== result) return fail('invalid_request'); return result; }

/** Storage preserves incomplete numeric input. Execution validation belongs to submission. */
function payload(raw: unknown, legacy = false): DraftPayload {
  const value = object(raw, ['task', 'dataset', 'name', 'options', 'template', 'reusedFrom'], legacy ? [] : ['task', 'dataset', 'name', 'options', 'template']);
  const rawOptions = legacy && value.options === undefined ? {} : value.options;
  const options = object(rawOptions, Object.keys(defaultOptions), legacy ? [] : Object.keys(defaultOptions));
  const merged = { ...(legacy ? defaultOptions : {}), ...options };
  const content: DraftPayload = {
    task: text(legacy && value.task === undefined ? '' : value.task),
    dataset: text(legacy && value.dataset === undefined ? '' : value.dataset),
    name: text(legacy && value.name === undefined ? '' : value.name),
    template: value.template === null || legacy && value.template === undefined ? null : text(value.template),
    options: {
      model_for_all_agents: merged.model_for_all_agents === null ? null : text(merged.model_for_all_agents),
      agent_steps: number(merged.agent_steps), exec_timeout: number(merged.exec_timeout),
      exec_parallel_workers: merged.exec_parallel_workers === 'auto' ? 'auto' : number(merged.exec_parallel_workers),
      agent_total_time_limit: number(merged.agent_total_time_limit), final_retrain_enabled: boolean(merged.final_retrain_enabled),
    },
  };
  if (value.reusedFrom !== undefined) {
    const source = object(value.reusedFrom, ['runId', 'title', 'requiresDataset', 'usedDefaults']);
    content.reusedFrom = { runId: text(source.runId), title: text(source.title), requiresDataset: boolean(source.requiresDataset), usedDefaults: boolean(source.usedDefaults) };
  }
  if (Buffer.byteLength(JSON.stringify(content)) > MAX_DRAFT_BYTES) return fail('too_large');
  return content;
}
function legacyContent(raw: string | null): DraftPayload | null {
  if (raw === null) return null;
  if (Buffer.byteLength(raw) > MAX_DRAFT_FILE_BYTES) return fail('too_large');
  let parsed: unknown;
  try { parsed = JSON.parse(raw); } catch { return fail('invalid_request'); }
  return payload(parsed, true);
}
function mutation(raw: unknown): DraftMutation {
  const request = object(raw, ['expectedRevision', 'mutationId', 'operation']);
  if (request.expectedRevision !== 'missing' && !hash(request.expectedRevision) || !identifier(request.mutationId)) return fail('invalid_request');
  const rawOperation = object(request.operation, ['type', 'content', 'title', 'id', 'raw'], ['type']);
  let operation: DraftOperation;
  switch (rawOperation.type) {
    case 'create': {
      const value = object(rawOperation, ['type', 'content', 'title'], ['type', 'content']);
      operation = { type: 'create', content: payload(value.content), ...(value.title === undefined ? {} : { title: title(value.title) }) }; break;
    }
    case 'update': { const value = object(rawOperation, ['type', 'id', 'content']); operation = { type: 'update', id: id(value.id), content: payload(value.content) }; break; }
    case 'rename': { const value = object(rawOperation, ['type', 'id', 'title']); operation = { type: 'rename', id: id(value.id), title: title(value.title) }; break; }
    case 'delete': { const value = object(rawOperation, ['type', 'id']); operation = { type: 'delete', id: id(value.id) }; break; }
    case 'activate': { const value = object(rawOperation, ['type', 'id']); operation = { type: 'activate', id: value.id === null ? null : id(value.id) }; break; }
    case 'migrate': { const value = object(rawOperation, ['type', 'raw']); const original = value.raw === null ? null : text(value.raw); legacyContent(original); operation = { type: 'migrate', raw: original }; break; }
    default: return fail('invalid_request');
  }
  return { expectedRevision: request.expectedRevision as string, mutationId: request.mutationId, operation };
}
function parseDocument(raw: Buffer): DraftDocument {
  let parsed: unknown;
  try { parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(raw)); }
  catch { return fail('corrupt'); }
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed) && typeof (parsed as { schema?: unknown }).schema === 'number' && (parsed as { schema: number }).schema !== 1) return fail('unsupported');
  try {
    const value = object(parsed, ['schema', 'generation', 'activeId', 'drafts', 'legacyMigration', 'mutations']);
    if (value.schema !== 1 || !nativeId(value.generation) || !Array.isArray(value.drafts) || value.drafts.length > MAX_DRAFTS || !Array.isArray(value.mutations) || value.mutations.length > MAX_MUTATIONS) return fail('corrupt');
    const drafts = value.drafts.map(rawRecord => {
      const record = object(rawRecord, ['id', 'title', 'createdAt', 'updatedAt', 'content']);
      const result: DraftRecord = { id: id(record.id), title: title(record.title), createdAt: date(record.createdAt), updatedAt: date(record.updatedAt), content: payload(record.content) };
      if (Buffer.byteLength(JSON.stringify(result)) > MAX_DRAFT_BYTES) return fail('too_large');
      return result;
    });
    if (new Set(drafts.map(record => record.id)).size !== drafts.length) return fail('corrupt');
    const activeId = value.activeId === null ? null : id(value.activeId);
    if (activeId !== null && !drafts.some(record => record.id === activeId)) return fail('corrupt');
    let legacyMigration: DraftDocument['legacyMigration'] = null;
    if (value.legacyMigration !== null) {
      const original = object(value.legacyMigration, ['sha256', 'draftId', 'raw']);
      if (!hash(original.sha256) || original.raw !== null && typeof original.raw !== 'string' || sha256(original.raw ?? '') !== original.sha256) return fail('corrupt');
      if (original.raw === null && original.draftId !== null || original.raw !== null && original.draftId === null) return fail('corrupt');
      legacyContent(original.raw);
      legacyMigration = { sha256: original.sha256, draftId: original.draftId === null ? null : id(original.draftId), raw: original.raw };
    }
    const mutations = value.mutations.map(rawMutation => {
      const entry = object(rawMutation, ['id', 'sha256']);
      if (!identifier(entry.id) || !hash(entry.sha256)) return fail('corrupt');
      return { id: entry.id, sha256: entry.sha256 };
    });
    if (new Set(mutations.map(entry => entry.id)).size !== mutations.length) return fail('corrupt');
    return { schema: 1, generation: value.generation, activeId, drafts, legacyMigration, mutations };
  } catch (error) { return fail(error instanceof DraftError && error.code === 'too_large' ? 'too_large' : 'corrupt'); }
}
function snapshot(document: DraftDocument | null, revision: string): DraftSnapshot {
  return structuredClone({ revision, activeId: document?.activeId ?? null, drafts: document?.drafts ?? [], legacyMigration: document?.legacyMigration ? { sha256: document.legacyMigration.sha256, draftId: document.legacyMigration.draftId } : null });
}
const signature = (value: fs.Stats) => `${value.dev}:${value.ino}:${value.mode}:${value.nlink}:${value.size}:${value.mtimeMs}:${value.ctimeMs}`;
const directorySignature = (value: fs.Stats) => `${value.dev}:${value.ino}:${value.mode}`;
type DraftFileSystem = Pick<typeof fs, 'realpathSync' | 'lstatSync' | 'openSync' | 'fstatSync' | 'readSync' | 'writeSync' | 'fsyncSync' | 'closeSync' | 'renameSync' | 'unlinkSync'>;
interface Loaded { document: DraftDocument | null; revision: string; signature: string | null }

/** One instance per native profile. Every transaction is synchronous: regular
 * invoke and beforeunload sendSync share the same main-process serialization.
 * No sidecar, renderer-selected path, or settings fallback participates. */
export class DraftStore {
  private readonly io: DraftFileSystem;
  private readonly directory: string;
  private readonly filename: string;
  private readonly directories: { name: string; signature: string }[];
  private working = false;

  constructor(directory: string, options: { io?: DraftFileSystem } = {}) {
    this.io = options.io ?? fs;
    try {
      this.directory = this.io.realpathSync(directory);
      this.filename = path.join(this.directory, DRAFT_FILENAME);
      this.directories = [];
      let name = this.directory;
      while (true) {
        const stats = this.io.lstatSync(name);
        if (!stats.isDirectory() || stats.isSymbolicLink()) return fail('unavailable');
        this.directories.push({ name, signature: directorySignature(stats) });
        const parent = path.dirname(name); if (name === parent) break; name = parent;
      }
    } catch { throw new DraftError('unavailable'); }
  }
  getDrafts(): DraftSnapshot { return this.transaction(() => { const current = this.read(); return snapshot(current.document, current.revision); }); }
  mutateDrafts(input: unknown): DraftSnapshot { return this.transaction(() => this.change(mutation(input), false)); }
  flushDrafts(input: unknown): DraftFlushResult {
    try { return { ok: true, snapshot: this.transaction(() => this.change(mutation(input), true)) }; }
    catch (error) { return { ok: false, code: draftErrorCode(error) }; }
  }
  private transaction<T>(callback: () => T): T {
    if (this.working) return fail('conflict');
    this.working = true;
    try { return callback(); }
    catch (error) { if (error instanceof DraftError) throw error; return fail('unavailable'); }
    finally { this.working = false; }
  }
  private checkDirectory(): void {
    for (const expected of this.directories) {
      let actual: fs.Stats;
      try { actual = this.io.lstatSync(expected.name); } catch { return fail('unavailable'); }
      if (!actual.isDirectory() || actual.isSymbolicLink() || directorySignature(actual) !== expected.signature) return fail('unavailable');
    }
  }
  private read(): Loaded {
    this.checkDirectory();
    let before: fs.Stats;
    try { before = this.io.lstatSync(this.filename); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { document: null, revision: 'missing', signature: null };
      return fail('unavailable');
    }
    if (!before.isFile() || before.isSymbolicLink() || before.nlink !== 1) return fail('unavailable');
    if (before.size > MAX_DRAFT_FILE_BYTES) return fail('too_large');
    if (typeof fs.constants.O_NOFOLLOW !== 'number') return fail('unavailable');
    let descriptor: number | undefined;
    try {
      descriptor = this.io.openSync(this.filename, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK);
      const opened = this.io.fstatSync(descriptor);
      if (!opened.isFile() || signature(opened) !== signature(before)) return fail('conflict');
      const buffer = Buffer.alloc(before.size + 1);
      let offset = 0;
      while (offset < buffer.length) { const count = this.io.readSync(descriptor, buffer, offset, buffer.length - offset, null); if (count === 0) break; offset += count; }
      if (offset !== before.size || signature(this.io.fstatSync(descriptor)) !== signature(before) || signature(this.io.lstatSync(this.filename)) !== signature(before)) return fail('conflict');
      this.checkDirectory();
      const bytes = buffer.subarray(0, offset);
      return { document: parseDocument(bytes), revision: sha256(bytes), signature: signature(before) };
    } catch (error) { if (error instanceof DraftError) throw error; return fail('unavailable'); }
    finally { if (descriptor !== undefined) try { this.io.closeSync(descriptor); } catch { /* A failed read never writes. */ } }
  }
  private change(request: DraftMutation, flush: boolean): DraftSnapshot {
    const current = this.read();
    const requestHash = sha256(JSON.stringify({ expectedRevision: request.expectedRevision, operation: request.operation }));
    const previous = current.document?.mutations.find(entry => entry.id === request.mutationId);
    if (previous) {
      if (previous.sha256 !== requestHash) return fail('invalid_request');
      return snapshot(current.document, current.revision);
    }
    if (flush && request.operation.type !== 'update') return fail('invalid_request');
    if (request.expectedRevision !== current.revision) return fail('conflict');
    const document: DraftDocument = current.document ?? { schema: 1, generation: randomUUID(), activeId: null, drafts: [], legacyMigration: null, mutations: [] };
    const operation = request.operation;
    const now = new Date().toISOString();
    const find = (wanted: string): DraftRecord => document.drafts.find(record => record.id === wanted) ?? fail('conflict');
    const create = (content: DraftPayload, name = '') => {
      if (document.drafts.length >= MAX_DRAFTS) return fail('too_large');
      const record: DraftRecord = { id: randomUUID(), title: name, createdAt: now, updatedAt: now, content };
      document.drafts.push(record); document.activeId = record.id; return record.id;
    };
    switch (operation.type) {
      case 'create': create(operation.content, operation.title); break;
      case 'update': { const record = find(operation.id); record.content = operation.content; record.updatedAt = now; break; }
      case 'rename': { const record = find(operation.id); record.title = operation.title; record.updatedAt = now; break; }
      case 'delete': find(operation.id); document.drafts = document.drafts.filter(record => record.id !== operation.id); if (document.activeId === operation.id) document.activeId = null; break;
      case 'activate': if (operation.id !== null) find(operation.id); document.activeId = operation.id; break;
      case 'migrate': {
        const digest = sha256(operation.raw ?? '');
        if (document.legacyMigration) {
          if (document.legacyMigration.sha256 !== digest || document.legacyMigration.raw !== operation.raw) return fail('conflict');
          // Never re-create a legacy draft that the user has since deleted.
          break;
        }
        const content = legacyContent(operation.raw);
        document.legacyMigration = { raw: operation.raw, sha256: digest, draftId: content === null ? null : create(content) }; break;
      }
    }
    document.generation = randomUUID();
    document.mutations = [...document.mutations, { id: request.mutationId, sha256: requestHash }].slice(-MAX_MUTATIONS);
    for (const record of document.drafts) if (Buffer.byteLength(JSON.stringify(record)) > MAX_DRAFT_BYTES) return fail('too_large');
    const bytes = Buffer.from(JSON.stringify(document));
    if (bytes.length > MAX_DRAFT_FILE_BYTES) return fail('too_large');
    this.commit(bytes, current);
    return snapshot(document, sha256(bytes));
  }
  private commit(bytes: Buffer, expected: Loaded): void {
    let descriptor: number | undefined;
    let directoryDescriptor: number | undefined;
    const temporary = path.join(this.directory, `.experiment-drafts-${randomUUID()}.tmp`);
    let temporaryCreated = false;
    let temporaryOwner: { dev: number; ino: number } | undefined;
    try {
      this.checkDirectory();
      descriptor = this.io.openSync(temporary, fs.constants.O_WRONLY | fs.constants.O_CREAT | fs.constants.O_EXCL | fs.constants.O_NOFOLLOW, 0o600);
      temporaryCreated = true;
      const created = this.io.fstatSync(descriptor);
      if (!created.isFile() || created.nlink !== 1) return fail('save_failed');
      temporaryOwner = { dev: created.dev, ino: created.ino };
      let offset = 0;
      while (offset < bytes.length) { const written = this.io.writeSync(descriptor, bytes, offset, bytes.length - offset, null); if (written <= 0) return fail('save_failed'); offset += written; }
      this.io.fsyncSync(descriptor);
      const prepared = this.io.fstatSync(descriptor);
      if (!prepared.isFile() || prepared.nlink !== 1) return fail('save_failed');
      const tempIdentity = signature(prepared);
      this.io.closeSync(descriptor); descriptor = undefined;
      const actual = this.read();
      if (actual.revision !== expected.revision || actual.signature !== expected.signature) return fail('conflict');
      this.checkDirectory();
      if (signature(this.io.lstatSync(temporary)) !== tempIdentity) return fail('conflict');
      directoryDescriptor = this.io.openSync(this.directory, fs.constants.O_RDONLY | fs.constants.O_DIRECTORY | fs.constants.O_NOFOLLOW);
      if (directorySignature(this.io.fstatSync(directoryDescriptor)) !== this.directories[0].signature) return fail('conflict');
      this.io.renameSync(temporary, this.filename); temporaryCreated = false;
      // If durability confirmation fails after rename, the mutation receipt is
      // already part of the file; retrying the identical request never replays it.
      this.io.fsyncSync(directoryDescriptor);
      this.checkDirectory();
    } catch (error) { if (error instanceof DraftError) throw error; return fail('save_failed'); }
    finally {
      for (const open of [descriptor, directoryDescriptor]) if (open !== undefined) try { this.io.closeSync(open); } catch { /* Preserve the original failure. */ }
      if (temporaryCreated && temporaryOwner) try {
        this.checkDirectory();
        const remaining = this.io.lstatSync(temporary);
        if (remaining.dev === temporaryOwner.dev && remaining.ino === temporaryOwner.ino) this.io.unlinkSync(temporary);
      } catch { /* Do not follow a changed directory or remove a replacement temporary file. */ }
    }
  }
}
