import { createHash, randomUUID } from 'node:crypto';
import { createWriteStream, renameSync, rmSync } from 'node:fs';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { validateArtifact } from './security';

export const MAX_PREVIEW_BYTES = 500_000;
// JSON may escape each source byte as six ASCII bytes (for example, NUL).
export const MAX_PREVIEW_RESPONSE_BYTES = MAX_PREVIEW_BYTES * 6 + 4096;
export const MAX_CONSOLE_DOWNLOAD_BYTES = 256 * 1024 * 1024;
const messages = {
  invalid_request: 'The artifact request is invalid.',
  unavailable: 'The artifact is unavailable. Refresh the experiment and try again.',
  changed: 'The artifact changed. Preview it again before saving.',
  too_large: 'The artifact is too large to preview. Save it from the file list.',
  invalid_text: 'The artifact is not valid UTF-8 text. Save it from the file list.',
  service_changed: 'The local service changed. Preview the artifact again before saving.',
  save_failed: 'The file could not be saved. Choose a writable location and try again.',
  busy: 'Another artifact is being saved. Wait for it to finish, then try again.',
} as const;
export type ArtifactCode = keyof typeof messages;
export class ArtifactError extends Error {
  constructor(readonly code: ArtifactCode) { super(`[artifact:${code}] ${messages[code]}`); }
}
export interface ArtifactTarget { runId: string; name: string }
export interface ArtifactSave extends ArtifactTarget { expectedSha256?: string; serviceRevision?: number }
export interface ArtifactPreview extends ArtifactTarget { content: string; sha256: string; size: number; serviceRevision: number }
export interface ConsoleSave { runId: string; serviceRevision: number }
interface ServiceState { status: string; revision: number }

/** A console export resolves the registered run in the service. Renderer input
 * cannot choose a source path, filename, content, or a different download URL. */
export function consoleRequest(raw: unknown): ConsoleSave {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new ArtifactError('invalid_request');
  const value = raw as Record<string, unknown>;
  if (Object.keys(value).some(key => key !== 'runId' && key !== 'serviceRevision') ||
      !Object.hasOwn(value, 'runId') || !Object.hasOwn(value, 'serviceRevision') ||
      typeof value.serviceRevision !== 'number' || !Number.isSafeInteger(value.serviceRevision) || value.serviceRevision < 0) throw new ArtifactError('invalid_request');
  try { validateArtifact(value.runId, 'console.log'); }
  catch { throw new ArtifactError('invalid_request'); }
  return { runId: value.runId as string, serviceRevision: value.serviceRevision };
}

export function artifactRequest(raw: unknown, saving = false): ArtifactSave {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new ArtifactError('invalid_request');
  const value = raw as Record<string, unknown>;
  const allowed = saving ? ['runId', 'name', 'expectedSha256', 'serviceRevision'] : ['runId', 'name'];
  if (Object.keys(value).some(key => !allowed.includes(key))) throw new ArtifactError('invalid_request');
  let target: ArtifactTarget;
  try { target = validateArtifact(value.runId, value.name); }
  catch { throw new ArtifactError('invalid_request'); }
  const hashPresent = Object.hasOwn(value, 'expectedSha256');
  const revisionPresent = Object.hasOwn(value, 'serviceRevision');
  if (hashPresent !== revisionPresent) throw new ArtifactError('invalid_request');
  if (!hashPresent) return target;
  if (typeof value.expectedSha256 !== 'string' || !/^[a-f0-9]{64}$/.test(value.expectedSha256) ||
      typeof value.serviceRevision !== 'number' || !Number.isSafeInteger(value.serviceRevision) || value.serviceRevision < 0) throw new ArtifactError('invalid_request');
  return { ...target, expectedSha256: value.expectedSha256, serviceRevision: value.serviceRevision };
}

/** Capture both the service object and its monotonically increasing state
 * revision. Reconnection can reuse a Sidecar object, so identity alone is not
 * enough. Checks surround every await and the final atomic file publication. */
export function artifactService<T extends object>(getService: () => T | null | undefined, getState: () => ServiceState, expectedRevision?: number) {
  const service = getService();
  const state = getState();
  if (expectedRevision !== undefined && state.revision !== expectedRevision) throw new ArtifactError('service_changed');
  if (!service || state.status !== 'ready') throw new ArtifactError('unavailable');
  const revision = state.revision;
  const assertCurrent = () => {
    const current = getState();
    if (getService() !== service || current.status !== 'ready' || current.revision !== revision) throw new ArtifactError('service_changed');
  };
  return { service, revision, assertCurrent };
}

async function boundedBody(response: Response, limit: number): Promise<Buffer> {
  const declared = Number(response.headers.get('content-length') ?? '0');
  if (declared > limit) { await response.body?.cancel().catch(() => {}); throw new ArtifactError('too_large'); }
  const reader = response.body?.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  if (!reader) throw new ArtifactError('unavailable');
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) throw new ArtifactError('too_large');
      chunks.push(value);
    }
    return Buffer.concat(chunks, size);
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally { reader.releaseLock(); }
}

export async function previewArtifact(target: ArtifactTarget, serviceRevision: number, send: () => Promise<Response>, assertCurrent: () => void): Promise<ArtifactPreview> {
  let response: Response | undefined;
  try {
    assertCurrent();
    response = await send();
    assertCurrent();
    const raw = await boundedBody(response, response.ok ? MAX_PREVIEW_RESPONSE_BYTES : 16_384);
    assertCurrent();
    const value: unknown = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(raw));
    if (!response.ok) {
      const code = value && typeof value === 'object' && !Array.isArray(value) ? (value as { code?: unknown }).code : null;
      throw new ArtifactError(typeof code === 'string' && Object.hasOwn(messages, code) ? code as ArtifactCode : 'unavailable');
    }
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new ArtifactError('unavailable');
    const record = value as Record<string, unknown>;
    if (record.name !== target.name || typeof record.content !== 'string' || typeof record.sha256 !== 'string' ||
        !/^[a-f0-9]{64}$/.test(record.sha256) || typeof record.size !== 'number' || !Number.isSafeInteger(record.size) || record.size < 0) throw new ArtifactError('unavailable');
    if (record.size > MAX_PREVIEW_BYTES || record.content.length > MAX_PREVIEW_BYTES) throw new ArtifactError('too_large');
    const bytes = Buffer.from(record.content, 'utf8');
    if (bytes.length > MAX_PREVIEW_BYTES) throw new ArtifactError('too_large');
    // Lone JSON surrogates must not be silently replaced when reconstructing
    // the original UTF-8 bytes. BOM and CRLF remain part of the hashed content.
    if (bytes.toString('utf8') !== record.content) throw new ArtifactError('invalid_text');
    if (record.size !== bytes.length || createHash('sha256').update(bytes).digest('hex') !== record.sha256) throw new ArtifactError('unavailable');
    return { ...target, content: record.content, sha256: record.sha256, size: record.size, serviceRevision };
  } catch (error) {
    await response?.body?.cancel().catch(() => {});
    assertCurrent();
    throw error instanceof ArtifactError ? error : new ArtifactError('unavailable');
  }
}

/** A preview save never writes text supplied by the renderer. The authenticated
 * original artifact is streamed to a private temporary file, checked against
 * the displayed snapshot, and only then replaces the chosen destination. */
export async function saveArtifact(target: ArtifactSave, chooseDestination: () => Promise<string | null>, send: () => Promise<Response>, assertCurrent: () => void, source: 'artifact' | 'console' = 'artifact'): Promise<string | null> {
  let temporary: string | undefined;
  let response: Response | undefined;
  let writing = false;
  try {
    assertCurrent();
    const destination = await chooseDestination();
    if (!destination) return null;
    assertCurrent();
    response = await send();
    assertCurrent();
    if (!response.ok) {
      // The registered console route has fixed status codes. Never return its
      // body, which might contain a source path or raw storage error.
      if (source === 'console' && response.status === 413) throw new ArtifactError('too_large');
      if (source === 'console' && response.status === 409) throw new ArtifactError('changed');
      throw new ArtifactError('unavailable');
    }
    if (!response.body) throw new ArtifactError('unavailable');
    let expectedSize: number | undefined;
    let expectedHash = target.expectedSha256;
    if (source === 'console') {
      const length = response.headers.get('content-length');
      const digest = response.headers.get('x-biomaker-console-sha256');
      const encoding = response.headers.get('content-encoding');
      if (response.status !== 200 || !length || !/^(0|[1-9][0-9]*)$/.test(length) ||
          !Number.isSafeInteger(Number(length)) || !digest || !/^[a-f0-9]{64}$/.test(digest) ||
          response.headers.get('content-type')?.split(';')[0].trim().toLowerCase() !== 'application/octet-stream' ||
          (encoding !== null && encoding.toLowerCase() !== 'identity')) throw new ArtifactError('unavailable');
      expectedSize = Number(length);
      if (expectedSize > MAX_CONSOLE_DOWNLOAD_BYTES) throw new ArtifactError('too_large');
      expectedHash = digest;
    }
    if (target.expectedSha256 && Number(response.headers.get('content-length') ?? '0') > MAX_PREVIEW_BYTES) throw new ArtifactError('changed');
    temporary = `${destination}.biomaker-${randomUUID()}.part`;
    const hash = createHash('sha256');
    let size = 0;
    const check = new Transform({ transform(chunk: Buffer, _encoding, callback) {
      size += chunk.length;
      if (source === 'console' && size > MAX_CONSOLE_DOWNLOAD_BYTES) { callback(new ArtifactError('too_large')); return; }
      if (expectedSize !== undefined && size > expectedSize) { callback(new ArtifactError('changed')); return; }
      if (target.expectedSha256 && size > MAX_PREVIEW_BYTES) { callback(new ArtifactError('changed')); return; }
      hash.update(chunk);
      callback(null, chunk);
    } });
    writing = true;
    await pipeline(Readable.fromWeb(response.body as any), check, createWriteStream(temporary, { flags: 'wx', mode: 0o600 }));
    assertCurrent();
    if (expectedSize !== undefined && size !== expectedSize) throw new ArtifactError('changed');
    if (expectedHash && hash.digest('hex') !== expectedHash) throw new ArtifactError('changed');
    assertCurrent();
    renameSync(temporary, destination);
    return destination;
  } catch (error) {
    await response?.body?.cancel().catch(() => {});
    assertCurrent();
    throw error instanceof ArtifactError ? error : new ArtifactError(writing ? 'save_failed' : 'unavailable');
  } finally {
    if (temporary) { try { rmSync(temporary, { force: true }); } catch { /* Preserve the original fixed failure. */ } }
  }
}

/** One app-owned instance covers window/run navigation as well as the native
 * dialog and streaming download. Refuse overlap instead of queuing a stale
 * export which could overwrite a newer user decision at the same destination. */
export function createArtifactSaver(): typeof saveArtifact {
  let active = false;
  return async (...args) => {
    if (active) throw new ArtifactError('busy');
    active = true;
    try { return await saveArtifact(...args); }
    finally { active = false; }
  };
}
