import { constants } from 'node:fs';
import { access, lstat, open, realpath, stat } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

export const MAX_DIAGNOSTIC_EVENTS = 50;
export const MAX_DIAGNOSTIC_MANIFEST_BYTES = 64 * 1024;
export const DIAGNOSTIC_FAILURE_CODES = [
  'SERVICE_CONSTRUCTION_FAILED', 'SERVICE_START_FAILED', 'SERVICE_FAILED', 'SERVICE_STOP_FAILED',
] as const;
export const DIAGNOSTIC_EVENT_CODES = [
  ...DIAGNOSTIC_FAILURE_CODES,
  'SERVICE_STARTING', 'SERVICE_READY', 'SERVICE_RECONNECTING', 'SERVICE_RETRY_REQUESTED', 'SERVICE_STOPPED',
] as const;

type Unknown = 'unknown';
type Status = 'starting' | 'ready' | 'reconnecting' | 'failed' | Unknown;
type FailureCode = typeof DIAGNOSTIC_FAILURE_CODES[number];
type EventCode = typeof DIAGNOSTIC_EVENT_CODES[number];
type ManifestStatus = 'ok' | 'missing' | 'unreadable' | 'invalid' | 'too_large' | 'unsafe' | 'changed';

export interface DiagnosticsInput {
  appVersion: string;
  packaged: boolean;
  // Trusted bundle/build locations supplied by the main process, never renderer paths.
  runtimeDirectory: string;
  servicePath: string;
  // Optional trusted profile root. Only root/logs/state directory metadata is inspected.
  dataDirectory?: string;
  serviceState: unknown;
  events?: readonly unknown[];
}

export interface DiagnosticsReport {
  schemaVersion: 1;
  createdAt: string;
  application: { version: string; packaged: boolean | Unknown };
  system: { platform: string; arch: string; osRelease: string; electronVersion: string; nodeVersion: string };
  service: { status: Status; revision: number | Unknown; errorCode: FailureCode | Unknown | null };
  events: { code: EventCode | 'UNKNOWN'; status: Status; timestamp: string }[];
  bundle: {
    applicationVersion: string;
    target: string;
    sourceCommit: string;
    engineSha256: string;
    dirtySource: boolean | Unknown;
    pythonVersion: string;
    pythonBuild: string;
  };
  runtime: {
    exists: boolean;
    readable: boolean;
    manifestStatus: ManifestStatus;
    files: { manifest: boolean; python: boolean; uv: boolean; fastapi: boolean; engine: boolean; service: boolean };
  };
  storage: { root: StorageDirectoryStatus; logs: StorageDirectoryStatus; state: StorageDirectoryStatus } | null;
}

export interface StorageDirectoryStatus {
  exists: boolean;
  directory: boolean;
  // Applies only to an existing non-symlink directory; says nothing about
  // whether a missing directory could be created by its parent.
  existingWritable: boolean;
}

const statuses = new Set(['starting', 'ready', 'reconnecting', 'failed']);
const failures = new Set<string>(DIAGNOSTIC_FAILURE_CODES);
const eventCodes = new Set<string>(DIAGNOSTIC_EVENT_CODES);
const platforms = new Set(['aix', 'android', 'darwin', 'freebsd', 'haiku', 'linux', 'openbsd', 'sunos', 'win32']);
const architectures = new Set(['arm', 'arm64', 'ia32', 'loong64', 'mips', 'mipsel', 'ppc', 'ppc64', 'riscv64', 's390', 's390x', 'x64']);

function object(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function format(value: unknown, pattern: RegExp, max = 64): string {
  return typeof value === 'string' && value.length <= max && pattern.test(value) ? value : 'unknown';
}
function version(value: unknown): string {
  // No arbitrary semver metadata: it could contain a username, hostname or path.
  return format(value, /^\d{1,4}\.\d{1,4}\.\d{1,4}(?:-(?:alpha|beta|rc|dev)(?:\.?\d{1,4})?)?$/, 40);
}
function status(value: unknown): Status {
  return typeof value === 'string' && statuses.has(value) ? value as Status : 'unknown';
}
function timestamp(value: unknown): string {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)) return 'unknown';
  const time = new Date(value);
  return Number.isFinite(time.getTime()) && time.toISOString() === value ? value : 'unknown';
}
function sanitizeService(value: unknown): DiagnosticsReport['service'] {
  const raw = object(value);
  const safeStatus = status(raw.status);
  const revision = typeof raw.revision === 'number' && Number.isSafeInteger(raw.revision) && raw.revision >= 0 ? raw.revision : 'unknown';
  const errorCode = typeof raw.code === 'string' && failures.has(raw.code)
    ? raw.code as FailureCode
    : safeStatus === 'failed' ? 'SERVICE_FAILED' : safeStatus === 'unknown' ? 'unknown' : null;
  return { status: safeStatus, revision, errorCode };
}
function sanitizeEvents(value: DiagnosticsInput['events']): DiagnosticsReport['events'] {
  if (!Array.isArray(value)) return [];
  return value.slice(-MAX_DIAGNOSTIC_EVENTS).map(item => {
    const raw = object(item);
    return {
      code: typeof raw.code === 'string' && eventCodes.has(raw.code) ? raw.code as EventCode : 'UNKNOWN',
      status: status(raw.status),
      timestamp: timestamp(raw.timestamp),
    };
  });
}
function sanitizeBundle(value: unknown): DiagnosticsReport['bundle'] {
  const raw = object(value);
  return {
    applicationVersion: version(raw.applicationVersion),
    target: format(raw.target, /^(?:darwin|linux|win32)-(?:arm64|x64|ia32|armv7l)$/, 24),
    sourceCommit: format(raw.sourceCommit, /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/),
    engineSha256: format(raw.engineSha256, /^[a-f0-9]{64}$/),
    dirtySource: typeof raw.sourceFilesMayIncludeUncommittedChanges === 'boolean' ? raw.sourceFilesMayIncludeUncommittedChanges : 'unknown',
    pythonVersion: format(raw.pythonVersion, /^3\.\d{1,3}\.\d{1,3}$/, 12),
    pythonBuild: format(raw.pythonBuild, /^\d{8}$/, 8),
  };
}
function inside(candidate: string, root: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

async function readableFile(file: string, runtimeRoot?: string): Promise<boolean> {
  try {
    // Python may be a bundle-internal symlink. Check its resolved boundary but
    // never read its contents or follow an external target for availability.
    const resolved = await realpath(file);
    if (runtimeRoot && !inside(resolved, runtimeRoot)) return false;
    if (!(await stat(resolved)).isFile()) return false;
    await access(resolved, constants.R_OK);
    return true;
  } catch { return false; }
}

async function readManifest(root: string): Promise<{ readable: boolean; status: ManifestStatus; value?: unknown }> {
  const file = path.join(root, 'manifest.json');
  let initial;
  try { initial = await lstat(file); }
  catch (error) {
    // Only classify an explicitly known missing-file condition; never export
    // an Error object, its message, errno, syscall or path.
    return { readable: false, status: object(error).code === 'ENOENT' ? 'missing' : 'unreadable' };
  }
  if (!initial.isFile() || initial.isSymbolicLink()) return { readable: false, status: 'unsafe' };
  if (initial.size > MAX_DIAGNOSTIC_MANIFEST_BYTES) {
    try { await access(file, constants.R_OK); return { readable: true, status: 'too_large' }; }
    catch { return { readable: false, status: 'unreadable' }; }
  }
  let handle;
  try {
    // O_NONBLOCK also prevents a raced-in FIFO from hanging before fstat.
    handle = await open(file, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0) | (constants.O_NONBLOCK ?? 0));
    const before = await handle.stat();
    if (!before.isFile() || before.dev !== initial.dev || before.ino !== initial.ino) return { readable: false, status: 'unsafe' };
    if (before.size > MAX_DIAGNOSTIC_MANIFEST_BYTES) return { readable: true, status: 'too_large' };
    const bytes = Buffer.alloc(MAX_DIAGNOSTIC_MANIFEST_BYTES);
    let length = 0;
    while (length < bytes.length) {
      const { bytesRead } = await handle.read(bytes, length, bytes.length - length, length);
      if (!bytesRead) break;
      length += bytesRead;
    }
    const after = await handle.stat();
    if (after.size > MAX_DIAGNOSTIC_MANIFEST_BYTES) return { readable: true, status: 'too_large' };
    if (before.size !== after.size || before.mtimeMs !== after.mtimeMs || length !== after.size) return { readable: true, status: 'changed' };
    let value: unknown;
    try { value = JSON.parse(bytes.subarray(0, length).toString('utf8')); }
    catch { return { readable: true, status: 'invalid' }; }
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return { readable: true, status: 'invalid' };
    return { readable: true, status: 'ok', value };
  } catch { return { readable: false, status: 'unreadable' }; }
  finally { if (handle) await handle.close().catch(() => {}); }
}

function absentDirectory(): StorageDirectoryStatus {
  return { exists: false, directory: false, existingWritable: false };
}
async function storageDirectory(directory: string): Promise<StorageDirectoryStatus> {
  const result = absentDirectory();
  try {
    const info = await lstat(directory);
    result.exists = true;
    result.directory = info.isDirectory() && !info.isSymbolicLink();
    if (result.directory) {
      try { await access(directory, constants.W_OK | constants.X_OK); result.existingWritable = true; }
      catch { /* An existing but unwritable directory remains distinguishable. */ }
    }
  } catch { /* No raw filesystem errors are included in a report. */ }
  return result;
}
async function collectStorage(dataDirectory: DiagnosticsInput['dataDirectory']): Promise<DiagnosticsReport['storage']> {
  if (dataDirectory === undefined) return null;
  const root = await storageDirectory(dataDirectory);
  // Never traverse a root that is a file or symlink, and never enumerate or
  // read profile contents. These are the only two child names inspected.
  const [logs, state] = root.directory
    ? await Promise.all(['logs', 'state'].map(name => storageDirectory(path.join(dataDirectory, name))))
    : [absentDirectory(), absentDirectory()];
  return { root, logs, state };
}

/** Collect a bounded, allowlisted report without a running sidecar or profile-content access. */
export async function collectDiagnostics(input: DiagnosticsInput): Promise<DiagnosticsReport> {
  const runtime: DiagnosticsReport['runtime'] = {
    exists: false, readable: false, manifestStatus: 'missing',
    files: { manifest: false, python: false, uv: false, fastapi: false, engine: false, service: false },
  };
  let bundle = sanitizeBundle(undefined);
  // No directory enumeration, subprocesses, profile/config/ledger/log reads or
  // environment collection. Only the fixed runtime manifest is read as bytes.
  runtime.files.service = await readableFile(input.servicePath);
  try {
    const info = await lstat(input.runtimeDirectory);
    runtime.exists = true;
    if (info.isSymbolicLink() || !info.isDirectory()) {
      runtime.manifestStatus = 'unsafe';
    } else {
      const root = await realpath(input.runtimeDirectory);
      await access(root, constants.R_OK | constants.X_OK);
      runtime.readable = true;
      const manifest = await readManifest(root);
      runtime.manifestStatus = manifest.status;
      runtime.files.manifest = manifest.readable;
      bundle = sanitizeBundle(manifest.value);
      const required = [
        process.platform === 'win32' ? 'python/python.exe' : 'python/bin/python3',
        process.platform === 'win32' ? 'bin/uv.exe' : 'bin/uv',
        'site-packages/fastapi/__init__.py', 'engine/pyproject.toml',
      ];
      [runtime.files.python, runtime.files.uv, runtime.files.fastapi, runtime.files.engine] = await Promise.all(required.map(file => readableFile(path.join(root, file), root)));
    }
  } catch (error) {
    runtime.manifestStatus = object(error).code === 'ENOENT' ? 'missing' : 'unreadable';
  }
  return {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    application: { version: version(input.appVersion), packaged: typeof input.packaged === 'boolean' ? input.packaged : 'unknown' },
    system: {
      platform: platforms.has(process.platform) ? process.platform : 'unknown',
      arch: architectures.has(process.arch) ? process.arch : 'unknown',
      osRelease: format(os.release(), /^\d{1,5}(?:\.\d{1,5}){1,3}$/, 24),
      electronVersion: version(process.versions.electron),
      nodeVersion: version(process.versions.node),
    },
    service: sanitizeService(input.serviceState),
    events: sanitizeEvents(input.events),
    bundle,
    runtime,
    storage: await collectStorage(input.dataDirectory),
  };
}
