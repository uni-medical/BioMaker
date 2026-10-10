import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, symlinkSync, chmodSync, realpathSync } from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  collectDiagnostics, DIAGNOSTIC_EVENT_CODES, DIAGNOSTIC_FAILURE_CODES,
  MAX_DIAGNOSTIC_EVENTS, MAX_DIAGNOSTIC_MANIFEST_BYTES, type DiagnosticsInput,
} from './diagnostics';

const sentinel = 'PRIVATE_SENTINEL_provider_key_user_hostname_dataset';
const instant = '2026-10-09T01:23:45.678Z';
const validManifest = {
  applicationVersion: '0.1.6', target: 'darwin-arm64', sourceCommit: 'a'.repeat(40), engineSha256: 'b'.repeat(64),
  sourceFilesMayIncludeUncommittedChanges: false, pythonVersion: '3.12.12', pythonBuild: '20260211',
};
function fixture() {
  const root = mkdtempSync(path.join(os.tmpdir(), 'biomaker-diagnostics-'));
  const runtime = path.join(root, sentinel, 'runtime');
  mkdirSync(runtime, { recursive: true });
  const servicePath = path.join(root, 'service', 'main.py');
  mkdirSync(path.dirname(servicePath));
  writeFileSync(servicePath, sentinel);
  const manifest = path.join(runtime, 'manifest.json');
  writeFileSync(manifest, JSON.stringify(validManifest));
  for (const file of [process.platform === 'win32' ? 'python/python.exe' : 'python/bin/python3', process.platform === 'win32' ? 'bin/uv.exe' : 'bin/uv', 'site-packages/fastapi/__init__.py', 'engine/pyproject.toml']) {
    const target = path.join(runtime, file); mkdirSync(path.dirname(target), { recursive: true }); writeFileSync(target, sentinel);
  }
  const input: DiagnosticsInput = { appVersion: '0.1.6', packaged: true, runtimeDirectory: runtime, servicePath, serviceState: { status: 'failed', revision: 4, code: 'SERVICE_CONSTRUCTION_FAILED', message: sentinel }, events: [] };
  return { root, runtime, manifest, input, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}
function privateValuesAbsent(report: unknown, extra: string[] = []) {
  const json = JSON.stringify(report);
  for (const value of [sentinel, ...extra]) assert.equal(json.includes(value), false, `Report exposed a private value: ${value}`);
  assert.ok(Buffer.byteLength(json) < 16 * 1024, 'Report must remain bounded');
  assert.deepEqual(JSON.parse(json), report, 'Report must be an ordinary JSON object');
}

test('collects only allowlisted bundle and system facts without a sidecar', async t => {
  const f = fixture(); t.after(f.cleanup);
  const report = await collectDiagnostics(f.input);
  assert.equal(report.schemaVersion, 1);
  assert.equal(new Date(report.createdAt).toISOString(), report.createdAt);
  assert.deepEqual(report.application, { version: '0.1.6', packaged: true });
  assert.deepEqual(report.service, { status: 'failed', revision: 4, errorCode: 'SERVICE_CONSTRUCTION_FAILED' });
  assert.deepEqual(report.bundle, { applicationVersion: '0.1.6', target: 'darwin-arm64', sourceCommit: 'a'.repeat(40), engineSha256: 'b'.repeat(64), dirtySource: false, pythonVersion: '3.12.12', pythonBuild: '20260211' });
  assert.deepEqual(report.runtime, { exists: true, readable: true, manifestStatus: 'ok', files: { manifest: true, python: true, uv: true, fastapi: true, engine: true, service: true } });
  assert.equal(report.storage, null, 'An omitted profile location is not reported as a missing profile');
  assert.deepEqual(Object.keys(report.system).sort(), ['arch', 'electronVersion', 'nodeVersion', 'osRelease', 'platform']);
  privateValuesAbsent(report, [f.root, f.runtime, f.input.servicePath]);
});

test('does not include arbitrary manifest fields, error text, profile content or environment values', async t => {
  const f = fixture(); t.after(f.cleanup);
  writeFileSync(f.manifest, JSON.stringify({ ...validManifest, pythonDistribution: sentinel, task: sentinel, provider: { key: sentinel }, paths: [f.root], environment: { secret: sentinel }, rawLogs: sentinel, hostname: sentinel, username: sentinel }));
  for (const file of ['.env', 'config.json', 'run-ledger.sqlite3', 'desktop-service.log', 'task.txt', 'datasets/private.csv']) {
    const target = path.join(f.root, 'profile', file); mkdirSync(path.dirname(target), { recursive: true }); writeFileSync(target, sentinel);
  }
  f.input.dataDirectory = path.join(f.root, 'profile');
  const originalOpen = fs.open;
  const manifestPath = realpathSync(f.manifest);
  const opened: string[] = [];
  t.mock.method(fs, 'open', (...args: Parameters<typeof fs.open>) => {
    const [file] = args;
    opened.push(String(file));
    assert.equal(String(file), manifestPath, 'Only the fixed bundle manifest may be opened');
    return originalOpen(...args);
  });
  t.mock.method(fs, 'readFile', () => assert.fail('Unbounded file reads are forbidden'));
  t.mock.method(fs, 'readdir', () => assert.fail('Directory scanning is forbidden'));
  const old = process.env.BIOMAKER_DIAGNOSTICS_TEST_SECRET;
  process.env.BIOMAKER_DIAGNOSTICS_TEST_SECRET = sentinel;
  t.after(() => { if (old === undefined) delete process.env.BIOMAKER_DIAGNOSTICS_TEST_SECRET; else process.env.BIOMAKER_DIAGNOSTICS_TEST_SECRET = old; });
  f.input.serviceState = { status: 'failed', revision: 8, code: sentinel, message: sentinel, error: new Error(`${sentinel}: ${f.root}`), env: process.env };
  f.input.events = [{ code: 'SERVICE_START_FAILED', status: 'failed', timestamp: instant, message: sentinel, error: sentinel, path: f.root }];
  const report = await collectDiagnostics(f.input);
  assert.deepEqual(report.events, [{ code: 'SERVICE_START_FAILED', status: 'failed', timestamp: instant }]);
  assert.equal(report.service.errorCode, 'SERVICE_FAILED');
  assert.deepEqual(opened, [manifestPath]);
  assert.equal(report.bundle.applicationVersion, '0.1.6');
  assert.deepEqual(report.storage, {
    root: { exists: true, directory: true, existingWritable: true },
    logs: { exists: false, directory: false, existingWritable: false },
    state: { exists: false, directory: false, existingWritable: false },
  });
  privateValuesAbsent(report, [f.root, 'BIOMAKER_DIAGNOSTICS_TEST_SECRET']);
});

test('rejects private or malformed values even in allowlisted manifest and application fields', async t => {
  const f = fixture(); t.after(f.cleanup);
  writeFileSync(f.manifest, JSON.stringify({ applicationVersion: `0.1.6+${sentinel}`, target: `darwin-${sentinel}`, sourceCommit: sentinel, engineSha256: sentinel, sourceFilesMayIncludeUncommittedChanges: sentinel, pythonVersion: `3.12.12/${sentinel}`, pythonBuild: sentinel }));
  f.input.appVersion = `0.1.6-${sentinel}`;
  f.input.packaged = sentinel as unknown as boolean;
  const report = await collectDiagnostics(f.input);
  assert.deepEqual(report.application, { version: 'unknown', packaged: 'unknown' });
  assert.ok(Object.values(report.bundle).every(value => value === 'unknown'));
  privateValuesAbsent(report);
});

test('missing runtime still produces a serializable report with failed service code', async t => {
  const f = fixture(); t.after(f.cleanup); rmSync(f.runtime, { recursive: true });
  const report = await collectDiagnostics(f.input);
  assert.equal(report.runtime.exists, false);
  assert.equal(report.runtime.readable, false);
  assert.equal(report.runtime.manifestStatus, 'missing');
  assert.equal(report.runtime.files.service, true);
  assert.ok(Object.values(report.bundle).every(value => value === 'unknown'));
  privateValuesAbsent(report, [f.root]);
});

test('missing manifest and required files are reported independently', async t => {
  const f = fixture(); t.after(f.cleanup);
  rmSync(f.manifest); rmSync(path.join(f.runtime, 'python'), { recursive: true }); rmSync(f.input.servicePath);
  const report = await collectDiagnostics(f.input);
  assert.equal(report.runtime.manifestStatus, 'missing');
  assert.deepEqual(report.runtime.files, { manifest: false, python: false, uv: true, fastapi: true, engine: true, service: false });
  privateValuesAbsent(report);
});

test('malformed or non-object manifests do not prevent exporting other diagnostics', async t => {
  const f = fixture(); t.after(f.cleanup);
  for (const content of ['', '{', `{"error":"${sentinel}","x":NaN}`, 'null', '[]', 'false', '42']) {
    writeFileSync(f.manifest, content);
    const report = await collectDiagnostics(f.input);
    assert.equal(report.runtime.manifestStatus, 'invalid');
    assert.ok(Object.values(report.bundle).every(value => value === 'unknown'));
    assert.equal(report.runtime.files.python, true);
    privateValuesAbsent(report);
  }
});

test('manifest byte limit accepts exactly 64 KiB and rejects larger files', async t => {
  const f = fixture(); t.after(f.cleanup);
  const valid = JSON.stringify(validManifest);
  writeFileSync(f.manifest, valid.padEnd(MAX_DIAGNOSTIC_MANIFEST_BYTES, ' '));
  assert.equal((await collectDiagnostics(f.input)).runtime.manifestStatus, 'ok');
  writeFileSync(f.manifest, valid.padEnd(MAX_DIAGNOSTIC_MANIFEST_BYTES + 1, ' '));
  const report = await collectDiagnostics(f.input);
  assert.equal(report.runtime.manifestStatus, 'too_large');
  assert.ok(Object.values(report.bundle).every(value => value === 'unknown'));
  privateValuesAbsent(report);
});

test('symlink manifests are rejected even when the external target is a valid manifest', { skip: process.platform === 'win32' }, async t => {
  const f = fixture(); t.after(f.cleanup);
  const outside = path.join(f.root, 'private-manifest.json');
  writeFileSync(outside, JSON.stringify({ ...validManifest, applicationVersion: '9.9.9' }));
  rmSync(f.manifest); symlinkSync(outside, f.manifest);
  const report = await collectDiagnostics(f.input);
  assert.equal(report.runtime.manifestStatus, 'unsafe');
  assert.equal(report.runtime.files.manifest, false);
  assert.equal(report.bundle.applicationVersion, 'unknown');
  privateValuesAbsent(report, [outside, '9.9.9']);
});

test('directory and FIFO manifests are rejected without opening or blocking', { skip: process.platform === 'win32', timeout: 2000 }, async t => {
  const f = fixture(); t.after(f.cleanup);
  rmSync(f.manifest); mkdirSync(f.manifest);
  assert.equal((await collectDiagnostics(f.input)).runtime.manifestStatus, 'unsafe');
  rmSync(f.manifest, { recursive: true }); execFileSync('mkfifo', [f.manifest]);
  const report = await collectDiagnostics(f.input);
  assert.equal(report.runtime.manifestStatus, 'unsafe');
  privateValuesAbsent(report);
});

test('symlink runtime roots are not traversed', { skip: process.platform === 'win32' }, async t => {
  const f = fixture(); t.after(f.cleanup);
  const link = path.join(f.root, 'linked-runtime'); symlinkSync(f.runtime, link);
  const report = await collectDiagnostics({ ...f.input, runtimeDirectory: link });
  assert.equal(report.runtime.exists, true);
  assert.equal(report.runtime.readable, false);
  assert.equal(report.runtime.manifestStatus, 'unsafe');
  assert.equal(report.bundle.applicationVersion, 'unknown');
  privateValuesAbsent(report);
});

test('availability accepts internal Python links but rejects links outside the runtime', { skip: process.platform === 'win32' }, async t => {
  const f = fixture(); t.after(f.cleanup);
  const python = path.join(f.runtime, 'python/bin/python3');
  const internal = path.join(f.runtime, 'python/bin/python3.12'); writeFileSync(internal, sentinel);
  rmSync(python); symlinkSync('python3.12', python);
  assert.equal((await collectDiagnostics(f.input)).runtime.files.python, true);
  rmSync(python); symlinkSync(f.input.servicePath, python);
  const report = await collectDiagnostics(f.input);
  assert.equal(report.runtime.files.python, false);
  privateValuesAbsent(report);
});

test('unreadable manifest returns a fixed status with no filesystem error text', { skip: process.platform === 'win32' || process.getuid?.() === 0 }, async t => {
  const f = fixture(); t.after(f.cleanup); chmodSync(f.manifest, 0);
  t.after(() => { try { chmodSync(f.manifest, 0o600); } catch {} });
  const report = await collectDiagnostics(f.input);
  assert.equal(report.runtime.manifestStatus, 'unreadable');
  privateValuesAbsent(report, [f.root, 'EACCES', 'permission denied']);
  chmodSync(f.manifest, 0o600); writeFileSync(f.manifest, ' '.repeat(MAX_DIAGNOSTIC_MANIFEST_BYTES + 1)); chmodSync(f.manifest, 0);
  const oversized = await collectDiagnostics(f.input);
  assert.equal(oversized.runtime.manifestStatus, 'unreadable');
  assert.equal(oversized.runtime.files.manifest, false);
});

test('service errors accept only the controller failure-code union and safe revisions', async t => {
  const f = fixture(); t.after(f.cleanup);
  for (const code of DIAGNOSTIC_FAILURE_CODES) assert.equal((await collectDiagnostics({ ...f.input, serviceState: { status: 'failed', revision: 0, code, message: sentinel } })).service.errorCode, code);
  for (const revision of [-1, 0.5, Infinity, Number.MAX_SAFE_INTEGER + 1, sentinel]) {
    const report = await collectDiagnostics({ ...f.input, serviceState: { status: sentinel, revision, code: sentinel } });
    assert.deepEqual(report.service, { status: 'unknown', revision: 'unknown', errorCode: 'unknown' });
    privateValuesAbsent(report);
  }
  assert.deepEqual((await collectDiagnostics({ ...f.input, serviceState: { status: 'ready', revision: 0 } })).service, { status: 'ready', revision: 0, errorCode: null });
});

test('events retain only the newest 50 entries and known structured fields', async t => {
  const f = fixture(); t.after(f.cleanup);
  const events = Array.from({ length: 5000 }, (_, i) => ({ code: DIAGNOSTIC_EVENT_CODES[i % DIAGNOSTIC_EVENT_CODES.length], status: 'failed', timestamp: new Date(Date.UTC(2026, 9, 9, 0, 0, i)).toISOString(), message: sentinel, sequence: i }));
  const report = await collectDiagnostics({ ...f.input, events });
  assert.equal(report.events.length, MAX_DIAGNOSTIC_EVENTS);
  assert.equal(report.events[0].timestamp, events[events.length - MAX_DIAGNOSTIC_EVENTS].timestamp);
  assert.deepEqual(Object.keys(report.events[0]).sort(), ['code', 'status', 'timestamp']);
  privateValuesAbsent(report);
});

test('invalid event codes, timestamps and statuses cannot become free-text channels', async t => {
  const f = fixture(); t.after(f.cleanup);
  const report = await collectDiagnostics({ ...f.input, events: [null, sentinel, { code: sentinel, status: sentinel, timestamp: sentinel }, { code: 'SERVICE_READY', status: 'ready', timestamp: '2026-02-30T01:23:45.678Z' }, { code: 'SERVICE_RETRY_REQUESTED', status: 'starting', timestamp: instant }] });
  assert.deepEqual(report.events.slice(0, 3), Array.from({ length: 3 }, () => ({ code: 'UNKNOWN', status: 'unknown', timestamp: 'unknown' })));
  assert.equal(report.events[3].timestamp, 'unknown');
  assert.deepEqual(report.events[4], { code: 'SERVICE_RETRY_REQUESTED', status: 'starting', timestamp: instant });
  privateValuesAbsent(report);
});

test('invalid OS release and version metadata are reduced to unknown', async t => {
  const f = fixture(); t.after(f.cleanup);
  t.mock.method(os, 'release', () => `25.0.0-${sentinel}`);
  const report = await collectDiagnostics({ ...f.input, appVersion: '1.2.3+host.private' });
  assert.equal(report.system.osRelease, 'unknown');
  assert.equal(report.application.version, 'unknown');
  privateValuesAbsent(report, ['host.private']);
});

test('storage diagnostics identify a regular logs file while the runtime is healthy', async t => {
  const f = fixture(); t.after(f.cleanup);
  const profile = path.join(f.root, sentinel, 'profile'); mkdirSync(profile);
  writeFileSync(path.join(profile, 'logs'), sentinel); mkdirSync(path.join(profile, 'state'));
  writeFileSync(path.join(profile, 'state', 'desktop-metadata.sqlite3'), sentinel);
  const report = await collectDiagnostics({ ...f.input, dataDirectory: profile });
  assert.equal(report.runtime.manifestStatus, 'ok');
  assert.deepEqual(report.storage, {
    root: { exists: true, directory: true, existingWritable: true },
    logs: { exists: true, directory: false, existingWritable: false },
    state: { exists: true, directory: true, existingWritable: true },
  });
  privateValuesAbsent(report, [profile, f.root]);
});

test('missing profile and missing subdirectories are reported without guessing creation ability', async t => {
  const f = fixture(); t.after(f.cleanup);
  const profile = path.join(f.root, sentinel, 'missing-profile');
  const absent = { exists: false, directory: false, existingWritable: false };
  const missing = await collectDiagnostics({ ...f.input, dataDirectory: profile });
  assert.deepEqual(missing.storage, { root: absent, logs: absent, state: absent });
  mkdirSync(profile);
  const empty = await collectDiagnostics({ ...f.input, dataDirectory: profile });
  assert.deepEqual(empty.storage, { root: { exists: true, directory: true, existingWritable: true }, logs: absent, state: absent });
  privateValuesAbsent(missing, [profile]); privateValuesAbsent(empty, [profile]);
});

test('profile inspection uses only lstat/access on the three fixed locations', async t => {
  const f = fixture(); t.after(f.cleanup);
  const profile = path.join(f.root, sentinel, 'profile');
  for (const directory of [profile, path.join(profile, 'logs'), path.join(profile, 'state')]) mkdirSync(directory, { recursive: true });
  for (const file of ['.env', 'config.json', 'logs/desktop-service.log', 'state/run-ledger.sqlite3']) writeFileSync(path.join(profile, file), sentinel);
  const originalLstat = fs.lstat, originalAccess = fs.access;
  const observedStats: string[] = [], observedAccess: string[] = [];
  t.mock.method(fs, 'lstat', ((...args: Parameters<typeof fs.lstat>) => {
    if (String(args[0]).startsWith(profile)) observedStats.push(String(args[0]));
    return originalLstat(...args);
  }) as typeof fs.lstat);
  t.mock.method(fs, 'access', (...args: Parameters<typeof fs.access>) => {
    if (String(args[0]).startsWith(profile)) observedAccess.push(String(args[0]));
    return originalAccess(...args);
  });
  const originalOpen = fs.open;
  const opened: string[] = [];
  t.mock.method(fs, 'open', (...args: Parameters<typeof fs.open>) => { opened.push(String(args[0])); return originalOpen(...args); });
  const readFile = t.mock.method(fs, 'readFile', () => assert.fail('No profile file may be read'));
  const readdir = t.mock.method(fs, 'readdir', () => assert.fail('No profile directory may be scanned'));
  const report = await collectDiagnostics({ ...f.input, dataDirectory: profile });
  const expected = [profile, path.join(profile, 'logs'), path.join(profile, 'state')].sort();
  assert.deepEqual(observedStats.sort(), expected);
  assert.deepEqual(observedAccess.sort(), expected);
  assert.deepEqual(opened, [realpathSync(f.manifest)]);
  assert.equal(readFile.mock.callCount(), 0); assert.equal(readdir.mock.callCount(), 0);
  privateValuesAbsent(report, [profile]);
});

test('storage inspection rejects symlink roots and children without accessing their targets', { skip: process.platform === 'win32' }, async t => {
  const f = fixture(); t.after(f.cleanup);
  const profile = path.join(f.root, sentinel, 'profile'); mkdirSync(profile);
  const outside = path.join(f.root, 'private-target'); mkdirSync(outside); mkdirSync(path.join(outside, 'logs')); mkdirSync(path.join(outside, 'state'));
  symlinkSync(outside, path.join(profile, 'logs')); symlinkSync(outside, path.join(profile, 'state'));
  const rootLink = path.join(f.root, 'profile-link'); symlinkSync(outside, rootLink);
  const originalAccess = fs.access;
  const accessed: string[] = [];
  t.mock.method(fs, 'access', (...args: Parameters<typeof fs.access>) => { accessed.push(String(args[0])); return originalAccess(...args); });
  const children = await collectDiagnostics({ ...f.input, dataDirectory: profile });
  const root = await collectDiagnostics({ ...f.input, dataDirectory: rootLink });
  const link = { exists: true, directory: false, existingWritable: false };
  const absent = { exists: false, directory: false, existingWritable: false };
  assert.deepEqual(children.storage, { root: { exists: true, directory: true, existingWritable: true }, logs: link, state: link });
  assert.deepEqual(root.storage, { root: link, logs: absent, state: absent });
  for (const forbidden of [rootLink, outside, path.join(profile, 'logs'), path.join(profile, 'state')]) assert.equal(accessed.includes(forbidden), false);
  privateValuesAbsent(children, [profile, outside]); privateValuesAbsent(root, [rootLink, outside]);
});

test('existing unwritable storage stays distinct from missing storage', { skip: process.platform === 'win32' || process.getuid?.() === 0 }, async t => {
  const f = fixture(); t.after(f.cleanup);
  const profile = path.join(f.root, sentinel, 'profile'); mkdirSync(profile); mkdirSync(path.join(profile, 'logs'));
  chmodSync(path.join(profile, 'logs'), 0o500);
  t.after(() => { try { chmodSync(path.join(profile, 'logs'), 0o700); } catch {} });
  const report = await collectDiagnostics({ ...f.input, dataDirectory: profile });
  assert.deepEqual(report.storage?.logs, { exists: true, directory: true, existingWritable: false });
  assert.deepEqual(report.storage?.state, { exists: false, directory: false, existingWritable: false });
  privateValuesAbsent(report, [profile]);
});
