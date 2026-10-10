import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, symlinkSync, utimesSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { prepareSelectedEngine, selectEngine } from './engine-selection';

function fixture() {
  const root = mkdtempSync(path.join(os.tmpdir(), 'BioMaker engine upgrade '));
  const runtime = path.join(root, 'bundle/runtime');
  const data = path.join(root, 'profile');
  const source = path.join(runtime, 'engine');
  mkdirSync(path.join(source, 'biomaker'), { recursive: true });
  writeFileSync(path.join(source, 'biomaker/run.py'), 'print("engine source")\n');
  writeFileSync(path.join(source, 'pyproject.toml'), '[project]\nname="biomaker"\n');
  writeFileSync(path.join(source, 'requirements.txt'), 'torch==2.14.1\n');
  writeFileSync(path.join(source, 'uv.lock'), 'locked dependencies\n');
  const sourceHash = () => {
    const hash = createHash('sha256');
    const visit = (directory: string) => { for (const item of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) { const file = path.join(directory, item.name); if (item.isDirectory()) visit(file); else { hash.update(path.relative(source, file)); hash.update(readFileSync(file)); } } };
    visit(source); return hash.digest('hex');
  };
  const manifest = { target: `${process.platform}-${process.arch}`, pythonVersion: '3.12.12', pythonBuild: '20260211', applicationVersion: '0.1.1', engineSha256: sourceHash() };
  const saveManifest = () => { manifest.engineSha256 = sourceHash(); writeFileSync(path.join(runtime, 'manifest.json'), JSON.stringify(manifest)); };
  saveManifest();
  const legacy = (version = '0.1.1') => {
    const engine = path.join(data, 'engines', `${version}-${manifest.engineSha256.slice(0, 12)}`);
    cpSync(source, engine, { recursive: true });
    writeFileSync(path.join(engine, '.desktop-engine-ready'), version);
    const retained = path.join(data, 'runtimes', `retained-${version}`);
    mkdirSync(path.join(retained, 'python/bin'), { recursive: true });
    const binary = path.join(retained, process.platform === 'win32' ? 'python/python.exe' : 'python/bin/python3.12');
    writeFileSync(binary, 'retained interpreter fixture');
    writeFileSync(path.join(retained, 'manifest.json'), JSON.stringify(manifest));
    writeFileSync(path.join(retained, '.desktop-runtime-ready'), version);
    const bin = path.join(engine, '.venv', process.platform === 'win32' ? 'Scripts' : 'bin');
    mkdirSync(bin, { recursive: true });
    symlinkSync(binary, path.join(bin, process.platform === 'win32' ? 'python.exe' : 'python'));
    writeFileSync(path.join(engine, '.venv/pyvenv.cfg'), 'implementation = CPython\nversion_info = 3.12.12\ninclude-system-site-packages = false\n');
    writeFileSync(path.join(engine, '.desktop-engine.json'), JSON.stringify({ version, fingerprint: 'service-owned-fingerprint', native_runtime: { revision: 'service-owned-revision' } }));
    return { engine, binary, retained };
  };
  return { root, runtime, data, source, manifest, saveManifest, legacy, cleanup: () => rmSync(root, { recursive: true, force: true }) };
}

test('a UI-only version bump reuses the newest complete legacy engine in place', () => {
  const f = fixture();
  try {
    const old = f.legacy('0.1.0');
    utimesSync(path.join(old.engine, '.desktop-engine.json'), new Date(1000), new Date(1000));
    const previous = f.legacy('0.1.1');
    const receipt = readFileSync(path.join(previous.engine, '.desktop-engine.json'), 'utf8');
    const inode = statSync(previous.binary).ino;
    mkdirSync(path.join(previous.engine, 'biomaker/__pycache__'));
    writeFileSync(path.join(previous.engine, 'biomaker/__pycache__/run.cpython-312.pyc'), 'generated cache');
    mkdirSync(path.join(previous.engine, 'biomaker.egg-info'));
    writeFileSync(path.join(previous.engine, 'biomaker.egg-info/PKG-INFO'), 'generated metadata');
    writeFileSync(path.join(previous.engine, '.desktop-engine-operation.lock'), '');
    f.manifest.applicationVersion = '0.1.2'; f.saveManifest();
    const selection = selectEngine(f.runtime, f.data);
    assert.equal(selection.directory, previous.engine);
    assert.equal(selection.reused, true);
    prepareSelectedEngine(selection);
    assert.equal(readFileSync(path.join(previous.engine, '.desktop-engine.json'), 'utf8'), receipt);
    assert.equal(statSync(previous.binary).ino, inode);
    assert.ok(existsSync(old.engine));
  } finally { f.cleanup(); }
});

test('new source copies use a version-independent identity and do not claim dependency readiness', () => {
  const f = fixture();
  try {
    const initial = selectEngine(f.runtime, f.data);
    prepareSelectedEngine(initial);
    assert.match(path.basename(initial.directory), /^engine-[a-f0-9]{64}$/);
    assert.equal(existsSync(path.join(initial.directory, '.desktop-engine.json')), false);
    f.manifest.applicationVersion = '0.9.0'; f.saveManifest();
    const next = selectEngine(f.runtime, f.data);
    assert.equal(next.directory, initial.directory);
    assert.equal(next.reused, true);
  } finally { f.cleanup(); }
});

test('changed source creates a separate engine while a process keeps using the old one', async () => {
  const f = fixture();
  const previous = f.legacy();
  const child = spawn(process.execPath, ['-e', 'process.send("ready");setInterval(()=>{},1000)'], { cwd: previous.engine, stdio: ['ignore', 'ignore', 'ignore', 'ipc'] });
  try {
    await once(child, 'message');
    const original = readFileSync(path.join(previous.engine, 'biomaker/run.py'), 'utf8');
    writeFileSync(path.join(f.source, 'biomaker/run.py'), 'print("new engine source")\n'); f.saveManifest();
    const next = selectEngine(f.runtime, f.data);
    assert.notEqual(next.directory, previous.engine);
    prepareSelectedEngine(next);
    assert.equal(readFileSync(path.join(previous.engine, 'biomaker/run.py'), 'utf8'), original);
    assert.equal(readFileSync(path.join(next.directory, 'biomaker/run.py'), 'utf8'), 'print("new engine source")\n');
    assert.doesNotThrow(() => process.kill(child.pid!, 0));
    assert.ok(existsSync(path.join(previous.engine, '.venv/bin/python')) || existsSync(path.join(previous.engine, '.venv/Scripts/python.exe')));
  } finally { const exited = once(child, 'exit'); child.kill(); await exited; f.cleanup(); }
});

for (const tamper of ['changed source', 'stale dependencies', 'injected module', 'partial source']) test(`a matching directory prefix cannot hide ${tamper}`, () => {
  const f = fixture();
  try {
    const previous = f.legacy();
    if (tamper === 'changed source') writeFileSync(path.join(previous.engine, 'biomaker/run.py'), 'changed');
    if (tamper === 'stale dependencies') writeFileSync(path.join(previous.engine, 'uv.lock'), 'stale lock');
    if (tamper === 'injected module') writeFileSync(path.join(previous.engine, 'biomaker/unexpected.py'), 'new module');
    if (tamper === 'partial source') rmSync(path.join(previous.engine, 'requirements.txt'));
    const before = readdirSync(previous.engine);
    const next = selectEngine(f.runtime, f.data);
    assert.notEqual(next.directory, previous.engine);
    prepareSelectedEngine(next);
    assert.deepEqual(readdirSync(previous.engine), before);
  } finally { f.cleanup(); }
});

test('platform and exact Python build/version compatibility isolate environments', () => {
  for (const field of ['target', 'pythonVersion', 'pythonBuild'] as const) {
    const f = fixture();
    try {
      const previous = f.legacy();
      f.manifest[field] = field === 'target' ? (process.platform === 'darwin' ? 'linux-x64' : 'darwin-arm64') : field === 'pythonVersion' ? '3.12.13' : '20260301';
      f.saveManifest();
      const next = selectEngine(f.runtime, f.data);
      assert.notEqual(next.directory, previous.engine);
      assert.ok(existsSync(previous.binary));
    } finally { f.cleanup(); }
  }
});

test('stale install receipts are preserved for the service to repair, never promoted by selection', () => {
  const f = fixture();
  try {
    const previous = f.legacy();
    writeFileSync(path.join(previous.engine, '.desktop-engine.json'), '{"fingerprint":"obsolete-native-revision"}');
    const next = selectEngine(f.runtime, f.data);
    assert.equal(next.directory, previous.engine);
    prepareSelectedEngine(next);
    assert.equal(readFileSync(path.join(previous.engine, '.desktop-engine.json'), 'utf8'), '{"fingerprint":"obsolete-native-revision"}');
  } finally { f.cleanup(); }
});

test('tampered stable directories are preserved and replaced with a separate safe identity variant', () => {
  const f = fixture();
  try {
    const original = selectEngine(f.runtime, f.data); prepareSelectedEngine(original);
    writeFileSync(path.join(original.directory, 'biomaker/run.py'), 'locally modified');
    const next = selectEngine(f.runtime, f.data); prepareSelectedEngine(next);
    assert.notEqual(next.directory, original.directory);
    assert.equal(readFileSync(path.join(original.directory, 'biomaker/run.py'), 'utf8'), 'locally modified');
    assert.equal(selectEngine(f.runtime, f.data).directory, next.directory);
  } finally { f.cleanup(); }
});

test('invalid bundled checksum and post-selection source changes fail without deleting data', () => {
  const f = fixture();
  try {
    const previous = f.legacy();
    const selection = selectEngine(f.runtime, f.data);
    writeFileSync(path.join(previous.engine, 'biomaker/run.py'), 'changed during startup');
    assert.throws(() => prepareSelectedEngine(selection), /changed during startup/);
    assert.ok(existsSync(previous.binary));
    f.manifest.engineSha256 = '0'.repeat(64);
    writeFileSync(path.join(f.runtime, 'manifest.json'), JSON.stringify(f.manifest));
    assert.throws(() => selectEngine(f.runtime, f.data), /source checksum/);
  } finally { f.cleanup(); }
});

test('an interpreter changed after selection is not silently reused', () => {
  const f = fixture();
  try {
    const previous = f.legacy();
    const selection = selectEngine(f.runtime, f.data);
    writeFileSync(path.join(previous.engine, '.venv/pyvenv.cfg'), 'implementation = CPython\nversion_info = 3.13.0\ninclude-system-site-packages = false\n');
    assert.throws(() => prepareSelectedEngine(selection), /changed during startup/);
    assert.ok(existsSync(previous.binary));
    assert.notEqual(selectEngine(f.runtime, f.data).directory, previous.engine);
  } finally { f.cleanup(); }
});
