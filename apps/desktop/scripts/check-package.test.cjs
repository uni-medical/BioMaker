const test = require('node:test');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { validateRuntime, validateEngineOverlay } = require('./check-package.cjs');

test('package gate rejects architecture mismatch, stale lock, private files and escaping symlinks', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'biomaker-package-gate-'));
  try {
    for (const item of ['python/bin', 'bin', 'site-packages/fastapi', 'engine']) mkdirSync(path.join(root, item), { recursive: true });
    for (const item of ['python/bin/python3', 'bin/uv', 'site-packages/fastapi/__init__.py', 'engine/pyproject.toml']) writeFileSync(path.join(root, item), '');
    const lock = path.join(root, 'service-requirements.txt'); writeFileSync(lock, 'fastapi==1\n');
    writeFileSync(path.join(root, 'manifest.json'), JSON.stringify({ target: 'darwin-arm64', serviceLockSha256: createHash('sha256').update('fastapi==1\n').digest('hex') }));
    validateRuntime(root, 'darwin', 'arm64', lock);
    assert.throws(() => validateRuntime(root, 'darwin', 'x64', lock), /package target/);
    assert.throws(() => validateRuntime(root, 'win32', 'arm64', lock), /package target/);
    writeFileSync(lock, 'fastapi==2\n');
    assert.throws(() => validateRuntime(root, 'darwin', 'arm64', lock), /lock changed/);
    writeFileSync(path.join(root, 'engine/.env'), 'PRIVATE');
    assert.throws(() => validateRuntime(root, 'darwin', 'arm64'), /Unexpected bundled engine entry|Private/);
    rmSync(path.join(root, 'engine/.env'));
    symlinkSync(os.tmpdir(), path.join(root, 'python/outside'));
    assert.throws(() => validateRuntime(root, 'darwin', 'arm64'), /symlink/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('desktop engine overlay is pinned to its source manifests', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'biomaker-overlay-gate-'));
  try {
    const project = path.join(root, 'apps/desktop');
    mkdirSync(path.join(project, 'engine'), { recursive: true });
    assert.throws(() => validateEngineOverlay(project), /incomplete/);
    writeFileSync(path.join(root, 'requirements.txt'), 'base');
    writeFileSync(path.join(root, 'pyproject.toml'), 'project');
    writeFileSync(path.join(project, 'engine/requirements.txt'), 'desktop');
    writeFileSync(path.join(project, 'engine/uv.lock'), 'lock');
    writeFileSync(path.join(project, 'engine/source.json'), JSON.stringify({ baseRequirementsSha256: createHash('sha256').update('base').digest('hex'), basePyprojectSha256: createHash('sha256').update('project').digest('hex') }));
    assert.equal(validateEngineOverlay(project).desktopEngineLockSha256, createHash('sha256').update('lock').digest('hex'));
    writeFileSync(path.join(root, 'requirements.txt'), 'changed');
    assert.throws(() => validateEngineOverlay(project), /Regenerate and audit/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
