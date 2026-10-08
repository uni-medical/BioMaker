const test = require('node:test');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const beforePack = require('./check-package.cjs');
const { validateRuntime, validateEngineOverlay } = beforePack;

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

test('beforePack requires a runtime prepared for the current application version', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'biomaker-version-gate-'));
  try {
    const project = path.join(root, 'apps/desktop');
    const runtime = path.join(project, 'build/runtime');
    const digest = (value) => createHash('sha256').update(value).digest('hex');
    for (const item of ['engine', 'service', 'build/runtime/python/bin', 'build/runtime/bin', 'build/runtime/site-packages/fastapi', 'build/runtime/engine']) mkdirSync(path.join(project, item), { recursive: true });
    writeFileSync(path.join(root, 'requirements.txt'), 'base');
    writeFileSync(path.join(root, 'pyproject.toml'), 'project');
    writeFileSync(path.join(project, 'engine/requirements.txt'), 'desktop');
    writeFileSync(path.join(project, 'engine/uv.lock'), 'lock');
    writeFileSync(path.join(project, 'engine/source.json'), JSON.stringify({ baseRequirementsSha256: digest('base'), basePyprojectSha256: digest('project') }));
    writeFileSync(path.join(project, 'service/requirements.txt'), 'service');
    writeFileSync(path.join(runtime, 'engine/requirements.txt'), 'desktop');
    writeFileSync(path.join(runtime, 'engine/uv.lock'), 'lock');
    writeFileSync(path.join(runtime, 'engine/pyproject.toml'), 'project');
    for (const item of ['python/bin/python3', 'bin/uv', 'site-packages/fastapi/__init__.py']) writeFileSync(path.join(runtime, item), '');
    const manifest = { target: 'darwin-arm64', applicationVersion: '0.1.2', serviceLockSha256: digest('service'), ...validateEngineOverlay(project) };
    writeFileSync(path.join(runtime, 'manifest.json'), JSON.stringify(manifest));
    writeFileSync(path.join(project, 'package.json'), JSON.stringify({ version: '0.1.2' }));
    const context = { packager: { projectDir: project }, electronPlatformName: 'darwin', arch: 3 };
    await assert.doesNotReject(beforePack(context));

    writeFileSync(path.join(project, 'package.json'), JSON.stringify({ version: '0.1.3' }));
    await assert.rejects(beforePack(context), /runtime application version is 0\.1\.2, but package version is 0\.1\.3\. Run npm run prepare:runtime again before packaging/);

    delete manifest.applicationVersion;
    writeFileSync(path.join(runtime, 'manifest.json'), JSON.stringify(manifest));
    await assert.rejects(beforePack(context), /runtime application version is missing.*npm run prepare:runtime/);

    manifest.applicationVersion = '0.1.3';
    writeFileSync(path.join(runtime, 'manifest.json'), JSON.stringify(manifest));
    await assert.doesNotReject(beforePack(context));

    writeFileSync(path.join(project, 'package.json'), '{}');
    await assert.rejects(beforePack(context), /package\.json must declare a non-empty version/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
