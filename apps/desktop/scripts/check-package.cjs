const { createHash } = require('node:crypto');
const { existsSync, lstatSync, readFileSync, readdirSync, realpathSync } = require('node:fs');
const path = require('node:path');

function inside(candidate, root) {
  const relative = path.relative(root, candidate);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}
function sha(file) { return createHash('sha256').update(readFileSync(file)).digest('hex'); }
const engineSourceEntries = ['biomaker', 'scripts', 'pyproject.toml', 'requirements.txt', 'uv.lock', 'README.md', 'LICENSE', 'apps/webui/webui', 'apps/webui/pyproject.toml', 'apps/webui/uv.lock'];
function sourceFilter(file) {
  const base = path.basename(file);
  return !['__pycache__', '.git', '.venv', '.DS_Store', 'node_modules', 'tests'].includes(base) && !base.startsWith('.env') && !base.endsWith('.pyc') && !base.endsWith('.pyo') && !lstatSync(file).isSymbolicLink();
}
function treeFiles(directory, filter = () => true) {
  const files = new Map();
  function visit(file) {
    if (!filter(file)) return;
    if (lstatSync(file).isDirectory()) {
      for (const entry of readdirSync(file).sort((a, b) => a.localeCompare(b))) visit(path.join(file, entry));
    } else files.set(path.relative(directory, file), file);
  }
  visit(directory);
  return files;
}
function digestTree(directory) {
  const hash = createHash('sha256');
  for (const [relative, file] of treeFiles(directory)) { hash.update(relative); hash.update(readFileSync(file)); }
  return hash.digest('hex');
}
function validateEngineSource(runtime, repo) {
  const expected = new Map();
  for (const entry of engineSourceEntries) {
    const source = path.join(repo, entry);
    if (!existsSync(source)) throw new Error(`Engine source is incomplete: ${entry}`);
    for (const [, file] of treeFiles(source, sourceFilter)) expected.set(path.relative(repo, file), file);
  }
  const actual = treeFiles(path.join(runtime, 'engine'));
  for (const [relative, source] of expected) {
    if (!actual.has(relative) || sha(actual.get(relative)) !== sha(source)) throw new Error(`Bundled engine source differs from the current repository: ${relative}. Run npm run prepare:runtime again before packaging.`);
  }
  for (const relative of actual.keys()) if (!expected.has(relative)) throw new Error(`Bundled engine source contains an obsolete or unexpected file: ${relative}. Run npm run prepare:runtime again before packaging.`);
  return { files: expected.size, engineSha256: digestTree(path.join(runtime, 'engine')) };
}
function validateEngineOverlay(project) {
  const repo = path.resolve(project, '../..');
  const overlay = path.join(project, 'engine');
  for (const name of ['requirements.txt', 'uv.lock', 'source.json']) if (!existsSync(path.join(overlay, name))) throw new Error(`Desktop engine dependency overlay is incomplete: engine/${name}`);
  const source = JSON.parse(readFileSync(path.join(overlay, 'source.json'), 'utf8'));
  const baseRequirementsSha256 = sha(path.join(repo, 'requirements.txt'));
  const basePyprojectSha256 = sha(path.join(repo, 'pyproject.toml'));
  if (source.baseRequirementsSha256 !== baseRequirementsSha256 || source.basePyprojectSha256 !== basePyprojectSha256) throw new Error('The core dependency manifest changed. Regenerate and audit the desktop engine overlay before packaging.');
  if (sha(path.join(overlay, 'requirements.txt')) !== baseRequirementsSha256 || sha(path.join(overlay, 'uv.lock')) !== sha(path.join(repo, 'uv.lock'))) throw new Error('Desktop dependencies differ from the core. Regenerate and audit the desktop engine overlay before packaging.');
  return { baseRequirementsSha256, basePyprojectSha256, desktopEngineLockSha256: sha(path.join(overlay, 'uv.lock')), desktopEngineRequirementsSha256: sha(path.join(overlay, 'requirements.txt')) };
}
function validateRuntime(runtime, platform, architecture, requirements, overlay, applicationVersion) {
  const manifest = JSON.parse(readFileSync(path.join(runtime, 'manifest.json'), 'utf8'));
  if (applicationVersion !== undefined && manifest.applicationVersion !== applicationVersion) throw new Error(`Portable runtime application version is ${manifest.applicationVersion ?? 'missing'}, but package version is ${applicationVersion}. Run npm run prepare:runtime again before packaging.`);
  const target = `${platform}-${architecture}`;
  if (manifest.target !== target) throw new Error(`Portable runtime is ${manifest.target}, but package target is ${target}. Run prepare:runtime on a ${target} build host; cross-platform runtime reuse is not supported.`);
  const executable = platform === 'win32' ? 'python/python.exe' : 'python/bin/python3';
  for (const item of [executable, platform === 'win32' ? 'bin/uv.exe' : 'bin/uv', 'site-packages/fastapi/__init__.py', 'engine/pyproject.toml']) {
    if (!existsSync(path.join(runtime, item))) throw new Error(`Portable runtime is incomplete: ${item}`);
  }
  if (requirements) {
    const digest = createHash('sha256').update(readFileSync(requirements)).digest('hex');
    if (manifest.serviceLockSha256 !== digest) throw new Error('Service dependency lock changed after runtime preparation. Run npm run prepare:runtime again.');
  }
  if (overlay) {
    for (const key of Object.keys(overlay)) if (manifest[key] !== overlay[key]) throw new Error(`Desktop engine overlay changed after runtime preparation (${key}). Run npm run prepare:runtime again.`);
    if (sha(path.join(runtime, 'engine/uv.lock')) !== overlay.desktopEngineLockSha256 || sha(path.join(runtime, 'engine/requirements.txt')) !== overlay.desktopEngineRequirementsSha256 || sha(path.join(runtime, 'engine/pyproject.toml')) !== overlay.basePyprojectSha256) throw new Error('Bundled engine dependency files do not match the audited desktop overlay.');
  }
  const allowedEngine = new Set(['biomaker', 'scripts', 'pyproject.toml', 'requirements.txt', 'uv.lock', 'README.md', 'LICENSE', 'apps']);
  for (const item of readdirSync(path.join(runtime, 'engine'))) if (!allowedEngine.has(item)) throw new Error(`Unexpected bundled engine entry: ${item}`);
  const forbidden = new Set(['.env', '.git', '.venv', '.DS_Store', 'node_modules', 'workspaces', 'datasets', 'experiments', 'logs']);
  const root = realpathSync(runtime);
  function visit(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      if (entry.name === '.env' || entry.name.startsWith('.env.') || (inside(file, path.join(root, 'engine')) && forbidden.has(entry.name))) throw new Error(`Private/generated content cannot be packaged: ${path.relative(root, file)}`);
      if (entry.isSymbolicLink()) {
        if (!existsSync(file) || !inside(realpathSync(file), root)) throw new Error(`Runtime symlink is broken or non-portable: ${path.relative(root, file)}`);
      } else if (entry.isDirectory()) visit(file);
    }
  }
  visit(root);
  if (manifest.engineSha256 !== digestTree(path.join(runtime, 'engine'))) throw new Error('Bundled engine source checksum does not match its runtime manifest. Run npm run prepare:runtime again before packaging.');
  return manifest;
}
module.exports = async function beforePack(context) {
  const arch = typeof context.arch === 'string' ? context.arch : ['ia32', 'x64', 'armv7l', 'arm64', 'universal'][context.arch];
  const project = context.packager.projectDir;
  const { version } = JSON.parse(readFileSync(path.join(project, 'package.json'), 'utf8'));
  if (typeof version !== 'string' || !version.trim()) throw new Error('Application package.json must declare a non-empty version before packaging.');
  const overlay = validateEngineOverlay(project);
  validateRuntime(path.join(project, 'build/runtime'), context.electronPlatformName, arch, path.join(project, 'service/requirements.txt'), overlay, version);
  validateEngineSource(path.join(project, 'build/runtime'), path.resolve(project, '../..'));
};
module.exports.validateRuntime = validateRuntime;
module.exports.validateEngineOverlay = validateEngineOverlay;
module.exports.validateEngineSource = validateEngineSource;
module.exports.sourceFilter = sourceFilter;
module.exports.digestTree = digestTree;
module.exports.engineSourceEntries = engineSourceEntries;
