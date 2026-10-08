import { createHash, randomBytes } from 'node:crypto';
import { cpSync, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';

interface Compatibility { target: string; pythonVersion: string; pythonBuild: string }
interface EngineIdentity extends Compatibility { schema: 1; engineSha256: string; key: string }
interface Inventory { files: Map<string, string>; digest: string }
export interface EngineSelection {
  directory: string;
  source: string;
  identity: EngineIdentity;
  inventory: Inventory;
  runtimesDirectory: string;
  reused: boolean;
}
const ROOT_GENERATED = new Set(['.venv', 'biomaker.egg-info', '.desktop-engine-ready', '.desktop-engine.json', '.desktop-engine-identity.json']);
const IDENTITY_FILE = '.desktop-engine-identity.json';

function sha(value: string | Buffer) { return createHash('sha256').update(value).digest('hex'); }
function inside(candidate: string, root: string) {
  const relative = path.relative(root, candidate);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}
function regularFile(file: string) {
  try { return lstatSync(file).isFile(); } catch { return false; }
}
function readJSON(file: string): any {
  if (!regularFile(file) || statSync(file).size > 64 * 1024) throw new Error('Missing or invalid engine metadata');
  return JSON.parse(readFileSync(file, 'utf8'));
}

/** Match the runtime builder's complete source hash. Installed mutable state is
 * excluded only at known locations; unexpected source files reject reuse. */
function inventory(root: string, installed: boolean): Inventory {
  if (!lstatSync(root).isDirectory()) throw new Error('Engine source is not a directory');
  const files = new Map<string, string>();
  const digest = createHash('sha256');
  const visit = (directory: string) => {
    for (const item of readdirSync(directory, { withFileTypes: true }).sort((left, right) => left.name.localeCompare(right.name))) {
      if (installed && ((directory === root && ROOT_GENERATED.has(item.name)) || item.name === '__pycache__' || item.name === '.DS_Store')) continue;
      const file = path.join(directory, item.name);
      if (item.isSymbolicLink()) throw new Error('Engine source contains a symbolic link');
      if (item.isDirectory()) visit(file);
      else if (item.isFile()) {
        const relative = path.relative(root, file);
        const data = readFileSync(file);
        files.set(relative, sha(data));
        digest.update(relative); digest.update(data);
      } else throw new Error('Unsupported engine source entry');
    }
  };
  visit(root);
  return { files, digest: digest.digest('hex') };
}

function sourceMatches(directory: string, expected: Inventory) {
  try {
    const actual = inventory(directory, true);
    return actual.digest === expected.digest && actual.files.size === expected.files.size && [...expected.files].every(([name, hash]) => actual.files.get(name) === hash);
  } catch { return false; }
}

function compatibility(manifest: any): Compatibility {
  if (typeof manifest?.target !== 'string' || !/^(darwin|linux|win32)-(arm64|x64|ia32|armv7l)$/.test(manifest.target) || typeof manifest.pythonVersion !== 'string' || !/^3\.\d+\.\d+$/.test(manifest.pythonVersion) || typeof manifest.pythonBuild !== 'string' || !/^\d{8}$/.test(manifest.pythonBuild)) throw new Error('The portable runtime has invalid Python/platform provenance.');
  return { target: manifest.target, pythonVersion: manifest.pythonVersion, pythonBuild: manifest.pythonBuild };
}

function matchesCompatibility(actual: Compatibility, expected: Compatibility) {
  return actual.target === expected.target && actual.pythonVersion === expected.pythonVersion && actual.pythonBuild === expected.pythonBuild;
}

/** Check provenance without executing a candidate's Python or moving its venv. */
function interpreterCompatible(directory: string, runtimesDirectory: string, expected: Compatibility): boolean {
  try {
    const venv = path.join(directory, '.venv');
    if (!lstatSync(venv).isDirectory()) return false;
    const executable = path.join(venv, expected.target.startsWith('win32-') ? 'Scripts/python.exe' : 'bin/python');
    const resolved = realpathSync(executable);
    const runtimes = realpathSync(runtimesDirectory);
    if (!inside(resolved, runtimes)) return false;
    const parts = path.relative(runtimes, resolved).split(path.sep);
    if (parts.length < 3 || parts[1] !== 'python') return false;
    const runtime = path.join(runtimes, parts[0]);
    if (!regularFile(path.join(runtime, '.desktop-runtime-ready')) || !inside(resolved, realpathSync(path.join(runtime, 'python')))) return false;
    if (!matchesCompatibility(compatibility(readJSON(path.join(runtime, 'manifest.json'))), expected)) return false;
    const cfg = Object.fromEntries(readFileSync(path.join(venv, 'pyvenv.cfg'), 'utf8').split(/\r?\n/).filter(line => line.includes('=')).map(line => { const split = line.indexOf('='); return [line.slice(0, split).trim(), line.slice(split + 1).trim()]; }));
    return cfg.implementation === 'CPython' && cfg.version_info === expected.pythonVersion && cfg['include-system-site-packages'] === 'false';
  } catch { return false; }
}

function candidate(directory: string, identity: EngineIdentity, expected: Inventory, runtimes: string): { directory: string; installed: boolean; receiptTime: number } | null {
  try {
    if (!regularFile(path.join(directory, '.desktop-engine-ready')) || !sourceMatches(directory, expected)) return null;
    const metadata = path.join(directory, IDENTITY_FILE);
    const hasVenv = existsSync(path.join(directory, '.venv'));
    if (existsSync(metadata)) {
      const saved = readJSON(metadata);
      if (saved.schema !== 1 || saved.key !== identity.key || saved.engineSha256 !== identity.engineSha256 || !matchesCompatibility(compatibility(saved), identity)) return null;
      if (hasVenv && !interpreterCompatible(directory, runtimes, identity)) return null;
    } else if (!hasVenv || !interpreterCompatible(directory, runtimes, identity)) {
      // Legacy completion markers contain only an application version. A real
      // retained interpreter is needed to establish their compatibility.
      return null;
    }
    const receipt = path.join(directory, '.desktop-engine.json');
    const installed = hasVenv && regularFile(receipt);
    return { directory, installed, receiptTime: installed ? statSync(receipt).mtimeMs : 0 };
  } catch { return null; }
}

export function selectEngine(runtimeDirectory: string, dataDirectory: string): EngineSelection {
  const manifest = readJSON(path.join(runtimeDirectory, 'manifest.json'));
  const compatible = compatibility(manifest);
  if (!/^[a-f0-9]{64}$/.test(manifest.engineSha256 ?? '')) throw new Error('The portable runtime has no valid engine source checksum.');
  const source = path.join(runtimeDirectory, 'engine');
  const expected = inventory(source, false);
  if (expected.digest !== manifest.engineSha256) throw new Error('The bundled engine does not match its source checksum. Reinstall BioMaker.');
  const identity: EngineIdentity = { schema: 1, ...compatible, engineSha256: expected.digest, key: sha(JSON.stringify({ ...compatible, engineSha256: expected.digest })) };
  const engines = path.join(dataDirectory, 'engines');
  const runtimes = path.join(dataDirectory, 'runtimes');
  const candidates = existsSync(engines) ? readdirSync(engines, { withFileTypes: true }).filter(entry => entry.isDirectory() && !entry.name.includes('.staging-')).map(entry => candidate(path.join(engines, entry.name), identity, expected, runtimes)).filter((item): item is NonNullable<typeof item> => item !== null) : [];
  candidates.sort((left, right) => Number(right.installed) - Number(left.installed) || right.receiptTime - left.receiptTime || left.directory.localeCompare(right.directory));
  if (candidates.length) return { directory: candidates[0].directory, source, identity, inventory: expected, runtimesDirectory: runtimes, reused: true };
  const base = path.join(engines, `engine-${identity.key}`);
  // An altered or interrupted old directory is preserved for investigation or
  // a running experiment. Never replace it just to claim a deterministic name.
  const directory = existsSync(base) ? `${base}-${randomBytes(6).toString('hex')}` : base;
  return { directory, source, identity, inventory: expected, runtimesDirectory: runtimes, reused: false };
}

export function prepareSelectedEngine(selection: EngineSelection): void {
  if (existsSync(selection.directory)) {
    if (!candidate(selection.directory, selection.identity, selection.inventory, selection.runtimesDirectory)) throw new Error('The selected engine changed during startup. Restart BioMaker to create an isolated environment.');
    return;
  }
  if (selection.reused) throw new Error('The selected existing engine is no longer available. Restart BioMaker.');
  mkdirSync(path.dirname(selection.directory), { recursive: true, mode: 0o700 });
  const stage = `${selection.directory}.staging-${process.pid}-${randomBytes(6).toString('hex')}`;
  try {
    cpSync(selection.source, stage, { recursive: true, verbatimSymlinks: true });
    if (!sourceMatches(stage, selection.inventory)) throw new Error('The bundled engine changed during startup.');
    writeFileSync(path.join(stage, IDENTITY_FILE), JSON.stringify(selection.identity, null, 2), { mode: 0o600 });
    writeFileSync(path.join(stage, '.desktop-engine-ready'), 'source-copy-complete', { mode: 0o600 });
    if (existsSync(selection.directory)) throw new Error('Another engine appeared during startup. Restart BioMaker.');
    renameSync(stage, selection.directory);
  } catch (error) { rmSync(stage, { recursive: true, force: true }); throw error; }
  // These markers certify source copying only. Dependency installation and
  // native repairs remain exclusively governed by the Python install receipt.
}
