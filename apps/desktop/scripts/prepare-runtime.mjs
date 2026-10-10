/** Build a self-contained sidecar runtime on the target OS/architecture.
 * Run on macOS arm64 for that release; Windows/Linux require native CI runners.
 * No system Python/Node is required on the recipient machine.
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { chmodSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const { validateEngineOverlay, validateRuntime, validateEngineSource, sourceFilter, digestTree, engineSourceEntries } = createRequire(import.meta.url)('./check-package.cjs');
const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(desktop, '../..');
const build = path.join(desktop, 'build');
const target = path.join(build, 'runtime');
const stage = path.join(build, `.runtime-stage-${process.pid}`);
const pythonVersion = '3.12.12';
const uvName = process.platform === 'win32' ? 'uv.exe' : 'uv';
const uvCommand = process.env.BIOMAKER_BUILD_UV || 'uv';
const engineOverlay = validateEngineOverlay(desktop);
function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: repo, encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${path.basename(command)} ${args[0]} failed (${result.status})`);
  return result.stdout?.trim() ?? '';
}
function sha(file) { return createHash('sha256').update(readFileSync(file)).digest('hex'); }
function findUV() {
  if (path.isAbsolute(uvCommand) || uvCommand.includes(path.sep)) return realpathSync(uvCommand);
  for (const directory of (process.env.PATH || '').split(path.delimiter)) {
    const candidate = path.join(directory, uvName); if (existsSync(candidate)) return realpathSync(candidate);
  }
  throw new Error('Build requires uv. Install https://docs.astral.sh/uv/getting-started/installation/ or set BIOMAKER_BUILD_UV.');
}
mkdirSync(build, { recursive: true });
mkdirSync(stage, { recursive: true });
try {
  const uvSource = findUV();
  const uvVersion = run(uvSource, ['--version']);
  console.log(`Preparing ${process.platform}/${process.arch}, CPython ${pythonVersion}, ${uvVersion}`);
  mkdirSync(path.join(stage, 'bin'));
  const uv = path.join(stage, 'bin', uvName); cpSync(uvSource, uv); if (process.platform !== 'win32') chmodSync(uv, 0o755);
  const installDir = path.join(stage, 'python-managed');
  const env = { ...process.env, UV_PYTHON_INSTALL_DIR: installDir, UV_PYTHON_BIN_DIR: path.join(stage, 'python-bin'), UV_NO_PROGRESS: '1' };
  run(uv, ['python', 'install', pythonVersion, '--no-bin'], { env, stdio: 'inherit' });
  const managed = readdirSync(installDir).find(name => name.startsWith(`cpython-${pythonVersion}-`));
  if (!managed) throw new Error('Managed CPython installation was not found.');
  cpSync(path.join(installDir, managed), path.join(stage, 'python'), { recursive: true, verbatimSymlinks: true });
  rmSync(installDir, { recursive: true, force: true }); rmSync(path.join(stage, 'python-bin'), { recursive: true, force: true });
  const python = process.platform === 'win32' ? path.join(stage, 'python', 'python.exe') : path.join(stage, 'python', 'bin', 'python3');
  const requirements = readFileSync(path.join(desktop, 'service/requirements.txt'), 'utf8');
  writeFileSync(path.join(stage, 'service-requirements.txt'), requirements + '\n');
  run(uv, ['pip', 'install', '--python', python, '--target', path.join(stage, 'site-packages'), '--require-hashes', '-r', path.join(stage, 'service-requirements.txt')], { stdio: 'inherit' });
  const engine = path.join(stage, 'engine'); mkdirSync(engine);
  // Explicit source allowlist avoids packaging credentials, experiments or datasets.
  // The generated engine manifests were validated byte-for-byte against the
  // unified root manifests above; source copying and package checks share one list.
  for (const item of engineSourceEntries) {
    mkdirSync(path.dirname(path.join(engine, item)), { recursive: true });
    cpSync(path.join(repo, item), path.join(engine, item), { recursive: true, filter: sourceFilter });
  }
  const version = JSON.parse(readFileSync(path.join(desktop, 'package.json'), 'utf8')).version;
  const pythonBuild = readFileSync(path.join(stage, 'python/BUILD'), 'utf8').trim();
  if (!/^\d{8}$/.test(pythonBuild)) throw new Error('Python standalone build tag is missing or invalid.');
  const manifest = { schema: 1, applicationVersion: version, builtAt: new Date().toISOString(), target: `${process.platform}-${process.arch}`, pythonVersion, pythonBuild, pythonDistribution: 'python-build-standalone, installed by uv', uvVersion, uvSha256: sha(uv), serviceLockSha256: sha(path.join(desktop, 'service/requirements.txt')), ...engineOverlay, engineSha256: digestTree(engine), sourceCommit: run('git', ['rev-parse', 'HEAD']), sourceFilesMayIncludeUncommittedChanges: Boolean(run('git', ['status', '--porcelain', '--untracked-files=normal'])) };
  writeFileSync(path.join(stage, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  const licenseDir = path.join(stage, 'licenses'); mkdirSync(licenseDir);
  cpSync(path.join(repo, 'LICENSE'), path.join(licenseDir, 'BioMaker-LICENSE.txt'));
  // uv is dual MIT/Apache-2.0. Fetch exact-version notices, never execute fetched text.
  const versionMatch = uvVersion.match(/^uv ([0-9.]+)/)?.[1];
  if (!versionMatch) throw new Error('Cannot determine uv version for its license notices.');
  for (const name of ['LICENSE-MIT', 'LICENSE-APACHE']) {
    const response = await fetch(`https://raw.githubusercontent.com/astral-sh/uv/${versionMatch}/${name}`, { signal: AbortSignal.timeout(30000) });
    if (!response.ok) throw new Error(`Unable to retrieve uv ${name} (${response.status})`);
    writeFileSync(path.join(licenseDir, `uv-${name}.txt`), await response.text());
  }
  // install_only Python distributions omit some dependency notices. Retain the
  // matching python-build-standalone release's complete license collection.
  // A single source archive avoids unauthenticated GitHub API rate limits.
  // Read only bounded, regular LICENSE files; never extract or execute the archive.
  const archiveResponse = await fetch(`https://codeload.github.com/astral-sh/python-build-standalone/tar.gz/refs/tags/${pythonBuild}`, { signal: AbortSignal.timeout(60000) });
  if (!archiveResponse.ok || !archiveResponse.body) throw new Error(`Cannot retrieve Python native dependency notices (${archiveResponse.status})`);
  const chunks = []; let archiveBytes = 0;
  for await (const chunk of archiveResponse.body) {
    archiveBytes += chunk.length;
    if (archiveBytes > 32 * 1024 * 1024) throw new Error('Unexpected Python notice archive size.');
    chunks.push(chunk);
  }
  const archive = path.join(stage, 'python-source-notices.tar.gz');
  writeFileSync(archive, Buffer.concat(chunks));
  const pythonNotices = path.join(licenseDir, 'python-build-standalone'); mkdirSync(pythonNotices);
  const extractNotices = `import pathlib,re,sys,tarfile\nroot=pathlib.Path(sys.argv[2]); count=0\nwith tarfile.open(sys.argv[1], 'r|gz') as archive:\n for entry in archive:\n  parts=entry.name.split('/')\n  if len(parts)!=2 or not re.fullmatch(r'LICENSE(?:\\.[A-Za-z0-9.-]+)?',parts[1]): continue\n  if not entry.isfile() or entry.size>1024*1024: raise ValueError('Invalid license entry')\n  count+=1\n  if count>64: raise ValueError('Too many license entries')\n  target=root/parts[1]\n  with target.open('xb') as output: output.write(archive.extractfile(entry).read())\nif not count: raise ValueError('No Python license notices found')\nprint(f'Retained {count} Python native dependency notices')`;
  run(python, ['-c', extractNotices, archive, pythonNotices], { stdio: 'inherit' });
  manifest.pythonNoticeArchiveSha256 = sha(archive);
  writeFileSync(path.join(stage, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  rmSync(archive);
  const metadataScript = `import importlib.metadata as m,json,sys\nitems=[]\nfor d in sorted(m.distributions(path=[sys.argv[1]]),key=lambda x:x.metadata['Name'].lower()):\n items.append({'name':d.metadata['Name'],'version':d.version,'license':d.metadata.get('License-Expression') or d.metadata.get('License','See distribution metadata and license files'),'homepage':d.metadata.get('Home-page',''),'notice_directory':str(d._path).replace(sys.argv[1],'site-packages')})\nprint(json.dumps(items,indent=2))`;
  writeFileSync(path.join(licenseDir, 'python-packages.json'), run(python, ['-c', metadataScript, path.join(stage, 'site-packages')]) + '\n');
  writeFileSync(path.join(stage, 'THIRD_PARTY_NOTICES.md'), `# BioMaker desktop portable runtime\n\nTarget: ${manifest.target}. CPython ${pythonVersion} from python-build-standalone via uv. uv: ${uvVersion}.\n\nBioMaker is MIT licensed (licenses/BioMaker-LICENSE.txt). uv is dual MIT/Apache-2.0 (licenses/uv-LICENSE-*). CPython's license is retained in python/lib/python3.12/LICENSE.txt; the complete matching python-build-standalone native dependency notice collection is in licenses/python-build-standalone/. Python service dependency metadata and notices remain in site-packages/*.dist-info; licenses/python-packages.json indexes them. Electron, Chromium, and Node notices are included by Electron in the application bundle.\n\nThe service requirements are pinned with hashes in apps/desktop/service/requirements.txt. manifest.json records source hashes and runtime provenance. The engine source snapshot intentionally excludes .env, virtual environments, datasets, logs, and experiments. ML dependencies are downloaded into the user's application support directory only after the user chooses to install the engine. Their dependency licenses accompany that installation.\n\nNo code-signing or notarization claim is made by this preparation step. A public release requires release-owner signing credentials and notarization. Windows and Linux must be built and verified on their native runners.\n`);
  run(python, ['-c', 'import fastapi,uvicorn,psutil,dotenv;print("Portable service imports OK")'], { env: { ...process.env, PYTHONPATH: path.join(stage, 'site-packages'), PYTHONNOUSERSITE: '1' }, stdio: 'inherit' });
  validateRuntime(stage, process.platform, process.arch, path.join(desktop, 'service/requirements.txt'), engineOverlay);
  validateEngineSource(stage, repo);
  // Replace only a previously generated runtime after the complete new one passes.
  const previous = `${target}.previous`;
  rmSync(previous, { recursive: true, force: true });
  if (existsSync(target)) renameSync(target, previous);
  try { renameSync(stage, target); } catch (error) { if (existsSync(previous)) renameSync(previous, target); throw error; }
  rmSync(previous, { recursive: true, force: true });
  console.log(`Portable runtime ready: ${target}`);
} catch (error) {
  rmSync(stage, { recursive: true, force: true }); throw error;
}
