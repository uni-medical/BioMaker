import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(desktop, '../..');
const overlay = path.join(desktop, 'engine');
mkdirSync(path.join(desktop, 'build'), { recursive: true });
const stage = mkdtempSync(path.join(desktop, 'build', 'engine-lock-'));
try {
  for (const name of ['pyproject.toml', 'README.md', 'LICENSE']) cpSync(path.join(repo, name), path.join(stage, name));
  cpSync(path.join(repo, 'biomaker'), path.join(stage, 'biomaker'), { recursive: true, filter: source => !['__pycache__', '.env'].includes(path.basename(source)) && !source.endsWith('.pyc') });
  cpSync(path.join(overlay, 'requirements.txt'), path.join(stage, 'requirements.txt'));
  if (existsSync(path.join(overlay, 'uv.lock'))) cpSync(path.join(overlay, 'uv.lock'), path.join(stage, 'uv.lock'));
  const result = spawnSync(process.env.BIOMAKER_BUILD_UV || 'uv', ['lock', '--project', stage, '--python', '3.12', ...(process.argv.includes('--upgrade') ? ['--upgrade'] : [])], { stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error('Desktop engine lock could not be resolved');
  cpSync(path.join(stage, 'uv.lock'), path.join(overlay, 'uv.lock'));
  const sha = name => createHash('sha256').update(readFileSync(path.join(repo, name))).digest('hex');
  writeFileSync(path.join(overlay, 'source.json'), JSON.stringify({ baseRequirementsSha256: sha('requirements.txt'), basePyprojectSha256: sha('pyproject.toml'), purpose: 'Desktop distribution dependency lock; historical research environment remains unchanged.' }, null, 2) + '\n');
  console.log('Desktop engine lock updated. Run real installation/import acceptance before release.');
} finally { rmSync(stage, { recursive: true, force: true }); }
