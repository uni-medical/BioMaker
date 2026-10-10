import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { cpSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(desktop, '../..');
const overlay = path.join(desktop, 'engine');
const result = spawnSync(process.env.BIOMAKER_BUILD_UV || 'uv', ['lock', '--project', repo, ...(process.argv.includes('--upgrade') ? ['--upgrade'] : ['--check'])], { stdio: 'inherit' });
if (result.error) throw result.error;
if (result.status !== 0) throw new Error('Resolve the root dependency lock before preparing Desktop');
mkdirSync(overlay, { recursive: true });
for (const name of ['requirements.txt', 'uv.lock']) cpSync(path.join(repo, name), path.join(overlay, name));
const sha = name => createHash('sha256').update(readFileSync(path.join(repo, name))).digest('hex');
writeFileSync(path.join(overlay, 'source.json'), JSON.stringify({ baseRequirementsSha256: sha('requirements.txt'), basePyprojectSha256: sha('pyproject.toml'), purpose: 'Generated distribution copies of the unified core dependencies and universal lock.' }, null, 2) + '\n');
console.log('Desktop engine now uses the root requirements and universal lock.');
