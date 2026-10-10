import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
if (!existsSync(path.join(root, 'build/runtime/manifest.json'))) {
  console.error('Prepare the portable runtime first: npm run prepare:runtime'); process.exit(1);
}
const build = spawnSync(process.execPath, ['scripts/build-electron.mjs'], { cwd: root, stdio: 'inherit' });
if (build.status !== 0) process.exit(build.status ?? 1);
const vite = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5173', '--strictPort'], { cwd: root, stdio: 'inherit' });
let electron;
let stopping = false;
const stop = () => { if (stopping) return; stopping = true; electron?.kill('SIGTERM'); vite.kill('SIGTERM'); };
process.once('SIGINT', stop); process.once('SIGTERM', stop);
vite.once('exit', () => { if (!stopping) { stop(); process.exitCode = 1; } });
for (let attempt = 0; attempt < 100 && !stopping; attempt++) {
  try { if ((await fetch('http://127.0.0.1:5173', { signal: AbortSignal.timeout(500) })).ok) break; } catch { /* server starting */ }
  if (attempt === 99) { console.error('Vite did not start.'); stop(); process.exitCode = 1; }
  await new Promise(resolve => setTimeout(resolve, 150));
}
if (!stopping) {
  const { default: executable } = await import('electron');
  const environment = { ...process.env, BIOMAKER_DESKTOP_DEV_URL: 'http://127.0.0.1:5173' };
  delete environment.ELECTRON_RUN_AS_NODE;
  electron = spawn(executable, ['.'], { cwd: root, stdio: 'inherit', env: environment });
  electron.once('exit', code => { stop(); process.exitCode = code ?? 0; });
}
