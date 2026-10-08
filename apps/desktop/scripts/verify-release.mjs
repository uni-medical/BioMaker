// Verify the actual distributable bytes after packaged acceptance, without
// executing Python from a signed bundle or modifying any application files.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { lstat, mkdir, mkdtemp, readFile, readdir, readlink, realpath, rm, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const desktop = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const version = JSON.parse(await readFile(path.join(desktop, 'package.json'), 'utf8')).version;
assert.equal(process.platform, 'darwin', 'macOS release verification requires a macOS host');
const release = path.join(desktop, 'release');
const output = path.join(desktop, 'output');
const bundle = path.join(release, 'mac-arm64/BioMaker.app');
const dmg = path.join(release, `BioMaker-${version}-mac-arm64.dmg`);
const zip = path.join(release, `BioMaker-${version}-mac-arm64.zip`);
const run = (command, args) => execFileSync(command, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 120_000 });
async function sha(file) {
  const digest = createHash('sha256');
  for await (const bytes of createReadStream(file)) digest.update(bytes);
  return digest.digest('hex');
}
async function fingerprint(directory) {
  const digest = createHash('sha256');
  let count = 0;
  async function visit(parent) {
    for (const name of (await readdir(parent)).sort()) {
      const file = path.join(parent, name);
      const relative = path.relative(directory, file);
      const info = await lstat(file);
      if (info.isSymbolicLink()) digest.update(JSON.stringify([relative, 'link', await readlink(file)]));
      else if (info.isDirectory()) { digest.update(JSON.stringify([relative, 'directory'])); await visit(file); }
      else if (info.isFile()) { digest.update(JSON.stringify([relative, 'file', await sha(file)])); count++; }
      else throw new Error(`Unexpected application entry: ${relative}`);
    }
  }
  await visit(directory);
  return { files: count, sha256: digest.digest('hex') };
}
async function verifyBundle(directory) {
  run('/usr/bin/codesign', ['--verify', '--deep', '--strict', directory]);
  const info = JSON.parse(run('/usr/bin/plutil', ['-convert', 'json', '-o', '-', path.join(directory, 'Contents/Info.plist')]));
  assert.equal(info.CFBundleShortVersionString, version);
  const manifest = JSON.parse(await readFile(path.join(directory, 'Contents/Resources/runtime/manifest.json'), 'utf8'));
  assert.equal(manifest.applicationVersion, version);
  assert.equal(manifest.target, 'darwin-arm64');
  assert.equal(manifest.sourceFilesMayIncludeUncommittedChanges, false, 'A release must record a clean source checkout');
  return { manifest, fingerprint: await fingerprint(directory) };
}

const original = await verifyBundle(bundle);
const temporary = await mkdtemp(path.join(os.tmpdir(), `biomaker-release-${version}-`));
const mount = path.join(temporary, 'mounted');
await mkdir(mount);
let mounted = false;
try {
  run('/usr/bin/hdiutil', ['attach', '-readonly', '-nobrowse', '-mountpoint', mount, dmg]);
  mounted = true;
  assert((await lstat(path.join(mount, 'Applications'))).isSymbolicLink());
  assert.equal(await realpath(path.join(mount, 'Applications')), '/Applications');
  const disk = await verifyBundle(path.join(mount, 'BioMaker.app'));
  assert.deepEqual(disk.fingerprint, original.fingerprint, 'DMG application differs from the tested unpacked application');
  run('/usr/bin/hdiutil', ['detach', mount]);
  mounted = false;
  const unpackedZip = path.join(temporary, 'zip');
  await mkdir(unpackedZip);
  run('/usr/bin/ditto', ['-x', '-k', zip, unpackedZip]);
  const zipped = await verifyBundle(path.join(unpackedZip, 'BioMaker.app'));
  assert.deepEqual(zipped.fingerprint, original.fingerprint, 'ZIP application differs from the tested unpacked application');
  const checksums = { [path.basename(dmg)]: await sha(dmg), [path.basename(zip)]: await sha(zip) };
  const receipt = {
    version, sourceCommit: original.manifest.sourceCommit, dirtySource: false,
    dmg_mounted_readonly: true, applications_shortcut: true,
    complete_bundle_matches_dmg: true, complete_bundle_matches_zip: true,
    bundle_fingerprint: original.fingerprint,
    signature: 'Deep strict verification passed on unpacked, DMG and ZIP applications; signing identity and notarization require separate verification',
    sha256: checksums,
  };
  await mkdir(output, { recursive: true });
  const report = path.join(output, `package-verification-${version}.json`);
  await writeFile(report, JSON.stringify(receipt, null, 2) + '\n');
  await writeFile(path.join(release, `SHA256SUMS-${version}.txt`), Object.entries(checksums).map(([name, hash]) => `${hash}  ${name}\n`).join(''));
  console.log(JSON.stringify({ passed: true, version, report, sha256: checksums }, null, 2));
} finally {
  if (mounted) run('/usr/bin/hdiutil', ['detach', mount]);
  await rm(temporary, { recursive: true, force: true });
}
