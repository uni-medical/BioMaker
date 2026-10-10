import sharp from 'sharp';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const build = path.join(root, 'build');
const svg = path.join(root, 'resources', 'icon.svg');
const iconset = path.join(build, 'icon.iconset');
await mkdir(iconset, { recursive: true });
await sharp(svg).resize(1024).png().toFile(path.join(build, 'icon.png'));
for (const size of [16, 32, 128, 256, 512]) {
  await sharp(svg).resize(size).png().toFile(path.join(iconset, `icon_${size}x${size}.png`));
  await sharp(svg).resize(size * 2).png().toFile(path.join(iconset, `icon_${size}x${size}@2x.png`));
}
if (process.platform === 'darwin') {
  execFileSync('iconutil', ['-c', 'icns', iconset, '-o', path.join(build, 'icon.icns')]);
}
// Windows ICO can contain a PNG image; no platform-specific converter is required.
const png = await sharp(svg).resize(256).png().toBuffer();
const header = Buffer.alloc(22);
header.writeUInt16LE(1, 2); header.writeUInt16LE(1, 4);
header.writeUInt16LE(1, 10); header.writeUInt16LE(32, 12);
header.writeUInt32LE(png.length, 14); header.writeUInt32LE(22, 18);
await writeFile(path.join(build, 'icon.ico'), Buffer.concat([header, png]));
console.log('BioMaker icons generated (PNG, ICO, macOS ICNS where supported).');
