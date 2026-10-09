// These standalone callbacks run in Electron's main process. Clipboard payloads
// never cross Playwright or reach a receipt; only format/size/hash metadata does.
export async function captureClipboard({ clipboard }) {
  if (globalThis.__nativeClipboardQA) throw new Error('An original clipboard snapshot already exists');
  const saved = [];
  const crypto = process.getBuiltinModule('node:crypto');
  const metadata = [];
  for (const item of await clipboard.read()) {
    const payloads = {}, types = [];
    for (const type of item.types) {
      const payload = await item.getType(type);
      payloads[type] = payload;
      const bytes = type === 'electron application/bookmark'
        ? Buffer.from(JSON.stringify(payload)) : Buffer.from(await payload.arrayBuffer());
      types.push({ type, bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') });
    }
    saved.push(payloads);
    metadata.push({ types });
  }
  globalThis.__nativeClipboardQA = { saved, originalMetadata: metadata, requests: [], forbidden: [] };
  return { itemCount: metadata.length, items: metadata };
}

export async function restoreClipboard({ clipboard, ClipboardItem }) {
  const qa = globalThis.__nativeClipboardQA;
  if (!qa) return { restored: true, untouched: true, requests: [], forbidden: [] };
  const crypto = process.getBuiltinModule('node:crypto');
  // Electron's high-level text/html writer can reserialize native HTML. Supplying
  // both the MIME alias and its raw native format can therefore change the bytes.
  // macOS exposes the original native formats: atomically write only those, then
  // verify every original raw format AND high-level alias, byte for byte below.
  const mode = process.platform === 'darwin' ? 'raw' : 'all';
  let writeError;
  try {
    const items = qa.saved.map(payloads => {
      const raw = Object.fromEntries(Object.entries(payloads).filter(([type]) => type.startsWith('electron application/osclipboard;')));
      if (mode === 'raw' && !Object.keys(raw).length) throw new Error('Original item has no raw native formats; refusing incomplete restoration');
      return new ClipboardItem(mode === 'raw' ? raw : payloads);
    });
    if (items.length) await clipboard.write(items); else clipboard.clear();
  } catch (error) { writeError = String(error); }
  const after = [];
  let readError;
  try {
    for (const item of await clipboard.read()) {
      const types = [];
      for (const type of item.types) {
        const payload = await item.getType(type);
        const bytes = type === 'electron application/bookmark'
          ? Buffer.from(JSON.stringify(payload)) : Buffer.from(await payload.arrayBuffer());
        types.push({ type, bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') });
      }
      after.push({ types });
    }
  } catch (error) { readError = String(error); }
  const before = qa.originalMetadata;
  // Exact item order, type order, sizes and bytes. No semantic normalization.
  const restored = !writeError && !readError && JSON.stringify(before) === JSON.stringify(after);
  const result = {
    restored, snapshotRetained: true, snapshotOwnerPid: process.pid, mode,
    diagnostics: { before: { itemCount: before.length, items: before }, after: { itemCount: after.length, items: after }, ...(writeError ? { writeError } : {}), ...(readError ? { readError } : {}) },
    requests: [...qa.requests], forbidden: [...qa.forbidden],
  };
  qa.lastRestoration = result;
  return result;
}
