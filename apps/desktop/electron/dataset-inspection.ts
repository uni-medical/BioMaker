import path from 'node:path';

export const MAX_DATASET_PATH_BYTES = 4096;
export const MAX_DATASET_ROOTS = 128;
export const MAX_DATASET_INSPECTION_BYTES = 512 * 1024;

function absoluteMetadataPath(value: unknown): value is string {
  return typeof value === 'string' && path.isAbsolute(value) &&
    !/[\x00-\x1f\x7f]/.test(value) && Buffer.byteLength(value) <= MAX_DATASET_PATH_BYTES;
}

/**
 * Build the trusted inspection envelope without touching the filesystem.
 * The bounded Python worker resolves paths and verifies containment before
 * enumerating any directory; a renderer-supplied grant is never forwarded.
 */
export function datasetInspectionBody(body: unknown, authorizedRoots: readonly string[]): string {
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error('[dataset:invalid_request] Invalid dataset inspection request');
  const supplied = body as Record<string, unknown>;
  if (Object.keys(supplied).some(key => key !== 'path' && key !== 'authorized_roots') || !absoluteMetadataPath(supplied.path)) {
    throw new Error('[dataset:invalid_request] Choose a supported absolute dataset folder path');
  }
  if (!Array.isArray(authorizedRoots) || authorizedRoots.length > MAX_DATASET_ROOTS) {
    throw new Error('Invalid dataset folder authorizations');
  }
  // Old preferences or a previously selected unusual filename must not poison
  // every later inspection. Dropping an invalid grant only narrows access.
  const roots = authorizedRoots.filter(absoluteMetadataPath);
  const encoded = JSON.stringify({ path: supplied.path, authorized_roots: [...new Set(roots)] });
  if (Buffer.byteLength(encoded) > MAX_DATASET_INSPECTION_BYTES) throw new Error('Dataset inspection request is too large');
  return encoded;
}
