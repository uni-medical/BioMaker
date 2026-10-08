import path from 'node:path';

const METHODS = new Set(['GET', 'POST', 'PUT', 'PATCH', 'DELETE']);
const EXTERNAL_HOSTS = new Set(['github.com', 'docs.astral.sh', 'platform.openai.com', 'huggingface.co']);
export const MAX_REQUEST_BYTES = 2 * 1024 * 1024;
export const MAX_RESPONSE_BYTES = 16 * 1024 * 1024;

export function validateRequest(rawPath: unknown, rawOptions: unknown = {}): { path: string; method: string; body?: string } {
  if (typeof rawPath !== 'string' || rawPath.length > 4096 || /[\\\x00-\x20\x7f]/.test(rawPath)) throw new Error('Invalid request path');
  if (!rawPath.startsWith('/api/') && !rawPath.startsWith('/artifacts/runs/')) throw new Error('Request path is not allowed');
  // Reject encoded separators, dot segments and nested encodings before WHATWG URL normalizes them.
  let decoded: string;
  try { decoded = decodeURIComponent(rawPath.split('?')[0]); } catch { throw new Error('Invalid request encoding'); }
  if (/[\\%\x00-\x1f\x7f]/.test(decoded) || decoded.split('/').some(part => part === '.' || part === '..')) throw new Error('Unsafe request path');
  if (/%(?:2f|5c)/i.test(rawPath.split('?')[0])) throw new Error('Encoded separators are not allowed');
  const options = rawOptions ?? {};
  if (typeof options !== 'object' || Array.isArray(options)) throw new Error('Invalid request options');
  const { method = 'GET', body, ...rest } = options as Record<string, unknown>;
  if (Object.keys(rest).length || typeof method !== 'string' || !METHODS.has(method.toUpperCase())) throw new Error('Unsupported request options');
  const normalized = method.toUpperCase();
  if (rawPath.startsWith('/artifacts/') && normalized !== 'GET') throw new Error('Artifacts are read only');
  if (body !== undefined && normalized === 'GET') throw new Error('GET requests cannot have a body');
  const encoded = body === undefined ? undefined : JSON.stringify(body);
  if (encoded !== undefined && Buffer.byteLength(encoded) > MAX_REQUEST_BYTES) throw new Error('Request is too large');
  return { path: rawPath, method: normalized, body: encoded };
}

export function isWithin(candidate: string, root: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}

export function validateArtifact(runId: unknown, name: unknown): { runId: string; name: string } {
  if (typeof runId !== 'string' || !runId || runId.length > 180 || runId.startsWith('.') || /[\\/\x00-\x1f\x7f]/.test(runId)) throw new Error('Invalid run identifier');
  if (typeof name !== 'string' || !name || name.length > 240 || name === '.' || name === '..' || name.startsWith('.') || /[\\/\x00-\x1f\x7f]/.test(name)) throw new Error('Invalid artifact name');
  return { runId, name };
}

export function externalURL(raw: unknown): string {
  if (typeof raw !== 'string' || raw.length > 4096) throw new Error('Invalid URL');
  const url = new URL(raw);
  if (url.protocol !== 'https:' || url.username || url.password || url.port || !EXTERNAL_HOSTS.has(url.hostname)) throw new Error('This external link is not allowed');
  return url.href;
}

export function validTheme(value: unknown): value is 'system' | 'light' | 'dark' {
  return value === 'system' || value === 'light' || value === 'dark';
}
