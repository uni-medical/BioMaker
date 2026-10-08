import { MAX_RESPONSE_BYTES } from './security';

/** API responses use strict JSON. Artifact previews preserve the original text,
 * including Python journal tokens such as NaN and Infinity. */
export async function readResponse(response: Response, parseJSON = true): Promise<unknown> {
  if (Number(response.headers.get('content-length') ?? '0') > MAX_RESPONSE_BYTES) throw new Error('内容过大，请使用“导出文件”。');
  const reader = response.body?.getReader(); const chunks: Uint8Array[] = []; let length = 0;
  if (reader) for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    length += value.byteLength;
    if (length > MAX_RESPONSE_BYTES) { await reader.cancel(); throw new Error('内容过大，请使用“导出文件”。'); }
    chunks.push(value);
  }
  const text = Buffer.concat(chunks).toString('utf8');
  let value: unknown = text;
  if ((parseJSON || !response.ok) && response.headers.get('content-type')?.includes('application/json')) {
    try { value = JSON.parse(text); } catch { throw new Error('本地服务返回了无效数据。'); }
  }
  if (!response.ok) {
    const detail = typeof value === 'object' && value !== null && 'detail' in value ? (value as { detail: unknown }).detail : null;
    throw new Error(typeof detail === 'string' ? detail : `操作失败（${response.status}）`);
  }
  return value;
}
