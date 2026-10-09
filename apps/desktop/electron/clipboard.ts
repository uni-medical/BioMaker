// Only text writes are exposed to the trusted application frame. Chromium's
// permission handlers continue to deny clipboard reads and all other grants.
export const MAX_CLIPBOARD_BYTES = 4 * 1024 * 1024;

export function clipboardText(value: unknown): string {
  if (typeof value !== 'string' || value.length > MAX_CLIPBOARD_BYTES || Buffer.byteLength(value, 'utf8') > MAX_CLIPBOARD_BYTES) {
    throw new Error('Clipboard text must be a string no larger than 4 MiB.');
  }
  return value;
}
