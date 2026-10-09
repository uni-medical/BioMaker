import assert from 'node:assert/strict';
import { test } from 'node:test';
import { clipboardText, MAX_CLIPBOARD_BYTES } from './clipboard';

test('clipboard write accepts exact Unicode text and bounds encoded bytes', () => {
  for (const value of ['', '训练结果\n| 模型 | AUROC |\n', 'x'.repeat(MAX_CLIPBOARD_BYTES)]) assert.equal(clipboardText(value), value);
  assert.throws(() => clipboardText('x'.repeat(MAX_CLIPBOARD_BYTES + 1)));
  assert.throws(() => clipboardText('🧬'.repeat(MAX_CLIPBOARD_BYTES / 4 + 1)));
});
test('clipboard write refuses object coercion and non-text payloads', () => {
  for (const value of [null, undefined, 1, [], { text: 'hello' }, Buffer.from('hello'), { toString: () => 'hello' }]) assert.throws(() => clipboardText(value));
});
