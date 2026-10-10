import test from 'node:test';
import assert from 'node:assert/strict';
import { saveProviderSettings } from './provider-settings';

const revision = 'a'.repeat(64);
const config = { settings_revision: revision, values: { OPENAI_API_KEY: { configured: true, masked: '••••••••', secret: true } } };
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } });
const isUncertain = (error: unknown) => {
  assert(error instanceof Error);
  assert.match(error.message, /^\[provider_settings:uncertain\]/);
  assert(!error.message.includes('private-sentinel'));
  return true;
};

test('a successful provider save retains the masked response and revision', async () => {
  let calls = 0;
  assert.deepEqual(await saveProviderSettings(async () => { calls++; return json(config); }), config);
  assert.equal(calls, 1);
});

test('transport failures become uncertain without retrying or returning private errors', async () => {
  for (const error of [new Error('private-sentinel'), new DOMException('private-sentinel', 'TimeoutError')]) {
    let calls = 0;
    await assert.rejects(saveProviderSettings(async () => { calls++; throw error; }), isUncertain);
    assert.equal(calls, 1);
  }
});

test('explicit HTTP save refusals preserve their existing codes and retry semantics', async () => {
  for (const [status, message] of [[409, '[provider_settings:conflict] Review settings.'], [422, '[provider_settings:key_required] Enter key.'], [503, 'Synthetic settings save failure']] as const) {
    await assert.rejects(saveProviderSettings(async () => json({ detail: message }, status)), error => {
      assert(error instanceof Error);
      assert.equal(error.message, message);
      return true;
    });
  }
});

test('a truncated successful response is uncertain after a potentially committed save', async () => {
  const body = new ReadableStream<Uint8Array>({ start(controller) { controller.error(new Error('private-sentinel')); } });
  await assert.rejects(saveProviderSettings(async () => new Response(body)), isUncertain);
});

test('malformed JSON after HTTP success requires review instead of reporting a saved configuration', async () => {
  await assert.rejects(saveProviderSettings(async () => new Response('{private-sentinel', { headers: { 'Content-Type': 'application/json' } })), isUncertain);
});

test('successful responses must contain a reviewable configuration and opaque revision', async () => {
  for (const value of [null, [], 'private-sentinel', {}, { values: {} }, { settings_revision: 'private-sentinel', values: {} }, { settings_revision: revision, values: [] }]) {
    await assert.rejects(saveProviderSettings(async () => json(value)), isUncertain);
  }
});
