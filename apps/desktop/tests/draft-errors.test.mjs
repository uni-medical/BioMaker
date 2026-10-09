import assert from 'node:assert/strict';
import { test } from 'node:test';
import { draftErrorTitle } from '../src/draftErrors.ts';
test('each draft failure names its own action in the currently selected language', () => {
  const titles = { save: ['草稿保存异常', 'Draft save issue'], delete: ['草稿删除异常', 'Draft deletion issue'], rename: ['草稿重命名异常', 'Draft rename issue'], copy: ['草稿复制异常', 'Draft copy issue'], load: ['草稿载入异常', 'Draft loading issue'] };
  for (const [operation, expected] of Object.entries(titles)) {
    assert.equal(draftErrorTitle(operation, (zh, _en) => zh), expected[0]);
    assert.equal(draftErrorTitle(operation, (_zh, en) => en), expected[1]);
  }
});
