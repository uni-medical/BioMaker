import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { datasetInspectionBody, MAX_DATASET_PATH_BYTES, MAX_DATASET_ROOTS } from './dataset-inspection';

const root = path.resolve('synthetic-dataset');
const workspace = path.resolve('synthetic-profile', 'workspace');

test('native dataset inspection replaces every renderer-supplied authorization', () => {
  for (const supplied of [[path.parse(root).root], ['../../../'], { arbitrary: 'root' }, null, 123]) {
    assert.deepEqual(JSON.parse(datasetInspectionBody({ path: root, authorized_roots: supplied }, [root, workspace, root])), {
      path: root, authorized_roots: [root, workspace],
    });
  }
});

test('metadata construction does not resolve, stat, or normalize the requested path', () => {
  const nonexistent = `${root}${path.sep}missing${path.sep}..${path.sep}alias`;
  assert.equal(JSON.parse(datasetInspectionBody({ path: nonexistent }, [root])).path, nonexistent);
  assert.deepEqual(JSON.parse(datasetInspectionBody({ path: root }, [])).authorized_roots, []);
});

test('dataset inspection requires a bounded absolute path and rejects malformed bodies', () => {
  for (const body of [undefined, null, [], 'dataset', 42, {}, { path: 1 }, { path: '' }, { path: 'relative/path' }, { path: `${root}\0` }, { path: `${root}\n` }, { path: `${root}\u007f` }, { path: root, recursive: true }]) {
    assert.throws(() => datasetInspectionBody(body, [workspace]), /^Error: \[dataset:invalid_request\]/);
  }
  const prefix = path.parse(root).root;
  const boundary = prefix + 'a'.repeat(MAX_DATASET_PATH_BYTES - Buffer.byteLength(prefix));
  assert.equal(JSON.parse(datasetInspectionBody({ path: boundary }, [workspace])).path, boundary);
  assert.throws(() => datasetInspectionBody({ path: boundary + 'a' }, [workspace]), /\[dataset:invalid_request\]/);
  assert.throws(() => datasetInspectionBody({ path: prefix + '数'.repeat(MAX_DATASET_PATH_BYTES / 2) }, [workspace]), /\[dataset:invalid_request\]/);
  const privateInvalidPath = `${root}/private-dataset-name\n`;
  assert.throws(() => datasetInspectionBody({ path: privateInvalidPath }, [workspace]), error => {
    assert(error instanceof Error);
    assert(error.message.startsWith('[dataset:invalid_request]'));
    assert(!error.message.includes('private-dataset-name'));
    return true;
  });
});

test('invalid persisted grants are ignored while valid authorizations survive', () => {
  const invalid = ['relative', root + '\0', root + '\n', root + '\u007f', root + 'x'.repeat(MAX_DATASET_PATH_BYTES)];
  assert.deepEqual(JSON.parse(datasetInspectionBody({ path: root }, [...invalid, root, workspace])).authorized_roots, [root, workspace]);
  assert.deepEqual(JSON.parse(datasetInspectionBody({ path: root }, invalid)).authorized_roots, []);
});

test('native authorization count remains bounded before filtering or deduplication', () => {
  assert.throws(() => datasetInspectionBody({ path: root }, Array(MAX_DATASET_ROOTS + 1).fill(root)));
  assert.throws(() => datasetInspectionBody({ path: root }, Array(MAX_DATASET_ROOTS + 1).fill('relative')));
  assert.equal(JSON.parse(datasetInspectionBody({ path: root }, Array(MAX_DATASET_ROOTS).fill(root))).authorized_roots.length, 1);
});

test('serialized inspection envelope has an independent byte limit', () => {
  const prefix = path.parse(root).root;
  // Quotes remain legal path metadata but double when JSON-encoded. No file
  // is opened; this tests the actual wire size after trusted grants are added.
  const roots = Array.from({ length: MAX_DATASET_ROOTS }, (_, index) => prefix + '"'.repeat(3000) + index);
  assert.throws(() => datasetInspectionBody({ path: root }, roots), /too large/);
});
