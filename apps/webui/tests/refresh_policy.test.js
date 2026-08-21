"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  BACKOFF_DELAYS,
  initialRun,
  isActive,
  isTerminal,
  pollDelay,
  treeDecision,
} = require("../webui/static/refresh_policy.js");

test("initialRun opens only active work and otherwise leaves a new experiment", () => {
  const completed = { id: "completed", status: "succeeded" };
  const queued = { id: "queued", status: "queued" };

  assert.equal(initialRun([completed]), null);
  assert.equal(initialRun([completed, queued]), queued);
  assert.equal(initialRun([]), null);
});

test("status helpers distinguish active, terminal, and unknown runs", () => {
  for (const status of ["queued", "running", "stopping"]) {
    assert.equal(isActive(status), true, `${status} should be active`);
    assert.equal(isTerminal(status), false, `${status} should not be terminal`);
  }

  for (const status of [
    "succeeded",
    "failed",
    "cancelled",
    "stopped",
    "interrupted",
  ]) {
    assert.equal(isTerminal(status), true, `${status} should be terminal`);
    assert.equal(isActive(status), false, `${status} should not be active`);
  }

  for (const status of [undefined, null, "", "unknown"]) {
    assert.equal(isActive(status), false);
    assert.equal(isTerminal(status), false);
  }
});

test("pollDelay uses an active cadence and a slower idle cadence", () => {
  assert.equal(pollDelay({ hasActive: true }), 3000);
  assert.equal(pollDelay({ hasActive: false }), 30000);
  assert.equal(pollDelay(), 30000);
});

test("pollDelay pauses while the page is hidden", () => {
  assert.equal(pollDelay({ hidden: true, hasActive: true }), null);
  assert.equal(pollDelay({ hidden: true, failures: 3 }), null);
});

test("pollDelay backs off after failures and caps at the longest delay", () => {
  assert.deepEqual(BACKOFF_DELAYS, [2000, 4000, 8000, 16000, 30000]);

  BACKOFF_DELAYS.forEach((expected, index) => {
    assert.equal(
      pollDelay({ hasActive: true, failures: index + 1 }),
      expected,
    );
  });

  assert.equal(pollDelay({ failures: BACKOFF_DELAYS.length + 10 }), 30000);
});

test("treeDecision treats loaded or in-flight revisions as current", () => {
  assert.equal(
    treeDecision({
      available: true,
      revision: "tree-7",
      loadedRevision: "tree-7",
    }),
    "current",
  );
  assert.equal(
    treeDecision({
      available: true,
      revision: "tree-7",
      loadingRevision: "tree-7",
    }),
    "current",
  );
});

test("treeDecision loads a new revision when auto-update is enabled", () => {
  assert.equal(
    treeDecision({
      available: true,
      revision: "tree-8",
      loadedRevision: "tree-7",
    }),
    "load",
  );
});

test("treeDecision records a new revision as pending when auto-update is off", () => {
  assert.equal(
    treeDecision({
      available: true,
      revision: "tree-8",
      loadedRevision: "tree-7",
      autoUpdate: false,
    }),
    "pending",
  );
});

test("treeDecision force-loads even a current revision", () => {
  assert.equal(
    treeDecision({
      available: true,
      revision: "tree-7",
      loadedRevision: "tree-7",
      autoUpdate: false,
      force: true,
    }),
    "load",
  );
});

test("treeDecision rejects missing artifacts before considering force", () => {
  assert.equal(treeDecision({ revision: "tree-1", force: true }), "unavailable");
  assert.equal(
    treeDecision({ available: true, revision: null, force: true }),
    "unavailable",
  );
});
