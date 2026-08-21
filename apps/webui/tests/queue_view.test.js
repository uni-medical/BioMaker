"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  actionForRun,
  readableRunName,
  reconcileRows,
} = require("../webui/static/queue_view.js");

class FakeRow {
  constructor(key) {
    this.dataset = { queueRunId: key };
    this.parent = null;
    this.updates = 0;
  }

  remove() {
    const index = this.parent?.children.indexOf(this) ?? -1;
    if (index >= 0) this.parent.children.splice(index, 1);
    this.parent = null;
  }
}

class FakeContainer {
  constructor(rows = []) {
    this.children = rows;
    rows.forEach((row) => { row.parent = this; });
  }

  insertBefore(row, current) {
    const previous = this.children.indexOf(row);
    if (previous >= 0) this.children.splice(previous, 1);
    const index = current == null ? this.children.length : this.children.indexOf(current);
    this.children.splice(index, 0, row);
    row.parent = this;
  }
}

test("queue actions follow server authorization and never duplicate Stop while stopping", () => {
  assert.equal(actionForRun({ status: "queued", allowed_actions: ["cancel"] }), "cancel");
  assert.equal(actionForRun({ status: "running", allowed_actions: ["stop"] }), "stop");
  assert.equal(actionForRun({ status: "stopping", allowed_actions: ["stop"] }), null);
  assert.equal(actionForRun({ status: "running", allowed_actions: [] }), null);
  assert.equal(actionForRun({ status: "queued" }), null);
});

test("readable run names never expose the internal UUID fallback", () => {
  assert.equal(readableRunName({
    id: "28a08c870cb04a13ae7995e2f0270b42",
    requested_exp_name: "requested-name",
    actual_exp_name: "actual-name",
    goal: "ignored goal",
  }, "Queued run"), "requested-name");
  assert.equal(readableRunName({ actual_exp_name: "actual-name" }), "actual-name");
  assert.equal(readableRunName({
    id: "28a08c870cb04a13ae7995e2f0270b42",
    project_snapshot: { task_body: "# Segment liver tumors\nMore detail" },
  }, "Queued run"), "Segment liver tumors");
  assert.equal(
    readableRunName({ id: "28a08c870cb04a13ae7995e2f0270b42" }, "Queued run"),
    "Queued run",
  );
});

test("queue reconciliation preserves row identity when polling returns the same runs", () => {
  const first = new FakeRow("run-1");
  const second = new FakeRow("run-2");
  const container = new FakeContainer([first, second]);
  const create = (key) => new FakeRow(key);
  const update = (row) => { row.updates += 1; };

  reconcileRows(container, [{ id: "run-1" }, { id: "run-2" }], create, update);

  assert.equal(container.children[0], first);
  assert.equal(container.children[1], second);
  assert.equal(first.updates, 1);
  assert.equal(second.updates, 1);
});

test("queue reconciliation reorders existing rows and removes only stale rows", () => {
  const stale = new FakeRow("stale");
  const first = new FakeRow("run-1");
  const second = new FakeRow("run-2");
  const container = new FakeContainer([stale, second, first]);

  reconcileRows(
    container,
    [{ id: "run-1" }, { id: "run-2" }, { id: "run-3" }],
    (key) => new FakeRow(key),
    () => {},
  );

  assert.deepEqual(container.children.map((row) => row.dataset.queueRunId), ["run-1", "run-2", "run-3"]);
  assert.equal(container.children[0], first);
  assert.equal(container.children[1], second);
  assert.equal(stale.parent, null);
});
