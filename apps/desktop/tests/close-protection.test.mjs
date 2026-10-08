import test from "node:test";
import assert from "node:assert/strict";
import { collectCloseBlockReason, registerCloseProtection } from "../src/closeProtection.ts";

test("all scopes run even when saving wins, regardless of registration order", () => {
  for (const reasons of [["saving", "unsaved", null], ["unsaved", null, "saving"]]) {
    const calls = [];
    assert.equal(collectCloseBlockReason(reasons.map((reason, index) => () => {
      calls.push(index);
      return reason;
    })), "saving");
    assert.deepEqual(calls, [0, 1, 2]);
  }
});

test("a failed flush is unsaved and does not prevent another scope from running", () => {
  let flushed = false;
  assert.equal(collectCloseBlockReason([
    () => { throw new Error("Storage unavailable"); },
    () => { flushed = true; return null; },
  ]), "unsaved");
  assert.equal(flushed, true);
  assert.equal(collectCloseBlockReason([() => "saving", () => { throw new Error("Storage unavailable"); }]), "saving");
});

test("clean or absent scopes allow the window to close", () => {
  assert.equal(collectCloseBlockReason([]), null);
  assert.equal(collectCloseBlockReason([() => null, () => null]), null);
});

function fixture(run) {
  const previousWindow = globalThis.window;
  const listeners = new Set();
  const reports = [];
  const cleanups = [];
  let attached = 0;
  let detached = 0;
  globalThis.window = {
    addEventListener(name, callback) { assert.equal(name, "beforeunload"); listeners.add(callback); attached += 1; },
    removeEventListener(name, callback) { assert.equal(name, "beforeunload"); listeners.delete(callback); detached += 1; },
    biomaker: { reportCloseBlock(reason) { reports.push(reason); } },
  };
  const f = {
    register(key, callback) {
      const cleanup = registerCloseProtection(key, callback);
      cleanups.push(cleanup);
      return cleanup;
    },
    close() {
      const event = { prevented: false, returnValue: "", preventDefault() { this.prevented = true; } };
      for (const callback of listeners) callback(event);
      return event;
    },
    reports,
    get attached() { return attached; },
    get detached() { return detached; },
    get listenerCount() { return listeners.size; },
  };
  try { run(f); }
  finally {
    for (const cleanup of cleanups) cleanup();
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
}

test("one listener reports the combined reason once and observes current ref values", () => fixture(f => {
  const settings = { saving: true, dirty: true };
  let draftFlushes = 0;
  f.register("settings", () => settings.saving ? "saving" : settings.dirty ? "unsaved" : null);
  f.register("drafts", () => { draftFlushes += 1; return "unsaved"; });
  assert.equal(f.attached, 1);
  assert.equal(f.listenerCount, 1);
  assert.equal(f.close().prevented, true);
  assert.deepEqual(f.reports, ["saving"]);
  assert.equal(draftFlushes, 1);
  settings.saving = false;
  assert.equal(f.close().prevented, true);
  assert.deepEqual(f.reports, ["saving", "unsaved"]);
  assert.equal(f.attached, 1);
}));

test("replacing a scope survives stale cleanup, including reuse of the same callback", () => fixture(f => {
  const callback = () => "unsaved";
  const oldCleanup = f.register("settings", callback);
  const cleanup = f.register("settings", callback);
  oldCleanup();
  assert.equal(f.listenerCount, 1);
  assert.equal(f.close().prevented, true);
  assert.deepEqual(f.reports, ["unsaved"]);
  cleanup();
  cleanup();
  assert.equal(f.listenerCount, 0);
  assert.equal(f.attached, 1);
  assert.equal(f.detached, 1);
}));

test("removing one scope preserves others, and the last removal detaches the listener", () => fixture(f => {
  const first = f.register("settings", () => "saving");
  const last = f.register("drafts", () => null);
  first();
  assert.equal(f.listenerCount, 1);
  assert.equal(f.close().prevented, false);
  assert.deepEqual(f.reports, []);
  last();
  assert.equal(f.listenerCount, 0);
  assert.equal(f.detached, 1);
  const replacement = f.register("settings", () => "unsaved");
  assert.equal(f.attached, 2);
  assert.equal(f.close().prevented, true);
  replacement();
}));

test("a failing native report cannot bypass renderer close prevention", () => fixture(f => {
  f.register("drafts", () => "unsaved");
  globalThis.window.biomaker.reportCloseBlock = () => { throw new Error("Bridge unavailable"); };
  const event = f.close();
  assert.equal(event.prevented, true);
  assert.equal(event.returnValue, "Unsaved changes");
}));
