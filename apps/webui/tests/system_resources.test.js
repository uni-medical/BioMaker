"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const {
  buildSummary,
  formatBytes,
  formatMemoryUsage,
  formatPercent,
  markStale,
} = require("../webui/static/system_resources.js");

test("resource formatting preserves unavailable values instead of inventing metrics", () => {
  assert.equal(formatPercent(null), "—");
  assert.equal(formatPercent(41.6), "42%");
  assert.equal(formatBytes(undefined), "—");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatMemoryUsage(null, 8 * 1024 ** 3), "— / 8.0 GB");
});

test("summary labels metrics as whole-system and keeps per-card detail out of the header", () => {
  assert.equal(buildSummary({
    stale: false,
    cpu: { utilization_percent: 12.5 },
    memory: { used_bytes: 2 * 1024 ** 3, total_bytes: 8 * 1024 ** 3, utilization_percent: 25 },
    nvidia_gpus: [
      { index: 0, utilization_percent: 87 },
      { index: 1, utilization_percent: null },
    ],
    apple_mps: { available: false },
  }), "System · CPU 13% · RAM 25% · GPU max 87%");
});

test("summary treats MPS capability as an unavailable metric and marks stale samples", () => {
  assert.equal(buildSummary({
    stale: true,
    cpu: { utilization_percent: null },
    memory: { used_bytes: null, total_bytes: null, utilization_percent: null },
    nvidia_gpus: [],
    apple_mps: { available: true },
  }), "System · CPU — · RAM — · GPU — · stale");
  assert.equal(buildSummary(null), "System · unavailable");
});

test("summary reports measured Apple GPU utilization before the MPS fallback", () => {
  assert.equal(buildSummary({
    stale: false,
    cpu: { utilization_percent: 12.5 },
    memory: { utilization_percent: 25 },
    nvidia_gpus: [],
    apple_mps: { available: true },
    apple_gpu: {
      source: "ioreg",
      utilization_percent: 97,
      shared_memory_used_bytes: 1_204_289_536,
    },
  }), "System · CPU 13% · RAM 25% · GPU 97%");
});

class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.className = "";
    this.hidden = false;
    this.textContent = "";
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = [...children];
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
}

function loadResourceRenderer() {
  const elements = new Map([
    ["#system-resource-summary", new FakeElement()],
    ["#system-resource-freshness", new FakeElement()],
    ["#system-resource-overview", new FakeElement()],
    ["#system-resource-gpus", new FakeElement()],
  ]);
  const document = {
    addEventListener() {},
    createElement(tagName) {
      return new FakeElement(tagName);
    },
    querySelector(selector) {
      return elements.get(selector) || null;
    },
    querySelectorAll() {
      return [];
    },
  };
  const context = vm.createContext({
    console,
    document,
    window: {
      BioMakerSystemResourceView: require("../webui/static/system_resources.js"),
      addEventListener() {},
    },
  });
  const appSource = fs.readFileSync(
    path.join(__dirname, "../webui/static/app.js"),
    "utf8",
  );
  vm.runInContext(appSource, context, { filename: "app.js" });
  return {
    elements,
    render(snapshot) {
      context.__resourceSnapshot = snapshot;
      vm.runInContext("renderSystemResources(__resourceSnapshot)", context);
    },
  };
}

test("Apple GPU detail renders utilization and shared unified memory", () => {
  const { elements, render } = loadResourceRenderer();
  render({
    sampled_at: null,
    stale: false,
    error: null,
    cpu: { utilization_percent: 12.5 },
    memory: {
      utilization_percent: 25,
      used_bytes: 2 * 1024 ** 3,
      total_bytes: 8 * 1024 ** 3,
    },
    nvidia_gpus: [],
    apple_mps: { available: true },
    apple_gpu: {
      source: "ioreg",
      utilization_percent: 97,
      shared_memory_used_bytes: 1_204_289_536,
    },
  });

  const gpuList = elements.get("#system-resource-gpus");
  assert.equal(gpuList.children.length, 1);
  const [appleCard] = gpuList.children;
  assert.equal(appleCard.children[0].textContent, "Apple GPU");
  const [utilization, sharedMemory] = appleCard.children[1].children;
  assert.equal(utilization.children[0].textContent, "Utilization");
  assert.equal(utilization.children[1].textContent, "97%");
  assert.equal(sharedMemory.children[0].textContent, "Shared memory");
  assert.equal(sharedMemory.children[1].textContent, "1.1 GB");
});

test("transport failures keep the last resource values but mark them stale", () => {
  const current = {
    sampled_at: "2026-08-21T08:09:10Z",
    stale: false,
    error: null,
    cpu: { utilization_percent: 12.5 },
  };

  const stale = markStale(current, "System resource refresh failed; retrying.");

  assert.notEqual(stale, current);
  assert.equal(current.stale, false);
  assert.equal(stale.stale, true);
  assert.equal(stale.cpu, current.cpu);
  assert.equal(stale.error, "System resource refresh failed; retrying.");
  assert.match(buildSummary(stale), /stale$/);
  assert.equal(markStale(null, "failed"), null);
});
