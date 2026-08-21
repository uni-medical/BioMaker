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
  }), "System · CPU 13% · RAM 25% · GPU max 87%");
});

test("summary omits unavailable GPU metrics and marks stale samples", () => {
  assert.equal(buildSummary({
    stale: true,
    cpu: { utilization_percent: null },
    memory: { used_bytes: null, total_bytes: null, utilization_percent: null },
    nvidia_gpus: [],
    apple_gpu: null,
  }), "System · CPU — · RAM — · stale");
  assert.equal(buildSummary(null), "System · unavailable");
});

test("summary reports measured Apple GPU utilization", () => {
  assert.equal(buildSummary({
    stale: false,
    cpu: { utilization_percent: 12.5 },
    memory: { utilization_percent: 25 },
    nvidia_gpus: [],
    apple_gpu: {
      source: "ioreg",
      utilization_percent: 97,
      shared_memory_used_bytes: 1_204_289_536,
    },
  }), "System · CPU 13% · RAM 25% · GPU 97%");
});

class FakeClassList {
  constructor(...names) {
    this.names = new Set(names);
  }

  add(...names) {
    names.forEach((name) => this.names.add(name));
  }

  contains(name) {
    return this.names.has(name);
  }

  remove(...names) {
    names.forEach((name) => this.names.delete(name));
  }

  toggle(name, force) {
    const enabled = force === undefined ? !this.names.has(name) : Boolean(force);
    if (enabled) this.names.add(name);
    else this.names.delete(name);
    return enabled;
  }
}

class FakeElement {
  constructor(tagName = "div") {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.classList = new FakeClassList();
    this.className = "";
    this.dataset = {};
    this.focused = false;
    this.hidden = false;
    this.id = "";
    this.tabIndex = 0;
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

  focus() {
    this.focused = true;
  }
}

function loadResourceRenderer({ fetchImpl = null, hidden = false } = {}) {
  const elements = new Map([
    ["#system-resource-summary", new FakeElement()],
    ["#system-resource-freshness", new FakeElement()],
    ["#system-resource-overview", new FakeElement()],
    ["#system-resource-gpus", new FakeElement()],
  ]);
  const documentListeners = new Map();
  const windowListeners = new Map();
  const document = {
    hidden,
    addEventListener(type, callback) {
      documentListeners.set(type, callback);
    },
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
  let nextTimerId = 1;
  const timers = new Map();
  const context = vm.createContext({
    AbortController,
    console,
    document,
    fetch: fetchImpl || (() => Promise.reject(new Error("unexpected fetch"))),
    window: {
      BioMakerSystemResourceView: require("../webui/static/system_resources.js"),
      addEventListener(type, callback) {
        windowListeners.set(type, callback);
      },
      clearTimeout(timerId) {
        timers.delete(timerId);
      },
      setTimeout(callback, delay) {
        const timerId = nextTimerId;
        nextTimerId += 1;
        timers.set(timerId, { callback, delay });
        return timerId;
      },
    },
  });
  const appSource = fs.readFileSync(
    path.join(__dirname, "../webui/static/app.js"),
    "utf8",
  );
  vm.runInContext(appSource, context, { filename: "app.js" });
  return {
    context,
    document,
    elements,
    documentListeners,
    timers,
    windowListeners,
    render(snapshot) {
      context.__resourceSnapshot = snapshot;
      vm.runInContext("renderSystemResources(__resourceSnapshot)", context);
    },
  };
}

function loadSystemTabController({ narrow = false } = {}) {
  const tabNames = ["events", "tree", "console", "artifacts", "system"];
  const tabs = tabNames.map((name, index) => {
    const tab = new FakeElement("button");
    tab.dataset.tab = name;
    tab.classList.toggle("active", index === 0);
    tab.tabIndex = index === 0 ? 0 : -1;
    return tab;
  });
  const panels = tabNames.map((name, index) => {
    const panel = new FakeElement("section");
    panel.id = `${name}-panel`;
    panel.classList.toggle("active", index === 0);
    return panel;
  });
  const detailPane = new FakeElement("section");
  const summary = new FakeElement("button");
  const workbench = new FakeElement("main");
  const document = {
    hidden: false,
    addEventListener() {},
    querySelector(selector) {
      if (selector === ".detail-pane") return detailPane;
      if (selector === "#system-resource-summary") return summary;
      if (selector === ".workbench") return workbench;
      if (selector === ".tab.active") return tabs.find((tab) => tab.classList.contains("active"));
      return null;
    },
    querySelectorAll(selector) {
      if (selector === ".tab") return tabs;
      if (selector === ".tab-panel") return panels;
      return [];
    },
  };
  const context = vm.createContext({
    AbortController,
    console,
    document,
    fetch: () => Promise.reject(new Error("unexpected fetch")),
    window: {
      BioMakerSystemResourceView: require("../webui/static/system_resources.js"),
      addEventListener() {},
      clearTimeout() {},
      matchMedia: () => ({ matches: narrow }),
      requestAnimationFrame(callback) { callback(); },
      setTimeout() { return 1; },
    },
  });
  const appSource = fs.readFileSync(
    path.join(__dirname, "../webui/static/app.js"),
    "utf8",
  );
  vm.runInContext(appSource, context, { filename: "app.js" });
  return { context, detailPane, panels, summary, tabs, workbench };
}

test("Apple GPU detail renders utilization and shared memory", () => {
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

test("missing Apple metrics do not render an Apple GPU card", () => {
  const { elements, render } = loadResourceRenderer();
  render({
    sampled_at: null,
    stale: false,
    error: "Apple GPU metrics are unavailable.",
    cpu: { utilization_percent: 12.5 },
    memory: { utilization_percent: 25, used_bytes: 2_000, total_bytes: 8_000 },
    nvidia_gpus: [],
    apple_gpu: null,
  });

  const [unavailable] = elements.get("#system-resource-gpus").children;
  assert.equal(unavailable.tagName, "P");
  assert.equal(unavailable.textContent, "No GPU metrics available.");
});

test("partial Apple GPU metrics keep the card and mark only the missing value unavailable", () => {
  const { elements, render } = loadResourceRenderer();
  render({
    sampled_at: null,
    stale: false,
    error: null,
    cpu: { utilization_percent: 12.5 },
    memory: { utilization_percent: 25, used_bytes: 2_000, total_bytes: 8_000 },
    nvidia_gpus: [],
    apple_gpu: {
      source: "ioreg",
      utilization_percent: 47,
      shared_memory_used_bytes: null,
    },
  });

  const [appleCard] = elements.get("#system-resource-gpus").children;
  const [utilization, sharedMemory] = appleCard.children[1].children;
  assert.equal(utilization.children[1].textContent, "47%");
  assert.equal(sharedMemory.children[1].textContent, "—");
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

test("resource polling does not overlap and stops scheduling while hidden", async () => {
  let fetchCalls = 0;
  let finishFetch;
  const snapshot = {
    sampled_at: "2026-08-21T08:09:10Z",
    stale: false,
    error: null,
    cpu: { utilization_percent: 12.5 },
    memory: { utilization_percent: 25, used_bytes: 2_000, total_bytes: 8_000 },
    nvidia_gpus: [],
    apple_gpu: null,
  };
  const fetchImpl = () => {
    fetchCalls += 1;
    return new Promise((resolve) => {
      finishFetch = () => resolve({
        ok: true,
        status: 200,
        headers: { get: () => "application/json" },
        json: async () => snapshot,
      });
    });
  };
  const { context, document, timers } = loadResourceRenderer({ fetchImpl });

  const first = vm.runInContext("pollSystemResources()", context);
  const overlapping = vm.runInContext("pollSystemResources()", context);
  await overlapping;
  assert.equal(fetchCalls, 1);

  finishFetch();
  await first;
  assert.equal(timers.size, 1);

  document.hidden = true;
  vm.runInContext("scheduleSystemResourcePoll(0)", context);
  assert.equal(timers.size, 0);
  await vm.runInContext("pollSystemResources()", context);
  assert.equal(fetchCalls, 1);
});

test("hiding the page aborts an in-flight resource request and polling can resume", async () => {
  const signals = [];
  const snapshot = {
    sampled_at: "2026-08-21T08:09:10Z",
    stale: false,
    error: null,
    cpu: { utilization_percent: 12.5 },
    memory: { utilization_percent: 25, used_bytes: 2_000, total_bytes: 8_000 },
    nvidia_gpus: [],
    apple_gpu: null,
  };
  const fetchImpl = (_path, options) => {
    signals.push(options.signal);
    if (signals.length > 1) {
      return Promise.resolve({
        ok: true,
        status: 200,
        headers: { get: () => "application/json" },
        json: async () => snapshot,
      });
    }
    return new Promise((_resolve, reject) => {
      options.signal.addEventListener("abort", () => {
        const error = new Error("aborted");
        error.name = "AbortError";
        reject(error);
      }, { once: true });
    });
  };
  const {
    context,
    document,
    timers,
  } = loadResourceRenderer({ fetchImpl });

  const first = vm.runInContext("pollSystemResources()", context);
  assert.equal(signals.length, 1);
  assert.equal(signals[0].aborted, false);

  document.hidden = true;
  vm.runInContext("handleVisibilityChange()", context);
  assert.equal(signals[0].aborted, true);
  await first;
  assert.equal(timers.size, 0);
  assert.equal(
    vm.runInContext(
      "state.systemResourceAbortController === null && state.systemResourcePolling === false",
      context,
    ),
    true,
  );

  document.hidden = false;
  vm.runInContext("handleVisibilityChange()", context);
  assert.equal([...timers.values()].some(({ delay }) => delay === 0), true);
  await vm.runInContext("pollSystemResources()", context);
  assert.equal(signals.length, 2);
  assert.equal(signals[1].aborted, false);
  assert.equal([...timers.values()].some(({ delay }) => delay === 2000), true);
});

test("a resource request is aborted after the bounded timeout and retries", async () => {
  let signal;
  const fetchImpl = (_path, options) => {
    signal = options.signal;
    return new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => {
        const error = new Error("aborted");
        error.name = "AbortError";
        reject(error);
      }, { once: true });
    });
  };
  const { context, timers } = loadResourceRenderer({ fetchImpl });

  const polling = vm.runInContext("pollSystemResources()", context);
  const timeoutEntry = [...timers.entries()].find(([, timer]) => timer.delay === 5000);
  assert.ok(timeoutEntry);
  const [timeoutId, timeout] = timeoutEntry;
  timers.delete(timeoutId);
  timeout.callback();
  await polling;

  assert.equal(signal.aborted, true);
  assert.equal(
    vm.runInContext(
      "state.systemResourceAbortController === null && state.systemResourcePolling === false",
      context,
    ),
    true,
  );
  assert.equal([...timers.values()].some(({ delay }) => delay === 2000), true);
});

test("the resource summary opens a non-overlay System tab and focuses it on narrow screens", () => {
  const {
    context,
    detailPane,
    panels,
    summary,
    tabs,
    workbench,
  } = loadSystemTabController({ narrow: true });

  vm.runInContext("openSystemResources()", context);

  assert.equal(vm.runInContext("state.activeTab", context), "system");
  assert.equal(detailPane.classList.contains("system-view"), true);
  assert.equal(summary.attributes["aria-pressed"], "true");
  assert.equal(workbench.dataset.mobileView, "run");
  assert.equal(tabs.at(-1).classList.contains("active"), true);
  assert.equal(tabs.at(-1).focused, true);
  assert.equal(panels.at(-1).classList.contains("active"), true);
  assert.equal(panels.at(-1).attributes["aria-hidden"], "false");
  assert.equal(panels[0].attributes["aria-hidden"], "true");
});

test("five detail tabs support End and wrapped ArrowRight keyboard navigation", () => {
  const { context, tabs } = loadSystemTabController();
  let prevented = false;
  context.__event = {
    key: "End",
    currentTarget: tabs[0],
    preventDefault() { prevented = true; },
  };

  vm.runInContext("handleTabKeydown(__event)", context);
  assert.equal(prevented, true);
  assert.equal(tabs.at(-1).focused, true);
  assert.equal(vm.runInContext("state.activeTab", context), "system");

  context.__event = {
    key: "ArrowRight",
    currentTarget: tabs.at(-1),
    preventDefault() {},
  };
  vm.runInContext("handleTabKeydown(__event)", context);
  assert.equal(vm.runInContext("state.activeTab", context), "events");
  assert.equal(tabs[0].classList.contains("active"), true);
});

test("system resources live in the detail pane without legacy popover markup or CSS", () => {
  const html = fs.readFileSync(
    path.join(__dirname, "../webui/static/index.html"),
    "utf8",
  );
  const styles = fs.readFileSync(
    path.join(__dirname, "../webui/static/styles.css"),
    "utf8",
  );

  assert.match(html, /<button id="system-resource-summary"[^>]+aria-controls="system-panel"/);
  assert.match(
    html,
    /<section id="system-panel" class="tab-panel system-resource-panel"[^>]+tabindex="0"/,
  );
  assert.match(html, />Whole-system resources</);
  assert.doesNotMatch(html, /<details id="system-resources"/);
  assert.doesNotMatch(html, /system-resource-popover/);
  assert.doesNotMatch(html, /MPS available|\bANE\b|Neural Engine|dedicated VRAM|current run/i);
  assert.match(styles, /\.detail-pane\.system-view \.run-summary\s*\{\s*display: none;/);
  assert.match(styles, /\.system-resource-panel\s*\{\s*overflow: auto;/);
  assert.doesNotMatch(styles, /\.system-resource-popover/);
  assert.doesNotMatch(styles, /\.system-resources\s*>\s*summary/);
});
