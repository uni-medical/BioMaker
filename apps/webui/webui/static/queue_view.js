(function exposeQueueView(root, factory) {
  const view = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = view;
  }
  if (root) {
    root.BioMakerQueueView = view;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function createQueueView() {
  "use strict";

  function actionForRun(run) {
    const status = String(run?.status || "").toLowerCase();
    const actions = new Set(Array.isArray(run?.allowed_actions) ? run.allowed_actions : []);
    if (status === "queued" && actions.has("cancel")) return "cancel";
    if (status === "running" && actions.has("stop")) return "stop";
    return null;
  }

  function readableRunName(run, fallback = "Run") {
    const requested = String(run?.requested_exp_name || "").trim();
    if (requested) return requested;
    const actual = String(run?.actual_exp_name || "").trim();
    if (actual) return actual;
    const goal = String(run?.goal || run?.project_snapshot?.task_body || "")
      .split(/\r?\n/)
      .map((line) => line.replace(/^\s*#+\s*/, "").trim())
      .find(Boolean);
    if (goal) {
      const sentence = goal.split(/(?<=[.!?。！？])\s+/)[0];
      return sentence.length > 64
        ? `${sentence.slice(0, 61).trimEnd()}...`
        : sentence;
    }
    return fallback;
  }

  function reconcileRows(container, runs, createRow, updateRow) {
    const existing = new Map(
      Array.from(container.children).map((row) => [row.dataset.queueRunId, row]),
    );
    const keep = new Set();

    runs.forEach((run, index) => {
      const key = String(run.id);
      const row = existing.get(key) || createRow(key);
      keep.add(key);
      updateRow(row, run);
      const current = container.children[index] || null;
      if (current !== row) container.insertBefore(row, current);
    });

    for (const [key, row] of existing) {
      if (!keep.has(key)) row.remove();
    }
  }

  return Object.freeze({ actionForRun, readableRunName, reconcileRows });
});
