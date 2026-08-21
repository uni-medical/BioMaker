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

  return Object.freeze({ actionForRun, reconcileRows });
});
