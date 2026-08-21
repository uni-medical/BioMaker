(function exposeRefreshPolicy(root, factory) {
  const policy = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = policy;
  }
  if (root) {
    root.BioMakerRefreshPolicy = policy;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function createRefreshPolicy() {
  "use strict";

  const ACTIVE_STATUSES = new Set(["queued", "running", "stopping"]);
  const TERMINAL_STATUSES = new Set([
    "succeeded",
    "failed",
    "cancelled",
    "stopped",
    "interrupted",
  ]);
  const BACKOFF_DELAYS = [2000, 4000, 8000, 16000, 30000];

  function isActive(status) {
    return ACTIVE_STATUSES.has(String(status || ""));
  }

  function isTerminal(status) {
    return TERMINAL_STATUSES.has(String(status || ""));
  }

  function initialRun(runs = []) {
    return runs.find((run) => isActive(run?.status)) || null;
  }

  function pollDelay({ hidden = false, hasActive = false, failures = 0 } = {}) {
    if (hidden) return null;
    if (failures > 0) {
      return BACKOFF_DELAYS[Math.min(failures - 1, BACKOFF_DELAYS.length - 1)];
    }
    return hasActive ? 3000 : 30000;
  }

  function treeDecision({
    available = false,
    revision = null,
    loadedRevision = null,
    loadingRevision = null,
    autoUpdate = true,
    force = false,
  } = {}) {
    if (!available || !revision) return "unavailable";
    if (force) return "load";
    if (revision === loadedRevision || revision === loadingRevision) return "current";
    return autoUpdate ? "load" : "pending";
  }

  return Object.freeze({
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    BACKOFF_DELAYS,
    isActive,
    isTerminal,
    initialRun,
    pollDelay,
    treeDecision,
  });
});
