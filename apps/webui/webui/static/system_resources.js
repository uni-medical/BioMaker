(function exposeSystemResourceView(root, factory) {
  const view = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = view;
  }
  if (root) {
    root.BioMakerSystemResourceView = view;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function createSystemResourceView() {
  "use strict";

  function finiteNumber(value) {
    if (value == null || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function formatPercent(value) {
    const number = finiteNumber(value);
    return number == null ? "—" : `${Math.round(number)}%`;
  }

  function formatBytes(value) {
    const bytes = finiteNumber(value);
    if (bytes == null || bytes < 0) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let scaled = bytes;
    let unit = 0;
    while (scaled >= 1024 && unit < units.length - 1) {
      scaled /= 1024;
      unit += 1;
    }
    const precision = unit === 0 || scaled >= 100 ? 0 : 1;
    return `${scaled.toFixed(precision)} ${units[unit]}`;
  }

  function formatMemoryUsage(used, total) {
    return `${formatBytes(used)} / ${formatBytes(total)}`;
  }

  function buildSummary(snapshot) {
    if (!snapshot || typeof snapshot !== "object") return "System · unavailable";
    const parts = [
      "System",
      `CPU ${formatPercent(snapshot.cpu?.utilization_percent)}`,
      `RAM ${formatPercent(snapshot.memory?.utilization_percent)}`,
    ];
    const gpus = Array.isArray(snapshot.nvidia_gpus) ? snapshot.nvidia_gpus : [];
    if (gpus.length) {
      const utilizations = gpus
        .map((gpu) => finiteNumber(gpu?.utilization_percent))
        .filter((value) => value != null);
      parts.push(`GPU max ${formatPercent(utilizations.length ? Math.max(...utilizations) : null)}`);
    }
    if (!gpus.length) {
      const appleUtilization = finiteNumber(snapshot.apple_gpu?.utilization_percent);
      if (appleUtilization != null) parts.push(`GPU ${formatPercent(appleUtilization)}`);
    }
    if (snapshot.stale) parts.push("stale");
    return parts.join(" · ");
  }

  function markStale(snapshot, error) {
    if (!snapshot || typeof snapshot !== "object") return null;
    return { ...snapshot, stale: true, error: String(error || "System resource refresh failed.") };
  }

  return Object.freeze({
    buildSummary,
    formatBytes,
    formatMemoryUsage,
    formatPercent,
    markStale,
  });
});
