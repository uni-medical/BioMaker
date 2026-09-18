"use strict";

const PREFERRED_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"];
const MAX_MODEL_SUGGESTIONS = 8;
const SYSTEM_RESOURCE_POLL_INTERVAL_MS = 2000;
const SYSTEM_RESOURCE_REQUEST_TIMEOUT_MS = 5000;
let secretReadGeneration = 0;
let modelDiscoveryGeneration = 0;

const DEFAULT_OPTIONS = {
  model_for_all_agents: null,
  agent_steps: 35,
  exec_timeout: 600,
  exec_parallel_workers: "auto",
  agent_total_time_limit: 86400,
  final_retrain_enabled: true,
};

const SECRET_FIELDS = {
  OPENAI_API_KEY: { input: "#settings-api-key", status: "#settings-api-key-status", label: "API key" },
  SERPER_API_KEY: { input: "#settings-serper-key", status: "#settings-serper-key-status", label: "Serper key" },
  JINA_API_KEY: { input: "#settings-jina-key", status: "#settings-jina-key-status", label: "Jina key" },
};

const state = {
  runs: [],
  datasets: [],
  models: [],
  env: {},
  envSource: {},
  selectedRunId: null,
  selectedRun: null,
  activeTab: "events",
  mobileView: "setup",
  dirty: false,
  polling: false,
  pollQueued: false,
  pollTimer: null,
  pollFailures: 0,
  pollStopped: false,
  updateSnapshot: null,
  resources: null,
  queue: { paused: false, reason: null, pause_generation: 0 },
  queueEndpointAvailable: null,
  systemResourceTimer: null,
  systemResourcePolling: false,
  systemResourceAbortController: null,
  systemResourceEndpointAvailable: null,
  systemResourceSnapshot: null,
  treeLoadToken: 0,
  treeLoadCancel: null,
  treeViewState: {},
  treeViewId: null,
  treeInteracting: false,
};

const refreshPolicy = window.BioMakerRefreshPolicy || {
  isActive: (status) => ["queued", "running", "stopping"].includes(status),
  isTerminal,
  initialRun: (runs = []) => runs.find((run) => ["queued", "running", "stopping"].includes(run?.status)) || null,
  pollDelay: ({ hidden, hasActive, failures }) => {
    if (hidden) return null;
    if (failures) return Math.min(30000, 2000 * (2 ** (failures - 1)));
    return hasActive ? 3000 : 30000;
  },
  treeDecision: ({ available, revision, loadedRevision, loadingRevision, autoUpdate, force }) => {
    if (!available || !revision) return "unavailable";
    if (force) return "load";
    if (revision === loadedRevision || revision === loadingRevision) return "current";
    return autoUpdate ? "load" : "pending";
  },
};

const systemResourceView = window.BioMakerSystemResourceView || {
  buildSummary: () => "System · unavailable",
  formatBytes: () => "—",
  formatMemoryUsage: () => "— / —",
  formatPercent: () => "—",
  markStale: () => null,
};
const queueView = window.BioMakerQueueView;

function emptyResourceState() {
  return {
    runRevision: null,
    artifactsRevision: null,
    finalBarrier: null,
    tree: { loaded: null, loading: null, pending: null },
    events: { loaded: null, loading: null },
    console: { loaded: null, loading: null },
  };
}

state.resources = emptyResourceState();

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
    document.body.classList.add("icons-ready");
  }
}

function emptyStateMarkup(icon, title) {
  return `
    <div class="empty-state rich-empty-state">
      <div class="empty-icon"><i data-lucide="${escapeHtml(icon)}" aria-hidden="true"></i></div>
      <strong>${escapeHtml(title)}</strong>
    </div>`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) {
        message = Array.isArray(body.detail)
          ? body.detail.map((item) => item.msg).join("; ")
          : body.detail;
      }
    } catch (_) {
      // Keep the HTTP status as the fallback message.
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  if (response.status === 204) {
    return null;
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json")
    ? response.json()
    : response.text();
}

function runApiPath(runId, suffix = "") {
  return `/api/runs/${encodeURIComponent(String(runId))}${suffix}`;
}

function toast(message, kind = "info") {
  const element = document.createElement("div");
  element.className = `toast ${kind}`;
  element.textContent = message;
  $("#toast-region").append(element);
  window.setTimeout(() => element.remove(), 4200);
}

function formatDate(value, includeDate = false) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(undefined, {
    ...(includeDate ? { month: "short", day: "numeric" } : {}),
    hour: "2-digit",
    minute: "2-digit",
    second: includeDate ? undefined : "2-digit",
  }).format(date);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function formatEventName(value) {
  return String(value || "Event")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function runById(runId) {
  return state.runs.find((run) => run.id === runId) || null;
}

function autoTaskName(datasetPath, goal) {
  const firstLine = String(goal || "")
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*#+\s*/, "").trim())
    .find(Boolean);
  if (firstLine) {
    const firstSentence = firstLine.split(/(?<=[.!?。！？])\s+/)[0];
    return firstSentence.length > 64 ? `${firstSentence.slice(0, 61).trimEnd()}...` : firstSentence;
  }
  const dataset = String(datasetPath || "").split(/[\\/]/).filter(Boolean).at(-1);
  return dataset || "New experiment";
}

function isTerminal(status) {
  return ["complete", "incomplete"].includes(canonicalStatus(status));
}

function canonicalStatus(status) {
  const value = String(status || "").trim().toLowerCase();
  if (["succeeded", "success", "complete", "completed"].includes(value)) return "complete";
  if (["queued", "running", "stopping"].includes(value)) return value;
  if (["failed", "cancelled", "stopped", "interrupted", "incomplete"].includes(value)) return "incomplete";
  return value || "incomplete";
}

function formatDuration(startValue, endValue = Date.now()) {
  const start = new Date(startValue || 0).getTime();
  const end = new Date(endValue || Date.now()).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end) || start <= 0 || end < start) return "";
  const totalSeconds = Math.max(0, Math.round((end - start) / 1000));
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function executionLabel(status) {
  return {
    queued: "Queued",
    running: "Running",
    stopping: "Stopping",
    complete: "Complete",
    incomplete: "Incomplete",
  }[canonicalStatus(status)] || String(status || "Incomplete");
}

function sidebarStatusLabel(status) {
  const normalized = canonicalStatus(status);
  return normalized === "complete" ? null : executionLabel(normalized);
}

function computeWorkerLabel(value) {
  if (value == null || String(value).toLowerCase() === "auto") return "Auto compute";
  const workers = Number(value);
  return Number.isFinite(workers)
    ? `${workers} worker${workers === 1 ? "" : "s"}`
    : "Auto compute";
}

function bestResultText(results) {
  const best = results.best;
  const specs = best?.metric_specs || [];
  const rawText = (spec) => {
    const value = best?.metrics?.[spec.name];
    return `${spec.name} ${spec.maximize ? "↑" : "↓"} ${value != null && Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "—"}`;
  };
  if (best && specs.length > 2) {
    return `Avg rank ${best.average_rank == null ? "—" : Number(best.average_rank).toFixed(2)} · ${rawText(specs[0])}`;
  }
  if (best && specs.length) return specs.map(rawText).join(" · ");
  return results.best_metric != null && Number.isFinite(Number(results.best_metric))
    ? Number(results.best_metric).toFixed(4)
    : null;
}

function resultPresentation(run) {
  const results = run?.observed?.results || {};
  const latest = results.latest || {};
  const bestText = bestResultText(results);
  if (bestText) {
    return { text: `Best: ${bestText}`, kind: "good" };
  }
  const outcome = String(latest.outcome || "").trim().toLowerCase();
  if (outcome) {
    const outcomeKinds = {
      bug: "bad",
      error: "bad",
      failed: "bad",
      failure: "bad",
      invalid: "bad",
      regressed_parent: "warning",
      first_success: "good",
      new_best: "good",
      improved_parent: "good",
      better: "good",
      matched_best: "neutral",
      valid_non_improving: "neutral",
    };
    return {
      text: `Result: ${formatEventName(outcome)}`,
      kind: outcomeKinds[outcome] || "neutral",
    };
  }
  if (isTerminal(run?.status)) {
    return { text: "No valid result", kind: canonicalStatus(run.status) === "complete" ? "warning" : "neutral" };
  }
  return null;
}

function revisionKey(snapshot) {
  if (!snapshot) return null;
  return [
    snapshot.run_revision,
    snapshot.tree?.revision,
    snapshot.events?.revision,
    snapshot.console?.revision,
    snapshot.artifacts?.revision,
  ].join(":");
}

function resetSelectedResources() {
  if (state.treeLoadCancel) state.treeLoadCancel();
  state.treeLoadCancel = null;
  state.treeInteracting = false;
  state.treeViewId = null;
  state.resources = emptyResourceState();
  state.updateSnapshot = null;
  state.treeLoadToken += 1;
  const frame = $("#tree-frame");
  if (frame) {
    frame.onload = null;
    frame.onerror = null;
  }
}

function renderDatasetOptions() {
  const datalist = $("#dataset-options");
  datalist.innerHTML = state.datasets
    .map((dataset) => `<option value="${escapeHtml(dataset.path)}">${escapeHtml(dataset.name)}</option>`)
    .join("");
}

function renderDatasetDialog() {
  const list = $("#dataset-list");
  if (!state.datasets.length) {
    list.innerHTML = emptyStateMarkup("folder-search", "No datasets found");
    refreshIcons();
    return;
  }
  list.innerHTML = state.datasets.map((dataset) => `
    <button class="dataset-option" type="button" data-dataset-path="${escapeHtml(dataset.path)}">
      <i data-lucide="folder" aria-hidden="true"></i>
      <span>${escapeHtml(dataset.name)}<small>${escapeHtml(dataset.path)}</small></span>
    </button>`).join("");
  list.querySelectorAll("[data-dataset-path]").forEach((button) => {
    button.addEventListener("click", () => {
      $("#dataset-path").value = button.dataset.datasetPath;
      markProjectDirty();
      $("#dataset-dialog").close();
    });
  });
  refreshIcons();
}

function runDatasetPath(run) {
  return String(run?.dataset_path || run?.project_snapshot?.dataset_path || "").trim();
}

function runLogDirectoryName(run, fallback = "") {
  const rawLogDir = String(run?.log_dir || "").replace(/[\\/]+$/, "");
  if (rawLogDir) return rawLogDir.split(/[\\/]/).filter(Boolean).at(-1) || rawLogDir;
  return queueView.readableRunName(run, fallback);
}

function runConfiguredModel(run) {
  return run?.options?.model_for_all_agents || run?.options?.main_model || "";
}

function runSetupSignature(run) {
  if (!run) return "";
  const snapshot = run.project_snapshot || {};
  const options = run.options || {};
  return JSON.stringify([
    Boolean(run.log_dir),
    run.log_dir ? runLogDirectoryName(run) : (run.actual_exp_name || run.requested_exp_name || ""),
    run.dataset_path || snapshot.dataset_path || "",
    snapshot.task_template_path || "",
    snapshot.task_body || "",
    runConfiguredModel(run) || null,
    options.agent_steps ?? null,
    options.exec_timeout ?? null,
    options.exec_parallel_workers ?? null,
    options.agent_total_time_limit ?? null,
    options.final_retrain_enabled ?? null,
  ]);
}

function groupedRunsByDataset() {
  const groups = new Map();
  for (const run of state.runs) {
    // The navigation is a view of logs/, so do not invent a temporary label
    // before the CLI has created the real log directory.
    if (!run?.log_dir) continue;
    const datasetPath = runDatasetPath(run);
    if (!datasetPath) continue;
    const datasetName = datasetPath.split(/[\\/]/).filter(Boolean).at(-1);
    if (!datasetName) continue;
    if (!groups.has(datasetName)) groups.set(datasetName, { datasetPath, datasetName, runs: [] });
    groups.get(datasetName).runs.push(run);
  }
  return Array.from(groups.values()).sort((left, right) =>
    left.datasetName.localeCompare(right.datasetName, undefined, { sensitivity: "base" }),
  );
}

function renderProjects() {
  const list = $("#project-list");
  const groups = groupedRunsByDataset();
  if (!groups.length) {
    list.innerHTML = emptyStateMarkup("folder-plus", "No experiments");
    refreshIcons();
    return;
  }
  if (list.querySelector(".empty-state")) list.textContent = "";
  const existingGroups = new Map(
    Array.from(list.querySelectorAll("[data-dataset-name]")).map((group) => [group.dataset.datasetName, group]),
  );
  const keepGroups = new Set();
  for (const group of groups) {
    let details = existingGroups.get(group.datasetName);
    if (!details) {
      details = document.createElement("details");
      details.className = "dataset-group";
      details.dataset.datasetName = group.datasetName;
      details.open = true;

      const summary = document.createElement("summary");
      summary.className = "dataset-group-summary";
      summary.title = group.datasetPath;
      const icon = document.createElement("i");
      icon.setAttribute("data-lucide", "folder");
      icon.setAttribute("aria-hidden", "true");
      const name = document.createElement("span");
      name.className = "dataset-folder-name";
      summary.append(icon, name);
      const runList = document.createElement("div");
      runList.className = "dataset-run-list";
      details.append(summary, runList);
    }
    keepGroups.add(group.datasetName);
    details.querySelector(".dataset-folder-name").textContent = group.datasetName;
    if (group.runs.some((run) => run.id === state.selectedRunId)) details.open = true;

    const runList = details.querySelector(".dataset-run-list");
    const existingRuns = new Map(
      Array.from(runList.querySelectorAll("[data-run-id]")).map((button) => [button.dataset.runId, button]),
    );
    const keepRuns = new Set();
    for (const run of group.runs) {
      let button = existingRuns.get(run.id);
      if (!button) {
        button = document.createElement("button");
        button.className = "nav-item run-nav-item";
        button.type = "button";
        button.dataset.runId = run.id;
        const title = document.createElement("span");
        title.className = "run-log-name";
        button.append(title);
        button.addEventListener("click", () => selectRun(button.dataset.runId).catch(showError));
      }
      keepRuns.add(run.id);
      const logName = runLogDirectoryName(run);
      button.querySelector(".run-log-name").textContent = logName;
      const statusLabel = sidebarStatusLabel(run.status);
      button.title = statusLabel ? `${logName} — ${statusLabel}` : logName;
      button.setAttribute("aria-label", statusLabel ? `${logName}, ${statusLabel}` : logName);
      button.classList.toggle("active", run.id === state.selectedRunId);
      button.setAttribute("aria-current", run.id === state.selectedRunId ? "true" : "false");
      let status = button.querySelector(".run-nav-status");
      if (statusLabel) {
        if (!status) {
          status = document.createElement("span");
          status.className = "run-nav-status";
          button.append(status);
        }
        status.className = `run-nav-status ${canonicalStatus(run.status)}`;
        status.textContent = statusLabel;
      } else if (status) {
        status.remove();
      }
      runList.append(button);
    }
    for (const [runId, button] of existingRuns) {
      if (!keepRuns.has(runId)) button.remove();
    }
    list.append(details);
  }
  for (const [datasetName, group] of existingGroups) {
    if (!keepGroups.has(datasetName)) group.remove();
  }
  refreshIcons();
}

function queueReasonText(reason) {
  if (!reason) return "Queue paused after an incomplete run.";
  if (typeof reason === "string") return reason;
  if (typeof reason === "object") return reason.message || reason.detail || reason.reason || "Queue paused after an incomplete run.";
  return String(reason);
}

function createQueueRow(runId) {
  const row = document.createElement("div");
  row.className = "queue-run-row";
  row.dataset.queueRunId = runId;
  row.setAttribute("role", "listitem");

  const select = document.createElement("button");
  select.className = "queue-run-select";
  select.type = "button";
  const name = document.createElement("span");
  name.className = "queue-run-name";
  const status = document.createElement("span");
  status.className = "queue-run-status";
  select.append(name, status);
  select.addEventListener("click", () => selectRun(row.dataset.queueRunId).catch(showError));

  const action = document.createElement("button");
  action.className = "queue-run-action";
  action.type = "button";
  action.hidden = true;
  action.addEventListener("click", (event) => queueRunAction(event).catch(showError));
  row.append(select, action);
  return row;
}

function updateQueueRow(row, run) {
  const normalized = canonicalStatus(run.status);
  const name = runLogDirectoryName(run, "Queued run");
  row.dataset.queueRunId = String(run.id);
  const select = row.querySelector(".queue-run-select");
  select.title = name;
  select.setAttribute("aria-label", `Open ${name}, ${executionLabel(normalized)}`);
  select.setAttribute("aria-current", run.id === state.selectedRunId ? "true" : "false");
  row.querySelector(".queue-run-name").textContent = name;
  const status = row.querySelector(".queue-run-status");
  status.className = `queue-run-status ${normalized}`;
  status.textContent = executionLabel(normalized);

  const action = row.querySelector(".queue-run-action");
  const actionName = queueView.actionForRun(run);
  if (!actionName) {
    if (document.activeElement === action) select.focus();
    action.hidden = true;
    action.disabled = false;
    delete action.dataset.runAction;
    delete action.dataset.runId;
    return;
  }
  const label = actionName === "cancel" ? "Cancel" : "Stop";
  action.hidden = false;
  action.disabled = false;
  action.dataset.runId = String(run.id);
  action.dataset.runAction = actionName;
  action.textContent = label;
  action.title = `${label} ${name}`;
  action.setAttribute("aria-label", `${label} ${name}`);
}

function renderQueue() {
  const section = $("#queue-section");
  const list = $("#queue-list");
  const queueRuns = state.runs
    .filter((run) => ["queued", "running", "stopping"].includes(canonicalStatus(run.status)))
    .sort((left, right) => {
      const leftActive = canonicalStatus(left.status) === "queued" ? 1 : 0;
      const rightActive = canonicalStatus(right.status) === "queued" ? 1 : 0;
      if (leftActive !== rightActive) return leftActive - rightActive;
      const leftPosition = Number(left.queue_position ?? left.queue_sequence);
      const rightPosition = Number(right.queue_position ?? right.queue_sequence);
      if (Number.isFinite(leftPosition) && Number.isFinite(rightPosition) && leftPosition !== rightPosition) {
        return leftPosition - rightPosition;
      }
      return String(left.timestamps?.created_at || "").localeCompare(String(right.timestamps?.created_at || ""));
    });
  const paused = Boolean(state.queue?.paused);
  section.hidden = !paused && !queueRuns.length;
  $("#resume-queue-button").hidden = !paused;
  const pausedReason = $("#queue-paused-reason");
  pausedReason.hidden = !paused;
  pausedReason.textContent = paused ? queueReasonText(state.queue?.reason) : "";

  queueView.reconcileRows(list, queueRuns, createQueueRow, updateQueueRow);
}

function renderRuns() {
  renderQueue();
  renderProjects();
  renderRunHistory();
}

function renderRunHistory() {
  const select = $("#run-history-select");
  const title = $("#selected-run-name");
  select.hidden = true;
  title.hidden = !state.selectedRun;
}

function emptyProject() {
  return {
    name: "",
    dataset_path: "",
    task_template_path: null,
    task_body: "",
    advanced_defaults: { ...DEFAULT_OPTIONS },
  };
}

function setProjectFormReadOnly(readOnly) {
  const form = $("#project-form");
  form.classList.toggle("run-snapshot", readOnly);
  form.setAttribute("aria-readonly", String(readOnly));
  form.querySelectorAll('input:not([type="hidden"]), textarea').forEach((field) => {
    if (field.type === "checkbox") field.disabled = readOnly;
    else field.readOnly = readOnly;
  });
  $("#browse-datasets-button").disabled = readOnly;
  $("#refresh-task-models-button").disabled = readOnly;
}

function populateProjectForm(project) {
  const value = project || emptyProject();
  const options = { ...DEFAULT_OPTIONS, ...(value.advanced_defaults || {}) };
  setProjectFormReadOnly(false);
  $("#project-name").value = value.name || "";
  $("#dataset-path").value = value.dataset_path || "";
  $("#task-template").value = value.task_template_path || "";
  $("#task-body").value = value.task_body || "";
  $("#task-model").value = options.model_for_all_agents || "";
  $("#agent-steps").value = options.agent_steps;
  $("#exec-timeout").value = options.exec_timeout;
  $("#total-time-limit").value = options.agent_total_time_limit;
  $("#final-retrain").checked = options.final_retrain_enabled;
  $("#run-name-label").textContent = "Run name";
  $("#run-name-optional").hidden = false;
  $("#task-model-label").textContent = "Model override";
  $("#task-model-status").textContent = "Leave blank to use the Settings default, then the repository default.";
  $("#editor-title").textContent = project?.name || "New experiment";
  $("#project-form-status").hidden = true;
  state.dirty = false;
  renderEditorState();
}

function populateRunForm(run) {
  if (!run) return;
  const snapshot = run.project_snapshot || {};
  const options = run.options || {};
  const name = runLogDirectoryName(run, "Run");
  setProjectFormReadOnly(true);
  $("#project-name").value = name;
  $("#dataset-path").value = run.dataset_path || snapshot.dataset_path || "";
  $("#task-template").value = snapshot.task_template_path || "";
  $("#task-body").value = snapshot.task_body || "";
  $("#task-model").value = runConfiguredModel(run);
  $("#agent-steps").value = options.agent_steps ?? "";
  $("#exec-timeout").value = options.exec_timeout ?? "";
  $("#total-time-limit").value = options.agent_total_time_limit ?? "";
  $("#final-retrain").checked = options.final_retrain_enabled === true;
  $("#run-name-label").textContent = "Run name";
  $("#run-name-optional").hidden = true;
  $("#task-model-label").textContent = "Model";
  $("#task-model-status").textContent = runConfiguredModel(run)
    ? "Recorded in the CLI run configuration."
    : "Not recorded by this run.";
  $("#editor-title").textContent = name || "Run configuration";
  $("#project-form-status").hidden = true;
  state.dirty = false;
  renderEditorState();
}

function updateNewTaskTitle() {
  const explicitName = $("#project-name").value.trim();
  $("#editor-title").textContent = explicitName
    || autoTaskName($("#dataset-path").value, $("#task-body").value);
}

function renderEditorState() {
  if (state.selectedRun) {
    $("#run-button").hidden = true;
    const options = state.selectedRun.options || {};
    const parts = [];
    const model = runConfiguredModel(state.selectedRun);
    if (model) parts.push(model);
    if (options.agent_steps != null) parts.push(`${options.agent_steps} steps`);
    parts.push(computeWorkerLabel(options.exec_parallel_workers));
    $("#run-config-summary").textContent = parts.length
      ? `${parts.join(" · ")} · CLI configuration`
      : "CLI configuration";
    return;
  }
  $("#run-button").hidden = false;
  $("#run-button-label").textContent = "Run";
  const options = collectProjectOptions();
  const settingsModel = configuredDefaultModel();
  const model = options.model_for_all_agents
    || (settingsModel ? `${settingsModel} (Settings)` : "repository default");
  $("#run-config-summary").textContent = `${model} · ${options.agent_steps} steps · ${computeWorkerLabel(options.exec_parallel_workers)}`;
}

function collectProjectOptions() {
  return {
    model_for_all_agents: $("#task-model").value.trim() || null,
    agent_steps: Number($("#agent-steps").value),
    exec_timeout: Number($("#exec-timeout").value),
    exec_parallel_workers: "auto",
    agent_total_time_limit: Number($("#total-time-limit").value),
    final_retrain_enabled: $("#final-retrain").checked,
  };
}

function markProjectDirty() {
  state.dirty = true;
  $("#project-form-status").hidden = true;
  updateNewTaskTitle();
  renderEditorState();
}

function collectProjectForm() {
  const datasetPath = $("#dataset-path").value.trim();
  const taskBody = $("#task-body").value.trim();
  return {
    dataset_path: datasetPath,
    task_template_path: $("#task-template").value.trim() || null,
    task_body: taskBody,
    advanced_defaults: collectProjectOptions(),
  };
}

function newProject() {
  if (state.dirty && !window.confirm("Discard unsaved changes?")) {
    return;
  }
  state.selectedRunId = null;
  state.selectedRun = null;
  resetSelectedResources();
  populateProjectForm(null);
  renderProjects();
  loadSelectedRun(true).catch(showError);
  setMobileView("setup");
  $("#project-name").focus();
}

async function queueRun() {
  const button = $("#run-button");
  button.disabled = true;
  try {
    const form = $("#project-form");
    const formStatus = $("#project-form-status");
    if (!form.reportValidity()) {
      formStatus.textContent = "Check the highlighted fields before running.";
      formStatus.hidden = false;
      throw new Error("Please correct the experiment fields");
    }
    const experiment = collectProjectForm();
    if (!experiment.dataset_path || !experiment.task_body) {
      formStatus.textContent = "Dataset and goal cannot be blank.";
      formStatus.hidden = false;
      throw new Error("Dataset and goal are required");
    }
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        experiment,
        requested_exp_name: $("#project-name").value.trim() || null,
      }),
    });
    formStatus.hidden = true;
    state.runs.unshift(run);
    state.selectedRunId = run.id;
    state.selectedRun = run;
    resetSelectedResources();
    populateRunForm(run);
    renderRuns();
    setMobileView("run", { focus: true });
    await loadSelectedRun(true);
    schedulePoll(0);
    toast("Run queued");
  } finally {
    button.disabled = false;
  }
}

async function selectRun(runId) {
  const run = runById(runId);
  if (!run) return;
  if (state.dirty && !window.confirm("Discard unsaved changes?")) {
    renderProjects();
    return;
  }
  state.selectedRunId = runId;
  state.selectedRun = run;
  resetSelectedResources();
  populateRunForm(run);
  renderRuns();
  setMobileView("run", { focus: true });
  await loadSelectedRun(true);
}

function renderRunSummary() {
  const run = state.selectedRun;
  const summary = $("#run-summary");
  summary.hidden = !run;
  $(".detail-pane").classList.toggle("no-run", !run);
  const name = run ? runLogDirectoryName(run, "Run") : "";
  $("#selected-run-name").textContent = name;
  renderRunHistory();
  if (!run) {
    $("#run-result-status").hidden = true;
    $("#run-action-button").hidden = true;
    $("#run-lifecycle-detail").hidden = true;
    return;
  }
  const execution = $("#run-execution-status");
  execution.className = `status-pill ${canonicalStatus(run.status)}`;
  execution.textContent = executionLabel(run.status);
  execution.title = `Execution status: ${executionLabel(run.status)}`;

  const result = resultPresentation(run);
  const resultPill = $("#run-result-status");
  resultPill.hidden = !result;
  if (result) {
    resultPill.className = `result-pill ${result.kind}`;
    resultPill.textContent = result.text;
  }

  const startedAt = run.timestamps?.started_at;
  const finishedAt = run.timestamps?.finished_at;
  const elapsed = formatDuration(startedAt, finishedAt || Date.now());
  const metaParts = [];
  const model = runConfiguredModel(run);
  const completedValue = run.progress?.completed_steps;
  const totalValue = run.progress?.total_steps ?? run.options?.agent_steps;
  const completedSteps = completedValue == null ? NaN : Number(completedValue);
  const totalSteps = totalValue == null ? NaN : Number(totalValue);
  if (model) metaParts.push(model);
  if (Number.isFinite(completedSteps) && Number.isFinite(totalSteps)) {
    metaParts.push(`${completedSteps}/${totalSteps} steps`);
  }
  if (startedAt) metaParts.push(`Started ${formatDate(startedAt, true)}`);
  if (elapsed) metaParts.push(elapsed);
  $("#selected-run-meta").textContent = metaParts.join(" · ");

  const actionButton = $("#run-action-button");
  const actionLabel = $("#run-action-label");
  const actionIcon = $("#run-action-icon");
  const allowedActions = new Set(run.allowed_actions || []);
  actionButton.hidden = false;
  actionButton.disabled = false;
  actionButton.className = "button secondary";
  if (allowedActions.has("cancel")) {
    actionButton.dataset.runAction = "cancel";
    actionButton.className = "button danger";
    actionLabel.textContent = "Cancel";
    actionIcon.setAttribute("data-lucide", "x");
  } else if (allowedActions.has("stop")) {
    actionButton.dataset.runAction = "stop";
    actionButton.className = "button danger";
    actionButton.disabled = run.status === "stopping";
    actionLabel.textContent = run.status === "stopping" ? "Stopping" : "Stop";
    actionIcon.setAttribute("data-lucide", "square");
  } else if (allowedActions.has("requeue")) {
    actionButton.dataset.runAction = "requeue";
    actionLabel.textContent = "Retry";
    actionIcon.setAttribute("data-lucide", "rotate-ccw");
  } else {
    actionButton.hidden = true;
  }

  const error = run.status_reason || run.reason || run.lifecycle?.reason || run.error || run.observed?.results?.latest?.error;
  const terminalEvent = run.terminal_event || run.lifecycle?.terminal_event || run.observed?.terminal_event;
  const terminalEventName = typeof terminalEvent === "string"
    ? terminalEvent
    : (terminalEvent?.event || terminalEvent?.type || terminalEvent?.name || "");
  const rawStatus = String(run.status || "");
  const detailedStatus = run.detailed_status
    || (rawStatus && rawStatus !== canonicalStatus(rawStatus) ? rawStatus : "");
  const lifecycleDetail = $("#run-lifecycle-detail");
  lifecycleDetail.className = canonicalStatus(run.status) === "incomplete"
    ? "run-lifecycle-detail incomplete"
    : "run-lifecycle-detail";
  const detailParts = [];
  if (detailedStatus) detailParts.push(`Detailed status: ${formatEventName(detailedStatus)}`);
  if (error) detailParts.push(`Reason: ${typeof error === "string" ? error : JSON.stringify(error)}`);
  if (terminalEventName) detailParts.push(`Terminal event: ${terminalEventName}`);
  lifecycleDetail.hidden = !detailParts.length;
  lifecycleDetail.textContent = detailParts.join(" · ");
  refreshIcons();
}

function clearSelectedRun() {
  state.selectedRunId = null;
  state.selectedRun = null;
  resetSelectedResources();
  populateProjectForm(null);
  renderRunSummary();
  renderTree();
  renderArtifacts();
  loadEvents();
  loadConsole();
}

async function runAction(event) {
  if (!state.selectedRunId) return;
  const action = event.currentTarget.dataset.runAction;
  const labels = { cancel: "cancel", stop: "stop", requeue: "requeue" };
  if (action !== "requeue" && !window.confirm(`${labels[action][0].toUpperCase()}${labels[action].slice(1)} this run?`)) {
    return;
  }
  const result = await api(runApiPath(state.selectedRunId, `/${action}`), { method: "POST" });
  if (action === "requeue") {
    state.runs.unshift(result);
    state.selectedRunId = result.id;
    state.selectedRun = result;
    resetSelectedResources();
    populateRunForm(result);
  }
  await refreshRuns(true);
  schedulePoll(0);
  toast(action === "requeue" ? "Run requeued" : `${action[0].toUpperCase()}${action.slice(1)} requested`);
}

async function queueRunAction(event) {
  const button = event.currentTarget;
  const runId = button.dataset.runId;
  const action = button.dataset.runAction;
  const run = runById(runId);
  if (!run || !["cancel", "stop"].includes(action)) return;
  const name = runLogDirectoryName(run, "Queued run");
  if (!window.confirm(`${action === "cancel" ? "Cancel" : "Stop"} ${name}?`)) return;
  button.disabled = true;
  try {
    await api(runApiPath(runId, `/${action}`), { method: "POST" });
    await refreshRuns(true);
    schedulePoll(0);
    toast(action === "cancel" ? "Queued run cancelled" : "Stop requested");
  } finally {
    if (button.isConnected) button.disabled = false;
  }
}

async function resumeQueue() {
  const button = $("#resume-queue-button");
  button.disabled = true;
  try {
    const generation = Number(state.queue?.pause_generation);
    if (!Number.isInteger(generation) || generation < 0) {
      throw new Error("Queue state is stale; refresh before resuming");
    }
    const response = await api("/api/queue/resume", {
      method: "POST",
      body: JSON.stringify({ pause_generation: generation }),
    });
    applyQueueControl(response);
    await refreshRuns(true);
    schedulePoll(0);
    if (state.queue?.paused) {
      toast("Queue changed while resuming; it remains paused", "error");
    } else {
      toast("Queue resumed");
    }
  } finally {
    button.disabled = false;
  }
}

async function loadSelectedRun(forceTree = false) {
  const requestedRunId = state.selectedRunId;
  if (!requestedRunId) {
    clearSelectedRun();
    return;
  }
  try {
    const snapshot = await api(runApiPath(requestedRunId, "/updates"));
    if (state.selectedRunId !== requestedRunId) return;
    await applyUpdateSnapshot(snapshot, { force: forceTree });
  } catch (error) {
    if (state.selectedRunId !== requestedRunId) return;
    if (error.message.includes("not found")) {
      clearSelectedRun();
      return;
    }
    throw error;
  }
}

async function loadRunDetail(snapshot, force = false) {
  const requestedRunId = state.selectedRunId;
  if (!requestedRunId) return;
  const runChanged = snapshot.run_revision !== state.resources.runRevision;
  const artifactsChanged = snapshot.artifacts?.revision !== state.resources.artifactsRevision;
  if (!force && !runChanged && !artifactsChanged && state.selectedRun?.observed) return;
  const selectedRun = await api(runApiPath(requestedRunId));
  if (state.selectedRunId !== requestedRunId) return;
  const previousSetupSignature = runSetupSignature(state.selectedRun);
  state.selectedRun = selectedRun;
  const index = state.runs.findIndex((item) => item.id === requestedRunId);
  if (index >= 0) state.runs[index] = selectedRun;
  state.resources.runRevision = snapshot.run_revision;
  state.resources.artifactsRevision = snapshot.artifacts?.revision || null;
  if (runSetupSignature(selectedRun) !== previousSetupSignature) {
    populateRunForm(selectedRun);
  }
  renderRuns();
  renderRunSummary();
  if (state.activeTab === "artifacts" || snapshot.final_barrier || force) renderArtifacts();
}

async function applyUpdateSnapshot(snapshot, { force = false } = {}) {
  if (!snapshot || snapshot.run_id !== state.selectedRunId) return;
  state.updateSnapshot = snapshot;
  const listedRun = runById(snapshot.run_id);
  if (listedRun && listedRun.status !== snapshot.status) listedRun.status = snapshot.status;
  if (state.selectedRun && state.selectedRun.id === snapshot.run_id) {
    state.selectedRun.status = snapshot.status;
  }
  await loadRunDetail(snapshot, force);
  if (snapshot.run_id !== state.selectedRunId) return;
  renderRunSummary();

  const loadFinalResources = Boolean(snapshot.final_barrier);
  const tasks = [];
  if (force || state.activeTab === "tree" || loadFinalResources) {
    tasks.push(renderTree(force, snapshot));
  } else {
    updateTreeAvailability(snapshot);
  }
  if ((force || state.activeTab === "events" || loadFinalResources) && snapshot.events?.available) {
    tasks.push(loadEvents({ revision: snapshot.events?.revision, force }));
  } else if (!snapshot.events?.available && state.resources.events.loaded !== "unavailable") {
    state.resources.events.loaded = "unavailable";
    $("#events-list").innerHTML = emptyStateMarkup("activity", "No activity yet");
    refreshIcons();
  }
  if ((force || state.activeTab === "console" || loadFinalResources) && snapshot.console?.available) {
    tasks.push(loadConsole({ revision: snapshot.console?.revision, force }));
  } else if (!snapshot.console?.available && state.resources.console.loaded !== "unavailable") {
    state.resources.console.loaded = "unavailable";
    $("#console-output").textContent = "No output yet.";
  }
  await Promise.all(tasks);
  if (snapshot.run_id !== state.selectedRunId) return;
  if (snapshot.final_barrier) state.resources.finalBarrier = revisionKey(snapshot);
}

function updateTreeAvailability(snapshot = state.updateSnapshot) {
  const run = state.selectedRun;
  const tree = snapshot?.tree;
  const status = $("#tree-update-status");
  const openLink = $("#tree-open-link");
  if (!run || !tree?.available || !tree.revision) {
    if (!state.resources.tree.loaded) {
      $("#tree-frame").hidden = true;
      $("#tree-empty").hidden = false;
      $("#tree-empty strong").textContent = run && refreshPolicy.isActive(run.status)
        ? "Tree will appear after the first search update"
        : "No tree data for this run";
    }
    openLink.hidden = true;
    status.textContent = run && refreshPolicy.isActive(run.status) ? "Waiting for tree" : "No tree";
    return;
  }
  openLink.hidden = false;
  openLink.href = tree.url;
  const decision = refreshPolicy.treeDecision({
    available: true,
    revision: tree.revision,
    loadedRevision: state.resources.tree.loaded,
    loadingRevision: state.resources.tree.loading,
    autoUpdate: $("#tree-auto-refresh").checked && !state.treeInteracting,
  });
  if (decision === "pending") {
    state.resources.tree.pending = tree.revision;
    status.textContent = "Update available";
    status.className = "tree-update-status pending";
  } else if (decision === "current") {
    status.textContent = state.resources.tree.loading ? "Updating…" : "Up to date";
    status.className = `tree-update-status${state.resources.tree.loading ? " loading" : ""}`;
  } else if (decision === "load") {
    state.resources.tree.pending = tree.revision;
    status.textContent = "New revision ready";
    status.className = "tree-update-status pending";
  }
}

function renderTree(force = false, snapshot = state.updateSnapshot) {
  const run = state.selectedRun;
  const frame = $("#tree-frame");
  const empty = $("#tree-empty");
  const openLink = $("#tree-open-link");
  const status = $("#tree-update-status");
  const tree = snapshot?.tree;
  if (!run || !tree?.available || !tree.revision || !tree.url) {
    updateTreeAvailability(snapshot);
    return Promise.resolve();
  }
  openLink.hidden = false;
  openLink.href = tree.url;
  const decision = refreshPolicy.treeDecision({
    available: true,
    revision: tree.revision,
    loadedRevision: state.resources.tree.loaded,
    loadingRevision: state.resources.tree.loading,
    autoUpdate: $("#tree-auto-refresh").checked && !state.treeInteracting,
    force,
  });
  if (decision === "current") {
    updateTreeAvailability(snapshot);
    return Promise.resolve();
  }
  if (decision === "pending") {
    state.resources.tree.pending = tree.revision;
    status.textContent = "Update available";
    status.className = "tree-update-status pending";
    return Promise.resolve();
  }

  const requestedRunId = run.id;
  const requestedRevision = tree.revision;
  const token = ++state.treeLoadToken;
  const viewId = `${token}:${requestedRevision}`;
  state.treeViewId = viewId;
  const previousCancel = state.treeLoadCancel;
  state.treeLoadCancel = null;
  if (previousCancel) previousCancel();
  state.resources.tree.loading = requestedRevision;
  status.textContent = "Updating…";
  status.className = "tree-update-status loading";
  empty.hidden = true;
  frame.hidden = false;
  const url = new URL(tree.url, window.location.origin);
  url.searchParams.set("embed", "1");
  url.searchParams.set("parent_origin", window.location.origin);
  url.searchParams.set("revision", requestedRevision);
  url.searchParams.set("view_id", viewId);
  if (force) url.searchParams.set("manual", String(Date.now()));

  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (loaded) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      if (state.treeLoadCancel === cancel) state.treeLoadCancel = null;
      if (state.selectedRunId !== requestedRunId || token !== state.treeLoadToken) {
        resolve();
        return;
      }
      state.resources.tree.loading = null;
      if (loaded) {
        state.resources.tree.loaded = requestedRevision;
        state.resources.tree.pending = null;
        status.textContent = `Updated ${formatDate(Date.now())}`;
        status.className = "tree-update-status";
      } else {
        frame.hidden = true;
        empty.hidden = false;
        empty.querySelector("strong").textContent = "Tree could not be loaded";
        status.textContent = "Update failed · retrying";
        status.className = "tree-update-status error";
        reject(new Error("Tree update could not be loaded"));
        return;
      }
      resolve();
    };
    const cancel = () => finish(false);
    state.treeLoadCancel = cancel;
    const timeout = window.setTimeout(() => finish(false), 15000);
    frame.onload = () => finish(true);
    frame.onerror = () => finish(false);
    frame.src = url.toString();
  });
}

async function loadConsole({ revision = state.updateSnapshot?.console?.revision, force = false } = {}) {
  const output = $("#console-output");
  const requestedRunId = state.selectedRunId;
  if (!requestedRunId) {
    output.textContent = "No output yet.";
    return;
  }
  if (!force && revision && revision === state.resources.console.loaded) return;
  if (!force && revision && revision === state.resources.console.loading) return;
  const loadingRevision = revision || `manual-${Date.now()}`;
  state.resources.console.loading = loadingRevision;
  try {
    const consoleText = await api(runApiPath(requestedRunId, "/log?tail=1200"));
    if (state.selectedRunId !== requestedRunId) return;
    output.textContent = consoleText;
    if (!output.textContent) output.textContent = "No output yet.";
    if ($("#console-follow").checked) output.scrollTop = output.scrollHeight;
    state.resources.console.loaded = revision || state.resources.console.loaded;
  } finally {
    if (state.selectedRunId === requestedRunId && state.resources.console.loading === loadingRevision) {
      state.resources.console.loading = null;
    }
  }
}

async function loadEvents({ revision = state.updateSnapshot?.events?.revision, force = false } = {}) {
  const list = $("#events-list");
  const requestedRunId = state.selectedRunId;
  if (!requestedRunId) {
    list.innerHTML = emptyStateMarkup("activity", "No activity yet");
    refreshIcons();
    return;
  }
  if (!force && revision && revision === state.resources.events.loaded) return;
  if (!force && revision && revision === state.resources.events.loading) return;
  const loadingRevision = revision || `manual-${Date.now()}`;
  state.resources.events.loading = loadingRevision;
  let response;
  try {
    response = await api(runApiPath(requestedRunId, "/events?tail=200"));
    if (state.selectedRunId !== requestedRunId) return;
    state.resources.events.loaded = revision || state.resources.events.loaded;
  } finally {
    if (state.selectedRunId === requestedRunId && state.resources.events.loading === loadingRevision) {
      state.resources.events.loading = null;
    }
  }
  if (!response || state.selectedRunId !== requestedRunId) return;
  if (!response.events.length) {
    list.innerHTML = emptyStateMarkup("activity", "No activity yet");
    refreshIcons();
    return;
  }
  list.innerHTML = response.events
    .slice()
    .reverse()
    .map((event) => {
      const payload = { ...event };
      delete payload.ts;
      delete payload.event;
      const payloadText = Object.keys(payload).length ? JSON.stringify(payload, null, 2) : "";
      return `
        <article class="event-row">
          <span class="event-marker" aria-hidden="true"></span>
          <span class="event-name">${escapeHtml(formatEventName(event.event))}</span>
          <time class="event-time">${escapeHtml(formatDate(event.ts))}</time>
          ${payloadText ? `<details class="event-details"><summary>Details</summary><pre class="event-payload">${escapeHtml(payloadText)}</pre></details>` : ""}
        </article>`;
    })
    .join("");
}

function renderArtifacts() {
  const list = $("#artifact-list");
  const observation = $("#run-observation");
  const run = state.selectedRun;
  if (!run) {
    observation.innerHTML = "";
    list.innerHTML = emptyStateMarkup("files", "No files yet");
    refreshIcons();
    return;
  }
  const observed = run.observed || {};
  const results = observed.results || {};
  const lastEvent = observed.last_event?.event || "—";
  observation.innerHTML = `
    <span>Events <strong>${escapeHtml(observed.event_count ?? 0)}</strong></span>
    <span>Nodes <strong>${escapeHtml(results.count ?? 0)}</strong></span>
    <span>Selected best ${results.best?.node_id ? escapeHtml(results.best.node_id) : ""} <strong>${escapeHtml(bestResultText(results) || "—")}</strong></span>
    <span>Last event <strong>${escapeHtml(lastEvent)}</strong></span>`;
  if (!run.artifacts?.length) {
    list.innerHTML = emptyStateMarkup("files", "No files yet");
    refreshIcons();
    return;
  }
  list.innerHTML = run.artifacts
    .map((artifact) => `
      <div class="artifact-row">
        <i data-lucide="file" aria-hidden="true"></i>
        <a class="artifact-name" href="${escapeHtml(artifact.url)}" target="_blank" rel="noopener">${escapeHtml(artifact.name)}</a>
        <span class="artifact-size">${escapeHtml(formatBytes(artifact.size))}</span>
        <time class="artifact-time">${escapeHtml(formatDate(artifact.modified_at * 1000))}</time>
      </div>`)
    .join("");
  refreshIcons();
}

function setActiveTab(tabName) {
  state.activeTab = tabName;
  const systemActive = tabName === "system";
  $(".detail-pane").classList.toggle("system-view", systemActive);
  $("#system-resource-summary").setAttribute("aria-pressed", String(systemActive));
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.tab === tabName;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  $$(".tab-panel").forEach((panel) => {
    const active = panel.id === `${tabName}-panel`;
    panel.classList.toggle("active", active);
    panel.setAttribute("aria-hidden", String(!active));
  });
  if (tabName === "tree") renderTree().catch(showError);
  if (tabName === "console" && state.updateSnapshot?.console?.available) loadConsole().catch(showError);
  if (tabName === "events" && state.updateSnapshot?.events?.available) loadEvents().catch(showError);
  if (tabName === "artifacts") renderArtifacts();
  if (tabName !== "tree" && $(".workbench").classList.contains("tree-focus")) toggleTreeFocus(false);
}

function openSystemResources() {
  setActiveTab("system");
  setMobileView("run", { focus: true });
}

function setMobileView(view, { focus = false } = {}) {
  if (!["tasks", "setup", "run"].includes(view)) return;
  state.mobileView = view;
  $(".workbench").dataset.mobileView = view;
  $$(".mobile-view-button").forEach((button) => {
    const active = button.dataset.mobileView === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  });
  if (focus && window.matchMedia("(max-width: 960px)").matches) {
    const target = {
      tasks: $("#new-project-button"),
      setup: $("#project-name"),
      run: $(".tab.active"),
    }[view];
    window.requestAnimationFrame(() => target?.focus({ preventScroll: true }));
  }
}

function toggleTreeFocus(force) {
  if (state.activeTab !== "tree" && force !== false) setActiveTab("tree");
  const workbench = $(".workbench");
  const next = typeof force === "boolean" ? force : !workbench.classList.contains("tree-focus");
  workbench.classList.toggle("tree-focus", next);
  const button = $("#run-focus-button");
  button.setAttribute("aria-pressed", String(next));
  button.title = next ? "Exit tree focus" : "Focus tree";
  button.setAttribute("aria-label", button.title);
  const icon = button.querySelector("[data-lucide]");
  if (icon) icon.setAttribute("data-lucide", next ? "minimize-2" : "maximize-2");
  refreshIcons();
}

function handleTabKeydown(event) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = $$(".tab");
  const currentIndex = tabs.indexOf(event.currentTarget);
  if (currentIndex < 0) return;
  event.preventDefault();
  let nextIndex;
  if (event.key === "Home") nextIndex = 0;
  else if (event.key === "End") nextIndex = tabs.length - 1;
  else if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % tabs.length;
  else nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
  tabs[nextIndex].focus();
  setActiveTab(tabs[nextIndex].dataset.tab);
}

function trapDialogFocus(event) {
  const dialog = event.currentTarget;
  if (event.key !== "Tab" || !dialog.open) return;
  const focusable = Array.from(
    dialog.querySelectorAll(
      'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])',
    ),
  ).filter(
    (element) => element.getClientRects().length > 0 && !element.closest("[inert]"),
  );
  const first = focusable[0];
  const last = focusable.at(-1);
  if (!first || !last) return;
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function renderEnvironmentFields() {
  const baseUrl = state.env.OPENAI_BASE_URL?.value || "";
  $("#settings-base-url").value = baseUrl;
  for (const [key, field] of Object.entries(SECRET_FIELDS)) {
    const input = $(field.input);
    input.value = "";
    input.type = "password";
    input.dataset.changed = "false";
    input.placeholder = state.env[key]?.masked || "";
    $(field.status).textContent = state.env[key]?.configured ? "" : "Not configured";
    updateSecretToggle(key, false);
  }
  const currentModel = configuredDefaultModel();
  $("#settings-model").value = currentModel;
  $("#settings-model").dataset.changed = "false";
  $("#settings-model-status").textContent = currentModel
    ? "Default for WebUI runs. Task overrides take precedence."
    : "Using the repository default. Task overrides take precedence.";
  const source = state.envSource || {};
  $("#settings-env-source").textContent = source.path
    ? `Saved in ${source.path}${source.exists ? "" : " (created when you save)"}`
    : "Settings apply to this local BioMaker workspace.";
  refreshIcons();
}

async function openSettings() {
  const response = await api("/api/env");
  state.env = response.values;
  state.envSource = response.source || {};
  renderEnvironmentFields();
  $("#settings-dialog").showModal();
  discoverModels("settings").catch((error) => showModelError(error, "settings"));
}

function configuredDefaultModel() {
  return state.env.BIOMAKER_MAIN_MODEL?.value || "";
}

function modelSuggestions(query = "", selected = "") {
  const normalizedQuery = query.trim().toLowerCase();
  return state.models
    .map((model, index) => ({ model, index, lower: model.toLowerCase() }))
    .filter(({ lower }) => !normalizedQuery || lower.includes(normalizedQuery))
    .sort((left, right) => {
      if (left.model === selected) return -1;
      if (right.model === selected) return 1;
      const leftPreferred = PREFERRED_MODELS.indexOf(left.model);
      const rightPreferred = PREFERRED_MODELS.indexOf(right.model);
      if (leftPreferred !== rightPreferred) {
        if (leftPreferred < 0) return 1;
        if (rightPreferred < 0) return -1;
        return leftPreferred - rightPreferred;
      }
      if (normalizedQuery) {
        const leftStarts = left.lower.startsWith(normalizedQuery);
        const rightStarts = right.lower.startsWith(normalizedQuery);
        if (leftStarts !== rightStarts) return leftStarts ? -1 : 1;
      }
      return left.index - right.index;
    })
    .slice(0, MAX_MODEL_SUGGESTIONS)
    .map(({ model }) => model);
}

function renderModelOptions(target, query = "", selected = "") {
  $(target).innerHTML = modelSuggestions(query, selected)
    .map((model) => `<option value="${escapeHtml(model)}"></option>`)
    .join("");
}

function setModelOptions(models) {
  state.models = [...new Set(models.filter(Boolean))];
  const taskModel = $("#task-model").value.trim();
  const settingsModel = $("#settings-model").value.trim();
  renderModelOptions("#task-model-options", taskModel, taskModel);
  renderModelOptions("#settings-model-options", settingsModel, settingsModel);
}

function showModelError(error, source = "task") {
  console.error(error);
  $(source === "settings" ? "#settings-model-status" : "#task-model-status").textContent = error.message || String(error);
}

function invalidateModelDiscovery() {
  modelDiscoveryGeneration += 1;
  state.models = [];
  $("#task-model-options").innerHTML = "";
  $("#settings-model-options").innerHTML = "";
  $("#task-model-status").textContent = "Leave blank to use the Settings default, then the repository default.";
  $("#settings-model-status").textContent = "Refresh to load models from the configured endpoint.";
  $("#refresh-task-models-button").disabled = false;
  $("#refresh-models-button").disabled = false;
}

async function discoverModels(source = "task") {
  const fromSettings = source === "settings";
  const button = $(fromSettings ? "#refresh-models-button" : "#refresh-task-models-button");
  const status = $(fromSettings ? "#settings-model-status" : "#task-model-status");
  const modelInput = $(fromSettings ? "#settings-model" : "#task-model");
  const discoveryGeneration = ++modelDiscoveryGeneration;
  button.disabled = true;
  state.models = [];
  $("#task-model-options").innerHTML = "";
  $("#settings-model-options").innerHTML = "";
  status.textContent = "Loading...";
  try {
    const apiKeyInput = $("#settings-api-key");
    const payload = fromSettings
      ? {
          base_url: $("#settings-base-url").value.trim() || null,
          api_key: apiKeyInput.dataset.changed === "true" ? (apiKeyInput.value.trim() || null) : null,
        }
      : {};
    const response = await api("/api/models/discover", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    if (discoveryGeneration !== modelDiscoveryGeneration) return;
    const models = response.models || [];
    const selected = modelInput.value.trim();
    setModelOptions(models);
    status.textContent = response.status === "ok"
      ? `${models.length} models available.${selected ? "" : " Enter or select one."}`
      : (response.detail || "Models unavailable");
  } catch (error) {
    if (discoveryGeneration !== modelDiscoveryGeneration) return;
    throw error;
  } finally {
    if (discoveryGeneration === modelDiscoveryGeneration) button.disabled = false;
  }
}

function updateSecretToggle(key, revealed) {
  const field = SECRET_FIELDS[key];
  if (!field) return;
  const button = $(`.secret-toggle[data-env-key="${key}"]`);
  if (!button) return;
  const action = revealed ? "Hide" : "Show";
  button.title = `${action} ${field.label}`;
  button.setAttribute("aria-label", `${action} ${field.label}`);
  button.setAttribute("aria-pressed", String(revealed));
  button.innerHTML = `<span class="icon-fallback" aria-hidden="true">${action}</span><i data-lucide="${revealed ? "eye-off" : "eye"}" aria-hidden="true"></i>`;
}

async function toggleSecret(event) {
  const button = event.currentTarget;
  const key = button.dataset.envKey;
  const field = SECRET_FIELDS[key];
  if (!field) return;
  const input = $(field.input);
  if (input.type === "text") {
    if (input.dataset.changed !== "true") input.value = "";
    input.type = "password";
    updateSecretToggle(key, false);
    refreshIcons();
    return;
  }
  if (!input.value && state.env[key]?.configured) {
    const revealGeneration = secretReadGeneration;
    button.disabled = true;
    try {
      const response = await api("/api/env/reveal", {
        method: "POST",
        body: JSON.stringify({ key }),
      });
      if (revealGeneration !== secretReadGeneration || !$("#settings-dialog").open) return;
      input.value = response.value || "";
    } finally {
      if (revealGeneration === secretReadGeneration) button.disabled = false;
    }
  }
  input.type = "text";
  updateSecretToggle(key, true);
  refreshIcons();
}

function clearVisibleSecrets() {
  secretReadGeneration += 1;
  invalidateModelDiscovery();
  for (const [key, field] of Object.entries(SECRET_FIELDS)) {
    const input = $(field.input);
    const button = $(`.secret-toggle[data-env-key="${key}"]`);
    input.value = "";
    input.type = "password";
    input.dataset.changed = "false";
    if (button) button.disabled = false;
    updateSecretToggle(key, false);
  }
  refreshIcons();
}

async function saveEnvironment(event) {
  event.preventDefault();
  if (event.submitter?.value === "cancel") {
    clearVisibleSecrets();
    $("#settings-dialog").close();
    return;
  }
  const model = $("#settings-model").value.trim();
  const updates = {
    OPENAI_BASE_URL: $("#settings-base-url").value.trim() || null,
    BIOMAKER_MAIN_MODEL: model || null,
  };
  for (const [key, field] of Object.entries(SECRET_FIELDS)) {
    const input = $(field.input);
    if (input.dataset.changed === "true") updates[key] = input.value.trim() || null;
  }
  if (Object.keys(updates).length) {
    const response = await api("/api/env", {
      method: "PUT",
      body: JSON.stringify(updates),
    });
    state.env = response.values;
    state.envSource = response.source || state.envSource;
    invalidateModelDiscovery();
    renderEditorState();
  }
  clearVisibleSecrets();
  $("#settings-dialog").close();
  toast("Settings saved");
}

function setConnectionIndicator(message = "", kind = "") {
  const indicator = $("#connection-indicator");
  indicator.textContent = message;
  indicator.className = `connection-indicator${kind ? ` ${kind}` : ""}`;
  indicator.hidden = !message;
}

function resourceMetric(label, value) {
  const metric = document.createElement("div");
  metric.className = "system-resource-metric";
  const name = document.createElement("span");
  name.textContent = label;
  const reading = document.createElement("strong");
  reading.textContent = value;
  metric.append(name, reading);
  return metric;
}

function renderSystemResources(snapshot) {
  state.systemResourceSnapshot = snapshot;
  const summary = $("#system-resource-summary");
  summary.textContent = systemResourceView.buildSummary(snapshot);

  const freshness = $("#system-resource-freshness");
  const sampledAt = formatDate(snapshot?.sampled_at, true);
  freshness.textContent = `${snapshot?.stale ? "Stale" : "Sampled"}${sampledAt ? ` ${sampledAt}` : ""}`;

  const overview = $("#system-resource-overview");
  overview.replaceChildren(
    resourceMetric("CPU", systemResourceView.formatPercent(snapshot?.cpu?.utilization_percent)),
    resourceMetric(
      "RAM",
      `${systemResourceView.formatPercent(snapshot?.memory?.utilization_percent)} · ${systemResourceView.formatMemoryUsage(snapshot?.memory?.used_bytes, snapshot?.memory?.total_bytes)}`,
    ),
  );

  const gpuList = $("#system-resource-gpus");
  gpuList.replaceChildren();
  const gpus = Array.isArray(snapshot?.nvidia_gpus) ? snapshot.nvidia_gpus : [];
  for (const gpu of gpus) {
    const card = document.createElement("section");
    card.className = "system-resource-gpu";
    const name = document.createElement("strong");
    name.textContent = `GPU ${gpu.index ?? "?"} · ${gpu.name || "NVIDIA GPU"}`;
    const metrics = document.createElement("div");
    metrics.className = "system-resource-gpu-metrics";
    metrics.append(
      resourceMetric("Utilization", systemResourceView.formatPercent(gpu.utilization_percent)),
      resourceMetric("Memory", systemResourceView.formatMemoryUsage(gpu.memory_used_bytes, gpu.memory_total_bytes)),
      resourceMetric(
        "Temperature",
        gpu.temperature_celsius != null && Number.isFinite(Number(gpu.temperature_celsius))
          ? `${Math.round(Number(gpu.temperature_celsius))} °C`
          : "—",
      ),
    );
    card.append(name, metrics);
    gpuList.append(card);
  }
  const appleGpu = snapshot?.apple_gpu;
  if (!gpus.length && appleGpu) {
    const apple = document.createElement("section");
    apple.className = "system-resource-gpu";
    const title = document.createElement("strong");
    title.textContent = "Apple GPU";
    const metrics = document.createElement("div");
    metrics.className = "system-resource-gpu-metrics";
    metrics.append(
      resourceMetric("Utilization", systemResourceView.formatPercent(appleGpu?.utilization_percent)),
      resourceMetric("Shared memory", systemResourceView.formatBytes(appleGpu?.shared_memory_used_bytes)),
    );
    apple.append(title, metrics);
    gpuList.append(apple);
  } else if (!gpus.length) {
    const unavailable = document.createElement("p");
    unavailable.className = "system-resource-gpu-empty";
    unavailable.textContent = "No GPU metrics available.";
    gpuList.append(unavailable);
  }

}

function renderSystemResourceFailure(message) {
  const staleSnapshot = systemResourceView.markStale(state.systemResourceSnapshot, message);
  if (staleSnapshot) {
    renderSystemResources(staleSnapshot);
    return;
  }
  $("#system-resource-summary").textContent = "System · unavailable";
  $("#system-resource-freshness").textContent = "Unavailable";
  $("#system-resource-overview").replaceChildren();
  $("#system-resource-gpus").replaceChildren();
}

function scheduleSystemResourcePoll(delay = SYSTEM_RESOURCE_POLL_INTERVAL_MS) {
  if (state.systemResourceTimer) window.clearTimeout(state.systemResourceTimer);
  state.systemResourceTimer = null;
  if (document.hidden || state.systemResourceEndpointAvailable === false) return;
  state.systemResourceTimer = window.setTimeout(() => {
    state.systemResourceTimer = null;
    pollSystemResources().catch((error) => console.error(error));
  }, Math.max(0, delay));
}

async function pollSystemResources() {
  if (state.systemResourcePolling || document.hidden || state.systemResourceEndpointAvailable === false) return;
  if (state.systemResourceTimer) window.clearTimeout(state.systemResourceTimer);
  state.systemResourceTimer = null;
  state.systemResourcePolling = true;
  const controller = new AbortController();
  state.systemResourceAbortController = controller;
  const timeout = window.setTimeout(
    () => controller.abort(),
    SYSTEM_RESOURCE_REQUEST_TIMEOUT_MS,
  );
  try {
    const snapshot = await api("/api/system/resources", { signal: controller.signal });
    state.systemResourceEndpointAvailable = true;
    renderSystemResources(snapshot);
  } catch (error) {
    if ([404, 405].includes(error.status)) {
      state.systemResourceEndpointAvailable = false;
      renderSystemResourceFailure("System resource monitoring is unavailable on this server.");
    } else {
      renderSystemResourceFailure("System resource refresh failed; retrying.");
    }
  } finally {
    window.clearTimeout(timeout);
    if (state.systemResourceAbortController === controller) {
      state.systemResourceAbortController = null;
    }
    state.systemResourcePolling = false;
    scheduleSystemResourcePoll();
  }
}

function handleVisibilityChange() {
  if (document.hidden) {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
    if (state.systemResourceTimer) window.clearTimeout(state.systemResourceTimer);
    state.systemResourceTimer = null;
    state.systemResourceAbortController?.abort();
    return;
  }
  schedulePoll(0);
  scheduleSystemResourcePoll(0);
}

function nextPollDelay() {
  return refreshPolicy.pollDelay({
    hidden: document.hidden,
    hasActive: state.runs.some((run) => refreshPolicy.isActive(run.status)),
    failures: state.pollFailures,
  });
}

function applyQueueControl(value) {
  if (!value || typeof value !== "object") return false;
  const control = value.queue_control || value.queue || value;
  if (!control || typeof control !== "object" || !("paused" in control)) return false;
  state.queue = {
    ...state.queue,
    ...control,
    paused: Boolean(control.paused),
    reason: control.reason ?? null,
  };
  return true;
}

async function fetchQueueControl() {
  if (state.queueEndpointAvailable === false) return;
  try {
    const control = await api("/api/queue");
    state.queueEndpointAvailable = true;
    applyQueueControl(control);
  } catch (error) {
    if ([404, 405].includes(error.status)) {
      state.queueEndpointAvailable = false;
      return;
    }
    console.warn("Queue status is temporarily unavailable", error);
  }
}

async function fetchRunsState() {
  const payload = await api("/api/runs");
  if (Array.isArray(payload)) {
    await fetchQueueControl();
    return payload;
  }
  if (!payload || !Array.isArray(payload.runs)) {
    throw new Error("Run list returned an invalid response");
  }
  const hasEmbeddedQueue = applyQueueControl(payload.queue_control || payload.queue);
  if (!hasEmbeddedQueue) await fetchQueueControl();
  return payload.runs;
}

function schedulePoll(delay = nextPollDelay(), options = {}) {
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
  if (delay == null || document.hidden || state.pollStopped) return;
  state.pollTimer = window.setTimeout(() => {
    state.pollTimer = null;
    pollNow(options).catch((error) => console.error(error));
  }, Math.max(0, delay));
}

async function refreshRunData(force = false) {
  const runs = await fetchRunsState();
  state.runs = runs;
  const listedSelectedRun = state.selectedRunId ? runById(state.selectedRunId) : null;
  if (listedSelectedRun) {
    const previousSetupSignature = runSetupSignature(state.selectedRun);
    state.selectedRun = state.selectedRun
      ? { ...state.selectedRun, ...listedSelectedRun }
      : listedSelectedRun;
    if (runSetupSignature(state.selectedRun) !== previousSetupSignature) {
      populateRunForm(state.selectedRun);
    }
  }
  renderRuns();
  renderRunSummary();
  if (state.selectedRunId && listedSelectedRun) {
    await loadSelectedRun(force);
  } else if (state.selectedRunId || state.selectedRun) {
    clearSelectedRun();
  }
}

async function pollNow({ force = false, manual = false } = {}) {
  if (state.polling) {
    const queued = state.pollQueued || { force: false, manual: false };
    state.pollQueued = { force: queued.force || force, manual: queued.manual || manual };
    return;
  }
  if (state.pollTimer) window.clearTimeout(state.pollTimer);
  state.pollTimer = null;
  state.polling = true;
  const refreshButton = $("#refresh-button");
  if (manual) refreshButton.setAttribute("aria-busy", "true");
  try {
    await refreshRunData(force);
    state.pollFailures = 0;
    setConnectionIndicator();
  } catch (error) {
    state.pollFailures += 1;
    const delay = refreshPolicy.pollDelay({ failures: state.pollFailures, hidden: document.hidden });
    const seconds = delay == null ? "when this tab is visible" : `in ${Math.round(delay / 1000)}s`;
    setConnectionIndicator(`Update paused · retrying ${seconds}`, "error");
    console.error(error);
    if (manual) toast(error.message || String(error), "error");
  } finally {
    state.polling = false;
    refreshButton.removeAttribute("aria-busy");
    const queued = state.pollQueued;
    state.pollQueued = false;
    if (queued) schedulePoll(0, queued);
    else schedulePoll(nextPollDelay());
  }
}

async function refreshRuns(force = false) {
  return pollNow({ force, manual: force });
}

function showError(error) {
  console.error(error);
  toast(error.message || String(error), "error");
}

function bindEvents() {
  $("#new-project-button").addEventListener("click", newProject);
  $("#project-form").addEventListener("submit", (event) => {
    event.preventDefault();
    queueRun().catch(showError);
  });
  $("#run-button").addEventListener("click", () => queueRun().catch(showError));
  $("#run-action-button").addEventListener("click", (event) => runAction(event).catch(showError));
  $("#resume-queue-button").addEventListener("click", () => resumeQueue().catch(showError));
  $("#settings-button").addEventListener("click", () => openSettings().catch(showError));
  $("#settings-form").addEventListener("submit", (event) => saveEnvironment(event).catch(showError));
  $("#settings-dialog").addEventListener("close", clearVisibleSecrets);
  $("#settings-dialog").addEventListener("keydown", trapDialogFocus);
  $("#dataset-dialog").addEventListener("keydown", trapDialogFocus);
  $("#settings-base-url").addEventListener("input", invalidateModelDiscovery);
  $("#refresh-task-models-button").addEventListener("click", () => discoverModels("task").catch(showModelError));
  $("#refresh-models-button").addEventListener("click", () => {
    discoverModels("settings").catch((error) => showModelError(error, "settings"));
  });
  $$(".secret-toggle").forEach((button) => {
    button.addEventListener("click", (event) => toggleSecret(event).catch(showError));
  });
  for (const [key, field] of Object.entries(SECRET_FIELDS)) {
    $(field.input).addEventListener("input", (event) => {
      event.currentTarget.dataset.changed = "true";
      if (key === "OPENAI_API_KEY") invalidateModelDiscovery();
    });
  }
  $("#task-model").addEventListener("input", (event) => {
    renderModelOptions("#task-model-options", event.currentTarget.value, event.currentTarget.value);
  });
  $("#settings-model").addEventListener("input", (event) => {
    event.currentTarget.dataset.changed = "true";
    renderModelOptions("#settings-model-options", event.currentTarget.value, event.currentTarget.value);
  });
  $("#system-resource-summary").addEventListener("click", openSystemResources);
  $("#refresh-button").addEventListener("click", () => refreshRuns(true).catch(showError));
  $("#tree-refresh-button").addEventListener("click", () => renderTree(true).catch(showError));
  $("#tree-auto-refresh").addEventListener("change", (event) => {
    if (event.currentTarget.checked) renderTree().catch(showError);
    else updateTreeAvailability();
  });
  $("#run-focus-button").addEventListener("click", () => toggleTreeFocus());
  $("#console-refresh-button").addEventListener("click", () => loadConsole({ force: true }).catch(showError));
  $("#browse-datasets-button").addEventListener("click", () => {
    renderDatasetDialog();
    $("#dataset-dialog").showModal();
    $("#dataset-list .dataset-option")?.focus();
  });
  $("#close-dataset-dialog").addEventListener("click", () => $("#dataset-dialog").close());
  $("#run-history-select").addEventListener("change", (event) => selectRun(event.currentTarget.value).catch(showError));
  $$(".mobile-view-button").forEach((button) => {
    button.addEventListener("click", () => setMobileView(button.dataset.mobileView));
  });
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
    tab.addEventListener("keydown", handleTabKeydown);
  });
  $$("#project-form input, #project-form select, #project-form textarea").forEach((field) => {
    field.addEventListener("input", markProjectDirty);
    field.addEventListener("change", markProjectDirty);
  });
  document.addEventListener("visibilitychange", handleVisibilityChange);
  window.addEventListener("online", () => {
    schedulePoll(0);
    if (state.systemResourceEndpointAvailable !== false) scheduleSystemResourcePoll(0);
  });
  window.addEventListener("offline", () => setConnectionIndicator("Offline · waiting for connection", "error"));
  window.addEventListener("message", (event) => {
    const frame = $("#tree-frame");
    if (!frame.contentWindow || event.source !== frame.contentWindow || !event.data || typeof event.data !== "object") return;
    if (!state.treeViewId || event.data.view_id !== state.treeViewId) return;
    const type = event.data.type;
    if (["biomaker:tree-state", "biomaker-tree-state"].includes(type)) {
      if (state.selectedRunId) state.treeViewState[state.selectedRunId] = event.data.state || event.data.payload || {};
    }
    if (type === "biomaker:tree-state-request") {
      const savedState = state.selectedRunId ? state.treeViewState[state.selectedRunId] : null;
      if (savedState && typeof event.data.key === "string") {
        frame.contentWindow.postMessage({
          type: "biomaker:tree-state-restore",
          key: event.data.key,
          state: savedState,
          view_id: state.treeViewId,
        }, "*");
      }
    }
    if (["biomaker:tree-interaction", "biomaker-tree-interaction"].includes(type)) {
      state.treeInteracting = Boolean(event.data.active);
      if (!state.treeInteracting && state.resources.tree.pending) renderTree().catch(showError);
    }
  });
  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
}

async function initialize() {
  bindEvents();
  refreshIcons();
  const [runs, datasets, environment] = await Promise.all([
    fetchRunsState(),
    api("/api/datasets"),
    api("/api/env"),
  ]);
  state.runs = runs;
  state.datasets = datasets;
  state.env = environment.values;
  state.envSource = environment.source || {};
  renderDatasetOptions();
  setMobileView("setup");
  populateProjectForm(null);
  renderRuns();
  const activeRun = refreshPolicy.initialRun(runs);
  if (activeRun) await selectRun(activeRun.id);
  else await loadSelectedRun(true);
  if (activeRun) setMobileView("run");
  schedulePoll(nextPollDelay());
  scheduleSystemResourcePoll(0);
}

document.addEventListener("DOMContentLoaded", () => initialize().catch(showError));
