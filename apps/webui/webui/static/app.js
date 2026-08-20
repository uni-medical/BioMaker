"use strict";

const DEFAULT_OPTIONS = {
  model_for_all_agents: null,
  agent_steps: 35,
  exec_timeout: 600,
  exec_parallel_workers: 1,
  agent_total_time_limit: 86400,
  final_retrain_enabled: true,
};

const ENV_KEYS = [
  "OPENAI_API_KEY",
  "OPENAI_BASE_URL",
  "SERPER_API_KEY",
  "JINA_API_KEY",
];

const state = {
  projects: [],
  runs: [],
  templates: [],
  datasets: [],
  env: {},
  selectedProjectId: null,
  selectedRunId: null,
  selectedRun: null,
  activeTab: "tree",
  dirty: false,
  polling: false,
  treeRunId: null,
  lastTreeRefresh: 0,
};

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
    throw new Error(message);
  }
  if (response.status === 204) {
    return null;
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json")
    ? response.json()
    : response.text();
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

function projectById(projectId) {
  return state.projects.find((project) => project.id === projectId) || null;
}

function runById(runId) {
  return state.runs.find((run) => run.id === runId) || null;
}

function isTerminal(status) {
  return ["succeeded", "failed", "cancelled", "stopped", "interrupted"].includes(status);
}

function renderTemplateOptions() {
  const select = $("#task-template");
  select.innerHTML = '<option value="">Custom task</option>';
  for (const template of state.templates) {
    const option = document.createElement("option");
    option.value = template.path;
    option.textContent = template.name;
    select.append(option);
  }
  const datalist = $("#dataset-options");
  datalist.innerHTML = state.datasets
    .map((dataset) => `<option value="${escapeHtml(dataset.path)}">${escapeHtml(dataset.name)}</option>`)
    .join("");
}

function renderProjects() {
  const list = $("#project-list");
  if (!state.projects.length) {
    list.innerHTML = '<div class="empty-state"><span>No saved projects</span></div>';
    return;
  }
  list.innerHTML = state.projects
    .map((project) => {
      const dataset = project.dataset_path.split(/[\\/]/).filter(Boolean).at(-1) || project.dataset_path;
      return `
        <button class="nav-item ${project.id === state.selectedProjectId ? "active" : ""}" type="button" data-project-id="${escapeHtml(project.id)}">
          <span class="nav-item-main">
            <span class="nav-item-title">${escapeHtml(project.name)}</span>
            <span class="nav-item-meta">${escapeHtml(dataset)}</span>
          </span>
          <i data-lucide="chevron-right" aria-hidden="true"></i>
        </button>`;
    })
    .join("");
  list.querySelectorAll("[data-project-id]").forEach((button) => {
    button.addEventListener("click", () => selectProject(button.dataset.projectId));
  });
  refreshIcons();
}

function renderRuns() {
  const list = $("#run-list");
  $("#run-count").textContent = String(state.runs.length);
  const queued = state.runs.filter((run) => run.status === "queued").length;
  const running = state.runs.filter((run) => ["running", "stopping"].includes(run.status)).length;
  const queueIndicator = $("#queue-indicator");
  queueIndicator.textContent = running || queued ? `${running} running · ${queued} queued` : "Queue idle";
  queueIndicator.classList.toggle("busy", Boolean(running || queued));

  if (!state.runs.length) {
    list.innerHTML = '<div class="empty-state"><span>No runs yet</span></div>';
    return;
  }
  list.innerHTML = state.runs
    .map((run) => {
      const name = run.actual_exp_name || run.requested_exp_name;
      const project = projectById(run.project_id);
      const projectName = project?.name || run.project_snapshot?.project_name || "Deleted project";
      return `
        <button class="nav-item ${run.id === state.selectedRunId ? "active" : ""}" type="button" data-run-id="${escapeHtml(run.id)}">
          <span class="nav-item-main">
            <span class="nav-item-title">${escapeHtml(name)}</span>
            <span class="nav-item-meta">${escapeHtml(projectName)} · ${escapeHtml(formatDate(run.timestamps?.created_at, true))}</span>
          </span>
          <span class="status-dot ${escapeHtml(run.status)}" title="${escapeHtml(run.status)}"></span>
        </button>`;
    })
    .join("");
  list.querySelectorAll("[data-run-id]").forEach((button) => {
    button.addEventListener("click", () => selectRun(button.dataset.runId));
  });
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

function populateProjectForm(project) {
  const value = project || emptyProject();
  const options = { ...DEFAULT_OPTIONS, ...(value.advanced_defaults || {}) };
  $("#project-name").value = value.name || "";
  $("#dataset-path").value = value.dataset_path || "";
  $("#task-template").value = value.task_template_path || "";
  $("#task-body").value = value.task_body || "";
  $("#model-name").value = options.model_for_all_agents || "";
  $("#agent-steps").value = options.agent_steps;
  $("#exec-timeout").value = options.exec_timeout;
  $("#parallel-workers").value = options.exec_parallel_workers;
  $("#total-time-limit").value = options.agent_total_time_limit;
  $("#final-retrain").checked = options.final_retrain_enabled;
  $("#experiment-name").value = "";
  $("#editor-title").textContent = project?.name || "New project";
  $("#header-project").textContent = project?.name || "New project";
  $("#delete-project-button").hidden = !project;
  state.dirty = false;
}

function collectProjectForm() {
  return {
    name: $("#project-name").value.trim(),
    dataset_path: $("#dataset-path").value.trim(),
    task_template_path: $("#task-template").value || null,
    task_body: $("#task-body").value.trim(),
    advanced_defaults: {
      model_for_all_agents: $("#model-name").value.trim() || null,
      agent_steps: Number($("#agent-steps").value),
      exec_timeout: Number($("#exec-timeout").value),
      exec_parallel_workers: Number($("#parallel-workers").value),
      agent_total_time_limit: Number($("#total-time-limit").value),
      final_retrain_enabled: $("#final-retrain").checked,
    },
  };
}

function selectProject(projectId) {
  if (state.dirty && !window.confirm("Discard unsaved project changes?")) {
    return;
  }
  const project = projectById(projectId);
  if (!project) return;
  state.selectedProjectId = projectId;
  populateProjectForm(project);
  renderProjects();
}

function newProject() {
  if (state.dirty && !window.confirm("Discard unsaved project changes?")) {
    return;
  }
  state.selectedProjectId = null;
  populateProjectForm(null);
  renderProjects();
  $("#project-name").focus();
}

async function saveProject({ quiet = false } = {}) {
  const payload = collectProjectForm();
  if (!payload.name || !payload.dataset_path || !payload.task_body) {
    throw new Error("Project name, dataset path, and task goal are required");
  }
  const existing = state.selectedProjectId;
  const project = await api(existing ? `/api/projects/${existing}` : "/api/projects", {
    method: existing ? "PUT" : "POST",
    body: JSON.stringify(payload),
  });
  const index = state.projects.findIndex((item) => item.id === project.id);
  if (index >= 0) state.projects[index] = project;
  else state.projects.unshift(project);
  state.selectedProjectId = project.id;
  state.dirty = false;
  populateProjectForm(project);
  renderProjects();
  if (!quiet) toast("Project saved");
  return project;
}

async function deleteProject() {
  if (!state.selectedProjectId) return;
  const project = projectById(state.selectedProjectId);
  if (!window.confirm(`Delete project “${project?.name || ""}”? Run records will be kept.`)) {
    return;
  }
  await api(`/api/projects/${state.selectedProjectId}`, { method: "DELETE" });
  state.projects = state.projects.filter((item) => item.id !== state.selectedProjectId);
  state.selectedProjectId = state.projects[0]?.id || null;
  populateProjectForm(projectById(state.selectedProjectId));
  renderProjects();
  toast("Project deleted");
}

async function queueRun() {
  const button = $("#run-button");
  button.disabled = true;
  try {
    const project = await saveProject({ quiet: true });
    const requestedName = $("#experiment-name").value.trim();
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        project_id: project.id,
        requested_exp_name: requestedName || null,
      }),
    });
    state.runs.unshift(run);
    state.selectedRunId = run.id;
    $("#experiment-name").value = "";
    renderRuns();
    await loadSelectedRun(true);
    toast("Run added to queue");
  } finally {
    button.disabled = false;
  }
}

async function selectRun(runId) {
  state.selectedRunId = runId;
  state.treeRunId = null;
  state.lastTreeRefresh = 0;
  renderRuns();
  await loadSelectedRun(true);
}

function renderRunSummary() {
  const run = state.selectedRun;
  const name = run?.actual_exp_name || run?.requested_exp_name || "No run selected";
  $("#selected-run-name").textContent = name;
  const actions = $("#selected-run-actions");
  if (!run) {
    actions.innerHTML = "";
    return;
  }
  let command = "";
  if (run.status === "queued") {
    command = '<button class="button danger" type="button" data-run-action="cancel"><i data-lucide="x" aria-hidden="true"></i>Cancel</button>';
  } else if (["running", "stopping"].includes(run.status)) {
    command = `<button class="button danger" type="button" data-run-action="stop" ${run.status === "stopping" ? "disabled" : ""}><i data-lucide="square" aria-hidden="true"></i>${run.status === "stopping" ? "Stopping" : "Stop"}</button>`;
  } else if (isTerminal(run.status)) {
    command = '<button class="button secondary" type="button" data-run-action="requeue"><i data-lucide="rotate-ccw" aria-hidden="true"></i>Requeue</button>';
  }
  actions.innerHTML = `<span class="status-pill ${escapeHtml(run.status)}">${escapeHtml(run.status)}</span>${command}`;
  const actionButton = actions.querySelector("[data-run-action]");
  if (actionButton) actionButton.addEventListener("click", runAction);
  refreshIcons();
}

async function runAction(event) {
  if (!state.selectedRunId) return;
  const action = event.currentTarget.dataset.runAction;
  const labels = { cancel: "cancel", stop: "stop", requeue: "requeue" };
  if (action !== "requeue" && !window.confirm(`${labels[action][0].toUpperCase()}${labels[action].slice(1)} this run?`)) {
    return;
  }
  const result = await api(`/api/runs/${state.selectedRunId}/${action}`, { method: "POST" });
  if (action === "requeue") {
    state.runs.unshift(result);
    state.selectedRunId = result.id;
    state.treeRunId = null;
  }
  await refreshRuns(true);
  toast(action === "requeue" ? "Run requeued" : `Run ${action} requested`);
}

async function loadSelectedRun(forceTree = false) {
  if (!state.selectedRunId) {
    state.selectedRun = null;
    renderRunSummary();
    renderTree();
    renderArtifacts();
    return;
  }
  try {
    state.selectedRun = await api(`/api/runs/${state.selectedRunId}`);
  } catch (error) {
    if (error.message.includes("not found")) {
      state.selectedRunId = null;
      state.selectedRun = null;
      renderRunSummary();
      return;
    }
    throw error;
  }
  renderRunSummary();
  if (state.activeTab === "tree") renderTree(forceTree);
  if (state.activeTab === "console") await loadConsole();
  if (state.activeTab === "events") await loadEvents();
  if (state.activeTab === "artifacts") renderArtifacts();
}

function renderTree(force = false) {
  const run = state.selectedRun;
  const frame = $("#tree-frame");
  const empty = $("#tree-empty");
  const openLink = $("#tree-open-link");
  const hasTree = Boolean(run?.artifacts?.some((artifact) => artifact.name === "tree_plot.html"));
  if (!run || !hasTree) {
    frame.hidden = true;
    openLink.hidden = true;
    openLink.removeAttribute("href");
    empty.hidden = false;
    empty.querySelector("span").textContent = run
      ? "Tree plot is not available yet"
      : "Select a run with a tree plot";
    return;
  }
  empty.hidden = true;
  frame.hidden = false;
  openLink.hidden = false;
  openLink.href = `/artifacts/runs/${encodeURIComponent(run.id)}/tree_plot.html`;
  const shouldRefresh =
    force ||
    state.treeRunId !== run.id ||
    (["running", "stopping"].includes(run.status) &&
      $("#tree-auto-refresh").checked &&
      Date.now() - state.lastTreeRefresh > 8000);
  if (shouldRefresh) {
    frame.src = `/artifacts/runs/${encodeURIComponent(run.id)}/tree_plot.html?v=${Date.now()}`;
    state.treeRunId = run.id;
    state.lastTreeRefresh = Date.now();
  }
}

async function loadConsole() {
  const output = $("#console-output");
  if (!state.selectedRunId) {
    output.textContent = "Select a run to view console output.";
    return;
  }
  output.textContent = await api(`/api/runs/${state.selectedRunId}/log?tail=1200`);
  if (!output.textContent) output.textContent = "Console output is not available yet.";
  if ($("#console-follow").checked) output.scrollTop = output.scrollHeight;
}

async function loadEvents() {
  const list = $("#events-list");
  if (!state.selectedRunId) {
    list.innerHTML = '<div class="empty-state"><span>Select a run to view events</span></div>';
    return;
  }
  const response = await api(`/api/runs/${state.selectedRunId}/events?tail=200`);
  if (!response.events.length) {
    list.innerHTML = '<div class="empty-state"><span>No structured events yet</span></div>';
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
          <span class="event-name">${escapeHtml(event.event || "event")}</span>
          <time class="event-time">${escapeHtml(formatDate(event.ts))}</time>
          ${payloadText ? `<pre class="event-payload">${escapeHtml(payloadText)}</pre>` : ""}
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
    list.innerHTML = '<div class="empty-state"><span>Select a run to view artifacts</span></div>';
    return;
  }
  const observed = run.observed || {};
  const results = observed.results || {};
  const lastEvent = observed.last_event?.event || "—";
  observation.innerHTML = `
    <span>Events <strong>${escapeHtml(observed.event_count ?? 0)}</strong></span>
    <span>Nodes <strong>${escapeHtml(results.count ?? 0)}</strong></span>
    <span>Best metric <strong>${results.best_metric == null ? "—" : escapeHtml(Number(results.best_metric).toFixed(4))}</strong></span>
    <span>Last event <strong>${escapeHtml(lastEvent)}</strong></span>`;
  if (!run.artifacts?.length) {
    list.innerHTML = '<div class="empty-state"><span>No artifacts yet</span></div>';
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
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.tab === tabName;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $$(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${tabName}-panel`);
  });
  if (tabName === "tree") renderTree();
  if (tabName === "console") loadConsole().catch(showError);
  if (tabName === "events") loadEvents().catch(showError);
  if (tabName === "artifacts") renderArtifacts();
}

function renderEnvironmentFields() {
  $("#env-fields").innerHTML = ENV_KEYS.map((key) => {
    const item = state.env[key] || { configured: false, masked: "" };
    return `
      <div class="env-row" data-env-key="${key}">
        <label class="field">
          <span>${key}</span>
          <input type="${key.includes("KEY") ? "password" : "text"}" autocomplete="off" placeholder="${escapeHtml(item.configured ? "Leave blank to keep current value" : "Not configured")}">
        </label>
        <span class="env-status">${escapeHtml(item.masked || "not set")}</span>
        <label class="clear-secret">
          <input type="checkbox" ${item.configured ? "" : "disabled"}>
          Clear
        </label>
      </div>`;
  }).join("");
}

async function openSettings() {
  const response = await api("/api/env");
  state.env = response.values;
  renderEnvironmentFields();
  $("#settings-dialog").showModal();
}

async function saveEnvironment(event) {
  event.preventDefault();
  if (event.submitter?.value === "cancel") {
    $("#settings-dialog").close();
    return;
  }
  const updates = {};
  $$(".env-row").forEach((row) => {
    const key = row.dataset.envKey;
    const input = row.querySelector("input:not([type=checkbox])");
    const clear = row.querySelector("input[type=checkbox]");
    if (clear.checked) updates[key] = null;
    else if (input.value) updates[key] = input.value;
  });
  if (Object.keys(updates).length) {
    const response = await api("/api/env", {
      method: "PUT",
      body: JSON.stringify(updates),
    });
    state.env = response.values;
  }
  $("#settings-dialog").close();
  toast("Environment settings saved");
}

function renderDatasetDialog() {
  const list = $("#dataset-list");
  if (!state.datasets.length) {
    list.innerHTML = '<div class="empty-state"><span>No workspace datasets found</span></div>';
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
      state.dirty = true;
      $("#dataset-dialog").close();
    });
  });
  refreshIcons();
}

function applyTemplate() {
  const selected = state.templates.find((template) => template.path === $("#task-template").value);
  if (!selected) return;
  $("#task-body").value = selected.body;
  if (selected.dataset_path) $("#dataset-path").value = selected.dataset_path;
  if (!$("#project-name").value.trim()) $("#project-name").value = selected.name;
  state.dirty = true;
}

async function refreshRuns(force = false) {
  if (state.polling) return;
  state.polling = true;
  try {
    state.runs = await api("/api/runs");
    renderRuns();
    if (state.selectedRunId && (force || runById(state.selectedRunId))) {
      await loadSelectedRun(force);
    }
  } finally {
    state.polling = false;
  }
}

function showError(error) {
  console.error(error);
  toast(error.message || String(error), "error");
}

function bindEvents() {
  $("#new-project-button").addEventListener("click", newProject);
  $("#save-project-button").addEventListener("click", () => saveProject().catch(showError));
  $("#delete-project-button").addEventListener("click", () => deleteProject().catch(showError));
  $("#run-button").addEventListener("click", () => queueRun().catch(showError));
  $("#task-template").addEventListener("change", applyTemplate);
  $("#settings-button").addEventListener("click", () => openSettings().catch(showError));
  $("#settings-form").addEventListener("submit", (event) => saveEnvironment(event).catch(showError));
  $("#refresh-button").addEventListener("click", () => refreshRuns(true).catch(showError));
  $("#tree-refresh-button").addEventListener("click", () => renderTree(true));
  $("#console-refresh-button").addEventListener("click", () => loadConsole().catch(showError));
  $("#browse-datasets-button").addEventListener("click", () => {
    renderDatasetDialog();
    $("#dataset-dialog").showModal();
  });
  $("#close-dataset-dialog").addEventListener("click", () => $("#dataset-dialog").close());
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => setActiveTab(tab.dataset.tab)));
  $$("#project-form input, #project-form select, #project-form textarea").forEach((field) => {
    field.addEventListener("input", () => { state.dirty = true; });
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
  const [projects, runs, templates, datasets, environment] = await Promise.all([
    api("/api/projects"),
    api("/api/runs"),
    api("/api/task-templates"),
    api("/api/datasets"),
    api("/api/env"),
  ]);
  state.projects = projects;
  state.runs = runs;
  state.templates = templates;
  state.datasets = datasets;
  state.env = environment.values;
  renderTemplateOptions();
  state.selectedProjectId = projects[0]?.id || null;
  populateProjectForm(projectById(state.selectedProjectId));
  renderProjects();
  renderRuns();
  if (runs[0]) {
    state.selectedRunId = runs[0].id;
    await loadSelectedRun(true);
    renderRuns();
  } else {
    renderRunSummary();
    renderTree();
  }
  window.setInterval(() => refreshRuns().catch(showError), 2000);
}

document.addEventListener("DOMContentLoaded", () => initialize().catch(showError));
