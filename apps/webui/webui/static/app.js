"use strict";

const DEFAULT_OPTIONS = {
  model_for_all_agents: null,
  agent_steps: 35,
  exec_timeout: 600,
  exec_parallel_workers: 1,
  agent_total_time_limit: 86400,
  final_retrain_enabled: true,
};

const SECRET_FIELDS = {
  OPENAI_API_KEY: { input: "#settings-api-key", status: "#settings-api-key-status", label: "API key" },
  SERPER_API_KEY: { input: "#settings-serper-key", status: "#settings-serper-key-status", label: "Serper key" },
  JINA_API_KEY: { input: "#settings-jina-key", status: "#settings-jina-key-status", label: "Jina key" },
};

const state = {
  projects: [],
  runs: [],
  datasets: [],
  env: {},
  envDefaults: {},
  selectedProjectId: null,
  selectedRunId: null,
  selectedRun: null,
  activeTab: "events",
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

function formatEventName(value) {
  return String(value || "Event")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function projectById(projectId) {
  return state.projects.find((project) => project.id === projectId) || null;
}

function runById(runId) {
  return state.runs.find((run) => run.id === runId) || null;
}

function runsForProject(projectId) {
  return state.runs.filter((run) => run.project_id === projectId);
}

function latestRunForProject(projectId) {
  return runsForProject(projectId)[0] || null;
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
  return dataset || "New task";
}

function isTerminal(status) {
  return ["succeeded", "failed", "cancelled", "stopped", "interrupted"].includes(status);
}

function renderDatasetOptions() {
  const datalist = $("#dataset-options");
  datalist.innerHTML = state.datasets
    .map((dataset) => `<option value="${escapeHtml(dataset.path)}">${escapeHtml(dataset.name)}</option>`)
    .join("");
}

function renderProjects() {
  const list = $("#project-list");
  if (!state.projects.length) {
    list.innerHTML = emptyStateMarkup("folder-plus", "No tasks");
    refreshIcons();
    return;
  }
  list.innerHTML = state.projects
    .map((project) => {
      const dataset = project.dataset_path.split(/[\\/]/).filter(Boolean).at(-1) || project.dataset_path;
      const latestRun = latestRunForProject(project.id);
      return `
        <button class="nav-item ${project.id === state.selectedProjectId ? "active" : ""}" type="button" data-project-id="${escapeHtml(project.id)}">
          <span class="nav-item-main">
            <span class="nav-item-title">${escapeHtml(project.name)}</span>
            <span class="nav-item-meta">${escapeHtml(dataset)}</span>
          </span>
          ${latestRun ? `<span class="status-dot ${escapeHtml(latestRun.status)}" title="${escapeHtml(latestRun.status)}" role="img" aria-label="Latest run: ${escapeHtml(latestRun.status)}"></span>` : ""}
        </button>`;
    })
    .join("");
  list.querySelectorAll("[data-project-id]").forEach((button) => {
    button.addEventListener("click", () => selectProject(button.dataset.projectId).catch(showError));
  });
  refreshIcons();
}

function renderRuns() {
  const queued = state.runs.filter((run) => run.status === "queued").length;
  const running = state.runs.filter((run) => ["running", "stopping"].includes(run.status)).length;
  const queueIndicator = $("#queue-indicator");
  const queueParts = [];
  if (running) queueParts.push(`${running} running`);
  if (queued) queueParts.push(`${queued} queued`);
  queueIndicator.textContent = queueParts.join(" · ");
  queueIndicator.hidden = !queueParts.length;
  queueIndicator.classList.toggle("busy", Boolean(queueParts.length));
  renderProjects();
  renderRunHistory();
}

function renderRunHistory() {
  const select = $("#run-history-select");
  const title = $("#selected-run-name");
  const runs = state.selectedProjectId ? runsForProject(state.selectedProjectId) : [];
  select.innerHTML = runs
    .map((run) => {
      const name = run.actual_exp_name || run.requested_exp_name;
      const label = `${name} · ${formatDate(run.timestamps?.created_at, true)}`;
      return `<option value="${escapeHtml(run.id)}" ${run.id === state.selectedRunId ? "selected" : ""}>${escapeHtml(label)}</option>`;
    })
    .join("");
  const hasHistory = runs.length > 1;
  select.hidden = !hasHistory;
  title.hidden = hasHistory || !state.selectedRun;
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
  $("#agent-steps").value = options.agent_steps;
  $("#exec-timeout").value = options.exec_timeout;
  $("#parallel-workers").value = options.exec_parallel_workers;
  $("#total-time-limit").value = options.agent_total_time_limit;
  $("#final-retrain").checked = options.final_retrain_enabled;
  $("#editor-title").textContent = project?.name || "New task";
  $("#delete-project-button").hidden = true;
  state.dirty = false;
}

function updateNewTaskTitle() {
  if (state.selectedProjectId) return;
  $("#editor-title").textContent = autoTaskName($("#dataset-path").value, $("#task-body").value);
}

function collectProjectForm() {
  const datasetPath = $("#dataset-path").value.trim();
  const taskBody = $("#task-body").value.trim();
  const existing = projectById(state.selectedProjectId);
  return {
    name: existing?.name || autoTaskName(datasetPath, taskBody),
    dataset_path: datasetPath,
    task_template_path: existing?.task_template_path || null,
    task_body: taskBody,
    advanced_defaults: {
      model_for_all_agents: null,
      agent_steps: Number($("#agent-steps").value),
      exec_timeout: Number($("#exec-timeout").value),
      exec_parallel_workers: Number($("#parallel-workers").value),
      agent_total_time_limit: Number($("#total-time-limit").value),
      final_retrain_enabled: $("#final-retrain").checked,
    },
  };
}

async function selectProject(projectId) {
  const project = projectById(projectId);
  if (!project) return;
  if (projectId === state.selectedProjectId) {
    renderProjects();
    renderRunHistory();
    return;
  }
  if (state.dirty && !window.confirm("Discard unsaved changes?")) {
    return;
  }
  state.selectedProjectId = projectId;
  populateProjectForm(project);
  renderProjects();
  const latestRun = latestRunForProject(projectId);
  state.selectedRunId = latestRun?.id || null;
  await loadSelectedRun(true);
  renderRuns();
}

function newProject() {
  if (state.dirty && !window.confirm("Discard unsaved changes?")) {
    return;
  }
  state.selectedProjectId = null;
  state.selectedRunId = null;
  state.selectedRun = null;
  populateProjectForm(null);
  renderProjects();
  loadSelectedRun(true).catch(showError);
  $("#dataset-path").focus();
}

async function saveProject({ quiet = false } = {}) {
  const payload = collectProjectForm();
  if (!payload.dataset_path || !payload.task_body) {
    throw new Error("Dataset and goal are required");
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
  if (!quiet) toast("Task saved");
  return project;
}

async function deleteProject(projectId = state.selectedProjectId) {
  if (!projectId) return;
  const project = projectById(projectId);
  if (!project) return;
  if (!window.confirm(`Delete task “${project?.name || ""}”? Runs will be kept.`)) {
    return;
  }
  const deletedIndex = state.projects.findIndex((item) => item.id === projectId);
  await api(`/api/projects/${projectId}`, { method: "DELETE" });
  state.projects = state.projects.filter((item) => item.id !== projectId);

  if (state.selectedProjectId !== projectId) {
    renderRuns();
    toast("Task deleted");
    return;
  }

  state.dirty = false;
  state.selectedProjectId = null;
  const nextIndex = Math.min(Math.max(deletedIndex, 0), state.projects.length - 1);
  const nextProject = state.projects[nextIndex] || null;
  if (nextProject) {
    await selectProject(nextProject.id);
  } else {
    newProject();
  }
  toast("Task deleted");
}

async function queueRun() {
  const button = $("#run-button");
  button.disabled = true;
  try {
    const project = await saveProject({ quiet: true });
    const run = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        project_id: project.id,
        requested_exp_name: null,
      }),
    });
    state.runs.unshift(run);
    state.selectedRunId = run.id;
    renderRuns();
    await loadSelectedRun(true);
    toast("Run queued");
  } finally {
    button.disabled = false;
  }
}

async function selectRun(runId) {
  const run = runById(runId);
  const project = run ? projectById(run.project_id) : null;
  if (project && project.id !== state.selectedProjectId) {
    if (state.dirty && !window.confirm("Discard unsaved changes?")) return;
    state.selectedProjectId = project.id;
    populateProjectForm(project);
  }
  state.selectedRunId = runId;
  state.treeRunId = null;
  state.lastTreeRefresh = 0;
  renderRuns();
  await loadSelectedRun(true);
}

function renderRunSummary() {
  const run = state.selectedRun;
  const summary = $("#run-summary");
  summary.hidden = !run;
  $(".detail-pane").classList.toggle("no-run", !run);
  const name = run?.actual_exp_name || run?.requested_exp_name || "";
  $("#selected-run-name").textContent = name;
  renderRunHistory();
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
  if (actionButton) actionButton.addEventListener("click", (event) => runAction(event).catch(showError));
  refreshIcons();
}

function clearSelectedRun() {
  state.selectedRunId = null;
  state.selectedRun = null;
  state.treeRunId = null;
  state.lastTreeRefresh = 0;
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
  const result = await api(`/api/runs/${state.selectedRunId}/${action}`, { method: "POST" });
  if (action === "requeue") {
    state.runs.unshift(result);
    state.selectedRunId = result.id;
    state.treeRunId = null;
  }
  await refreshRuns(true);
  toast(action === "requeue" ? "Run requeued" : `${action[0].toUpperCase()}${action.slice(1)} requested`);
}

async function loadSelectedRun(forceTree = false) {
  const requestedRunId = state.selectedRunId;
  if (!requestedRunId) {
    clearSelectedRun();
    return;
  }
  let selectedRun;
  try {
    selectedRun = await api(`/api/runs/${requestedRunId}`);
  } catch (error) {
    if (state.selectedRunId !== requestedRunId) return;
    if (error.message.includes("not found")) {
      clearSelectedRun();
      return;
    }
    throw error;
  }
  if (state.selectedRunId !== requestedRunId) return;
  state.selectedRun = selectedRun;
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
    frame.removeAttribute("src");
    openLink.hidden = true;
    openLink.removeAttribute("href");
    empty.hidden = false;
    empty.querySelector("strong").textContent = "No search data yet";
    state.treeRunId = null;
    state.lastTreeRefresh = 0;
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
  const requestedRunId = state.selectedRunId;
  if (!requestedRunId) {
    output.textContent = "No output yet.";
    return;
  }
  const consoleText = await api(`/api/runs/${requestedRunId}/log?tail=1200`);
  if (state.selectedRunId !== requestedRunId) return;
  output.textContent = consoleText;
  if (!output.textContent) output.textContent = "No output yet.";
  if ($("#console-follow").checked) output.scrollTop = output.scrollHeight;
}

async function loadEvents() {
  const list = $("#events-list");
  const requestedRunId = state.selectedRunId;
  if (!requestedRunId) {
    list.innerHTML = emptyStateMarkup("activity", "No activity yet");
    refreshIcons();
    return;
  }
  const response = await api(`/api/runs/${requestedRunId}/events?tail=200`);
  if (state.selectedRunId !== requestedRunId) return;
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
    <span>Best metric <strong>${results.best_metric == null ? "—" : escapeHtml(Number(results.best_metric).toFixed(4))}</strong></span>
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
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.tab === tabName;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  $$(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${tabName}-panel`);
  });
  if (tabName === "tree") renderTree();
  if (tabName === "console") loadConsole().catch(showError);
  if (tabName === "events") loadEvents().catch(showError);
  if (tabName === "artifacts") renderArtifacts();
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

function renderEnvironmentFields() {
  const baseUrl = state.env.OPENAI_BASE_URL?.value || "";
  $("#settings-base-url").value = baseUrl;
  for (const [key, field] of Object.entries(SECRET_FIELDS)) {
    const input = $(field.input);
    input.value = "";
    input.type = "password";
    input.dataset.changed = "false";
    $(field.status).textContent = state.env[key]?.configured ? "Configured" : "Not configured";
    updateSecretToggle(key, false);
  }
  const currentModel = state.envDefaults.main_model
    || state.env.BIOMAKER_MAIN_MODEL?.value
    || state.env.OPENAI_MODEL?.value
    || "";
  setModelOptions(currentModel ? [currentModel] : [], currentModel);
  $("#settings-model-status").textContent = "";
  refreshIcons();
}

async function openSettings() {
  const response = await api("/api/env");
  state.env = response.values;
  state.envDefaults = response.defaults || {};
  renderEnvironmentFields();
  $("#settings-dialog").showModal();
  discoverModels().catch(showModelError);
}

function setModelOptions(models, selected) {
  const uniqueModels = [...new Set([selected, ...models].filter(Boolean))];
  $("#settings-model-options").innerHTML = uniqueModels
    .map((model) => `<option value="${escapeHtml(model)}"></option>`)
    .join("");
  $("#settings-model").value = selected || "";
}

function showModelError(error) {
  console.error(error);
  $("#settings-model-status").textContent = error.message || String(error);
}

async function discoverModels() {
  const button = $("#refresh-models-button");
  const status = $("#settings-model-status");
  button.disabled = true;
  status.textContent = "Loading...";
  try {
    const apiKeyInput = $("#settings-api-key");
    const apiKey = apiKeyInput.dataset.changed === "true" ? apiKeyInput.value : null;
    const response = await api("/api/models/discover", {
      method: "POST",
      body: JSON.stringify({
        base_url: $("#settings-base-url").value.trim() || null,
        api_key: apiKey || null,
      }),
    });
    const selected = $("#settings-model").value || response.default_model || "";
    setModelOptions(response.models || [], selected);
    status.textContent = response.status === "ok" ? "" : (response.detail || "Models unavailable");
  } finally {
    button.disabled = false;
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
    input.type = "password";
    updateSecretToggle(key, false);
    refreshIcons();
    return;
  }
  if (!input.value && state.env[key]?.configured) {
    button.disabled = true;
    try {
      const response = await api("/api/env/reveal", {
        method: "POST",
        body: JSON.stringify({ key }),
      });
      input.value = response.value || "";
    } finally {
      button.disabled = false;
    }
  }
  input.type = "text";
  updateSecretToggle(key, true);
  refreshIcons();
}

function clearVisibleSecrets() {
  for (const [key, field] of Object.entries(SECRET_FIELDS)) {
    const input = $(field.input);
    input.value = "";
    input.type = "password";
    input.dataset.changed = "false";
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
    BIOMAKER_HELPER_MODEL: model || null,
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
    state.envDefaults = response.defaults || state.envDefaults;
  }
  clearVisibleSecrets();
  $("#settings-dialog").close();
  toast("Settings saved");
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
      state.dirty = true;
      $("#dataset-dialog").close();
    });
  });
  refreshIcons();
}

async function refreshRuns(force = false) {
  if (state.polling) return;
  state.polling = true;
  try {
    state.runs = await api("/api/runs");
    renderRuns();
    if (state.selectedRunId) {
      await loadSelectedRun(force);
    } else if (state.selectedRun) {
      clearSelectedRun();
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
  $("#settings-button").addEventListener("click", () => openSettings().catch(showError));
  $("#settings-form").addEventListener("submit", (event) => saveEnvironment(event).catch(showError));
  $("#settings-dialog").addEventListener("close", clearVisibleSecrets);
  $("#refresh-models-button").addEventListener("click", () => discoverModels().catch(showModelError));
  $$(".secret-toggle").forEach((button) => {
    button.addEventListener("click", (event) => toggleSecret(event).catch(showError));
  });
  for (const field of Object.values(SECRET_FIELDS)) {
    $(field.input).addEventListener("input", (event) => { event.currentTarget.dataset.changed = "true"; });
  }
  $("#refresh-button").addEventListener("click", () => refreshRuns(true).catch(showError));
  $("#tree-refresh-button").addEventListener("click", () => renderTree(true));
  $("#console-refresh-button").addEventListener("click", () => loadConsole().catch(showError));
  $("#browse-datasets-button").addEventListener("click", () => {
    renderDatasetDialog();
    $("#dataset-dialog").showModal();
  });
  $("#close-dataset-dialog").addEventListener("click", () => $("#dataset-dialog").close());
  $("#run-history-select").addEventListener("change", (event) => selectRun(event.currentTarget.value).catch(showError));
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
    tab.addEventListener("keydown", handleTabKeydown);
  });
  $$("#project-form input, #project-form select, #project-form textarea").forEach((field) => {
    field.addEventListener("input", () => {
      state.dirty = true;
      updateNewTaskTitle();
    });
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
  const [projects, runs, datasets, environment] = await Promise.all([
    api("/api/projects"),
    api("/api/runs"),
    api("/api/datasets"),
    api("/api/env"),
  ]);
  state.projects = projects;
  state.runs = runs;
  state.datasets = datasets;
  state.env = environment.values;
  state.envDefaults = environment.defaults || {};
  renderDatasetOptions();
  const activeRun = runs.find((run) => ["running", "stopping", "queued"].includes(run.status));
  const initialProject = projectById(activeRun?.project_id) || projects[0] || null;
  if (initialProject) {
    await selectProject(initialProject.id);
  } else {
    newProject();
  }
  window.setInterval(() => refreshRuns().catch(showError), 2000);
}

document.addEventListener("DOMContentLoaded", () => initialize().catch(showError));
