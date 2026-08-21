const bgCol = "#F7F7F7";
const accentCol = "#3B82F6";
const bestCol = "#B88712";
const nodeCol = "#666666";
const bugCol = "#B4493D";
const rootCol = "#50565F";
const edgeCol = "#AAA9A4";
const isEmbeddedTree = document.documentElement.dataset.embed === "true";
const treeViewId = new URLSearchParams(window.location.search).get("view_id") || "";
const prefersReducedMotion = window.matchMedia?.(
  "(prefers-reduced-motion: reduce)",
)?.matches;
const treeStatePrefix = "biomaker-tree-state:v1:";

if (window.hljs?.initHighlightingOnLoad) {
  window.hljs.initHighlightingOnLoad();
}

const updateTargetDims = () => {
  if (isEmbeddedTree) return [windowWidth, windowHeight];
  if (windowWidth <= 900) return [windowWidth, windowHeight * 0.48];
  return [windowWidth * 0.58, windowHeight];
};

const escapeHtml = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const compact = (value, fallback = "n/a") => {
  const text = String(value ?? "").trim();
  return text.length ? text : fallback;
};

const nodeStateKey = (treeInd) => {
  const summary = (treeStructData.summaries || [])[treeInd] || {};
  const nodeId = compact(summary.node_id, "");
  if (nodeId) return `node:${nodeId}`;
  const label = compact((treeStructData.labels || [])[treeInd], "");
  return label ? `label:${label}` : `index:${treeInd}`;
};

const treeStateKey = () => {
  const run = treeStructData.run_summary || {};
  const runName = compact(run.exp_name || treeStructData.exp_name, "run");
  return `${treeStatePrefix}${window.location.pathname}:${runName}`;
};

const validateTreeUiState = (candidate) => {
  if (!candidate || typeof candidate !== "object") return null;
  const expanded = {};
  if (candidate.expanded && typeof candidate.expanded === "object") {
    Object.entries(candidate.expanded)
      .slice(0, 160)
      .forEach(([key, value]) => {
        if (typeof key === "string" && key.length <= 160) {
          expanded[key] = Boolean(value);
        }
      });
  }
  const selectedIndex = Number(candidate.selectedIndex);
  const panelScroll = Number(candidate.panelScroll);
  return {
    drawerOpen: Boolean(candidate.drawerOpen),
    expanded,
    panelScroll:
      Number.isFinite(panelScroll) && panelScroll >= 0
        ? Math.min(panelScroll, 10_000_000)
        : 0,
    selectedIndex:
      Number.isInteger(selectedIndex) && selectedIndex >= 0
        ? selectedIndex
        : null,
    selectedKey:
      typeof candidate.selectedKey === "string" &&
      candidate.selectedKey.length <= 512
        ? candidate.selectedKey
        : "",
  };
};

const readTreeUiState = () => {
  const key = treeStateKey();
  try {
    const stored = window.sessionStorage.getItem(key);
    if (stored) return validateTreeUiState(JSON.parse(stored));
  } catch (_error) {
    // Sandboxed iframes without allow-same-origin expose opaque storage origins.
  }
  try {
    if (window.name.startsWith(treeStatePrefix)) {
      const payload = JSON.parse(window.name.slice(treeStatePrefix.length));
      if (payload?.key === key) return validateTreeUiState(payload.state);
    }
  } catch (_error) {
    // Ignore stale or non-BioMaker window.name values.
  }
  return null;
};

const detailStateKey = (element, index) =>
  element.id || element.dataset.stateKey || `detail:${index}`;

const collectExpandedState = () => {
  const expanded = {};
  document.querySelectorAll("#side-panel details").forEach((element, index) => {
    expanded[detailStateKey(element, index)] = element.open;
  });
  return expanded;
};

const applyExpandedState = (expanded) => {
  if (!expanded || typeof expanded !== "object") return;
  document.querySelectorAll("#side-panel details").forEach((element, index) => {
    const key = detailStateKey(element, index);
    if (Object.hasOwn(expanded, key)) element.open = Boolean(expanded[key]);
  });
};

const metricLine = (label, value) => `
  <div class="metric">
    <span>${escapeHtml(label)}</span>
    <strong>${escapeHtml(compact(value))}</strong>
  </div>`;

const formatNumber = (value) => {
  if (value === null || value === undefined || value === "") return "n/a";
  const num = Number(value);
  return Number.isFinite(num) ? num.toLocaleString() : String(value);
};

const formatSeconds = (value) => {
  if (value === null || value === undefined || value === "") return "n/a";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return `${num.toFixed(num >= 10 ? 0 : 1)}s`;
};

const formatMetric = (value) => {
  if (value === null || value === undefined || value === "") return "n/a";
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(4) : String(value);
};

const promptTokenText = (stats) => {
  const exact = stats?.input_tokens;
  const estimated = stats?.estimated_input_tokens;
  if (exact !== null && exact !== undefined) return `${formatNumber(exact)} tokens`;
  if (estimated !== null && estimated !== undefined) {
    return `~${formatNumber(estimated)} tokens`;
  }
  return "n/a";
};

const renderCodeBlock = (code) => {
  const text = String(code || "");
  if (!text) return '<p class="muted">No code recorded for this node.</p>';
  const lines = text.replace(/\n$/, "").split("\n");
  const rows = lines
    .map((line, index) => {
      const highlighted = window.hljs?.highlight
        ? window.hljs.highlight(line || " ", {
            language: "python",
            ignoreIllegals: true,
          }).value
        : escapeHtml(line || " ");
      return `<span class="code-line"><span class="line-number">${
        index + 1
      }</span><span class="line-content">${
        highlighted || "&nbsp;"
      }</span></span>`;
    })
    .join("");
  return `<pre class="code-with-lines"><code class="language-python">${rows}</code></pre>`;
};

const renderPromptSections = (sections) => {
  if (!Array.isArray(sections) || sections.length === 0) {
    return '<p class="muted">No prompt sections were saved for this node.</p>';
  }
  return sections
    .map((section, index) => {
      const tokens = section.estimated_tokens ?? "";
      const chars = section.chars ?? "";
      const sectionText = section.text || "";
      const body =
        section.name === "Detail records" || section.name === "Tree Records"
          ? renderDetailRecords(sectionText)
          : `<pre>${escapeHtml(sectionText)}</pre>`;
      return `
        <details class="prompt-section" data-state-key="prompt:${escapeHtml(
          section.name || "section",
        )}:${index}">
          <summary>
            ${escapeHtml(section.name || "section")}
            <span>${formatNumber(tokens)} est. tokens · ${formatNumber(chars)} chars</span>
          </summary>
          ${body}
        </details>
      `;
    })
    .join("");
};

const renderDetailRecords = (text) => {
  const records = String(text || "")
    .split(/\n\n---\n\n/g)
    .map((record) => record.trim())
    .filter(Boolean);
  if (records.length === 0) return '<p class="muted">No detail records.</p>';

  const roles = detailRecordRoles(records);
  return records
    .map((record, index) => {
      const summary = detailRecordSummary(record, index, roles[index]);
      return `
        <details class="prompt-section detail-record" data-state-key="record:${index}">
          <summary>
            ${escapeHtml(summary)}
            <span>${formatNumber(Math.round(record.length / 4))} est. tokens · ${formatNumber(record.length)} chars</span>
          </summary>
          ${renderDetailRecordBody(record, index)}
        </details>
      `;
    })
    .join("");
};

const isNodeDetailRecord = (record) =>
  /^Node\s+[0-9a-fA-F]+(?:\s+full record|\s+·)/.test(String(record || ""));

const lineageRole = (distanceFromParent) => {
  if (distanceFromParent === 0) return "parent";
  if (distanceFromParent === 1) return "grandparent";
  if (distanceFromParent === 2) return "grandgrandparent";
  return `ancestor-${distanceFromParent + 1}`;
};

const detailRecordRoles = (records) => {
  const roles = Array(records.length).fill("");
  const lineageStart = records.findIndex((record) =>
    String(record || "").startsWith("Parent Lineage"),
  );
  const siblingStart = records.findIndex((record) =>
    String(record || "").startsWith("Existing Children") ||
      String(record || "").startsWith("Existing Root Drafts") ||
      String(record || "").startsWith("Siblings of the Buggy Parent") ||
      String(record || "").startsWith("Same-Round") ||
      String(record || "").startsWith("Same-Parent"),
  );

  if (lineageStart >= 0) {
    const lineageEnd = siblingStart >= 0 ? siblingStart : records.length;
    const lineageNodeIndices = [];
    for (let i = lineageStart + 1; i < lineageEnd; i += 1) {
      if (isNodeDetailRecord(records[i])) lineageNodeIndices.push(i);
    }
    lineageNodeIndices.forEach((recordIndex, position) => {
      const distanceFromParent = lineageNodeIndices.length - position - 1;
      roles[recordIndex] = lineageRole(distanceFromParent);
    });
  }

  if (siblingStart >= 0) {
    for (let i = siblingStart + 1; i < records.length; i += 1) {
      if (isNodeDetailRecord(records[i])) roles[i] = "sibling";
    }
  }
  return roles;
};

const renderDetailRecordBody = (record, recordIndex = 0) => {
  const parsed = parseNodeDetailRecord(record);
  if (!parsed) return `<pre>${escapeHtml(record)}</pre>`;
  return `
    <div class="detail-record-body">
      <details class="prompt-section" data-state-key="record:${recordIndex}:metadata" open>
        <summary>Metadata</summary>
        <pre>${escapeHtml(parsed.metadata)}</pre>
      </details>
      ${
        parsed.artifact
          ? `<details class="prompt-section" data-state-key="record:${recordIndex}:artifact">
              <summary>Output Check</summary>
              <pre>${escapeHtml(parsed.artifact)}</pre>
            </details>`
          : ""
      }
      <details class="prompt-section" data-state-key="record:${recordIndex}:plan">
        <summary>Plan</summary>
        <pre>${escapeHtml(parsed.plan)}</pre>
      </details>
      <details class="prompt-section" data-state-key="record:${recordIndex}:code">
        <summary>Code</summary>
        ${renderCodeBlock(parsed.code)}
      </details>
      ${
        parsed.output
          ? `<details class="prompt-section" data-state-key="record:${recordIndex}:output">
              <summary>Execution Output</summary>
              <pre>${escapeHtml(parsed.output)}</pre>
            </details>`
          : ""
      }
    </div>
  `;
};

const parseNodeDetailRecord = (record) => {
  const text = String(record || "");
  if (!/^Node\s+[0-9a-fA-F]+(?:\s+full record|\s+·)/.test(text)) return null;
  const outputCheckMarker = "\n\nOutput Check:\n";
  const planMarker = "\n\nPlan:\n";
  const outputCheckStart = text.indexOf(outputCheckMarker);
  const planStart = text.indexOf(planMarker);
  if (planStart < 0) return null;

  const hasOutputCheck = outputCheckStart >= 0 && outputCheckStart < planStart;
  const metadata = text.slice(0, hasOutputCheck ? outputCheckStart : planStart).trim();
  const artifact = hasOutputCheck
    ? text.slice(outputCheckStart + outputCheckMarker.length, planStart).trim()
    : "";
  const rest = text.slice(planStart + planMarker.length);
  const codeMarker = "\n\nCode:\n```python\n";
  const codeStart = rest.indexOf(codeMarker);
  if (codeStart < 0) return null;

  const plan = rest.slice(0, codeStart).trim();
  const afterCodeMarker = rest.slice(codeStart + codeMarker.length);
  const outputMarker = "\n```\n\nExecution Output:\n```\n";
  const outputStart = afterCodeMarker.indexOf(outputMarker);
  if (outputStart < 0) {
    const codeOnly = afterCodeMarker.replace(/\n```[\s\n]*$/, "").trim();
    return { metadata, artifact, plan, code: codeOnly, output: "" };
  }

  const code = afterCodeMarker.slice(0, outputStart).trim();
  const afterOutputMarker = afterCodeMarker.slice(outputStart + outputMarker.length);
  const findingMarker = "\n```\n\nEmpirical finding:\n";
  const findingStart = afterOutputMarker.indexOf(findingMarker);
  const rawOutput =
    findingStart >= 0
      ? afterOutputMarker.slice(0, findingStart)
      : afterOutputMarker;
  const output = rawOutput.replace(/\n```[\s\n]*$/, "").trim();
  return { metadata, artifact, plan, code, output };
};

const detailRecordSummary = (record, index, role = "") => {
  const lines = String(record || "").split("\n");
  const firstLine = lines.find((line) => line.trim()) || `Detail record ${index + 1}`;
  const nodeMatch = firstLine.match(/^Node\s+([0-9a-fA-F]+)(?:\s+full record|\s+·)?/);
  if (!nodeMatch) return firstLine;

  const fields = {};
  for (const line of lines) {
    const match = line.match(/^([a-zA-Z_]+):\s*(.*)$/);
    if (match) fields[match[1]] = match[2];
  }
  const nodeId = fields.node_id || nodeMatch[1];
  const firstLineParts = firstLine.split(" · ").map((part) => part.trim()).filter(Boolean);
  const parts =
    firstLineParts.length > 1
      ? firstLineParts
      : [`Node ${nodeId}`, detailRecordStepLabel(fields.step, fields.result)];
  if (role) parts.push(role);
  const valMetric = fields.val_metric || fields.metric;
  if (valMetric && valMetric !== "n/a") {
    parts.push(`val metric ${formatMetric(valMetric)}`);
  }
  return parts.join(" · ");
};

const detailRecordStepLabel = (step, result) => {
  const normalized = String(step ?? "").trim();
  if (normalized && normalized !== "None") {
    if (normalized === "generated_not_scored") return "generated, not scored";
    if (normalized === "generating") return "generating";
    return `Step ${normalized}`;
  }
  const resultText = String(result ?? "").trim();
  if (resultText === "generated_not_scored_yet") return "generated, not scored";
  if (resultText === "generation_in_progress") return "generating";
  return "not scored";
};

const renderPromptDetails = (treeInd) => {
  const prompt = (treeStructData.prompts || [])[treeInd] || {};
  if (!prompt.available) {
    return '<p class="muted">No full input prompt was recorded for this node.</p>';
  }
  return renderPromptSections(prompt.sections);
};

const setNodeDetails = (treeInd, { restoreView = false } = {}) => {
  const summary = (treeStructData.summaries || [])[treeInd] || {};
  const selectedLabel = document.getElementById("selected-label");
  if (selectedLabel) {
    const step = summary.step || "";
    const nodeId = summary.node_id || "";
    selectedLabel.textContent = nodeId
      ? `${step ? `${step}: ` : ""}${nodeId}`
      : "Search start";
  }

  const summaryElm = document.getElementById("node-summary");
  if (summaryElm) {
    const prompt = (treeStructData.prompts || [])[treeInd] || {};
    const stats = prompt.stats || {};
    summaryElm.innerHTML = `
      <div class="metrics-grid">
        ${metricLine("val metric", formatMetric(summary.metric))}
        ${metricLine("llm time", formatSeconds(stats.request_time_seconds))}
        ${metricLine("execution time", summary.runtime)}
        ${metricLine("cpu usage", summary.cpu)}
        ${metricLine("system memory", summary.memory)}
        ${metricLine("gpu memory", summary.gpu_memory)}
        ${metricLine("model", stats.model)}
        ${metricLine("input tokens", promptTokenText(stats))}
        ${metricLine("output tokens", formatNumber(stats.output_tokens))}
      </div>
      ${summary.is_best ? '<p class="best-pill">best node</p>' : ""}
    `;
  }

  const codeElm = document.getElementById("code");
  if (codeElm) {
    const code = (treeStructData.code || [])[treeInd] || "";
    codeElm.innerHTML = renderCodeBlock(code);
  }

  const outputElm = document.getElementById("output");
  if (outputElm) {
    const termOut = (treeStructData.term_out || [])[treeInd] || "";
    outputElm.textContent = termOut || "";
  }

  const promptElm = document.getElementById("input-prompt");
  if (promptElm) {
    promptElm.innerHTML = renderPromptDetails(treeInd);
  }

  const planElm = document.getElementById("plan");
  if (planElm) {
    planElm.textContent = summary.plan || "";
  }

  const detailsTrigger = document.getElementById("details-trigger");
  const selectedText = selectedLabel?.textContent || "selected node";
  if (detailsTrigger) {
    detailsTrigger.setAttribute(
      "aria-label",
      `Open details for ${selectedText}`,
    );
  }
  canvas?.elt?.setAttribute(
    "aria-label",
    `BioMaker search tree. Selected ${selectedText}. Use arrow keys to change nodes and Enter to open details.`,
  );

  if (restoreView && restoredUiState) {
    applyExpandedState(restoredUiState.expanded);
    window.requestAnimationFrame(() => {
      const panel = document.getElementById("side-panel");
      if (panel) panel.scrollTop = restoredUiState?.panelScroll || 0;
    });
  }
};

const setRunSummary = () => {
  const elm = document.getElementById("run-summary");
  if (!elm) return;
  const s = treeStructData.run_summary || {};
  elm.innerHTML = `
    <span>${escapeHtml(s.exp_name || "BioMaker run")}</span>
    <b>best ${escapeHtml(compact(s.best_metric))}</b>
    <span>valid ${escapeHtml(s.valid ?? 0)}</span>
    <span>bug ${escapeHtml(s.bug ?? 0)}</span>
    <span>steps ${escapeHtml(s.steps ?? 0)}</span>
  `;

  const toolbar = document.getElementById("tree-status");
  if (toolbar) {
    toolbar.innerHTML = `
      <strong title="${escapeHtml(s.exp_name || "BioMaker run")}">${escapeHtml(
        s.exp_name || "BioMaker run",
      )}</strong>
      <span class="best-value">best ${escapeHtml(compact(s.best_metric))}</span>
      <span>${escapeHtml(s.steps ?? 0)} steps</span>
    `;
  }
};

const preferredNode = (visibleNodes) => {
  const bestInd = (treeStructData.is_best || []).findIndex(Boolean);
  if (bestInd >= 0) {
    const bestNode = visibleNodes.find((node) => node.treeInd === bestInd);
    if (bestNode) return bestNode;
  }
  return visibleNodes.find((node) => {
    const status = (treeStructData.statuses || [])[node.treeInd] || "unknown";
    return status !== "virtual";
  }) || visibleNodes[0];
};

const findNodeIndexFromState = (state) => {
  if (!state) return -1;
  const count = (treeStructData.layout || []).length;
  if (state.selectedKey) {
    for (let index = 0; index < count; index += 1) {
      if (nodeStateKey(index) === state.selectedKey) return index;
    }
  }
  return Number.isInteger(state.selectedIndex) && state.selectedIndex < count
    ? state.selectedIndex
    : -1;
};

const collectTreeUiState = () => {
  const panel = document.getElementById("side-panel");
  return {
    drawerOpen:
      isEmbeddedTree &&
      document.documentElement.dataset.drawerOpen === "true",
    expanded: collectExpandedState(),
    panelScroll: panel?.scrollTop || 0,
    selectedIndex: selectedTreeIndex,
    selectedKey:
      Number.isInteger(selectedTreeIndex) && selectedTreeIndex >= 0
        ? nodeStateKey(selectedTreeIndex)
        : "",
  };
};

const parentOrigin = (() => {
  if (!isEmbeddedTree) return "";
  try {
    const requested = new URLSearchParams(window.location.search).get(
      "parent_origin",
    );
    if (!requested) return "";
    const parsed = new URL(requested);
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.origin : "";
  } catch (_error) {
    return "";
  }
})();

const postTreeUiState = (state) => {
  if (!parentOrigin || window.parent === window) return;
  window.parent.postMessage(
    {
      key: treeStateKey(),
      state,
      type: "biomaker:tree-state",
      version: 1,
      view_id: treeViewId,
    },
    parentOrigin,
  );
};

const postTreeInteraction = (active) => {
  if (!parentOrigin || window.parent === window) return;
  window.parent.postMessage(
    {
      active: Boolean(active),
      type: "biomaker:tree-interaction",
      version: 1,
      view_id: treeViewId,
    },
    parentOrigin,
  );
};

const beginTreeInteraction = () => {
  window.clearTimeout(treeInteractionTimer);
  postTreeInteraction(true);
  treeInteractionTimer = window.setTimeout(() => {
    treeInteractionTimer = undefined;
    postTreeInteraction(false);
  }, 5000);
};

const endTreeInteraction = (delay = 240) => {
  window.clearTimeout(treeInteractionTimer);
  treeInteractionTimer = window.setTimeout(() => {
    treeInteractionTimer = undefined;
    postTreeInteraction(false);
  }, delay);
};

const markTreeInteraction = () => {
  beginTreeInteraction();
  endTreeInteraction(700);
};

const persistTreeUiState = () => {
  const key = treeStateKey();
  const state = validateTreeUiState(collectTreeUiState());
  if (!state) return;
  restoredUiState = state;
  try {
    window.sessionStorage.setItem(key, JSON.stringify(state));
  } catch (_error) {
    // window.name below is the storage fallback for opaque sandbox origins.
  }
  try {
    window.name = `${treeStatePrefix}${JSON.stringify({ key, state })}`;
  } catch (_error) {
    // Persistence is best effort; the tree stays usable without it.
  }
  postTreeUiState(state);
};

const scheduleTreeUiPersist = () => {
  window.clearTimeout(treeStatePersistTimer);
  treeStatePersistTimer = window.setTimeout(persistTreeUiState, 100);
};

const setDrawerOpen = (open, { focus = false, persist = true } = {}) => {
  if (!isEmbeddedTree) return;
  const expanded = Boolean(open);
  const root = document.documentElement;
  const panel = document.getElementById("side-panel");
  const trigger = document.getElementById("details-trigger");
  root.dataset.drawerOpen = expanded ? "true" : "false";
  trigger?.setAttribute("aria-expanded", String(expanded));
  panel?.setAttribute("aria-hidden", String(!expanded));
  if (panel) panel.inert = !expanded;
  if (focus) {
    window.requestAnimationFrame(() => {
      if (expanded) document.getElementById("drawer-close")?.focus();
      else trigger?.focus();
    });
  }
  if (persist) scheduleTreeUiPersist();
};

const selectTreeNode = (
  treeInd,
  {
    manual = false,
    openDrawer = false,
    persist = true,
    restoreView = false,
  } = {},
) => {
  if (!Number.isInteger(treeInd) || !nodes[treeInd]) return;
  selectedTreeIndex = treeInd;
  nodes.forEach((node) => {
    node.selected = node.treeInd === treeInd;
  });
  manualSelection = manual;
  setNodeDetails(treeInd, { restoreView });
  if (!restoreView) {
    const panel = document.getElementById("side-panel");
    if (panel) panel.scrollTop = 0;
  }
  if (openDrawer) setDrawerOpen(true, { persist: false });
  if (persist) scheduleTreeUiPersist();
};

const restoreTreeUiState = (state) => {
  const validated = validateTreeUiState(state);
  if (!validated) return;
  restoredUiState = validated;
  setDrawerOpen(validated.drawerOpen, { persist: false });
  const restoredIndex = findNodeIndexFromState(validated);
  if (restoredIndex >= 0 && nodes[restoredIndex]) {
    selectTreeNode(restoredIndex, {
      manual: true,
      persist: false,
      restoreView: true,
    });
  } else {
    applyExpandedState(validated.expanded);
  }
};

const initializeTreeUi = () => {
  const panel = document.getElementById("side-panel");
  if (isEmbeddedTree && panel) panel.setAttribute("role", "dialog");
  setDrawerOpen(restoredUiState?.drawerOpen, { persist: false });

  document
    .getElementById("details-trigger")
    ?.addEventListener("click", () => setDrawerOpen(true, { focus: true }));
  document
    .getElementById("drawer-close")
    ?.addEventListener("click", () => setDrawerOpen(false, { focus: true }));
  document
    .getElementById("drawer-scrim")
    ?.addEventListener("click", () => setDrawerOpen(false));
  panel?.addEventListener("scroll", () => {
    scheduleTreeUiPersist();
    markTreeInteraction();
  }, { passive: true });
  panel?.addEventListener("toggle", () => {
    scheduleTreeUiPersist();
    markTreeInteraction();
  }, true);
  panel?.addEventListener("pointerdown", beginTreeInteraction);
  panel?.addEventListener("pointerup", () => endTreeInteraction());

  document.addEventListener("keydown", (event) => {
    const drawerOpen =
      document.documentElement.dataset.drawerOpen === "true";
    if (drawerOpen) markTreeInteraction();
    if (
      event.key === "Tab" &&
      drawerOpen &&
      panel
    ) {
      const focusable = Array.from(
        panel.querySelectorAll(
          'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => !element.closest("[inert]"));
      const first = focusable[0];
      const last = focusable.at(-1);
      if (first && last) {
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    }
    if (
      event.key === "Escape" &&
      drawerOpen
    ) {
      setDrawerOpen(false, { focus: true });
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) noLoop();
    else if (animationsSettled()) redraw();
    else loop();
  });
  window.addEventListener("pagehide", () => {
    persistTreeUiState();
    postTreeInteraction(false);
  });
  window.addEventListener("pointerup", () => endTreeInteraction());
  window.addEventListener("pointercancel", () => endTreeInteraction());

  if (parentOrigin && window.parent !== window) {
    window.addEventListener("message", (event) => {
      if (event.source !== window.parent || event.origin !== parentOrigin) return;
      const data = event.data;
      if (
        data?.type !== "biomaker:tree-state-restore" ||
        data.key !== treeStateKey() ||
        data.view_id !== treeViewId
      ) {
        return;
      }
      restoreTreeUiState(data.state);
    });
    window.parent.postMessage(
      {
        key: treeStateKey(),
        type: "biomaker:tree-state-request",
        version: 1,
        view_id: treeViewId,
      },
      parentOrigin,
    );
  }
};

windowResized = () => {
  persistTreeUiState();
  resizeCanvas(...updateTargetDims());
  nodes = [];
  edges = [];
  treeInitialized = false;
  globalTime = 0;
  firstFrameTime = undefined;
  manualSelection = Boolean(restoredUiState?.selectedKey);
  loop();
};

const animEase = (t) => 1 - (1 - Math.min(t, 1.0)) ** 5;

// ---- global constants ----

const globalAnimSpeed = 2.2;
const initialSpeedRampMs = 4000;
const childEdgeStaggerMs = 500;
const scaleFactor = 1.0;

// ---- global vars ----

let globalTime = 0;
let manualSelection = false;
let selectedTreeIndex = null;
let restoredUiState = null;
let treeStatePersistTimer = undefined;
let treeInteractionTimer = undefined;
let canvas = null;

let treeStructData = <placeholder>

let firstFrameTime = undefined;

let nodes = [];
let edges = [];
let treeInitialized = false;

setup = () => {
  canvas = createCanvas(...updateTargetDims());
  canvas.attribute(
    "aria-label",
    "BioMaker search tree. Use arrow keys to change nodes and Enter to open details.",
  );
  canvas.attribute("role", "img");
  canvas.attribute("tabindex", "0");
  canvas.elt.addEventListener("keydown", handleTreeKeydown);
  canvas.elt.addEventListener("pointerdown", beginTreeInteraction);
  frameRate(60);
  document.documentElement.dataset.treeReady = "true";
  restoredUiState = readTreeUiState();
  setRunSummary();
  initializeTreeUi();
};

class Node {
  x;
  y;
  size;
  xT;
  yT;
  xB;
  yB;
  treeInd;
  color;
  relSize;
  animationStart = Number.MAX_VALUE;
  animationProgress = 0;
  isStatic = false;
  hasChildren = false;
  isRootNode = true;
  isStarred = false;
  selected = false;
  renderSize = 10;
  edges = [];
  bgCol;

  constructor(x, y, relSize, treeInd) {
    const nodeCount = Math.max(treeStructData.layout.length, 1);
    const crowdScale = Math.max(0.6, Math.min(1.0, 22 / nodeCount));
    const minSize = 24 * crowdScale;
    const maxSize = 38 * crowdScale;

    const maxColor = 10;
    const minColor = 125;

    this.relSize = relSize;
    this.treeInd = treeInd;
    this.size = minSize + (maxSize - minSize) * relSize;
    this.color = minColor + (maxColor - minColor) * relSize;
    this.bgCol = Math.round(Math.max(this.color / 2, 0));

    this.x = x;
    this.y = y;
    this.xT = x;
    this.yT = y - this.size / 2;
    this.xB = x;
    this.yB = y + this.size / 2;

    nodes.push(this);
  }

  startAnimation = (offset = 0) => {
    if (this.animationStart == Number.MAX_VALUE)
      this.animationStart = globalTime + offset;
  };

  child = (node) => {
    let edge = new Edge(this, node);
    this.edges.push(edge);
    edges.push(edge);
    this.hasChildren = true;
    node.isRootNode = false;
    return node;
  };

  render = () => {
    if (globalTime - this.animationStart < 0) return;

    const mouseXlocalCoords = (mouseX - width / 2) / scaleFactor;
    const mouseYlocalCoords = (mouseY - height / 2) / scaleFactor;
    const status = (treeStructData.statuses || [])[this.treeInd] || "unknown";
    const isMouseOver =
      dist(mouseXlocalCoords, mouseYlocalCoords, this.x, this.y) <
      this.renderSize / 1.5;
    if (isMouseOver) cursor(HAND);

    this.renderSize = this.size;
    if (!this.isStatic) {
      this.animationProgress = animEase(
        (globalTime - this.animationStart) / 1000,
      );
      if (this.animationProgress >= 1) {
        this.isStatic = true;
      } else {
        this.renderSize =
          this.size *
          (0.8 +
            0.2 *
              (-3.33 * this.animationProgress ** 2 +
                4.33 * this.animationProgress));
      }
    }

    fill(this.color);
    if (status === "virtual") {
      fill(rootCol);
    } else if (status === "bug") {
      fill(bugCol);
    } else {
      fill(nodeCol);
    }
    const isBest = (treeStructData.is_best || [])[this.treeInd];
    noStroke();
    if (status === "virtual") {
      noFill();
      stroke(rootCol);
      strokeWeight(2);
      circle(this.x, this.y, Math.max(8, this.renderSize * 0.42));
      noStroke();
    } else {
      square(
        this.x - this.renderSize / 2,
        this.y - this.renderSize / 2,
        this.renderSize,
        8,
      );
    }

    if (isBest || this.selected) {
      noFill();
      stroke(this.selected ? accentCol : bestCol);
      strokeWeight(this.selected ? 4 : 3);
      if (status === "virtual") {
        circle(this.x, this.y, Math.max(14, this.renderSize * 0.7));
      } else {
        square(
          this.x - this.renderSize / 2 - 4,
          this.y - this.renderSize / 2 - 4,
          this.renderSize + 8,
          9,
        );
      }
      noStroke();
    }

    noStroke();
    textAlign(CENTER, CENTER);
    const label = (treeStructData.display_labels || [])[this.treeInd] || "";
    textSize(status === "virtual" ? this.renderSize / 3 : this.renderSize / 3.2);
    fill(255);
    if (status !== "virtual") {
      text(label, this.x, this.y - 1);
    }
    if (isBest) {
      textSize(this.renderSize / 2.2);
      fill(bestCol);
      text("*", this.x + this.renderSize * 0.55, this.y - this.renderSize * 0.5);
    }

    const dotAnimThreshold = 0.85;
    if (this.isStarred && this.animationProgress >= dotAnimThreshold) {
      let dotAnimProgress =
        (this.animationProgress - dotAnimThreshold) / (1 - dotAnimThreshold);
      textSize(
        ((-3.33 * dotAnimProgress ** 2 + 4.33 * dotAnimProgress) *
          this.renderSize) /
          2,
      );
      if (this.selected) {
        fill(0);
        stroke(0);
      } else {
        fill(accentCol);
        stroke(accentCol);
      }
      strokeWeight((-(dotAnimProgress ** 2) + dotAnimProgress) * 2);
      text("*", this.x + 20, this.y - 11);
      noStroke();
    }

    if (!this.isStatic) {
      fill(bgCol);
      const progressAnimBaseSize = this.renderSize + 5;
      rect(
        this.x - progressAnimBaseSize / 2,
        this.y -
          progressAnimBaseSize / 2 +
          progressAnimBaseSize * this.animationProgress,
        progressAnimBaseSize,
        progressAnimBaseSize * (1 - this.animationProgress),
      );
    }
    if (this.animationProgress >= 0.9) {
      this.edges
        .sort((a, b) => a.color() - b.color())
        .forEach((e, i) => {
          e.startAnimation((i / this.edges.length) ** 2 * childEdgeStaggerMs);
        });
    }
  };
}

class Edge {
  nodeT;
  nodeB;
  animX = 0;
  animY = 0;
  animationStart = Number.MAX_VALUE;
  animationProgress = 0;
  isStatic = false;
  weight = 0;

  constructor(nodeT, nodeB) {
    this.nodeT = nodeT;
    this.nodeB = nodeB;
    this.weight = 2 + nodeB.relSize * 1;
  }

  color = () => this.nodeB.color;

  startAnimation = (offset = 0) => {
    if (this.animationStart == Number.MAX_VALUE)
      this.animationStart = globalTime + offset;
  };

  render = () => {
    if (globalTime - this.animationStart < 0) return;

    if (!this.isStatic) {
      this.animationProgress = animEase(
        (globalTime - this.animationStart) / 1000,
      );
      if (this.animationProgress >= 1) {
        this.isStatic = true;
        this.animX = this.nodeB.xT;
        this.animY = this.nodeB.yT;
      } else {
        this.animX = bezierPoint(
          this.nodeT.xB,
          this.nodeT.xB,
          this.nodeB.xT,
          this.nodeB.xT,
          this.animationProgress,
        );

        this.animY = bezierPoint(
          this.nodeT.yB,
          (this.nodeT.yB + this.nodeB.yT) / 2,
          (this.nodeT.yB + this.nodeB.yT) / 2,
          this.nodeB.yT,
          this.animationProgress,
        );
      }
    }
    if (this.animationProgress >= 0.97) {
      this.nodeB.startAnimation();
    }

    strokeWeight(this.weight);
    noFill();
    const status = (treeStructData.statuses || [])[this.nodeB.treeInd] || "unknown";
    const edgeColor = status === "bug" ? bugCol : status === "virtual" ? rootCol : edgeCol;
    stroke(lerpColor(color(bgCol), color(edgeColor), 0.75));
    bezier(
      this.nodeT.xB,
      this.nodeT.yB,
      this.nodeT.xB,
      (this.nodeT.yB + this.nodeB.yT) / 2,
      this.animX,
      (this.nodeT.yB + this.nodeB.yT) / 2,
      this.animX,
      this.animY,
    );
  };
}

const animationsSettled = () =>
  treeInitialized &&
  nodes.every((node) => node.isStatic) &&
  edges.every((edge) => edge.isStatic);

const finishAnimationsImmediately = () => {
  nodes.forEach((node) => {
    node.animationStart = 0;
    node.animationProgress = 1;
    node.isStatic = true;
    node.renderSize = node.size;
  });
  edges.forEach((edge) => {
    edge.animationStart = 0;
    edge.animationProgress = 1;
    edge.isStatic = true;
    edge.animX = edge.nodeB.xT;
    edge.animY = edge.nodeB.yT;
  });
};

const nodeAtPointer = () => {
  const localX = (mouseX - width / 2) / scaleFactor;
  const localY = (mouseY - height / 2) / scaleFactor;
  return [...nodes].reverse().find(
    (node) =>
      node.animationProgress > 0 &&
      dist(localX, localY, node.x, node.y) < node.renderSize / 1.5,
  );
};

const handleTreeKeydown = (event) => {
  const navigation = {
    ArrowDown: 1,
    ArrowRight: 1,
    ArrowUp: -1,
    ArrowLeft: -1,
  };
  if (event.key in navigation || ["Home", "End"].includes(event.key)) {
    event.preventDefault();
    if (!nodes.length) return;
    markTreeInteraction();
    let nextIndex;
    if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = nodes.length - 1;
    else {
      const current = Number.isInteger(selectedTreeIndex) ? selectedTreeIndex : 0;
      nextIndex = (current + navigation[event.key] + nodes.length) % nodes.length;
    }
    selectTreeNode(nextIndex, { manual: true });
    redraw();
    return;
  }
  if (["Enter", " "].includes(event.key)) {
    event.preventDefault();
    markTreeInteraction();
    const nextIndex = Number.isInteger(selectedTreeIndex) ? selectedTreeIndex : 0;
    selectTreeNode(nextIndex, { manual: true, persist: false });
    if (isEmbeddedTree) setDrawerOpen(true, { focus: true });
    else scheduleTreeUiPersist();
    redraw();
  }
};

mousePressed = (event) => {
  if (event?.target !== canvas?.elt) return;
  const hit = nodeAtPointer();
  if (!hit) return;
  selectTreeNode(hit.treeInd, {
    manual: true,
    openDrawer: isEmbeddedTree,
  });
  redraw();
  return false;
};

mouseMoved = (event) => {
  if (event?.target === canvas?.elt && animationsSettled()) redraw();
};

draw = () => {
  cursor(ARROW);
  if (firstFrameTime === undefined) {
    firstFrameTime = millis();
  }
  // ---- update global animation state ----
  const initialSpeedScalingEaseIO =
    (cos(min((millis() - firstFrameTime) / initialSpeedRampMs, 1.0) * PI) +
      1) /
    2;
  const initAnimationSpeedFactor = 1.0 - 0.4 * initialSpeedScalingEaseIO;
  // update global scaling-aware clock
  globalTime += globalAnimSpeed * initAnimationSpeedFactor * deltaTime;

  if (!treeInitialized) {
    treeInitialized = true;
    const nodeSafeInset = 28;
    const toolbarBottom = document
      .getElementById("tree-toolbar")
      ?.getBoundingClientRect().bottom;
    const bottomInset = isEmbeddedTree
      ? Math.min(nodeSafeInset, height / 2)
      : height * 0.08;
    const requestedTopInset = Math.max(
      nodeSafeInset,
      (toolbarBottom || 0) + nodeSafeInset,
    );
    const topInset = isEmbeddedTree
      ? Math.min(requestedTopInset, Math.max(0, height - bottomInset))
      : height * 0.08;
    const spacingHeight = Math.max(0, height - topInset - bottomInset);
    const spacingWidth = Math.max(0, width - nodeSafeInset * 2);
    (treeStructData.layout || []).forEach((lay, index) => {
      new Node(
        spacingWidth * (lay[0] - 0.5),
        topInset + spacingHeight * lay[1] - height / 2,
        1 - ((treeStructData.metrics || [])[index] || 0),
        index,
      );
    });
    (treeStructData.edges || []).forEach((ind) => {
      if (nodes[ind[0]] && nodes[ind[1]]) nodes[ind[0]].child(nodes[ind[1]]);
    });
    nodes.forEach((n) => {
      if (n.isRootNode) n.startAnimation();
    });
    if (nodes.length > 0) {
      const restoredIndex = findNodeIndexFromState(restoredUiState);
      const initialNode =
        (restoredIndex >= 0 && nodes[restoredIndex]) || preferredNode(nodes);
      selectTreeNode(initialNode.treeInd, {
        manual: restoredIndex >= 0,
        persist: false,
        restoreView: restoredIndex >= 0,
      });
    }
    if (prefersReducedMotion) finishAnimationsImmediately();
  }

  const staticNodes = nodes.filter(
    (n) => n.isStatic || n.animationProgress >= 0.7,
  );
  if (staticNodes.length > 0) {
    const selectedNode = preferredNode(staticNodes);
    if (!manualSelection) {
      if (selectedTreeIndex !== selectedNode.treeInd) {
        selectTreeNode(selectedNode.treeInd, {
          manual: false,
          persist: false,
        });
      }
    }
  }
  background(bgCol);
  // global animation transforms
  translate(width / 2, height / 2);
  scale(scaleFactor);

  
  // ---- fg render ----
  edges.forEach((e) => e.render());
  nodes.forEach((n) => n.render());

  if (animationsSettled()) noLoop();
};
