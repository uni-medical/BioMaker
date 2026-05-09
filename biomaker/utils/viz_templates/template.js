const bgCol = "#F2F0E7";
const accentCol = "#2F6FED";
const bestCol = "#D6A11E";
const okCol = "#2F7D59";
const bugCol = "#B4493D";
const rootCol = "#50565F";
const edgeCol = "#9B9486";

hljs.initHighlightingOnLoad();

const updateTargetDims = () => {
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

const renderPromptSections = (sections) => {
  if (!Array.isArray(sections) || sections.length === 0) {
    return '<p class="muted">No prompt sections were saved for this node.</p>';
  }
  return sections
    .map((section) => {
      const tokens = section.estimated_tokens ?? "";
      const chars = section.chars ?? "";
      const sectionText = section.text || "";
      const body =
        section.name === "Detail records" || section.name === "Tree Records"
          ? renderDetailRecords(sectionText)
          : `<pre>${escapeHtml(sectionText)}</pre>`;
      return `
        <details class="prompt-section">
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
        <details class="prompt-section detail-record">
          <summary>
            ${escapeHtml(summary)}
            <span>${formatNumber(Math.round(record.length / 4))} est. tokens · ${formatNumber(record.length)} chars</span>
          </summary>
          ${renderDetailRecordBody(record)}
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

const renderDetailRecordBody = (record) => {
  const parsed = parseNodeDetailRecord(record);
  if (!parsed) return `<pre>${escapeHtml(record)}</pre>`;
  return `
    <div class="detail-record-body">
      <details class="prompt-section" open>
        <summary>Metadata</summary>
        <pre>${escapeHtml(parsed.metadata)}</pre>
      </details>
      ${
        parsed.artifact
          ? `<details class="prompt-section">
              <summary>Output Check</summary>
              <pre>${escapeHtml(parsed.artifact)}</pre>
            </details>`
          : ""
      }
      <details class="prompt-section">
        <summary>Plan</summary>
        <pre>${escapeHtml(parsed.plan)}</pre>
      </details>
      <details class="prompt-section">
        <summary>Code</summary>
        <pre><code class="language-python">${escapeHtml(parsed.code)}</code></pre>
      </details>
      ${
        parsed.output
          ? `<details class="prompt-section">
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

const setNodeDetails = (treeInd) => {
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
      ${summary.error ? `<h3>Error</h3><pre>${escapeHtml(summary.error)}</pre>` : ""}
    `;
  }

  const codeElm = document.getElementById("code");
  if (codeElm) {
    const code = treeStructData.code[treeInd] || "";
    codeElm.innerHTML = code
      ? hljs.highlight(code, { language: "python" }).value
      : "";
  }

  const outputElm = document.getElementById("output");
  if (outputElm) {
    const termOut = treeStructData.term_out[treeInd] || "";
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

windowResized = () => {
  resizeCanvas(...updateTargetDims());
  nodes = [];
  edges = [];
  manualSelection = false;
  awaitingPostResizeOps = true;
};

const animEase = (t) => 1 - (1 - Math.min(t, 1.0)) ** 5;

// ---- global constants ----

const globalAnimSpeed = 1.1;
const scaleFactor = 1.0;

// ---- global vars ----

let globalTime = 0;
let manualSelection = false;

let currentElemInd = 0;

let treeStructData = <placeholder>

let lastClick = 0;
let firstFrameTime = undefined;

let nodes = [];
let edges = [];

let lastScrollPos = 0;

setup = () => {
  canvas = createCanvas(...updateTargetDims());
  setRunSummary();
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
      status !== "virtual" &&
      dist(mouseXlocalCoords, mouseYlocalCoords, this.x, this.y) <
      this.renderSize / 1.5;
    if (isMouseOver) cursor(HAND);
    if (isMouseOver && mouseIsPressed) {
      nodes.forEach((n) => (n.selected = false));
      this.selected = true;
      setNodeDetails(this.treeInd);
      manualSelection = true;
    }

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
      fill(okCol);
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
      if (status !== "virtual") {
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
          e.startAnimation((i / this.edges.length) ** 2 * 1000);
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

draw = () => {
  cursor(ARROW);
  frameRate(120);
  if (!firstFrameTime && frameCount <= 1) {
    firstFrameTime = millis();
  }
  // ---- update global animation state ----
  const initialSpeedScalingEaseIO =
    (cos(min((millis() - firstFrameTime) / 8000, 1.0) * PI) + 1) / 2;
  const initialSpeedScalingEase =
    (cos(min((millis() - firstFrameTime) / 8000, 1.0) ** (1 / 2) * PI) + 1) / 2;
  const initAnimationSpeedFactor = 1.0 - 0.4 * initialSpeedScalingEaseIO;
  // update global scaling-aware clock
  globalTime += globalAnimSpeed * initAnimationSpeedFactor * deltaTime;

  if (nodes.length == 0) {
    const spacingHeight = height * 0.84;
    const spacingWidth = width * 0.92;
    treeStructData.layout.forEach((lay, index) => {
      new Node(
        spacingWidth * (lay[0] - 0.5),
        spacingHeight * (lay[1] - 0.5),
        1 - treeStructData.metrics[index],
        index,
      );
    });
    treeStructData.edges.forEach((ind) => {
      nodes[ind[0]].child(nodes[ind[1]]);
    });
    nodes.forEach((n) => {
      if (n.isRootNode) n.startAnimation();
    });
    if (nodes.length > 0) {
      const initialNode = preferredNode(nodes);
      initialNode.selected = true;
      setNodeDetails(initialNode.treeInd);
    }
  }

  const staticNodes = nodes.filter(
    (n) => n.isStatic || n.animationProgress >= 0.7,
  );
  if (staticNodes.length > 0) {
    const selectedNode = preferredNode(staticNodes);
    if (!manualSelection) {
      if (!selectedNode.selected) {
        setNodeDetails(selectedNode.treeInd);
      }
      staticNodes.forEach((node) => {
        node.selected = node === selectedNode;
      });
    }
  }
  background(bgCol);
  // global animation transforms
  translate(width / 2, height / 2);
  scale(scaleFactor);

  
  // ---- fg render ----
  edges.forEach((e) => e.render());
  nodes.forEach((n) => n.render());
  
};
