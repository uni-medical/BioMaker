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

const setNodeDetails = (treeInd) => {
  const summary = (treeStructData.summaries || [])[treeInd] || {};
  const selectedLabel = document.getElementById("selected-label");
  if (selectedLabel) selectedLabel.textContent = summary.title || "Search start";

  const summaryElm = document.getElementById("node-summary");
  if (summaryElm) {
    summaryElm.innerHTML = `
      <div class="metrics-grid">
        ${metricLine("stage", summary.stage)}
        ${metricLine("metric", summary.metric)}
        ${metricLine("runtime", summary.runtime)}
        ${metricLine("memory", summary.memory)}
        ${metricLine("cpu", summary.cpu)}
        ${metricLine("outcome", summary.outcome)}
      </div>
      ${summary.parent ? `<p><b>parent</b>: ${escapeHtml(summary.parent)}</p>` : ""}
      ${summary.is_best ? '<p class="best-pill">best node</p>' : ""}
      ${summary.selected ? `<h3>Selection</h3><p>${escapeHtml(summary.selected)}</p>` : ""}
      ${summary.plan ? `<h3>Plan</h3><p>${escapeHtml(summary.plan)}</p>` : ""}
      ${summary.analysis ? `<h3>Analysis</h3><p>${escapeHtml(summary.analysis)}</p>` : ""}
      ${summary.artifact ? `<h3>Artifact</h3><pre>${escapeHtml(summary.artifact)}</pre>` : ""}
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

  const artifactElm = document.getElementById("artifact-json");
  if (artifactElm) {
    const report = summary.artifact || "";
    artifactElm.textContent = report || "";
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
