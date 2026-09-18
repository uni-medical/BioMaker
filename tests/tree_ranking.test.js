"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const template = fs.readFileSync(path.join(__dirname, "../biomaker/utils/viz_templates/template.js"), "utf8");

function makeView() {
  const elements = {};
  for (const id of ["candidate-table", "candidate-sort", "candidate-filter"]) {
    elements[id] = { innerHTML: "", value: "", listeners: {}, addEventListener(type, handler) { this.listeners[type] = handler; } };
  }
  const specs = [{ name: "dice", maximize: true }, { name: "distance", maximize: false }];
  const candidates = [
    { node_id: "b", step: "1", tree_index: 2, metrics: { dice: 5, distance: 5 }, average_rank: 1.5, worst_rank: 2, is_best: true },
    { node_id: "a", step: "0", tree_index: 1, metrics: { dice: 9, distance: 1 }, average_rank: 2, worst_rank: 3, is_best: false },
    { node_id: "c", step: "2", tree_index: 3, metrics: { dice: 3, distance: 4 }, average_rank: 2.5, worst_rank: 3, is_best: false },
  ];
  const data = { ranking_snapshot: { metric_specs: specs, selection_policy: "average_rank" }, candidates };
  const context = vm.createContext({
    document: { documentElement: { dataset: {} }, getElementById: (id) => elements[id] },
    window: { location: { search: "" }, matchMedia: () => ({ matches: true }), clearTimeout() {}, setTimeout() {} },
    URLSearchParams,
    URL,
    redraw() {},
  });
  vm.runInContext(template.replace("<placeholder>", JSON.stringify(data)), context);
  return { context, elements, specs, candidates, run: (expression) => vm.runInContext(expression, context) };
}

test("candidate table preserves exported selection order and ranks during metric sorting and filtering", () => {
  const view = makeView();
  const ids = (expression) => JSON.parse(view.run(`JSON.stringify(${expression}.map(row => row.node_id))`));
  const before = view.run("JSON.stringify(treeStructData)");
  assert.deepEqual(ids("candidateRowsForView()"), ["b", "a", "c"]);
  assert.deepEqual(ids('candidateRowsForView("dice")'), ["a", "b", "c"]);
  assert.deepEqual(ids('candidateRowsForView("distance")'), ["a", "c", "b"]);
  assert.deepEqual(ids('candidateRowsForView("dice", "b")'), ["b"]);
  assert.equal(view.run("JSON.stringify(treeStructData)"), before);
  view.run("initializeCandidateTable()");
  assert.match(view.elements["candidate-table"].innerHTML, /Selected|Best/);
  view.elements["candidate-sort"].value = "distance";
  view.elements["candidate-sort"].listeners.change();
  assert.equal(view.run("treeStructData.candidates[0].is_best"), true);
});

test("candidate rows select their original tree node after a different view sort", () => {
  const view = makeView();
  view.run("nodes = [0, 1, 2, 3].map(treeInd => ({ treeInd })); initializeCandidateTable()");
  view.elements["candidate-sort"].value = "distance";
  view.elements["candidate-sort"].listeners.change();
  view.elements["candidate-table"].listeners.click({ target: { closest: () => ({ dataset: { treeIndex: "2" } }) } });
  assert.equal(view.run("selectedTreeIndex"), 2);
  assert.equal(view.run("nodes[2].selected"), true);
});

test("metric details escape names, separate diagnostics and resources, and retain all objectives", () => {
  const view = makeView();
  const name = '<img src=x onerror="unsafe()">' + "long".repeat(40);
  const summary = { node_id: "b", selection_policy: "average_rank", candidate_count: 3, average_rank: 1.5, worst_rank: 2,
    metric_specs: [...view.specs, { name, maximize: false }], metrics: { dice: 5, distance: 5, [name]: 0.4 },
    metric_ranks: { dice: 2, distance: 1, [name]: 1 }, diagnostic_metrics: { diagnostic: 300 }, is_best: true, status: "completed" };
  const rendered = view.run(`renderRankingDetails(${JSON.stringify(summary)})`);
  assert.match(rendered, /Current ranking/);
  assert.match(rendered, /Average rank/);
  assert.match(rendered, /Worst rank/);
  assert.match(rendered, /Additional diagnostic metrics/);
  assert.match(rendered, /&lt;img/);
  assert.doesNotMatch(rendered, /<img/);
  assert.equal((rendered.match(/scope="row"/g) || []).length, 4);
});
