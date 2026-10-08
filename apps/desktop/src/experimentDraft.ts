import { defaultOptions, runTitle, type Options, type Run } from "./models";

export interface Draft {
  task: string;
  dataset: string;
  name: string;
  options: Options;
  template: string | null;
  reusedFrom?: { runId: string; title: string; requiresDataset: boolean; usedDefaults: boolean };
}
export const emptyDraft: Draft = {
  task: "", dataset: "", name: "", options: defaultOptions, template: null,
};
export function hasDraft(draft: Draft): boolean {
  return !!(draft.task || draft.dataset || draft.name || draft.template || draft.reusedFrom) ||
    (Object.keys(defaultOptions) as (keyof Options)[]).some(key => draft.options[key] !== defaultOptions[key]);
}
export function reuseDraft(run: Run): Draft {
  const source = run.options || {};
  const options = { ...defaultOptions };
  let usedDefaults = false;
  const model = source.model_for_all_agents ?? source.main_model;
  if (typeof model === "string" && Array.from(model).length <= 200) options.model_for_all_agents = model.trim() || null;
  else if (Object.hasOwn(source, "model_for_all_agents") && source.model_for_all_agents === null && source.main_model == null) options.model_for_all_agents = null;
  else usedDefaults = true;
  const numericKeys = ["agent_steps", "exec_timeout", "agent_total_time_limit"] as const;
  const limits = { agent_steps: 1000, exec_timeout: 86400, agent_total_time_limit: 2592000 };
  for (const key of numericKeys) {
    const value = source[key];
    if (typeof value === "number" && Number.isInteger(value) && value >= 1 && value <= limits[key]) options[key] = value;
    else usedDefaults = true;
  }
  const workers = source.exec_parallel_workers;
  if (workers === "auto" || (typeof workers === "number" && Number.isInteger(workers) && workers >= 1 && workers <= 64)) options.exec_parallel_workers = workers;
  else usedDefaults = true;
  if (typeof source.final_retrain_enabled === "boolean") options.final_retrain_enabled = source.final_retrain_enabled;
  else usedDefaults = true;
  const requiresDataset = !!run.history_import || !!run.read_only || run.managed === false;
  return {
    task: run.project_snapshot?.task_body || "",
    dataset: requiresDataset ? "" : run.project_snapshot?.dataset_path || run.dataset_path || "",
    name: "", template: null, options,
    reusedFrom: { runId: run.id, title: runTitle(run), requiresDataset, usedDefaults },
  };
}
