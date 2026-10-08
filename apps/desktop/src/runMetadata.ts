import { isActive, type Run } from "./models";

export interface RunMetadata {
  run_id: string;
  display_name: string | null;
  archived: boolean;
  revision: number;
  updated_at: string | null;
}
export type MetadataMap = Record<string, RunMetadata>;
export function defaultMetadata(runId: string): RunMetadata {
  return { run_id: runId, display_name: null, archived: false, revision: 0, updated_at: null };
}
// Poll responses may have started before a PUT. Never roll a committed revision
// back, and never interpret an omitted row as deletion of local metadata.
export function mergeMetadata(current: MetadataMap, rows: RunMetadata[]): MetadataMap {
  let next = current;
  for (const row of rows) {
    if (current[row.run_id] && current[row.run_id].revision >= row.revision) continue;
    if (next === current) next = { ...current };
    next[row.run_id] = row;
  }
  return next;
}
export function decorateRun(run: Run, metadata: MetadataMap): Run {
  const row = metadata[run.id];
  return { ...run, display_name: row?.display_name ?? null, archived: row?.archived ?? false };
}
export function isCurrentRun(run: Run): boolean {
  return !run.archived || isActive(run) || !!run.cleanup_pending;
}
export function canArchive(run: Run): boolean {
  if (run.cleanup_pending) return false;
  if (run.managed === false) return true;
  return ["succeeded", "failed", "cancelled", "stopped", "interrupted"].includes(run.detailed_status || run.status);
}
