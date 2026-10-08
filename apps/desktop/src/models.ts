export type Language = "zh" | "en";
export type Theme = "system" | "light" | "dark";
export type Translate = (zh: string, en: string) => string;
export interface Options {
  model_for_all_agents: string | null;
  agent_steps: number;
  exec_timeout: number;
  exec_parallel_workers: number | "auto";
  agent_total_time_limit: number;
  final_retrain_enabled: boolean;
}
export const defaultOptions: Options = {
  model_for_all_agents: null,
  agent_steps: 35,
  exec_timeout: 600,
  exec_parallel_workers: "auto",
  agent_total_time_limit: 86400,
  final_retrain_enabled: true,
};
export interface Run {
  id: string;
  status: string;
  detailed_status: string;
  requested_exp_name?: string;
  actual_exp_name?: string;
  dataset_name?: string;
  dataset_path?: string;
  read_only?: boolean;
  status_reason?: string;
  error?: string;
  allowed_actions?: string[];
  options?: Partial<Options> & { main_model?: string };
  project_snapshot?: {
    dataset_path?: string;
    task_body?: string;
    task_template_path?: string;
  };
  timestamps?: {
    created_at?: string;
    started_at?: string;
    finished_at?: string;
  };
  progress?: {
    completed_steps?: number;
    total_steps?: number;
    last_event?: string;
  };
  log_dir?: string;
  workspace_dir?: string;
  observed?: {
    event_count: number;
    results: {
      count: number;
      best_metric: number | null;
      latest?: Record<string, string | null>;
    };
    last_event?: RunEvent;
  };
  artifacts?: Artifact[];
}
export interface Artifact {
  name: string;
  size: number;
  modified_at: number;
  revision?: string;
  url: string;
}
export interface RunEvent {
  event?: string;
  ts?: string;
  step?: number;
  round?: number;
  message?: string;
  error?: string;
  reason?: string;
  [key: string]: unknown;
}
export interface Revisions {
  run_revision?: string;
  events?: { revision?: string; available?: boolean };
  console?: { revision?: string; available?: boolean };
  artifacts?: { revision?: string };
  final_barrier?: boolean;
}
export interface EngineStatus {
  version: string;
  engine: {
    ready: boolean;
    installing: boolean;
    error: string | null;
    log: string;
  };
  paths: { workspace: string; logs: string; engine?: string };
  capabilities?: {
    local_execution?: boolean;
    platform?: string;
    background_runs?: boolean;
  };
}
export interface EnvConfig {
  values: Record<
    string,
    { configured: boolean; value?: string; masked?: string; secret?: boolean }
  >;
  source?: { path: string };
}
export interface Health {
  status: string;
  queue: {
    active_run_id?: string;
    running: number;
    queued: number;
    paused: boolean;
    pause_reason?: string;
  };
}
export interface QueueState {
  paused: boolean;
  reason?: string;
  pause_generation: number;
}
export interface Template {
  id: string;
  name: string;
  path: string;
  body: string;
  dataset_path?: string;
}
export interface Resources {
  stale?: boolean;
  sampled_at?: string;
  error?: string;
  cpu?: { utilization_percent?: number | null };
  memory?: {
    used_bytes?: number;
    total_bytes?: number;
    utilization_percent?: number;
  };
  nvidia_gpus?: {
    name: string;
    utilization_percent?: number | null;
    memory_used_bytes?: number;
    memory_total_bytes?: number;
  }[];
  apple_mps?: { available: boolean | null };
  apple_gpu?: { utilization_percent?: number | null } | null;
}
export function request<T>(
  path: string,
  method: "GET" | "POST" | "PUT" = "GET",
  body?: unknown,
): Promise<T> {
  return window.biomaker.request<T>(path, {
    method,
    ...(body === undefined ? {} : { body }),
  });
}
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
export function runTitle(run: Run): string {
  return (
    run.requested_exp_name ||
    run.actual_exp_name ||
    run.project_snapshot?.task_body
      ?.split("\n")
      .find(Boolean)
      ?.replace(/^#+\s*/, "")
      .slice(0, 64) ||
    run.id
  );
}
export function basename(path?: string): string {
  return (
    path
      ?.replace(/[\\/]+$/, "")
      .split(/[\\/]/)
      .pop() || ""
  );
}
export function isActive(run?: Run): boolean {
  return !!run && ["running", "queued", "stopping"].includes(run.status);
}
export function formatBytes(value?: number): string {
  if (value === undefined || value === null) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(1)} GB`;
}
export function statusLabel(run: Run, t: Translate): string {
  const status = run.detailed_status || run.status;
  const labels: Record<string, [string, string]> = {
    queued: ["排队中", "Queued"],
    running: ["运行中", "Running"],
    stopping: ["正在停止", "Stopping"],
    succeeded: ["已完成", "Complete"],
    complete: ["已完成", "Complete"],
    failed: ["运行失败", "Failed"],
    stopped: ["已停止", "Stopped"],
    cancelled: ["已取消", "Cancelled"],
    interrupted: ["已中断", "Interrupted"],
    incomplete: ["未完成", "Incomplete"],
    orphaned: ["未完成", "Incomplete"],
  };
  return labels[status] ? t(...labels[status]) : status;
}
export function dateText(
  value: string | undefined,
  lang: Language,
  time = false,
): string {
  if (!value) return "—";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "—";
  return time
    ? date.toLocaleTimeString(lang === "zh" ? "zh-CN" : "en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      })
    : date.toLocaleDateString(lang === "zh" ? "zh-CN" : "en-US", {
        month: "short",
        day: "numeric",
      });
}
export function readStored<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}
export function persist(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* App remains usable when storage is unavailable. */
  }
}
