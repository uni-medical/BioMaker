import type { ServiceState } from "./bridge";

export type Language = "zh" | "en";
export type Theme = "system" | "light" | "dark";
export type Translate = (zh: string, en: string) => string;
export interface ProviderCheckResult {
  status: "ok" | "error";
  code: string;
  elapsed_ms: number;
  checked_at: string;
}
export function providerCheckMessage(code: string, t: Translate): string {
  switch (code) {
    case "ok": return t("模型响应测试通过", "Model response test passed");
    case "invalid_url": return t("API 地址无效，请填写完整的 http:// 或 https:// 基础地址。", "Invalid API address. Enter the full http:// or https:// base URL.");
    case "missing_key": return t("请填写此 API 地址的密钥，再手动测试。", "Enter an API key for this address, then test again manually.");
    case "invalid_key": return t("密钥格式无效，请检查是否包含换行或其它无效字符。", "Invalid key format. Check for line breaks or other invalid characters.");
    case "missing_model": return t("请填写要测试的模型名称。", "Enter the model to test.");
    case "auth": return t("服务拒绝了身份验证，请检查密钥、账户权限和 API 地址。", "The provider rejected authentication. Check the key, account permissions, and API address.");
    case "model": return t("服务未找到此模型或当前账户无权使用，请核对模型 ID 和权限。", "The provider could not find this model, or your account cannot use it. Check the model ID and permissions.");
    case "rate_limit": return t("服务达到速率或额度限制，请检查账户额度后手动重试。", "The provider reported a rate or quota limit. Check your account before retrying manually.");
    case "timeout": return t("本次请求超时，未自动重试。可检查服务状态后手动测试。", "This request timed out and was not retried. Check the provider, then test again manually.");
    case "network": return t("无法连接模型服务，请检查网络、代理和 API 地址。", "Could not connect to the provider. Check the network, proxy, and API address.");
    case "request_rejected": return t("服务拒绝了本次短请求。请确认接口支持 Chat Completions 和 max_completion_tokens；这不代表模型不可用。", "The provider rejected this short request. Check support for Chat Completions and max_completion_tokens; this does not mean the model is unavailable.");
    case "invalid_request": return t("测试参数无效，请检查填写内容后重试。", "The test parameters are invalid. Check the entered settings before trying again.");
    case "invalid_response": return t("本次短请求未返回可读的助手文本。可能是接口格式不兼容或生成预算不足，不能据此判断模型不可用。", "This short request returned no readable assistant text. The response format or generation budget may be incompatible; this does not establish that the model is unavailable.");
    case "response_too_large": return t("服务返回的内容超出本次短请求的读取限制，请检查接口兼容性。", "The response exceeded this short test's size limit. Check API compatibility.");
    case "redirect": return t("服务要求重定向，测试已停止。请直接填写最终 API 基础地址。", "The provider requested a redirect, so the test stopped. Enter the final API base URL directly.");
    case "busy": return t("已有模型测试正在进行，请稍后手动测试。", "A model test is already running. Test again manually after it finishes.");
    case "unavailable": return t("模型服务暂时无法完成请求，请稍后手动重试。", "The provider could not complete this request. Try again manually later.");
    case "local_service": return t("本地服务未能完成测试，未自动重试。请检查本地服务状态。", "The local service could not complete the test. It was not retried; check the local service status.");
    default: return t("本次测试未能完成，未自动重试。请检查服务状态后再试。", "This test could not finish and was not retried. Check the service before trying again.");
  }
}
export function sameProviderBaseUrl(first: string, second: string): boolean {
  const normalize = (value: string) => {
    if (value.length > 4096 || /[\u0000-\u001f\u007f]/.test(value)) return null;
    const raw = value.trim() || "https://api.openai.com/v1";
    if (/\s|[?#\\]/u.test(raw)) return null;
    const parts = /^(https?):\/\/([^/]+)(\/.*)?$/i.exec(raw);
    if (!parts || parts[2].includes("@")) return null;
    // Parse the authority only: WHATWG URL parsing must not collapse path
    // segments or repeated slashes before comparing the provider address.
    const url = new URL(`${parts[1]}://${parts[2]}/`);
    if (url.port === "0") return null;
    const rawHost = parts[2].startsWith("[") ? url.hostname : parts[2].split(":")[0];
    const host = /[^\x00-\x7f]/.test(rawHost) ? url.hostname : rawHost.toLowerCase();
    if (!host.startsWith("[") && (!/^[a-z0-9.-]+$/.test(host) || host.replace(/\.$/, "").split(".").some(part => !part))) return null;
    let path = parts[3] || "";
    if (/%(?![0-9a-f]{2})/i.test(path) || path.split("/").some(part => /^\.{1,2}$/.test(part.replace(/%2e/gi, ".")))) return null;
    path = encodeURI(path).replace(/%25([0-9a-f]{2})/gi, "%$1");
    if (path.endsWith("/")) path = path.slice(0, -1);
    return `${url.protocol}//${host}${url.port ? `:${url.port}` : ""}${path}`;
  };
  try {
    const normalized = normalize(first);
    return normalized !== null && normalized === normalize(second);
  } catch { return false; }
}
export function serviceStatusLabel(state: ServiceState, t: Translate): string {
  switch (state.status) {
    case "starting": return t("正在启动本地服务…", "Starting the local service…");
    case "reconnecting": return t("正在恢复本地服务…", "Reconnecting to the local service…");
    case "failed": return t("本地服务不可用", "Local service unavailable");
    case "ready": return t("本地服务已就绪", "Local service ready");
  }
}
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
  managed?: boolean;
  cleanup_pending?: boolean;
  display_name?: string | null;
  archived?: boolean;
  history_import?: { mode: "copy"; source_id: string; source_path: string; name: string; imported_at: string };
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
    operation?: "install" | "check" | "repair" | null;
    checking?: boolean;
    busy_external?: boolean;
    maintenance_unavailable?: boolean;
    health?: "unknown" | "healthy" | "failed";
    checked_at?: string | null;
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
export function runtimeOperation(engine?: EngineStatus["engine"]): "install" | "check" | "repair" | null {
  return engine?.operation || (engine?.checking ? "check" : engine?.installing ? "install" : null);
}
export function isRuntimeBusy(engine?: EngineStatus["engine"]): boolean {
  return !!engine?.busy_external || !!runtimeOperation(engine);
}
export function runtimeOperationLabel(engine: EngineStatus["engine"] | undefined, t: Translate): string {
  if (engine?.busy_external) return t("另一进程正在维护运行环境", "Another process is maintaining the runtime");
  const operation = runtimeOperation(engine);
  return operation === "check" ? t("正在检查运行环境", "Checking runtime")
    : operation === "repair" ? t("正在修复运行环境", "Repairing runtime")
      : t("正在安装运行环境", "Installing runtime");
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
    run.display_name ||
    run.history_import?.name ||
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
    snapshot: ["历史快照", "Historical snapshot"],
    unknown: ["状态未知", "Unknown status"],
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
