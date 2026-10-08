import { useState } from "react";
import ReactMarkdown from "react-markdown";
import Candidates from "./Candidates";
import {
  Activity,
  ArrowDownToLine,
  Check,
  ChevronDown,
  ChevronRight,
  Clock3,
  Code2,
  Copy,
  File,
  FileCode2,
  FileText,
  FolderOpen,
  Layers3,
  Play,
  RotateCcw,
  Square,
  Terminal,
  X,
} from "lucide-react";
import { IconButton, Mark, Modal, Notice, Spinner } from "./components";
import {
  dateText,
  errorMessage,
  formatBytes,
  isActive,
  request,
  runTitle,
  statusLabel,
  type Artifact,
  type Language,
  type Resources,
  type Run,
  type RunEvent,
  type Translate,
} from "./models";

const eventNames: Record<string, [string, string]> = {
  run_started: ["实验已启动", "Experiment started"],
  run_finished: ["实验已完成", "Experiment completed"],
  run_failed: ["实验运行失败", "Experiment failed"],
  run_interrupted: ["实验已中断", "Experiment interrupted"],
  round_started: ["开始新一轮探索", "Exploration round started"],
  round_finished: ["本轮探索完成", "Exploration round completed"],
  node_started: ["执行候选方案", "Running a candidate"],
  node_finished: ["候选方案执行完成", "Candidate execution completed"],
  node_failed: ["候选方案执行失败", "Candidate execution failed"],
  final_retrain_started: ["开始最终重训", "Final retraining started"],
  final_retrain_finished: ["最终重训完成", "Final retraining completed"],
  final_retrain_failed: ["最终重训失败", "Final retraining failed"],
  agent_started: ["智能体开始工作", "Agent started"],
  agent_finished: ["智能体工作完成", "Agent finished"],
  code_generated: ["已生成实验代码", "Experiment code generated"],
  evaluation_started: ["开始评估", "Evaluation started"],
  evaluation_finished: ["评估完成", "Evaluation completed"],
  search_started: ["开始检索", "Research started"],
  report_generated: ["实验报告已生成", "Experiment report created"],
  checkpoint_saved: ["检查点已保存", "Checkpoint saved"],
  budget_exhausted: ["已达到资源上限", "Resource limit reached"],
  run_stopping: ["正在停止实验", "Stopping experiment"],
  final_retrain_skipped: ["已跳过最终重训", "Final retraining skipped"],
  node_evaluated: ["候选方案已评估", "Candidate evaluated"],
  candidate_started: ["开始候选方案", "Candidate started"],
  candidate_finished: ["候选方案完成", "Candidate completed"],
  agent_step_started: ["智能体执行步骤", "Agent step started"],
  agent_step_finished: ["智能体步骤完成", "Agent step completed"],
  model_request_started: ["正在请求模型", "Requesting model"],
  model_request_finished: ["模型响应已返回", "Model response received"],
  tool_started: ["正在执行工具", "Running tool"],
  tool_finished: ["工具执行完成", "Tool completed"],
};
function EventItem({
  event,
  lang,
  t,
}: {
  event: RunEvent;
  lang: Language;
  t: Translate;
}) {
  const [open, setOpen] = useState(false);
  const name = event.event || "event";
  const title = eventNames[name]
    ? t(...eventNames[name])
    : name.replaceAll("_", " ");
  const summary = event.message || event.error || event.reason;
  const failure = name.includes("failed") || !!event.error;
  const fields = Object.fromEntries(
    Object.entries(event).filter(
      ([key]) => !["event", "ts", "message", "error", "reason"].includes(key),
    ),
  );
  return (
    <div className={`event-item ${failure ? "event-error" : ""}`}>
      <span className="event-node">
        {failure ? (
          <X size={12} />
        ) : name.endsWith("finished") || name.endsWith("saved") ? (
          <Check size={12} />
        ) : (
          <span />
        )}
      </span>
      <div className="event-body">
        <button
          className="event-heading"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
        >
          <strong>{title}</strong>
          {event.step !== undefined && (
            <span className="event-step">
              {t("步骤", "Step")} {event.step}
            </span>
          )}
          <time>{dateText(event.ts, lang, true)}</time>
          <ChevronRight size={13} className={open ? "rotate-down" : ""} />
        </button>
        {summary && <p>{String(summary)}</p>}
        {open && (
          <pre className="event-data">{JSON.stringify(fields, null, 2)}</pre>
        )}
      </div>
    </div>
  );
}
export default function RunView({
  run,
  events,
  log,
  loading,
  detailError,
  lang,
  t,
  busy,
  onAction,
  onRetryLoad,
  toast,
}: {
  run: Run;
  events: RunEvent[];
  log: string;
  loading: boolean;
  detailError: string;
  lang: Language;
  t: Translate;
  busy: string;
  onAction: (action: "stop" | "cancel" | "requeue", run: Run) => void;
  onRetryLoad: () => void;
  toast: (message: string) => void;
}) {
  const [tab, setTab] = useState<
    "activity" | "candidates" | "artifacts" | "log"
  >("activity");
  const [promptOpen, setPromptOpen] = useState(false);
  const [preview, setPreview] = useState<{
    name: string;
    content: string;
  } | null>(null);
  const [fileBusy, setFileBusy] = useState("");
  const results = run.observed?.results;
  const artifacts = run.artifacts || [];
  const actions = run.allowed_actions || [];
  const done = run.progress?.completed_steps;
  const steps = run.progress?.total_steps || run.options?.agent_steps;
  const progress =
    done !== undefined && done !== null && steps
      ? Math.max(0, Math.min(100, (done / steps) * 100))
      : null;
  async function save(artifact: Artifact) {
    setFileBusy(artifact.name);
    try {
      const path = await window.biomaker.saveArtifact({
        runId: run.id,
        name: artifact.name,
      });
      if (path) toast(t("文件已保存", "File saved"));
    } catch (e) {
      toast(errorMessage(e));
    } finally {
      setFileBusy("");
    }
  }
  async function showPreview(artifact: Artifact) {
    if (artifact.size > 500_000) {
      toast(
        t("文件较大，请保存后查看。", "This file is large. Save it to view."),
      );
      return;
    }
    setFileBusy(artifact.name);
    try {
      const content = await request<unknown>(artifact.url);
      setPreview({
        name: artifact.name,
        content:
          typeof content === "string"
            ? content
            : JSON.stringify(content, null, 2),
      });
    } catch (e) {
      toast(errorMessage(e));
    } finally {
      setFileBusy("");
    }
  }
  async function copy(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      toast(t("已复制", "Copied"));
    } catch {
      toast(
        t(
          "复制失败，请手动选择文本。",
          "Could not copy. Select the text manually.",
        ),
      );
    }
  }
  return (
    <div className="run-view">
      <div className="run-intro">
        <div className="run-title-row">
          <div>
            <div className={`status-label status-${run.status}`}>
              <span className="status-dot" />
              <span>{statusLabel(run, t)}</span>
              <span className="date-separator" />
              {dateText(run.timestamps?.created_at, lang)}
            </div>
            <h1>{runTitle(run)}</h1>
          </div>
          <div className="run-actions">
            {actions.includes("requeue") && (
              <button
                className="secondary-button"
                onClick={() => onAction("requeue", run)}
                disabled={!!busy}
              >
                {busy === "requeue" ? <Spinner /> : <RotateCcw size={14} />}
                {t("再次运行", "Run again")}
              </button>
            )}
            {actions.includes("stop") && (
              <button
                className="secondary-button"
                onClick={() => onAction("stop", run)}
                disabled={!!busy}
              >
                {busy === "stop" ? <Spinner /> : <Square size={13} />}
                {t("停止", "Stop")}
              </button>
            )}
            {actions.includes("cancel") && (
              <button
                className="secondary-button"
                onClick={() => onAction("cancel", run)}
                disabled={!!busy}
              >
                {busy === "cancel" ? <Spinner /> : <X size={14} />}
                {t("取消排队", "Cancel queued run")}
              </button>
            )}
          </div>
        </div>
        {run.project_snapshot?.task_body && (
          <div className="task-message">
            <button
              className="task-message-toggle"
              onClick={() => setPromptOpen(!promptOpen)}
              aria-expanded={promptOpen}
            >
              <FileText size={15} />
              <span>{t("实验任务", "Experiment task")}</span>
              <ChevronDown
                size={14}
                className={promptOpen ? "" : "rotate-left"}
              />
            </button>
            <div className={`task-markdown ${promptOpen ? "expanded" : ""}`}>
              <ReactMarkdown>{run.project_snapshot.task_body}</ReactMarkdown>
            </div>
          </div>
        )}
        {(run.status_reason || run.error) && (
          <Notice danger={run.detailed_status === "failed"}>
            {run.status_reason || run.error}
          </Notice>
        )}
        {run.read_only && (
          <p className="read-only-note">
            {t(
              "从历史日志导入，仅供查看。",
              "Imported from historical logs. View only.",
            )}
          </p>
        )}
        {results?.count || progress !== null ? (
          <div className="run-summary">
            {progress !== null ? (
              <div>
                <span>{t("探索进度", "Exploration progress")}</span>
                <strong>
                  {done}
                  <small> / {steps}</small>
                </strong>
                <div className="progress-track">
                  <span style={{ width: `${progress}%` }} />
                </div>
              </div>
            ) : null}
            {!!results?.count && (
              <div>
                <span>{t("已记录的候选方案", "Recorded candidates")}</span>
                <strong>{results.count}</strong>
              </div>
            )}
            {results?.best_metric !== null &&
              results?.best_metric !== undefined && (
                <div>
                  <span>{t("最佳记录指标", "Best recorded metric")}</span>
                  <strong>
                    {Number.isFinite(results.best_metric)
                      ? Number(results.best_metric.toPrecision(6)).toString()
                      : "—"}
                  </strong>
                  <small>
                    {t("以实验定义的指标为准", "As defined by this experiment")}
                  </small>
                </div>
              )}
          </div>
        ) : null}
      </div>
      <div
        className="run-tabs"
        role="tablist"
        aria-label={t("实验详情", "Experiment details")}
      >
        {(
          [
            { id: "activity", label: t("活动", "Activity"), Icon: Activity },
            {
              id: "candidates",
              label: t("候选方案", "Candidates"),
              Icon: Code2,
            },
            { id: "artifacts", label: t("文件", "Files"), Icon: Layers3 },
            { id: "log", label: t("运行日志", "Console"), Icon: Terminal },
          ] as const
        ).map(({ id, label, Icon }) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={tab === id ? "selected" : ""}
          >
            <Icon size={15} />
            {label}
            {id === "artifacts" && artifacts.length > 0 && (
              <span className="tab-count">{artifacts.length}</span>
            )}
          </button>
        ))}
        <div className="tabs-right">
          {loading && <Spinner size={14} />}
          {isActive(run) && (
            <span className="live-label">
              <span />
              {t("实时更新", "Live")}
            </span>
          )}
        </div>
      </div>
      <div
        role="tabpanel"
        className={`run-tab-content ${tab === "log" ? "log-panel" : ""}`}
      >
        {detailError && (
          <Notice danger>
            <span>{detailError}</span>
            <button className="text-button" onClick={onRetryLoad}>
              {t("重试", "Retry")}
            </button>
          </Notice>
        )}
        {tab === "activity" && (
          <>
            {events.length > 0 ? (
              <div className="timeline">
                {events.map((event, i) => (
                  <EventItem
                    key={`${event.ts}-${event.event}-${i}`}
                    event={event}
                    lang={lang}
                    t={t}
                  />
                ))}
                {isActive(run) && (
                  <div className="event-item current">
                    <span className="event-node">
                      <Spinner size={12} />
                    </span>
                    <div className="event-body">
                      <span>
                        {run.status === "queued"
                          ? t("等待运行", "Waiting to run")
                          : run.status === "stopping"
                            ? t(
                                "正在结束当前进程…",
                                "Ending the current process…",
                              )
                            : t(
                                "等待下一条活动…",
                                "Waiting for the next activity…",
                              )}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="detail-empty">
                <Activity size={28} />
                <h3>
                  {loading
                    ? t("正在读取实验活动", "Loading activity")
                    : run.status === "queued"
                      ? t("实验已加入队列", "Experiment is queued")
                      : t("尚无活动记录", "No activity recorded yet")}
                </h3>
                <p>
                  {run.status === "queued"
                    ? t(
                        "轮到此实验时会自动开始，运行过程将在这里记录。",
                        "This experiment starts automatically when its turn arrives. Activity will appear here.",
                      )
                    : t(
                        "实验产生的步骤和结果会按时间显示在这里。",
                        "Experiment steps and results appear here in chronological order.",
                      )}
                </p>
              </div>
            )}
            {(run.observed?.event_count || 0) > events.length && (
              <p className="tail-note">
                {t(
                  `显示最近 ${events.length} 条活动，完整记录可在文件中保存。`,
                  `Showing the latest ${events.length} events. Save the event file for the full history.`,
                )}
              </p>
            )}
          </>
        )}
        {tab === "candidates" && <Candidates run={run} t={t} />}
        {tab === "artifacts" && (
          <>
            {artifacts.length > 0 ? (
              <div className="artifact-list">
                {artifacts.map((artifact) => {
                  const canPreview =
                    /\.(md|txt|json|jsonl|log|csv|tsv|py|yaml|yml)$/i.test(
                      artifact.name,
                    );
                  return (
                    <div className="artifact-row" key={artifact.name}>
                      <span className="file-icon">
                        {/\.(py|json|jsonl|yaml|html)$/.test(artifact.name) ? (
                          <FileCode2 size={20} />
                        ) : (
                          <FileText size={20} />
                        )}
                      </span>
                      <div className="artifact-name">
                        <strong>{artifact.name}</strong>
                        <span>
                          {formatBytes(artifact.size)} ·{" "}
                          {dateText(
                            new Date(artifact.modified_at * 1000).toISOString(),
                            lang,
                          )}
                        </span>
                      </div>
                      {canPreview && (
                        <button
                          className="text-button"
                          onClick={() => showPreview(artifact)}
                          disabled={!!fileBusy}
                        >
                          {t("预览", "Preview")}
                        </button>
                      )}
                      <IconButton
                        label={t(
                          `保存 ${artifact.name}`,
                          `Save ${artifact.name}`,
                        )}
                        disabled={!!fileBusy}
                        onClick={() => save(artifact)}
                      >
                        {fileBusy === artifact.name ? (
                          <Spinner />
                        ) : (
                          <ArrowDownToLine size={16} />
                        )}
                      </IconButton>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="detail-empty">
                <File size={28} />
                <h3>
                  {t("输出文件会保存在这里", "Output files will appear here")}
                </h3>
                <p>
                  {t(
                    "实验代码、日志、指标和报告会随实验进展逐步生成。",
                    "Code, logs, metrics, and reports are generated as the experiment progresses.",
                  )}
                </p>
              </div>
            )}
          </>
        )}
        {tab === "log" && (
          <>
            <div className="log-toolbar">
              <span>{t("最近 400 行", "Last 400 lines")}</span>
              <button
                className="text-button"
                onClick={() => copy(log)}
                disabled={!log}
              >
                <Copy size={13} />
                {t("复制", "Copy")}
              </button>
            </div>
            <pre className="console-log" tabIndex={0}>
              {log || t("运行日志尚未生成。", "No console output yet.")}
            </pre>
          </>
        )}
      </div>
      {preview && (
        <Modal
          title={preview.name}
          onClose={() => setPreview(null)}
          t={t}
          className="preview-modal"
        >
          <div className="file-preview">
            {preview.name.endsWith(".md") ? (
              <div className="markdown">
                <ReactMarkdown>{preview.content}</ReactMarkdown>
              </div>
            ) : (
              <pre tabIndex={0}>{preview.content}</pre>
            )}
          </div>
          <div className="preview-footer">
            <button
              className="secondary-button"
              onClick={() => copy(preview.content)}
            >
              <Copy size={14} />
              {t("复制内容", "Copy contents")}
            </button>
            <button
              className="primary-button"
              onClick={() => {
                const artifact = artifacts.find((a) => a.name === preview.name);
                if (artifact) void save(artifact);
              }}
              disabled={!!fileBusy}
            >
              <ArrowDownToLine size={14} />
              {t("保存文件", "Save file")}
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
export function ResourcesView({
  data,
  t,
}: {
  data: Resources | null;
  t: Translate;
}) {
  const cpu = data?.cpu?.utilization_percent;
  const memory = data?.memory?.utilization_percent;
  return (
    <section className="inspector-section resource-section">
      <h3>{t("这台电脑", "This computer")}</h3>
      <div className="resource-row">
        <span>CPU</span>
        <strong>
          {cpu !== undefined && cpu !== null ? `${Math.round(cpu)}%` : "—"}
        </strong>
      </div>
      <div className="resource-meter">
        <span style={{ width: `${cpu ?? 0}%` }} />
      </div>
      <div className="resource-row">
        <span>{t("内存", "Memory")}</span>
        <strong>
          {memory !== undefined && memory !== null
            ? `${Math.round(memory)}%`
            : "—"}
        </strong>
      </div>
      <div className="resource-meter">
        <span style={{ width: `${memory ?? 0}%` }} />
      </div>
      {data?.memory?.total_bytes ? (
        <div className="resource-caption">
          {formatBytes(data.memory.used_bytes)} /{" "}
          {formatBytes(data.memory.total_bytes)}
        </div>
      ) : null}
      <div className="accelerator">
        <Code2 size={14} />
        <span>
          {data?.nvidia_gpus?.length
            ? data.nvidia_gpus.map((g) => g.name).join(", ")
            : data?.apple_mps?.available === true
              ? "Apple Silicon · MPS"
              : data?.apple_mps?.available === false
                ? t("CPU 计算", "CPU compute")
                : t("加速设备待确认", "Accelerator not yet verified")}
        </span>
      </div>
      {data?.stale && (
        <p className="resource-caption">
          {t("资源数据可能已过期", "Resource data may be stale")}
        </p>
      )}
    </section>
  );
}
