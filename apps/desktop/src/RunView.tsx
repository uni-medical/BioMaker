import { historyMessage } from "./historyMessages";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import Candidates from "./Candidates";
import RunActivity from "./RunActivity";
import RunLogViewer from "./RunLogViewer";
import type { ActivityWindow, LogWindow } from "./runStreams";
import RunResults from "./RunResults";
import RunOrganization from "./RunOrganization";
import type { RunMetadata } from "./runMetadata";
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
  RefreshCw,
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
  runTitle,
  statusLabel,
  type Artifact,
  type Language,
  type Resources,
  type Run,
  type Translate,
} from "./models";

type RunTab = "results" | "activity" | "candidates" | "artifacts" | "log";
type FilePreview = {
  runId: string; name: string; content: string; sha256: string;
  size: number; serviceRevision: number; listedRevision: string;
};
const fileRevision = (artifact: Artifact) => artifact.revision || `${artifact.size}:${artifact.modified_at}`;
function artifactMessage(error: unknown, t: Translate): string {
  const code = /\[artifact:([a-z_]+)\]/.exec(errorMessage(error))?.[1];
  switch (code) {
    case "busy": return t("已有文件正在导出，请完成后再保存。", "A file export is already in progress. Wait for it to finish before saving again.");
    case "changed": return t("文件已更新。请重新预览后再保存；目标文件未被替换。", "The file changed. Preview it again before saving; the destination was not replaced.");
    case "service_changed": return t("本地服务已变化。请重新预览后再保存；目标文件未被替换。", "The local service changed. Preview again before saving; the destination was not replaced.");
    case "too_large": return t("文件较大，请从文件列表保存后查看。", "This file is large. Save it from Files to view it.");
    case "invalid_text": return t("此文件无法作为文本预览。可以从文件列表保存。", "This file cannot be previewed as text. You can save it from Files.");
    case "unavailable": return t("文件暂时无法读取。请刷新实验文件后重试。", "The file could not be read. Refresh the experiment files and try again.");
    case "save_failed": return t("文件未能保存。请检查目标位置的权限和可用空间后重试。", "The file could not be saved. Check access and free space at the destination, then retry.");
    default: return t("文件操作未能完成。请刷新实验后重试。", "The file operation could not finish. Refresh the experiment and try again.");
  }
}

export default function RunView({
  run,
  activity,
  activityError,
  log,
  logError,
  connected,
  onLoadOlderLog,
  onLoadOlderActivity,
  loading,
  detailError,
  lang,
  t,
  busy,
  serviceReady,
  serviceRevision,
  onAction,
  onReuse,
  metadata,
  metadataReady,
  onMetadataChange,
  onRetryLoad,
  toast,
}: {
  run: Run;
  activity: ActivityWindow | null;
  activityError: string;
  log: LogWindow | null;
  logError: string;
  connected: boolean;
  onLoadOlderLog: (cursor: string) => Promise<LogWindow>;
  onLoadOlderActivity: (cursor: string) => Promise<ActivityWindow>;
  loading: boolean;
  detailError: string;
  lang: Language;
  t: Translate;
  busy: string;
  serviceReady: boolean;
  serviceRevision: number;
  onAction: (action: "stop" | "cancel", run: Run) => void;
  onReuse: (run: Run) => void;
  metadata: RunMetadata;
  metadataReady: boolean;
  onMetadataChange: (row: RunMetadata) => void;
  onRetryLoad: () => void;
  toast: (message: string) => void;
}) {
  const [tab, setTab] = useState<RunTab>(() => isActive(run) ? "activity" : "results");
  const [requestedCandidate, setRequestedCandidate] = useState<string | null>(null);
  const tabButtons = useRef<Partial<Record<RunTab, HTMLButtonElement | null>>>({});
  const [promptOpen, setPromptOpen] = useState(false);
  const [preview, setPreview] = useState<FilePreview | null>(null);
  const [previewError, setPreviewError] = useState("");
  const [fileBusy, setFileBusy] = useState("");
  const fileSequence = useRef(0);
  const saving = useRef(false);
  const alive = useRef(true);
  const current = useRef({ run, serviceReady, serviceRevision, preview });
  current.current = { run, serviceReady, serviceRevision, preview };
  useEffect(() => { alive.current = true; return () => { alive.current = false; fileSequence.current++; }; }, []);
  const results = run.observed?.results;
  const artifacts = run.artifacts || [];
  const actions = run.allowed_actions || [];
  const currentPreviewFile = preview ? artifacts.find(file => file.name === preview.name) : undefined;
  const previewStale = !!preview && (!serviceReady || preview.serviceRevision !== serviceRevision
    || !currentPreviewFile || fileRevision(currentPreviewFile) !== preview.listedRevision);
  function switchTab(next: RunTab, focus = false) {
    setTab(next);
    if (focus) tabButtons.current[next]?.focus();
  }
  function closePreview() {
    // Closing the preview abandons an in-flight read, but an approved export
    // still owns its busy state until native code has finished or refused it.
    if (!saving.current) { fileSequence.current++; setFileBusy(""); }
    current.current.preview = null;
    setPreview(null);
    setPreviewError("");
  }
  const done = run.progress?.completed_steps;
  const steps = run.progress?.total_steps || run.options?.agent_steps;
  const progress =
    done !== undefined && done !== null && steps
      ? Math.max(0, Math.min(100, (done / steps) * 100))
      : null;
  async function save(artifact: Artifact, viewed?: FilePreview) {
    if (!serviceReady || saving.current) return;
    if (viewed && previewStale) return;
    saving.current = true;
    const sequence = ++fileSequence.current;
    const runId = run.id;
    setFileBusy(artifact.name);
    setPreviewError("");
    try {
      const path = await window.biomaker.saveArtifact({
        runId: run.id,
        name: artifact.name,
        ...(viewed ? { expectedSha256: viewed.sha256, serviceRevision: viewed.serviceRevision } : {}),
      });
      if (alive.current && current.current.run.id === runId && sequence === fileSequence.current && path) toast(t("文件已保存", "File saved"));
    } catch (e) {
      if (alive.current && current.current.run.id === runId && sequence === fileSequence.current) {
        const message = artifactMessage(e, t);
        if (viewed && current.current.preview?.sha256 === viewed.sha256) setPreviewError(message); else toast(message);
      }
    } finally {
      saving.current = false;
      if (alive.current && sequence === fileSequence.current) setFileBusy("");
    }
  }
  async function saveConsole() {
    if (!serviceReady || saving.current) return;
    saving.current = true;
    const sequence = ++fileSequence.current;
    const runId = run.id;
    setFileBusy("__console__");
    try {
      const path = await window.biomaker.saveConsole({ runId, serviceRevision });
      if (alive.current && current.current.run.id === runId && sequence === fileSequence.current && path) toast(t("运行日志已保存", "Console log saved"));
    } finally {
      saving.current = false;
      if (alive.current && sequence === fileSequence.current) setFileBusy("");
    }
  }
  async function showPreview(artifact: Artifact) {
    if (!serviceReady || saving.current) return;
    if (artifact.size > 500_000) {
      toast(
        t("文件较大，请保存后查看。", "This file is large. Save it to view."),
      );
      return;
    }
    const sequence = ++fileSequence.current;
    const runId = run.id;
    const generation = serviceRevision;
    const listedRevision = fileRevision(artifact);
    setFileBusy(artifact.name);
    setPreviewError("");
    const valid = () => alive.current && current.current.run.id === runId && current.current.serviceReady
      && current.current.serviceRevision === generation && sequence === fileSequence.current;
    try {
      const value = await window.biomaker.previewArtifact({ runId, name: artifact.name });
      if (!valid()) return;
      const latest = current.current.run.artifacts?.find(file => file.name === artifact.name);
      if (!latest || fileRevision(latest) !== listedRevision || value.serviceRevision !== generation) {
        toast(t("文件在读取时发生了变化，请重新预览。", "The file changed while loading. Preview it again."));
        return;
      }
      setPreview({ ...value, listedRevision });
    } catch (e) {
      if (valid()) { const message = artifactMessage(e, t); setPreviewError(message); toast(message); }
    } finally {
      if (alive.current && sequence === fileSequence.current) setFileBusy("");
    }
  }
  async function copy(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      toast(t("已复制", "Copied"));
      return true;
    } catch {
      toast(
        t(
          "复制失败，请手动选择文本。",
          "Could not copy. Select the text manually.",
        ),
      );
      return false;
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
            <RunOrganization run={run} metadata={metadata} ready={serviceReady && metadataReady && !busy} serviceRevision={serviceRevision} t={t} onChange={onMetadataChange} toast={toast} />
            <button
                className="secondary-button"
                onClick={() => onReuse(run)}
                disabled={!!busy || loading || !run.project_snapshot?.task_body?.trim()}
                title={!run.project_snapshot?.task_body?.trim() ? t("原实验没有保存任务内容，无法复用。", "The original task was not saved, so it cannot be reused.") : undefined}
              >
                <RotateCcw size={14} />
                {t("复用为新实验", "Reuse as new experiment")}
            </button>
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
        {run.archived && <div className="archived-note"><Notice>{isActive(run) || run.cleanup_pending
          ? t("此实验已归档，但仍在运行或清理，因此也显示在当前实验中。", "This archived experiment is still running or cleaning up, so it also appears in current experiments.")
          : t("此实验已归档。实验记录和文件均保留，可以随时恢复。", "This experiment is archived. Its records and files are kept, and you can restore it at any time.")}</Notice></div>}
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
            {run.history_import ? historyMessage(run.status_reason || run.error || "", t) : run.status_reason || run.error}
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
                <span>{t("已记录探索候选 / 配置上限", "Recorded exploration candidates / configured limit")}</span>
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
            { id: "results", label: t("结果", "Results"), Icon: FileText },
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
            id={`run-tab-${id}`}
            ref={element => { tabButtons.current[id] = element; }}
            role="tab"
            aria-selected={tab === id}
            aria-controls="run-detail-panel"
            tabIndex={tab === id ? 0 : -1}
            onClick={() => switchTab(id)}
            onKeyDown={event => {
              const order: RunTab[] = ["results", "activity", "candidates", "artifacts", "log"];
              const index = order.indexOf(id);
              const next = event.key === "ArrowRight" ? order[(index + 1) % order.length]
                : event.key === "ArrowLeft" ? order[(index + order.length - 1) % order.length]
                  : event.key === "Home" ? order[0] : event.key === "End" ? order.at(-1) : undefined;
              if (next) { event.preventDefault(); switchTab(next, true); }
            }}
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
          {isActive(run) && connected && !detailError && !activityError && !logError && (
            <span className="live-label">
              <span />
              {t("实时更新", "Live")}
            </span>
          )}
        </div>
      </div>
      <div
        id="run-detail-panel"
        role="tabpanel"
        aria-labelledby={`run-tab-${tab}`}
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
        {tab === "results" && <RunResults
          run={run} t={t} serviceReady={serviceReady} serviceRevision={serviceRevision}
          fileBusy={fileBusy} onPreview={artifact => void showPreview(artifact)}
          onSave={artifact => void save(artifact)} onFiles={() => switchTab("artifacts", true)}
          onCandidate={id => { setRequestedCandidate(id); switchTab("candidates", true); }}
        />}
        <div hidden={tab !== "activity"}>
          <RunActivity run={run} snapshot={activity} error={activityError} loading={loading}
            connected={connected} serviceRevision={serviceRevision} lang={lang} t={t}
            onLoadOlder={onLoadOlderActivity} onRefresh={onRetryLoad} />
        </div>
        {tab === "candidates" && <Candidates run={run} t={t} serviceReady={serviceReady} serviceRevision={serviceRevision} requestedCandidate={requestedCandidate} />}
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
                          disabled={!!fileBusy || !serviceReady}
                        >
                          {t("预览", "Preview")}
                        </button>
                      )}
                      <IconButton
                        label={t(
                          `保存 ${artifact.name}`,
                          `Save ${artifact.name}`,
                        )}
                        disabled={!!fileBusy || !serviceReady}
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
        <div hidden={tab !== "log"}>
          <RunLogViewer key={run.id} visible={tab === "log"} runId={run.id} snapshot={log} loading={loading} error={logError || null}
            connected={connected} active={isActive(run)} lang={lang} t={t}
            onLoadOlder={onLoadOlderLog} onRefresh={onRetryLoad} onCopy={copy}
            onSaveFullLog={saveConsole} exportBusy={!!fileBusy} />
        </div>
      </div>
      {preview && (
        <Modal
          title={preview.name}
          onClose={closePreview}
          t={t}
          className="preview-modal"
        >
          <div className="preview-version-note" role="status">
            {previewStale ? t("文件或本地服务已变化。当前仍显示上次预览，请重新预览后再保存。", "The file or local service changed. The previous preview is still shown. Preview again before saving.")
              : t("保存时会核对文件内容，确保与此预览一致。", "Saving checks the file contents against this preview.")}
          </div>
          {previewError && <Notice danger>{previewError}</Notice>}
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
            {(previewStale || previewError) && <button className="secondary-button"
              disabled={!!fileBusy || !serviceReady || !currentPreviewFile}
              onClick={() => { if (currentPreviewFile) void showPreview(currentPreviewFile); }}>
              {fileBusy ? <Spinner size={14} /> : <RefreshCw size={14} />}
              {t("重新预览", "Preview again")}
            </button>}
            <button
              className="primary-button"
              onClick={() => {
                if (currentPreviewFile) void save(currentPreviewFile, preview);
              }}
              disabled={!!fileBusy || previewStale}
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
