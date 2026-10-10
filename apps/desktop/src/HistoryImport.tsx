import { useEffect, useRef, useState } from "react";
import { Check, ChevronRight, FileText, FolderOpen, Import, RefreshCw } from "lucide-react";
import type { ServiceState } from "./bridge";
import { historyMessage } from "./historyMessages";
import { Modal, Notice, Spinner } from "./components";
import { dateText, errorMessage, formatBytes, persist, readStored, request, serviceStatusLabel, statusLabel, type Language, type Translate } from "./models";

interface PreviewRun {
  source_id: string;
  name: string;
  created_at?: string | null;
  status: string;
  total_bytes: number;
  files: { name: string; size: number }[];
  capabilities: { candidates: boolean; code: boolean; report: boolean; results: boolean; logs: boolean };
  already_imported: boolean;
  existing_run_id?: string | null;
  eligible: boolean;
  warnings: string[];
}
interface Preview {
  preview_id: string;
  source_path: string;
  source_kind: "run" | "logs";
  expires_at: string;
  limits: { max_runs: number; max_files_per_run: number; max_file_bytes: number; max_batch_bytes: number };
  runs: PreviewRun[];
  warnings: string[];
}
interface ImportJob {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  completed_runs: number;
  total_runs: number;
  current_name?: string | null;
  imported: { source_id: string; run_id: string; name: string }[];
  skipped: { source_id: string; name: string; reason: string; run_id?: string | null }[];
  errors: { source_id?: string; name?: string; message: string }[];
}
const jobStorage = "biomaker.history-import-job";
const finished = (job: ImportJob) => job.status === "completed" || job.status === "failed";

export default function HistoryImport({ t, lang, serviceState, onClose, onImported, onOpenRun }: {
  t: Translate;
  lang: Language;
  serviceState: ServiceState;
  onClose: () => void;
  onImported: () => void;
  onOpenRun: (id: string) => void;
}) {
  const [source, setSource] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [pending, setPending] = useState<"" | "scan" | "submit">("");
  const [error, setError] = useState("");
  const [needsRescan, setNeedsRescan] = useState(false);
  const [jobId, setJobId] = useState<string | null>(() => {
    const stored = readStored<unknown>(jobStorage, null);
    return typeof stored === "string" ? stored : null;
  });
  const [job, setJob] = useState<ImportJob | null>(null);
  const [pollRetry, setPollRetry] = useState(0);
  const mounted = useRef(true);
  const sequence = useRef(0);
  const currentService = useRef(serviceState);
  currentService.current = serviceState;
  const importedCallback = useRef(onImported);
  importedCallback.current = onImported;
  const reportedJobs = useRef(new Set<string>());
  const serviceReady = serviceState.status === "ready";
  const eligible = preview?.runs.filter(run => run.eligible && !run.already_imported) || [];
  const chosen = eligible.filter(run => selected.has(run.source_id));
  const totalBytes = chosen.reduce((sum, run) => sum + run.total_bytes, 0);
  const tooLarge = !!preview && totalBytes > preview.limits.max_batch_bytes;
  const missingJob = error.includes("Import job was not found.");
  const unfinishedRuns = job ? Math.max(0, job.total_runs - job.imported.length - job.skipped.length) : 0;

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; sequence.current += 1; };
  }, []);

  useEffect(() => {
    if (!serviceReady && preview) setNeedsRescan(true);
  }, [serviceReady, preview]);

  useEffect(() => {
    if (!jobId || !serviceReady) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const revision = serviceState.revision;
    const poll = async () => {
      try {
        const next = await request<ImportJob>(`/api/desktop/history/imports/${encodeURIComponent(jobId)}`);
        if (stopped || currentService.current.revision !== revision) return;
        setJob(next);
        setError("");
        if (finished(next)) {
          if (readStored(jobStorage, null) === jobId) persist(jobStorage, null);
          if (!reportedJobs.current.has(jobId)) {
            reportedJobs.current.add(jobId);
            importedCallback.current();
          }
        } else timer = setTimeout(poll, 1000);
      } catch (e) {
        if (!stopped && currentService.current.revision === revision) setError(errorMessage(e));
      }
    };
    void poll();
    return () => { stopped = true; clearTimeout(timer); };
  }, [jobId, serviceReady, serviceState.revision, pollRetry]);

  async function scan(path: string) {
    if (!serviceReady || pending) return;
    const ticket = ++sequence.current;
    const revision = currentService.current.revision;
    setSource(path); setPending("scan"); setError(""); setPreview(null); setNeedsRescan(false);
    try {
      const next = await request<Preview>("/api/desktop/history/preview", "POST", { path });
      if (!mounted.current || ticket !== sequence.current) return;
      if (currentService.current.revision !== revision) { setNeedsRescan(true); return; }
      setPreview(next);
      setSelected(new Set(next.runs.filter(run => run.eligible && !run.already_imported).map(run => run.source_id)));
    } catch (e) {
      if (mounted.current && ticket === sequence.current) setError(errorMessage(e));
    } finally {
      if (mounted.current && ticket === sequence.current) setPending("");
    }
  }
  async function chooseFolder() {
    if (!serviceReady || pending) return;
    try {
      const path = await window.biomaker.chooseDirectory({ purpose: "history" });
      if (path && mounted.current) await scan(path);
    } catch (e) { if (mounted.current) setError(errorMessage(e)); }
  }
  async function startImport() {
    if (!preview || !serviceReady || pending || !chosen.length || tooLarge || needsRescan) return;
    if (Date.parse(preview.expires_at) <= Date.now()) { setNeedsRescan(true); return; }
    setPending("submit"); setError("");
    try {
      const next = await request<ImportJob>("/api/desktop/history/import", "POST", { preview_id: preview.preview_id, source_ids: chosen.map(run => run.source_id) });
      persist(jobStorage, next.job_id);
      if (mounted.current) { setJobId(next.job_id); setJob(next); }
    } catch (e) {
      if (mounted.current) { setError(errorMessage(e)); setNeedsRescan(true); }
    } finally { if (mounted.current) setPending(""); }
  }
  function startOver() {
    persist(jobStorage, null); setJobId(null); setJob(null); setPreview(null); setError(""); setNeedsRescan(false);
  }
  const close = () => { if (pending !== "submit") onClose(); };

  return <Modal title={t("导入已有实验", "Import existing experiments")} t={t} onClose={close} className="history-import-modal">
    <div className="history-import-body">
      <p className="muted leading">{t("导入实验记录的只读副本，查看以往的结果、代码和日志。数据集、训练工作目录与模型权重不会复制，也不会启动实验。", "Import read-only copies of past results, code, and logs. Datasets, training workspaces, and model weights are not copied, and no experiments are started.")}</p>
      {!serviceReady && <Notice danger={serviceState.status === "failed"}>{serviceStatusLabel(serviceState, t)}</Notice>}
      {error && <Notice danger>{historyMessage(error, t)}</Notice>}
      {jobId ? <>
        <div className="import-progress" role="status">
          {job && finished(job) ? (job.status === "completed" ? <Check size={24} /> : <FileText size={24} />) : <Spinner size={24} />}
          <div><strong>{!job ? t("正在读取导入进度…", "Loading import progress…") : finished(job) ? (job.status === "failed" ? t("导入未完成", "Import did not complete") : job.errors.length ? t("部分实验已导入", "Some experiments were imported") : t("导入完成", "Import complete")) : t("正在导入实验记录…", "Importing experiment records…")}</strong>
            {job && <p>{t(`已处理 ${job.completed_runs} / ${job.total_runs} 个实验`, `${job.completed_runs} of ${job.total_runs} experiments processed`)}{job.current_name && !finished(job) && ` · ${job.current_name}`}</p>}
          </div>
        </div>
        {job && finished(job) && <p className="import-result-summary">{t(`已导入 ${job.imported.length} 个 · 跳过 ${job.skipped.length} 个 · 未完成 ${unfinishedRuns} 个`, `${job.imported.length} imported · ${job.skipped.length} skipped · ${unfinishedRuns} not imported`)}</p>}
        {!!job?.imported.length && <div className="import-result-list">{job.imported.map(run => <button key={run.run_id} aria-label={t(`打开 ${run.name}`, `Open ${run.name}`)} onClick={() => onOpenRun(run.run_id)}><Check size={15} /><span>{run.name}</span><span className="muted">{t("打开实验", "Open experiment")}</span><ChevronRight size={15} /></button>)}</div>}
        {job?.skipped.map((run, index) => <div className="import-result-note" key={`${run.source_id}-${index}`}><strong>{run.name}</strong><p>{historyMessage(run.reason, t)}</p>{run.run_id && <button className="text-button" onClick={() => onOpenRun(run.run_id!)}>{t("打开已导入实验", "Open imported experiment")}</button>}</div>)}
        {job?.errors.map((item, index) => <Notice danger key={`${item.source_id}-${index}`}>{item.name && <strong>{item.name} · </strong>}{historyMessage(item.message, t)}</Notice>)}
        {error && <div className="import-recovery"><p className="muted">{missingJob ? t("导入记录不存在，请重新扫描源文件夹。", "The import record was not found. Scan the source folder again.") : t("暂时无法读取进度，导入任务记录已保留。可以重新读取或稍后回来查看。", "Progress is temporarily unavailable. The import job is kept so you can retry or return later.")}</p><button className="secondary-button" disabled={!serviceReady} onClick={() => setPollRetry(value => value + 1)}><RefreshCw size={14} />{t("重新读取", "Retry loading")}</button>{(missingJob || (job && finished(job))) && <button className="text-button" onClick={startOver}>{t("重新扫描", "Scan again")}</button>}</div>}
      </> : <>
        <div className="import-source">
          <FolderOpen size={20} />
          <div><strong>{source || t("选择以往 BioMaker 的实验文件夹", "Choose a folder from a previous BioMaker run")}</strong><small>{t("支持单个实验文件夹，或包含多个实验的 logs 文件夹。", "Choose a single experiment folder or a logs folder containing multiple experiments.")}</small></div>
          <button className="secondary-button" disabled={!!pending || !serviceReady} onClick={() => void chooseFolder()}>{source ? t("更换", "Change") : t("选择实验目录", "Choose experiment folder")}</button>
        </div>
        {pending === "scan" && <div className="import-progress" role="status"><Spinner size={20} /><span>{t("正在检查实验记录…", "Checking experiment records…")}</span></div>}
        {needsRescan && <Notice>{t("预览已失效，或源文件发生了变化。请重新扫描后再导入。", "The preview is no longer valid or source files have changed. Scan again before importing.")}</Notice>}
        {!!source && !pending && (error || needsRescan) && <button className="secondary-button" disabled={!serviceReady} onClick={() => void scan(source)}><RefreshCw size={14} />{t("重新扫描", "Scan again")}</button>}
        {preview && <>
          {preview.warnings.map((warning, index) => <Notice key={index}>{historyMessage(warning, t)}</Notice>)}
          <div className="import-selection-heading"><strong>{t(`发现 ${preview.runs.length} 个实验`, `${preview.runs.length} experiments found`)}</strong><button className="text-button" disabled={!!pending || !eligible.length} onClick={() => setSelected(selected.size ? new Set() : new Set(eligible.map(run => run.source_id)))}>{selected.size ? t("清空选择", "Clear selection") : t("全选可导入实验", "Select available experiments")}</button></div>
          {!preview.runs.length && <div className="import-empty"><FileText size={25} /><p>{t("未找到可识别的实验记录。请选择单个实验文件夹或它的 logs 上级目录。", "No experiment records were found. Choose an experiment folder or its parent logs folder.")}</p></div>}
          <div className="import-run-list">{preview.runs.map(run => <div className={`import-run ${!run.eligible || run.already_imported ? "unavailable" : ""}`} key={run.source_id}>
            <label className="import-run-choice"><input type="checkbox" aria-label={t(`选择 ${run.name}`, `Select ${run.name}`)} checked={selected.has(run.source_id)} disabled={!!pending || !run.eligible || run.already_imported} onChange={event => setSelected(previous => { const next = new Set(previous); if (event.target.checked) next.add(run.source_id); else next.delete(run.source_id); return next; })} /><span><strong>{run.name}</strong><small>{dateText(run.created_at || undefined, lang)} · {statusLabel({ id: run.source_id, status: run.status, detailed_status: run.status }, t)} · {formatBytes(run.total_bytes)}</small></span><span className="import-readonly">{run.already_imported ? t("已导入", "Imported") : !run.eligible ? t("不可导入", "Unavailable") : t("只读副本", "Read-only copy")}</span></label>
            <div className="import-capabilities">{([["candidates", "候选方案", "Candidates"], ["code", "代码", "Code"], ["report", "报告", "Report"], ["results", "指标", "Metrics"], ["logs", "日志", "Logs"]] as const).filter(([key]) => run.capabilities[key]).map(([key, zh, en]) => <span key={key}>{t(zh, en)}</span>)}</div>
            {run.warnings.map((warning, index) => <p className="import-warning" key={index}>{historyMessage(warning, t)}</p>)}
            {run.already_imported && run.existing_run_id && <button className="text-button" disabled={!!pending} onClick={() => onOpenRun(run.existing_run_id!)}>{t("打开已导入实验", "Open imported experiment")}</button>}
            <details><summary>{t(`${run.files.length} 个文件`, `${run.files.length} files`)}</summary><ul className="import-files">{run.files.map(file => <li key={file.name}><span>{file.name}</span><small>{formatBytes(file.size)}</small></li>)}</ul></details>
          </div>)}</div>
          {tooLarge && <Notice danger>{t(`所选文件超过单批次 ${formatBytes(preview.limits.max_batch_bytes)} 的上限，请减少选择。`, `The selected files exceed the ${formatBytes(preview.limits.max_batch_bytes)} batch limit. Select fewer experiments.`)}</Notice>}
        </>}
      </>}
    </div>
    <div className="history-import-footer">
      {jobId ? <>
        {job && finished(job) && <button className="secondary-button" onClick={startOver}>{t("导入其它实验", "Import more experiments")}</button>}
        <button className="primary-button" onClick={onClose}>{job && finished(job) ? t("完成", "Done") : t("稍后查看", "View later")}</button>
      </> : <>
        <span className="muted">{chosen.length ? t(`已选 ${chosen.length} 个 · ${formatBytes(totalBytes)}`, `${chosen.length} selected · ${formatBytes(totalBytes)}`) : t("选择要导入的实验", "Select experiments to import")}</span>
        <button className="secondary-button" disabled={pending === "submit"} onClick={close}>{t("取消", "Cancel")}</button>
        <button className="primary-button" disabled={!preview || !chosen.length || tooLarge || needsRescan || !!pending || !serviceReady} onClick={() => void startImport()}>{pending === "submit" ? <Spinner size={15} /> : <Import size={15} />}{pending === "submit" ? t("正在提交…", "Submitting…") : t("导入所选实验", "Import selected experiments")}</button>
      </>}
    </div>
  </Modal>;
}
