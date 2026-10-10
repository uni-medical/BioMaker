import { useEffect, useId, useRef, useState } from "react";
import { ArrowDownToLine, Check, Code2, FileSpreadsheet, FileText, GitBranch, RefreshCw, RotateCcw } from "lucide-react";
import { Notice, Spinner } from "./components";
import { isActive, request, type Artifact, type Run, type Translate } from "./models";
import "./results.css";

interface ResultArtifact {
  name: string;
  state: "available" | "missing" | "too_large" | "unsafe" | "invalid" | "unreadable" | "changing";
  revision: string | null;
  reason?: string;
}
interface ResultOverview {
  run_id: string;
  origin: "managed" | "imported_snapshot" | "historical";
  run_status: string;
  snapshot: { status: "stable" | "changing" | "unavailable"; revision: string | null };
  selection: {
    status: "recorded" | "code_match" | "none" | "unavailable" | "conflict";
    source: "step_saved" | "best_solution.py" | null;
    node_id: string | null;
    candidate_step: number | null;
    metric: { value: number; maximize: boolean | null; name?: string; source: "journal" } | null;
    metrics?: Record<string, number>;
    metric_specs?: { name: string; maximize: boolean }[];
    ranking?: { average_rank?: number; worst_rank?: number; candidate_count?: number; metric_ranks?: Record<string, number>; snapshot_version?: string };
    code_match: boolean | null;
    reason?: string;
  };
  final_retrain: {
    state: "succeeded" | "failed" | "disabled" | "recorded_without_receipt" | "unknown";
    source: "final_retrain_result.json" | "events.jsonl" | "run_manifest.json" | null;
    node_id: string | null;
    reason?: string;
    train_rows_before?: number;
    train_rows_after?: number;
    score_rows_added?: number;
  };
  artifacts: ResultArtifact[];
  warnings: string[];
}

function isOverview(value: unknown, runId: string): value is ResultOverview {
  if (!value || typeof value !== "object") return false;
  const data = value as Partial<ResultOverview>;
  return data.run_id === runId && !!data.snapshot && !!data.selection && !!data.final_retrain
    && Array.isArray(data.artifacts) && Array.isArray(data.warnings)
    && ["stable", "changing", "unavailable"].includes(data.snapshot.status)
    && ["recorded", "code_match", "none", "unavailable", "conflict"].includes(data.selection.status)
    && ["succeeded", "failed", "disabled", "recorded_without_receipt", "unknown"].includes(data.final_retrain.state);
}

function selectionExplanation(selection: ResultOverview["selection"], t: Translate): string {
  if (selection.status === "conflict") return t("运行记录与保存的方案不一致。请查看候选记录和方案代码后再使用。", "The run record and saved solution disagree. Review the candidates and solution code before using them.");
  if (selection.status === "code_match") return t("保存的方案代码与此候选一致；目前无法确认运行中的选择记录。", "The saved solution matches this candidate’s code; the run’s selection record could not be confirmed.");
  if (selection.status === "recorded") return selection.code_match === true
    ? t("依据运行中的选择记录，保存的方案代码与此候选一致。", "Based on the run’s selection record; the saved solution matches this candidate’s code.")
    : t("依据运行中的选择记录，尚未核对保存的方案代码。", "Based on the run’s selection record; the saved solution code could not be checked.");
  switch (selection.reason) {
    case "recorded_no_selection": return t("运行记录中没有选定候选方案。", "The run record contains no selected candidate.");
    case "no_valid_candidates": return t("尚未记录可用的候选方案。可以查看候选执行情况和运行日志。", "No usable candidate was recorded. Review candidate execution and the console.");
    case "best_code_ambiguous": return t("多个候选使用相同代码，无法据此确认选定方案。", "Several candidates contain the same code, so it cannot identify the selected candidate.");
    case "best_code_no_match": return t("保存的方案代码与候选记录没有匹配项。", "The saved solution code does not match a recorded candidate.");
    case "selected_candidate_missing": return t("选择记录所指的候选不在已保存记录中。", "The candidate named in the selection record is missing from the saved records.");
    case "journal_invalid":
    case "duplicate_candidate_ids":
    case "selection_event_invalid":
    case "selected_candidate_invalid": return t("部分候选或选择记录无法读取。请查看原始文件。", "Some candidate or selection records could not be read. Review the original files.");
    case "snapshot_changed": return t("文件正在更新，刷新后再确认选定方案。", "Files are being updated. Refresh to check the selected candidate.");
    default: return t("现有记录不足以确认选定方案。已保存的文件仍可在下方查看。", "The available records do not identify a selected candidate. Saved files are listed below.");
  }
}

function retrainDescription(retrain: ResultOverview["final_retrain"], t: Translate): { title: string; text: string; tone: string } {
  switch (retrain.state) {
    case "succeeded": return { title: t("最终重训已完成", "Final retraining completed"), text: t("重训记录标记为成功。可查看记录中的执行情况和指标。", "The retraining receipt records success. Review it for execution details and metrics."), tone: "success" };
    case "failed": return { title: t("最终重训未成功", "Final retraining failed"), text: t("查看重训记录和运行日志，确认失败原因；探索阶段的文件仍保留。", "Review the retraining receipt and console for the failure reason. Exploration files are retained."), tone: "warning" };
    case "disabled": return { title: t("未启用最终重训", "Final retraining was disabled"), text: t("此实验的配置没有启用最终重训。", "Final retraining was disabled in this experiment’s configuration."), tone: "neutral" };
    case "recorded_without_receipt": return { title: t("有重训活动，结果记录缺失", "Retraining activity found; receipt missing"), text: t("活动记录不能替代重训结果文件，请查看已有活动和文件。", "Activity records do not replace the retraining receipt. Review the available activity and files."), tone: "neutral" };
    default:
      if (retrain.reason === "events_unavailable") return { title: t("重训阶段无法确认", "Retraining attempt unconfirmed"), text: t("事件记录暂不可用，无法确认此重训记录对应的阶段。", "Event records are unavailable, so this retraining receipt cannot be associated with the current attempt."), tone: "neutral" };
      if (retrain.reason === "final_retrain_in_progress") return { title: t("正在最终重训", "Final retraining in progress"), text: t("已记录重训开始，结果尚未写入。", "Retraining has started; its result has not been recorded yet."), tone: "neutral" };
      if (retrain.reason === "final_retrain_incomplete") return { title: t("重训未留下完成记录", "No retraining completion recorded"), text: t("实验已结束，但没有找到完整的重训结果。", "The experiment ended without a complete retraining result."), tone: "neutral" };
      if (retrain.reason === "conflicting_retrain_records") return { title: t("重训记录不一致", "Retraining records disagree"), text: t("现有记录不足以确认重训结果，请核对原始文件。", "The records do not establish the retraining result. Review the original files."), tone: "warning" };
      if (retrain.reason === "receipt_invalid") return { title: t("重训记录无法读取", "Retraining receipt unreadable"), text: t("结果文件格式无法识别，可保存原始文件查看。", "The receipt format could not be recognized. Save the original file to inspect it."), tone: "neutral" };
      return { title: t("尚无重训结果记录", "No retraining result recorded"), text: retrain.reason === "imported_receipt_not_included"
        ? t("导入的历史快照没有包含重训结果文件。", "The imported snapshot does not include a retraining receipt.")
        : t("当前文件未提供可确认的重训结果。", "The available files do not establish a retraining result."), tone: "neutral" };
  }
}

export default function RunResults({ run, t, serviceReady, serviceRevision, refreshKey, onPreview, onSave, onFiles, onCandidate, fileBusy }: {
  run: Run;
  t: Translate;
  serviceReady: boolean;
  serviceRevision: number;
  refreshKey?: string | number;
  onPreview: (artifact: Artifact) => void;
  onSave: (artifact: Artifact) => void;
  onFiles: () => void;
  onCandidate: (id: string) => void;
  fileBusy: string;
}) {
  const [loaded, setLoaded] = useState<{ overview: ResultOverview; signature: string } | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  const [retry, setRetry] = useState(0);
  const generation = useRef(0);
  const headingId = useId();
  const artifactRevision = (run.artifacts || []).map(file => `${file.name}:${file.revision || ""}:${file.size}:${file.modified_at}`).sort().join("|");
  const signature = JSON.stringify([run.id, run.status, run.detailed_status, artifactRevision, serviceRevision, refreshKey ?? null]);
  useEffect(() => {
    const sequence = ++generation.current;
    let valid = true;
    setFailed(false);
    if (!serviceReady) { setLoading(false); return; }
    setLoading(true);
    request<unknown>(`/api/desktop/runs/${encodeURIComponent(run.id)}/result`).then(value => {
      if (!valid || sequence !== generation.current) return;
      if (!isOverview(value, run.id)) throw new Error("Invalid result overview");
      setLoaded({ overview: value, signature });
    }).catch(() => { if (valid && sequence === generation.current) setFailed(true); })
      .finally(() => { if (valid && sequence === generation.current) setLoading(false); });
    return () => { valid = false; };
  }, [signature, serviceReady, retry]);

  const overview = loaded?.signature === signature ? loaded.overview : null;
  const stable = !!overview && overview.snapshot.status === "stable" && !failed;
  const selection = stable ? overview.selection : null;
  const matched = selection?.status === "recorded" || selection?.status === "code_match";
  const metric = matched && selection.metric && typeof selection.metric.value === "number" && Number.isFinite(selection.metric.value) ? selection.metric : null;
  const metrics = matched && selection.metrics && typeof selection.metrics === "object"
    ? Object.entries(selection.metrics).filter(([, value]) => typeof value === "number" && Number.isFinite(value)) : [];
  const averageRank = matched && typeof selection.ranking?.average_rank === "number" && Number.isFinite(selection.ranking.average_rank) ? selection.ranking.average_rank : null;
  const retrain = stable ? retrainDescription(overview.final_retrain, t) : null;
  const readyForFiles = serviceReady && !fileBusy;
  const resources = [
    { name: "report.md", title: t("实验报告", "Experiment report"), description: t("根据候选记录生成的文字总结，阅读时请对照指标与执行记录。", "A generated summary of candidate records. Read it alongside metrics and execution records."), Icon: FileText },
    { name: "best_solution.py", title: t("保存的方案代码", "Saved solution code"), description: t("实验保存的 Python 方案，可预览或另存。", "The Python solution saved by this experiment."), Icon: Code2 },
    { name: "results.tsv", title: t("候选指标表", "Candidate metrics table"), description: t("每个候选的记录指标、执行状态和资源信息。", "Recorded metrics, execution states, and resources for each candidate."), Icon: FileSpreadsheet },
    { name: "final_retrain_result.json", title: t("最终重训记录", "Final retraining receipt"), description: t("重训是否成功、执行情况及已记录的指标。", "Retraining outcome, execution details, and recorded metrics."), Icon: RotateCcw },
  ];

  return <section className="run-results" aria-labelledby={headingId} aria-busy={loading}>
    <div className="result-heading">
      <div><h2 id={headingId}>{t("结果概览", "Results overview")}</h2><p>{overview?.origin === "imported_snapshot"
        ? t("这是导入时保存的历史快照，结果仅反映其中的文件。", "This imported snapshot reflects only the files saved at import time.")
        : overview?.origin === "historical"
          ? t("以下内容来自已有的实验文件；缺少记录时不会推断结果。", "Based on saved experiment files. Missing records are left unconfirmed.")
          : isActive(run)
            ? t("实验仍在进行，已保存的方案和文件可能继续更新。", "This experiment is still running. Saved solutions and files may change.")
            : t("查看保存的方案、候选指标与最终重训记录。", "Review the saved solution, candidate metrics, and final retraining receipt.")}</p></div>
      <button className="text-button result-refresh" onClick={() => setRetry(value => value + 1)} disabled={!serviceReady || loading}>
        {loading ? <Spinner size={13} /> : <RefreshCw size={13} />}{t("刷新结果", "Refresh results")}
      </button>
    </div>
    <div className="result-status" role="status">
      {!serviceReady ? <Notice>{t("等待本地服务恢复后读取结果。", "Results will load when the local service is available.")}</Notice>
        : failed ? <Notice danger>{t("暂时无法读取结果。请重试，或在文件中查看已保存内容。", "Results could not be loaded. Retry, or view the saved files.")}<button className="text-button" onClick={() => setRetry(value => value + 1)}>{t("重试", "Retry")}</button></Notice>
        : !overview ? <div className="result-loading"><Spinner size={20} /><span>{t("正在读取结果记录", "Loading result records")}</span></div>
        : overview.snapshot.status === "changing" ? <Notice>{t("实验文件正在更新，暂不显示选择和重训结论。请稍后刷新。", "Experiment files are changing. Refresh shortly to check selection and retraining results.")}</Notice>
        : overview.snapshot.status === "unavailable" ? <Notice>{t("目前无法读取足够的结果记录。可查看已保存文件，或稍后刷新。", "The result records are currently unavailable. View saved files or refresh later.")}</Notice>
        : null}
    </div>
    {selection && <section className={`result-selection ${selection.status === "conflict" ? "result-warning" : ""}`} aria-label={t("保存的候选方案", "Saved candidate")}>
      <div className="result-selection-heading"><GitBranch size={20} /><div><h3>{selection.status === "recorded" ? t("运行记录中的选择", "Recorded selection")
        : selection.status === "code_match" ? t("与保存代码匹配的候选", "Candidate matching saved code")
          : selection.status === "conflict" ? t("选择记录不一致", "Selection records disagree")
            : selection.status === "none" ? t("尚无选定方案", "No selected candidate") : t("无法确认选定方案", "Selection unavailable")}</h3><p>{selectionExplanation(selection, t)}</p></div></div>
      {matched && <>
        <dl className="result-facts">
          <div><dt>{t("候选方案", "Candidate")}</dt><dd>{typeof selection.candidate_step === "number" && Number.isInteger(selection.candidate_step) && selection.candidate_step >= 0 ? `#${selection.candidate_step + 1}` : selection.node_id?.slice(0, 8) || "—"}</dd></div>
          <div><dt>{metric?.name || t("记录指标", "Recorded metric")}</dt><dd>{metric ? Number(metric.value.toPrecision(6)).toString() : "—"}</dd></div>
          <div><dt>{t("指标方向", "Metric direction")}</dt><dd className="result-fact-text">{metric?.maximize === true ? t("越大越好", "Maximize") : metric?.maximize === false ? t("越小越好", "Minimize") : t("未记录", "Not recorded")}</dd></div>
        </dl>
        {averageRank !== null && <dl className="result-facts">
          <div><dt>{t("记录的平均排名", "Recorded average rank")}</dt><dd>{Number(averageRank.toPrecision(6)).toString()}</dd></div>
          {typeof selection.ranking?.candidate_count === "number" && Number.isSafeInteger(selection.ranking.candidate_count) && <div><dt>{t("参与排名的候选数", "Ranked candidates")}</dt><dd>{selection.ranking.candidate_count}</dd></div>}
        </dl>}
        {metrics.length > 0 && <dl className="result-facts" aria-label={t("全部验证指标", "All validation metrics")}>
          {metrics.map(([name, value]) => {
            const spec = Array.isArray(selection.metric_specs) ? selection.metric_specs.find(item => item?.name === name) : undefined;
            return <div key={name}><dt>{name}{spec?.maximize === true ? " ↑" : spec?.maximize === false ? " ↓" : ""}</dt><dd>{Number(value.toPrecision(6)).toString()}</dd></div>;
          })}
        </dl>}
        <div className="result-selection-footer"><p>{t("指标来自该候选的实验记录，不代表独立测试或临床验证。", "Metrics come from this candidate’s experiment record; they do not establish independent test or clinical validation.")}</p>{selection.node_id && <button className="secondary-button" disabled={!serviceReady} onClick={() => onCandidate(selection.node_id!)}><GitBranch size={14} />{t("查看候选方案", "View candidate")}</button>}</div>
      </>}
    </section>}
    {retrain && <section className={`result-retrain result-retrain-${retrain.tone}`} aria-label={t("最终重训状态", "Final retraining status")}>
      <span className="result-retrain-icon">{retrain.tone === "success" ? <Check size={18} /> : <RotateCcw size={18} />}</span><div className="result-retrain-body"><h3>{retrain.title}</h3><p>{retrain.text}</p>
        {overview && typeof overview.final_retrain.train_rows_after === "number" && Number.isFinite(overview.final_retrain.train_rows_after) && overview.final_retrain.train_rows_after >= 0 && <p className="result-retrain-count">{t("记录的重训样本数", "Recorded retraining rows")}: {overview.final_retrain.train_rows_after}</p>}
      </div>
    </section>}
    <section className="result-resources" aria-label={t("结果文件", "Result files")}>
      <div className="result-resources-heading"><h3>{t("结果文件", "Result files")}</h3><button className="text-button" onClick={onFiles}>{t("查看全部文件", "View all files")}</button></div>
      {resources.map(({ name, title, description, Icon }) => {
        const record = overview?.artifacts.find(file => file?.name === name);
        const artifact = run.artifacts?.find(file => file.name === name);
        const readable = record?.state === "available" || record?.state === "invalid" || record?.state === "too_large";
        const available = !!artifact && !!readable;
        const canPreview = available && artifact.size <= 500_000;
        const missing = record?.state === "missing";
        const stateText = record?.state === "available" && artifact ? null
          : record?.state === "too_large" ? t("文件较大，可保存后查看", "Large file; save to view")
            : record?.state === "invalid" ? t("记录格式无法识别，可查看原文件", "Unrecognized record format; inspect the original")
              : missing ? (isActive(run) ? t("尚未生成", "Not generated yet") : t("未保存此文件", "This file was not saved"))
                : record?.state === "changing" ? t("文件正在更新", "File is being updated")
                  : record ? t("文件暂时无法读取", "File currently unavailable") : t("等待读取文件记录", "Waiting for file records");
        return <div className="result-resource" key={name}>
          <Icon size={19} className="result-resource-icon" /><div className="result-resource-content"><h4>{title}</h4><p>{description}</p><div className="result-resource-meta"><span>{name}</span>{stateText && <span>{stateText}</span>}</div></div>
          <div className="result-resource-actions">
            <button className="text-button" aria-label={t(`预览${title}`, `Preview ${title.toLowerCase()}`)} disabled={!readyForFiles || !canPreview} onClick={() => { if (artifact) onPreview(artifact); }} title={available && !canPreview ? t("文件较大，请保存后查看。", "This file is large. Save it to view.") : undefined}>{t("预览", "Preview")}</button>
            <button className="secondary-button" aria-label={t(`保存 ${name}`, `Save ${name}`)} disabled={!readyForFiles || !available} onClick={() => { if (artifact) onSave(artifact); }}>{fileBusy === name ? <Spinner size={13} /> : <ArrowDownToLine size={13} />}{t("保存", "Save")}</button>
          </div>
        </div>;
      })}
    </section>
    {stable && overview.warnings.some(warning => warning !== "history_snapshot") && <p className="result-partial-note">{t("部分文件或记录不完整；概览仅展示能够确认的内容。可在全部文件中查看现有记录。", "Some files or records are incomplete. This overview shows only what can be confirmed; review the available records in Files.")}</p>}
  </section>;
}
