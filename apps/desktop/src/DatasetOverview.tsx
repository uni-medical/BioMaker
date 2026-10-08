import { AlertCircle, ChevronDown, FileText, Folder, Link, RefreshCw } from "lucide-react";
import { Spinner } from "./components";
import { basename, formatBytes, type Language, type Translate } from "./models";
import { datasetInspectionMessage, datasetWarningMessage, type DatasetInspectionState } from "./datasetInspection";

export default function DatasetOverview({ path, inspection, serviceReady, submitting, lang, t, onChoose, onOpen, onRefresh }: {
  path: string;
  inspection: DatasetInspectionState;
  serviceReady: boolean;
  submitting: boolean;
  lang: Language;
  t: Translate;
  onChoose: () => void;
  onOpen: () => void;
  onRefresh: () => void;
}) {
  const { result, error, pending } = inspection;
  const failed = error || result?.status === "error";
  const code = result?.code || inspection.errorCode || "";
  const partial = result?.status === "ok" && !result.complete;
  const warnings = [...new Set(result?.warnings || [])];
  const status = !serviceReady ? t("本地服务恢复后将检查目录。", "The folder will be checked when the local service is ready.")
    : pending ? t("正在读取目录概览…", "Inspecting directory metadata…")
      : failed ? datasetInspectionMessage(code, t)
        : partial ? t("目录可访问，概览不完整", "Folder is accessible; overview is incomplete")
          : datasetInspectionMessage("ok", t);
  return <section className={`dataset-overview ${failed ? "failed" : partial ? "partial" : ""}`}
    aria-label={t("数据集概览", "Dataset overview")} data-code={code}
    data-state={pending ? "checking" : failed ? "error" : result ? "ready" : "waiting"}>
    <div className="dataset-overview-heading">
      <div><Folder size={15} /><strong>{basename(path)}</strong></div>
      <div className="dataset-overview-actions">
        <button type="button" className="text-button" disabled={submitting} aria-label={t("重新选择数据集", "Choose dataset again")} onClick={onChoose}>{t("更换", "Change")}</button>
        <button type="button" className="text-button" disabled={submitting || pending || !serviceReady} aria-label={t("重新检查数据集", "Refresh dataset inspection")} onClick={onRefresh}>
          <RefreshCw size={12} />{t("刷新", "Refresh")}
        </button>
      </div>
    </div>
    <p className="dataset-inspection-status" role="status">{pending ? <Spinner size={13} /> : failed ? <AlertCircle size={13} /> : null}{status}</p>
    {result?.status === "ok" && <p className="dataset-overview-counts">
      {partial && result.files === 0 ? t("尚未统计到普通文件", "No regular files inspected yet")
        : <>{partial ? t("已统计 ", "Inspected ") : ""}{result.files.toLocaleString()} {t("个普通文件", "regular files")}</>}
      <span>·</span>{partial || warnings.includes("size_limit") ? t("至少 ", "at least ") : ""}{formatBytes(result.bytes)}
      {!!result.symlinks && <><span>·</span>{result.symlinks.toLocaleString()} {t("个符号链接", "symbolic links")}</>}
      {!!result.other && <><span>·</span>{result.other.toLocaleString()} {t("个其它条目", "other entries")}</>}
    </p>}
    {warnings.length > 0 && <p className="dataset-overview-warnings">{warnings.map(warning => datasetWarningMessage(warning, t)).join(t("；", "; "))}{t("。", ".")}</p>}
    <details className="dataset-overview-details">
      <summary><ChevronDown size={12} />{t("查看目录概览", "View directory overview")}</summary>
      <p className="path-text">{path}</p>
      <button type="button" className="text-button" disabled={submitting} onClick={onOpen}><Folder size={12} />{t("在访达中打开", "Open in Finder")}</button>
      {result?.status === "ok" && <>
        <div className="dataset-overview-extensions">{result.extensions.map(extension => <span key={extension.extension}>
          {extension.extension || t("无扩展名", "No extension")} <b>{extension.count.toLocaleString()}</b>
        </span>)}</div>
        {result.entries.length > 0 && <>
          <p className="dataset-overview-caption">{t("顶层条目预览（最多 10 项）", "Top-level entries (up to 10)")}</p>
          <ul className="dataset-overview-entries">{result.entries.slice(0, 10).map((entry, index) => <li key={`${index}:${entry.kind}:${entry.name}`}>
            {entry.kind === "directory" ? <Folder size={13} /> : entry.kind === "symlink" ? <Link size={13} /> : <FileText size={13} />}
            <span>{entry.name}</span><small>{entry.kind === "directory" ? t("目录", "Folder") : entry.kind === "symlink" ? t("符号链接", "Symbolic link") : entry.bytes === null ? t("其它", "Other") : formatBytes(entry.bytes)}</small>
          </li>)}</ul>
        </>}
      </>}
      {result?.checked_at && Number.isFinite(Date.parse(result.checked_at)) && <p className="dataset-overview-caption">
        {t("检查时间：", "Checked: ")}{new Date(result.checked_at).toLocaleString(lang === "zh" ? "zh-CN" : "en-US")}
      </p>}
    </details>
    <p className="dataset-overview-note">{t("仅统计本机目录元数据，不读取文件内容或调用模型。结果是当时的快照，不验证数据或标注有效性。", "Only local directory metadata is inspected; file contents are not read and no model is called. This is a point-in-time snapshot, not validation of data or labels.")}</p>
  </section>;
}
