import { useEffect, useRef, useState } from "react";
import { Download, RefreshCw } from "lucide-react";
import type { ServiceState } from "./bridge";
import { Spinner } from "./components";
import { errorMessage, type Translate } from "./models";

export default function ServiceRecovery({ state, onState, t, toast }: {
  state: ServiceState;
  onState: (state: ServiceState) => void;
  t: Translate;
  toast: (message: string) => void;
}) {
  const [retrying, setRetrying] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<{ action: "retry" | "export"; detail: string } | null>(null);
  const pending = useRef({ retry: false, export: false });
  const mounted = useRef(true);
  const currentState = useRef(state);
  currentState.current = state;
  const failureAdvice = state.code === "SERVICE_CONSTRUCTION_FAILED"
    ? t("本地服务所需资源未能准备好。请先重试；如果持续失败，请重新安装 BioMaker。", "Required local service resources could not be prepared. Retry first; if this keeps happening, reinstall BioMaker.")
    : state.code === "SERVICE_STOP_FAILED"
      ? t("旧服务尚未正常结束。请正常退出 BioMaker 后重新打开，再尝试恢复。", "The previous service has not stopped cleanly. Quit BioMaker normally, reopen it, then try to recover.")
      : state.code === "SERVICE_START_FAILED"
        ? t("未能启动或连接本地服务。请重试，也可以导出诊断以便检查。", "The local service could not start or connect. Retry, or export diagnostics to help investigate.")
        : t("本地服务暂时不可用。请重试；如果仍然失败，可以导出诊断以便检查。", "The local service is unavailable. Retry; if it still fails, export diagnostics to help investigate.");
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  async function retry() {
    if (pending.current.retry || currentState.current.status !== "failed") return;
    pending.current.retry = true;
    setRetrying(true);
    setError(null);
    try {
      // Native lifecycle pushes and this result share the App revision guard.
      // Recovery never navigates, reloads the renderer, or touches its draft.
      onState(await window.biomaker.retryService());
    } catch (failure) {
      if (mounted.current) setError({ action: "retry", detail: errorMessage(failure) });
      else toast(t("未能重试本地服务，请再试一次。", "Could not retry the local service. Please try again."));
    } finally {
      pending.current.retry = false;
      if (mounted.current) setRetrying(false);
    }
  }

  async function exportDiagnostics() {
    if (pending.current.export) return;
    pending.current.export = true;
    setExporting(true);
    setError(null);
    try {
      const path = await window.biomaker.exportDiagnostics();
      if (path) toast(t("诊断文件已保存到本机，可自行分享。", "Diagnostics saved on this computer. You can share the file yourself."));
    } catch (failure) {
      if (mounted.current) setError({ action: "export", detail: errorMessage(failure) });
      else toast(t("未能导出诊断，请再试一次。", "Could not export diagnostics. Please try again."));
    } finally {
      pending.current.export = false;
      if (mounted.current) setExporting(false);
    }
  }

  return <div className="service-recovery">
    {state.status === "failed" && <p className="service-recovery-advice">{failureAdvice}</p>}
    <div className="service-recovery-actions">
      {state.status !== "ready" && <button className="secondary-button" disabled={state.status !== "failed" || retrying} onClick={() => void retry()}>
        {retrying ? <Spinner size={14} /> : <RefreshCw size={14} />}
        {t("重试本地服务", "Retry local service")}
      </button>}
      <button className="secondary-button" disabled={exporting} onClick={() => void exportDiagnostics()}>
        {exporting ? <Spinner size={14} /> : <Download size={14} />}
        {t("导出诊断", "Export diagnostics")}
      </button>
    </div>
    <p className="service-diagnostics-note">{t(
      "诊断只包含应用与服务状态，不含原始日志、任务、密钥或数据。文件保存到本机，可自行分享，不会自动上传。",
      "Diagnostics contain only app and service status, without raw logs, tasks, keys, or data. The file stays on this computer for you to share; nothing is uploaded automatically.",
    )}</p>
    {error && <div className="service-recovery-error" role="alert">
      <p>{error.action === "retry"
        ? t("未能重试本地服务，请再试一次或导出诊断。", "Could not retry the local service. Try again or export diagnostics.")
        : t("未能导出诊断，请再试一次。", "Could not export diagnostics. Please try again.")}</p>
      <details><summary>{t("本地错误详情", "Local error details")}</summary><p>{error.detail}</p></details>
    </div>}
  </div>;
}
