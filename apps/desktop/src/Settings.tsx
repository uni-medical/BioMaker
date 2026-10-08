import { useEffect, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  Download,
  ExternalLink,
  FolderOpen,
  KeyRound,
  Monitor,
  Moon,
  RefreshCw,
  Settings2,
  Sun,
  Terminal,
  Trash2,
  Wrench,
} from "lucide-react";
import { Modal, Notice, SettingRow, Spinner } from "./components";
import type { ServiceState } from "./bridge";
import ServiceRecovery from "./ServiceRecovery";
import {
  errorMessage,
  request,
  serviceStatusLabel,
  runtimeOperation,
  runtimeOperationLabel,
  isRuntimeBusy,
  type EngineStatus,
  type EnvConfig,
  type Language,
  type Theme,
  type Translate,
} from "./models";

type Section = "provider" | "runtime" | "appearance";
export default function Settings({
  t,
  lang,
  setLang,
  theme,
  setTheme,
  notificationsEnabled,
  setNotificationsEnabled,
  engine,
  serviceState,
  onServiceState,
  env,
  version,
  initialSection,
  onClose,
  onSaved,
  onReturnToExperiment,
  modelOverride,
  onRefresh,
  onEngineChanged,
  toast,
}: {
  t: Translate;
  lang: Language;
  setLang: (v: Language) => void;
  theme: Theme;
  setTheme: (v: Theme) => void;
  notificationsEnabled: boolean;
  setNotificationsEnabled: (enabled: boolean) => void;
  engine: EngineStatus | null;
  serviceState: ServiceState;
  onServiceState: (state: ServiceState) => void;
  env: EnvConfig | null;
  version: string;
  initialSection?: Section;
  onClose: () => void;
  onSaved: (config: EnvConfig) => void;
  onReturnToExperiment?: () => void;
  modelOverride?: string | null;
  onRefresh: () => Promise<void>;
  onEngineChanged: (engine: EngineStatus["engine"]) => void;
  toast: (s: string) => void;
}) {
  const [section, setSection] = useState<Section>(initialSection || "provider");
  const [base, setBase] = useState(env?.values.OPENAI_BASE_URL?.value || "");
  const [key, setKey] = useState("");
  const [model, setModel] = useState(
    env?.values.BIOMAKER_MAIN_MODEL?.value || "",
  );
  const [serper, setSerper] = useState("");
  const [jina, setJina] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [pending, setPending] = useState("");
  const [error, setError] = useState("");
  const [discovery, setDiscovery] = useState("");
  const [dirty, setDirty] = useState(false);
  const [saved, setSaved] = useState(false);
  const [removeKey, setRemoveKey] = useState(false);
  const [optional, setOptional] = useState(false);
  const [confirmClose, setConfirmClose] = useState(false);
  const [confirmRepair, setConfirmRepair] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);
  const runtimeRequest = useRef(false);
  const repairCancel = useRef<HTMLButtonElement>(null);
  const repairButton = useRef<HTMLButtonElement>(null);
  const currentService = useRef(serviceState);
  currentService.current = serviceState;
  const closeState = useRef({ dirty: false, saving: false });
  const saving = pending === "save";
  const serviceReady = serviceState.status === "ready";
  const operation = runtimeOperation(engine?.engine);
  const runtimeBusy = isRuntimeBusy(engine?.engine);
  const externalBusy = !!engine?.engine.busy_external;
  const runtimeCancelled = /^(Installation|Engine check|Engine repair) cancelled\.$/.test(engine?.engine.error || "");
  const checkedAt = engine?.engine.checked_at && Number.isFinite(Date.parse(engine.engine.checked_at))
    ? new Date(engine.engine.checked_at).toLocaleString(lang === "zh" ? "zh-CN" : "en-US") : null;
  const configured = !!env?.values.OPENAI_API_KEY?.configured;
  const effectiveKey = !!key.trim() || (configured && !removeKey);
  const effectiveModel = !!model.trim() || !!modelOverride;
  const savedProviderReady =
    configured && !!(env?.values.BIOMAKER_MAIN_MODEL?.value || modelOverride);
  const changed = () => {
    closeState.current.dirty = true;
    setDirty(true);
    setSaved(false);
    setError("");
  };
  const close = () => {
    if (confirmRepair) { setConfirmRepair(false); repairButton.current?.focus(); return; }
    if (closeState.current.saving) {
      toast(t("配置正在保存，请稍候。", "Settings are being saved. Please wait."));
      return;
    }
    if (dirty) setConfirmClose(true);
    else onClose();
  };
  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      const state = closeState.current;
      if (!state.dirty && !state.saving) return;
      event.preventDefault();
      event.returnValue = "Unsaved model settings";
      window.biomaker.reportCloseBlock(state.saving ? "saving" : "unsaved");
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, []);
  useEffect(() => {
    if (env && !dirty) {
      setBase(env.values.OPENAI_BASE_URL?.value || "");
      setModel(env.values.BIOMAKER_MAIN_MODEL?.value || "");
    }
  }, [env, dirty]);
  useEffect(() => {
    if (runtimeBusy) setSection("runtime");
  }, []);
  useEffect(() => {
    if (!operation) setCancelRequested(false);
  }, [operation]);
  useEffect(() => {
    if (section !== "runtime" || runtimeBusy || !serviceReady) setConfirmRepair(false);
  }, [section, runtimeBusy, serviceReady]);
  useEffect(() => {
    if (confirmRepair) repairCancel.current?.focus();
  }, [confirmRepair]);

  async function save() {
    if (pending || !env || !serviceReady) return;
    closeState.current.saving = true;
    setPending("save");
    setConfirmClose(false);
    setError("");
    try {
      if (base.trim()) {
        const url = new URL(base);
        if (!["https:", "http:"].includes(url.protocol))
          throw new Error(
            t(
              "API 地址需要以 https:// 或 http:// 开头。",
              "Use an API address beginning with https:// or http://.",
            ),
          );
      }
      const values: Record<string, string | null> = {
        OPENAI_BASE_URL: base.trim() || null,
        BIOMAKER_MAIN_MODEL: model.trim() || null,
      };
      if (key.trim()) values.OPENAI_API_KEY = key.trim();
      else if (removeKey) values.OPENAI_API_KEY = null;
      if (serper.trim()) values.SERPER_API_KEY = serper.trim();
      if (jina.trim()) values.JINA_API_KEY = jina.trim();
      const result = await request<EnvConfig>("/api/env", "PUT", values);
      onSaved(result);
      setKey("");
      setSerper("");
      setJina("");
      setRemoveKey(false);
      closeState.current.dirty = false;
      setDirty(false);
      setSaved(true);
      toast(
        !result.values.OPENAI_API_KEY?.configured
          ? t(
              "配置已保存；添加 API Key 后才能运行实验",
              "Settings saved. Add an API key before running an experiment.",
            )
          : !(result.values.BIOMAKER_MAIN_MODEL?.value || modelOverride)
            ? t(
                "配置已保存；还需填写默认模型",
                "Settings saved. A default model is still needed.",
              )
            : t("模型配置已保存", "Model settings saved"),
      );
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      closeState.current.saving = false;
      setPending("");
    }
  }
  async function discover() {
    if (pending || !env || !serviceReady) return;
    setPending("discover");
    setError("");
    setDiscovery("");
    try {
      const result = await request<{
        status: string;
        models: string[];
        detail?: string;
      }>("/api/models/discover", "POST", {
        base_url: base.trim() || "https://api.openai.com/v1",
        ...(key.trim() ? { api_key: key.trim() } : {}),
      });
      setModels(result.models || []);
      if (result.models?.length)
        setDiscovery(
          t(
            `连接成功，发现 ${result.models.length} 个模型。`,
            `Connected. Found ${result.models.length} models.`,
          ),
        );
      else
        setError(
          result.detail ||
            t(
              "服务没有返回模型列表。你可以手动填写模型名称。",
              "The service returned no models. You can enter a model name manually.",
            ),
        );
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setPending("");
    }
  }
  async function engineAction(action: "install" | "check" | "repair" | "cancel") {
    if (!serviceReady || pending || runtimeRequest.current || externalBusy || (action !== "cancel" && runtimeBusy)) return;
    runtimeRequest.current = true;
    setPending(action);
    setError("");
    setConfirmRepair(false);
    const revision = serviceState.revision;
    const current = () => currentService.current.status === "ready" && currentService.current.revision === revision;
    try {
      const snapshot = await request<EngineStatus["engine"]>(`/api/desktop/engine/${action}`, "POST");
      if (!current()) return;
      onEngineChanged(snapshot);
      if (action === "cancel" && runtimeOperation(snapshot)) setCancelRequested(true);
      await onRefresh();
    } catch (e) {
      if (current()) {
        const message = errorMessage(e);
        setError(message.includes("The engine operation lock is unavailable.")
          ? t("本机磁盘访问异常，暂时无法检查或修复运行环境。请检查访问权限后重试。", "Local disk access prevents runtime checks or repairs. Check access permissions, then try again.")
          : message);
      }
    } finally {
      runtimeRequest.current = false;
      setPending("");
    }
  }
  async function openPath(path?: string) {
    if (!path) return;
    try {
      await window.biomaker.openPath(path);
    } catch (e) {
      setError(errorMessage(e));
    }
  }
  return (
    <Modal
      title={t("设置", "Settings")}
      onClose={close}
      t={t}
      className="settings-modal"
    >
      <div className="settings-layout">
        <nav
          className="settings-nav"
          aria-label={t("设置分类", "Settings sections")}
        >
          <button
            className={section === "provider" ? "selected" : ""}
            disabled={saving}
            onClick={() => setSection("provider")}
          >
            <KeyRound size={16} />
            {t("模型服务", "Model provider")}
          </button>
          <button
            className={section === "runtime" ? "selected" : ""}
            disabled={saving}
            onClick={() => setSection("runtime")}
          >
            <Terminal size={16} />
            {t("运行环境", "Runtime")}
          </button>
          <button
            className={section === "appearance" ? "selected" : ""}
            disabled={saving}
            onClick={() => setSection("appearance")}
          >
            <Settings2 size={16} />
            {t("通用", "General")}
          </button>
          <span className="settings-version">BioMaker {version}</span>
        </nav>
        <div className="settings-content">
          {!serviceReady && <Notice danger={serviceState.status === "failed"}>
            {serviceStatusLabel(serviceState, t)}
            {section !== "runtime" && <ServiceRecovery state={serviceState} onState={onServiceState} t={t} toast={toast} />}
          </Notice>}
          {section === "provider" && (
            <>
              {!env && serviceReady && (
                <Notice>
                  {t("正在读取已保存的配置…", "Loading saved settings…")}
                </Notice>
              )}
              <h3>{t("连接你的模型服务", "Connect your model provider")}</h3>
              <p className="muted leading">
                {t(
                  "使用兼容 OpenAI 的 API。模型用于规划实验、编写代码和分析结果。",
                  "Use an OpenAI-compatible API for experiment planning, code generation, and result analysis.",
                )}
              </p>
              <label className="field">
                <span>{t("API 地址", "API base URL")}</span>
                <input
                  disabled={!env || saving}
                  type="url"
                  value={base}
                  onChange={(e) => {
                    setBase(e.target.value);
                    changed();
                  }}
                  placeholder="https://api.openai.com/v1"
                  autoComplete="off"
                  spellCheck={false}
                />
                <small>
                  {t(
                    "留空时使用 OpenAI。其它服务请填写完整的 API 基础地址。",
                    "Leave blank for OpenAI. For other providers, enter the full API base URL.",
                  )}
                </small>
              </label>
              <label className="field">
                <span>
                  API Key{" "}
                  {configured && !removeKey && (
                    <span className="configured">
                      <Check size={12} />
                      {t("已配置", "Configured")}
                    </span>
                  )}
                </span>
                <div className="input-action">
                  <input
                    disabled={!env || saving}
                    type="password"
                    value={key}
                    onChange={(e) => {
                      setKey(e.target.value);
                      setRemoveKey(false);
                      changed();
                    }}
                    placeholder={
                      configured && !removeKey
                        ? t(
                            "留空保留现有密钥",
                            "Leave blank to keep existing key",
                          )
                        : t(
                            "输入服务提供商的 API Key",
                            "Enter your provider API key",
                          )
                    }
                    autoComplete="new-password"
                    spellCheck={false}
                  />
                  {configured && (
                    <button
                      type="button"
                      className={`icon-button ${removeKey ? "danger-text" : ""}`}
                      disabled={saving}
                      aria-label={t("移除已保存的密钥", "Remove saved key")}
                      title={t("移除已保存的密钥", "Remove saved key")}
                      onClick={() => {
                        setRemoveKey(!removeKey);
                        changed();
                      }}
                    >
                      <Trash2 size={16} />
                    </button>
                  )}
                </div>
                {removeKey && (
                  <small className="danger-text">
                    {t(
                      "保存后将移除密钥。再次点击垃圾桶可撤销。",
                      "Saving will remove the key. Click the trash icon again to undo.",
                    )}
                  </small>
                )}
              </label>
              <div className="model-label">
                <label htmlFor="provider-model">
                  {t("默认模型", "Default model")}
                </label>
                <button
                  className="text-button"
                  onClick={discover}
                  disabled={!!pending || !serviceReady || !(key.trim() || configured)}
                >
                  {pending === "discover" ? (
                    <Spinner size={13} />
                  ) : (
                    <RefreshCw size={13} />
                  )}
                  {t("连接并获取模型", "Connect & find models")}
                </button>
              </div>
              <input
                disabled={!env || saving}
                id="provider-model"
                value={model}
                onChange={(e) => {
                  setModel(e.target.value);
                  changed();
                }}
                placeholder={t(
                  "输入模型 ID，或从服务获取",
                  "Enter a model ID, or find available models",
                )}
                list="provider-models"
                autoComplete="off"
                spellCheck={false}
              />
              <datalist id="provider-models">
                {models.map((m) => (
                  <option key={m} value={m} />
                ))}
              </datalist>
              {models.length > 0 && (
                <select
                  className="model-select"
                  disabled={saving}
                  aria-label={t("选择可用模型", "Choose available model")}
                  value={models.includes(model) ? model : ""}
                  onChange={(e) => {
                    if (e.target.value) {
                      setModel(e.target.value);
                      changed();
                    }
                  }}
                >
                  <option value="">
                    {t("从可用模型中选择…", "Choose an available model…")}
                  </option>
                  {models.map((m) => (
                    <option key={m}>{m}</option>
                  ))}
                </select>
              )}
              {env && (
                <div
                  className={`provider-readiness ${effectiveKey && effectiveModel ? "complete" : ""}`}
                  role="status"
                >
                  {effectiveKey && effectiveModel ? (
                    <Check size={14} />
                  ) : (
                    <KeyRound size={14} />
                  )}
                  <span>
                    {!effectiveKey
                      ? t(
                          "添加 API Key 后，再填写或获取模型名称。",
                          "Add an API key, then enter or discover a model name.",
                        )
                      : !effectiveModel
                        ? t(
                            "还差一步：填写默认模型，或点击“连接并获取模型”选择。",
                            "One more step: enter a default model, or use “Connect & find models”.",
                          )
                        : dirty
                          ? t(
                              "配置已填写完整，保存后继续。",
                              "The required settings are filled in. Save to continue.",
                            )
                          : t(
                              "模型配置完整，可以返回实验。",
                              "Model settings are complete. You can return to your experiment.",
                            )}
                  </span>
                </div>
              )}
              {discovery && (
                <p className="success-text">
                  <Check size={14} />
                  {discovery}
                </p>
              )}
              <button
                className="disclosure optional-toggle"
                onClick={() => setOptional(!optional)}
                aria-expanded={optional}
              >
                <ChevronDown
                  size={15}
                  className={optional ? "" : "rotate-left"}
                />
                {t("可选的搜索服务", "Optional search services")}
              </button>
              {optional && (
                <div className="optional-fields">
                  <label className="field">
                    <span>
                      Serper API Key{" "}
                      {env?.values.SERPER_API_KEY?.configured && (
                        <Check size={13} />
                      )}
                    </span>
                    <input
                      disabled={!env || saving}
                      type="password"
                      value={serper}
                      onChange={(e) => {
                        setSerper(e.target.value);
                        changed();
                      }}
                      placeholder={t(
                        "留空保留现有配置",
                        "Leave blank to keep existing settings",
                      )}
                      autoComplete="new-password"
                    />
                  </label>
                  <label className="field">
                    <span>
                      Jina API Key{" "}
                      {env?.values.JINA_API_KEY?.configured && (
                        <Check size={13} />
                      )}
                    </span>
                    <input
                      disabled={!env || saving}
                      type="password"
                      value={jina}
                      onChange={(e) => {
                        setJina(e.target.value);
                        changed();
                      }}
                      placeholder={t(
                        "留空保留现有配置",
                        "Leave blank to keep existing settings",
                      )}
                      autoComplete="new-password"
                    />
                  </label>
                </div>
              )}
              <p className="privacy-note">
                {t(
                  "密钥保存在此电脑的应用配置中，不会显示在实验记录里。所选模型服务会收到任务内容和运行中读取的相关数据。",
                  "Keys are stored in the app configuration on this computer and kept out of run history. Your model provider receives task content and relevant data read during execution.",
                )}
              </p>
              {error && <Notice danger>{error}</Notice>}
              <div className="settings-footer">
                <span className="muted">
                  {saving
                    ? t("正在保存…", "Saving…")
                    : saved
                    ? t("所有更改已保存", "All changes saved")
                    : dirty
                      ? t("有未保存的更改", "Unsaved changes")
                      : ""}
                </span>
                {onReturnToExperiment && !dirty && savedProviderReady && (
                  <button
                    className="secondary-button"
                    onClick={onReturnToExperiment}
                  >
                    {t("返回实验", "Return to experiment")}
                  </button>
                )}
                <button
                  className="primary-button"
                  disabled={!!pending || !dirty || !env || !serviceReady}
                  onClick={save}
                >
                  {pending === "save" ? <Spinner /> : <Check size={15} />}
                  {t("保存配置", "Save settings")}
                </button>
              </div>
            </>
          )}
          {section === "runtime" && (
            <>
              <h3>{t("本地运行环境", "Local runtime")}</h3>
              <p className="muted leading">
                {t(
                  "BioMaker 使用独立的 Python 环境执行实验。首次安装需要联网下载依赖。",
                  "BioMaker runs experiments in an isolated Python environment. The first installation downloads dependencies from the internet.",
                )}
              </p>
              <div className="runtime-state">
                <span
                  className={`runtime-icon ${engine?.engine.health === "healthy" && !runtimeBusy ? "ready" : ""}`}
                >
                  {!serviceReady || !engine ? (
                    serviceState.status === "failed" ? <Terminal size={23} /> : <Spinner size={23} />
                  ) : runtimeBusy ? (
                    <Spinner size={23} />
                  ) : engine?.engine.health === "healthy" ? (
                    <Check size={23} />
                  ) : engine?.engine.ready ? (
                    <Terminal size={23} />
                  ) : (
                    <Download size={23} />
                  )}
                </span>
                <div>
                  <strong>
                    {!serviceReady
                      ? serviceStatusLabel(serviceState, t)
                      : !engine
                        ? t("正在读取运行环境…", "Loading runtime status…")
                        : runtimeBusy
                      ? cancelRequested && !externalBusy ? t("正在取消操作…", "Cancelling operation…") : runtimeOperationLabel(engine.engine, t)
                      : runtimeCancelled
                        ? t("操作已取消", "Operation cancelled")
                        : engine?.engine.maintenance_unavailable
                          ? t("运行环境暂时无法维护", "Runtime maintenance is unavailable")
                        : engine?.engine.health === "failed"
                          ? t("运行环境需要修复", "Runtime needs attention")
                      : engine?.engine.ready
                        ? engine.engine.health === "healthy"
                          ? t("运行环境检查通过", "Runtime check passed")
                          : t("运行环境已安装", "Runtime installed")
                        : t(
                            "安装实验所需环境",
                            "Install the experiment runtime",
                          )}
                  </strong>
                  <p>
                    {!serviceReady || !engine
                      ? t("连接就绪后将更新运行环境状态。", "Runtime status will update when the service is ready.")
                      : externalBusy
                        ? t("另一进程正在维护运行环境，请稍后重试。可以刷新以查看最新状态。", "Another process is maintaining the runtime. Please try again later, or refresh to see the latest status.")
                      : runtimeBusy
                        ? operation === "check"
                          ? t("正在检查依赖并执行固定的 CPU 自检。完成后会显示结果。", "Checking dependencies and running a fixed CPU self-test. Results appear when it finishes.")
                          : t("正在下载和验证依赖，进度会显示在下方。", "Downloading and checking dependencies. Progress appears below.")
                      : runtimeCancelled
                        ? t("操作已停止。可以重新检查运行环境，或修复后再试。", "The operation stopped. Check the runtime again, or repair it and retry.")
                      : engine?.engine.maintenance_unavailable
                        ? t("无法准备运行环境维护。请检查本机磁盘访问权限后重试。", "Runtime maintenance could not be prepared. Check local disk access, then try again.")
                      : engine?.engine.health === "failed"
                        ? t("运行环境检查或维护未通过。查看下方详细信息后重试或修复。", "A runtime check or maintenance operation failed. Review the details below, then retry or repair the runtime.")
                      : engine?.engine.ready
                        ? engine.engine.health === "healthy"
                          ? t("依赖和固定 CPU 自检已通过。实验会按任务与预算单独运行。", "Dependencies and the fixed CPU self-test passed. Experiments run separately with their own tasks and budgets.")
                          : t("已检测到此前安装的环境。可运行检查，确认依赖与固定 CPU 自检是否通过。", "A previously installed runtime was found. Run a check to verify dependencies and a fixed CPU self-test.")
                      : t(
                          "下载通常需要几分钟，进度会显示在下方。",
                          "The download may take several minutes. Progress appears below.",
                        )}
                  </p>
                </div>
              </div>
              {checkedAt && <p className="runtime-checked-at">{t("最近检查或维护：", "Latest check or maintenance: ")}<time dateTime={engine?.engine.checked_at || undefined}>{checkedAt}</time></p>}
              <div className="runtime-actions">
                {externalBusy ? <>
                  <button className="secondary-button" disabled><Check size={14} />{t("检查运行环境", "Check runtime")}</button>
                  <button className="secondary-button" disabled><Wrench size={14} />{t("修复运行环境", "Repair runtime")}</button>
                </> : runtimeBusy ? (
                  <button
                    className="secondary-button"
                    disabled={!!pending || !serviceReady || cancelRequested}
                    onClick={() => engineAction("cancel")}
                  >
                    {pending === "cancel" && <Spinner />}
                    {operation === "check" ? t("取消检查", "Cancel check") : operation === "repair" ? t("取消修复", "Cancel repair") : t("取消安装", "Cancel installation")}
                  </button>
                ) : !engine?.engine.ready && engine?.engine.health !== "failed" && !runtimeCancelled ? (
                  <button
                    className="primary-button"
                    disabled={!!pending || !serviceReady || !engine}
                    onClick={() => engineAction("install")}
                  >
                    {pending === "install" ? (
                      <Spinner />
                    ) : (
                      <Download size={15} />
                    )}
                    {!serviceReady || !engine
                      ? t("等待本地服务", "Waiting for the local service")
                      : engine?.engine.error
                        ? t("重新安装", "Retry installation")
                        : t("安装运行环境", "Install runtime")}
                  </button>
                ) : null}
                {!runtimeBusy && !!engine && (engine.engine.ready || engine.engine.health === "failed" || runtimeCancelled) && <>
                  <button className="secondary-button" disabled={!!pending || !serviceReady} onClick={() => void engineAction("check")}>
                    {pending === "check" ? <Spinner size={14} /> : <Check size={14} />}
                    {t("检查运行环境", "Check runtime")}
                  </button>
                  <button ref={repairButton} className="secondary-button" disabled={!!pending || !serviceReady} onClick={() => { setConfirmRepair(true); setError(""); }}>
                    {pending === "repair" ? <Spinner size={14} /> : <Wrench size={14} />}
                    {t("修复运行环境", "Repair runtime")}
                  </button>
                </>}
                <button
                  className="secondary-button"
                  disabled={!serviceReady}
                  onClick={() =>
                    onRefresh().catch((e) => setError(errorMessage(e)))
                  }
                >
                  <RefreshCw size={14} />
                  {t("刷新", "Refresh")}
                </button>
              </div>
              {confirmRepair && <section className="runtime-repair-confirmation" role="group" aria-label={t("修复运行环境？", "Repair the runtime?")}>
                <h4>{t("修复运行环境？", "Repair the runtime?")}</h4>
                <p>{t("将重新下载并安装实验依赖，然后执行固定的 CPU 自检，可能需要几分钟。已有实验、日志、数据和模型服务设置会保留。", "Dependencies will be downloaded and installed again, followed by a fixed CPU self-test. This may take several minutes. Existing experiments, logs, data, and model provider settings are kept.")}</p>
                <div className="runtime-confirm-actions">
                  <button ref={repairCancel} className="secondary-button" onClick={() => { setConfirmRepair(false); repairButton.current?.focus(); }}>{t("取消返回", "Go back")}</button>
                  <button className="primary-button" disabled={!!pending || !serviceReady || runtimeBusy} onClick={() => void engineAction("repair")}>{t("开始修复", "Start repair")}</button>
                </div>
              </section>}
              <section className="runtime-diagnostics" aria-label={t("本地服务诊断", "Local service diagnostics")}>
                <h4>{t("本地服务诊断", "Local service diagnostics")}</h4>
                <ServiceRecovery state={serviceState} onState={onServiceState} t={t} toast={toast} />
                {serviceState.status === "failed" && serviceState.message && <details className="service-local-details"><summary>{t("本地错误详情", "Local error details")}</summary><p>{serviceState.message}</p></details>}
              </section>
              {((engine?.engine.error && !externalBusy) || error) && (
                <Notice danger={!!error || !runtimeCancelled}>{error || (runtimeCancelled ? t("运行环境操作已取消，可重新检查或修复。", "The runtime operation was cancelled. You can check or repair it again.") : engine?.engine.maintenance_unavailable ? t("本机磁盘访问异常，暂时无法检查或修复运行环境。请检查访问权限后重试。", "Local disk access prevents runtime checks or repairs. Check access permissions, then try again.") : engine?.engine.error)}</Notice>
              )}
              {engine?.engine.log && (
                <details
                  className="install-log"
                  open={runtimeBusy || !!engine.engine.error}
                >
                  <summary>{t("运行环境日志", "Runtime log")}</summary>
                  <pre tabIndex={0}>{engine.engine.log}</pre>
                </details>
              )}
              <div className="runtime-paths">
                <SettingRow
                  title={t("实验工作区", "Experiment workspace")}
                  detail={engine?.paths.workspace || "—"}
                >
                  <button
                    className="icon-button"
                    title={t("在访达中打开", "Open in Finder")}
                    aria-label={t(
                      "打开实验工作区",
                      "Open experiment workspace",
                    )}
                    onClick={() => openPath(engine?.paths.workspace)}
                    disabled={!engine?.paths.workspace}
                  >
                    <FolderOpen size={17} />
                  </button>
                </SettingRow>
                <SettingRow
                  title={t("实验日志", "Experiment logs")}
                  detail={engine?.paths.logs || "—"}
                >
                  <button
                    className="icon-button"
                    title={t("在访达中打开", "Open in Finder")}
                    aria-label={t(
                      "打开实验日志文件夹",
                      "Open experiment logs folder",
                    )}
                    onClick={() => openPath(engine?.paths.logs)}
                    disabled={!engine?.paths.logs}
                  >
                    <FolderOpen size={17} />
                  </button>
                </SettingRow>
              </div>
              <p className="privacy-note">
                {t(
                  "实验会运行自动生成的 Python 代码。请使用可信的任务和专用数据目录；执行环境并非安全沙箱。",
                  "Experiments execute generated Python code. Use trusted tasks and a dedicated data directory; the runtime is not a security sandbox.",
                )}
              </p>
            </>
          )}
          {section === "appearance" && (
            <>
              <h3>{t("让 BioMaker 适合你", "Make BioMaker your own")}</h3>
              <SettingRow
                title={t("外观", "Appearance")}
                detail={t(
                  "跟随系统，或选择固定主题。",
                  "Follow your system, or choose a theme.",
                )}
              >
                <div className="segmented" aria-label={t("外观", "Appearance")}>
                  {(
                    [
                      {
                        value: "system",
                        Icon: Monitor,
                        zh: "自动",
                        en: "Auto",
                      },
                      { value: "light", Icon: Sun, zh: "浅色", en: "Light" },
                      { value: "dark", Icon: Moon, zh: "深色", en: "Dark" },
                    ] as const
                  ).map(({ value, Icon, zh, en }) => (
                    <button
                      key={value}
                      className={theme === value ? "selected" : ""}
                      aria-pressed={theme === value}
                      onClick={() => setTheme(value)}
                    >
                      <Icon size={15} />
                      {t(zh, en)}
                    </button>
                  ))}
                </div>
              </SettingRow>
              <SettingRow
                title={t("语言", "Language")}
                detail={t("界面语言会立即更新。", "Changes apply immediately.")}
              >
                <select
                  aria-label={t("界面语言", "Interface language")}
                  value={lang}
                  onChange={(e) => setLang(e.target.value as Language)}
                >
                  <option value="zh">简体中文</option>
                  <option value="en">English</option>
                </select>
              </SettingRow>
              <div className="setting-row notification-setting">
                <div>
                  <label className="setting-title" htmlFor="run-notifications">
                    {t(
                      "实验完成或失败时通知我",
                      "Notify when experiments finish or fail",
                    )}
                  </label>
                  <p id="run-notifications-help">
                    {t(
                      "应用在后台运行时提醒；完全退出后不会通知。通知还需获得系统允许。",
                      "Notify while BioMaker is running in the background; unavailable after quitting the app. Notifications also require system permission.",
                    )}
                  </p>
                </div>
                <input
                  id="run-notifications"
                  type="checkbox"
                  checked={notificationsEnabled}
                  onChange={(e) => setNotificationsEnabled(e.target.checked)}
                  aria-describedby="run-notifications-help"
                />
              </div>
              <div className="keyboard-help">
                <h4>{t("键盘快捷键", "Keyboard shortcuts")}</h4>
                {[
                  [t("新建实验", "New experiment"), "⌘ N"],
                  [t("搜索实验", "Search experiments"), "⌘ K"],
                  [t("设置", "Settings"), "⌘ ,"],
                  [t("提交实验", "Submit experiment"), "⌘ ↵"],
                ].map(([label, keys]) => (
                  <div key={keys}>
                    <span>{label}</span>
                    <kbd>{keys}</kbd>
                  </div>
                ))}
              </div>
              <div className="about-row">
                <span>BioMaker</span>
                <span>{version}</span>
              </div>
              <p className="muted leading">
                {t(
                  "从一个问题，到可追溯的实验。",
                  "From a question to a traceable experiment.",
                )}
              </p>
            </>
          )}
        </div>
      </div>
      {confirmClose && (
        <div className="unsaved-bar" role="alert">
          <span>
            {t("模型配置尚未保存。", "Model settings have not been saved.")}
          </span>
          <button
            className="text-button"
            onClick={() => setConfirmClose(false)}
          >
            {t("继续编辑", "Keep editing")}
          </button>
          <button className="secondary-button" disabled={saving} onClick={onClose}>
            {t("放弃更改", "Discard changes")}
          </button>
        </div>
      )}
    </Modal>
  );
}
