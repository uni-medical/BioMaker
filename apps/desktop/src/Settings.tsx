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
} from "lucide-react";
import { Modal, Notice, SettingRow, Spinner } from "./components";
import {
  errorMessage,
  request,
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
  env,
  version,
  initialSection,
  onClose,
  onSaved,
  onReturnToExperiment,
  modelOverride,
  onRefresh,
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
  env: EnvConfig | null;
  version: string;
  initialSection?: Section;
  onClose: () => void;
  onSaved: (config: EnvConfig) => void;
  onReturnToExperiment?: () => void;
  modelOverride?: string | null;
  onRefresh: () => Promise<void>;
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
  const closeState = useRef({ dirty: false, saving: false });
  const saving = pending === "save";
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
    if (engine?.engine.installing) setSection("runtime");
  }, []);

  async function save() {
    if (pending || !env) return;
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
    if (pending || !env) return;
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
  async function engineAction(action: "install" | "cancel") {
    setPending(action);
    setError("");
    try {
      await request(`/api/desktop/engine/${action}`, "POST");
      await onRefresh();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
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
          {section === "provider" && (
            <>
              {!env && (
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
                  disabled={!!pending || !(key.trim() || configured)}
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
                  disabled={!!pending || !dirty || !env}
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
                  className={`runtime-icon ${engine?.engine.ready ? "ready" : ""}`}
                >
                  {engine?.engine.installing ? (
                    <Spinner size={23} />
                  ) : engine?.engine.ready ? (
                    <Check size={23} />
                  ) : (
                    <Download size={23} />
                  )}
                </span>
                <div>
                  <strong>
                    {engine?.engine.installing
                      ? t("正在安装运行环境", "Installing runtime")
                      : engine?.engine.ready
                        ? t("运行环境已就绪", "Runtime ready")
                        : t(
                            "安装实验所需环境",
                            "Install the experiment runtime",
                          )}
                  </strong>
                  <p>
                    {engine?.engine.ready
                      ? t(
                          "可以在此电脑创建和运行实验。",
                          "Ready to create and run experiments on this computer.",
                        )
                      : t(
                          "下载通常需要几分钟，进度会显示在下方。",
                          "The download may take several minutes. Progress appears below.",
                        )}
                  </p>
                </div>
              </div>
              <div className="runtime-actions">
                {engine?.engine.installing ? (
                  <button
                    className="secondary-button"
                    disabled={!!pending}
                    onClick={() => engineAction("cancel")}
                  >
                    {pending === "cancel" && <Spinner />}
                    {t("取消安装", "Cancel installation")}
                  </button>
                ) : (
                  <button
                    className="primary-button"
                    disabled={!!pending || !!engine?.engine.ready}
                    onClick={() => engineAction("install")}
                  >
                    {pending === "install" ? (
                      <Spinner />
                    ) : (
                      <Download size={15} />
                    )}
                    {engine?.engine.ready
                      ? t("已安装", "Installed")
                      : engine?.engine.error
                        ? t("重新安装", "Retry installation")
                        : t("安装运行环境", "Install runtime")}
                  </button>
                )}
                <button
                  className="secondary-button"
                  onClick={() =>
                    onRefresh().catch((e) => setError(errorMessage(e)))
                  }
                >
                  <RefreshCw size={14} />
                  {t("刷新", "Refresh")}
                </button>
              </div>
              {(engine?.engine.error || error) && (
                <Notice danger>{error || engine?.engine.error}</Notice>
              )}
              {engine?.engine.log && (
                <details
                  className="install-log"
                  open={engine.engine.installing || !!engine.engine.error}
                >
                  <summary>{t("安装日志", "Installation log")}</summary>
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
