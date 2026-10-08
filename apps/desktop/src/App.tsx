import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Box,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Command,
  Cpu,
  FileText,
  Folder,
  FolderOpen,
  History,
  Import,
  Layers3,
  KeyRound,
  MoreHorizontal,
  PanelLeft,
  PanelRight,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Search,
  Settings2,
  SlidersHorizontal,
  Sparkles,
  SquarePen,
  X,
} from "lucide-react";
import {
  CheckItem,
  IconButton,
  Mark,
  Modal,
  Notice,
  Spinner,
} from "./components";
import Settings from "./Settings";
import RunView, { ResourcesView } from "./RunView";
import type { ActivityWindow, LogWindow } from "./runStreams";
import HistoryImport from "./HistoryImport";
import ServiceRecovery from "./ServiceRecovery";
import DatasetOverview from "./DatasetOverview";
import { datasetAdmissionCode, datasetInspectionMessage, useDatasetInspection } from "./datasetInspection";
import { reuseDraft, type Draft } from "./experimentDraft";
import DraftLibrary, { draftEntryTitle } from "./DraftLibrary";
import { useDraftWorkspace } from "./useDraftWorkspace";
import { useToast } from "./useToast";
import { decorateRun, defaultMetadata, isCurrentRun, mergeMetadata, type MetadataMap, type RunMetadata } from "./runMetadata";
import type { ServiceState } from "./bridge";
import {
  basename,
  dateText,
  defaultOptions,
  errorMessage,
  isActive,
  persist,
  readStored,
  request,
  runTitle,
  isRuntimeBusy,
  runtimeOperationLabel,
  serviceStatusLabel,
  statusLabel,
  type EngineStatus,
  type EnvConfig,
  type Health,
  type Language,
  type Options,
  type QueueState,
  type Resources,
  type Revisions,
  type Run,
  type Template,
  type Theme,
  type Translate,
} from "./models";

const exampleTasks = [
  {
    id: "classification",
    zh: "图像分类",
    en: "Image classification",
    hintZh: "从数据检查到模型评估",
    hintEn: "Inspect, train, and evaluate",
    Icon: Layers3,
    promptZh:
      "为这个数据集建立图像分类模型。先检查文件结构、标签和已有的数据划分，明确训练集与验证集。以验证集上的宏平均 F1 为主要指标，建立可靠基线，再在预算内改进模型。保留每轮实验代码、配置和指标；不要使用测试集选择模型。最终输出最佳模型、可复现实验说明和结果报告。",
    promptEn:
      "Build an image classification model for this dataset. Inspect the files, labels, and existing data splits first. Establish a reliable baseline using macro F1 on the validation split as the primary metric, then improve it within the budget. Keep the code, configuration, and metrics for each experiment. Do not use the test set for model selection. Deliver the best model, reproducibility instructions, and a results report.",
  },
  {
    id: "segmentation",
    zh: "医学图像分割",
    en: "Medical segmentation",
    hintZh: "明确标签与分割评估指标",
    hintEn: "Define labels and metrics",
    Icon: Box,
    promptZh:
      "为这个医学影像数据集建立分割模型。先核对图像、标注配对、类别定义、空间分辨率与已有划分，避免患者跨训练集和验证集泄漏。以验证集 Dice 为主要指标，建立基线并逐步改进。记录资源使用与每轮结果，不接触保留测试集。输出模型、推理脚本和可复现报告。",
    promptEn:
      "Build a segmentation model for this medical imaging dataset. Verify image–label pairing, class definitions, spatial resolution, and existing splits. Avoid patient-level leakage across training and validation. Establish and improve a baseline using validation Dice as the primary metric. Record resource use and each experiment result. Keep the held-out test set untouched. Deliver the model, inference script, and a reproducible report.",
  },
  {
    id: "baseline",
    zh: "快速基线",
    en: "Quick baseline",
    hintZh: "3 步，探索预算 30 分钟",
    hintEn: "3 steps, 30 min exploration",
    Icon: Cpu,
    promptZh:
      "为这个数据集建立一个可复现的快速基线模型，探索阶段预算为 30 分钟（不含数据准备和报告，单次请求可能延后结束）。先检查数据结构、标签和已有的训练与验证划分，选择符合任务类型的评估指标。优先使用适合当前电脑资源的简单模型和较短的训练流程，最多探索 3 步，不进行最终重训。不使用保留测试集调参。保留实验代码、配置与真实验证指标，说明本次预算内尚未完成的部分。",
    promptEn:
      "Build a reproducible baseline model for this dataset with a 30-minute exploration budget. Data preparation and reporting are outside this budget, and an in-flight request may finish after the limit. Inspect the data structure, labels, and existing training and validation splits, then choose a metric appropriate for the task. Prefer a simple model and short training schedule suitable for this computer. Use at most 3 exploration steps and skip final retraining. Do not tune on the held-out test set. Keep the code, configuration, and actual validation metrics, and report any work not completed within the budget.",
  },
];

export default function App() {
  const [lang, setLang] = useState<Language>(() =>
    readStored("biomaker.language", "zh"),
  );
  const [theme, setTheme] = useState<Theme>(() =>
    readStored("biomaker.theme", "system"),
  );
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [notificationRunToAcknowledge, setNotificationRunToAcknowledge] =
    useState<string | null>(null);
  const t: Translate = useCallback(
    (zh, en) => (lang === "zh" ? zh : en),
    [lang],
  );
  const drafts = useDraftWorkspace(t);
  const draft = drafts.draft;
  const [rawRuns, setRuns] = useState<Run[]>([]);
  const [metadata, setMetadata] = useState<MetadataMap>({});
  const [metadataLoaded, setMetadataLoaded] = useState(false);
  const [metadataError, setMetadataError] = useState("");
  const mergeRunMetadata = useCallback((row: RunMetadata) => setMetadata(current => mergeMetadata(current, [row])), []);
  const runs = useMemo(() => rawRuns.map(run => decorateRun(run, metadata)), [rawRuns, metadata]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<Run | null>(null);
  const [streamRunId, setStreamRunId] = useState<string | null>(null);
  const [activity, setActivity] = useState<ActivityWindow | null>(null);
  const [log, setLog] = useState<LogWindow | null>(null);
  const [activityError, setActivityError] = useState("");
  const [logError, setLogError] = useState("");
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [engine, setEngine] = useState<EngineStatus | null>(null);
  const engineEpoch = useRef(0);
  const applyEngineSnapshot = useCallback((snapshot: EngineStatus["engine"]) => {
    engineEpoch.current += 1;
    setEngine(current => current ? { ...current, engine: snapshot } : current);
  }, []);
  const [env, setEnv] = useState<EnvConfig | null>(null);
  const environmentEpoch = useRef(0);
  const [health, setHealth] = useState<Health | null>(null);
  const [resources, setResources] = useState<Resources | null>(null);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [version, setVersion] = useState("…");
  const [loading, setLoading] = useState(true);
  const [connectionError, setConnectionError] = useState("");
  const [serviceState, setServiceState] = useState<ServiceState>({ status: "starting", revision: -1 });
  const serviceRef = useRef(serviceState);
  const serviceReady = serviceState.status === "ready";
  const [submitError, setSubmitError] = useState("");
  const [busy, setBusy] = useState("");
  const [sidebar, setSidebar] = useState(() => window.innerWidth > 580);
  const [inspector, setInspector] = useState(() => window.innerWidth > 1050);
  const [advanced, setAdvanced] = useState(false);
  const [history, setHistory] = useState(false);
  const [draftLibrary, setDraftLibrary] = useState(false);
  const [historyQuery, setHistoryQuery] = useState("");
  const [historyFilter, setHistoryFilter] = useState("all");
  const [historyScope, setHistoryScope] = useState("current");
  const [importOpen, setImportOpen] = useState(false);
  const [settings, setSettings] = useState<
    "provider" | "runtime" | "appearance" | null
  >(null);
  const [settingsReturnToExperiment, setSettingsReturnToExperiment] =
    useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [templatesOpen, setTemplatesOpen] = useState(false);
  const [chosenTemplate, setChosenTemplate] = useState<Template | null>(null);
  const [templateQuery, setTemplateQuery] = useState("");
  const [confirmation, setConfirmation] = useState<{
    action: "stop" | "cancel";
    run: Run;
  } | null>(null);
  const [runNotice, setRunNotice] = useState<{
    runId: string; navigation: number; lang: Language; serviceRevision: number; scope: object; message: string;
  } | null>(null);
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(
    new Set(),
  );
  const textarea = useRef<HTMLTextAreaElement>(null);
  const draftEpoch = useRef(0);
  const submitInFlight = useRef(false);
  const pickerEpoch = useRef(0);
  const navigationEpoch = useRef(0);
  const form = useRef<HTMLFormElement>(null);
  const selectedRef = useRef(selectedId);
  selectedRef.current = selectedId;
  const revisions = useRef<Revisions | null>(null);
  const detailSequence = useRef(0);
  const refreshPromise = useRef<Promise<void> | null>(null);
  const pollState = useRef({ active: false, failures: 0 });
  const configLoaded = useRef({ env: false, templates: false });
  const themeEdited = useRef(false);
  const languageEdited = useRef(false);
  const notificationsEdited = useRef(false);
  const alive = useRef(true);
  // These transitions must invalidate target-view feedback even after A -> B
  // -> A. Run navigation has its own epoch; confirmation close is intentional.
  const noticeContext = JSON.stringify([lang, serviceState.status, serviceState.revision, settings, searchOpen, templatesOpen, importOpen, advanced]);
  const noticeScope = useRef({ context: noticeContext, token: {} });
  if (noticeScope.current.context !== noticeContext) noticeScope.current = { context: noticeContext, token: {} };
  const toastContext = JSON.stringify([
    lang, serviceState.status, serviceState.revision,
    history ? "history" : draftLibrary ? "drafts" : selectedId ? `run:${selectedId}` : `draft:${drafts.activeId || ""}`,
    settings, searchOpen, templatesOpen, importOpen, advanced, confirmation?.run.id, confirmation?.action,
  ]);
  const { toast, systemToast, dismissToast, toastMessage } = useToast(toastContext);
  const datasetInspection = useDatasetInspection(draft.dataset, drafts.ready && !selectedId && !history && !draftLibrary, serviceState, drafts.activeId || '');

  useEffect(() => {
    if (!runNotice) return;
    // Submission navigates, and stop/cancel closes a confirmation. Publish
    // only after the intended run view has rendered, without making other
    // old async completions eligible to notify a new view.
    if (selectedId === runNotice.runId && !history && !draftLibrary && !settings && !searchOpen && !importOpen &&
        navigationEpoch.current === runNotice.navigation && noticeScope.current.token === runNotice.scope && lang === runNotice.lang &&
        serviceState.status === "ready" && serviceState.revision === runNotice.serviceRevision) {
      toast(runNotice.message);
    }
    setRunNotice(null);
  }, [runNotice, selectedId, history, draftLibrary, settings, searchOpen, importOpen, lang, serviceState, toast]);

  useEffect(() => {
    if (drafts.ready && !drafts.activeId && !selectedId && !history) setDraftLibrary(true);
  }, [drafts.ready, drafts.activeId, selectedId, history]);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  useEffect(() => {
    if (!window.biomaker) return;
    let valid = true;
    window.biomaker
      .getPreferences?.()
      .then((preferences) => {
        if (valid && !themeEdited.current) setTheme(preferences.theme);
        if (
          valid &&
          !notificationsEdited.current &&
          typeof preferences.notifications === "boolean"
        )
          setNotificationsEnabled(preferences.notifications);
        if (valid && !languageEdited.current) {
          if (preferences.language) setLang(preferences.language);
          else
            void window.biomaker
              .setPreferences?.({ language: lang })
              .catch(() => {});
        }
      })
      .catch(() => {});
    return () => {
      valid = false;
    };
  }, []);
  const changeNotifications = (value: boolean) => {
    notificationsEdited.current = true;
    setNotificationsEnabled(value);
    void window.biomaker
      .setPreferences?.({ notifications: value })
      .catch((e) => systemToast(errorMessage(e)));
  };
  const changeLanguage = (value: Language) => {
    languageEdited.current = true;
    setLang(value);
    void window.biomaker
      .setPreferences?.({ language: value })
      .catch((e) => systemToast(errorMessage(e)));
  };
  const changeTheme = (value: Theme) => {
    themeEdited.current = true;
    setTheme(value);
    void window.biomaker
      .setPreferences?.({ theme: value })
      .catch((e) => systemToast(errorMessage(e)));
  };
  const openNotificationRun = useCallback((runId: string) => {
    if (typeof runId !== "string" || !runId.trim()) return;
    pickerEpoch.current += 1;
    navigationEpoch.current++;
    setSelectedId(runId);
    setHistory(false);
    setDraftLibrary(false);
    setSearchOpen(false);
    setTemplatesOpen(false);
    setConfirmation(null);
    // Keep the current draft and any unsaved provider settings intact.
    setNotificationRunToAcknowledge(runId);
  }, []);
  useEffect(() => {
    if (!window.biomaker) return;
    let valid = true;
    let receivedLiveEvent = false;
    const unsubscribe = window.biomaker.onOpenRun?.((runId) => {
      receivedLiveEvent = true;
      if (valid) openNotificationRun(runId);
    });
    void window.biomaker
      .getPendingRun?.()
      .then((runId) => {
        if (valid && !receivedLiveEvent && runId) openNotificationRun(runId);
      })
      .catch((e) => {
        if (valid) systemToast(errorMessage(e));
      });
    return () => {
      valid = false;
      unsubscribe?.();
    };
  }, [openNotificationRun, systemToast]);
  useEffect(() => {
    if (
      !notificationRunToAcknowledge ||
      selectedId !== notificationRunToAcknowledge ||
      !window.biomaker
    )
      return;
    const runId = notificationRunToAcknowledge;
    void window.biomaker
      .acknowledgeOpenRun?.(runId)
      .catch((e) => systemToast(errorMessage(e)))
      .finally(() => {
        setNotificationRunToAcknowledge((current) =>
          current === runId ? null : current,
        );
      });
  }, [selectedId, notificationRunToAcknowledge, systemToast]);
  useEffect(() => {
    persist("biomaker.language", lang);
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  }, [lang]);
  useEffect(() => {
    persist("biomaker.theme", theme);
    const media = matchMedia("(prefers-color-scheme: dark)");
    const apply = () =>
      (document.documentElement.dataset.theme =
        theme === "system" ? (media.matches ? "dark" : "light") : theme);
    apply();
    media.addEventListener("change", apply);
    return () => media.removeEventListener("change", apply);
  }, [theme]);
  useEffect(() => {
    const compact = matchMedia("(max-width: 1050px)");
    const narrow = matchMedia("(max-width: 580px)");
    const adapt = () => {
      if (compact.matches) setInspector(false);
      if (narrow.matches) setSidebar(false);
    };
    compact.addEventListener("change", adapt);
    narrow.addEventListener("change", adapt);
    return () => {
      compact.removeEventListener("change", adapt);
      narrow.removeEventListener("change", adapt);
    };
  }, []);

  const loadDetail = useCallback(async (id: string, force = false) => {
    if (serviceRef.current.status !== "ready") return;
    const serviceRevision = serviceRef.current.revision;
    const sequence = ++detailSequence.current;
    const current = () => selectedRef.current === id && alive.current && sequence === detailSequence.current && serviceRef.current.status === "ready" && serviceRef.current.revision === serviceRevision;
    if (force) {
      setDetailLoading(true);
      setDetailError("");
    }
    try {
      // Read streams independently of catalog/detail refreshes. Each response is
      // bounded and carries its own source identity and observation timestamp.
      const results = await Promise.allSettled([
        (async () => {
          const next = await request<Revisions>(`/api/runs/${encodeURIComponent(id)}/updates`);
          const previous = force ? null : revisions.current;
          const changed = !previous || previous.run_revision !== next.run_revision
            || previous.artifacts?.revision !== next.artifacts?.revision
            || previous.events?.revision !== next.events?.revision;
          const value = changed ? await request<Run>(`/api/runs/${encodeURIComponent(id)}`) : null;
          return { next, value };
        })(),
        request<ActivityWindow>(`/api/desktop/runs/${encodeURIComponent(id)}/activity`),
        request<LogWindow>(`/api/desktop/runs/${encodeURIComponent(id)}/console`),
      ]);
      if (!current()) return;
      const [detailResult, activityResult, logResult] = results;
      setStreamRunId(id);
      if (detailResult.status === "fulfilled") {
        if (detailResult.value.value) setDetail(detailResult.value.value);
        revisions.current = detailResult.value.next;
        setDetailError("");
      } else setDetailError(errorMessage(detailResult.reason));
      if (activityResult.status === "fulfilled") {
        setActivity(activityResult.value);
        setActivityError("");
      } else setActivityError(errorMessage(activityResult.reason));
      if (logResult.status === "fulfilled") {
        setLog(logResult.value);
        setLogError("");
      } else setLogError(errorMessage(logResult.reason));
    } finally {
      if (current()) setDetailLoading(false);
    }
  }, []);

  const loadOlderStream = useCallback(async <T extends ActivityWindow | LogWindow>(id: string, kind: "activity" | "console", cursor: string): Promise<T> => {
    const generation = serviceRef.current.revision;
    const navigation = navigationEpoch.current;
    if (serviceRef.current.status !== "ready" || selectedRef.current !== id) throw new Error("[stream:changed]");
    const value = await request<T>(`/api/desktop/runs/${encodeURIComponent(id)}/${kind}?before=${encodeURIComponent(cursor)}`);
    if (!alive.current || selectedRef.current !== id || navigationEpoch.current !== navigation
      || serviceRef.current.status !== "ready" || serviceRef.current.revision !== generation) throw new Error("[stream:changed]");
    return value;
  }, []);

  const refresh = useCallback((): Promise<void> => {
    if (serviceRef.current.status !== "ready") return Promise.resolve();
    if (refreshPromise.current) return refreshPromise.current;
    const serviceRevision = serviceRef.current.revision;
    const requestedEngineEpoch = engineEpoch.current;
    const current = () => alive.current && serviceRef.current.status === "ready" && serviceRef.current.revision === serviceRevision;
    const task = (async () => {
      const result = await Promise.allSettled([
        request<Run[]>("/api/runs"),
        request<Health>("/api/health"),
        request<EngineStatus>("/api/desktop/status"),
        request<Resources>("/api/system/resources"),
        request<{ items: RunMetadata[] }>("/api/desktop/run-metadata"),
      ]);
      if (!current()) return;
      if (result[0].status === "fulfilled") setRuns(result[0].value);
      if (result[1].status === "fulfilled") setHealth(result[1].value);
      if (result[2].status === "fulfilled" && requestedEngineEpoch === engineEpoch.current) setEngine(result[2].value);
      if (result[3].status === "fulfilled") setResources(result[3].value);
      if (result[4].status === "fulfilled") {
        const rows = result[4].value.items;
        setMetadata(current => mergeMetadata(current, rows));
        setMetadataLoaded(true);
        setMetadataError("");
      } else setMetadataError(errorMessage(result[4].reason));
      const requiredError = result
        .slice(0, 3)
        .find((item) => item.status === "rejected");
      if (requiredError?.status === "rejected") {
        setConnectionError(errorMessage(requiredError.reason));
        pollState.current.failures += 1;
      } else {
        setConnectionError("");
        pollState.current.failures = 0;
      }
      pollState.current.active =
        (result[0].status === "fulfilled" && result[0].value.some(isActive)) ||
        (result[2].status === "fulfilled" && isRuntimeBusy(result[2].value.engine)) ||
        requestedEngineEpoch !== engineEpoch.current;
      if (
        result[1].status === "fulfilled" &&
        (!configLoaded.current.env || !configLoaded.current.templates)
      ) {
        const requestedEnvironmentEpoch = environmentEpoch.current;
        const config = await Promise.allSettled([
          configLoaded.current.env
            ? Promise.resolve(null)
            : request<EnvConfig>("/api/env"),
          configLoaded.current.templates
            ? Promise.resolve(null)
            : request<Template[]>("/api/task-templates"),
        ]);
        if (!current()) return;
        if (config[0].status === "fulfilled" && config[0].value && requestedEnvironmentEpoch === environmentEpoch.current) {
          setEnv(config[0].value);
          configLoaded.current.env = true;
        }
        if (config[1].status === "fulfilled" && config[1].value) {
          setTemplates(config[1].value);
          configLoaded.current.templates = true;
        }
        const configError = config.find((item, index) => item.status === "rejected" &&
          (index !== 0 || requestedEnvironmentEpoch === environmentEpoch.current));
        if (configError?.status === "rejected") {
          setConnectionError(errorMessage(configError.reason));
          pollState.current.failures += 1;
        }
      }
      if (selectedRef.current) await loadDetail(selectedRef.current);
      if (current()) setLoading(false);
    })();
    refreshPromise.current = task;
    void task.finally(() => {
      if (refreshPromise.current === task) refreshPromise.current = null;
    });
    return task;
  }, [loadDetail]);

  const receiveServiceState = useCallback((next: ServiceState) => {
    if (!alive.current || next.revision <= serviceRef.current.revision) return;
    serviceRef.current = next;
    setServiceState(next);
    // Detach requests from an older process generation. Their responses are
    // ignored, and their finally handlers cannot clear the new request slot.
    refreshPromise.current = null;
    detailSequence.current += 1;
    setDetailLoading(false);
    setConnectionError("");
    setDetailError("");
    setMetadataLoaded(false);
    setMetadataError("");
    pollState.current.failures = 0;
    if (next.status === "ready") {
      configLoaded.current = { env: false, templates: false };
      revisions.current = null;
      void refresh();
    } else if (next.status === "failed") {
      setLoading(false);
    }
  }, [refresh]);

  useEffect(() => {
    if (!window.biomaker) return;
    let stopped = false;
    const receive = (next: ServiceState) => { if (!stopped) receiveServiceState(next); };
    const unsubscribe = window.biomaker.onServiceState(receive);
    void window.biomaker.getServiceState().then(receive).catch((error) => {
      if (!stopped && serviceRef.current.revision < 0)
        receive({ status: "failed", revision: 0, message: errorMessage(error) });
    });
    return () => { stopped = true; unsubscribe(); };
  }, [receiveServiceState]);

  const activelyPolling = isRuntimeBusy(engine?.engine) || runs.some(isActive);
  useEffect(() => {
    if (!window.biomaker) {
      setLoading(false);
      return;
    }
    let stopped = false;
    let running = false;
    let timer: ReturnType<typeof setTimeout>;
    void window.biomaker
      .getAppInfo()
      .then((info) => {
        if (!stopped) setVersion(info.version);
      })
      .catch(() => {});
    const tick = async () => {
      clearTimeout(timer);
      if (stopped || running || serviceRef.current.status !== "ready") return;
      running = true;
      try {
        await refresh();
      } finally {
        running = false;
      }
      if (stopped) return;
      const { failures, active } = pollState.current;
      const delay = failures
        ? Math.min(60000, 3000 * 2 ** failures)
        : document.hidden
          ? 30000
          : active
            ? 3000
            : 30000;
      timer = setTimeout(tick, delay);
    };
    void tick();
    const visible = () => {
      if (!document.hidden) void tick();
    };
    document.addEventListener("visibilitychange", visible);
    return () => {
      stopped = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", visible);
    };
  }, [refresh, activelyPolling, serviceState.revision]);

  useEffect(() => {
    revisions.current = null;
    setDetail(null);
    setStreamRunId(null);
    setActivity(null);
    setLog(null);
    setActivityError("");
    setLogError("");
    setDetailError("");
    if (selectedId) void loadDetail(selectedId, true);
  }, [selectedId, loadDetail]);
  const showComposer = useCallback(() => {
    if (submitInFlight.current) return;
    pickerEpoch.current += 1;
    setSelectedId(null);
    setHistory(false);
    setDraftLibrary(false);
    setSearchOpen(false);
    setSubmitError("");
    requestAnimationFrame(() => textarea.current?.focus());
  }, []);
  const changeDraftContext = useCallback(() => {
    draftEpoch.current++;
    pickerEpoch.current++;
    datasetInspection.invalidate();
  }, [datasetInspection.invalidate]);
  const startNew = useCallback(async () => {
    if (submitInFlight.current) return false;
    changeDraftContext();
    const navigation = ++navigationEpoch.current;
    const ok = await drafts.create();
    if (ok && navigation === navigationEpoch.current) { showComposer(); return true; }
    return false;
  }, [drafts.create, changeDraftContext, showComposer]);
  const openDraft = async (id: string) => {
    if (submitInFlight.current) return;
    changeDraftContext();
    const navigation = ++navigationEpoch.current;
    if (await drafts.activate(id) && navigation === navigationEpoch.current) showComposer();
  };
  const duplicateDraft = async (id: string) => {
    if (submitInFlight.current) return false;
    changeDraftContext();
    const navigation = ++navigationEpoch.current;
    const ok = await drafts.duplicate(id);
    if (ok && navigation === navigationEpoch.current) showComposer();
    return ok;
  };
  const deleteDraft = async (id: string) => {
    if (submitInFlight.current) return false;
    changeDraftContext();
    return drafts.remove(id);
  };
  const openSearch = useCallback(() => {
    navigationEpoch.current++;
    pickerEpoch.current++;
    setSearch("");
    setSearchOpen(true);
  }, []);
  useEffect(() => {
    const action = (value: string) => {
      if (submitInFlight.current && value !== "toggle-sidebar") return;
      if (document.querySelector("[role=dialog]") && value !== "refresh")
        return;
      if (value === "new-experiment" || value === "new-task") startNew();
      if (value === "settings") setSettings("provider");
      if (value === "search" || value === "command-palette") openSearch();
      if (value === "toggle-sidebar") setSidebar((v) => !v);
      if (value === "refresh") void refresh();
    };
    const stop = window.biomaker?.onMenuAction(action);
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !document.querySelector("[role=dialog]")) {
        if (window.innerWidth <= 1050) setInspector(false);
        if (window.innerWidth <= 580) setSidebar(false);
      }
      if (!(event.metaKey || event.ctrlKey)) return;
      if (event.key.toLowerCase() === "n") {
        event.preventDefault();
        action("new-experiment");
      }
      if (event.key.toLowerCase() === "k") {
        event.preventDefault();
        action("search");
      }
      if (event.key === ",") {
        event.preventDefault();
        action("settings");
      }
      if (
        event.key === "Enter" &&
        !selectedRef.current &&
        !settings &&
        !importOpen &&
        !searchOpen &&
        !templatesOpen &&
        !draftLibrary
      ) {
        event.preventDefault();
        form.current?.requestSubmit();
      }
    };
    document.addEventListener("keydown", key);
    return () => {
      stop?.();
      document.removeEventListener("keydown", key);
    };
  }, [startNew, openSearch, refresh, settings, searchOpen, templatesOpen, importOpen, draftLibrary]);

  const openTaskSettings = (section: "provider" | "runtime") => {
    if (submitInFlight.current) return;
    setSettingsReturnToExperiment(true);
    setSettings(section);
  };
  const closeSettings = () => {
    setSettings(null);
    setSettingsReturnToExperiment(false);
  };
  const returnToExperiment = () => {
    closeSettings();
    if (!selectedId && !history && !draftLibrary)
      requestAnimationFrame(() => textarea.current?.focus());
  };
  const updateDraft = (value: Partial<Draft>) => {
    if (submitInFlight.current) return;
    draftEpoch.current += 1;
    if (Object.hasOwn(value, "dataset")) datasetInspection.invalidate();
    drafts.update((d) => ({ ...d, ...value }));
    setSubmitError("");
  };
  const updateOptions = (value: Partial<Options>) => {
    if (submitInFlight.current) return;
    draftEpoch.current += 1;
    drafts.update((d) => ({ ...d, options: { ...d.options, ...value } }));
  };
  const selectRun = (id: string) => {
    navigationEpoch.current++;
    pickerEpoch.current += 1;
    setSelectedId(id);
    setHistory(false);
    setDraftLibrary(false);
    setSearchOpen(false);
  };
  const requestReuse = async (run: Run) => {
    if (submitInFlight.current) return;
    if (!run.project_snapshot?.task_body?.trim()) return;
    changeDraftContext();
    const navigation = ++navigationEpoch.current;
    if (await drafts.create(reuseDraft(run)) && navigation === navigationEpoch.current) { showComposer(); setAdvanced(true); }
  };
  const chooseFolder = async () => {
    if (submitInFlight.current || !drafts.ready) return;
    const sequence = ++pickerEpoch.current;
    const epoch = draftEpoch.current;
    const draftId = drafts.currentId();
    if (!draftId) return;
    try {
      const folder = await window.biomaker.chooseDirectory();
      if (alive.current && sequence === pickerEpoch.current && epoch === draftEpoch.current && draftId === drafts.currentId() && !submitInFlight.current && folder) updateDraft({ dataset: folder });
    } catch (e) {
      if (alive.current && sequence === pickerEpoch.current) toast(errorMessage(e));
    }
  };
  const openPath = async (path?: string) => {
    if (!path) return;
    try {
      await window.biomaker.openPath(path);
    } catch (e) {
      toast(errorMessage(e));
    }
  };
  const engineReady = !!engine?.engine.ready && !isRuntimeBusy(engine?.engine);
  const keyReady = !!env?.values.OPENAI_API_KEY?.configured;
  const modelReady = !!(
    draft.options.model_for_all_agents || env?.values.BIOMAKER_MAIN_MODEL?.value
  );
  const providerReady = keyReady && modelReady;
  const draftDisabled = !drafts.ready || !drafts.activeId || drafts.operating || busy === 'submit';
  const activeDraft = drafts.items.find(item => item.id === drafts.activeId);
  const draftReady = drafts.ready && !!drafts.activeId && !drafts.operating && !!draft.task.trim() && !!draft.dataset;
  const preparation = !engineReady
    ? "runtime"
    : !providerReady
      ? "provider"
      : null;
  const primaryNeedsSetup = serviceReady && draftReady && !!preparation;
  const canContinue = serviceReady && draftReady && datasetInspection.ready && !busy && !loading && !connectionError;
  const canSubmit = canContinue && !preparation;
  const selectedRun =
    detail?.id === selectedId
      ? decorateRun(detail, metadata)
      : runs.find((run) => run.id === selectedId);
  const inspectedOptions = selectedId ? selectedRun?.options : draft.options;
  const groups = useMemo(() => {
    const map = new Map<string, Run[]>();
    for (const run of runs) {
      if (!isCurrentRun(run)) continue;
      const key =
        run.dataset_name ||
        basename(run.dataset_path || run.project_snapshot?.dataset_path) ||
        t("未命名数据集", "Unnamed dataset");
      map.set(key, [...(map.get(key) || []), run]);
    }
    return Array.from(map);
  }, [runs, t]);
  const queryRuns = (query: string) =>
    runs.filter((run) =>
      [
        runTitle(run),
        run.history_import?.name,
        run.requested_exp_name,
        run.actual_exp_name,
        run.dataset_name,
        run.dataset_path,
        run.id,
        run.project_snapshot?.task_body,
      ]
        .join(" ")
        .toLocaleLowerCase()
        .includes(query.toLocaleLowerCase()),
    );
  const filteredHistory = queryRuns(historyQuery).filter(run =>
    (historyScope === "archived" ? !!run.archived : isCurrentRun(run)) &&
    (historyFilter === "all" || (historyFilter === "active" ? isActive(run) : run.status === historyFilter)),
  );

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (busy || submitInFlight.current || serviceRef.current.status !== "ready" || !drafts.ready || drafts.operating || !drafts.activeId || draftLibrary) return;
    setSubmitError("");
    if (!draft.task.trim()) {
      textarea.current?.focus();
      return;
    }
    if (
      draft.name.trim() &&
      !/^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(draft.name.trim())
    ) {
      setSubmitError(
        t(
          "实验名称需以英文字母或数字开头，仅使用字母、数字、点、下划线和连字符。",
          "Experiment names must start with a letter or number and use only letters, numbers, dots, underscores, or hyphens.",
        ),
      );
      setAdvanced(true);
      return;
    }
    if (!draft.dataset) {
      setSubmitError(
        t("请先选择数据集文件夹。", "Choose a dataset folder first."),
      );
      return;
    }
    if (!datasetInspection.isReady()) {
      setSubmitError(datasetInspection.pending
        ? t("请等待当前数据集目录检查完成。", "Wait for the current dataset folder inspection to finish.")
        : datasetInspectionMessage(datasetInspection.result?.code || datasetInspection.errorCode || "internal", t));
      return;
    }
    if (!engineReady) {
      openTaskSettings("runtime");
      return;
    }
    if (!keyReady || !modelReady) {
      openTaskSettings("provider");
      return;
    }
    const {
      agent_steps,
      exec_timeout,
      agent_total_time_limit,
      exec_parallel_workers,
    } = draft.options;
    if (
      ![agent_steps, exec_timeout, agent_total_time_limit].every(
        Number.isInteger,
      ) ||
      agent_steps < 1 ||
      agent_steps > 1000 ||
      exec_timeout < 1 ||
      exec_timeout > 86400 ||
      agent_total_time_limit < 1 ||
      agent_total_time_limit > 2592000 ||
      (exec_parallel_workers !== "auto" &&
        (!Number.isInteger(exec_parallel_workers) ||
          exec_parallel_workers < 1 ||
          exec_parallel_workers > 64))
    ) {
      setSubmitError(
        t(
          "请检查高级设置中的范围：步数 1–1000，单次执行 1–86400 秒，探索时长最多 720 小时，并发数 1–64。",
          "Check advanced settings: 1–1000 steps, 1–86400 seconds per execution, up to 720 exploration hours, and 1–64 workers.",
        ),
      );
      setAdvanced(true);
      return;
    }
    const snapshot = structuredClone(draft);
    const submittedDraftId = drafts.currentId()!;
    const submittedNoticeScope = noticeScope.current.token;
    const epoch = draftEpoch.current;
    const settingsEpoch = environmentEpoch.current;
    const serviceRevision = serviceRef.current.revision;
    const current = () => alive.current && drafts.currentId() === submittedDraftId && epoch === draftEpoch.current && serviceRef.current.status === "ready" && serviceRef.current.revision === serviceRevision;
    submitInFlight.current = true;
    pickerEpoch.current += 1;
    setBusy("submit");
    let dispatched = false;
    try {
      if (!await drafts.flush()) {
        setSubmitError(t('草稿尚未保存，实验未提交。请先解决草稿保存问题。', 'The draft has not been saved and no experiment was submitted. Resolve the draft save issue first.'));
        return;
      }
      const [currentEngine, currentEnv] = await Promise.all([
        request<EngineStatus>("/api/desktop/status"),
        request<EnvConfig>("/api/env"),
      ]);
      if (!current() || settingsEpoch !== environmentEpoch.current || !datasetInspection.isReady()) {
        if (alive.current) setSubmitError(t("开始前检查期间状态已变化，尚未提交实验。请检查当前配置后再开始。", "The state changed during the pre-start check. No experiment was submitted; review the current settings before starting again."));
        return;
      }
      applyEngineSnapshot(currentEngine.engine);
      environmentEpoch.current += 1;
      setEnv(currentEnv);
      configLoaded.current.env = true;
      if (!currentEngine.engine.ready || isRuntimeBusy(currentEngine.engine))
        throw new Error(
          t(
            "运行环境尚未就绪，请在设置中完成安装。",
            "The runtime is not ready. Complete installation in Settings.",
          ),
        );
      if (
        !currentEnv.values.OPENAI_API_KEY?.configured ||
        !(
          snapshot.options.model_for_all_agents ||
          currentEnv.values.BIOMAKER_MAIN_MODEL?.value
        )
      )
        throw new Error(
          t("请配置 API Key 和模型。", "Configure an API key and model."),
        );
      dispatched = true;
      const run = await request<Run>("/api/runs", "POST", {
        experiment: {
          dataset_path: snapshot.dataset,
          task_template_path: snapshot.template,
          task_body: snapshot.task.trim(),
          advanced_defaults: snapshot.options,
        },
        requested_exp_name: snapshot.name.trim() || null,
      });
      if (!current()) {
        if (alive.current) {
          setSubmitError(t("提交后服务状态已变化，实验可能已加入队列。草稿已保留，请先查看全部实验再决定是否重新提交。", "The service state changed after submission; the experiment may already be queued. Your draft is preserved. Check All experiments before submitting again."));
          void refresh();
        }
        return;
      }
      setRuns((previous) => [
        run,
        ...previous.filter((item) => item.id !== run.id),
      ]);
      selectRun(run.id);
      const destinationNavigation = navigationEpoch.current;
      draftEpoch.current += 1;
      const consumed = await drafts.remove(submittedDraftId);
      if (alive.current) setRunNotice({
        runId: run.id, navigation: destinationNavigation, lang, serviceRevision, scope: submittedNoticeScope,
        message: consumed ? t("实验已加入队列", "Experiment added to the queue")
          : t("实验已加入队列，但未能移除已提交草稿。请查看此实验，避免重复提交。", "The experiment was queued, but its draft could not be removed. Review this experiment before submitting it again."),
      });
      await refresh();
    } catch (e) {
      if (alive.current) {
        const message = errorMessage(e);
        const datasetCode = datasetAdmissionCode(message);
        if (datasetCode) {
          setSubmitError(`${datasetInspectionMessage(datasetCode, t)} ${t("实验尚未提交，草稿已保留。", "The experiment was not submitted. Your draft is preserved.")}`);
          datasetInspection.invalidate();
        } else {
          setSubmitError(dispatched
            ? `${message} ${t("未能确认实验提交结果，草稿已保留。请先查看全部实验，确认没有新增记录后再重试。", "The submission result could not be confirmed. Your draft is preserved. Check All experiments for a new record before retrying.")}`
            : current() ? message
              : t("开始前检查期间服务状态已变化，尚未提交实验。", "The service state changed during the pre-start check. No experiment was submitted."));
          if (dispatched) void refresh();
        }
      }
    } finally {
      submitInFlight.current = false;
      if (alive.current) setBusy("");
    }
  }
  async function runAction(
    action: "stop" | "cancel",
    run: Run,
    confirmed = false,
  ) {
    if (busy || submitInFlight.current || serviceRef.current.status !== "ready") return;
    if ((action === "stop" || action === "cancel") && !confirmed) {
      setConfirmation({ action, run });
      return;
    }
    setBusy(action);
    setConfirmation(null);
    const actionNavigation = navigationEpoch.current;
    const actionServiceRevision = serviceRef.current.revision;
    const actionNoticeScope = noticeScope.current.token;
    const notify = (message: string) => {
      if (alive.current) setRunNotice({ runId: run.id, navigation: actionNavigation, lang, serviceRevision: actionServiceRevision, scope: actionNoticeScope, message });
    };
    try {
      await request<Run>(
        `/api/runs/${encodeURIComponent(run.id)}/${action}`,
        "POST",
      );
      notify(
        action === "stop"
            ? t("正在停止实验", "Stopping experiment")
            : t("已取消排队", "Queued run cancelled"),
      );
      await refresh();
      if (selectedRef.current) await loadDetail(selectedRef.current, true);
    } catch (e) {
      const message = errorMessage(e);
      const datasetCode = datasetAdmissionCode(message);
      notify(datasetCode ? datasetInspectionMessage(datasetCode, t) : message);
    } finally {
      setBusy("");
    }
  }
  async function resumeQueue() {
    if (busy || submitInFlight.current || serviceRef.current.status !== "ready") return;
    setBusy("resume");
    try {
      const queue = await request<QueueState>("/api/queue");
      await request("/api/queue/resume", "POST", {
        pause_generation: queue.pause_generation,
      });
      await refresh();
      toast(t("队列已恢复", "Queue resumed"));
    } catch (e) {
      toast(errorMessage(e));
    } finally {
      setBusy("");
    }
  }
  async function refreshAll() {
    if (serviceRef.current.status !== "ready") return;
    configLoaded.current = { env: false, templates: false };
    await refresh();
  }

  if (!window.biomaker)
    return (
      <main className="fatal">
        <Mark size={52} />
        <h1>BioMaker</h1>
        <p>
          {t(
            "请通过 BioMaker 桌面应用打开此界面。",
            "Open this interface using the BioMaker desktop app.",
          )}
        </p>
        <p className="muted">
          {t(
            "桌面应用提供本地文件选择、实验执行与安全的配置管理。",
            "The desktop app provides local file access, experiment execution, and configuration management.",
          )}
        </p>
      </main>
    );
  return (
    <div
      className={`app-shell ${sidebar ? "" : "sidebar-hidden"} ${inspector && !history && !draftLibrary ? "" : "inspector-hidden"}`}
    >
      {sidebar && (
        <aside className="sidebar">
          <div className="traffic-space" />
          <div className="brand">
            <span className="brand-mark">
              <Mark size={27} />
            </span>
            <strong>BioMaker</strong>
            <IconButton
              label={t("收起侧边栏", "Hide sidebar")}
              onClick={() => setSidebar(false)}
            >
              <PanelLeft size={16} />
            </IconButton>
          </div>
          <nav className="primary-nav">
            <button
              onClick={() => void startNew()}
              disabled={!drafts.ready || drafts.operating || busy === 'submit'}
            >
              <SquarePen size={17} />
              <span>{t("新建实验", "New experiment")}</span>
              <kbd>⌘ N</kbd>
            </button>
            <button onClick={openSearch}>
              <Search size={17} />
              <span>{t("搜索", "Search")}</span>
              <kbd>⌘ K</kbd>
            </button>
            <button className={draftLibrary || (!selectedId && !history) ? 'selected' : ''} disabled={busy === 'submit'} onClick={() => { navigationEpoch.current++; pickerEpoch.current++; setDraftLibrary(true); setHistory(false); setSelectedId(null); }}>
              <FileText size={17} /><span>{t('草稿', 'Drafts')}</span>
              {drafts.items.length > 0 && <span className="nav-count">{drafts.items.length}</span>}
            </button>
            <button
              className={history ? "selected" : ""}
              disabled={busy === 'submit'}
              onClick={() => {
                navigationEpoch.current++;
                setHistory(true);
                setDraftLibrary(false);
                pickerEpoch.current++;
                setSelectedId(null);
              }}
            >
              <History size={17} />
              <span>{t("全部实验", "All experiments")}</span>
              {runs.length > 0 && (
                <span className="nav-count">{runs.length}</span>
              )}
            </button>
          </nav>
          <div className="sidebar-section-label">
            <span>{t("数据集与实验", "Datasets & experiments")}</span>
            <IconButton
              label={t(
                "选择数据集并新建实验",
                "Choose dataset for a new experiment",
              )}
              onClick={() => {
                void startNew().then(ok => { if (ok) void chooseFolder(); });
              }}
              disabled={!drafts.ready || drafts.operating || busy === 'submit'}
            >
              <Plus size={14} />
            </IconButton>
          </div>
          <div className="sidebar-runs">
            {loading && !runs.length ? (
              <div className="sidebar-loading">
                <Spinner size={15} />
                <span>{t("正在载入", "Loading")}</span>
              </div>
            ) : !groups.length ? (
              <div className="sidebar-empty">
                <Folder size={22} />
                <p>
                  {t(
                    "实验会按数据集整理在这里。",
                    "Your experiments appear here, grouped by dataset.",
                  )}
                </p>
                <button className="text-button" disabled={!serviceReady} onClick={() => setImportOpen(true)}>{t("导入已有实验", "Import existing experiments")}</button>
              </div>
            ) : (
              groups.map(([name, groupRuns]) => (
                <div className="run-group" key={name}>
                  <button
                    className="group-title"
                    onClick={() =>
                      setCollapsedGroups((previous) => {
                        const next = new Set(previous);
                        if (next.has(name)) next.delete(name);
                        else next.add(name);
                        return next;
                      })
                    }
                    aria-expanded={!collapsedGroups.has(name)}
                  >
                    <ChevronDown
                      size={12}
                      className={collapsedGroups.has(name) ? "rotate-left" : ""}
                    />
                    <Folder size={15} />
                    <span>{name}</span>
                    <small>{groupRuns.length}</small>
                  </button>
                  {!collapsedGroups.has(name) &&
                    groupRuns.slice(0, 15).map((run) => (
                      <button
                        key={run.id}
                        className={`sidebar-run ${selectedId === run.id ? "selected" : ""}`}
                        onClick={() => selectRun(run.id)}
                        title={`${runTitle(run)} — ${statusLabel(run, t)}`}
                      >
                        <span className={`run-dot dot-${run.status}`} />
                        <span>{runTitle(run)}</span>
                        {isActive(run) ? (
                          <span className="run-mini-state">
                            {run.status === "queued" ? (
                              <ClockIcon />
                            ) : (
                              <Activity size={12} />
                            )}
                          </span>
                        ) : (
                          <time>
                            {dateText(run.timestamps?.created_at, lang)}
                          </time>
                        )}
                      </button>
                    ))}
                  {!collapsedGroups.has(name) && groupRuns.length > 15 && (
                    <button
                      className="group-more"
                      disabled={busy === 'submit'}
                      onClick={() => {
                        navigationEpoch.current++;
                        pickerEpoch.current++;
                        setDraftLibrary(false);
                        setHistory(true);
                        setSelectedId(null);
                        setHistoryQuery(name);
                        setHistoryScope("current");
                        setHistoryFilter("all");
                      }}
                    >
                      {t(
                        `查看全部 ${groupRuns.length} 个实验`,
                        `View all ${groupRuns.length} experiments`,
                      )}
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
          <div className="sidebar-bottom">
            <button
              className="local-status"
              disabled={busy === "submit"}
              onClick={() => { if (!submitInFlight.current) setSettings("runtime"); }}
            >
              <span
                className={`connection-dot ${serviceState.status === "failed" ? "error" : !serviceReady ? "installing" : engineReady ? "ready" : isRuntimeBusy(engine?.engine) ? "installing" : ""}`}
              />
              <span>
                {!serviceReady
                  ? serviceStatusLabel(serviceState, t)
                  : !engine
                    ? t("正在读取运行环境…", "Loading runtime status…")
                  : isRuntimeBusy(engine?.engine)
                    ? runtimeOperationLabel(engine?.engine, t)
                    : engineReady
                      ? t("本地运行环境已就绪", "Local runtime ready")
                      : t("设置运行环境", "Set up runtime")}
              </span>
              <ChevronRight size={13} />
            </button>
            <button
              className="sidebar-settings"
              aria-label={t("设置", "Settings")}
              disabled={busy === "submit"}
              onClick={() => { if (!submitInFlight.current) setSettings("provider"); }}
            >
              <Settings2 size={17} />
              <span>{t("设置", "Settings")}</span>
              <span className="version">{version}</span>
            </button>
          </div>
        </aside>
      )}
      <main className="workspace">
        <header className="app-header">
          <div className="header-left">
            {!sidebar && (
              <IconButton
                label={t("展开侧边栏", "Show sidebar")}
                onClick={() => setSidebar(true)}
              >
                <PanelLeft size={17} />
              </IconButton>
            )}
            <span>
              {draftLibrary
                ? t('草稿', 'Drafts')
                : history
                ? t("全部实验", "All experiments")
                : selectedRun
                  ? selectedRun.dataset_name ||
                    basename(selectedRun.dataset_path) ||
                    t("实验", "Experiment")
                  : activeDraft ? draftEntryTitle(activeDraft, t) : t("新实验", "New experiment")}
            </span>
            {selectedRun && (
              <>
                <ChevronRight size={13} />
                <span className="header-run-title">
                  {runTitle(selectedRun)}
                </span>
              </>
            )}
          </div>
          <div className="header-actions">
            {drafts.ready && ((!selectedId && !history) || drafts.dirty || drafts.saving || drafts.operating || drafts.error) && <span className={`draft-save-state ${drafts.error ? 'failed' : ''}`} role="status" data-testid="draft-save-state">
              {drafts.saving || drafts.operating ? <Spinner size={12} /> : null}
              {drafts.error ? t('草稿保存异常', 'Draft save issue')
                : drafts.saving || drafts.operating ? t('正在保存…', 'Saving…')
                : drafts.dirty ? t('等待保存', 'Unsaved changes') : t('草稿已保存', 'Drafts saved')}
            </span>}
            <IconButton
              label={t("刷新", "Refresh")}
              onClick={() => void refreshAll()}
              disabled={!serviceReady}
            >
              <RefreshCw size={16} />
            </IconButton>
            {!history && !draftLibrary && (
              <IconButton
                label={
                  inspector
                    ? t("收起实验信息", "Hide experiment details")
                    : t("显示实验信息", "Show experiment details")
                }
                onClick={() => setInspector(!inspector)}
                active={inspector}
              >
                <PanelRight size={17} />
              </IconButton>
            )}
          </div>
        </header>
        {!serviceReady && (
          <div className={`service-banner ${serviceState.status === "failed" ? "failed" : ""}`} role={serviceState.status === "failed" ? "alert" : "status"}>
            {serviceState.status !== "failed" && <Spinner size={15} />}
            <span>{serviceStatusLabel(serviceState, t)}</span>
            {serviceState.status === "reconnecting" && <small>{t("连接恢复后将自动更新。", "Your workspace will update when the connection returns.")}</small>}
            <ServiceRecovery state={serviceState} onState={receiveServiceState} t={t} toast={toast} />
            {serviceState.status === "failed" && serviceState.message && <details><summary>{t("本地错误详情", "Local error details")}</summary><p>{serviceState.message}</p></details>}
          </div>
        )}
        {serviceReady && connectionError && (
          <div className="connection-banner" role="alert">
            <span>
              {t(
                "部分工作区数据未能读取，正在重试。",
                "Some workspace data could not be loaded. Retrying.",
              )}
            </span>
            <button onClick={() => void refreshAll()}>
              {t("立即重试", "Retry now")}
            </button>
            <details>
              <summary>{t("详情", "Details")}</summary>
              <p>{connectionError}</p>
            </details>
          </div>
        )}
        {health?.queue.paused && (
          <div className="queue-banner">
            <Pause size={16} />
            <div>
              <strong>{t("实验队列已暂停", "Experiment queue paused")}</strong>
              <span>
                {health.queue.pause_reason ||
                  t(
                    "检查运行状态后可恢复后续实验。",
                    "Review the run status before resuming queued experiments.",
                  )}
              </span>
            </div>
            <button
              className="secondary-button"
              disabled={!!busy || !serviceReady}
              onClick={resumeQueue}
            >
              {busy === "resume" ? <Spinner size={13} /> : <Play size={13} />}
              {t("恢复队列", "Resume queue")}
            </button>
          </div>
        )}
        {metadataError && <Notice danger>{t("实验整理信息暂时无法载入：", "Experiment organization is unavailable: ")}{metadataError} <button className="text-button" disabled={!serviceReady} onClick={() => void refresh()}>{t("重试", "Retry")}</button></Notice>}
        {drafts.error && <div className="draft-save-error" role="alert">
          <p>{drafts.error}</p>
          <div>
            <button className="secondary-button" disabled={drafts.saving || drafts.operating || busy === 'submit'} onClick={() => { if (!submitInFlight.current) void drafts.retry(); }}>{t('重试草稿保存', 'Retry drafts')}</button>
            {drafts.dirty && <button className="secondary-button" disabled={drafts.saving || drafts.operating || busy === 'submit'} onClick={() => {
              changeDraftContext();
              const navigation = ++navigationEpoch.current;
              void drafts.recoverAsCopy().then(ok => { if (ok && navigation === navigationEpoch.current) showComposer(); });
            }}>{t('另存为新草稿', 'Save as a new draft')}</button>}
            {(drafts.dirty || drafts.legacyRaw) && <button className="text-button" onClick={() => {
              const content = drafts.dirty ? JSON.stringify(draft, null, 2) : drafts.legacyRaw!;
              void navigator.clipboard.writeText(content).then(() => toast(t('草稿内容已复制', 'Draft content copied'))).catch(() => toast(t('无法复制，请手动保留当前内容。', 'Could not copy. Preserve the content manually.')));
            }}>{t('复制草稿内容', 'Copy draft content')}</button>}
          </div>
        </div>}
        <div className="workspace-content">
          {draftLibrary ? (
            <DraftLibrary items={drafts.items} activeId={drafts.activeId} lang={lang} t={t}
              busy={!drafts.ready || drafts.operating || busy === 'submit'} onOpen={id => void openDraft(id)}
              onNew={() => void startNew()} onRename={drafts.rename} onDuplicate={duplicateDraft} onDelete={deleteDraft} />
          ) : history ? (
            <div className="history-view">
              <div className="history-heading">
                <div>
                  <h1>{t("全部实验", "All experiments")}</h1>
                  <p>
                    {t(
                      "每一次探索，都有迹可循。",
                      "Every experiment, with a traceable record.",
                    )}
                  </p>
                </div>
                <div className="history-heading-actions">
                  <button className="secondary-button" disabled={!serviceReady} onClick={() => setImportOpen(true)}><Import size={15} />{t("导入已有实验", "Import existing experiments")}</button>
                  <button className="primary-button" disabled={!drafts.ready || drafts.operating || busy === 'submit'} onClick={() => void startNew()}><Plus size={15} />{t("新建实验", "New experiment")}</button>
                </div>
              </div>
              <div className="history-toolbar">
                <div className="search-input">
                  <Search size={16} />
                  <input
                    value={historyQuery}
                    onChange={(e) => setHistoryQuery(e.target.value)}
                    placeholder={t(
                      "搜索任务或数据集",
                      "Search tasks or datasets",
                    )}
                    aria-label={t("搜索全部实验", "Search all experiments")}
                  />
                </div>
                <select aria-label={t("实验范围", "Experiment scope")} value={historyScope} onChange={event => { setHistoryScope(event.target.value); setHistoryFilter("all"); }}>
                  <option value="current">{t("当前实验", "Current experiments")}</option>
                  <option value="archived">{t("已归档", "Archived")}</option>
                </select>
                <select
                  aria-label={t("筛选实验状态", "Filter run status")}
                  value={historyFilter}
                  onChange={(e) => setHistoryFilter(e.target.value)}
                >
                  <option value="all">{t("全部状态", "All statuses")}</option>
                  <option value="active">{t("进行中", "Active")}</option>
                  <option value="complete">{t("已完成", "Complete")}</option>
                  <option value="incomplete">
                    {t("未完成", "Incomplete")}
                  </option>
                </select>
              </div>
              {filteredHistory.length ? (
                <div className="history-list">
                  {filteredHistory.map((run) => (
                    <button key={run.id} onClick={() => selectRun(run.id)}>
                      <span className={`history-status dot-${run.status}`}>
                        <Activity size={17} />
                      </span>
                      <div>
                        <strong>{runTitle(run)} {run.archived && <span className="archive-badge">{t("已归档", "Archived")}</span>}</strong>
                        <span>
                          {run.dataset_name || basename(run.dataset_path)} ·{" "}
                          {dateText(run.timestamps?.created_at, lang)}
                        </span>
                      </div>
                      <span className={`status-label status-${run.status}`}>
                        {statusLabel(run, t)}
                      </span>
                      <ChevronRight size={15} />
                    </button>
                  ))}
                </div>
              ) : (
                <div className="detail-empty">
                  <History size={30} />
                  <h3>
                    {runs.length
                      ? t("没有匹配的实验", "No matching experiments")
                      : t("从第一个实验开始", "Start your first experiment")}
                  </h3>
                  <p>
                    {runs.length
                      ? t(
                          "尝试其它关键词或状态。",
                          "Try a different search or status.",
                        )
                      : t(
                          "选择数据集，写下你希望解决的问题。",
                          "Choose a dataset and describe what you want to solve.",
                        )}
                  </p>
                  <button
                    className="secondary-button"
                    onClick={
                      runs.length
                        ? () => {
                            setHistoryQuery("");
                            setHistoryFilter("all");
                          }
                        : startNew
                    }
                  >
                    {runs.length
                      ? t("清除筛选", "Clear filters")
                      : t("新建实验", "New experiment")}
                  </button>
                </div>
              )}
            </div>
          ) : selectedId ? (
            selectedRun ? (
              <RunView
                key={selectedId}
                run={selectedRun}
                activity={streamRunId === selectedId ? activity : null}
                activityError={streamRunId === selectedId ? activityError : ""}
                log={streamRunId === selectedId ? log : null}
                logError={streamRunId === selectedId ? logError : ""}
                connected={serviceReady && !connectionError}
                onLoadOlderLog={cursor => loadOlderStream<LogWindow>(selectedId, "console", cursor)}
                onLoadOlderActivity={cursor => loadOlderStream<ActivityWindow>(selectedId, "activity", cursor)}
                loading={detailLoading}
                detailError={detailError}
                lang={lang}
                t={t}
                busy={busy || (!serviceReady ? "service" : "")}
                serviceReady={serviceReady}
                serviceRevision={serviceState.revision}
                metadata={metadata[selectedId] || defaultMetadata(selectedId)}
                metadataReady={metadataLoaded}
                onMetadataChange={mergeRunMetadata}
                onReuse={requestReuse}
                onAction={runAction}
                onRetryLoad={() => void loadDetail(selectedId, true)}
                toast={toast}
              />
            ) : (
              <div className="detail-empty">
                {detailLoading || !serviceReady ? <Spinner size={24} /> : <FileText size={26} />}
                <h3>
                  {!serviceReady
                    ? serviceStatusLabel(serviceState, t)
                    : detailLoading
                    ? t("正在读取实验", "Loading experiment")
                    : t("实验暂时无法读取", "Experiment unavailable")}
                </h3>
                {detailError && <p>{detailError}</p>}
                <button
                  className="secondary-button"
                  onClick={() => void loadDetail(selectedId, true)}
                  disabled={!serviceReady}
                >
                  {t("重试", "Retry")}
                </button>
              </div>
            )
          ) : !drafts.ready || !drafts.activeId ? (
            <div className="detail-empty" role="status">
              {!drafts.error && <Spinner size={24} />}
              <h3>{drafts.error ? t('草稿暂时无法载入', 'Drafts are unavailable') : t('正在载入草稿', 'Loading drafts')}</h3>
              <p>{t('你的草稿保存在这台电脑上。', 'Your drafts are stored on this computer.')}</p>
            </div>
          ) : (
            <div
              className={`new-experiment ${!engineReady || !keyReady || !modelReady ? "needs-setup" : ""}`}
            >
              <div className="welcome">
                <span className="welcome-mark">
                  <Mark size={47} />
                </span>
                <h1>
                  {t(
                    "让下一个发现，从这里开始。",
                    "Your next discovery starts here.",
                  )}
                </h1>
                <p>
                  {t(
                    "描述你的目标，BioMaker 会规划、编写并运行实验。",
                    "Describe your goal. BioMaker plans, codes, and runs the experiment.",
                  )}
                </p>
              </div>
              {draft.reusedFrom && <div className="reuse-notice"><Notice>
                <strong>{t("已从实验复用草稿：", "Draft reused from: ")}{draft.reusedFrom.title}</strong>
                <p>{t("请检查任务、数据集、模型与预算，点击开始后才会创建新实验。", "Review the task, dataset, model, and budget. A new experiment is created only when you start it.")}</p>
                {draft.reusedFrom.requiresDataset && !draft.dataset && <p>{t("历史记录中的数据集路径未沿用，请重新选择本机数据集。", "Historical dataset paths are not reused. Choose a dataset on this computer.")}</p>}
                {draft.reusedFrom.usedDefaults && <p>{t("部分原设置未记录或不可用，已填入默认值，请逐项检查。", "Some original settings were missing or unavailable. Defaults were filled in; review each setting.")}</p>}
              </Notice></div>}
              <form ref={form} onSubmit={submit} className="experiment-form">
                <div
                  className={`composer ${busy === "submit" ? "is-submitting" : ""}`}
                >
                  <textarea
                    ref={textarea}
                    value={draft.task}
                    onChange={(e) =>
                      updateDraft({ task: e.target.value, template: null })
                    }
                    placeholder={t(
                      "你希望用这个数据集解决什么问题？",
                      "What would you like to solve with this dataset?",
                    )}
                    aria-label={t("实验任务", "Experiment task")}
                    rows={5}
                    disabled={draftDisabled}
                    maxLength={500000}
                  />
                  <div className="composer-tools">
                    <button
                      type="button"
                      className={`dataset-button ${draft.dataset ? "has-dataset" : ""}`}
                      disabled={draftDisabled}
                      onClick={chooseFolder}
                      title={
                        draft.dataset ||
                        t("选择数据集文件夹", "Choose dataset folder")
                      }
                    >
                      <FolderOpen size={16} />
                      <span>
                        {draft.dataset
                          ? basename(draft.dataset)
                          : t("选择数据集", "Choose dataset")}
                      </span>
                      {draft.dataset && <small>{t("已选择", "Selected")}</small>}
                    </button>
                    <div className="composer-right">
                      <button
                        type="button"
                        className={`composer-tool ${advanced ? "active" : ""}`}
                        title={t("实验设置", "Experiment settings")}
                        aria-label={t("实验设置", "Experiment settings")}
                        aria-expanded={advanced}
                        disabled={draftDisabled}
                        onClick={() => setAdvanced(!advanced)}
                      >
                        <SlidersHorizontal size={16} />
                      </button>
                      <button
                        type="submit"
                        className={`submit-button ${primaryNeedsSetup ? "setup-submit" : ""}`}
                        aria-label={
                          primaryNeedsSetup
                            ? preparation === "provider"
                              ? t(
                                  "配置模型并继续",
                                  "Configure model to continue",
                                )
                              : t("准备运行环境", "Set up runtime")
                            : t("开始实验", "Start experiment")
                        }
                        title={
                          !serviceReady
                            ? serviceStatusLabel(serviceState, t)
                            : canSubmit
                            ? t("开始实验 ⌘ ↵", "Start experiment ⌘ ↵")
                            : !draft.task.trim()
                              ? t(
                                  "先描述实验目标",
                                  "Describe your experiment first",
                                )
                            : !draft.dataset
                                ? t("请先选择数据集", "Choose a dataset first")
                                : !datasetInspection.ready
                                  ? datasetInspection.pending
                                    ? t("正在检查数据集目录", "Checking dataset folder")
                                    : t("请先处理数据集目录问题", "Resolve the dataset folder issue first")
                                : !engineReady
                                  ? t(
                                      "请先安装运行环境",
                                      "Install the runtime first",
                                    )
                                  : !keyReady || !modelReady
                                    ? t(
                                        "请先配置模型服务",
                                        "Configure a model provider first",
                                      )
                                    : t("开始实验", "Start experiment")
                        }
                        disabled={!canContinue}
                      >
                        {busy === "submit" ? (
                          <Spinner size={19} />
                        ) : primaryNeedsSetup ? (
                          <>
                            {preparation === "provider" ? (
                              <KeyRound size={15} />
                            ) : (
                              <Cpu size={15} />
                            )}
                            <span>
                              {preparation === "provider"
                                ? t("配置模型并继续", "Configure model")
                                : t("准备运行环境", "Set up runtime")}
                            </span>
                          </>
                        ) : (
                          <ArrowUp size={20} />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
                <div className="composer-meta">
                  <button
                    type="button"
                    disabled={busy === "submit"}
                    onClick={() => openTaskSettings("provider")}
                  >
                    <span
                      className={`model-dot ${keyReady && modelReady ? "ready" : ""}`}
                    />
                    {draft.options.model_for_all_agents ||
                      env?.values.BIOMAKER_MAIN_MODEL?.value ||
                      t("配置模型服务", "Configure a model provider")}
                    <ChevronDown size={12} />
                  </button>
                  <span>
                    {Number.isFinite(draft.options.agent_steps) ? draft.options.agent_steps : '—'} {t("步", "steps")}
                    <span className="meta-dot" />
                    {Number.isFinite(draft.options.agent_total_time_limit) ? (
                      draft.options.agent_total_time_limit / 3600
                    ).toLocaleString(lang === "zh" ? "zh-CN" : "en-US", {
                      maximumFractionDigits: 2,
                    }) : '—'}{" "}
                    {t("小时探索上限", "h exploration limit")}
                  </span>
                </div>
                {draft.dataset && <DatasetOverview
                  path={draft.dataset} inspection={datasetInspection} serviceReady={serviceReady}
                  submitting={draftDisabled} lang={lang} t={t}
                  onChoose={() => void chooseFolder()} onOpen={() => void openPath(draft.dataset)}
                  onRefresh={() => { if (!submitInFlight.current) { datasetInspection.invalidate(); setSubmitError(""); } }}
                />}
                {advanced && (
                  <fieldset className="advanced-panel" disabled={draftDisabled}>
                    <div className="advanced-heading">
                      <h3>{t("实验设置", "Experiment settings")}</h3>
                      <button
                        type="button"
                        className="text-button"
                        onClick={() => updateOptions(defaultOptions)}
                      >
                        {t("恢复默认", "Reset defaults")}
                      </button>
                    </div>
                    <label className="field">
                      <span>
                        {t("实验名称", "Experiment name")}{" "}
                        <small>{t("可选", "Optional")}</small>
                      </span>
                      <input
                        value={draft.name}
                        maxLength={80}
                        pattern="[A-Za-z0-9][A-Za-z0-9._\-]{0,79}"
                        title={t(
                          "仅使用英文字母、数字、点、下划线和连字符",
                          "Use letters, numbers, dots, underscores, and hyphens",
                        )}
                        placeholder={t(
                          "例如：lung-nodule-baseline",
                          "e.g. lung-nodule-baseline",
                        )}
                        onChange={(e) => updateDraft({ name: e.target.value })}
                      />
                      <small>
                        {t(
                          "用于输出文件夹；支持英文字母、数字、点、下划线和连字符。",
                          "Used for output folders; accepts letters, numbers, dots, underscores, and hyphens.",
                        )}
                      </small>
                    </label>
                    <div className="advanced-grid">
                      <label className="field">
                        <span>{t("最大探索步数", "Maximum steps")}</span>
                        <input
                          type="number"
                          min={1}
                          max={1000}
                          value={
                            Number.isFinite(draft.options.agent_steps)
                              ? draft.options.agent_steps
                              : ""
                          }
                          onChange={(e) =>
                            updateOptions({
                              agent_steps: e.target.valueAsNumber,
                            })
                          }
                        />
                      </label>
                      <label className="field">
                        <span>
                          {t(
                            "探索时长上限（小时）",
                            "Exploration time limit (hours)",
                          )}
                        </span>
                        <input
                          type="number"
                          min={1 / 3600}
                          max={720}
                          step="any"
                          value={
                            Number.isFinite(
                              draft.options.agent_total_time_limit,
                            )
                              ? Number(
                                  (
                                    draft.options.agent_total_time_limit / 3600
                                  ).toFixed(4),
                                )
                              : ""
                          }
                          onChange={(e) =>
                            updateOptions({
                              agent_total_time_limit: Math.round(
                                e.target.valueAsNumber * 3600,
                              ),
                            })
                          }
                        />
                        <small>
                          {t(
                            "不含数据准备、最终重训与报告；单次请求可能延后结束。",
                            "Excludes data preparation, final retraining, and reporting; an in-flight request may finish later.",
                          )}
                        </small>
                      </label>
                      <label className="field">
                        <span>
                          {t(
                            "单次执行时限（秒）",
                            "Execution timeout (seconds)",
                          )}
                        </span>
                        <input
                          type="number"
                          min={1}
                          max={86400}
                          value={
                            Number.isFinite(draft.options.exec_timeout)
                              ? draft.options.exec_timeout
                              : ""
                          }
                          onChange={(e) =>
                            updateOptions({
                              exec_timeout: e.target.valueAsNumber,
                            })
                          }
                        />
                      </label>
                      <label className="field">
                        <span>{t("并发工作进程", "Parallel workers")}</span>
                        <select
                          value={draft.options.exec_parallel_workers}
                          onChange={(e) =>
                            updateOptions({
                              exec_parallel_workers:
                                e.target.value === "auto"
                                  ? "auto"
                                  : Number(e.target.value),
                            })
                          }
                        >
                          <option value="auto">{t("自动", "Auto")}</option>
                          {typeof draft.options.exec_parallel_workers === "number" && ![1, 2, 4, 8, 16, 32, 64].includes(draft.options.exec_parallel_workers) && <option value={draft.options.exec_parallel_workers}>{draft.options.exec_parallel_workers}</option>}
                          {[1, 2, 4, 8, 16, 32, 64].map((n) => (
                            <option key={n} value={n}>
                              {n}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <label className="field">
                      <span>
                        {t("此次实验使用的模型", "Model for this experiment")}
                      </span>
                      <input
                        value={draft.options.model_for_all_agents || ""}
                        placeholder={t(
                          "留空使用默认模型",
                          "Leave blank to use the default model",
                        )}
                        onChange={(e) =>
                          updateOptions({
                            model_for_all_agents: e.target.value.trim() || null,
                          })
                        }
                      />
                    </label>
                    <label className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={draft.options.final_retrain_enabled}
                        onChange={(e) =>
                          updateOptions({
                            final_retrain_enabled: e.target.checked,
                          })
                        }
                      />
                      <span>
                        <strong>{t("最终重训", "Final retraining")}</strong>
                        <small>
                          {t(
                            "探索结束后，按实验协议重训最佳方案。",
                            "Retrain the best candidate after exploration, following the experiment protocol.",
                          )}
                        </small>
                      </span>
                    </label>
                  </fieldset>
                )}
                {submitError && <Notice danger>{submitError}</Notice>}
              </form>
              {serviceReady && !loading && engine && env && (!engineReady || !keyReady || !modelReady) && (
                <section className="onboarding">
                  <div className="onboarding-heading">
                    <h2>
                      {t(
                        "开始前，完成两项准备",
                        "Two steps before your first run",
                      )}
                    </h2>
                    <span>
                      {Number(engineReady) + Number(keyReady && modelReady)} / 2
                    </span>
                  </div>
                  <CheckItem
                    done={engineReady}
                    title={t("准备本地运行环境", "Prepare the local runtime")}
                    detail={
                      isRuntimeBusy(engine?.engine)
                        ? t(
                            "运行环境正在维护，打开可查看进度。",
                            "Runtime maintenance is in progress. Open to view progress.",
                          )
                        : t(
                            "为实验安装独立的 Python 环境。",
                            "Install an isolated Python environment for experiments.",
                          )
                    }
                    label={
                      engineReady
                        ? t("就绪", "Ready")
                        : isRuntimeBusy(engine?.engine)
                          ? t("查看", "View")
                          : t("安装", "Install")
                    }
                    onClick={() => openTaskSettings("runtime")}
                  />
                  <CheckItem
                    done={keyReady && modelReady}
                    title={t("连接模型服务", "Connect a model provider")}
                    detail={t(
                      "添加你的 API Key，并选择模型。",
                      "Add your API key and choose a model.",
                    )}
                    label={
                      keyReady && modelReady
                        ? t("就绪", "Ready")
                        : t("配置", "Configure")
                    }
                    onClick={() => openTaskSettings("provider")}
                  />
                </section>
              )}
              {!draft.task && (
                <div className="task-suggestions">
                  {exampleTasks.map(
                    ({
                      id,
                      zh,
                      en,
                      hintZh,
                      hintEn,
                      Icon,
                      promptZh,
                      promptEn,
                    }) => (
                      <button
                        key={id}
                        disabled={draftDisabled}
                        onClick={() => {
                          updateDraft({
                            task: t(promptZh, promptEn),
                            template: null,
                            ...(id === "baseline"
                              ? {
                                  options: {
                                    ...draft.options,
                                    agent_steps: 3,
                                    exec_timeout: 300,
                                    exec_parallel_workers: 1 as const,
                                    agent_total_time_limit: 1800,
                                    final_retrain_enabled: false,
                                  },
                                }
                              : {}),
                          });
                          requestAnimationFrame(() =>
                            textarea.current?.focus(),
                          );
                        }}
                      >
                        <Icon size={19} />
                        <span>
                          <strong>{t(zh, en)}</strong>
                          <small>{t(hintZh, hintEn)}</small>
                        </span>
                      </button>
                    ),
                  )}
                </div>
              )}
              {templates.length > 0 && (
                <button
                  className="template-link"
                  disabled={draftDisabled}
                  onClick={() => {
                    if (submitInFlight.current) return;
                    setTemplatesOpen(true);
                    setChosenTemplate(null);
                  }}
                >
                  <BookOpen size={14} />
                  {t("从实验模板开始", "Start from an experiment template")}
                  <ChevronRight size={13} />
                </button>
              )}
              <p className="new-footnote">
                {t(
                  "实验保存在本机。运行会使用所选模型服务，可能产生 API 费用。",
                  "Experiments are saved locally. Runs use your selected model provider and may incur API charges.",
                )}
              </p>
            </div>
          )}
        </div>
        <footer className="workspace-statusbar">
          <span>
            <span
              className={`connection-dot ${serviceState.status === "failed" || connectionError ? "error" : serviceReady ? "ready" : "installing"}`}
            />
            {!serviceReady
              ? serviceStatusLabel(serviceState, t)
              : connectionError
                ? t("数据更新失败", "Data update failed")
              : t("本地工作区", "Local workspace")}
          </span>
          <span>
            {!serviceReady
              ? ""
              : health?.queue.running
              ? t(
                  `${health.queue.running} 个实验运行中`,
                  `${health.queue.running} running`,
                )
              : loading
                ? t("连接中", "Connecting")
                : !engineReady
                  ? t("等待运行环境", "Runtime needed")
                  : !providerReady
                    ? t("等待配置模型", "Model setup needed")
                    : t("就绪", "Ready")}
            {serviceReady && (health?.queue.queued || 0) > 0 &&
              ` · ${health?.queue.queued} ${t("排队中", "queued")}`}
          </span>
        </footer>
      </main>
      {inspector && !history && !draftLibrary && (
        <aside className="inspector">
          <header>
            <span>{t("实验信息", "Experiment details")}</span>
            <IconButton
              label={t("收起实验信息", "Hide experiment details")}
              onClick={() => setInspector(false)}
            >
              <PanelRight size={15} />
            </IconButton>
          </header>
          <div className="inspector-scroll">
            <section className="inspector-section">
              <h3>{t("数据集", "Dataset")}</h3>
              {selectedRun?.dataset_path ||
              selectedRun?.project_snapshot?.dataset_path ||
              (!selectedId && draft.dataset) ? (
                <>
                  <div className="inspector-dataset">
                    <FolderOpen size={20} />
                    <strong>
                      {basename(
                        selectedRun?.dataset_path ||
                          selectedRun?.project_snapshot?.dataset_path ||
                          draft.dataset,
                      )}
                    </strong>
                  </div>
                  <p className="path-text">
                    {selectedRun?.dataset_path ||
                      selectedRun?.project_snapshot?.dataset_path ||
                      draft.dataset}
                  </p>
                  <button
                    className="text-button"
                    onClick={() =>
                      openPath(
                        selectedRun?.dataset_path ||
                          selectedRun?.project_snapshot?.dataset_path ||
                          draft.dataset,
                      )
                    }
                  >
                    <Folder size={13} />
                    {t("在访达中打开", "Open in Finder")}
                  </button>
                </>
              ) : selectedId ? (
                <p className="muted">
                  {t("未记录数据集路径", "No dataset path recorded")}
                </p>
              ) : (
                <button className="choose-dataset-empty" disabled={draftDisabled} onClick={chooseFolder}>
                  <FolderOpen size={23} />
                  <span>{t("选择本地数据集", "Choose a local dataset")}</span>
                  <small>
                    {t(
                      "文件夹中的数据用于此实验",
                      "Files in this folder are used for this experiment",
                    )}
                  </small>
                  <Plus size={14} />
                </button>
              )}
            </section>
            <section className="inspector-section">
              <h3>{t("运行预算", "Run budget")}</h3>
              <dl className="inspector-facts">
                <div>
                  <dt>{t("探索步数", "Exploration steps")}</dt>
                  <dd>{Number.isFinite(inspectedOptions?.agent_steps) ? inspectedOptions?.agent_steps : "—"}</dd>
                </div>
                <div>
                  <dt>{t("探索时长上限", "Exploration time limit")}</dt>
                  <dd>
                    {inspectedOptions?.agent_total_time_limit != null && Number.isFinite(inspectedOptions.agent_total_time_limit)
                      ? `${Math.round((inspectedOptions.agent_total_time_limit / 3600) * 100) / 100} ${t("小时", "hours")}`
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt>{t("并发进程", "Workers")}</dt>
                  <dd>
                    {inspectedOptions?.exec_parallel_workers === "auto"
                      ? t("自动", "Auto")
                      : Number.isFinite(inspectedOptions?.exec_parallel_workers) ? inspectedOptions?.exec_parallel_workers : "—"}
                  </dd>
                </div>
                <div>
                  <dt>{t("最终重训", "Final retraining")}</dt>
                  <dd>
                    {typeof inspectedOptions?.final_retrain_enabled ===
                    "boolean"
                      ? inspectedOptions.final_retrain_enabled
                        ? t("开启", "On")
                        : t("关闭", "Off")
                      : "—"}
                  </dd>
                </div>
              </dl>
              {!selectedId && (
                <button
                  className="text-button"
                  disabled={draftDisabled}
                  onClick={() => setAdvanced(!advanced)}
                >
                  <SlidersHorizontal size={13} />
                  {t("调整预算", "Adjust budget")}
                </button>
              )}
            </section>
            <ResourcesView data={resources} t={t} />
            {selectedRun?.log_dir && (
              <section className="inspector-section">
                <h3>{t("输出位置", "Output location")}</h3>
                <p className="path-text">{selectedRun.log_dir}</p>
                <button
                  className="text-button"
                  onClick={() => openPath(selectedRun.log_dir)}
                >
                  <FolderOpen size={14} />
                  {t("打开输出文件夹", "Open output folder")}
                </button>
              </section>
            )}
            <section className="inspector-note">
              <Sparkles size={17} />
              <p>
                {t(
                  "一个目标，完整的实验记录。代码、评估和结果都保留在你的工作区。",
                  "One goal, a complete experiment record. Code, evaluations, and results stay in your workspace.",
                )}
              </p>
            </section>
          </div>
        </aside>
      )}
      {settings && (
        <Settings
          t={t}
          lang={lang}
          setLang={changeLanguage}
          theme={theme}
          setTheme={changeTheme}
          notificationsEnabled={notificationsEnabled}
          setNotificationsEnabled={changeNotifications}
          engine={engine}
          serviceState={serviceState}
          onServiceState={receiveServiceState}
          env={env}
          version={version}
          initialSection={settings}
          onSectionChange={setSettings}
          onClose={closeSettings}
          onReturnToExperiment={
            settingsReturnToExperiment ? returnToExperiment : undefined
          }
          modelOverride={
            settingsReturnToExperiment
              ? selectedRun?.options?.model_for_all_agents ||
                selectedRun?.options?.main_model ||
                (!selectedId ? draft.options.model_for_all_agents : null)
              : null
          }
          onSaved={(value) => {
            // A background refresh started before a save or explicit review
            // must not replace the newer configuration the user just accepted.
            environmentEpoch.current += 1;
            setEnv(value);
            configLoaded.current.env = true;
          }}
          onRefresh={refreshAll}
          onEngineChanged={applyEngineSnapshot}
          toast={toast}
        />
      )}
      {importOpen && <HistoryImport t={t} lang={lang} serviceState={serviceState} onClose={() => setImportOpen(false)} onImported={() => void refreshAll()} onOpenRun={id => { selectRun(id); setImportOpen(false); }} />}
      {searchOpen && (
        <Modal
          title={t("搜索实验", "Search experiments")}
          onClose={() => setSearchOpen(false)}
          t={t}
          className="search-modal"
        >
          <div className="command-search">
            <Search size={19} />
            <input
              autoFocus
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t(
                "按任务、数据集或实验 ID 搜索…",
                "Search tasks, datasets, or run IDs…",
              )}
              aria-label={t("搜索实验", "Search experiments")}
            />
          </div>
          <div className="command-results">
            {!search && (
              <button className="command-item" disabled={!drafts.ready || drafts.operating || busy === 'submit'} onClick={() => void startNew()}>
                <SquarePen size={18} />
                <span>{t("新建实验", "New experiment")}</span>
                <kbd>⌘ N</kbd>
              </button>
            )}
            {queryRuns(search)
              .slice(0, 30)
              .map((run) => (
                <button
                  className="command-item"
                  key={run.id}
                  onClick={() => selectRun(run.id)}
                >
                  <span className={`run-dot dot-${run.status}`} />
                  <span>
                    <strong>{runTitle(run)}</strong>
                    <small>
                      {run.dataset_name || basename(run.dataset_path)}
                    </small>
                  </span>
                  {run.archived && <span className="archive-badge">{t("已归档", "Archived")}</span>}
                  <span className="command-status">{statusLabel(run, t)}</span>
                </button>
              ))}
            {search && !queryRuns(search).length && (
              <div className="search-empty">
                {t("没有找到相关实验。", "No matching experiments.")}
              </div>
            )}
          </div>
          <div className="command-footer">
            <span>
              {t(
                "搜索此电脑保存的实验",
                "Search experiments saved on this computer",
              )}
            </span>
            <kbd>Esc</kbd>
          </div>
        </Modal>
      )}
      {templatesOpen && (
        <Modal
          title={t("实验模板", "Experiment templates")}
          onClose={() => setTemplatesOpen(false)}
          t={t}
          className="templates-modal"
        >
          <p className="template-intro">
            {t(
              "模板提供任务和评估约定，使用前请核对你的数据结构。",
              "Templates define task and evaluation contracts. Check that they match your dataset before running.",
            )}
          </p>
          <input
            value={templateQuery}
            onChange={(e) => setTemplateQuery(e.target.value)}
            placeholder={t("搜索模板…", "Search templates…")}
            aria-label={t("搜索模板", "Search templates")}
          />
          <div className="template-grid">
            <div className="template-list">
              {templates
                .filter((template) =>
                  `${template.name} ${template.id}`
                    .toLowerCase()
                    .includes(templateQuery.toLowerCase()),
                )
                .map((template) => (
                  <button
                    className={
                      chosenTemplate?.id === template.id ? "selected" : ""
                    }
                    key={template.id}
                    onClick={() => setChosenTemplate(template)}
                  >
                    <FileText size={15} />
                    <span>{template.name}</span>
                    <ChevronRight size={13} />
                  </button>
                ))}
            </div>
            <div className="template-preview">
              {chosenTemplate ? (
                <>
                  <h3>{chosenTemplate.name}</h3>
                  <pre>{chosenTemplate.body}</pre>
                </>
              ) : (
                <div className="detail-empty">
                  <BookOpen size={26} />
                  <p>
                    {t(
                      "选择模板以查看任务内容",
                      "Choose a template to preview its task",
                    )}
                  </p>
                </div>
              )}
            </div>
          </div>
          <div className="template-footer">
            <small>
              {draft.task
                ? t(
                    "使用模板会替换当前任务描述。",
                    "Using a template replaces your current task description.",
                  )
                : ""}
            </small>
            <button
              className="primary-button"
              disabled={!chosenTemplate || busy === "submit"}
              onClick={() => {
                if (!chosenTemplate || submitInFlight.current) return;
                updateDraft({
                  task: chosenTemplate.body,
                  template: chosenTemplate.path,
                  ...(chosenTemplate.dataset_path
                    ? { dataset: chosenTemplate.dataset_path }
                    : {}),
                });
                setTemplatesOpen(false);
              }}
            >
              {t("使用模板", "Use template")}
              <ArrowRight size={14} />
            </button>
          </div>
        </Modal>
      )}
      {confirmation && (
        <Modal
          title={
            confirmation.action === "stop"
              ? t("停止这个实验？", "Stop this experiment?")
              : t("取消排队？", "Cancel this queued run?")
          }
          onClose={() => setConfirmation(null)}
          t={t}
          className="confirm-modal"
        >
          <p>
            {confirmation.action === "stop"
              ? t(
                  "将结束当前实验进程，已生成的日志和文件会保留。停止后，可以创建一次新的运行。",
                  "The current experiment process will end. Existing logs and files are kept. You can create a new run afterward.",
                )
              : t(
                  "这个实验还没有开始。取消后会保留记录，之后可以再次运行。",
                  "This experiment has not started. Cancelling keeps its record, and you can run it again later.",
                )}
          </p>
          <strong className="confirm-run-title">
            {runTitle(confirmation.run)}
          </strong>
          <div className="confirm-actions">
            <button
              className="secondary-button"
              onClick={() => setConfirmation(null)}
            >
              {t("返回", "Go back")}
            </button>
            <button
              className="danger-button"
              onClick={() =>
                void runAction(confirmation.action, confirmation.run, true)
              }
            >
              {confirmation.action === "stop"
                ? t("停止实验", "Stop experiment")
                : t("取消排队", "Cancel queued run")}
            </button>
          </div>
        </Modal>
      )}
      {toastMessage && (
        <div key={toastMessage.id} className="toast" role="status">
          <span>{toastMessage.message}</span>
          <IconButton
            label={t("关闭提示", "Dismiss")}
            onClick={() => dismissToast(toastMessage.id)}
          >
            <X size={14} />
          </IconButton>
        </div>
      )}
    </div>
  );
}
function ClockIcon() {
  return <span className="queue-mini-dot" />;
}
