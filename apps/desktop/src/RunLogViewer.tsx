import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowDownToLine, ArrowUp, ArrowUpToLine, Check, ChevronDown, ChevronUp, Copy, Pause, RefreshCw, Search, Terminal, WrapText, X } from "lucide-react";
import { IconButton, Spinner } from "./components";
import { formatBytes, type Language, type Translate } from "./models";
import type { LogWindow } from "./runStreams";
import "./RunLogViewer.css";

// Keep work bounded even if a malformed response contains one enormous line.
// The service returns at most 262,144 bytes, so a valid decoded UTF-8 window
// always fits. This extra guard must never remove the end of a valid tail.
const MAX_RENDER_CHARACTERS = 262_144;
const MAX_HIGHLIGHTS = 1_000;

export interface RunLogViewerProps {
  runId: string;
  lang: Language;
  t: Translate;
  connected: boolean;
  active: boolean;
  snapshot: LogWindow | null;
  loading: boolean;
  error: string | null;
  onLoadOlder: (cursor: string) => Promise<LogWindow>;
  onRefresh: () => void;
  onCopy: (text: string) => Promise<boolean>;
  onSaveFullLog: () => Promise<void>;
  exportBusy: boolean;
  /** Keep mounted across tabs; only the visible reader handles keyboard/follow. */
  visible?: boolean;
}

function sameWindow(left: LogWindow | null, right: LogWindow | null): boolean {
  return left === right || !!left && !!right && left.status === right.status
    && left.source_id === right.source_id && left.revision === right.revision
    && left.start === right.start && left.end === right.end && left.text === right.text;
}

function streamMessage(error: unknown, t: Translate): string {
  const message = error instanceof Error ? error.message : String(error || "");
  const code = /\[stream:([a-z_]+)\]/.exec(message)?.[1];
  if (code === "changed") return t("日志已变化，无法读取此前片段。请查看最新日志。", "The log changed and the earlier window is no longer available. View the latest log.");
  if (code === "unavailable") return t("日志暂时无法读取。已显示的内容仍然保留，可以稍后重试。", "The log is unavailable. The displayed text is kept; try again later.");
  if (code === "invalid_request") return t("无法读取这个日志片段。请刷新日志后重试。", "This log window could not be read. Refresh the log and try again.");
  return t("日志读取失败。已显示的内容仍然保留，请重试。", "The log could not be read. The displayed text is kept; please retry.");
}

function exportMessage(error: unknown, t: Translate): string {
  const message = error instanceof Error ? error.message : String(error || "");
  const code = /\[artifact:([a-z_]+)\]/.exec(message)?.[1];
  if (code === "too_large") return t("完整日志超过 256 MiB 导出上限。", "The full log exceeds the 256 MiB export limit.");
  if (code === "busy") return t("已有文件正在导出，请完成后再保存。", "A file export is already in progress. Wait for it to finish.");
  if (code === "changed") return t("日志文件在导出时发生变化，目标文件未被替换。请刷新后重试。", "The log changed during export. The destination was not replaced. Refresh and retry.");
  if (code === "service_changed") return t("本地服务已变化，目标文件未被替换。请连接恢复后重试。", "The local service changed. The destination was not replaced. Retry after reconnecting.");
  if (code === "unavailable") return t("完整日志暂时无法读取，请刷新后重试。", "The full log is unavailable. Refresh and retry.");
  return t("日志未能保存。请检查目标位置的权限和可用空间后重试。", "The log could not be saved. Check access and free space at the destination, then retry.");
}

export default function RunLogViewer(props: RunLogViewerProps) {
  // Keep reading state across hidden tabs, but never across experiment identity.
  return <LogReader key={props.runId} {...props} />;
}

function LogReader({ runId, lang, t, connected, active, snapshot, loading, error, onLoadOlder, onRefresh, onCopy, onSaveFullLog, exportBusy, visible: shown = true }: RunLogViewerProps) {
  const [visible, setVisible] = useState<LogWindow | null>(snapshot);
  const [following, setFollowing] = useState(active);
  const [olderPage, setOlderPage] = useState(false);
  const [olderBusy, setOlderBusy] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [copyState, setCopyState] = useState<"idle" | "copying" | "copied" | "failed">("idle");
  const [saving, setSaving] = useState(false);
  const [sourceNotice, setSourceNotice] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedMatch, setSelectedMatch] = useState(0);
  const [jump, setJump] = useState(0);
  const [wrap, setWrap] = useState(true);
  const root = useRef<HTMLElement>(null);
  const viewport = useRef<HTMLDivElement>(null);
  const body = useRef<HTMLPreElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const actionSequence = useRef(0);
  const followingRef = useRef(following);
  const mounted = useRef(true);
  const copyPending = useRef(false);
  const savePending = useRef(false);
  const previousSource = useRef(snapshot?.status === "ready" ? snapshot.source_id : null);
  const scrollTarget = useRef<"top" | "bottom" | null>("bottom");
  const programmaticScroll = useRef(false);
  const scrollFrame = useRef<number | null>(null);
  const latest = useRef({ snapshot, connected });
  latest.current = { snapshot, connected };
  followingRef.current = following;
  const searchId = useId();
  const scopeId = useId();

  const rawText = visible?.status === "ready" ? visible.text : "";
  let text = rawText.slice(0, MAX_RENDER_CHARACTERS);
  if (text.length < rawText.length && /[\uD800-\uDBFF]$/.test(text)) text = text.slice(0, -1);
  const clipped = text.length < rawText.length;
  const matches = useMemo(() => {
    const values: { start: number; end: number }[] = [];
    if (!query) return { values, capped: false };
    const pattern = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "giu");
    let found: RegExpExecArray | null;
    while ((found = pattern.exec(text))) {
      if (values.length === MAX_HIGHLIGHTS) return { values, capped: true };
      values.push({ start: found.index, end: found.index + found[0].length });
    }
    return { values, capped: false };
  }, [text, query]);
  const currentMatch = matches.values.length ? Math.min(selectedMatch, matches.values.length - 1) : -1;
  const sourceChanged = !!visible?.source_id && snapshot?.status === "ready" && visible.source_id !== snapshot.source_id;
  const pending = !!visible && !!snapshot && !sameWindow(visible, snapshot);
  const latestMissing = snapshot?.status === "missing";
  const showingSavedText = !!text && (!connected || !!error || latestMissing);
  const canLoadOlder = connected && !olderBusy && !!visible?.has_older && !!visible.older_cursor
    && !sourceChanged && !latestMissing;
  const observed = visible?.observed_at ? Date.parse(visible.observed_at) : NaN;
  const observedTime = Number.isFinite(observed) ? new Date(observed).toLocaleTimeString(lang === "zh" ? "zh-CN" : "en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }) : null;

  function invalidateOlder() {
    actionSequence.current++;
    setOlderBusy(false);
  }
  function pause(cancelOlder = true) {
    if (cancelOlder) invalidateOlder();
    followingRef.current = false;
    setFollowing(false);
  }
  function moveScroll(top: number) {
    const element = viewport.current;
    if (!element || !element.getClientRects().length) return;
    programmaticScroll.current = true;
    element.scrollTop = top;
    if (scrollFrame.current !== null) cancelAnimationFrame(scrollFrame.current);
    scrollFrame.current = requestAnimationFrame(() => { programmaticScroll.current = false; scrollFrame.current = null; });
  }
  function followLatest() {
    invalidateOlder();
    setPageError(null);
    setSourceNotice(false);
    setOlderPage(false);
    setQuery("");
    setSelectedMatch(0);
    setCopyState("idle");
    followingRef.current = true;
    setFollowing(true);
    const value = latest.current.snapshot;
    if (value?.status === "ready" || !text) setVisible(value);
    scrollTarget.current = "bottom";
    setJump(value => value + 1);
  }
  function openSearch() {
    pause();
    setSearchOpen(true);
    requestAnimationFrame(() => { search.current?.focus(); search.current?.select(); });
  }
  function showWindowStart() {
    pause();
    scrollTarget.current = "top";
    setJump(value => value + 1);
  }
  function closeSearch() {
    invalidateOlder();
    setSearchOpen(false);
    setQuery("");
    setSelectedMatch(0);
    viewport.current?.focus({ preventScroll: true });
  }
  function changeMatch(direction: number) {
    pause();
    if (!matches.values.length) return;
    setSelectedMatch((Math.max(0, currentMatch) + direction + matches.values.length) % matches.values.length);
    setJump(value => value + 1);
  }

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      actionSequence.current++;
      if (scrollFrame.current !== null) cancelAnimationFrame(scrollFrame.current);
    };
  }, []);
  useEffect(() => {
    if (!connected) invalidateOlder();
  }, [connected]);
  useEffect(() => {
    if (!snapshot) return;
    if (snapshot.status === "ready") {
      if (previousSource.current && previousSource.current !== snapshot.source_id) {
        invalidateOlder();
        setSourceNotice(true);
      }
      previousSource.current = snapshot.source_id;
    }
    if ((followingRef.current && snapshot.status === "ready") || !visible || visible.status === "missing") {
      if (!visible || visible.status === "missing") scrollTarget.current = "bottom";
      setVisible(snapshot);
      if (followingRef.current) { setOlderPage(false); scrollTarget.current = "bottom"; }
    }
  }, [snapshot]);
  useEffect(() => { setSelectedMatch(0); setCopyState("idle"); }, [text, query]);

  useLayoutEffect(() => {
    if (!shown || !viewport.current) return;
    if (scrollTarget.current) {
      moveScroll(scrollTarget.current === "bottom" ? viewport.current.scrollHeight : 0);
      scrollTarget.current = null;
    } else if (followingRef.current) moveScroll(viewport.current.scrollHeight);
    else if (query && currentMatch >= 0) {
      const mark = body.current?.querySelector<HTMLElement>(`[data-log-match="${currentMatch}"]`);
      if (mark) {
        const box = viewport.current.getBoundingClientRect();
        const target = mark.getBoundingClientRect();
        moveScroll(viewport.current.scrollTop + target.top - box.top - Math.max(16, viewport.current.clientHeight / 3));
        if (!wrap) viewport.current.scrollLeft = Math.max(0, viewport.current.scrollLeft + target.left - box.left - 28);
      }
    }
  }, [text, query, currentMatch, jump, wrap, following, shown]);

  useEffect(() => {
    const element = viewport.current;
    if (!element) return;
    const observer = new ResizeObserver(() => {
      if (followingRef.current && element.getClientRects().length) moveScroll(element.scrollHeight);
    });
    observer.observe(element);
    if (body.current) observer.observe(body.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      if (!(event.metaKey || event.ctrlKey) || event.key.toLowerCase() !== "f" || event.altKey
        || !root.current?.getClientRects().length || document.querySelector('[role="dialog"]')) return;
      event.preventDefault(); event.stopPropagation(); openSearch();
    };
    const selection = () => {
      const selected = window.getSelection();
      if (!root.current?.getClientRects().length || !viewport.current || !selected || selected.isCollapsed || !selected.rangeCount) return;
      try { if (selected.getRangeAt(0).intersectsNode(viewport.current)) pause(); } catch { /* Detached selection during navigation. */ }
    };
    document.addEventListener("keydown", keyboard, true);
    document.addEventListener("selectionchange", selection);
    return () => { document.removeEventListener("keydown", keyboard, true); document.removeEventListener("selectionchange", selection); };
  }, []);

  async function loadOlder() {
    if (!canLoadOlder || !visible?.older_cursor) return;
    const cursor = visible.older_cursor;
    const source = visible.source_id;
    pause(false);
    const sequence = ++actionSequence.current;
    setOlderBusy(true); setPageError(null); setCopyState("idle");
    const valid = () => mounted.current && sequence === actionSequence.current && !followingRef.current
      && latest.current.connected && latest.current.snapshot?.status === "ready" && latest.current.snapshot.source_id === source;
    try {
      const value = await onLoadOlder(cursor);
      if (!valid()) return;
      if (value.status !== "ready" || value.source_id !== source) throw new Error("[stream:changed]");
      scrollTarget.current = "top";
      setVisible(value); setOlderPage(true); setSelectedMatch(0); setJump(value => value + 1);
    } catch (reason) {
      if (valid()) setPageError(streamMessage(reason, t));
    } finally {
      if (mounted.current && sequence === actionSequence.current) setOlderBusy(false);
    }
  }
  async function copySnippet() {
    if (!text || copyPending.current) return;
    pause(); copyPending.current = true; setCopyState("copying");
    try { const copied = await onCopy(text); if (mounted.current) setCopyState(copied ? "copied" : "failed"); }
    catch { if (mounted.current) setCopyState("failed"); }
    finally { copyPending.current = false; }
  }
  async function saveFullLog() {
    if (!connected || exportBusy || savePending.current || snapshot?.status !== "ready") return;
    savePending.current = true; setSaving(true); setExportError(null);
    try { await onSaveFullLog(); }
    catch (reason) { if (mounted.current) setExportError(exportMessage(reason, t)); }
    finally { savePending.current = false; if (mounted.current) setSaving(false); }
  }

  const chunks = [];
  let position = 0;
  for (let index = 0; index < matches.values.length; index++) {
    const match = matches.values[index];
    if (match.start > position) chunks.push(text.slice(position, match.start));
    chunks.push(<mark key={`${match.start}-${match.end}`} data-log-match={index} className={index === currentMatch ? "is-current" : ""}>{text.slice(match.start, match.end)}</mark>);
    position = match.end;
  }
  if (position < text.length) chunks.push(text.slice(position));

  return <section className="run-log-reader" ref={root} aria-label={t("日志阅读器", "Log reader")} data-run-id={runId} data-following={following} data-older-page={olderPage}>
    <div className="run-log-toolbar">
      <div className="run-log-heading"><Terminal size={15} aria-hidden="true" /><strong>{olderPage ? t("较早的运行日志", "Earlier run log") : t("运行日志", "Run log")}</strong>
        <span className={`run-log-follow-state ${following && connected ? "is-following" : ""}`} role="status">
          {following ? connected ? active ? t("自动跟随", "Following") : t("已到最新", "At latest") : t("等待连接", "Waiting for connection") : t("已暂停跟随", "Following paused")}
        </span>
      </div>
      <div className="run-log-actions" role="group" aria-label={t("日志操作", "Log actions")}>
        <button type="button" className="run-log-action" onClick={() => following ? pause() : followLatest()} aria-pressed={following}>
          {following ? <Pause size={14} aria-hidden="true" /> : <ArrowDown size={14} aria-hidden="true" />}{following ? t("暂停跟随", "Pause following") : t("跟随最新", "Follow latest")}
        </button>
        <IconButton label={t("搜索当前日志片段", "Search current log window")} active={searchOpen} onClick={openSearch}><Search size={15} /></IconButton>
        <button type="button" className={`icon-button ${wrap ? "active" : ""}`} aria-label={t("自动换行", "Wrap lines")} title={t("自动换行", "Wrap lines")} aria-pressed={wrap} onClick={() => { invalidateOlder(); setWrap(value => !value); }}><WrapText size={15} aria-hidden="true" /></button>
        <IconButton label={t("刷新日志", "Refresh log")} disabled={!connected || loading} onClick={() => { invalidateOlder(); setPageError(null); onRefresh(); }}>{loading ? <Spinner size={15} /> : <RefreshCw size={15} />}</IconButton>
      </div>
    </div>
    {searchOpen && <div className="run-log-search-row">
      <div className="run-log-search"><Search size={14} aria-hidden="true" /><input ref={search} id={searchId} type="search" maxLength={256} value={query} aria-label={t("在当前片段中搜索", "Search within current window")} aria-describedby={scopeId}
        placeholder={t("在当前片段中搜索", "Search within current window")} onFocus={() => pause()} onChange={event => { pause(); setQuery(event.target.value); setSelectedMatch(0); }}
        onKeyDown={event => {
          if (event.nativeEvent.isComposing) return;
          if (event.key === "Enter") { event.preventDefault(); event.stopPropagation(); changeMatch(event.shiftKey ? -1 : 1); }
          if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); closeSearch(); }
        }} /></div>
      <span className="run-log-match-count" role="status">{query ? matches.values.length ? `${currentMatch + 1} / ${matches.values.length}${matches.capped ? "+" : ""}` : t("无匹配", "No matches") : t("当前片段", "Current window")}</span>
      <IconButton label={t("上一个匹配", "Previous match")} disabled={!matches.values.length} onClick={() => changeMatch(-1)}><ChevronUp size={16} /></IconButton>
      <IconButton label={t("下一个匹配", "Next match")} disabled={!matches.values.length} onClick={() => changeMatch(1)}><ChevronDown size={16} /></IconButton>
      <IconButton label={t("关闭日志搜索", "Close log search")} onClick={closeSearch}><X size={15} /></IconButton>
    </div>}
    <div className="run-log-scope" id={scopeId}>
      <span>{visible?.status === "ready" ? visible.start === 0 && visible.end === visible.source_size && !clipped
        ? t(`已载入全部 ${formatBytes(visible.source_size)} 日志，可在下方滚动查看。`, `All ${formatBytes(visible.source_size)} of the log is loaded. Scroll below to read it.`)
        : t(`已载入 ${formatBytes(Math.max(0, visible.end - visible.start))} / 共 ${formatBytes(visible.source_size)}；可查看更早内容或保存完整日志。`, `${formatBytes(Math.max(0, visible.end - visible.start))} loaded of ${formatBytes(visible.source_size)}. Read earlier output or save the full log.`)
        : t("日志会在运行时持续写入。", "Output is written as the experiment runs.")}</span>
      {observedTime && <time dateTime={visible?.observed_at}>{t("读取于 ", "Read at ")}{observedTime}</time>}
    </div>
    {(sourceChanged || sourceNotice) && <div className="run-log-notice" role="status">{sourceChanged ? t("日志来源已更新，仍显示此前片段。点击「跟随最新」查看最新内容。", "The log source changed. The earlier window is still shown. Choose Follow latest to read the current source.") : t("日志来源已更新，当前显示新来源的片段。", "The log source changed. A window from the current source is shown.")}</div>}
    {!connected && <div className="run-log-notice" role="status">{text ? t("本地服务未连接，保留上次读取的内容。", "The local service is disconnected. Previously read text is kept.") : t("等待本地服务连接后读取日志。", "Waiting for the local service to read the log.")}</div>}
    {latestMissing && text && <div className="run-log-notice" role="status">{t("当前日志文件不可用，保留上次读取的内容。", "The current log file is unavailable. Previously read text is kept.")}</div>}
    {(pageError || error) && <div className="run-log-notice is-error" role="status">{pageError || streamMessage(error, t)}<button type="button" disabled={!connected || loading} onClick={() => { invalidateOlder(); setPageError(null); onRefresh(); }}>{t("重试读取", "Retry reading")}</button></div>}
    {exportError && <div className="run-log-notice is-error" role="status">{exportError}</div>}
    {copyState === "failed" && <div className="run-log-notice is-error" role="status">{t("复制失败，请在日志中选择文本后手动复制。", "Copy failed. Select text in the log and copy it manually.")}</div>}
    {visible?.partial_first_line && <div className="run-log-boundary">{t("片段从一行中间开始，首行内容不完整。", "This window starts within a line; its first line is incomplete.")}</div>}
    {visible?.invalid_utf8 && <div className="run-log-boundary">{t("部分字节无法解码为文本，显示为替换字符。保存完整日志可保留原始字节。", "Some bytes could not be decoded and appear as replacement characters. Save the full log to retain the original bytes.")}</div>}
    {clipped && <div className="run-log-boundary">{t("片段超出展示限制，最多显示 262,144 个字符；搜索和复制限于显示部分。请保存完整日志查看其余内容。", "This window exceeds the display limit; at most 262,144 characters are displayed. Search and copy cover the displayed text. Save the full log to read the rest.")}</div>}
    {matches.capped && <div className="run-log-boundary">{t("仅高亮前 1,000 处匹配，请增加关键词以缩小范围。", "Only the first 1,000 matches are highlighted. Use a longer search to narrow the results.")}</div>}
    <div className="run-log-page-actions">
      <button type="button" className="run-log-action" disabled={!canLoadOlder} onClick={() => void loadOlder()}>{olderBusy ? <Spinner size={13} /> : <ArrowUp size={13} aria-hidden="true" />}{olderBusy ? t("读取此前片段…", "Loading earlier window…") : t("查看此前片段", "Earlier window")}</button>
      <span>{visible?.has_older ? t("搜索和复制限于已载入内容。", "Search and copy cover the loaded text.") : visible?.status === "ready" ? t("已载入日志开头", "The start of the log is loaded") : ""}</span>
      <div className="run-log-jump-actions">
        <button type="button" className="run-log-action" disabled={!text} onClick={showWindowStart}><ArrowUpToLine size={13} aria-hidden="true" />{t("跳到片段开头", "Top of window")}</button>
        <button type="button" className="run-log-action" disabled={snapshot?.status !== "ready"} onClick={followLatest}><ArrowDownToLine size={13} aria-hidden="true" />{t("跳到最新", "Jump to latest")}</button>
      </div>
    </div>
    <div ref={viewport} className={`console-log run-log-viewport ${wrap ? "is-wrapped" : ""}`} tabIndex={0} role="region" aria-label={t("日志内容", "Log contents")} aria-describedby={scopeId}
      data-source-id={visible?.source_id || ""} data-revision={visible?.revision || ""} data-stale={showingSavedText} data-start={visible?.start ?? ""} data-end={visible?.end ?? ""}
      onPointerDown={event => { if (event.button === 0) pause(); }} onWheel={event => { if (event.deltaY < 0) pause(); }}
      onScroll={() => { const element = viewport.current; if (element && !programmaticScroll.current && followingRef.current && element.getClientRects().length && element.scrollHeight - element.scrollTop - element.clientHeight > 48) pause(); }}
      onKeyDown={event => { if (["ArrowUp", "PageUp", "Home"].includes(event.key)) pause(); }}>
      <pre ref={body} className="run-log-text">{text ? chunks : <span className="run-log-empty">{loading && !visible ? t("正在读取日志…", "Reading the log…") : !connected ? t("日志暂不可用。", "The log is unavailable.") : error ? t("日志读取失败。", "The log could not be read.") : latestMissing ? t("尚无日志文件。", "No log file yet.") : t("这个日志片段没有文本。", "This log window has no text.")}</span>}</pre>
    </div>
    <div className="run-log-footer">
      <div className="run-log-footer-state" role="status">{!following && pending && !sourceChanged && !latestMissing ? <button type="button" className="run-log-action is-accent" onClick={followLatest}><ArrowDown size={13} aria-hidden="true" />{olderPage ? t("返回最新片段", "Return to latest window") : t("有新日志 · 跟随最新", "Log updated · Follow latest")}</button> : <span>{!following ? t("阅读已暂停跟随，实验不会因此停止。", "Following is paused; this does not stop the experiment.") : active ? t("新内容会自动显示在底部。", "New text appears at the bottom.") : t("显示最近读取的日志片段。", "Showing the most recently read log window.")}</span>}</div>
      <div className="run-log-export-actions">
        <button type="button" className="run-log-action" disabled={!text || copyState === "copying"} onClick={() => void copySnippet()}>{copyState === "copying" ? <Spinner size={13} /> : copyState === "copied" ? <Check size={13} aria-hidden="true" /> : <Copy size={13} aria-hidden="true" />}{copyState === "copied" ? t("片段已复制", "Window copied") : t("复制当前片段", "Copy current window")}</button>
        <button type="button" className="run-log-action" disabled={!connected || snapshot?.status !== "ready" || exportBusy || saving} onClick={() => void saveFullLog()}>{saving ? <Spinner size={13} /> : <ArrowDownToLine size={13} aria-hidden="true" />}{t("保存完整日志", "Save full log")}</button>
      </div>
    </div>
  </section>;
}
