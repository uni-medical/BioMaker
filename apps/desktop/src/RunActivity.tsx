import { useEffect, useRef, useState } from "react";
import { Activity, Check, ChevronRight, RefreshCw, X } from "lucide-react";
import { Notice, Spinner } from "./components";
import { dateText, errorMessage, isActive, type Language, type Run, type Translate } from "./models";
import { eventPresentation } from "./eventPresentation";
import type { ActivityWindow } from "./runStreams";

function EventItem({ event, lang, t }: { event: Record<string, unknown>; lang: Language; t: Translate }) {
  const [open, setOpen] = useState(false);
  const view = eventPresentation(event, t);
  const fields = Object.fromEntries(Object.entries(event).filter(([key]) => !["event", "ts"].includes(key)));
  return <div className={`event-item ${view.failure ? "event-error" : ""}`}>
    <span className="event-node">{view.failure ? <X size={12} /> : view.complete ? <Check size={12} /> : <span />}</span>
    <div className="event-body">
      <button className="event-heading" onClick={() => setOpen(!open)} aria-expanded={open}>
        <strong>{view.title}</strong>
        {view.step && <span className="event-step">{t("步骤", "Step")} {view.step}</span>}
        <time>{dateText(view.timestamp, lang, true)}</time>
        <ChevronRight size={13} className={open ? "rotate-down" : ""} />
      </button>
      {view.summary && <p>{view.summary}</p>}
      {open && <pre className="event-data">{JSON.stringify(fields, null, 2)}</pre>}
    </div>
  </div>;
}

export default function RunActivity({ run, snapshot, error, loading, connected, serviceRevision, lang, t, onLoadOlder, onRefresh }: {
  run: Run; snapshot: ActivityWindow | null; error: string; loading: boolean; connected: boolean;
  serviceRevision: number; lang: Language; t: Translate; onLoadOlder: (cursor: string) => Promise<ActivityWindow>; onRefresh: () => void;
}) {
  const [page, setPage] = useState<ActivityWindow | null>(null);
  const [olderBusy, setOlderBusy] = useState(false);
  const [olderError, setOlderError] = useState("");
  const request = useRef(0);
  useEffect(() => { request.current++; setOlderBusy(false); return () => { request.current++; }; }, [run.id, serviceRevision]);
  const visible = page || snapshot;
  const events = visible?.events || [];
  const latest = snapshot?.events.at(-1)?.data;
  const last = latest ? eventPresentation(latest, t) : null;
  const changed = !!page && !!snapshot && page.source_id !== snapshot.source_id;
  const stale = !connected || !!error;
  async function older() {
    if (!visible?.older_cursor || olderBusy || !connected) return;
    const sequence = ++request.current;
    setOlderBusy(true); setOlderError("");
    try {
      const value = await onLoadOlder(visible.older_cursor);
      if (sequence === request.current) setPage(value);
    } catch (e) {
      if (sequence === request.current) setOlderError(errorMessage(e));
    } finally { if (sequence === request.current) setOlderBusy(false); }
  }
  return <div className="run-activity">
    <div className="activity-context">
      <div><strong>{t("最近记录的活动", "Last recorded activity")}</strong>
        <p>{last ? <>{last.title} · {dateText(last.timestamp, lang, true)}</> : t("尚无记录", "No recorded activity")}</p>
      </div>
      <button className="text-button" onClick={onRefresh} disabled={!connected || loading}><RefreshCw size={13} />{t("刷新活动", "Refresh activity")}</button>
    </div>
    <p className="activity-explanation">{t("活动按记录顺序显示；并行候选可能交错。最近一条记录不代表整个实验的当前阶段。", "Records appear in file order; parallel candidates can interleave. The latest record does not establish the current stage of the whole experiment.")}</p>
    {stale && <Notice danger={!!error}>{t("更新暂停，显示上次读取的记录。", "Updates paused. Showing the last records read.")}</Notice>}
    {(olderError || changed) && <Notice danger>{t("活动文件已变化或暂时无法读取。请返回最新活动后重试。", "Activity changed or could not be read. Return to the latest activity and retry.")}</Notice>}
    <div className="activity-navigation">
      <span>{page ? t("较早片段", "Earlier segment") : t("最新片段", "Latest segment")} · {events.length} {t("条记录", "records")}</span>
      <div>
        {visible?.has_older && <button className="text-button" onClick={() => void older()} disabled={!connected || olderBusy || changed}>{olderBusy && <Spinner size={12} />}{t("查看更早活动", "Earlier activity")}</button>}
        {page && <button className="text-button" onClick={() => { request.current++; setPage(null); setOlderBusy(false); setOlderError(""); onRefresh(); }}>{t("返回最新活动", "Latest activity")}</button>}
      </div>
    </div>
    {events.length ? <div className="timeline">
      {events.map(({ id, data }) => <EventItem key={id} event={data} lang={lang} t={t} />)}
      {!page && isActive(run) && <div className="event-item current"><span className="event-node">{!stale ? <Spinner size={12} /> : <span />}</span><div className="event-body"><span>{stale ? t("等待连接恢复", "Waiting for connection") : run.status === "queued" ? t("等待运行", "Waiting to run") : run.status === "stopping" ? t("正在结束当前进程…", "Ending the current process…") : t("实验仍在运行，等待更多记录", "Experiment running; waiting for more records")}</span></div></div>}
    </div> : <div className="detail-empty"><Activity size={28} /><h3>{loading ? t("正在读取实验活动", "Loading activity") : stale ? t("活动暂时无法读取", "Activity unavailable") : run.status === "queued" ? t("实验已加入队列", "Experiment is queued") : t("尚无活动记录", "No activity recorded yet")}</h3></div>}
    {!!visible && (visible.has_older || visible.partial_first_line || visible.skipped_records > 0 || visible.invalid_utf8) && <p className="tail-note">
      {visible.has_older && t("此片段之前还有活动。", "Earlier activity is available. ")}
      {(visible.partial_first_line || visible.skipped_records > 0 || visible.invalid_utf8) && t("片段中存在不完整或无法解析的记录，已跳过；不能据此推断实验成功或失败。", "Incomplete or unreadable records were skipped; they do not establish success or failure.")}
    </p>}
  </div>;
}
