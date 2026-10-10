import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Copy, FilePenLine, Folder, Pencil, Plus, Search, Trash2, X } from "lucide-react";
import { IconButton, Modal, Spinner } from "./components";
import type { Draft } from "./experimentDraft";
import { basename, type Language, type Translate } from "./models";
import "./draft-library.css";

export interface DraftEntry {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  content: Draft;
}

export interface DraftLibraryProps {
  items: DraftEntry[];
  activeId: string | null;
  lang: Language;
  t: Translate;
  busy: boolean;
  onOpen: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => Promise<boolean>;
  onDuplicate: (id: string) => Promise<boolean>;
  onDelete: (id: string) => Promise<boolean>;
}

export function draftEntryTitle(entry: DraftEntry, t: Translate): string {
  const explicit = entry.title.trim();
  if (explicit) return explicit;
  const firstLine = entry.content.task.split(/\r?\n/).map(line => line.trim()).find(Boolean);
  return firstLine ? Array.from(firstLine).slice(0, 80).join("") : t("未命名草稿", "Untitled draft");
}

function updatedTime(entry: DraftEntry): number {
  const updated = Date.parse(entry.updatedAt);
  if (Number.isFinite(updated)) return updated;
  const created = Date.parse(entry.createdAt);
  return Number.isFinite(created) ? created : 0;
}

export default function DraftLibrary({
  items, activeId, lang, t, busy, onOpen, onNew, onRename, onDuplicate, onDelete,
}: DraftLibraryProps) {
  const [query, setQuery] = useState("");
  const [renaming, setRenaming] = useState<DraftEntry | null>(null);
  const [title, setTitle] = useState("");
  const [deleting, setDeleting] = useState<DraftEntry | null>(null);
  const [pending, setPending] = useState<{ id: string; action: "rename" | "duplicate" | "delete" } | null>(null);
  const mutationInFlight = useRef(false);
  const mounted = useRef(true);
  const titleInputId = useId();
  const titleHintId = useId();
  const titleCountId = useId();
  const disabled = busy || !!pending;
  const normalizedQuery = query.trim().toLocaleLowerCase(lang === "zh" ? "zh-CN" : "en-US");
  const visible = useMemo(() => items.filter(entry => {
    if (!normalizedQuery) return true;
    return [draftEntryTitle(entry, t), entry.content.task, entry.content.dataset]
      .some(value => value.toLocaleLowerCase(lang === "zh" ? "zh-CN" : "en-US").includes(normalizedQuery));
  }).sort((left, right) => updatedTime(right) - updatedTime(left) || left.id.localeCompare(right.id)), [items, normalizedQuery, lang, t]);
  const dates = useMemo(() => new Intl.DateTimeFormat(lang === "zh" ? "zh-CN" : "en-US", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  }), [lang]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  async function mutate(entry: DraftEntry, action: "rename" | "duplicate" | "delete") {
    if (busy || mutationInFlight.current) return;
    mutationInFlight.current = true;
    setPending({ id: entry.id, action });
    try {
      const succeeded = await (action === "rename" ? onRename(entry.id, title.trim())
        : action === "duplicate" ? onDuplicate(entry.id) : onDelete(entry.id));
      if (succeeded && mounted.current) {
        if (action === "rename") setRenaming(null);
        if (action === "delete") setDeleting(null);
      }
    } catch {
      // The parent owns persistence errors and user-facing recovery messages.
      // Keep modal input intact if a callback unexpectedly rejects.
    } finally {
      mutationInFlight.current = false;
      if (mounted.current) setPending(null);
    }
  }

  const closeRename = () => { if (!busy && !mutationInFlight.current) setRenaming(null); };
  const closeDelete = () => { if (!busy && !mutationInFlight.current) setDeleting(null); };

  return <section className="draft-library" aria-label={t("草稿", "Drafts")} aria-busy={disabled}>
    <div className="draft-library-heading">
      <div>
        <h1>{t("草稿", "Drafts")}</h1>
        <p>{t("任务和设置保存在本机，准备好后再开始实验。", "Keep tasks and settings on this computer. Start an experiment when you are ready.")}</p>
      </div>
      <button type="button" className="primary-button" disabled={disabled} onClick={onNew}>
        <Plus size={15} aria-hidden="true" />{t("新建实验", "New experiment")}
      </button>
    </div>

    {items.length > 0 && <div className="draft-library-toolbar">
      <div className="search-input draft-library-search">
        <Search size={16} aria-hidden="true" />
        <input type="search" value={query} onChange={event => setQuery(event.target.value)}
          placeholder={t("搜索标题、任务或数据集", "Search titles, tasks, or datasets")}
          aria-label={t("搜索草稿", "Search drafts")} />
        {query && <IconButton label={t("清除草稿搜索", "Clear draft search")} onClick={() => setQuery("")}><X size={14} /></IconButton>}
      </div>
      <span className="draft-library-count" role="status">
        {normalizedQuery ? t(`${visible.length} / ${items.length} 份草稿`, `${visible.length} of ${items.length} drafts`)
          : t(`${items.length} 份草稿`, `${items.length} ${items.length === 1 ? "draft" : "drafts"}`)}
      </span>
    </div>}

    {visible.length > 0 ? <ul className="draft-library-list" aria-label={t("草稿列表", "Draft list")}>
      {visible.map(entry => {
        const displayTitle = draftEntryTitle(entry, t);
        const timestamp = updatedTime(entry);
        const date = timestamp ? new Date(timestamp) : null;
        const summary = entry.content.task.trim().replace(/\s+/g, " ");
        return <li className={`draft-library-row ${entry.id === activeId ? "is-current" : ""}`} key={entry.id} data-draft-id={entry.id}>
          <button type="button" className="draft-library-open" disabled={disabled}
            onClick={() => onOpen(entry.id)} aria-label={t(`继续编辑草稿：${displayTitle}`, `Continue editing draft: ${displayTitle}`)}>
            <FilePenLine className="draft-library-icon" size={19} aria-hidden="true" />
            <span className="draft-library-content">
              <span className="draft-library-title"><strong title={displayTitle}>{displayTitle}</strong>
                {entry.id === activeId && <span className="draft-library-current">{t("当前", "Current")}</span>}
              </span>
              <span className="draft-library-summary">{summary || t("尚未填写任务描述", "No task description yet")}</span>
              <span className="draft-library-meta">
                <span className="draft-library-dataset" title={entry.content.dataset || undefined}><Folder size={12} aria-hidden="true" />
                  <span>{entry.content.dataset ? basename(entry.content.dataset) : t("未选择数据集", "No dataset selected")}</span>
                </span>
                {date && <time dateTime={date.toISOString()} title={date.toLocaleString(lang === "zh" ? "zh-CN" : "en-US")}>
                  {t("修改于 ", "Updated ")}{dates.format(date)}
                </time>}
              </span>
            </span>
          </button>
          <div className="draft-library-actions" role="group" aria-label={t(`草稿操作：${displayTitle}`, `Draft actions: ${displayTitle}`)}>
            <IconButton disabled={disabled} label={t(`重命名草稿：${displayTitle}`, `Rename draft: ${displayTitle}`)} onClick={() => {
              setTitle(entry.title); setRenaming(entry);
            }}><Pencil size={15} /></IconButton>
            <IconButton disabled={disabled} label={t(`复制草稿：${displayTitle}`, `Duplicate draft: ${displayTitle}`)} onClick={() => void mutate(entry, "duplicate")}>
              {pending?.id === entry.id && pending.action === "duplicate" ? <Spinner size={15} /> : <Copy size={15} />}
            </IconButton>
            <IconButton disabled={disabled} className="draft-library-delete" label={t(`删除草稿：${displayTitle}`, `Delete draft: ${displayTitle}`)} onClick={() => setDeleting(entry)}><Trash2 size={15} /></IconButton>
          </div>
        </li>;
      })}
    </ul> : <div className="draft-library-empty">
      {items.length ? <Search size={28} aria-hidden="true" /> : <FilePenLine size={28} aria-hidden="true" />}
      <h2>{items.length ? t("没有匹配的草稿", "No matching drafts") : t("还没有草稿", "No drafts yet")}</h2>
      <p>{items.length ? t("试试其它关键词，或清除搜索。", "Try another keyword or clear the search.")
        : t("先写下任务，选择数据集。你可以随时回来继续编辑。", "Describe a task and choose a dataset. Come back to continue editing whenever you need.")}</p>
      {items.length ? <button type="button" className="secondary-button" onClick={() => setQuery("")}>{t("清除搜索", "Clear search")}</button>
        : <button type="button" className="secondary-button" disabled={disabled} onClick={onNew}><Plus size={15} aria-hidden="true" />{t("新建实验", "New experiment")}</button>}
    </div>}

    {renaming && <Modal title={t("重命名草稿", "Rename draft")} onClose={closeRename} t={t} className="draft-library-modal">
      <form className="draft-library-rename" onSubmit={event => { event.preventDefault(); void mutate(renaming, "rename"); }} aria-busy={disabled}>
        <label htmlFor={titleInputId}>{t("草稿标题", "Draft title")}</label>
        <input id={titleInputId} value={title} disabled={disabled} autoFocus data-modal-autofocus
          onChange={event => setTitle(Array.from(event.target.value).slice(0, 120).join(""))}
          placeholder={draftEntryTitle({ ...renaming, title: "" }, t)}
          aria-describedby={`${titleHintId} ${titleCountId}`} />
        <div className="draft-library-title-help"><p id={titleHintId}>{t("留空时按任务内容命名。不会更改实验目录名称。", "Leave blank to use the task description. The experiment folder name stays unchanged.")}</p>
          <span id={titleCountId}>{Array.from(title).length} / 120</span>
        </div>
        <div className="draft-library-modal-actions">
          <button type="button" className="secondary-button" disabled={disabled} onClick={closeRename}>{t("取消", "Cancel")}</button>
          <button type="submit" className="primary-button" disabled={disabled}>
            {pending?.action === "rename" && <Spinner size={14} />}{t("保存标题", "Save title")}
          </button>
        </div>
      </form>
    </Modal>}

    {deleting && <Modal title={t("删除这份草稿？", "Delete this draft?")} onClose={closeDelete} t={t} className="draft-library-modal">
      <div className="draft-library-delete-body">
        <strong>{draftEntryTitle(deleting, t)}</strong>
        <p>{t("只删除这份未提交的草稿，不会删除已有实验、数据集或磁盘文件。删除后无法恢复。", "This deletes only this unsubmitted draft. Existing experiments, datasets, and files on disk are kept. This cannot be undone.")}</p>
        <div className="draft-library-modal-actions">
          <button type="button" className="secondary-button" disabled={disabled} autoFocus data-modal-autofocus onClick={closeDelete}>{t("保留草稿", "Keep draft")}</button>
          <button type="button" className="danger-button" disabled={disabled} onClick={() => void mutate(deleting, "delete")}>
            {pending?.action === "delete" && <Spinner size={14} />}{t("删除草稿", "Delete draft")}
          </button>
        </div>
      </div>
    </Modal>}
  </section>;
}
