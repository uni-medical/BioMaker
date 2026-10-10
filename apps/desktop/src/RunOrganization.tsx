import { useEffect, useRef, useState } from "react";
import { Archive, ArchiveRestore, Pencil } from "lucide-react";
import { Modal, Notice, Spinner } from "./components";
import { errorMessage, request, runTitle, type Run, type Translate } from "./models";
import { canArchive, type RunMetadata } from "./runMetadata";

export default function RunOrganization({ run, metadata, ready, serviceRevision, t, onChange, toast }: {
  run: Run;
  metadata: RunMetadata;
  ready: boolean;
  serviceRevision: number;
  t: Translate;
  onChange: (row: RunMetadata) => void;
  toast: (message: string) => void;
}) {
  const [rename, setRename] = useState(false);
  const [name, setName] = useState("");
  const [expectedRevision, setExpectedRevision] = useState(metadata.revision);
  const [pending, setPending] = useState("");
  const [error, setError] = useState("");
  const state = useRef({ ready, serviceRevision, mounted: true });
  state.current.ready = ready;
  state.current.serviceRevision = serviceRevision;
  useEffect(() => {
    state.current.mounted = true;
    return () => { state.current.mounted = false; };
  }, []);
  const path = `/api/desktop/run-metadata/${encodeURIComponent(run.id)}`;
  async function save(changes: { display_name?: string | null; archived?: boolean }, revision: number, action: string) {
    if (!ready || pending) return;
    setPending(action);
    setError("");
    const current = () => state.current.mounted && state.current.ready && state.current.serviceRevision === serviceRevision;
    try {
      const row = await request<RunMetadata>(path, "PUT", { expected_revision: revision, ...changes });
      if (!current()) return;
      onChange(row);
      if (action === "rename") setRename(false);
      toast(action === "rename" ? t("显示名称已保存", "Display name saved") : row.archived ? t("实验已归档，可在全部实验中恢复", "Experiment archived. Restore it from All experiments.") : t("实验已恢复", "Experiment restored"));
    } catch (e) {
      if (!current()) return;
      let message = errorMessage(e);
      // Recover the current revision on any failed PUT. In particular, a CAS
      // conflict must preserve the user's input and make the next save deliberate.
      try {
        const latest = await request<RunMetadata>(path);
        if (!current()) return;
        onChange(latest);
        if (action === "rename") setExpectedRevision(latest.revision);
        if (latest.revision !== revision) message = t(
          `此实验已在其他位置更新。最新名称：${runTitle({ ...run, display_name: latest.display_name })}。你的输入已保留，请检查你的输入后再次保存。`,
          `This experiment was updated elsewhere. Latest name: ${runTitle({ ...run, display_name: latest.display_name })}. Your input is preserved. Review it, then save again.`,
        );
      } catch { /* Preserve the original failure and input for a manual retry. */ }
      if (!current()) return;
      if (action === "rename") setError(message);
      else toast(message);
    } finally {
      if (state.current.mounted) setPending("");
    }
  }
  const invalidName = /[\p{Cc}\p{Cs}\p{Zl}\p{Zp}]/u.test(name) || Array.from(name.trim()).length > 120;
  const originalTitle = runTitle({ ...run, display_name: null });
  return <>
    <button className="secondary-button" disabled={!ready || !!pending} onClick={() => {
      setName(metadata.display_name ?? ""); setExpectedRevision(metadata.revision); setError(""); setRename(true);
    }}><Pencil size={13} />{t("重命名", "Rename")}</button>
    <button className="secondary-button" disabled={!ready || !!pending || (!metadata.archived && !canArchive(run))}
      title={!metadata.archived && !canArchive(run) ? t("实验结束并完成清理后可以归档", "Archive after the experiment and cleanup finish") : undefined}
      onClick={() => void save({ archived: !metadata.archived }, metadata.revision, "archive")}>
      {pending === "archive" ? <Spinner size={13} /> : metadata.archived ? <ArchiveRestore size={13} /> : <Archive size={13} />}
      {metadata.archived ? t("恢复实验", "Restore experiment") : t("归档实验", "Archive experiment")}
    </button>
    {rename && <Modal title={t("重命名实验", "Rename experiment")} onClose={() => { if (!pending) setRename(false); }} t={t} className="confirm-modal rename-modal">
      <form onSubmit={event => { event.preventDefault(); if (!invalidName) void save({ display_name: name.trim() || null }, expectedRevision, "rename"); }}>
        <p>{t("设置便于查找的显示名称，原始名称与实验文件会保留。", "Choose a display name for easier browsing. The original name and experiment files are kept.")}</p>
        <label className="field"><span>{t("显示名称", "Display name")}</span>
          <input autoFocus aria-label={t("显示名称", "Display name")} value={name} placeholder={originalTitle} disabled={!!pending} onChange={event => { setName(event.target.value); setError(""); }} />
          <small>{t("最多 120 个字符，留空使用原名称。", "Up to 120 characters. Leave blank to use the original name.")}</small>
        </label>
        <p className="organization-original">{t("原名称：", "Original name: ")}{originalTitle}</p>
        {invalidName && <Notice danger>{t("显示名称最多 120 个字符，不能含换行或控制字符。", "Use up to 120 characters, without line breaks or control characters.")}</Notice>}
        {error && <Notice danger>{error}</Notice>}
        <div className="confirm-actions">
          <button type="button" className="text-button" disabled={!!pending} onClick={() => { setName(""); setError(""); }}>{t("使用原名称", "Use original name")}</button>
          <button type="button" className="secondary-button" disabled={!!pending} onClick={() => setRename(false)}>{t("取消", "Cancel")}</button>
          <button type="submit" className="primary-button" disabled={!ready || !!pending || invalidName}>{pending === "rename" && <Spinner size={13} />}{t("保存名称", "Save name")}</button>
        </div>
      </form>
    </Modal>}
  </>;
}
