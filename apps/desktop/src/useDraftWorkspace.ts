import { useCallback, useEffect, useRef, useState } from 'react';
import type { DraftMutation, DraftOperation, DraftPayload, DraftSnapshot } from '../electron/drafts';
import { registerCloseProtection } from './closeProtection';
import { emptyDraft, hasDraft, type Draft } from './experimentDraft';
import type { DraftEntry } from './DraftLibrary';
import type { Translate } from './models';

const freshDraft = () => structuredClone(emptyDraft);
function toStored(draft: Draft): DraftPayload {
  // A cleared numeric input is NaN in the form and null in the saved document.
  // Drafts preserve incomplete values; the experiment submission validates them.
  return JSON.parse(JSON.stringify(draft)) as DraftPayload;
}
function fromStored(content: DraftPayload): Draft {
  const options = { ...content.options };
  for (const key of ['agent_steps', 'exec_timeout', 'exec_parallel_workers', 'agent_total_time_limit'] as const) {
    if (options[key] === null) options[key] = Number.NaN;
  }
  return { ...content, options } as Draft;
}
function equalContent(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object') return false;
  const a = left as Record<string, unknown>, b = right as Record<string, unknown>;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every(key => Object.hasOwn(b, key) && equalContent(a[key], b[key]));
}
export function draftStorageMessage(error: unknown, t: Translate): string {
  const code = /\[drafts:([a-z_]+)\]/.exec(String(error))?.[1];
  switch (code) {
    case 'conflict': return t('本机保存的草稿已变化，当前编辑仍保留。可以将当前内容另存为新草稿。', 'The saved drafts changed. Your edits are retained; save this content as a new draft.');
    case 'corrupt': return t('草稿文件无法读取，原文件未被覆盖。请保留当前内容后检查本机存储。', 'The draft file could not be read and was not overwritten. Keep your current content and check local storage.');
    case 'unsupported': return t('草稿来自更新的版本，当前版本无法修改。请使用相应版本打开。', 'These drafts use a newer format. Open them with the corresponding application version.');
    case 'too_large': return t('草稿数量或内容超过本机保存上限，请缩短内容或整理已有草稿。', 'The draft count or content exceeds the storage limit. Shorten the content or organize existing drafts.');
    case 'invalid_request': return t('部分草稿内容无法保存。当前编辑仍保留，原文件未被覆盖。', 'Some draft content could not be saved. Your edits are retained and the original file was not overwritten.');
    case 'save_failed': return t('未能确认草稿是否保存成功，当前编辑仍保留。请重试以确认保存状态。', 'The draft save could not be confirmed. Your edits are retained. Retry to confirm the saved state.');
    default: return t('无法确认本机草稿的保存状态。请检查存储权限和可用空间后重试，当前编辑仍保留。', 'The local draft state could not be confirmed. Check storage access and free space, then retry. Your edits are retained.');
  }
}

/** One editor owns a draft at a time. A save acknowledgment updates the disk
 * snapshot without replacing newer input. Document changes flush before moving
 * ownership, and the native close path synchronously persists the last edit. */
export function useDraftWorkspace(t: Translate) {
  const [draft, setDraft] = useState<Draft>(freshDraft);
  const [snapshot, setSnapshot] = useState<DraftSnapshot | null>(null);
  const [ready, setReady] = useState(false);
  const [saving, setSaving] = useState(false);
  const [operating, setOperating] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState('');
  const [legacyRaw, setLegacyRaw] = useState<string | null>(null);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const latest = useRef({ draft: freshDraft(), snapshot: null as DraftSnapshot | null, epoch: 0, dirty: false, operating: false, ready: false });
  const translate = useRef(t); translate.current = t;
  const mounted = useRef(true);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pending = useRef<Promise<boolean> | null>(null);
  const unacknowledged = useRef<{ input: DraftMutation; epoch: number } | null>(null);
  const recovery = useRef<{ input: DraftMutation; content: DraftPayload; knownIds: Set<string>; epoch: number } | null>(null);
  const initialization = useRef<Promise<DraftSnapshot> | null>(null);
  const loadGeneration = useRef(0);
  const flushRef = useRef<() => Promise<boolean>>(async () => false);
  const reportError = (failure: unknown) => { if (mounted.current) setError(draftStorageMessage(failure, translate.current)); };
  const stopTimer = () => { if (timer.current) { clearTimeout(timer.current); timer.current = null; } };
  const mutation = (operation: DraftOperation): DraftMutation => {
    if (!latest.current.snapshot) throw new Error('[drafts:unavailable]');
    return { expectedRevision: latest.current.snapshot.revision, mutationId: crypto.randomUUID(), operation };
  };
  const acknowledgeUpdate = (value: DraftSnapshot, input: DraftMutation) => {
    const operation = input.operation;
    if (operation.type !== 'update' || value.activeId !== operation.id || latest.current.snapshot?.activeId !== operation.id ||
        !equalContent(value.drafts.find(item => item.id === operation.id)?.content, operation.content)) {
      // An idempotent replay returns the current document, which may have been
      // edited since the original save. It must not transfer editor ownership.
      throw new Error('[drafts:conflict]');
    }
    apply(value, false);
  };
  const apply = (value: DraftSnapshot, replaceEditor: boolean) => {
    latest.current.snapshot = value;
    if (mounted.current) setSnapshot(value);
    if (replaceEditor) {
      unacknowledged.current = null;
      recovery.current = null;
      const item = value.drafts.find(item => item.id === value.activeId);
      const content = item ? fromStored(item.content) : freshDraft();
      latest.current.draft = content;
      latest.current.epoch++;
      latest.current.dirty = false;
      if (mounted.current) { setDraft(content); setDirty(false); }
    }
  };

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; stopTimer(); };
  }, []);
  useEffect(() => {
    let valid = true;
    const generation = ++loadGeneration.current;
    if (!window.biomaker) return;
    if (!initialization.current) initialization.current = (async () => {
      let value = await window.biomaker.getDrafts();
      const initializing = !value.legacyMigration;
      if (!value.legacyMigration) {
        const raw = localStorage.getItem('biomaker.draft');
        if (mounted.current) setLegacyRaw(raw);
        value = await window.biomaker.mutateDrafts({ expectedRevision: value.revision, mutationId: crypto.randomUUID(), operation: { type: 'migrate', raw } });
      }
      // Only the initial empty workspace needs a blank composer. An existing
      // library with no active draft remains a library on the next launch.
      if (initializing && value.drafts.length === 0) {
        value = await window.biomaker.mutateDrafts({ expectedRevision: value.revision, mutationId: crypto.randomUUID(), operation: { type: 'create', content: toStored(freshDraft()) } });
      }
      return value;
    })();
    void initialization.current.then(value => {
      if (!valid || generation !== loadGeneration.current) return;
      apply(value, true); latest.current.ready = true; setReady(true); setError('');
    }).catch(failure => { if (valid && generation === loadGeneration.current) reportError(failure); });
    return () => { valid = false; };
  }, [loadAttempt]);

  const flush = useCallback(async (): Promise<boolean> => {
    stopTimer();
    if (pending.current) {
      const ok = await pending.current;
      return ok ? flushRef.current() : false;
    }
    const state = latest.current;
    if (!state.ready) return false;
    if (!state.dirty) return true;
    if (!state.snapshot?.activeId) return false;
    const attempt = unacknowledged.current || { epoch: state.epoch,
      input: mutation({ type: 'update', id: state.snapshot.activeId, content: toStored(state.draft) }) };
    unacknowledged.current = attempt;
    const { epoch, input } = attempt;
    setSaving(true);
    // Start after pending is assigned, including a synchronous bridge failure.
    const operation = Promise.resolve().then(async () => {
      try {
        const value = await window.biomaker.mutateDrafts(input);
        acknowledgeUpdate(value, input);
        unacknowledged.current = null;
        if (latest.current.epoch === epoch) {
          latest.current.dirty = false;
          if (mounted.current) setDirty(false);
        }
        if (mounted.current) setError('');
        return true;
      } catch (failure) {
        // Rejected content never reached disk; allow a corrected edit to form
        // a new request. Ambiguous failures retain their idempotency key.
        if (/\[drafts:(invalid_request|too_large)\]/.test(String(failure))) unacknowledged.current = null;
        reportError(failure); return false;
      }
      finally { pending.current = null; if (mounted.current) setSaving(false); }
    });
    pending.current = operation;
    const ok = await operation;
    return ok && latest.current.dirty ? flushRef.current() : ok;
  }, []);
  flushRef.current = flush;

  const update = useCallback((updater: (current: Draft) => Draft) => {
    const state = latest.current;
    if (!state.ready || state.operating || !state.snapshot?.activeId) return;
    const next = updater(state.draft);
    state.draft = next; state.epoch++; state.dirty = true;
    setDraft(next); setDirty(true);
    stopTimer();
    timer.current = setTimeout(() => { void flushRef.current(); }, 350);
  }, []);

  useEffect(() => registerCloseProtection('drafts', () => {
    if (pending.current || latest.current.operating) return 'saving';
    if (!latest.current.dirty) return null;
    const state = latest.current;
    if (!state.snapshot?.activeId) return 'unsaved';
    stopTimer();
    try {
      const save = (input: DraftMutation) => {
        const response = window.biomaker.flushDrafts(input);
        if (!response.ok) throw new Error(`[drafts:${response.code}]`);
        acknowledgeUpdate(response.snapshot, input);
      };
      const prior = unacknowledged.current;
      if (prior) { save(prior.input); unacknowledged.current = null; }
      if (!prior || prior.epoch !== state.epoch) {
        const input = mutation({ type: 'update', id: state.snapshot!.activeId!, content: toStored(state.draft) });
        unacknowledged.current = { input, epoch: state.epoch };
        save(input); unacknowledged.current = null;
      }
      state.dirty = false;
      if (mounted.current) { setDirty(false); setError(''); }
      return null;
    } catch (failure) {
      if (/\[drafts:(invalid_request|too_large)\]/.test(String(failure))) unacknowledged.current = null;
      reportError(failure); return 'unsaved';
    }
  }), []);

  const act = useCallback(async (operation: () => DraftOperation): Promise<boolean> => {
    if (latest.current.operating || !latest.current.ready) return false;
    latest.current.operating = true; setOperating(true);
    try {
      if (!await flushRef.current()) return false;
      const value = await window.biomaker.mutateDrafts(mutation(operation()));
      apply(value, true); setError(''); return true;
    } catch (failure) { reportError(failure); return false; }
    finally { latest.current.operating = false; if (mounted.current) setOperating(false); }
  }, []);

  const create = useCallback(async (content?: Draft, title?: string): Promise<boolean> => {
    // Repeated New on a pristine composer should not fill the library with
    // empty documents. Explicit reuse/duplicate always creates a document.
    const state = latest.current;
    if (!state.ready || state.operating) return false;
    const active = state.snapshot?.drafts.find(item => item.id === state.snapshot?.activeId);
    if (!content && active && !active.title && !hasDraft(state.draft)) return flushRef.current();
    return act(() => ({ type: 'create', content: toStored(content || freshDraft()), ...(title ? { title } : {}) }));
  }, [act]);
  const activate = useCallback((id: string) => {
    const state = latest.current;
    // Returning to the same editor must remain possible when its contents are
    // too large to save; the user needs the form to shorten or copy that input.
    if (state.ready && !state.operating && state.snapshot?.activeId === id) return Promise.resolve(true);
    return act(() => ({ type: 'activate', id }));
  }, [act]);
  const rename = useCallback((id: string, title: string) => act(() => ({ type: 'rename', id, title })), [act]);
  const remove = useCallback((id: string) => act(() => ({ type: 'delete', id })), [act]);
  const duplicate = useCallback((id: string) => act(() => {
    const item = latest.current.snapshot?.drafts.find(item => item.id === id);
    if (!item) throw new Error('[drafts:conflict]');
    return { type: 'create', content: item.content, title: item.title ? `${Array.from(item.title).slice(0, 112).join('')} ${translate.current('副本', 'copy')}` : '' };
  }), [act]);
  const recoverAsCopy = useCallback(async (): Promise<boolean> => {
    if (latest.current.operating || pending.current || !latest.current.dirty) return false;
    latest.current.operating = true; setOperating(true); stopTimer();
    try {
      if (!recovery.current) {
        const content = toStored(latest.current.draft);
        const current = await window.biomaker.getDrafts();
        recovery.current = { content, knownIds: new Set(current.drafts.map(item => item.id)), epoch: latest.current.epoch,
          input: { expectedRevision: current.revision, mutationId: crypto.randomUUID(), operation: { type: 'create', content, title: translate.current('恢复的草稿', 'Recovered draft') } } };
      }
      const attempt = recovery.current;
      const value = await window.biomaker.mutateDrafts(attempt.input).catch(failure => {
        // Native CAS/validation refusal has no write; a fresh explicit recovery
        // may take the now-current revision. Ambiguous acknowledgments keep ID.
        if (/\[drafts:(conflict|invalid_request|too_large)\]/.test(String(failure))) recovery.current = null;
        throw failure;
      });
      const matches = value.drafts.filter(item => !attempt.knownIds.has(item.id) && equalContent(item.content, attempt.content));
      if (matches.length !== 1 || matches[0].id !== value.activeId) throw new Error('[drafts:conflict]');
      recovery.current = null; unacknowledged.current = null;
      if (latest.current.epoch === attempt.epoch) apply(value, true);
      else {
        // The recovery may have committed before its acknowledgment failed.
        // Retry it once, then save edits made after that attempt into its copy.
        apply(value, false);
        if (!await flushRef.current()) return false;
      }
      setError(''); return true;
    } catch (failure) {
      if (/\[drafts:(invalid_request|too_large)\]/.test(String(failure))) recovery.current = null;
      reportError(failure); return false;
    }
    finally { latest.current.operating = false; if (mounted.current) setOperating(false); }
  }, []);
  const retry = useCallback(async () => {
    if (latest.current.operating || pending.current) return;
    if (latest.current.ready) {
      if (latest.current.dirty) { await flushRef.current(); return; }
      latest.current.operating = true; setOperating(true);
      try { apply(await window.biomaker.getDrafts(), true); setError(''); }
      catch (failure) { reportError(failure); }
      finally { latest.current.operating = false; if (mounted.current) setOperating(false); }
      return;
    }
    initialization.current = null; setLoadAttempt(value => value + 1);
  }, []);
  const items: DraftEntry[] = (snapshot?.drafts || []).map(item => ({ ...item, content: item.id === snapshot?.activeId ? draft : fromStored(item.content) }));
  const currentId = useCallback(() => latest.current.snapshot?.activeId || null, []);
  return { draft, activeId: snapshot?.activeId || null, items, ready, saving, dirty, operating, error, legacyRaw,
    update, flush, create, activate, rename, remove, duplicate, recoverAsCopy, retry, currentId };
}
