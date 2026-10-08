export type CloseBlockReason = "saving" | "unsaved";
export type CloseProtectionCallback = () => CloseBlockReason | null;

/** Run every callback: even when saving blocks close, another scope may need to flush. */
export function collectCloseBlockReason(callbacks: Iterable<CloseProtectionCallback>): CloseBlockReason | null {
  let reason: CloseBlockReason | null = null;
  for (const callback of callbacks) {
    let result: CloseBlockReason | null;
    try {
      result = callback();
    } catch {
      result = "unsaved";
    }
    if (result === "saving") reason = "saving";
    else if (result === "unsaved" && reason === null) reason = "unsaved";
  }
  return reason;
}

const protections = new Map<string, { callback: CloseProtectionCallback }>();

function beforeUnload(event: BeforeUnloadEvent): void {
  // Snapshot registrations so callbacks can flush or unregister synchronously.
  const reason = collectCloseBlockReason(Array.from(protections.values(), entry => entry.callback));
  if (!reason) return;
  try {
    window.biomaker.reportCloseBlock(reason);
  } catch {
    // A bridge error must not bypass the renderer's close protection.
  } finally {
    event.preventDefault();
    event.returnValue = "Unsaved changes";
  }
}

/** Register once per scope; read mutable state from refs inside the callback. */
export function registerCloseProtection(key: string, callback: CloseProtectionCallback): () => void {
  const entry = { callback };
  const attach = protections.size === 0;
  protections.set(key, entry);
  if (attach) window.addEventListener("beforeunload", beforeUnload);
  return () => {
    // An older effect cleanup must not remove a newer registration for this scope.
    if (protections.get(key) !== entry) return;
    protections.delete(key);
    if (protections.size === 0) window.removeEventListener("beforeunload", beforeUnload);
  };
}
