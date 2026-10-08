import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface ToastRecord {
  id: number;
  message: string;
  scope: object;
}

/** A notice belongs to the view, language and service that started its action. */
export function useToast(context: string) {
  const current = useRef({ context, token: {} });
  if (current.current.context !== context) current.current = { context, token: {} };
  const token = current.current.token;
  const sequence = useRef(0);
  const mounted = useRef(true);
  const [record, setRecord] = useState<ToastRecord | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const publish = useCallback((message: string, scope: object) => {
    if (!mounted.current || !message || scope !== current.current.token) return;
    setRecord({ id: ++sequence.current, message, scope });
  }, []);

  // Each render context has its own notifier. Async completions retain the
  // original token, including across A -> B -> A navigation.
  const toast = useMemo(() => (message: string) => publish(message, token), [publish, token]);
  // Only app-wide native subscription / preference persistence errors use this.
  // Business operations must retain their scoped notifier instead.
  const systemToast = useCallback((message: string) => publish(message, current.current.token), [publish]);
  const dismissToast = useCallback((id: number) => {
    setRecord(value => value?.id === id ? null : value);
  }, []);

  useEffect(() => {
    if (!record) return;
    const timeout = setTimeout(() => dismissToast(record.id), 6000);
    return () => clearTimeout(timeout);
  }, [record?.id, dismissToast]);

  return { toast, systemToast, dismissToast, toastMessage: record?.scope === token ? record : null };
}
