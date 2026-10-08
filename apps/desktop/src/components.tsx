import { useEffect, useRef, type ReactNode } from "react";
import { X, LoaderCircle, Check, AlertCircle } from "lucide-react";
import type { Translate } from "./models";

export function Mark({ size = 32 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="18 17 46 46"
      fill="none"
      aria-hidden="true"
    >
      <g
        stroke="currentColor"
        strokeWidth="4.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M27 56V24h11c10 0 15 3 15 8 0 5-5 8-14 8H27" />
        <path d="M27 40h14c9 0 14 3 14 8s-5 8-15 8H27" />
        <path
          d="M39 24c-6 4-9 9-9 16s3 12 9 16"
          strokeOpacity=".6"
          strokeWidth="2.4"
        />
      </g>
    </svg>
  );
}
export function Spinner({ size = 16 }: { size?: number }) {
  return <LoaderCircle size={size} className="spin" aria-hidden="true" />;
}
export function IconButton({
  label,
  children,
  onClick,
  active = false,
  disabled = false,
  className = "",
}: {
  label: string;
  children: ReactNode;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      className={`icon-button ${active ? "active" : ""} ${className}`}
      title={label}
      aria-label={label}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}
export function Modal({
  title,
  children,
  onClose,
  t,
  className = "",
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  t: Translate;
  className?: string;
}) {
  const container = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const dialog = container.current;
    const frame = requestAnimationFrame(() => {
      // Do not steal focus if a user (or an autofocus control) already chose a
      // field before the first animation frame of the dialog.
      if (dialog?.contains(document.activeElement)) return;
      const target =
        dialog?.querySelector<HTMLElement>("[autofocus], [data-modal-autofocus]") ||
        dialog?.querySelector<HTMLElement>(
          "input:not(:disabled), textarea:not(:disabled), select:not(:disabled)",
        ) ||
        dialog?.querySelector<HTMLElement>("button");
      target?.focus();
    });
    function keyboard(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.stopPropagation();
        closeRef.current();
      }
      if (event.key !== "Tab" || !dialog) return;
      const elements = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex="0"]',
        ),
      ).filter((el) => el.offsetParent !== null);
      if (!elements.length) return;
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", keyboard, true);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("keydown", keyboard, true);
      previous?.focus();
    };
  }, []);
  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={`modal ${className}`}
        ref={container}
      >
        <div className="modal-heading">
          <h2>{title}</h2>
          <IconButton label={t("关闭", "Close")} onClick={onClose}>
            <X size={18} />
          </IconButton>
        </div>
        {children}
      </div>
    </div>
  );
}
export function Notice({
  children,
  danger = false,
}: {
  children: ReactNode;
  danger?: boolean;
}) {
  return (
    <div
      className={`notice ${danger ? "danger" : ""}`}
      role={danger ? "alert" : "status"}
    >
      <AlertCircle size={17} />
      <div>{children}</div>
    </div>
  );
}
export function SettingRow({
  title,
  detail,
  children,
}: {
  title: string;
  detail?: string;
  children: ReactNode;
}) {
  return (
    <div className="setting-row">
      <div>
        <div className="setting-title">{title}</div>
        {detail && <p>{detail}</p>}
      </div>
      {children}
    </div>
  );
}
export function CheckItem({
  done,
  title,
  detail,
  onClick,
  label,
}: {
  done: boolean;
  title: string;
  detail: string;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      className={`setup-item ${done ? "done" : ""}`}
      onClick={onClick}
    >
      <span className="setup-check">
        {done ? <Check size={14} /> : <span />}
      </span>
      <span>
        <strong>{title}</strong>
        <small>{detail}</small>
      </span>
      <span className="setup-action">{label}</span>
    </button>
  );
}
