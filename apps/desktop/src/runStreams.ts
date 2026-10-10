/** A bounded window from one observed stream. Cursors are opaque service values. */
export interface StreamBase {
  status: "ready" | "missing";
  source_id: string | null;
  revision: string;
  observed_at: string;
  start: number;
  end: number;
  source_size: number;
  has_older: boolean;
  older_cursor: string | null;
  partial_first_line: boolean;
  invalid_utf8: boolean;
}

export interface LogWindow extends StreamBase {
  text: string;
}

export interface ActivityWindow extends StreamBase {
  events: { id: string; data: Record<string, unknown> }[];
  skipped_records: number;
}
