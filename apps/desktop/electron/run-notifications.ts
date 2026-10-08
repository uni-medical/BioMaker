export interface RunCompletion { id: string; outcome: 'complete' | 'failed' | 'interrupted' }
const ACTIVE = new Set(['queued', 'running', 'stopping']);
const TERMINAL = new Set(['succeeded', 'complete', 'failed', 'interrupted', 'incomplete', 'orphaned', 'stopped', 'cancelled']);

/** Observe authenticated server responses, never renderer-supplied lifecycle data.
 * Historical terminal runs are silent. A retry is a new active transition.
 * Tickets prevent older concurrent HTTP responses from resurrecting a finished run.
 */
export class RunNotifications {
  private active = new Map<string, string>();
  private issued = 0;
  private accepted = new Map<string, number>();
  issue(): number { return ++this.issued; }
  get activeCount(): number { return this.active.size; }
  observe(payload: unknown, ticket: number): RunCompletion[] {
    if (!Number.isSafeInteger(ticket) || ticket < 1) return [];
    const completed: RunCompletion[] = [];
    for (const item of Array.isArray(payload) ? payload : [payload]) {
      if (!item || typeof item !== 'object') continue;
      const { id, status, detailed_status } = item as Record<string, unknown>;
      if (typeof id !== 'string' || !id || id.length > 180 || /[\\/\x00-\x1f\x7f]/.test(id)) continue;
      const state = typeof detailed_status === 'string' ? detailed_status : status;
      if (typeof state !== 'string' || (!ACTIVE.has(state) && !TERMINAL.has(state))) continue;
      // Concurrent detail responses may describe entirely different runs.
      // Only the same run's newer observation can supersede this response.
      if (ticket < (this.accepted.get(id) ?? 0)) continue;
      this.accepted.set(id, ticket);
      if (ACTIVE.has(state)) {
        this.active.set(id, state);
        // The execution pool is small; also bound malformed/unbounded responses.
        if (this.active.size > 10_000) this.active.delete(this.active.keys().next().value!);
        continue;
      }
      if (!TERMINAL.has(state)) continue;
      const previous = this.active.get(id);
      this.active.delete(id);
      if (!previous || previous === 'stopping' || state === 'stopped' || state === 'cancelled') continue;
      completed.push({ id, outcome: state === 'succeeded' || state === 'complete' ? 'complete' : state === 'failed' ? 'failed' : 'interrupted' });
    }
    if (this.accepted.size > 20_000) {
      for (const id of this.accepted.keys()) {
        if (!this.active.has(id)) this.accepted.delete(id);
        if (this.accepted.size <= 10_000) break;
      }
    }
    return completed;
  }
}
