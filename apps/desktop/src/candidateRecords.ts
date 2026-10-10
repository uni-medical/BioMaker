import type { Translate } from './models';

export type CandidatePhase = 'generating' | 'queued' | 'running' | 'evaluating' | 'valid' | 'failed' | 'interrupted' | 'unconfirmed';
export interface Candidate {
  id: string; step: number | null; scheduled_step?: number | null; recorded: boolean; phase: CandidatePhase;
  plan?: string; analysis?: string; code?: string; selection_reason?: string; content_truncated?: boolean;
  is_buggy?: boolean | null; agent_name?: string; outcome?: string;
  metric?: { value?: number | null; maximize?: boolean | null };
  exec_time?: number; exc_type?: string; _term_out?: string[];
}
export interface CandidateRecords { nodes: Candidate[]; node2parent: Record<string, string>; total: number; warnings: string[]; revision: string }
const phases = new Set<CandidatePhase>(['generating', 'queued', 'running', 'evaluating', 'valid', 'failed', 'interrupted', 'unconfirmed']);
const integer = (value: unknown): value is number => typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;

export function readCandidateRecords(value: unknown): CandidateRecords {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Candidate records unavailable.');
  const raw = value as Record<string, unknown>;
  if (raw.state !== 'ready' || !Array.isArray(raw.nodes)) throw new Error('Candidate records are updating or unavailable.');
  const nodes: Candidate[] = [];
  const ids = new Set<string>();
  for (const item of raw.nodes) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) continue;
    const data = item as Record<string, unknown>;
    if (typeof data.id !== 'string' || !data.id || data.id.length > 256) continue;
    if (ids.has(data.id)) throw new Error('Candidate identifiers are duplicated.');
    ids.add(data.id);
    const recorded = data.recorded === true;
    let phase = phases.has(data.phase as CandidatePhase) ? data.phase as CandidatePhase : 'unconfirmed';
    // A live placeholder can never become a successful result just because a
    // display snapshot says "valid". Only a saved result may carry a score.
    if (phase === 'valid' && (!recorded || data.is_buggy !== false)) phase = 'unconfirmed';
    const node: Candidate = { id: data.id, step: integer(data.step) ? data.step : null, scheduled_step: integer(data.scheduled_step) ? data.scheduled_step : null, recorded, phase };
    for (const key of ['plan', 'analysis', 'code', 'agent_name', 'outcome', 'exc_type', 'selection_reason'] as const) if (typeof data[key] === 'string') node[key] = data[key];
    node.content_truncated = data.content_truncated === true;
    if (recorded && typeof data.is_buggy === 'boolean') node.is_buggy = data.is_buggy;
    if (recorded && typeof data.exec_time === 'number' && Number.isFinite(data.exec_time) && data.exec_time >= 0) node.exec_time = data.exec_time;
    if (Array.isArray(data._term_out)) node._term_out = data._term_out.filter((line): line is string => typeof line === 'string');
    if (recorded && data.metric && typeof data.metric === 'object' && !Array.isArray(data.metric)) {
      const metric = data.metric as Record<string, unknown>;
      node.metric = { value: typeof metric.value === 'number' && Number.isFinite(metric.value) ? metric.value : null, maximize: typeof metric.maximize === 'boolean' ? metric.maximize : null };
    }
    nodes.push(node);
  }
  const node2parent: Record<string, string> = Object.create(null);
  if (raw.node2parent && typeof raw.node2parent === 'object' && !Array.isArray(raw.node2parent)) {
    for (const [id, parent] of Object.entries(raw.node2parent)) if (typeof parent === 'string') node2parent[id] = parent;
  }
  return { nodes, node2parent, total: integer(raw.total) ? raw.total : nodes.length, warnings: Array.isArray(raw.warnings) ? raw.warnings.filter((warning): warning is string => typeof warning === 'string') : [], revision: typeof raw.revision === 'string' ? raw.revision : '' };
}

export function candidatePhaseLabel(phase: CandidatePhase, t: Translate): string {
  switch (phase) {
    case 'generating': return t('生成方案中', 'Generating');
    case 'queued': return t('等待执行', 'Queued');
    case 'running': return t('执行中', 'Running');
    case 'evaluating': return t('评估与保存中', 'Evaluating and saving');
    case 'valid': return t('有效记录', 'Valid record');
    case 'failed': return t('失败', 'Failed');
    case 'interrupted': return t('已中断', 'Interrupted');
    default: return t('结果未确认', 'Result unconfirmed');
  }
}

export function keepCandidateSelection(nodes: Candidate[], current: string | null): string | null {
  return nodes.some(node => node.id === current) ? current : nodes.at(-1)?.id || null;
}

export function candidateNumber(node: Candidate): string {
  // Results uses the saved journal step. Scheduling may differ when execution
  // completes out of order, so it is only a pre-journal display fallback.
  const step = node.recorded && node.step !== null ? node.step : node.scheduled_step ?? node.step;
  return step !== null && step !== undefined ? `#${step + 1}` : node.id.slice(0, 6);
}
