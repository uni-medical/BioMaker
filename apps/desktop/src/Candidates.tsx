import { useEffect, useMemo, useRef, useState } from 'react';
import JSON5 from 'json5';
import { Check, Code2, GitBranch, GitFork, RefreshCw, X } from 'lucide-react';
import { Notice, Spinner } from './components';
import { errorMessage, request, type Run, type Translate } from './models';
import './candidates.css';

interface Candidate {
  id: string; step: number | null; plan?: string; analysis?: string; code?: string;
  is_buggy?: boolean | null; tag?: string; agent_name?: string; outcome?: string;
  metric?: { value?: number | null; maximize?: boolean; name?: string };
  exec_time?: number; exc_type?: string; _term_out?: string[];
}
interface Journal { nodes: Candidate[]; node2parent: Record<string, string>; total: number }

function readCandidate(value: unknown): Candidate | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  if (typeof raw.id !== 'string' || !raw.id || raw.id.length > 256) return null;
  const node: Candidate = { id: raw.id, step: typeof raw.step === 'number' && Number.isSafeInteger(raw.step) && raw.step >= 0 ? raw.step : null };
  for (const key of ['plan', 'analysis', 'code', 'tag', 'agent_name', 'outcome', 'exc_type'] as const) if (typeof raw[key] === 'string') node[key] = raw[key];
  if (typeof raw.is_buggy === 'boolean') node.is_buggy = raw.is_buggy;
  if (typeof raw.exec_time === 'number' && Number.isFinite(raw.exec_time) && raw.exec_time >= 0) node.exec_time = raw.exec_time;
  if (Array.isArray(raw._term_out)) node._term_out = raw._term_out.filter((line): line is string => typeof line === 'string');
  if (raw.metric && typeof raw.metric === 'object' && !Array.isArray(raw.metric)) {
    const metric = raw.metric as Record<string, unknown>;
    node.metric = {
      value: typeof metric.value === 'number' && Number.isFinite(metric.value) ? metric.value : null,
      maximize: typeof metric.maximize === 'boolean' ? metric.maximize : undefined,
      name: typeof metric.name === 'string' ? metric.name : undefined,
    };
  }
  return node;
}

export default function Candidates({ run, t, serviceReady, serviceRevision, requestedCandidate }: { run: Run; t: Translate; serviceReady: boolean; serviceRevision: number; requestedCandidate?: string | null }) {
  const artifact = run.artifacts?.find(file => file.name === 'journal.json');
  const [journal, setJournal] = useState<Journal | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [reload, setReload] = useState(0);
  const [section, setSection] = useState<'summary' | 'code' | 'output'>('summary');
  const detailHeading = useRef<HTMLHeadingElement>(null);
  const appliedRequest = useRef<string | null>(null);
  useEffect(() => {
    let valid = true;
    if (!serviceReady) { setBusy(false); return; }
    setError('');
    if (!artifact) { setJournal(null); return; }
    if (artifact.size > 12 * 1024 * 1024) { setError(t('候选记录较大，请在文件标签中导出 journal.json 查看。', 'This journal is large. Export journal.json from Files to inspect it.')); return; }
    setBusy(true);
    request<unknown>(artifact.url).then(raw => {
      // Python's journal serializer may emit NaN/Infinity for failed candidates.
      // JSON5 parses those tokens without evaluating code; nonfinite scores render as —.
      const value = typeof raw === 'string' ? JSON5.parse(raw) : raw;
      if (!value || !Array.isArray(value.nodes)) throw new Error(t('候选记录格式无法识别。', 'Unrecognized candidate journal format.'));
      const allNodes: Candidate[] = value.nodes.map(readCandidate).filter((node: Candidate | null): node is Candidate => node !== null);
      if (new Set(allNodes.map((node: Candidate) => node.id)).size !== allNodes.length) throw new Error(t('候选记录中存在重复标识，无法确定方案。请导出 journal.json 核对。', 'Candidate identifiers are duplicated. Export journal.json to review the records.'));
      const nodes = allNodes.slice(0, 1500);
      // Keep the recorded result reachable without rendering thousands of rows.
      if (requestedCandidate && !nodes.some(node => node.id === requestedCandidate)) {
        const requested = allNodes.find(node => node.id === requestedCandidate);
        if (requested) nodes.push(requested);
      }
      const node2parent: Record<string, string> = Object.create(null);
      if (value.node2parent && typeof value.node2parent === 'object' && !Array.isArray(value.node2parent)) {
        for (const [id, parent] of Object.entries(value.node2parent)) if (typeof parent === 'string') node2parent[id] = parent;
      }
      if (valid) { setJournal({ nodes, node2parent, total: allNodes.length }); setSelected(current => nodes.some(node => node.id === current) ? current : nodes.at(-1)?.id || null); }
    }).catch(e => { if (valid) setError(errorMessage(e)); }).finally(() => { if (valid) setBusy(false); });
    return () => { valid = false; };
  }, [run.id, artifact?.revision, artifact?.url, reload, serviceReady, serviceRevision, requestedCandidate]);
  useEffect(() => {
    if (!requestedCandidate || !journal || appliedRequest.current === requestedCandidate) return;
    if (journal.nodes.some(node => node.id === requestedCandidate)) {
      appliedRequest.current = requestedCandidate;
      setSelected(requestedCandidate);
      setSection('summary');
    }
  }, [requestedCandidate, journal]);
  useEffect(() => {
    if (requestedCandidate && selected === requestedCandidate && appliedRequest.current === requestedCandidate) detailHeading.current?.focus();
  }, [requestedCandidate, selected]);
  const rows = useMemo(() => {
    if (!journal) return [];
    const byId = new Map(journal.nodes.map(node => [node.id, node]));
    return journal.nodes.map(node => {
      const ancestors = new Set([node.id]); let current = node.id; let depth = 0;
      while (journal.node2parent[current] && byId.has(journal.node2parent[current]) && !ancestors.has(journal.node2parent[current])) {
        current = journal.node2parent[current]; ancestors.add(current); depth++;
      }
      return { node, depth: Math.min(depth, 5), parent: byId.get(journal.node2parent[node.id]) };
    });
  }, [journal]);
  const active = journal?.nodes.find(node => node.id === selected);
  const metric = (node: Candidate) => typeof node.metric?.value === 'number' && Number.isFinite(node.metric.value) ? Number(node.metric.value.toPrecision(6)).toString() : '—';
  const number = (node: Candidate) => node.step !== null && node.step !== undefined ? `#${node.step + 1}` : node.id.slice(0, 6);
  if (!artifact) return <div className="detail-empty"><GitBranch size={28}/><h3>{t('候选方案会随着实验展开', 'Candidates appear as the experiment develops')}</h3><p>{t('这里展示每个候选方案的来源、代码、执行结果与记录指标。', 'Explore each candidate’s lineage, code, execution output, and recorded metric.')}</p></div>;
  if (!serviceReady && !journal) return <div className="detail-empty"><Spinner size={24}/><p>{t('等待本地服务恢复后读取候选方案', 'Candidates will load when the local service is available')}</p></div>;
  if (error) return <Notice danger>{error}<button className="text-button" disabled={!serviceReady} onClick={() => setReload(value => value + 1)}><RefreshCw size={13}/>{t('重试', 'Retry')}</button></Notice>;
  if (busy && !journal) return <div className="detail-empty"><Spinner size={24}/><p>{t('正在读取候选方案', 'Loading candidates')}</p></div>;
  if (!journal?.nodes.length) return <div className="detail-empty"><GitBranch size={28}/><h3>{t('尚未记录候选方案', 'No candidates recorded yet')}</h3></div>;
  return <div className="candidates-view">
    {journal.total > journal.nodes.length && <p className="candidate-window-note">{t(`显示前 1500 个候选${requestedCandidate ? '及从结果页打开的方案' : ''}。完整记录可在文件中导出 journal.json。`, `Showing the first 1500 candidates${requestedCandidate ? ' and the candidate opened from Results' : ''}. Export journal.json from Files for the full record.`)}</p>}
    {requestedCandidate && !journal.nodes.some(node => node.id === requestedCandidate) && <Notice danger>{t('此候选方案已不在当前记录中。请返回结果页刷新后再查看。', 'This candidate is no longer in the current record. Refresh Results before opening it again.')}</Notice>}
    <div className="candidate-list" aria-label={t('候选方案列表', 'Candidate list')}>
      <div className="candidate-list-caption"><span>{t(`${journal.nodes.length} 个候选方案`, `${journal.nodes.length} candidates`)}</span><span>{t('记录指标', 'Metric')}</span></div>
      {rows.map(({ node, depth, parent }) => <button key={node.id} className={`candidate-row ${selected === node.id ? 'selected' : ''}`} onClick={() => { setSelected(node.id); setSection('summary'); }} aria-pressed={selected === node.id}>
        <span className="candidate-lineage" style={{ marginLeft: depth * 13 }}>{journal.node2parent[node.id] ? <GitFork size={15}/> : <GitBranch size={15}/>}</span>
        <span className="candidate-row-title"><strong>{number(node)} <small>{node.agent_name || (journal.node2parent[node.id] ? t('改进', 'Refinement') : t('初始方案', 'Draft'))}</small></strong><span>{parent ? `${t('基于', 'From')} ${number(parent)}` : journal.node2parent[node.id] ? t('上级候选未显示', 'Parent candidate not shown') : t('独立候选方案', 'Independent candidate')}</span></span>
        <span className={`candidate-verdict ${node.is_buggy ? 'failed' : node.is_buggy === false ? 'valid' : ''}`}>{node.is_buggy ? <X size={12}/> : node.is_buggy === false ? <Check size={12}/> : null}{metric(node)}</span>
      </button>)}
    </div>
    {active && <section className="candidate-detail" aria-label={t('候选方案详情', 'Candidate details')}>
      <div className="candidate-detail-heading"><h3 ref={detailHeading} tabIndex={-1}>{t('候选方案', 'Candidate')} {number(active)}</h3><span className={`status-badge ${active.is_buggy ? 'incomplete' : 'complete'}`}>{active.is_buggy ? t('执行有误', 'Execution error') : active.is_buggy === false ? t('有效记录', 'Valid record') : active.tag || t('处理中', 'In progress')}</span></div>
      <div className="candidate-facts"><div><span>{t('记录指标', 'Recorded metric')}</span><strong>{metric(active)}</strong></div><div><span>{t('执行耗时', 'Execution time')}</span><strong>{typeof active.exec_time === 'number' ? `${Math.round(active.exec_time)} s` : '—'}</strong></div><div><span>{t('优化方向', 'Metric direction')}</span><strong>{active.metric?.maximize === true ? t('越大越好', 'Maximize') : active.metric?.maximize === false ? t('越小越好', 'Minimize') : '—'}</strong></div></div>
      <div className="candidate-sections" role="tablist" aria-label={t('候选内容', 'Candidate content')}>{(['summary', 'code', 'output'] as const).map(id => <button key={id} role="tab" aria-selected={section === id} className={section === id ? 'selected' : ''} onClick={() => setSection(id)}>{id === 'summary' ? t('方案与分析', 'Plan & analysis') : id === 'code' ? t('代码', 'Code') : t('执行输出', 'Output')}</button>)}</div>
      <div role="tabpanel" className="candidate-content">{section === 'summary' ? <>{active.plan && <><h4>{t('方案', 'Plan')}</h4><p>{active.plan}</p></>}{active.analysis && <><h4>{t('分析', 'Analysis')}</h4><p>{active.analysis}</p></>}{active.exc_type && <Notice danger>{active.exc_type}</Notice>}{!active.plan && !active.analysis && <p className="muted">{t('此候选方案没有记录说明。', 'No description was recorded for this candidate.')}</p>}<p className="candidate-evidence-note">{t('指标来自该实验的候选记录；它们不代表独立测试或临床验证。', 'Metrics are recorded within this experiment; they do not imply independent test or clinical validation.')}</p></> : <pre tabIndex={0}>{section === 'code' ? (active.code || t('尚无代码记录', 'No code recorded')) : (active._term_out?.join('\n') || t('尚无执行输出', 'No execution output recorded'))}</pre>}</div>
    </section>}
  </div>;
}
