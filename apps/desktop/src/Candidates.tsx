import { useEffect, useMemo, useState } from 'react';
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
interface Journal { nodes: Candidate[]; node2parent: Record<string, string> }

export default function Candidates({ run, t, serviceReady, serviceRevision }: { run: Run; t: Translate; serviceReady: boolean; serviceRevision: number }) {
  const artifact = run.artifacts?.find(file => file.name === 'journal.json');
  const [journal, setJournal] = useState<Journal | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [reload, setReload] = useState(0);
  const [section, setSection] = useState<'summary' | 'code' | 'output'>('summary');
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
      const nodes: Candidate[] = value.nodes.filter((node: unknown): node is Candidate => !!node && typeof node === 'object' && typeof (node as Candidate).id === 'string').slice(0, 1500);
      const node2parent = value.node2parent && typeof value.node2parent === 'object' && !Array.isArray(value.node2parent) ? value.node2parent : {};
      if (valid) { setJournal({ nodes, node2parent }); setSelected(current => nodes.some(node => node.id === current) ? current : nodes.at(-1)?.id || null); }
    }).catch(e => { if (valid) setError(errorMessage(e)); }).finally(() => { if (valid) setBusy(false); });
    return () => { valid = false; };
  }, [run.id, artifact?.revision, artifact?.url, reload, serviceReady, serviceRevision]);
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
    <div className="candidate-list" aria-label={t('候选方案列表', 'Candidate list')}>
      <div className="candidate-list-caption"><span>{t(`${journal.nodes.length} 个候选方案`, `${journal.nodes.length} candidates`)}</span><span>{t('记录指标', 'Metric')}</span></div>
      {rows.map(({ node, depth, parent }) => <button key={node.id} className={`candidate-row ${selected === node.id ? 'selected' : ''}`} onClick={() => { setSelected(node.id); setSection('summary'); }} aria-pressed={selected === node.id}>
        <span className="candidate-lineage" style={{ marginLeft: depth * 13 }}>{parent ? <GitFork size={15}/> : <GitBranch size={15}/>}</span>
        <span className="candidate-row-title"><strong>{number(node)} <small>{node.agent_name || (parent ? t('改进', 'Refinement') : t('初始方案', 'Draft'))}</small></strong><span>{parent ? `${t('基于', 'From')} ${number(parent)}` : t('独立候选方案', 'Independent candidate')}</span></span>
        <span className={`candidate-verdict ${node.is_buggy ? 'failed' : node.is_buggy === false ? 'valid' : ''}`}>{node.is_buggy ? <X size={12}/> : node.is_buggy === false ? <Check size={12}/> : null}{metric(node)}</span>
      </button>)}
    </div>
    {active && <section className="candidate-detail" aria-label={t('候选方案详情', 'Candidate details')}>
      <div className="candidate-detail-heading"><h3>{t('候选方案', 'Candidate')} {number(active)}</h3><span className={`status-badge ${active.is_buggy ? 'incomplete' : 'complete'}`}>{active.is_buggy ? t('执行有误', 'Execution error') : active.is_buggy === false ? t('有效记录', 'Valid record') : active.tag || t('处理中', 'In progress')}</span></div>
      <div className="candidate-facts"><div><span>{t('记录指标', 'Recorded metric')}</span><strong>{metric(active)}</strong></div><div><span>{t('执行耗时', 'Execution time')}</span><strong>{typeof active.exec_time === 'number' ? `${Math.round(active.exec_time)} s` : '—'}</strong></div><div><span>{t('优化方向', 'Metric direction')}</span><strong>{active.metric?.maximize === true ? t('越大越好', 'Maximize') : active.metric?.maximize === false ? t('越小越好', 'Minimize') : '—'}</strong></div></div>
      <div className="candidate-sections" role="tablist" aria-label={t('候选内容', 'Candidate content')}>{(['summary', 'code', 'output'] as const).map(id => <button key={id} role="tab" aria-selected={section === id} className={section === id ? 'selected' : ''} onClick={() => setSection(id)}>{id === 'summary' ? t('方案与分析', 'Plan & analysis') : id === 'code' ? t('代码', 'Code') : t('执行输出', 'Output')}</button>)}</div>
      <div role="tabpanel" className="candidate-content">{section === 'summary' ? <>{active.plan && <><h4>{t('方案', 'Plan')}</h4><p>{active.plan}</p></>}{active.analysis && <><h4>{t('分析', 'Analysis')}</h4><p>{active.analysis}</p></>}{active.exc_type && <Notice danger>{active.exc_type}</Notice>}{!active.plan && !active.analysis && <p className="muted">{t('此候选方案没有记录说明。', 'No description was recorded for this candidate.')}</p>}<p className="candidate-evidence-note">{t('指标来自该实验的候选记录；它们不代表独立测试或临床验证。', 'Metrics are recorded within this experiment; they do not imply independent test or clinical validation.')}</p></> : <pre tabIndex={0}>{section === 'code' ? (active.code || t('尚无代码记录', 'No code recorded')) : (active._term_out?.join('\n') || t('尚无执行输出', 'No execution output recorded'))}</pre>}</div>
    </section>}
  </div>;
}
