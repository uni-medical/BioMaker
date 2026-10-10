import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, GitBranch, GitFork, RefreshCw, X } from 'lucide-react';
import { Notice, Spinner } from './components';
import { errorMessage, request, type Run, type Translate } from './models';
import { candidateNumber, candidatePhaseLabel, keepCandidateSelection, readCandidateRecords, type Candidate, type CandidateRecords } from './candidateRecords';
import './candidates.css';

export default function Candidates({ run, t, serviceReady, serviceRevision, requestedCandidate }: { run: Run; t: Translate; serviceReady: boolean; serviceRevision: number; requestedCandidate?: string | null }) {
  const artifact = run.artifacts?.find(file => file.name === 'journal.json');
  const [journal, setJournal] = useState<CandidateRecords | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [reload, setReload] = useState(0);
  const [section, setSection] = useState<'summary' | 'code' | 'output'>('summary');
  const detailHeading = useRef<HTMLHeadingElement>(null);
  const appliedRequest = useRef<string | null>(null);
  const live = ['queued', 'starting', 'running', 'stopping'].includes(run.status);
  useEffect(() => {
    setJournal(null); setSelected(null); setSection('summary'); setError(''); appliedRequest.current = null;
  }, [run.id]);
  useEffect(() => {
    let valid = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (!serviceReady) { setBusy(false); return; }
    async function refresh() {
      setBusy(true);
      try {
        const query = requestedCandidate ? `?requested_id=${encodeURIComponent(requestedCandidate)}` : '';
        const raw = await request<unknown>(`/api/desktop/runs/${encodeURIComponent(run.id)}/candidates${query}`);
        const next = readCandidateRecords(raw);
        if (valid) { setJournal(next); setSelected(current => keepCandidateSelection(next.nodes, current)); setError(''); }
      } catch (e) {
        if (valid) setError(errorMessage(e));
      } finally {
        if (valid) { setBusy(false); if (live) timer = setTimeout(() => void refresh(), 1500); }
      }
    }
    void refresh();
    return () => { valid = false; if (timer) clearTimeout(timer); };
  }, [run.id, run.status, live, artifact?.revision, reload, serviceReady, serviceRevision, requestedCandidate]);
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
  const plan = active?.phase === 'generating' ? active.selection_reason || active.plan : active?.plan || active?.selection_reason;
  const metric = (node: Candidate) => typeof node.metric?.value === 'number' && Number.isFinite(node.metric.value) ? Number(node.metric.value.toPrecision(6)).toString() : '—';
  const number = (node: Candidate) => candidateNumber(node) + (node.recorded ? '' : t('（生成序号）', ' (generation order)'));
  if (!serviceReady && !journal) return <div className="detail-empty"><Spinner size={24}/><p>{t('等待本地服务恢复后读取候选方案', 'Candidates will load when the local service is available')}</p></div>;
  const updateNotice = (!serviceReady || error) && <Notice danger={!!error}>{t('更新暂停，保留上次读取的候选记录。', 'Updates paused. Showing the last candidate records read.')}{error && <span className="candidate-update-error">{error}</span>}<button className="text-button" disabled={!serviceReady || busy} onClick={() => setReload(value => value + 1)}><RefreshCw size={13}/>{t('重试', 'Retry')}</button></Notice>;
  if (error && !journal) return updateNotice;
  if (busy && !journal) return <div className="detail-empty"><Spinner size={24}/><p>{t('正在读取候选方案', 'Loading candidates')}</p></div>;
  if (!journal?.nodes.length) return <>{updateNotice}<div className="detail-empty"><GitBranch size={28}/><h3>{t('尚未记录候选方案', 'No candidates recorded yet')}</h3><p>{live ? t('实验正在准备；生成候选后会自动显示在这里。', 'The experiment is preparing. Candidates will appear here automatically when generation starts.') : t('此运行未保存候选记录。可在活动和日志中查看结束原因。', 'This run did not save candidate records. Review activity and logs for its outcome.')}</p></div></>;
  return <div className="candidates-view">
    {updateNotice}
    {journal.warnings.length > 0 && <Notice>{t('部分活动记录无法读取，以下内容来自可读取的记录。', 'Some activity could not be read. The view uses the available records.')}</Notice>}
    {journal.total > journal.nodes.length && <p className="candidate-window-note">{t(`显示前 1500 个候选${requestedCandidate ? '及从结果页打开的方案' : ''}。完整记录可在文件中导出 journal.json。`, `Showing the first 1500 candidates${requestedCandidate ? ' and the candidate opened from Results' : ''}. Export journal.json from Files for the full record.`)}</p>}
    {requestedCandidate && !journal.nodes.some(node => node.id === requestedCandidate) && <Notice danger>{t('此候选方案已不在当前记录中。请返回结果页刷新后再查看。', 'This candidate is no longer in the current record. Refresh Results before opening it again.')}</Notice>}
    <div className="candidate-list" aria-label={t('候选方案列表', 'Candidate list')}>
      <div className="candidate-list-caption"><span>{t(`${journal.nodes.length} 个候选方案`, `${journal.nodes.length} candidates`)}</span><span>{t('状态 / 指标', 'Status / metric')}</span></div>
      {rows.map(({ node, depth, parent }) => <button key={node.id} className={`candidate-row ${selected === node.id ? 'selected' : ''}`} onClick={() => { setSelected(node.id); setSection('summary'); }} aria-pressed={selected === node.id}>
        <span className="candidate-lineage" style={{ marginLeft: depth * 13 }}>{journal.node2parent[node.id] ? <GitFork size={15}/> : <GitBranch size={15}/>}</span>
        <span className="candidate-row-title"><strong>{number(node)} <small>{node.agent_name || (journal.node2parent[node.id] ? t('改进', 'Refinement') : t('初始方案', 'Draft'))}</small></strong><span>{parent ? `${t('基于', 'From')} ${number(parent)}` : journal.node2parent[node.id] ? t('上级候选未显示', 'Parent candidate not shown') : t('独立候选方案', 'Independent candidate')}</span></span>
        <span className={`candidate-verdict ${node.phase}`}><span>{node.phase === 'failed' ? <X size={12}/> : node.phase === 'valid' ? <Check size={12}/> : serviceReady && !error && live && ['generating', 'running', 'evaluating'].includes(node.phase) ? <Spinner size={12}/> : null}{candidatePhaseLabel(node.phase, t)}</span>{node.recorded && <small>{metric(node)}</small>}</span>
      </button>)}
    </div>
    {active && <section className="candidate-detail" aria-label={t('候选方案详情', 'Candidate details')}>
      <div className="candidate-detail-heading"><h3 ref={detailHeading} tabIndex={-1}>{t('候选方案', 'Candidate')} {number(active)}</h3><span className={`candidate-status ${active.phase}`}>{candidatePhaseLabel(active.phase, t)}</span></div>
      <div className="candidate-facts"><div><span>{t('记录指标', 'Recorded metric')}</span><strong>{metric(active)}</strong></div><div><span>{t('执行耗时', 'Execution time')}</span><strong>{typeof active.exec_time === 'number' ? `${Math.round(active.exec_time)} s` : '—'}</strong></div><div><span>{t('优化方向', 'Metric direction')}</span><strong>{active.metric?.maximize === true ? t('越大越好', 'Maximize') : active.metric?.maximize === false ? t('越小越好', 'Minimize') : '—'}</strong></div></div>
      <div className="candidate-sections" role="tablist" aria-label={t('候选内容', 'Candidate content')}>{(['summary', 'code', 'output'] as const).map(id => <button key={id} role="tab" aria-selected={section === id} className={section === id ? 'selected' : ''} onClick={() => setSection(id)}>{id === 'summary' ? t('方案与分析', 'Plan & analysis') : id === 'code' ? t('代码', 'Code') : t('执行输出', 'Output')}</button>)}</div>
      <div role="tabpanel" className="candidate-content">{active.content_truncated && <Notice>{t('较长内容已截取；完整记录可从文件中导出。', 'Long content is truncated. Export the files for the full record.')}</Notice>}{section === 'summary' ? <>{plan && <><h4>{t('方案', 'Plan')}</h4><p>{plan}</p></>}{active.analysis && <><h4>{t('分析', 'Analysis')}</h4><p>{active.analysis}</p></>}{active.exc_type && <Notice danger>{active.exc_type}</Notice>}{!plan && !active.analysis && <p className="muted">{t('此候选方案没有记录说明。', 'No description was recorded for this candidate.')}</p>}<p className="candidate-evidence-note">{t('指标来自已保存的候选记录；执行完成不等于评估通过，也不代表独立测试或临床验证。', 'Metrics come from saved candidate records. Finishing execution does not establish successful evaluation, independent testing, or clinical validation.')}</p></> : <pre tabIndex={0}>{section === 'code' ? (active.code || t('代码生成后会显示在这里。', 'Code will appear here after generation.')) : (active._term_out?.join('\n') || (active.recorded ? t('尚无执行输出。可在运行日志或导出的记录中查看现有内容。', 'No execution output was recorded. Check Logs or exported records for available content.') : t('执行结果尚未保存；实时输出可在日志中查看。', 'Execution output has not been saved yet. View live output in Logs.')))}</pre>}</div>
    </section>}
  </div>;
}
