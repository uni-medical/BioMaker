// Historical JSON objects are not guaranteed to match the current engine schema.
// Keep arbitrary fields out of JSX and use only explicit failure evidence.
const eventNames: Record<string, [string, string]> = {
  workspace_prepared: ["工作目录已准备", "Workspace prepared"],
  split_protocol_prepared: ["数据划分协议已准备", "Split protocol prepared"],
  evaluator_prepared: ["评估器已准备", "Evaluator prepared"],
  exp_guide_written: ["实验指导已保存", "Experiment guidance saved"],
  run_overview: ["实验配置已记录", "Experiment configuration recorded"],
  run_warning: ["实验提醒", "Experiment warning"],
  run_manifest_written: ["实验清单已保存", "Experiment manifest saved"],
  run_manifest_updated: ["实验清单已更新", "Experiment manifest updated"],
  node_generation_started: ["开始生成候选方案", "Candidate generation started"],
  node_generation_finished: ["候选方案已生成", "Candidate generated"],
  node_generation_failed: ["候选方案生成失败", "Candidate generation failed"],
  node_queued_for_execution: ["候选方案等待执行", "Candidate queued for execution"],
  node_submitted: ["候选方案已提交执行", "Candidate submitted for execution"],
  execution_started: ["开始执行与校验候选方案", "Candidate execution and validation started"],
  execution_finished: ["候选方案执行与校验结束", "Candidate execution and validation ended"],
  context_built: ["上下文已准备", "Context prepared"],
  draft_plan_batch_generated: ["候选计划已生成", "Candidate plans generated"],
  node_selected: ["已选择探索候选", "Exploration candidate selected"],
  tool_observation: ["工具结果已记录", "Tool result recorded"],
  plan_code_extraction_retry: ["重试提取计划与代码", "Retrying plan and code extraction"],
  prompt_budget_applied: ["已应用上下文预算", "Context budget applied"],
  step_saved: ["候选记录已保存", "Candidate record saved"],
  run_started: ["实验已启动", "Experiment started"],
  run_finished: ["实验已完成", "Experiment completed"],
  run_failed: ["实验运行失败", "Experiment failed"],
  run_interrupted: ["实验已中断", "Experiment interrupted"],
  round_started: ["开始新一轮探索", "Exploration round started"],
  round_finished: ["本轮探索完成", "Exploration round completed"],
  node_started: ["执行候选方案", "Running a candidate"],
  node_finished: ["候选方案执行完成", "Candidate execution completed"],
  node_failed: ["候选方案执行失败", "Candidate execution failed"],
  final_retrain_started: ["开始最终重训", "Final retraining started"],
  final_retrain_finished: ["最终重训完成", "Final retraining completed"],
  final_retrain_failed: ["最终重训失败", "Final retraining failed"],
  agent_started: ["智能体开始工作", "Agent started"],
  agent_finished: ["智能体工作完成", "Agent finished"],
  code_generated: ["已生成实验代码", "Experiment code generated"],
  evaluation_started: ["开始评估", "Evaluation started"],
  evaluation_finished: ["评估完成", "Evaluation completed"],
  search_started: ["开始检索", "Research started"],
  report_generated: ["实验报告已生成", "Experiment report created"],
  report_started: ["正在生成实验报告", "Generating experiment report"],
  report_finished: ["实验报告已生成", "Experiment report created"],
  checkpoint_saved: ["检查点已保存", "Checkpoint saved"],
  budget_exhausted: ["已达到资源上限", "Resource limit reached"],
  run_stopping: ["正在停止实验", "Stopping experiment"],
  final_retrain_skipped: ["已跳过最终重训", "Final retraining skipped"],
  node_evaluated: ["候选方案已评估", "Candidate evaluated"],
  candidate_started: ["开始候选方案", "Candidate started"],
  candidate_finished: ["候选方案完成", "Candidate completed"],
  agent_step_started: ["智能体执行步骤", "Agent step started"],
  agent_step_finished: ["智能体步骤完成", "Agent step completed"],
  model_request_started: ["正在请求模型", "Requesting model"],
  model_request_finished: ["模型响应已返回", "Model response received"],
  tool_started: ["正在执行工具", "Running tool"],
  tool_finished: ["工具执行完成", "Tool completed"],
};

export function eventPresentation(event: Record<string, unknown>, t: (zh: string, en: string) => string) {
  const name = typeof event.event === "string" && event.event ? event.event : "event";
  const text = (value: unknown): string => typeof value === "string" ? value : typeof value === "number" && Number.isFinite(value) ? String(value) : "";
  const exception = text(event.exc_type);
  const failure = name.includes("failed") || !!text(event.error)
    || (name === "final_retrain_finished" && event.ok === false)
    || ((name === "execution_finished" || name === "node_evaluated") && !!exception)
    || (name === "node_evaluated" && event.is_buggy === true);
  const title = name === "final_retrain_finished" && failure ? t("最终重训未成功", "Final retraining failed")
    : name === "execution_finished" && failure ? t("候选方案执行或校验失败", "Candidate execution or validation failed")
    : name === "node_evaluated" && failure ? t("候选方案评估未通过", "Candidate evaluation failed")
    : Object.hasOwn(eventNames, name) ? t(...eventNames[name]) : name.replaceAll("_", " ");
  return {
    name, title, failure,
    summary: text(event.message) || text(event.error) || text(event.reason) || exception,
    step: text(event.step), timestamp: typeof event.ts === "string" ? event.ts : undefined,
    complete: !failure && (name.endsWith("finished") || name.endsWith("saved")),
  };
}
