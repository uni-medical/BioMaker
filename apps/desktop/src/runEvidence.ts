import { historyMessage } from "./historyMessages";
import type { Run, Translate } from "./models";

const missingTerminalReasons = new Set([
  "The process is not known to be alive and no terminal lifecycle event was recorded.",
  "No terminal lifecycle event was found in the last up to 5,000 valid event records. This does not determine whether the original process is running.",
]);

export function runStatusExplanation(run: Run, t: Translate): string {
  const rawReason: unknown = run.status_reason || run.error || "";
  const reason = typeof rawReason === "string" ? rawReason : JSON.stringify(rawReason) ?? "";
  if (missingTerminalReasons.has(reason)) {
    return t(
      "登记这条实验记录时，读取的历史片段未包含结束事件。状态保留登记时的判断，无法据此确认原进程是否仍在运行。",
      "The history read when this experiment was registered contained no terminal event. Its saved status preserves that assessment and does not establish whether the original process is still running.",
    );
  }
  const known = new Map([
    ["Cancelled before the run started.", "实验开始前已取消。"],
    ["Stopped by the user.", "已按用户请求停止实验。"],
    ["The run was interrupted before completion.", "实验在完成前中断。"],
    ["The run failed before completion.", "实验未完成，运行失败。"],
    ["The run did not record successful completion.", "已保存记录未确认实验成功完成。"],
  ]);
  const translated = known.get(reason);
  return translated ? t(translated, reason) : run.history_import ? historyMessage(reason, t) : reason;
}

/** Describe current file evidence without changing the saved status or actions. */
export function runTailExplanation(run: Run, t: Translate): string | null {
  const tail = run.event_tail;
  if (!tail || !Number.isSafeInteger(tail.valid_event_count) || tail.valid_event_count <= 0 ||
      !Number.isSafeInteger(tail.valid_event_limit) || tail.valid_event_limit < tail.valid_event_count) return null;
  const state = run.detailed_status;
  const ended = run.status === "complete" || ["succeeded", "failed", "stopped", "interrupted", "cancelled"].includes(state);
  if (!ended && !run.read_only) return null;
  if (tail.reason_code === "no_terminal_in_tail" && tail.terminal_event === null && ended &&
      !missingTerminalReasons.has(run.status_reason || run.error || "")) {
    return t(
      `当前读取的 ${tail.valid_event_count.toLocaleString("zh-CN")} 条历史事件中未包含结束事件。上方状态来自已保存的实验记录。`,
      `The ${tail.valid_event_count.toLocaleString("en-US")} historical events currently read contain no terminal event. The status above comes from the saved experiment record.`,
    );
  }
  const terminal = tail.terminal_event;
  const matches = terminal === "run_finished" ? state === "succeeded" || run.status === "complete"
    : terminal === "run_failed" ? state === "failed"
    : terminal === "run_interrupted" ? ["stopped", "interrupted"].includes(state) : true;
  if (matches) return null;
  const label = terminal === "run_finished" ? t("完成", "completion") : terminal === "run_failed" ? t("失败", "failure") : t("中断", "interruption");
  return t(
    `当前历史片段记录了“${label}”事件，与已保存状态不同。上方仍显示已保存的实验状态，可在活动中核对事件。`,
    `The current history records a ${label} event that differs from the saved status. The status above is preserved; inspect the events in Activity.`,
  );
}
