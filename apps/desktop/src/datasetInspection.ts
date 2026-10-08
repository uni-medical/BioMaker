import { useCallback, useEffect, useRef, useState } from "react";
import type { ServiceState } from "./bridge";
import { errorMessage, request, type Translate } from "./models";

export interface DatasetInspection {
  status: "ok" | "error";
  code: string;
  checked_at: string;
  elapsed_ms: number;
  complete: boolean;
  files: number;
  directories: number;
  symlinks: number;
  other: number;
  bytes: number;
  extensions: { extension: string; count: number }[];
  entries: { name: string; kind: "file" | "directory" | "symlink" | "other"; bytes: number | null }[];
  warnings: string[];
}
export function datasetAdmissionCode(message: string): string | null {
  return /\[dataset:(empty|not_found|not_directory|permission|unauthorized|invalid_request|timeout|busy|internal)\]/.exec(message)?.[1] || null;
}
export function datasetInspectionMessage(code: string, t: Translate): string {
  switch (code) {
    case "ok": return t("目录可访问", "Folder is accessible");
    case "empty": return t("目录中未发现普通文件，请选择包含数据的目录。", "No regular files were found. Choose a folder containing data.");
    case "not_found": return t("目录已不存在。请连接存储设备或重新选择目录。", "The folder no longer exists. Connect the storage device or choose another folder.");
    case "not_directory": return t("此路径不是目录，请重新选择数据集文件夹。", "This path is not a directory. Choose a dataset folder again.");
    case "permission": return t("无法读取此目录，请检查系统权限后重试或重新选择。", "This folder cannot be read. Check system permissions and retry, or select it again.");
    case "unauthorized": return t("请重新选择此目录，授权 BioMaker 读取目录概览。", "Select this folder again to allow BioMaker to inspect its directory metadata.");
    case "invalid_request": return t("目录路径无效，请重新选择数据集文件夹。", "The folder path is invalid. Choose the dataset folder again.");
    case "timeout": return t("目录检查超时，请检查存储设备后重试。", "Folder inspection timed out. Check the storage device and retry.");
    case "busy": return t("另一项目录检查仍在进行，请稍后刷新。", "Another folder inspection is in progress. Refresh after it finishes.");
    default: return t("暂时无法完成目录检查，请刷新重试。", "Folder inspection could not finish. Refresh to try again.");
  }
}
export function datasetWarningMessage(code: string, t: Translate): string {
  switch (code) {
    case "entry_limit": return t("已达到条目数量上限", "Entry limit reached");
    case "depth_limit": return t("较深目录未统计", "Deeper folders were not inspected");
    case "soft_timeout":
    case "hard_timeout": return t("已达到检查时间上限", "Inspection time limit reached");
    case "descendant_unreadable": return t("部分子项无法读取", "Some child entries could not be read");
    case "symlinks_not_followed": return t("符号链接仅计数，未读取链接目标", "Symbolic links were counted without inspecting their targets");
    case "changed_during_scan": return t("检查期间目录发生变化", "The folder changed during inspection");
    case "admission_short_circuit": return t("本次仅完成快速目录检查", "Only a quick folder check was completed");
    case "root_symlink_resolved": return t("所选目录已解析到实际位置", "The selected folder was resolved to its actual location");
    case "name_truncated": return t("部分条目名称已缩短或处理不可显示字符", "Some displayed names were shortened or had non-displayable characters replaced");
    case "extension_limit": return t("文件类型摘要仅显示部分类型", "The file-type summary shows only some types");
    case "size_limit": return t("大小统计达到显示上限，此数值为下界", "The size display limit was reached; this value is a lower bound");
    default: return t("部分目录信息未统计", "Some directory metadata was not inspected");
  }
}

type InspectionRecord = { generation: number; path: string; revision: number; result: DatasetInspection | null; errorCode: string | null };
export function useDatasetInspection(path: string, enabled: boolean, service: ServiceState) {
  const [record, setRecord] = useState<InspectionRecord | null>(null);
  const [refreshEpoch, setRefreshEpoch] = useState(0);
  const generation = useRef(0);
  const currentRecord = useRef<InspectionRecord | null>(null);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const context = useRef({ path, enabled, service });
  context.current = { path, enabled, service };
  const invalidate = useCallback(() => {
    generation.current += 1;
    currentRecord.current = null;
    setRecord(null);
    setRefreshEpoch(value => value + 1);
  }, []);
  useEffect(() => {
    const sequence = ++generation.current;
    currentRecord.current = null;
    setRecord(null);
    if (!path || !enabled || service.status !== "ready") return;
    const revision = service.revision;
    const current = () => generation.current === sequence && context.current.path === path &&
      context.current.enabled && context.current.service.status === "ready" && context.current.service.revision === revision;
    // A newer folder waits for the old bounded request, then skips all obsolete
    // queued requests. This avoids overlapping scans and automatic busy retries.
    queue.current = queue.current.catch(() => {}).then(async () => {
      if (!current()) return;
      let next: InspectionRecord;
      try {
        const result = await request<DatasetInspection>("/api/desktop/dataset/inspect", "POST", { path });
        next = { generation: sequence, path, revision, result, errorCode: null };
      } catch (error) {
        next = { generation: sequence, path, revision, result: null, errorCode: datasetAdmissionCode(errorMessage(error)) || "internal" };
      }
      if (current()) {
        currentRecord.current = next;
        setRecord(next);
      }
    });
    return () => { generation.current += 1; currentRecord.current = null; };
  }, [path, enabled, service.status, service.revision, refreshEpoch]);
  const matches = (value: InspectionRecord | null) => !!value && value.generation === generation.current &&
    value.path === context.current.path && context.current.enabled && context.current.service.status === "ready" &&
    value.revision === context.current.service.revision;
  const visible = matches(record) ? record : null;
  const isReady = () => matches(currentRecord.current) && currentRecord.current?.result?.status === "ok";
  return {
    result: visible?.result || null,
    error: !!visible?.errorCode,
    errorCode: visible?.errorCode || null,
    pending: !!path && enabled && service.status === "ready" && !visible,
    ready: !!visible?.result && visible.result.status === "ok",
    isReady,
    invalidate,
  };
}
export type DatasetInspectionState = ReturnType<typeof useDatasetInspection>;
