import type { Translate } from "./models";

/** Translate known import diagnostics without discarding unknown source details. */
export function historyMessage(message: string, t: Translate): string {
  const plain = message.replace(/^Error invoking remote method ['"]biomaker:request['"]:\s*(?:Error:\s*)?/, "");
  const known: Record<string, string> = {
    "Only listed experiment records are copied. Datasets, checkpoints, workspaces, private ground truth, and subfolders are not migrated.": "只复制下方列出的实验记录；数据集、模型权重、训练工作目录、私有标注和子文件夹保持原位。",
    "No completion event was observed. This is a static snapshot; the original experiment may still be running.": "未找到完成事件，导入的是当前记录快照；原实验可能仍在运行。",
    "Imported static snapshot without a completion event. The original experiment may still be running and is not controlled by this app.": "这是没有完成事件的历史记录快照。原实验可能仍在运行，本软件仅查看该副本。",
    "Already imported; the existing copy was preserved.": "已经导入，保留现有副本。",
    "run_manifest.json must be a regular file, not a symbolic link.": "实验清单 run_manifest.json 必须是普通文件，不能是符号链接。",
    "No regular run_manifest.json was found in this experiment folder.": "此文件夹中未找到有效的实验清单 run_manifest.json。",
    "run_manifest.json requires a nonempty data_dir string.": "实验清单缺少有效的数据目录记录（data_dir），无法识别此实验。",
    "No recorded candidates, results, report, or solution was found; an empty record cannot be imported.": "未找到候选方案、结果、报告或代码，无法导入空记录。",
    "journal.json exceeds the 12 MiB candidate viewer limit; it will be available for file export only.": "候选记录 journal.json 超过 12 MiB，可导出文件查看，无法在候选方案页面展开。",
    "report.md exceeds the 16 MiB viewer limit; it will be available for file export only.": "报告 report.md 超过 16 MiB，可导出文件查看。",
    "Source files changed; preview the experiment again.": "源文件发生了变化，请重新扫描实验。",
    "Source files changed while being read; preview again.": "读取过程中源文件发生了变化，请重新扫描。",
    "Source files changed during preview; scan again.": "预览过程中源文件发生了变化，请重新扫描。",
    "Source directory changed or is a symbolic link; preview it again.": "源文件夹发生了变化，或属于符号链接，请重新选择并扫描。",
    "Source directory was replaced; preview again.": "源文件夹已被替换，请重新扫描。",
    "This preview expired. Scan the source folder again.": "预览已过期，请重新扫描源文件夹。",
    "Import job was not found. Preview the source again.": "导入记录不存在，请重新扫描源文件夹。",
    "The desktop service is stopping. Reopen BioMaker and preview again.": "本地服务正在退出，请重新打开 BioMaker 后扫描。",
    "An import is already in progress. Wait for it to finish.": "已有导入任务正在进行，请等待它完成。",
    "Choose between 1 and 100 distinct preview entries.": "请选择 1 至 100 个不同的实验。",
    "An entry does not belong to this preview. Scan the folder again.": "所选实验已不属于当前预览，请重新扫描文件夹。",
    "A selected experiment is not eligible. Review the preview warnings.": "部分所选实验无法导入，请查看预览中的说明。",
    "The selected records exceed 256 MiB. Import a smaller batch.": "所选记录超过单批次 256 MiB 上限，请减少选择。",
    "This experiment exceeds the 256 MiB import limit.": "此实验记录超过 256 MiB 导入上限。",
    "There is not enough free disk space for these records and import metadata.": "磁盘剩余空间不足，无法保存这些实验记录及导入信息。",
    "Import was interrupted when the desktop service stopped. Completed copies are preserved; preview again to continue.": "本地服务停止导致导入中断。已完成的副本已保留，请重新扫描后继续。",
    "Import interrupted because the desktop service is closing. Completed copies are preserved.": "本地服务退出导致导入中断，已完成的副本已保留。",
    "Import interrupted because the desktop service is closing.": "本地服务正在退出，导入已中断。",
    "Choose an absolute, regular experiment or logs folder.": "请通过文件夹选择器选择实验目录或 logs 文件夹。",
    "The selected folder is unavailable or is not a directory.": "所选文件夹无法访问，或已不再是文件夹，请重新选择。",
    "This folder already belongs to the desktop workspace. Open its existing records instead.": "此文件夹已属于桌面工作区，可以直接打开其中的实验记录。",
    "This folder is too large or slow to scan. Select an individual experiment folder.": "此文件夹内容过多或读取较慢，请选择单个实验文件夹。",
    "More than 100 child folders were found. Select one experiment folder; no entries were silently omitted.": "文件夹中包含超过 100 个子目录，请选择单个实验文件夹后扫描。",
    "Preview scanning exceeded 15 seconds. Select one experiment folder.": "扫描超过 15 秒，请选择单个实验文件夹后重试。",
    "No experiment folders were found. Select a folder containing run_manifest.json or its parent logs folder.": "未找到实验记录，请选择包含 run_manifest.json 的实验文件夹，或它的 logs 上级目录。",
    "A local import directory already exists but is not registered. Restart BioMaker to recover it; it will not be overwritten.": "本地已有尚未注册的导入副本。请重启 BioMaker 恢复，该副本不会被覆盖。",
    "The copied experiment could not be read by BioMaker; it was not imported.": "BioMaker 无法读取复制的实验记录，本次未导入该实验。",
    "The staged experiment is not readable by BioMaker.": "BioMaker 无法读取临时副本，本次未导入该实验。请检查源实验记录。",
    "The copied experiment contains metadata that BioMaker cannot safely display. No records were published.": "实验记录中的部分信息无法正确显示，本次未导入该实验。请检查源记录后重试。",
    "A managed run already uses this import ID; it will not be overwritten.": "已有实验使用此导入标识，现有记录不会被覆盖。",
    "The destination already exists and will not be overwritten. Preview again.": "目标副本已经存在，不会覆盖。请重新扫描。",
    "The destination appeared during copying; it will not be overwritten.": "复制期间目标副本已由另一操作创建，现有副本不会被覆盖。",
    "A complete local copy was saved but registration failed. Restart BioMaker to recover it; the source was not changed.": "完整副本已保存，但注册失败。请重启 BioMaker 恢复，源文件未修改。",
  };
  if (known[plain]) return t(known[plain], plain);
  let match: RegExpMatchArray | null;
  if ((match = plain.match(/^Not copied \(folders, links, or unsupported files\): (.+)$/s))) return t(`以下项目不复制（文件夹、链接或不支持的文件）：${match[1]}`, plain);
  if ((match = plain.match(/^(.+) is not a valid JSON object\.$/s))) return t(`文件 ${match[1]} 的 JSON 格式无效，无法读取。`, plain);
  if ((match = plain.match(/^(.+) must contain a JSON object\.$/s))) return t(`文件 ${match[1]} 的内容格式不正确，需要 JSON 对象。`, plain);
  if ((match = plain.match(/^(.+) contains a nonfinite JSON number\.$/s))) return t(`文件 ${match[1]} 包含无法读取的非有限数值，请修复源记录后重试。`, plain);
  if ((match = plain.match(/^events\.jsonl line (\d+) is not a valid finite JSON event\. The records were not published; repair the source or select another experiment\.$/))) return t(`events.jsonl 第 ${match[1]} 行的事件格式或数值无效，本次未导入该实验。请修复源记录或选择其他实验。`, plain);
  if ((match = plain.match(/^Source file (.+) (?:changed|changed during copying|grew during copying); preview again\.$/s))) return t(`源文件 ${match[1]} 发生了变化，请重新扫描后再导入。`, plain);
  if ((match = plain.match(/^(.+) exceeds the 64 MiB file limit\.$/s))) return t(`文件 ${match[1]} 超过单文件 64 MiB 上限。`, plain);
  if ((match = plain.match(/^(.+) is not a supported regular file or exceeds its size limit\.$/s))) return t(`文件 ${match[1]} 不是支持的普通文件，或超过大小上限。`, plain);
  if ((match = plain.match(/^This experiment has more than (\d+) top-level entries\. Select a smaller records folder\.$/))) return t(`此实验文件夹包含超过 ${match[1]} 个顶层项目，请选择更小的记录目录。`, plain);
  if (/\[Errno 28\]|No space left on device/.test(plain)) return t(`磁盘空间不足。原始详情：${plain}`, plain);
  return message;
}
