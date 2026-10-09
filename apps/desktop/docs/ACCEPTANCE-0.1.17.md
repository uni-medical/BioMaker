# BioMaker 0.1.17：候选进度与运行日志修复

源码版本：`d3132c2a71b170d3b8e616ea7e6fb502b025e82d`。安装目标：`/Applications/BioMaker.app`，macOS arm64。

## 修复

- 候选页以前只读取执行与评估之后才保存的 journal，因此生成和训练期间可能一直为空。现在合并实时快照、生命周期事件与最终 journal，显示生成、等待、执行、评估、失败和已保存结果；自动刷新保留选中方案与代码页。保存前使用“生成序号”，保存后编号与结果页一致。只有保存的记录提供有效性和指标。
- 候选 Python 改为无缓冲输出；框架在进程仍运行时转发新增 stdout/stderr，带候选与输出通道标识，并对跨分块凭据进行脱敏。完整原始执行结果仍保留。普通文件日志不再由 Rich 强制按终端宽度插入断行。
- 日志页按窗口剩余高度布局，长任务和侧栏路径不会把日志底部推出窗口。新增跳到片段开头/最新，明确已载入与总字节数，并保留暂停跟随、搜索、较早片段与完整导出。
- 候选读取使用有界缓存；异常或超大活动/实时快照不会遮蔽可读取的 journal。响应包含父节点结构在内仍受大小限制。

## 已验证证据

| 检查 | 结果 | 收据 |
| --- | --- | --- |
| 引擎源码 | 124 passed | `output/live-progress-0.1.17/core-tests.xml` |
| 桌面服务 | 577 passed | `output/live-progress-0.1.17/service-tests.xml` |
| WebUI | 212 passed | `output/live-progress-0.1.17/webui.xml` |
| 开发版 Electron | 17 套、211 项通过，另通过数据集检查 14 项 | `output/live-progress-0.1.17/source-acceptance.json`、`output/development-dataset-0.1.17.json` |
| 打包版 Electron | 20 套、246 项通过 | `output/packaged-acceptance-0.1.17.json` |
| DMG / ZIP / 应用本体 | 内容一致，deep strict 签名与干净源码来源检查通过 | `output/package-verification-0.1.17.json` |
| 实际升级 | 0.1.16 → 0.1.17，旧应用备份，原配置/草稿/偏好/完整数据库记录未变 | `output/live-progress-0.1.17/retained-profile-before-launch.json` |
| 新引擎安装 | 原生设置界面安装，运行环境与 CPU 自检通过 | `output/live-progress-0.1.17/installed-runtime-ready.png` |
| 已安装应用的合成训练工作流 | 27.836 秒完成，真实队列/引擎/结果/原生导出，安装文件未变 | `output/live-progress-0.1.17/desktop-workflow.json` |
| 安装版引擎与队列 | 引擎离线修复、原生依赖、退出继续/重连/停止、真实 CPU 工作流四类检查通过；测试进程无残留 | `output/live-progress-0.1.17/headless-gates.json`、`headless-cleanup.json` |
| 环境安装及隔离测试后的保留检查 | 新真实实验提交前，原配置/草稿/偏好/全部数据库表/旧引擎与备份均未变 | `output/live-progress-0.1.17/retained-profile-after-setup.json` |

候选界面测试使用隔离文件与真实服务，覆盖 journal 出现前的状态和代码、断线重连、失败/停止、超过 1500 个候选的结果跳转。日志测试覆盖自动追加、暂停选择、来源更换、长行、分页、导出与 960×680 小窗口。它们的合成数据与本地模型响应不作为真实 DeepSeek 或医学性能证据。

合成训练的旧测试选择器在代码页全局匹配 LogisticRegression；新增实时日志后同名内容也出现在隐藏日志页，导致首次测试误选隐藏元素。测试现限定在候选代码容器，随后完整工作流通过。应用源码和安装包不因该测试选择器修正而变化。

安装版引擎检查使用隔离副本及本地响应服务；其中 CPU 工作流产生 1 个真实候选、5 次本地 fixture 请求，外部模型请求为 0。首次沙箱包装遇到 macOS 不允许嵌套 sandbox 的限制；调整测试包装后通过，首次失败、调整原因和脚本哈希均保存在收据中，未修改生产源码。

## 安装后真实运行

真实运行 `72d0d0c2ec7a4ec5b6cbad0f5324ba72`（`dermamnist-live-progress-017`）由安装版原生界面提交，使用保存的 DeepSeek / deepseek-flash 配置和完整 DermaMNIST 划分，没有使用模型响应替身。预算为 1 步、1 worker、候选 300 秒、探索 0.25 小时，最终重训关闭。

2026-10-09 北京时间 12:46:38.640 开始，12:51:22.324 结束，引擎生命周期约 283.683 秒。候选 ID 为 `05d870e85eaa42cfa33d6f09d5f6f122`。以下观测时间来自每 0.25 秒轮询的只读观察器，不等同于精确写入时间：

| 北京时间 | 实际证据 |
| --- | --- |
| 12:50:01.820 | 首次生成快照，journal 尚不存在；原生界面随后自动出现候选行，未点击刷新 |
| 12:50:34.665 | 执行中快照含 17,819 字节代码，journal 尚不存在 |
| 12:50:36.965 | 首次候选 stdout；读取前后 journal 均不存在 |
| 12:51:03.928 | 首次观测到 journal，比首条 stdout 晚 26.963 秒 |
| 12:51:22.514 | 观测到 `run_finished`；原生界面显示已完成 |

观察器收据：`output/live-progress-0.1.17/real-run-observation.json`。它最初等待了错误的预测目录；确认队列实际分配的 `1-` 目录后重新开始，原始空等待保留。短暂的 queued 阶段未被本次轮询捕获，不能声称真实运行观察到全部阶段。

原生界面实际核验了：候选在 journal 前自动出现；选中的代码页保持不变并自动显示生成代码；完成后结果与候选均为 #1，记录指标 0.849805、执行约 27 秒；日志显示全部 17.7 KB，搜索可定位每轮 110 个批次中的 11 条进度行、两轮完成信息、保存预测及检查点信息，底部可见退出码 0 和完整导出按钮。此处“全部”指队列保存的该运行日志，不代表所有底层进程的任意输出。

截图位于 `output/live-progress-0.1.17/`：`real-candidate-generating-before-journal.png`、`real-candidate-completed-code.png`、`real-run-results.png`、`real-run-training-batches.png`、`real-run-completed-log-latest.png`。代码截图拍摄时运行已完成；没有训练期间的原生日志页截图。训练输出提前到达由只读文件观察器证明，界面自动追加由打包版交互测试验证。

最终只读核验确认队列状态 `succeeded`、整次运行和候选退出码均为 0，候选产物校验及评估通过，运行 PID 已退出。保存的代码与 stdout 记录 CompactCNN（288,359 参数）在 MPS 上完成两轮训练，每轮 7,007 个样本，输入缩放到 64×64；检查点为 1,169,661 字节。验证/测试预测分别为 1,003/2,005 行，split、ID 顺序和七类概率结构核对通过。原始数据集 10,023 个文件全部与测试前 SHA256 清单一致，无新增或缺失。收据：`output/live-progress-0.1.17/real-run-final-verification.json`。

验证集 macro AUROC 记录值为 0.849804803882517，本次仅引用保存的评估记录；未独立重算指标、重训或评估私有测试集，不能据此推断临床效果或可复现性能。

## 最终安装完整性

真实运行结束并正常退出应用后，原 provider 配置、旧运行的完整数据库记录、19 个已记录旧产物、旧引擎已记录源码/收据/解释器及旧应用备份均与基线一致。运行库共 2 条，仅新增上述已授权实验；草稿随提交产生正常变更。核验前后无残留应用或实验进程。核验在数据库及 WAL/SHM 的一致副本上进行，未改写原库。

已安装应用与 release 应用均为 2,821 个文件，指纹 `15554d8986e5dd2c9bf2ab47222be570942c94ece78028a380338824e8ef8f65`；deep strict 签名校验通过。DMG/ZIP 哈希与打包验收一致。收据：`output/live-progress-0.1.17/final-installed-integrity.json`。最终验收文档及测试选择器单独提交，安装包仍对应文首的生产源码版本。

## 历史与边界

升级前已有的真实 DermaMNIST 实验仍可打开，结果和候选页均指向 #2，记录 AUROC 为 0.927266。历史日志保持原字节；旧引擎当时未转发的训练行无法补进旧 console，可从候选执行输出查看已保存内容。

本次应用仍采用 ad-hoc 签名；公开分发所需 Developer ID 和 Apple notarization 沿用发布文档的要求。其他系统与硬件需独立验收。
