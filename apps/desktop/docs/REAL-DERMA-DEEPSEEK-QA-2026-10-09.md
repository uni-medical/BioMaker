# BioMaker 0.1.16：真实 DeepSeek 与 DermaMNIST 运行验收

**真实链路已完成；当前软件仍有明确缺陷，不能判定为成熟、零故障交付。**

本次从 `/Applications/BioMaker.app` 的实际用户界面配置服务、选择数据、提交任务、查看活动/候选/代码/执行输出/文件/结果，并通过真实 macOS 保存面板导出报告。调用了真实 DeepSeek，没有固定模型响应、合成数据、测试服务或 UI 测试替换。运行期间没有手工修改生成代码、预测文件或自动报告。

## 运行与独立结果

| 项目 | 实际记录 |
| --- | --- |
| 安装版 | BioMaker 0.1.16，macOS arm64；本轮未重建或替换应用 |
| 服务与模型 | `https://api.deepseek.com`，实际发现并使用 `deepseek-flash`；短响应测试通过 |
| 数据目录 | `/Users/huangziyan/projects/BioMaker/datasets/dermamnist_224` |
| 实验名称 / ID | `dermamnist-deepseek-real-20261009` / `d7dbdd531eaa419c9b33265c42665733` |
| 时间（北京时间） | 2026-10-09 11:28:46 至 11:38:48，端到端 **601.828 秒** |
| 配置 | 3 个探索步骤、1800 秒探索预算、1 个候选执行 worker、单候选 600 秒；最终重训关闭 |
| 完成状态 | SQLite `succeeded`，进程退出码 `0`，55 条事件；完成后无本次实验引擎进程残留 |
| 完整划分 | 训练 **7007** / 验证 **1003** / 测试 **2005**，共 10015 张真实 224×224 RGB 图像 |

1800 秒是探索预算，数据准备和报告等开销不等同于探索时间。1 个 worker 限制候选执行并发，不禁止候选内部数据加载或分类器线程；本轮三份候选代码并行生成，执行串行。

| 候选 | 实际运行方案 | 产物状态 | 验证 macro OvR AUROC | 验证准确率 | 框架执行耗时 |
| --- | --- | --- | ---: | ---: | ---: |
| #1 `55c2a915` | 从头训练 SmallCNN，MPS，2 个完整 epoch | 有效 | 0.825553 | 45.76% | 29.731 秒 |
| **#2 `4de31b50`** | **冻结 MobileNetV3 图像特征 + LightGBM** | **有效、选中** | **0.927266** | **77.77%** | **198.957 秒** |
| #3 `a35a7ffc` | 冻结 ResNet18 特征 + LogisticRegression | 提交缺少 `split`，校验失败 | 无有效排名分数 | — | 24.191 秒 |

第 2 个候选在 MPS 上提取全部 **10015×1024** 图像特征；LightGBM 使用全部 **7007×1024** 训练特征及训练标签拟合，并进行了仅使用训练标签的 3 折 OOF 检查。实际分支由 stdout 和 `run_log.json` 共同确认，没有执行其备用 CNN。图像缩放到 128×128，使用 ImageNet 标准化，seed=42。执行中获取了约 10.24 MB 的 MobileNetV3 权重；这次运行不是完全离线测试。ResNet18 使用生成代码前已存在的缓存权重。

选定后，审计脚本针对最终提交的 **2005 张测试图片** 独立复算，得到 **macro OvR AUROC 0.923346，准确率 77.06%**。此复算在 `run_finished` 后进行，未改变选定方案。两个有效候选的验证分数及最终测试分数均由独立 Mann–Whitney 秩统计计算，并与已安装 sklearn 交叉核验；逐类 AUC 最大差为 `1.11e-16`。没有执行生成评估器来充当独立复算。

验证/测试预测的 ID、顺序和行数与原始 CSV 完全一致；七类概率有限、均在 `[0,1]` 内且逐行归一。最终保存的 `best_solution.py`、`submission.csv` 及 `working_best` 预测与选中候选逐字节匹配。

证据：[最终运行收据](../output/real-dermamnist-20261009/final-verification.json)、[独立审计摘要](../output/real-dermamnist-20261009/independent_audit_summary.txt)、[完整审计](../output/real-dermamnist-20261009/independent_audit_final.json)、[最终测试复算](../output/real-dermamnist-20261009/independent_audit_final_test_4de31b50.json)。

## 数据与评估边界

- 数据准备保留全部原始划分、样本和顺序。候选公开表只有 `agent_train.csv` 含标签；`score_val.csv`、`agent_test.csv` 仅含 `Id,split,path`。原始 val/test、重复 processed 表及含真实测试标签的 `sample_submission.csv` 均未发布到候选 `input`。真值保存在候选工作目录外。
- 初轮数据准备因跨划分裸 `Id` 重复和私有表缺少 `path` 被拒绝；第二轮自动使用 `(split,Id)` 并补齐公开列，校验通过。
- 后台实际上在每个候选完成后计算私有测试指标，写入单独的 `private_metrics.json`，包含失败候选。因此本轮**不满足“只在最终选择后计算一次测试指标”**的严格协议。
- 三个候选在首份私有测试结果之前已经全部生成。审计未发现记录的测试指标进入候选提示、分析、journal 或选择流程；安装代码按有效候选的验证指标选 best，再单独记录私有指标。此结论不构成操作系统级访问隔离证明。
- 数据集执行前后 **10023 个文件** SHA-256 一致，没有增删或修改。所有图像跨划分无字节级重复；训练集和验证集各有一对内部重复，依照保留划分要求未删除。缺少患者标识，无法排除近重复或患者交叉。
- `class=5` 约占各划分 67%，准确率需结合类别不平衡解读。没有验证类别疾病名称映射、官方数据来源、多随机种子方差或临床适用性；这是短程产品验收结果。

证据：[数据审计](../output/real-dermamnist-20261009/audit_results.json)、[原始文件清单](../output/real-dermamnist-20261009/SHA256SUMS.txt)、[结束后文件校验](../output/real-dermamnist-20261009/dataset-final-verification.json)、[评估器审计](../output/real-dermamnist-20261009/independent_audit_evaluator_review.json)。

## 实际桌面操作

| 操作 | 结果 |
| --- | --- |
| 配置真实服务、模型发现与短响应测试 | 通过，保存后实际任务使用 `deepseek-flash` |
| 完整数据任务提交、活动刷新、展开评估事件 | 通过，可见真实执行及评估事件 |
| 候选列表与最终结果跳转 | 通过：3 个候选、2 个有效、#2 被选中；#3 显示 `ArtifactValidationError` |
| 候选代码、执行输出及产物规范预览 | 通过，内容对应本次生成与实际 stdout |
| 报告预览、真实 macOS 保存面板 | 导出成功，5926 字节，与原报告逐字节一致 |
| 完成后的结果页与工作区状态 | 显示已完成、工作区就绪，模型设置和真实实验保留供继续使用 |

截图：[服务连通](../output/real-dermamnist-20261009/01-provider-connected.png)、[真实预算](../output/real-dermamnist-20261009/02-real-task-budget.png)、[完成结果](../output/real-dermamnist-20261009/06-real-run-completed.png)、[原生导出](../output/real-dermamnist-20261009/07-native-real-report-export.png)、[失败候选](../output/real-dermamnist-20261009/09-invalid-candidate-excluded.png)。导出内容核对见[收据](../output/real-dermamnist-20261009/native-export-verification.json)。

## 本次发现和确认的问题

1. **生成评估器的格式校验不完整。** 第 3 个候选的测试提交缺少契约要求的 `split` 列；`evaluator.validate_submission()` 自行推断该列并返回成功，但框架的独立产物校验随后正确拒绝。训练进程退出 0 不等于产物有效，该候选未进入排名。3 步预算已用完，本轮没有自动修复并重跑它。
2. **最佳方案没有保存拟合模型。** 选中候选留下代码、验证/测试预测和运行日志，但未序列化 LightGBM 模型。现有代码可以重新训练，不能直接当作已经保存的模型进行新图片推理。首个 CNN 则保存了 `model.pt`。
3. **自动报告与运行证据不完全一致。** 报告遗漏第 3 个失败候选；把 `unified_memory_rss_proxy` 写成 Peak MPS；声称无法确认最佳分支及缺少逐类 AUC，而 stdout/run_log 和 journal 实际保存了这些信息。最佳 2377.65625 MiB 是进程树峰值 RSS，不能作为独立测得的 GPU 内存。自动报告原文保留，纠正记录见[独立报告审计](../output/real-dermamnist-20261009/report-consistency-audit.md)。
4. **报告 Markdown 表格没有渲染。** 在真实预览中，结果表显示为管道符分隔的普通段落。已目视确认并保存[截图](../output/real-dermamnist-20261009/08-report-table-rendering-issue.png)；当前 `ReactMarkdown` 预览未配置 GFM 表格支持。
5. **关闭最终重训仍出现文件不完整警告。** 本次成功结果页一边正确显示“未启用最终重训”，一边仍显示“部分文件或记录不完整”；与上一轮已确认的可选重训文件误报一致。
6. **严格评估协议仍有差距。** 每步后台计算测试指标、生成评估器在缺类或指标异常时回退到 0.5，都需要明确处理。本轮七类齐全、有效候选 `fallback_used=false`，独立指标吻合，未受回退影响。

静态审计还发现最佳代码中未执行的备用 CNN 分支存在类别权重维度错误与 `drop_last=True`。本次实际走预训练主分支，因此不将其写成已经发生的训练失败。

之前的[实机操作检查](MANUAL-QA-2026-10-09.md)另记录了剪贴板权限、小窗口顶栏裁切和草稿删除错误标题问题。本次保持同一安装版，没有修复交付这些问题。之前 19 套 / 237 项断言是上一轮测试结果，本次真实实验没有重新执行那 19 套，不能混为新的全套回归。

## 产物位置

实际运行日志与结果：
`/Users/huangziyan/Library/Application Support/BioMaker/workspace/logs/0-dermamnist-deepseek-real-20261009`

候选与最佳工作目录：
`/Users/huangziyan/Library/Application Support/BioMaker/workspace/workspaces/0-dermamnist-deepseek-real-20261009`

独立复核脚本、收据及桌面截图：
`/Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/real-dermamnist-20261009`

本次只新增测试证据与本验收文档，未改运行源码、未替换安装包、未推送 GitHub。真实服务配置和实验保留在当前用户的 BioMaker 中。
