# BioMaker 0.1.24 验收记录（软件流程通过，保留候选科研限制）

生产冻结提交：`f24b8f2a6401d7445f0bc95343a9fb2e867fda62`。目标为 macOS Apple Silicon 的 `/Applications/BioMaker.app`。

**结论：本轮源码回归、历史源码预检、发布包、实际升级、原生新引擎、安装版合成流程、新真实运行、原生界面/完整导出及最终完整性均通过。真实运行的两个候选均通过软件检查，验证集选中模型并完成一次选定后的私有评估。仍存在下文明确列出的候选任务遵循/科研语义限制和证据范围，不能称模型满足全部科研要求、医学有效性或所有可能问题均已解决。** 本记录汇总当前发布版本的修复和实际验收回执。

## 累计修复：从界面问题到真实运行缺陷

下表概括 `.18–.24` 当前版本保留的修复；候选/日志基础修复始于 `.17`，在后续版本沿用。历史 UI 证据按本轮门禁明确复用，当前后端源码则已重新完整回归，不把不同版本测试数相加。

| 用户可见问题 | 当前行为与证据 |
| --- | --- |
| 候选和代码更新迟、刷新后丢失选择 | 合并实时快照、生命周期事件和最终 journal，显示生成/执行/评估状态，刷新保留选中方案与代码页；只有保存记录确立指标与有效性。[原修复及实测记录][history-live]、本轮[来源门禁][gate] |
| 日志延迟、长行或小窗口裁切、无法取得完整内容 | 无缓冲转发 stdout/stderr，带候选/通道标识及跨块脱敏；文件日志避免终端宽度强制断行。视口约束、暂停、搜索、前翻、回到最新、已载入/总字节及完整导出均保留；前翻替换当前片段，完整文件通过导出获取。[历史 UI 检查][history-gui]及本轮[原生操作/导出][native-ui] |
| 报告表格、失败尝试、关闭重训与错误提示不准确 | 从保存记录确定性生成事实报告，保留失败尝试；GFM 表格可渲染，关闭最终重训有明确说明。报告/日志/草稿走只写原生复制接口，草稿按真实失败操作提示。[累计修复记录][history18]、本轮源码回归和原生结果检查；剪贴板未在本轮重复操作 |
| 指标别名导致最终评分不可用，训练标签丢失或所有候选失败仍退出成功 | `.19/.20` 规范明确等价的指标并在评估器冻结前检查任务/语义/方向；不把不同 AUC 定义或无效分数静默替代。`.21` 守卫训练 target，仅隐藏验证/测试答案；零有效候选保存失败报告、发出失败状态并非零退出。验证集选优、选定后一次私有测试及模型保存/受限重载要求保留。[训练标签修复回执][training-fix]、本轮[完整 Core][core]与[真实运行审计][real] |
| 合法本地推理模块误判为不允许依赖 | `.21–.24` 依次支持固定源码写入、普通 open 写法、可证明数值模板及未调用默认目录 wrapper；递归检查真实源码，未知/冲突写入保守拒绝。四条历史原源码只读预检与本轮两个新候选通过，旧失败不改写 |
| 子进程身份检查持续失败导致清理循环无界 | `.23` 加入清理重试预算、隔离未确认清理的计算租约，避免日志异常覆盖原错误或删除工作区证据；本轮 162 项相关测试属于完整 Core 子集。[故障注入与修复审计][cleanup-history] |

**原 `.20` 极短 metadata 子进程的偶发等待没有被重现。** 复现并修复的是单独注入持续进程组检查失败后的无界清理缺陷，不能声称已找到原偶发事件的唯一根因。默认五秒是清理重试预算，不是整个任务的退出时限或硬实时保证。旧引擎当时未捕获的日志不能补回；旧失败实验、旧评估和原产物均保持原样。

## 本轮修复与约束

`.23` 的真实候选 `b1b7fd36f2154572ba8f37e9f27cdd4b` 在 `main()` 中把固定源码写入 `working/inference.py` 并导入，同时还定义了一个未调用的便利 wrapper。该 wrapper 通过定义时默认目录再次导入同一模块；原先各函数隔离分析不能恢复这段源码证明，导致预检误拒。

本轮仅为**未引用、未调用、未装饰且不经相关反射/命名空间逃逸的顶层函数**，使用定义时捕获的默认路径和已递归验证的唯一模块源码进行额外隔离检查。已有路径分析保持不变，不把默认参数当作实际调用的路径证明，不把函数内部写入提升为模块级事实。真实调用、覆盖参数、`*args`、别名/回调、装饰器、相关反射、未知或冲突写入仍保守拒绝。没有将 `inference` 加入第三方依赖白名单。

独立复审发现并修复了三处具体缺陷：

- 补齐异常处理和模式匹配的名字绑定，以及相对 `pathlib` 导入，避免被覆盖的路径相关名字继续提供证明。
- 对可能的 Path 对象、构造器、模块/对象别名进行保守来源追踪，使未知目标和动态模式的写入失效证明；该信息只用于负向失效，不作为路径或模块的正向证明。
- 第一次正式历史预检仍拒绝实际 `.23` 源码：反射检查把普通 PyTorch `model.eval()` 错当内置 `eval()`。区分普通方法和内置动态求值后，原始保存源码通过；内置 `eval()` 等负向检查保留。

依赖、安装命令、会话逃逸和计算设备约束继续递归检查。**这是静态依赖预检，不是 Python 沙箱，也不是通用调用图证明。** 仍可能保守拒绝未支持的其他合法动态结构。

[第一次正式预检失败][preflight-before]与[定位回执][diagnosis]均保留；修正后重新执行全部正式源码回归。[四个最终独立负向探针][negative]全部按预期拒绝，临时目录始终为空，没有执行样例代码。最终负向探针使用与冻结提交一致的模块源码。

## 本轮新执行的源码测试

| 来源 | 通过 | 跳过 |
| --- | ---: | ---: |
| 完整 Core | 644 | 0 |
| Desktop service | 581 | 0 |
| WebUI Python | 212 | 0 |
| WebUI Node | 19 | 0 |
| Desktop npm | 178 | 1 |
| **源码测试合计** | **1634** | **1** |

另有 TypeScript typecheck 命令通过。177 项本地模块和 162 项进程清理定向测试是 644 项 Core 的子集，不额外累加。唯一跳过是既有 opt-in portable-sidecar integration。Core 命令耗时 56.469 秒，零 failure/error，保留三个既有 SWIG deprecation warnings。

[统一 Core 回归][core]以 `-I -B` 隔离模式使用研究解释器，显式插入并核验当前 checkout 的导入位置；82 个 Python 文件与全部 91 个 Core 源码/测试文件前后哈希相同。[最终 Desktop/WebUI 回归][source]也是本轮新执行，251 个声明范围内源码文件前后相同。本轮早于 `model.eval()` 修正的 `source-regression-final` 不作为最终验收；最终使用 `source-regression-confirmed`。

[冻结源码绑定][binding]独立重读 JUnit、日志、汇总和哈希，将两个源码集合合并后的 **280 个唯一文件**与冻结 Git 提交及当前字节逐一核验；全部匹配，验证时 HEAD 等于冻结提交且 tracked 工作区干净。源码快照覆盖声明范围，未声称覆盖整个 checkout 或应用包。

| 最终收据 | SHA256 |
| --- | --- |
| patch-regression.json | `2044403e20bc13f2d85c43e285ed7622d094e26f4a4b8a00157d493bcabf6f6d` |
| source-regression-confirmed/summary.json | `52afcea9ae17d02ea403285673df647848e193bb8268112e1db70437b2b3a34c` |
| source-binding-verification.json | `fafb95301f7980b17db44c9d784131cc6ba1a082717a974dc12e0cd44999725e` |
| review-negative-probes-final.json | `a6c3d1612360af7514066bac5ff94176abaf55abb45a0af75fd52230be9f4d9c` |

## 四条历史源码兼容性预检

[正式只读预检][preflight]从各历史 journal 提取精确 SHA 对应的源码，在四个空临时目录分别检查：

| 原版本 | 候选 | 主要写入/导入形式 | 本轮静态预检 |
| --- | --- | --- | --- |
| 0.1.20 | `7f991ea79f324dfaac507971f31b69d9` | `Path.write_text` 字面源码 | 通过 |
| 0.1.21 | `fbcb5b3b91284d29bd11e7e157b8010d` | `os.path.join` 加普通 `with open().write()` | 通过 |
| 0.1.22 | `fe73e64d702f435fb0dc8131ce2a6652` | 连续 `replace/repr` 数值模板 | 通过 |
| 0.1.23 | `b1b7fd36f2154572ba8f37e9f27cdd4b` | 固定源码加未调用的默认路径 wrapper；包含普通 `model.eval()` | 通过 |

四个临时目录保持为空，原 journal 在预检前后和冻结绑定时哈希一致。**没有执行旧候选、模型或评估器，没有重训、重新评分或改写历史失败。** 本轮通过只证明这些精确保存源码通过当前准入，不证明它们全部运行逻辑或模型性能正确。`.23` 的失败记录与验收结论保持原样。

最终预检 SHA256：`ddfd036f702ee38cbcedd9b31e9fead2f99919e20d88bc747886368038991659`。所有源码修正、正式回归、历史预检和负向探针均绑定同一最终提交。

## 本轮发布包、实际升级与首启前保留

[发布门禁][gate]已通过，绑定最终提交、冻结的 [release bindings][bindings]、本轮全部源码回归与包验证。`release-bindings.json` 已冻结，SHA 为 `51783a73916f7c47e4c48b2330410a202c505533bb7e44af3e30468f7839ee7a`，后续验收只读取，不改写。

[包验证][package]确认解包应用与 DMG、ZIP 内完整 bundle 一致；三份应用均通过 deep strict 签名验证，DMG 只读挂载且包含 Applications 快捷入口。该检查不等同于已验证开发者签名身份或 Apple 公证。

| 产物 | SHA256 |
| --- | --- |
| 2825 文件的应用 bundle | `b05ee7b296b552362ba8c4a4915adda65bc1003549d8a8847c03c6a0e75827eb` |
| BioMaker-0.1.24-mac-arm64.dmg | `07fcf4cea5210b3f1fa12e6786713515f9ea973acb6c611793344809e3cb17c6` |
| BioMaker-0.1.24-mac-arm64.zip | `457aa4c6e2023b88f0df70f70ca429b14a06f55730151c6c908247a5387c12af` |

[实际升级回执][installation]确认已从 `.23` 升级到 `/Applications/BioMaker.app`，来源、暂存及最终安装 bundle 指纹一致。旧 `.23` 应用完整备份并核验 2825 文件指纹 `32ce9e605d6c52ccd7bd3ff301f4d9e058498fa4a3752b71aec01adc2a761215`。

[首启前保留检查][retention]通过：配置、草稿/偏好、运行数据库全部旧记录与 schema、desktop metadata、8 条旧运行及其产物树均保持原样；12 个旧引擎来源与解释器字节、`.23` 完整引擎与完整证据目录、历史成功/失败记录、旧归档和应用备份均保留。检查时安装应用/profile 相关进程无残留，安装 bundle 与已验证构建一致。这是本轮首启前状态，本轮新运行后的最终保留检查也已通过，见下文。

门禁明确复用 `.18` 的 21 组、253 项历史 GUI 检查：已核验桌面 UI/service/Electron 源码和对应测试夹具保持一致；**这些不是本轮新执行的 GUI 测试，也未加入 1634 项源码通过数。** 本轮新引擎、安装版合成完整流程及真实原生操作另有下文的新回执。

| 收据 | SHA256 |
| --- | --- |
| package-verification-0.1.24.json | `4ae6a1347f398e8dad2c08decdeb10cd34831dc99f1a586c346f4abac7397db5` |
| packaged-acceptance-0.1.24.json | `ff632d60cba54512a1d0c41b36b8838cb27198c11bbd23acb239ebb416491360` |
| package-installation-0.1.24.json | `23363a30378c1a472e229b1e7e1bf7e71ba91c3791b74edf8f192b425feb1009` |
| retained-profile-before-launch.json | `389b87f3da41ea5053005e3076c41512b47070c520ced29c9a15f4c8c353fd0e` |

## 本轮原生新引擎与安装版合成流程

[新引擎回执][engine]已通过：原生设置界面实际完成新 `.24` 引擎安装，依赖检查和固定 CPU 自检显示 ready，[截图][engine-image]绑定 SHA `3aa307b25151006b8b1e7c5f0a04c6c97bbb8b1fe8ec7d5f1aae5c648762e059`。新引擎为 `engine-1f5d31c4251af06117fc174d463da02c1139fefb8e18c75061a6ecc82dc77c5f`；13 个累计变化生产模块与冻结提交/发布门禁一致，依赖 fingerprint 和 native runtime receipt 均核验为当前。此回执只是只读记录已完成安装，没有再次运行健康检查。

**本轮安装版合成完整流程通过，耗时 27.598 秒。** [流程回执][workflow]和[身份/清理绑定][workflow-binding]确认使用实际 `/Applications/BioMaker.app`、隔离 profile 及刚安装的新 `.24` 引擎，没有替换产品源码。界面提交后队列进入 queued → running → succeeded，实际使用 12 行合成训练数据在 CPU 拟合 sklearn LogisticRegression，保存模型，完成两例 `[2,2]` 输出的离线重定位重载探针，并查看结果、候选代码与执行日志。renderer errors 和 fixture errors 均为空，回执保留 7 张流程截图。

合成流程还通过界面按钮调用真实报告导出处理器并读取导出的文件；**保存路径由测试桩提供，没有操作真实 macOS 保存对话框，也未进行导出文件与源文件全字节一致性断言。** 本轮真实原生导出另有下文的通过回执。

该流程使用 4 次本地 fixture 响应，外部 provider 调用为 0，最终重训关闭，不构成真实 DeepSeek 或医学模型性能证据。测试包装限制 Python/uv/训练子进程外网连接，精确 `/bin/ps` 路径仅保留只读进程身份检查例外；候选外部连接拒绝断言通过，未改变产品源码或 Electron Chromium sandbox。安装 bundle 前后保持 2825 文件和原指纹，隔离 profile 相关进程均已清理。

## 新真实运行的软件验收

[原生启动核验][launch]绑定新运行 `0e68daa6c189493a95ce01c9e415e2a6`、最终 `.24` 引擎、精确任务 SHA 和实际启动参数：`deepseek-flash`、两步探索、一个 worker、每候选 600 秒、探索预算 1800 秒、最终重训关闭。生成的 public split 与原始数据逐行/按序一致：训练 7007 行，验证 1003 行，测试 2005 行；只有训练表带规范化的 `label`，其值与原始 `class` 一致，验证/测试标签未出现在这些候选输入表中。

[最终真实运行审计][real]通过：ledger 为 succeeded、return code 0、唯一 run_finished 终态事件，cleanup_pending=false，已记录任务 PID 不再存在。两个候选均成功执行、通过产物及评估检查，未观察到本轮已知的依赖兼容错误。候选验证结果如下，仅按验证集选择，不披露或重新计算私有测试分数：

| 候选 | 验证 AUC | 软件检查 | 选择 |
| --- | ---: | --- | --- |
| `3415a2d293b34e68b6d120b96147f93e` | 0.8639779142087403 | 通过 | 选中 |
| `2ad4f46becfa4742a2653c446a8a258b` | 0.8435289494761707 | 通过 | 未选中 |

两个候选的执行源码、全部原始训练路径/标签、无放回 DataLoader 遍历和保存的运行计数一致，支持各自完成两轮、每轮覆盖 7007 张唯一训练图像；每轮 110 批，最后一批 31 张。这个结论来自精确源码及已有运行证据，没有重新训练或回放采样器，也不是独立证明每次优化器操作。

选中源码 SHA 为 `cd1e0bf14fd3de8f931f95ab88d8a1a62c10969b06369bd8feecf8252feae57b`。其验证/测试预测分别含 1003/2005 行，键和次序与原始 split 一致；七类概率有限、范围正确且逐行归一。选中模型有 1 个 checkpoint、8 个声明文件，文件哈希与运行时重载回执一致；已保存的离线重定位、禁止训练的两例 `[2,7]` 探针通过。审计没有再次加载或执行模型。小样本重载证明有限，不等于对训练正确性或全部预测依赖的通用证明。

私有测试在全部探索与选择结束之后恰好评估一次，选中节点、阶段、提交文件 SHA 和评估回执一致；公共事件不含私有指标值，本次审计没有复制或重算私有分数。确定性报告覆盖全部候选，报告生成区间没有模型请求事件；源码检查也未发现报告函数调用模型，此结论不来自网络抓包。原始数据清单中的 **10023 个文件全部哈希一致**，没有新增、缺失或变更。

[候选任务遵循/科研语义限制][candidate-notes]保留两项具体偏差，不能把“两个有效候选”写成“两个候选都满足全部科研细节”。审计中的 `candidate_training_limitations: []` 只表示覆盖度分类未发现限制，不表示候选没有其他训练或任务遵循限制：

- 选中模型在正式两轮训练之前，对前 8 张训练图像执行了设备 sanity forward/backward，没有 optimizer step；成功路径保留了 BatchNorm running statistics 的变化，源码注释所称的重置没有发生。因此支持“两轮完整优化器训练”，不声称模型只接触过这两轮图像。
- 未选中候选把 `smoke_expected.json` 写自重载后的预测。虽然运行日志记录与内存模型的最大差 `3.576e-07`，它保存的 expected 不是独立的序列化前参考值。选中候选则保存拟合权重的内存 CPU 副本参考值，与重载输出记录差为 0。

审计器最初未识别实际日志中的 `epoch_log` 与 `of_epochs` 字段；[字段兼容修正回执][audit-schema]记录修正前后，仍要求精确 `[7007,7007]`、两轮编号、stdout、源码遍历与原始训练数据一致。仅修改忽略目录中的读取脚本，没有更改产品、运行产物、模型、分数或历史记录，没有重新执行候选。

## 原生日志观察的证据范围

[启动时 AX 记录][launch-ax]显示任务运行中、自动跟随开启，读取于本地 17:37:01 的日志大小为 3.1 KB；[终态 AX 记录][terminal-ax]及[截图][terminal-image]显示本地 17:40:36 已完成，总日志 46.6 KB，其中页面载入 39.5 KB。该原生操作支持观察到日志界面从运行期内容自动更新至终态内容。

**本轮没有保存“训练 batch 正在运行时显示在窗口”的截图，不能据终态 batch 文本回推该项已实测。** [文件观察器][observer]在 17:40:03 才启动，当时第一条 journal 已存在；`live_snapshot_observed_before_journal=false` 与 `candidate_stdout_observed_before_journal=false` 如实保留。它们表示这次较晚观察没有建立“先于 journal”的时间证据，既不是产品未流式输出的证明，也不能改写成通过。观察器确认终态事件，未把较早写入的时间戳当成本轮现场观察时间。

| 本轮回执 | SHA256 |
| --- | --- |
| installed-engine.json | `f7aace721825ec2078a3fa19a3e40bfb54053b95df89d804d9e25f809e04d1e2` |
| desktop-workflow.json | `c2728c324b2a318b295f07d550ef21ae1d313622243e53a2f6fcd2ffa2d4cb56` |
| desktop-workflow-binding.json | `9411f47b700c0d92414243e3aba3961311607bb38d655cc517853e228dbf1c86` |
| real-launch-verification.json | `51034a5a17bb84c1554c5e0f3a15edcce47d3f537f31b6b42e423964d543dcbe` |
| real-run-final-verification.json | `d2ac1398becfedb2690750cc3fdb3c93bf48ec32864fba4cf3958e340fc358fd` |
| candidate-review-notes.json | `b9f74b87f15967a2f54044ba3927a3db082b231d8d6b8ff081abcc4d43aef65a` |

## 本轮原生操作、完整导出与最终完整性

[原生界面回执][native-ui]确认实际安装应用通过 macOS AX 操作并结合截图检查：从运行日志自动更新到终态日志；点击较早内容后显示 7.1 KB 并到达开头，再回到最新的 39.5 KB / 总计 46.6 KB。**前翻替换当前可见片段，不会把所有片段累积到窗口中。** 搜索 `samples_seen=7007` 找到 19 处文本匹配，点击下一项到 2/19 并暂停跟随；匹配包含任务/报告回显，不是 19 个独立训练 epoch。

同轮实际打开两个有效候选的列表和选中候选代码，核验结果显示验证分数 `0.863978` 与最终重训关闭；原生滚动检查报告 GFM 表格。通过 Command-Q 正常退出，进程清理由最终完整性检查另行确认。本轮没有改动或重新测试剪贴板。

[真实原生导出][native-export]通过：分别点击应用导出按钮，在实际 macOS 保存对话框输入目录和文件名并保存，读取导出文件与源文件作完整字节/SHA 对比：

| 导出 | 字节数 | SHA256 |
| --- | ---: | --- |
| 完整 console 日志 | 47765 | `8703ffa8147cfbf62522b0bdfc82f631ea4a85c9045c879e6c568c0620bb93dc` |
| Markdown 报告 | 23049 | `c59198fba75ba6cd3f096f4bcf799dadbdd58462a61865d69bb00a757f264302` |

两者与对应源文件逐字节相同。完整导出独立于当前页面可见片段，未使用合成测试的保存路径桩。为避免文件名误导，将 17:40:36 的终态截图和 AX 文本从 `real-native-live-log` 改名为 `real-native-terminal-log`；[更名回执][filename-correction]确认仅改文件名，字节和 SHA 不变，没有将终态截图转述为训练中的现场截图。

[最终完整性回执][integrity]通过：数据库保留全部 8 条旧运行，只新增授权运行 `0e68daa6c189493a95ce01c9e415e2a6`，合计 9 条；16 棵旧 log/workspace 目录中的 **909 个旧产物文件**及目录/符号链接身份一致，未追随外部符号链接目标。provider 配置不变；草稿随授权任务提交正常变化，不声称整个 profile 或数据库字节完全不变。

12 个旧引擎的源码/回执/解释器身份及字节保留，`.22/.23` 完整引擎与完整证据目录保持原样，旧归档与 `.23` 应用备份通过核验。新引擎源码、实际运行 cwd 与最终提交一致；安装应用和解包 bundle 均保持 2825 文件及既定指纹，新 DMG/ZIP 哈希一致、deep strict 校验通过。检查前后应用/profile 所属进程均为 0，审计没有改写原数据库。

完整性检查结束后，已重新打开应用并停留在本轮已有实验的结果页，方便直接查看；[交接回执][handoff]及[截图][handoff-image]确认 `.24`、原生环境 ready、仍为 9 条运行且未启动新任务。零所属进程结论仅适用于此前完整性检查时刻，交接时应用按意图保持打开。

| 最终回执 | SHA256 |
| --- | --- |
| real-native-ui-verification.json | `ee674780d35c1ef75f9a50b38c1dfff209e298c08c5963c3b339057bb2026b9a` |
| real-native-export-verification.json | `b6775ac81dcdbab3d3d5a410da8f6a45db2ab6c7a921975280b059520652b8db` |
| final-installed-integrity.json | `e4814f25b35ae153b268cb2f81a3485605630ef7b8acf616d643a48c81d68a4e` |
| native-handoff-verification.json | `e87af29bd7aa6fc1f9194d38c6fe61e0f8f00940e7055eb9fe537c70873c812c` |

## 发布与证据范围

本次验收目标为 **macOS Apple Silicon**。[打包日志][package-log]记录的是 ad-hoc 签名，并跳过了 Apple notarization；没有 Developer ID 签名或 Apple 公证，deep strict 通过仅证明所检查包的签名封装完整性。Intel Mac、Windows、Linux 和另一台机器均未作本轮安装验收。

1634 passed / 1 skipped 与 typecheck 是本轮新源码回归；21 套/253 项 GUI 是经身份门禁核对的 `.18` 历史复用；安装版合成流程与本轮原生/真实数据工作流分别有新回执。这些数字和证据类别不相加、不互相替代。软件流程通过不是医学或临床验证，也不消除上述候选任务遵循/科研语义限制。原偶发 metadata 等待未重现、无本轮 batch-live 截图和较晚观察器的限制继续保留。本轮没有 GitHub 同步。

[preflight-before]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/saved-candidate-preflight-before-eval-fix.json
[diagnosis]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/actual-source-preflight-diagnostic.json
[negative]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/review-negative-probes-final.json
[core]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/patch-regression.json
[source]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/source-regression-confirmed/summary.json
[binding]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/source-binding-verification.json
[preflight]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/saved-candidate-preflight.json
[gate]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/packaged-acceptance-0.1.24.json
[bindings]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/release-bindings.json
[package]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/package-verification-0.1.24.json
[installation]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/package-installation-0.1.24.json
[retention]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/retained-profile-before-launch.json
[engine]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/installed-engine.json
[engine-image]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/installed-runtime-ready.jpg
[workflow]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/desktop-workflow.json
[workflow-binding]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/desktop-workflow-binding.json
[launch]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-launch-verification.json
[real]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-run-final-verification.json
[candidate-notes]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/candidate-review-notes.json
[audit-schema]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-run-auditor-schema-adjustment.json
[launch-ax]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-native-launch-ax.txt
[terminal-ax]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-native-terminal-log-ax.txt
[terminal-image]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-native-terminal-log.jpg
[observer]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-run-observation.json
[native-ui]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-native-ui-verification.json
[native-export]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-native-export-verification.json
[filename-correction]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/native-capture-filename-correction.json
[integrity]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/final-installed-integrity.json
[history-live]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/docs/ACCEPTANCE-0.1.17.md
[history-gui]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/packaged-acceptance-0.1.18.json
[history18]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/ACCEPTANCE-0.1.18.draft.md
[training-fix]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.21/training-label-fix-verification.json
[cleanup-history]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.23/process-cleanup-audit/summary.json
[handoff]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/native-handoff-verification.json
[handoff-image]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/real-native-handoff-final.jpg
[package-log]: /Users/huangziyan/projects/BioMaker/desktop-worktree/apps/desktop/output/all-fixes-0.1.24/package-mac.log
