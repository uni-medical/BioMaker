# 桌面交互修复与回归记录（2026-10-09）

此记录对应开发源码验收，构建时版本仍为 0.1.17；输出目录 `output/all-fixes-0.1.18/` 按待发布版本归档。没有在本轮测试中重建安装包或覆盖已安装应用。全部 Electron 套件串行执行，使用各自的临时隔离 profile 和合成历史记录，不调用外部模型，不安装运行引擎，不训练模型。

## 已修复

- **复制**：报告预览、日志片段及草稿恢复统一使用原生 `biomaker:write-clipboard`。只接受字符串，UTF-8 上限 4 MiB。原有主窗口/主 frame 授权及浏览器权限默认拒绝继续生效，renderer 没有剪贴板读取接口。
- **报告表格**：使用 `remark-gfm` 渲染 GFM 表格，保留 ReactMarkdown 默认 URL 检查，并显式跳过原始 HTML。任务描述和 Markdown 文件预览共用这一配置。
- **草稿错误标题**：区分保存、删除、重命名、复制及载入失败；前置自动保存失败仍显示保存失败，避免错误归因。删除失败的真实操作路径已有 UI 断言。
- **可选回执告警**：仅在原始 manifest 明确禁用 final retrain、可用事件与此一致且结果状态为 disabled 时，省略缺失 `final_retrain_result.json` 告警。其他必需文件缺失、回执损坏/过大及冲突事件继续显示问题。
- **小窗口回归**：验证既有布局修复，在 1180×780 和 900×620 内容窗口中展开长路径信息面板、将其滚到底部，再进行设置→英文→返回→中文→返回。没有额外修改现有布局样式。
- **报告设置核对**：设置只提供默认模型及当前实验覆盖模型，没有可编辑的报告模型选项；无需移除 UI。后台兼容配置值未修改。

## 验收结果

| 套件 | 通过检查组 | 证据 |
| --- | ---: | --- |
| 真实系统剪贴板 | 6 | `output/all-fixes-0.1.18/native-clipboard/acceptance.json` |
| 文件预览与导出 | 11 | `output/all-fixes-0.1.18/artifact-preview/acceptance.json` |
| 提示消息生命周期 | 17 | `output/all-fixes-0.1.18/toast-feedback/acceptance.json` |
| 导航、窗口与语言往返 | 15 | `output/all-fixes-0.1.18/run-navigation/acceptance.json` |
| 日志读取、分页与恢复 | 15 | `output/all-fixes-0.1.18/log-reader/acceptance.json` |
| 合计 | **64** | `output/all-fixes-0.1.18/frontend-ui-summary.json` |

真实剪贴板套件不替换生产 IPC、clipboard 或权限处理器。实际点击报告复制、日志复制和真实本地 CAS 冲突后的草稿恢复复制，读取系统剪贴板验证内容，并检查超限/非文本请求和外来窗口写入被拒绝。此前的系统剪贴板数据仅保留在 Electron 主进程内存，退出时恢复并逐格式核对，结果 `originalClipboardRestored: true`；不将原剪贴板内容写入日志或截图。

本机 Electron 44 的原生文本剪贴板会省略开头的编码 BOM；正文 Unicode 及 CRLF 保持一致。报告保存/导出另有字节级验证，原始 BOM、CRLF 均保留。该区别已写入剪贴板验收回执。测试编写期的 API/选择器适配记录保留在 `native-clipboard/harness-notes.json` 和 `preflight-harness-failure.*`，最终结果以 `acceptance.json` 为准。

小窗口两种尺寸共取得 12 次设置前/设置内/返回后的几何记录，右侧面板均存在且边界受限；`shellScroll = 0`、`shellOverflow = 0`。另有普通尺寸语言切换与导航记录。已查看 900×620 的实际截图，顶部栏、信息面板内部滚动条与底部状态栏可见。

Toast 与日志测试中的剪贴板延迟/失败适配器仅用于时序测试，不作为系统剪贴板工作正常的证据。它们已从旧 renderer `navigator.clipboard` 适配点迁移到测试专用 native IPC 适配点。实际系统剪贴板能力由独立的未 mock 套件验证。

全部五个套件结束后再次检查其精确 profile/canonical profile 关联进程，结果为空。测试没有修改真实用户 profile、旧实验或 provider 凭据。

## 非 UI 检查

- `npm run typecheck` 与 `npm run build` 通过。
- Native/package unit tests：159 项，其中 158 通过、1 个 opt-in portable-sidecar 测试跳过、0 失败。真实 sidecar 生命周期在上述 UI 套件中另行覆盖。
- 新增前端内容测试：3 通过，包括 GFM 表格、安全 URL/HTML，以及五类草稿标题的中英文。已加入正常 `npm test` 链路，命令为 `npm run test:report-ui`。
- Result overview：118 项通过。
- `git diff --check -- apps/desktop` 通过。
- 构建保留 Vite 大 bundle 提示；恶意 image URL 测试有 React 空 `src` 提示。这两条均未使检查失败。

这些结果证明所列桌面交互和记录展示路径通过验收，不构成真实模型训练、科学性能、签名/公证或安装包发行验收。
