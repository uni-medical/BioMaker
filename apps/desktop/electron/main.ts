import { app, BrowserWindow, dialog, ipcMain, Menu, nativeTheme, Notification, screen, session, shell, type IpcMainInvokeEvent, type MenuItemConstructorOptions } from 'electron';
import { createWriteStream, existsSync, mkdirSync, readFileSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { Sidecar, type ServiceState } from './sidecar';
import { ServiceController } from './service-controller';
import { collectDiagnostics } from './diagnostics';
import { externalURL, isWithin, validateArtifact, validateRequest, validTheme } from './security';
import { readResponse } from './response';
import { RunNotifications, type RunCompletion } from './run-notifications';
import { CloseGuard, type CloseBlockReason } from './close-guard';
import { datasetInspectionBody } from './dataset-inspection';
import { saveProviderSettings } from './provider-settings';

app.setName('BioMaker');
// Test/review installs can use a separate profile without touching real projects.
const customData = process.env.BIOMAKER_DESKTOP_USER_DATA || (!app.isPackaged ? process.env.BIOMAKER_DESKTOP_HOME : undefined);
if (customData) app.setPath('userData', path.resolve(customData));
const locked = app.requestSingleInstanceLock();
if (!locked) app.quit();
let mainWindow: BrowserWindow | null = null;
let serviceController: ServiceController<Sidecar> | null = null;
let diagnosticsExport: Promise<string | null> | null = null;
let quitting = false;
let serviceState: ServiceState = { status: 'starting', revision: 0 };
let preferences: { theme: 'system' | 'light' | 'dark'; language?: 'zh' | 'en'; notifications: boolean; directories: string[]; bounds?: Electron.Rectangle } = { theme: 'system', notifications: true, directories: [] };
const runNotifications = new RunNotifications();
let notificationTimer: NodeJS.Timeout | null = null;
let notificationPollRunning = false;
let pendingRunId: string | null = null;
const visibleNotifications = new Set<Electron.Notification>();
let closeBlockReason: CloseBlockReason = 'unsaved';
const t = (zh: string, en: string) => preferences.language === 'en' ? en : zh;
const closeGuard = new CloseGuard({
  windowId: () => mainWindow?.id ?? null,
  closeWindow: () => mainWindow?.close(),
  confirmClose: reason => {
    const saving = reason === 'saving';
    const response = dialog.showMessageBoxSync(mainWindow!, {
      type: 'warning', title: t('模型配置', 'Model settings'),
      message: saving ? t('配置正在保存，请稍候。', 'Settings are being saved. Please wait.') : t('模型配置尚未保存。', 'Model settings have not been saved.'),
      detail: saving ? t('保存完成后可以关闭窗口或退出。', 'You can close the window or quit after saving finishes.') : t('放弃更改后，未保存的内容将丢失。', 'Discarding changes will lose the settings you have not saved.'),
      buttons: saving ? [t('继续等待', 'Keep waiting')] : [t('继续编辑', 'Keep editing'), t('放弃更改', 'Discard changes')],
      defaultId: 0, cancelId: 0, noLink: true,
    });
    return !saving && response === 1;
  },
  stop: async () => {
    quitting = true;
    if (notificationTimer) clearTimeout(notificationTimer);
    for (const notification of visibleNotifications) notification.close();
    await serviceController?.stop();
  },
  quit: () => app.quit(),
});
const desktopRoot = path.resolve(__dirname, '..');
const indexPath = path.join(desktopRoot, 'dist', 'index.html');
const devURL = !app.isPackaged && process.env.BIOMAKER_DESKTOP_DEV_URL === 'http://127.0.0.1:5173' ? process.env.BIOMAKER_DESKTOP_DEV_URL : null;
const CSP = `default-src 'self'; script-src 'self'${devURL ? " 'unsafe-inline'" : ''}; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self'${devURL ? ' http://127.0.0.1:5173 ws://127.0.0.1:5173' : ''}; object-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'`;
const preferencesPath = () => path.join(app.getPath('userData'), 'desktop-preferences.json');
function savePreferences() {
  const target = preferencesPath(); const temp = `${target}.tmp`;
  writeFileSync(temp, JSON.stringify(preferences, null, 2), { mode: 0o600 }); renameSync(temp, target);
}
function isRendererURL(raw: string): boolean {
  try { const url = new URL(raw); return devURL ? url.origin === devURL : url.protocol === 'file:' && fileURLToPath(url) === indexPath; } catch { return false; }
}
function senderAllowed(event: IpcMainInvokeEvent) {
  if (!mainWindow || event.sender !== mainWindow.webContents || event.senderFrame !== mainWindow.webContents.mainFrame || !isRendererURL(event.senderFrame.url)) throw new Error('Untrusted application frame');
}
function handle(channel: string, callback: (...args: any[]) => unknown) {
  ipcMain.handle(channel, (event, ...args) => { senderAllowed(event); return callback(...args); });
}
function action(name: string) { mainWindow?.webContents.send('biomaker:menu-action', name); }
function publishServiceState(state: ServiceState) {
  serviceState = state;
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('biomaker:service-state', state);
  if (state.status === 'ready') scheduleRunMonitor(0);
}
async function fetchService(url: string, options: RequestInit): Promise<Response> {
  const current = serviceController?.getService();
  const revision = serviceState.revision;
  try { return await fetch(url, options); }
  catch (error) {
    // HTTP errors are handled by the caller. Only a transport failure needs a
    // health probe, so invalid settings or an unavailable artifact never turn
    // a healthy local service into a disconnected UI.
    // A request from a previous service generation must not disturb one that
    // has already reconnected while that request was still settling.
    if (serviceController?.getService() === current && serviceState.revision === revision) void current?.checkConnection();
    throw error;
  }
}
function showWindow() { if (!app.isReady()) { app.once('ready', showWindow); return; } if (!mainWindow) void createWindow(); else { if (mainWindow.isMinimized()) mainWindow.restore(); mainWindow.show(); mainWindow.focus(); } }
function notifyCompletion(completion: RunCompletion) {
  if (quitting || !preferences.notifications || mainWindow?.isFocused() || !Notification.isSupported()) return;
  // System notifications can appear on a lock screen: omit task names, datasets,
  // provider errors and metrics, while retaining an internal navigation target.
  const notification = new Notification({
    title: completion.outcome === 'complete' ? t('实验已完成', 'Experiment complete') : completion.outcome === 'failed' ? t('实验运行失败', 'Experiment failed') : t('实验已中断', 'Experiment interrupted'),
    body: t('点击打开 BioMaker，查看实验结果与日志。', 'Open BioMaker to view the experiment results and logs.'),
    silent: true,
  });
  notification.on('click', () => {
    pendingRunId = completion.id;
    showWindow();
    mainWindow?.webContents.send('biomaker:open-run', completion.id);
  });
  const release = () => visibleNotifications.delete(notification);
  notification.on('close', release);
  notification.on('failed', release);
  visibleNotifications.add(notification);
  if (visibleNotifications.size > 20) {
    const oldest = visibleNotifications.values().next().value!;
    oldest.close(); visibleNotifications.delete(oldest);
  }
  try { notification.show(); } catch { release(); }
}
function observeRuns(payload: unknown, ticket: number) {
  const hadActiveRuns = runNotifications.activeCount > 0;
  for (const completion of runNotifications.observe(payload, ticket)) notifyCompletion(completion);
  if (!hadActiveRuns && runNotifications.activeCount) scheduleRunMonitor(1000);
}
function scheduleRunMonitor(delay: number) {
  if (quitting) return;
  if (notificationTimer) clearTimeout(notificationTimer);
  notificationTimer = setTimeout(() => { void pollNotificationRuns(); }, delay);
  notificationTimer.unref();
}
async function pollNotificationRuns() {
  if (quitting || notificationPollRunning) return;
  notificationPollRunning = true;
  try {
    const sidecar = serviceController?.getService();
    if (sidecar && serviceState.status === 'ready') {
      const ticket = runNotifications.issue();
      const response = await fetchService(`${sidecar.origin}/api/runs`, { headers: { Authorization: `Bearer ${sidecar.token}` }, redirect: 'error', signal: AbortSignal.timeout(10_000) });
      observeRuns(await readResponse(response, true), ticket);
    }
  } catch { /* Preserve known active runs across temporary service failures. */ }
  finally {
    notificationPollRunning = false;
    scheduleRunMonitor(runNotifications.activeCount ? 5000 : 30000);
  }
}
const runtimePath = () => app.isPackaged ? path.join(process.resourcesPath, 'runtime') : path.join(desktopRoot, 'build', 'runtime');
const servicePath = () => app.isPackaged ? path.join(process.resourcesPath, 'service', 'main.py') : path.join(desktopRoot, 'service', 'main.py');
function exportDiagnostics(): Promise<string | null> {
  if (diagnosticsExport) return diagnosticsExport;
  const operation = (async () => {
    const options: Electron.SaveDialogOptions = {
      title: t('导出诊断', 'Export diagnostics'),
      defaultPath: path.join(app.getPath('downloads'), `BioMaker-diagnostics-${new Date().toISOString().replace(/[:.]/g, '-')}.json`),
      buttonLabel: t('保存', 'Save'),
      filters: [{ name: t('诊断报告', 'Diagnostic report'), extensions: ['json'] }],
      message: t('仅包含应用版本、运行环境与服务状态。不含原始日志、任务、API Key 或数据文件。', 'Includes application versions, runtime availability and service state. Excludes raw logs, tasks, API keys and data files.'),
    };
    const result = mainWindow && !mainWindow.isDestroyed() ? await dialog.showSaveDialog(mainWindow, options) : await dialog.showSaveDialog(options);
    if (result.canceled || !result.filePath) return null;
    const report = await collectDiagnostics({ appVersion: app.getVersion(), packaged: app.isPackaged, runtimeDirectory: runtimePath(), servicePath: servicePath(), dataDirectory: app.getPath('userData'), serviceState, events: serviceController?.getDiagnostics() ?? [] });
    const temporary = `${result.filePath}.biomaker-${randomUUID()}.tmp`;
    try {
      writeFileSync(temporary, JSON.stringify(report, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
      renameSync(temporary, result.filePath);
    } catch {
      try { rmSync(temporary, { force: true }); } catch { /* Preserve the export failure. */ }
      throw new Error(t('无法保存诊断文件，请选择可写的位置后重试。', 'Could not save diagnostics. Choose a writable location and try again.'));
    }
    return result.filePath;
  })();
  diagnosticsExport = operation;
  void operation.finally(() => { if (diagnosticsExport === operation) diagnosticsExport = null; }).catch(() => {});
  return operation;
}
function installMenu() {
  const template: MenuItemConstructorOptions[] = [
    ...(process.platform === 'darwin' ? [{ label: 'BioMaker', submenu: [{ role: 'about', label: t('关于 BioMaker', 'About BioMaker') }, { type: 'separator' }, { label: t('设置…', 'Settings…'), accelerator: 'CmdOrCtrl+,', click: () => action('settings') }, { type: 'separator' }, { role: 'hide', label: t('隐藏 BioMaker', 'Hide BioMaker') }, { role: 'hideOthers', label: t('隐藏其它应用', 'Hide Others') }, { role: 'unhide', label: t('显示全部', 'Show All') }, { type: 'separator' }, { role: 'quit', label: t('退出 BioMaker', 'Quit BioMaker') }] } as MenuItemConstructorOptions] : []),
    { label: t('文件', 'File'), submenu: [{ label: t('新建实验', 'New experiment'), accelerator: 'CmdOrCtrl+N', click: () => action('new-experiment') }, { label: t('搜索实验…', 'Search experiments…'), accelerator: 'CmdOrCtrl+K', click: () => action('search') }, { type: 'separator' }, { label: t('打开工作目录', 'Open workspace'), click: () => { void shell.openPath(path.join(app.getPath('userData'), 'workspace')); } }, { type: 'separator' }, { role: process.platform === 'darwin' ? 'close' : 'quit', label: process.platform === 'darwin' ? t('关闭窗口', 'Close Window') : t('退出', 'Quit') }] },
    { label: t('编辑', 'Edit'), submenu: [{ role: 'undo', label: t('撤销', 'Undo') }, { role: 'redo', label: t('重做', 'Redo') }, { type: 'separator' }, { role: 'cut', label: t('剪切', 'Cut') }, { role: 'copy', label: t('复制', 'Copy') }, { role: 'paste', label: t('粘贴', 'Paste') }, { role: 'selectAll', label: t('全选', 'Select All') }] },
    { label: t('显示', 'View'), submenu: [{ label: t('切换侧边栏', 'Toggle sidebar'), accelerator: 'CmdOrCtrl+B', click: () => action('toggle-sidebar') }, { label: t('刷新数据', 'Refresh data'), accelerator: 'CmdOrCtrl+R', click: () => action('refresh') }, { type: 'separator' }, { role: 'resetZoom', label: t('实际大小', 'Actual Size') }, { role: 'zoomIn', label: t('放大', 'Zoom In') }, { role: 'zoomOut', label: t('缩小', 'Zoom Out') }, { type: 'separator' }, { role: 'togglefullscreen', label: t('切换全屏', 'Toggle Full Screen') }, ...(!app.isPackaged ? [{ role: 'toggleDevTools', label: t('开发者工具', 'Developer Tools') } as MenuItemConstructorOptions] : [])] },
    { label: t('窗口', 'Window'), submenu: [{ role: 'minimize', label: t('最小化', 'Minimize') }, { role: 'zoom', label: t('缩放窗口', 'Zoom') }, { label: t('显示 BioMaker', 'Show BioMaker'), click: showWindow }] },
    { label: t('帮助', 'Help'), submenu: [{ label: t('BioMaker 使用文档', 'BioMaker documentation'), click: () => { void shell.openExternal('https://github.com/uni-medical/BioMaker'); } }, { label: t('导出诊断…', 'Export diagnostics…'), click: () => { void exportDiagnostics().catch(error => dialog.showErrorBox(t('诊断导出失败', 'Diagnostic export failed'), String(error))); } }, { label: t('打开本地日志文件夹', 'Open local log folder'), click: () => { void shell.openPath(path.join(app.getPath('userData'), 'logs')); } }, ...(process.platform !== 'darwin' ? [{ role: 'about', label: t('关于 BioMaker', 'About BioMaker') } as MenuItemConstructorOptions] : [])] },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}
function setupIPC() {
  // Synchronous only at beforeunload: the reason must arrive before Electron's
  // will-prevent-unload event. No form values or secrets cross this channel.
  ipcMain.on('biomaker:close-block', (event, reason) => {
    try {
      senderAllowed(event);
      if (reason !== 'unsaved' && reason !== 'saving') throw new Error('Invalid close protection');
      closeBlockReason = reason;
      event.returnValue = true;
    } catch { event.returnValue = false; }
  });
  handle('biomaker:request', async (rawPath, options) => {
    const request = validateRequest(rawPath, options);
    const pathname = decodeURIComponent(request.path.split('?')[0]).replace(/\/+$/, '');
    if (request.method === 'POST' && pathname === '/api/desktop/dataset/inspect') {
      request.body = datasetInspectionBody(request.body ? JSON.parse(request.body) : null, [
        ...preferences.directories,
        path.join(app.getPath('userData'), 'workspace'),
      ]);
    }
    if (request.method === 'POST' && pathname === '/api/desktop/history/preview') {
      const body = request.body ? JSON.parse(request.body) : null;
      if (!body || typeof body.path !== 'string' || !path.isAbsolute(body.path) || !existsSync(body.path)) throw new Error(t('请选择已有实验文件夹。', 'Choose an existing experiment folder.'));
      const resolved = realpathSync(body.path);
      const authorized = preferences.directories.some(root => {
        try { return realpathSync(root) === root && isWithin(resolved, root); } catch { return false; }
      });
      if (!statSync(resolved).isDirectory() || !authorized) throw new Error(t('请先通过目录选择器授权访问此实验文件夹。', 'Choose this experiment folder using the folder picker to grant access first.'));
      request.body = JSON.stringify({ ...body, path: resolved });
    }
    const sidecar = serviceController?.getService();
    if (!sidecar || serviceState.status !== 'ready') throw new Error(serviceState.message ?? (serviceState.status === 'failed' ? t('本地服务不可用。请重试本地服务，或导出诊断。', 'The local service is unavailable. Retry the service or export diagnostics.') : serviceState.status === 'reconnecting' ? t('正在重新连接 BioMaker 服务，请稍后重试。', 'BioMaker is reconnecting. Please try again shortly.') : t('BioMaker 服务正在启动，请稍后重试。', 'BioMaker is starting. Please try again shortly.')));
    const isRunResponse = /^\/api\/runs(?:\/[^/]+(?:\/(?:requeue|stop|cancel))?)?$/.test(request.path.split('?')[0]);
    const ticket = isRunResponse ? runNotifications.issue() : 0;
    const send = () => fetchService(`${sidecar.origin}${request.path}`, { method: request.method, body: request.body, headers: { Authorization: `Bearer ${sidecar.token}`, ...(request.body === undefined ? {} : { 'Content-Type': 'application/json' }) }, redirect: 'error', signal: AbortSignal.timeout(45_000) });
    if (request.method === 'PUT' && pathname === '/api/env') return saveProviderSettings(send);
    const response = await send();
    const result = await readResponse(response, request.path.startsWith('/api/'));
    if (isRunResponse) observeRuns(result, ticket);
    return result;
  });
  handle('biomaker:choose-directory', async (options) => {
    if (options !== undefined && (!options || typeof options !== 'object' || Array.isArray(options) || Object.keys(options).some(key => key !== 'purpose') || (options.purpose !== undefined && options.purpose !== 'dataset' && options.purpose !== 'history'))) throw new Error('Invalid directory purpose');
    const history = options?.purpose === 'history';
    const result = await dialog.showOpenDialog(mainWindow!, { title: history ? t('选择已有实验文件夹', 'Choose an existing experiment folder') : t('选择数据目录', 'Choose a dataset folder'), properties: history ? ['openDirectory'] : ['openDirectory', 'createDirectory'], buttonLabel: history ? t('选择此文件夹', 'Choose folder') : t('选择此目录', 'Choose folder') });
    if (result.canceled || !result.filePaths[0]) return null;
    const selected = realpathSync(result.filePaths[0]);
    preferences.directories = [...new Set([...preferences.directories, selected])].slice(-100); savePreferences();
    return selected;
  });
  handle('biomaker:open-path', async (candidate) => {
    if (typeof candidate !== 'string' || !path.isAbsolute(candidate) || !existsSync(candidate)) throw new Error(t('目录不存在。', 'The folder does not exist.'));
    const resolved = realpathSync(candidate);
    if (!statSync(resolved).isDirectory() || /\.(?:app|workflow|bundle|plugin)$/i.test(resolved)) throw new Error(t('此操作只允许打开普通目录。', 'Only regular folders can be opened.'));
    const roots = [app.getPath('userData'), ...preferences.directories].filter(existsSync).map(value => realpathSync(value));
    if (!roots.some(root => isWithin(resolved, root))) throw new Error(t('请先通过目录选择器授权访问此位置。', 'Choose this location using the folder picker to grant access first.'));
    const error = await shell.openPath(resolved); if (error) throw new Error(error);
  });
  handle('biomaker:save-artifact', async (artifact) => {
    const { runId, name } = validateArtifact(artifact?.runId, artifact?.name);
    const sidecar = serviceController?.getService();
    if (!sidecar || serviceState.status !== 'ready') throw new Error(t('本地服务不可用。', 'The local service is unavailable.'));
    const result = await dialog.showSaveDialog(mainWindow!, { title: t('导出实验产物', 'Export experiment output'), defaultPath: path.join(app.getPath('downloads'), name), buttonLabel: t('导出', 'Export') });
    if (result.canceled || !result.filePath) return null;
    const response = await fetchService(`${sidecar.origin}/artifacts/runs/${encodeURIComponent(runId)}/${encodeURIComponent(name)}`, { headers: { Authorization: `Bearer ${sidecar.token}` }, redirect: 'error', signal: AbortSignal.timeout(300_000) });
    if (!response.ok || !response.body) throw new Error(`${t('导出失败', 'Export failed')} (${response.status}).`);
    const temporary = `${result.filePath}.biomaker-${process.pid}.part`;
    try { await pipeline(Readable.fromWeb(response.body as any), createWriteStream(temporary, { flags: 'wx', mode: 0o600 })); renameSync(temporary, result.filePath); } catch (error) { rmSync(temporary, { force: true }); throw error; }
    return result.filePath;
  });
  handle('biomaker:app-info', () => ({ version: app.getVersion(), platform: process.platform }));
  handle('biomaker:get-service-state', () => ({ ...serviceState }));
  handle('biomaker:retry-service', () => {
    if (quitting || !serviceController) throw new Error(t('应用正在退出，请稍后重新打开。', 'The application is quitting. Open it again shortly.'));
    return serviceController.retry();
  });
  handle('biomaker:export-diagnostics', () => exportDiagnostics());
  handle('biomaker:get-preferences', () => ({ theme: preferences.theme, language: preferences.language, notifications: preferences.notifications }));
  handle('biomaker:set-preferences', (value) => {
    if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(key => !['theme', 'language', 'notifications'].includes(key))) throw new Error('Invalid preferences');
    if ('theme' in value && !validTheme(value.theme)) throw new Error('Invalid theme');
    if ('language' in value && value.language !== 'zh' && value.language !== 'en') throw new Error('Invalid language');
    if ('notifications' in value && typeof value.notifications !== 'boolean') throw new Error('Invalid notification preference');
    if ('theme' in value) { preferences.theme = value.theme; nativeTheme.themeSource = value.theme; }
    if ('language' in value) preferences.language = value.language;
    if ('notifications' in value) preferences.notifications = value.notifications;
    if (!preferences.notifications) for (const notification of visibleNotifications) notification.close();
    savePreferences(); installMenu();
    return { theme: preferences.theme, language: preferences.language, notifications: preferences.notifications };
  });
  handle('biomaker:get-pending-run', () => pendingRunId);
  handle('biomaker:acknowledge-open-run', (id) => { if (typeof id !== 'string' || id.length > 180) throw new Error('Invalid run identifier'); if (id === pendingRunId) pendingRunId = null; });
  handle('biomaker:open-external', (url) => shell.openExternal(externalURL(url)));
}
async function createWindow() {
  if (mainWindow) { mainWindow.show(); mainWindow.focus(); return; }
  let bounds = preferences.bounds;
  if (bounds && (!Number.isFinite(bounds.width) || !Number.isFinite(bounds.height) || !screen.getAllDisplays().some(display => isRectVisible(bounds!, display.workArea)))) bounds = undefined;
  mainWindow = new BrowserWindow({ width: bounds?.width ?? 1360, height: bounds?.height ?? 900, x: bounds?.x, y: bounds?.y, minWidth: 900, minHeight: 620, title: 'BioMaker', show: false, backgroundColor: nativeTheme.shouldUseDarkColors ? '#171918' : '#f7f7f5', titleBarStyle: 'hiddenInset', trafficLightPosition: { x: 20, y: 21 }, autoHideMenuBar: process.platform !== 'darwin', webPreferences: { preload: path.join(__dirname, 'preload.js'), nodeIntegration: false, contextIsolation: true, sandbox: true, webSecurity: true, spellcheck: false, devTools: !app.isPackaged } });
  const window = mainWindow;
  window.webContents.setWindowOpenHandler(({ url }) => { try { void shell.openExternal(externalURL(url)); } catch { /* Unapproved windows stay closed. */ } return { action: 'deny' }; });
  window.webContents.on('will-navigate', (event, url) => { if (!isRendererURL(url)) event.preventDefault(); });
  window.webContents.on('will-attach-webview', event => event.preventDefault());
  window.webContents.on('will-prevent-unload', event => {
    if (closeGuard.blockedUnload(closeBlockReason)) event.preventDefault();
  });
  window.on('close', () => { if (!window.isMaximized() && !window.isFullScreen()) { preferences.bounds = window.getBounds(); savePreferences(); } });
  window.on('closed', () => { if (mainWindow === window) mainWindow = null; closeGuard.windowClosed(); });
  window.once('ready-to-show', () => window.show());
  if (devURL) await window.loadURL(devURL); else await window.loadFile(indexPath);
}
function isRectVisible(bounds: Electron.Rectangle, area: Electron.Rectangle) { return bounds.x + bounds.width > area.x + 80 && bounds.x < area.x + area.width - 80 && bounds.y + bounds.height > area.y + 80 && bounds.y < area.y + area.height - 80; }
app.on('second-instance', showWindow);
app.on('activate', () => { if (app.isReady()) showWindow(); });
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('before-quit', event => {
  if (closeGuard.readyToQuit) return;
  event.preventDefault();
  closeGuard.requestQuit();
});
if (locked) void app.whenReady().then(async () => {
  mkdirSync(app.getPath('userData'), { recursive: true, mode: 0o700 });
  try { const saved = JSON.parse(readFileSync(preferencesPath(), 'utf8')); preferences = { theme: validTheme(saved?.theme) ? saved.theme : 'system', language: saved?.language === 'en' || saved?.language === 'zh' ? saved.language : undefined, notifications: saved?.notifications !== false, directories: Array.isArray(saved?.directories) ? saved.directories.filter((item: unknown) => typeof item === 'string' && path.isAbsolute(item)).slice(-100) : [], bounds: saved?.bounds && ['x', 'y', 'width', 'height'].every(key => Number.isFinite(saved.bounds[key])) ? saved.bounds : undefined }; } catch { /* Fresh profile. */ }
  nativeTheme.themeSource = preferences.theme;
  app.setAboutPanelOptions({ applicationName: 'BioMaker', applicationVersion: app.getVersion(), copyright: 'BioMaker · Biomedical AI research workspace', website: 'https://github.com/uni-medical/BioMaker' });
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  session.defaultSession.setPermissionCheckHandler(() => false);
  session.defaultSession.on('will-download', event => event.preventDefault());
  session.defaultSession.webRequest.onBeforeRequest((details, callback) => {
    let allowed = false;
    try {
      const url = new URL(details.url);
      allowed = devURL ? url.origin === devURL || url.origin === 'ws://127.0.0.1:5173' : url.protocol === 'file:' && isWithin(fileURLToPath(url), path.join(desktopRoot, 'dist'));
      if (url.protocol === 'data:' || url.protocol === 'blob:' || details.url === 'about:blank') allowed = true;
    } catch { /* Malformed URLs remain blocked. */ }
    callback({ cancel: !allowed });
  });
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => callback({ responseHeaders: { ...details.responseHeaders, 'Content-Security-Policy': [CSP], 'X-Content-Type-Options': ['nosniff'] } }));
  installMenu(); setupIPC();
  serviceController = new ServiceController<Sidecar>({
    factory: callbacks => new Sidecar({ runtime: runtimePath(), service: servicePath(), data: app.getPath('userData'), version: app.getVersion(), ...callbacks }),
    onState: publishServiceState,
  });
  // Window and IPC remain usable even if service construction or startup fails.
  await createWindow();
  await serviceController.start();
}).catch(error => { dialog.showErrorBox(t('BioMaker 无法启动', 'BioMaker could not start'), String(error)); app.quit(); });
