import { app, BrowserWindow, dialog, ipcMain, Menu, nativeTheme, Notification, screen, session, shell, type IpcMainInvokeEvent, type MenuItemConstructorOptions } from 'electron';
import { createWriteStream, existsSync, mkdirSync, readFileSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { Sidecar } from './sidecar';
import { externalURL, isWithin, validateArtifact, validateRequest, validTheme } from './security';
import { readResponse } from './response';
import { RunNotifications, type RunCompletion } from './run-notifications';

app.setName('BioMaker');
// Test/review installs can use a separate profile without touching real projects.
const customData = process.env.BIOMAKER_DESKTOP_USER_DATA || (!app.isPackaged ? process.env.BIOMAKER_DESKTOP_HOME : undefined);
if (customData) app.setPath('userData', path.resolve(customData));
const locked = app.requestSingleInstanceLock();
if (!locked) app.quit();
let mainWindow: BrowserWindow | null = null;
let sidecar: Sidecar | null = null;
let quitting = false;
let serviceReady = false;
let preferences: { theme: 'system' | 'light' | 'dark'; language?: 'zh' | 'en'; notifications: boolean; directories: string[]; bounds?: Electron.Rectangle } = { theme: 'system', notifications: true, directories: [] };
const runNotifications = new RunNotifications();
let notificationTimer: NodeJS.Timeout | null = null;
let notificationPollRunning = false;
let pendingRunId: string | null = null;
const visibleNotifications = new Set<Electron.Notification>();
const t = (zh: string, en: string) => preferences.language === 'en' ? en : zh;
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
    if (sidecar && serviceReady) {
      const ticket = runNotifications.issue();
      const response = await fetch(`${sidecar.origin}/api/runs`, { headers: { Authorization: `Bearer ${sidecar.token}` }, redirect: 'error', signal: AbortSignal.timeout(10_000) });
      observeRuns(await readResponse(response, true), ticket);
    }
  } catch { /* Preserve known active runs across temporary service failures. */ }
  finally {
    notificationPollRunning = false;
    scheduleRunMonitor(runNotifications.activeCount ? 5000 : 30000);
  }
}
function installMenu() {
  const template: MenuItemConstructorOptions[] = [
    ...(process.platform === 'darwin' ? [{ label: 'BioMaker', submenu: [{ role: 'about', label: t('关于 BioMaker', 'About BioMaker') }, { type: 'separator' }, { label: t('设置…', 'Settings…'), accelerator: 'CmdOrCtrl+,', click: () => action('settings') }, { type: 'separator' }, { role: 'hide', label: t('隐藏 BioMaker', 'Hide BioMaker') }, { role: 'hideOthers', label: t('隐藏其它应用', 'Hide Others') }, { role: 'unhide', label: t('显示全部', 'Show All') }, { type: 'separator' }, { role: 'quit', label: t('退出 BioMaker', 'Quit BioMaker') }] } as MenuItemConstructorOptions] : []),
    { label: t('文件', 'File'), submenu: [{ label: t('新建实验', 'New experiment'), accelerator: 'CmdOrCtrl+N', click: () => action('new-experiment') }, { label: t('搜索实验…', 'Search experiments…'), accelerator: 'CmdOrCtrl+K', click: () => action('search') }, { type: 'separator' }, { label: t('打开工作目录', 'Open workspace'), click: () => { void shell.openPath(path.join(app.getPath('userData'), 'workspace')); } }, { type: 'separator' }, { role: process.platform === 'darwin' ? 'close' : 'quit', label: process.platform === 'darwin' ? t('关闭窗口', 'Close Window') : t('退出', 'Quit') }] },
    { label: t('编辑', 'Edit'), submenu: [{ role: 'undo', label: t('撤销', 'Undo') }, { role: 'redo', label: t('重做', 'Redo') }, { type: 'separator' }, { role: 'cut', label: t('剪切', 'Cut') }, { role: 'copy', label: t('复制', 'Copy') }, { role: 'paste', label: t('粘贴', 'Paste') }, { role: 'selectAll', label: t('全选', 'Select All') }] },
    { label: t('显示', 'View'), submenu: [{ label: t('切换侧边栏', 'Toggle sidebar'), accelerator: 'CmdOrCtrl+B', click: () => action('toggle-sidebar') }, { label: t('刷新数据', 'Refresh data'), accelerator: 'CmdOrCtrl+R', click: () => action('refresh') }, { type: 'separator' }, { role: 'resetZoom', label: t('实际大小', 'Actual Size') }, { role: 'zoomIn', label: t('放大', 'Zoom In') }, { role: 'zoomOut', label: t('缩小', 'Zoom Out') }, { type: 'separator' }, { role: 'togglefullscreen', label: t('切换全屏', 'Toggle Full Screen') }, ...(!app.isPackaged ? [{ role: 'toggleDevTools', label: t('开发者工具', 'Developer Tools') } as MenuItemConstructorOptions] : [])] },
    { label: t('窗口', 'Window'), submenu: [{ role: 'minimize', label: t('最小化', 'Minimize') }, { role: 'zoom', label: t('缩放窗口', 'Zoom') }, { label: t('显示 BioMaker', 'Show BioMaker'), click: showWindow }] },
    { label: t('帮助', 'Help'), submenu: [{ label: t('BioMaker 使用文档', 'BioMaker documentation'), click: () => { void shell.openExternal('https://github.com/uni-medical/BioMaker'); } }, { label: t('打开诊断日志', 'Open diagnostic logs'), click: () => { void shell.openPath(path.join(app.getPath('userData'), 'logs')); } }, ...(process.platform !== 'darwin' ? [{ role: 'about', label: t('关于 BioMaker', 'About BioMaker') } as MenuItemConstructorOptions] : [])] },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}
function setupIPC() {
  handle('biomaker:request', async (rawPath, options) => {
    const request = validateRequest(rawPath, options);
    if (!sidecar || !serviceReady) throw new Error(t('BioMaker 服务正在启动，请稍后重试。', 'BioMaker is starting. Please try again shortly.'));
    const isRunResponse = /^\/api\/runs(?:\/[^/]+(?:\/(?:requeue|stop|cancel))?)?$/.test(request.path.split('?')[0]);
    const ticket = isRunResponse ? runNotifications.issue() : 0;
    const response = await fetch(`${sidecar.origin}${request.path}`, { method: request.method, body: request.body, headers: { Authorization: `Bearer ${sidecar.token}`, ...(request.body === undefined ? {} : { 'Content-Type': 'application/json' }) }, redirect: 'error', signal: AbortSignal.timeout(45_000) });
    const result = await readResponse(response, request.path.startsWith('/api/'));
    if (isRunResponse) observeRuns(result, ticket);
    return result;
  });
  handle('biomaker:choose-directory', async () => {
    const result = await dialog.showOpenDialog(mainWindow!, { title: t('选择数据目录', 'Choose a dataset folder'), properties: ['openDirectory', 'createDirectory'], buttonLabel: t('选择此目录', 'Choose folder') });
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
    if (!sidecar) throw new Error(t('本地服务不可用。', 'The local service is unavailable.'));
    const result = await dialog.showSaveDialog(mainWindow!, { title: t('导出实验产物', 'Export experiment output'), defaultPath: path.join(app.getPath('downloads'), name), buttonLabel: t('导出', 'Export') });
    if (result.canceled || !result.filePath) return null;
    const response = await fetch(`${sidecar.origin}/artifacts/runs/${encodeURIComponent(runId)}/${encodeURIComponent(name)}`, { headers: { Authorization: `Bearer ${sidecar.token}` }, redirect: 'error', signal: AbortSignal.timeout(300_000) });
    if (!response.ok || !response.body) throw new Error(`${t('导出失败', 'Export failed')} (${response.status}).`);
    const temporary = `${result.filePath}.biomaker-${process.pid}.part`;
    try { await pipeline(Readable.fromWeb(response.body as any), createWriteStream(temporary, { flags: 'wx', mode: 0o600 })); renameSync(temporary, result.filePath); } catch (error) { rmSync(temporary, { force: true }); throw error; }
    return result.filePath;
  });
  handle('biomaker:app-info', () => ({ version: app.getVersion(), platform: process.platform }));
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
  window.on('close', () => { if (!window.isMaximized() && !window.isFullScreen()) { preferences.bounds = window.getBounds(); savePreferences(); } });
  window.on('closed', () => { if (mainWindow === window) mainWindow = null; });
  window.once('ready-to-show', () => window.show());
  if (devURL) await window.loadURL(devURL); else await window.loadFile(indexPath);
}
function isRectVisible(bounds: Electron.Rectangle, area: Electron.Rectangle) { return bounds.x + bounds.width > area.x + 80 && bounds.x < area.x + area.width - 80 && bounds.y + bounds.height > area.y + 80 && bounds.y < area.y + area.height - 80; }
app.on('second-instance', showWindow);
app.on('activate', () => { if (app.isReady()) showWindow(); });
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('before-quit', event => {
  if (quitting) return;
  event.preventDefault(); quitting = true;
  if (notificationTimer) clearTimeout(notificationTimer);
  for (const notification of visibleNotifications) notification.close();
  void (sidecar?.stop() ?? Promise.resolve()).finally(() => app.quit());
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
  sidecar = new Sidecar({ runtime: app.isPackaged ? path.join(process.resourcesPath, 'runtime') : path.join(desktopRoot, 'build', 'runtime'), service: app.isPackaged ? path.join(process.resourcesPath, 'service', 'main.py') : path.join(desktopRoot, 'service', 'main.py'), data: app.getPath('userData'), version: app.getVersion(), onFailure: message => { serviceReady = false; void dialog.showMessageBox({ type: 'error', title: t('BioMaker 服务', 'BioMaker service'), message, buttons: [t('重新启动', 'Restart'), t('打开诊断日志', 'Open diagnostic logs'), t('稍后', 'Later')], defaultId: 0, cancelId: 2 }).then(async result => { if (result.response === 0) { app.relaunch(); app.quit(); } else if (result.response === 1) await shell.openPath(path.join(app.getPath('userData'), 'logs')); }); } });
  // Render early so users see progress while the bundled service initializes.
  await createWindow();
  try { await sidecar.start(); serviceReady = true; action('refresh'); scheduleRunMonitor(0); } catch (error) {
    const result = await dialog.showMessageBox(mainWindow!, { type: 'error', title: t('BioMaker 启动失败', 'BioMaker could not start'), message: String(error), detail: t('你的实验数据保存在应用工作目录中。可以重试启动，或打开诊断日志。', 'Your experiments are saved in the application workspace. Retry or open the diagnostic logs.'), buttons: [t('重试', 'Retry'), t('打开日志', 'Open logs'), t('退出', 'Quit')], defaultId: 0, cancelId: 2 });
    if (result.response === 0) { app.relaunch(); app.quit(); } else if (result.response === 1) { await shell.openPath(path.join(app.getPath('userData'), 'logs')); } else app.quit();
  }
}).catch(error => { dialog.showErrorBox(t('BioMaker 无法启动', 'BioMaker could not start'), String(error)); app.quit(); });
