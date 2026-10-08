import { app, BrowserWindow, dialog, ipcMain, Menu, nativeTheme, screen, session, shell, type IpcMainInvokeEvent, type MenuItemConstructorOptions } from 'electron';
import { createWriteStream, existsSync, mkdirSync, readFileSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { Sidecar } from './sidecar';
import { externalURL, isWithin, validateArtifact, validateRequest, validTheme } from './security';
import { readResponse } from './response';

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
let preferences: { theme: 'system' | 'light' | 'dark'; directories: string[]; bounds?: Electron.Rectangle } = { theme: 'system', directories: [] };
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
function installMenu() {
  const template: MenuItemConstructorOptions[] = [
    ...(process.platform === 'darwin' ? [{ label: 'BioMaker', submenu: [{ role: 'about' }, { type: 'separator' }, { label: '设置…', accelerator: 'CmdOrCtrl+,', click: () => action('settings') }, { type: 'separator' }, { role: 'hide' }, { role: 'hideOthers' }, { role: 'unhide' }, { type: 'separator' }, { role: 'quit' }] } as MenuItemConstructorOptions] : []),
    { label: '文件', submenu: [{ label: '新建实验', accelerator: 'CmdOrCtrl+N', click: () => action('new-experiment') }, { label: '搜索实验…', accelerator: 'CmdOrCtrl+K', click: () => action('search') }, { type: 'separator' }, { label: '打开工作目录', click: () => { void shell.openPath(path.join(app.getPath('userData'), 'workspace')); } }, { type: 'separator' }, { role: process.platform === 'darwin' ? 'close' : 'quit' }] },
    { label: '编辑', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: '显示', submenu: [{ label: '切换侧边栏', accelerator: 'CmdOrCtrl+B', click: () => action('toggle-sidebar') }, { label: '刷新数据', accelerator: 'CmdOrCtrl+R', click: () => action('refresh') }, { type: 'separator' }, { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { type: 'separator' }, { role: 'togglefullscreen' }, ...(!app.isPackaged ? [{ role: 'toggleDevTools' } as MenuItemConstructorOptions] : [])] },
    { label: '窗口', submenu: [{ role: 'minimize' }, { role: 'zoom' }, { label: '显示 BioMaker', click: showWindow }] },
    { label: '帮助', submenu: [{ label: 'BioMaker 使用文档', click: () => { void shell.openExternal('https://github.com/uni-medical/BioMaker'); } }, { label: '打开诊断日志', click: () => { void shell.openPath(path.join(app.getPath('userData'), 'logs')); } }, ...(process.platform !== 'darwin' ? [{ role: 'about' } as MenuItemConstructorOptions] : [])] },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}
function setupIPC() {
  handle('biomaker:request', async (rawPath, options) => {
    const request = validateRequest(rawPath, options);
    if (!sidecar || !serviceReady) throw new Error('BioMaker 服务正在启动，请稍后重试。');
    const response = await fetch(`${sidecar.origin}${request.path}`, { method: request.method, body: request.body, headers: { Authorization: `Bearer ${sidecar.token}`, ...(request.body === undefined ? {} : { 'Content-Type': 'application/json' }) }, redirect: 'error', signal: AbortSignal.timeout(45_000) });
    return readResponse(response, request.path.startsWith('/api/'));
  });
  handle('biomaker:choose-directory', async () => {
    const result = await dialog.showOpenDialog(mainWindow!, { title: '选择数据目录', properties: ['openDirectory', 'createDirectory'], buttonLabel: '选择此目录' });
    if (result.canceled || !result.filePaths[0]) return null;
    const selected = realpathSync(result.filePaths[0]);
    preferences.directories = [...new Set([...preferences.directories, selected])].slice(-100); savePreferences();
    return selected;
  });
  handle('biomaker:open-path', async (candidate) => {
    if (typeof candidate !== 'string' || !path.isAbsolute(candidate) || !existsSync(candidate)) throw new Error('目录不存在。');
    const resolved = realpathSync(candidate);
    if (!statSync(resolved).isDirectory() || /\.(?:app|workflow|bundle|plugin)$/i.test(resolved)) throw new Error('此操作只允许打开普通目录。');
    const roots = [app.getPath('userData'), ...preferences.directories].filter(existsSync).map(value => realpathSync(value));
    if (!roots.some(root => isWithin(resolved, root))) throw new Error('请先通过目录选择器授权访问此位置。');
    const error = await shell.openPath(resolved); if (error) throw new Error(error);
  });
  handle('biomaker:save-artifact', async (artifact) => {
    const { runId, name } = validateArtifact(artifact?.runId, artifact?.name);
    if (!sidecar) throw new Error('本地服务不可用。');
    const result = await dialog.showSaveDialog(mainWindow!, { title: '导出实验产物', defaultPath: path.join(app.getPath('downloads'), name), buttonLabel: '导出' });
    if (result.canceled || !result.filePath) return null;
    const response = await fetch(`${sidecar.origin}/artifacts/runs/${encodeURIComponent(runId)}/${encodeURIComponent(name)}`, { headers: { Authorization: `Bearer ${sidecar.token}` }, redirect: 'error', signal: AbortSignal.timeout(300_000) });
    if (!response.ok || !response.body) throw new Error(`导出失败（${response.status}）。`);
    const temporary = `${result.filePath}.biomaker-${process.pid}.part`;
    try { await pipeline(Readable.fromWeb(response.body as any), createWriteStream(temporary, { flags: 'wx', mode: 0o600 })); renameSync(temporary, result.filePath); } catch (error) { rmSync(temporary, { force: true }); throw error; }
    return result.filePath;
  });
  handle('biomaker:app-info', () => ({ version: app.getVersion(), platform: process.platform }));
  handle('biomaker:get-preferences', () => ({ theme: preferences.theme }));
  handle('biomaker:set-preferences', (value) => { if (!validTheme(value?.theme)) throw new Error('Invalid theme'); preferences.theme = value.theme; nativeTheme.themeSource = value.theme; savePreferences(); return { theme: preferences.theme }; });
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
  void (sidecar?.stop() ?? Promise.resolve()).finally(() => app.quit());
});
if (locked) void app.whenReady().then(async () => {
  mkdirSync(app.getPath('userData'), { recursive: true, mode: 0o700 });
  try { const saved = JSON.parse(readFileSync(preferencesPath(), 'utf8')); preferences = { theme: validTheme(saved?.theme) ? saved.theme : 'system', directories: Array.isArray(saved?.directories) ? saved.directories.filter((item: unknown) => typeof item === 'string' && path.isAbsolute(item)).slice(-100) : [], bounds: saved?.bounds && ['x', 'y', 'width', 'height'].every(key => Number.isFinite(saved.bounds[key])) ? saved.bounds : undefined }; } catch { /* Fresh profile. */ }
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
  sidecar = new Sidecar({ runtime: app.isPackaged ? path.join(process.resourcesPath, 'runtime') : path.join(desktopRoot, 'build', 'runtime'), service: app.isPackaged ? path.join(process.resourcesPath, 'service', 'main.py') : path.join(desktopRoot, 'service', 'main.py'), data: app.getPath('userData'), version: app.getVersion(), onFailure: message => { serviceReady = false; void dialog.showMessageBox({ type: 'error', title: 'BioMaker 服务', message, buttons: ['重新启动', '打开诊断日志', '稍后'], defaultId: 0, cancelId: 2 }).then(async result => { if (result.response === 0) { app.relaunch(); app.quit(); } else if (result.response === 1) await shell.openPath(path.join(app.getPath('userData'), 'logs')); }); } });
  // Render early so users see progress while the bundled service initializes.
  await createWindow();
  try { await sidecar.start(); serviceReady = true; action('refresh'); } catch (error) {
    const result = await dialog.showMessageBox(mainWindow!, { type: 'error', title: 'BioMaker 启动失败', message: String(error), detail: '你的实验数据保存在应用工作目录中。可以重试启动，或打开诊断日志。', buttons: ['重试', '打开日志', '退出'], defaultId: 0, cancelId: 2 });
    if (result.response === 0) { app.relaunch(); app.quit(); } else if (result.response === 1) { await shell.openPath(path.join(app.getPath('userData'), 'logs')); } else app.quit();
  }
}).catch(error => { dialog.showErrorBox('BioMaker 无法启动', String(error)); app.quit(); });
