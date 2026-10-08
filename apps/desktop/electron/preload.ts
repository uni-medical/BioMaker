import { contextBridge, ipcRenderer } from 'electron';

const invoke = (channel: string, ...args: unknown[]) => ipcRenderer.invoke(channel, ...args);
contextBridge.exposeInMainWorld('biomaker', Object.freeze({
  request: (path: string, options?: { method?: string; body?: unknown }) => invoke('biomaker:request', path, options),
  chooseDirectory: () => invoke('biomaker:choose-directory'),
  openPath: (path: string) => invoke('biomaker:open-path', path),
  saveArtifact: (artifact: { runId: string; name: string }) => invoke('biomaker:save-artifact', artifact),
  getAppInfo: () => invoke('biomaker:app-info'),
  getPreferences: () => invoke('biomaker:get-preferences'),
  setPreferences: (preferences: { theme?: 'system' | 'light' | 'dark'; language?: 'zh' | 'en' }) => invoke('biomaker:set-preferences', preferences),
  openExternal: (url: string) => invoke('biomaker:open-external', url),
  onMenuAction: (callback: (action: string) => void) => {
    if (typeof callback !== 'function') throw new TypeError('Expected a callback');
    const listener = (_event: Electron.IpcRendererEvent, action: string) => callback(action);
    ipcRenderer.on('biomaker:menu-action', listener);
    return () => ipcRenderer.removeListener('biomaker:menu-action', listener);
  },
}));
