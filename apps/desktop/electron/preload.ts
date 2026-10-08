import { contextBridge, ipcRenderer } from 'electron';
import type { ServiceState } from './sidecar';

const invoke = (channel: string, ...args: unknown[]) => ipcRenderer.invoke(channel, ...args);
contextBridge.exposeInMainWorld('biomaker', Object.freeze({
  request: (path: string, options?: { method?: string; body?: unknown }) => invoke('biomaker:request', path, options),
  chooseDirectory: (options?: { purpose?: 'dataset' | 'history' }) => invoke('biomaker:choose-directory', options),
  openPath: (path: string) => invoke('biomaker:open-path', path),
  saveArtifact: (artifact: { runId: string; name: string }) => invoke('biomaker:save-artifact', artifact),
  getAppInfo: () => invoke('biomaker:app-info'),
  getServiceState: () => invoke('biomaker:get-service-state'),
  onServiceState: (callback: (state: ServiceState) => void) => {
    if (typeof callback !== 'function') throw new TypeError('Expected a callback');
    const listener = (_event: Electron.IpcRendererEvent, state: ServiceState) => callback(state);
    ipcRenderer.on('biomaker:service-state', listener);
    return () => ipcRenderer.removeListener('biomaker:service-state', listener);
  },
  getPreferences: () => invoke('biomaker:get-preferences'),
  setPreferences: (preferences: { theme?: 'system' | 'light' | 'dark'; language?: 'zh' | 'en'; notifications?: boolean }) => invoke('biomaker:set-preferences', preferences),
  getPendingRun: () => invoke('biomaker:get-pending-run'),
  acknowledgeOpenRun: (id: string) => invoke('biomaker:acknowledge-open-run', id),
  reportCloseBlock: (reason: 'unsaved' | 'saving') => ipcRenderer.sendSync('biomaker:close-block', reason),
  onOpenRun: (callback: (id: string) => void) => {
    if (typeof callback !== 'function') throw new TypeError('Expected a callback');
    const listener = (_event: Electron.IpcRendererEvent, id: string) => callback(id);
    ipcRenderer.on('biomaker:open-run', listener);
    return () => ipcRenderer.removeListener('biomaker:open-run', listener);
  },
  openExternal: (url: string) => invoke('biomaker:open-external', url),
  onMenuAction: (callback: (action: string) => void) => {
    if (typeof callback !== 'function') throw new TypeError('Expected a callback');
    const listener = (_event: Electron.IpcRendererEvent, action: string) => callback(action);
    ipcRenderer.on('biomaker:menu-action', listener);
    return () => ipcRenderer.removeListener('biomaker:menu-action', listener);
  },
}));
