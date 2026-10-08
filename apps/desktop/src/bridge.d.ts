export interface ServiceState { status: 'starting' | 'ready' | 'reconnecting' | 'failed'; revision: number; message?: string; code?: string }
export interface ArtifactPreview { runId: string; name: string; content: string; sha256: string; size: number; serviceRevision: number }

declare global {
  interface Window {
    biomaker: {
      request<T = unknown>(path: string, options?: { method?: 'GET' | 'POST' | 'PUT'; body?: unknown }): Promise<T>;
      chooseDirectory(options?: { purpose?: 'dataset' | 'history' }): Promise<string | null>;
      openPath(path: string): Promise<void>;
      previewArtifact(artifact: { runId: string; name: string }): Promise<ArtifactPreview>;
      saveArtifact(artifact: { runId: string; name: string; expectedSha256?: string; serviceRevision?: number }): Promise<string | null>;
      getAppInfo(): Promise<{ version: string; platform: string }>;
      getServiceState(): Promise<ServiceState>;
      retryService(): Promise<ServiceState>;
      exportDiagnostics(): Promise<string | null>;
      onServiceState(callback: (state: ServiceState) => void): () => void;
      getPreferences(): Promise<{ theme: 'system' | 'light' | 'dark'; language?: 'zh' | 'en'; notifications: boolean }>;
      setPreferences(preferences: { theme?: 'system' | 'light' | 'dark'; language?: 'zh' | 'en'; notifications?: boolean }): Promise<{ theme: 'system' | 'light' | 'dark'; language?: 'zh' | 'en'; notifications: boolean }>;
      getPendingRun(): Promise<string | null>;
      acknowledgeOpenRun(id: string): Promise<void>;
      reportCloseBlock(reason: 'unsaved' | 'saving'): boolean;
      onOpenRun(callback: (id: string) => void): () => void;
      openExternal(url: string): Promise<void>;
      onMenuAction(callback: (action: string) => void): () => void;
    };
  }
}
