export {};

declare global {
  interface Window {
    biomaker: {
      request<T = unknown>(path: string, options?: { method?: 'GET' | 'POST' | 'PUT'; body?: unknown }): Promise<T>;
      chooseDirectory(): Promise<string | null>;
      openPath(path: string): Promise<void>;
      saveArtifact(artifact: { runId: string; name: string }): Promise<string | null>;
      getAppInfo(): Promise<{ version: string; platform: string }>;
      getPreferences(): Promise<{ theme: 'system' | 'light' | 'dark' }>;
      setPreferences(preferences: { theme: 'system' | 'light' | 'dark' }): Promise<{ theme: 'system' | 'light' | 'dark' }>;
      openExternal(url: string): Promise<void>;
      onMenuAction(callback: (action: string) => void): () => void;
    };
  }
}
