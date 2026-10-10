export type CloseBlockReason = 'unsaved' | 'saving';

/** Keep application shutdown behind the renderer's last chance to retain edits. */
export class CloseGuard {
  private quitRequested = false;
  private stopping = false;
  private shutdownComplete = false;
  private prompting = false;

  constructor(private readonly actions: {
    windowId: () => number | null;
    closeWindow: () => void;
    confirmClose: (reason: CloseBlockReason) => boolean;
    stop: () => Promise<void>;
    quit: () => void;
  }) {}

  /** True only for the final quit, after the window closed and shutdown finished. */
  get readyToQuit(): boolean { return this.shutdownComplete; }

  requestQuit(): void {
    if (this.stopping) return;
    this.quitRequested = true;
    if (this.actions.windowId() === null) this.shutdown();
    else this.actions.closeWindow();
  }

  /** Decide within will-prevent-unload; Electron cannot reliably retry it later. */
  blockedUnload(reason: CloseBlockReason): boolean {
    if (this.prompting) return false;
    this.prompting = true;
    try {
      if (this.actions.confirmClose(reason) && reason !== 'saving') return true;
    } catch { /* Keep edits if the native dialog cannot be displayed. */ }
    finally { this.prompting = false; }
    this.quitRequested = false;
    return false;
  }

  windowClosed() {
    if (this.quitRequested) this.shutdown();
  }

  private shutdown() {
    if (this.stopping) return;
    this.stopping = true;
    void this.actions.stop().catch(() => { /* Still release the application after a shutdown failure. */ }).finally(() => {
      this.shutdownComplete = true;
      this.actions.quit();
    });
  }
}
