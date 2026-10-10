import { spawn, type ChildProcess } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { appendFileSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { prepareSelectedEngine, selectEngine, type EngineSelection } from './engine-selection';

export interface ServiceState { status: 'starting' | 'ready' | 'reconnecting' | 'failed'; revision: number; message?: string }
export interface SidecarOptions { runtime: string; service: string; data: string; version: string; onFailure: (message: string) => void; onState?: (state: ServiceState) => void }
export class Sidecar {
  readonly token = randomBytes(32).toString('hex');
  readonly engine: string;
  readonly runtimeDirectory: string;
  private child: ChildProcess | null = null;
  private port: number | null = null;
  private stopping = false;
  private restarting: Promise<void> | null = null;
  private probing: Promise<void> | null = null;
  private restartCount = 0;
  private state: ServiceState = { status: 'starting', revision: 0 };
  private logPath: string;
  private engineSelection: EngineSelection;
  constructor(private options: SidecarOptions) {
    const manifest = path.join(options.runtime, 'manifest.json');
    this.engineSelection = selectEngine(options.runtime, options.data);
    this.engine = this.engineSelection.directory;
    const identity = createHash('sha256');
    if (existsSync(manifest)) identity.update(readFileSync(manifest));
    const serviceDirectory = path.dirname(options.service);
    for (const entry of readdirSync(serviceDirectory).filter(name => name.endsWith('.py')).sort()) { identity.update(entry); identity.update(readFileSync(path.join(serviceDirectory, entry))); }
    this.runtimeDirectory = path.join(options.data, 'runtimes', identity.digest('hex').slice(0, 24));
    mkdirSync(path.join(options.data, 'logs'), { recursive: true, mode: 0o700 });
    this.logPath = path.join(options.data, 'logs', 'desktop-service.log');
    // The service log is diagnostic only. Bound its size across app launches.
    try { if (readFileSync(this.logPath).length > 2 * 1024 * 1024) writeFileSync(this.logPath, ''); } catch { /* first launch */ }
  }
  get origin(): string {
    if (!this.port) throw new Error('BioMaker 服务正在启动，请稍后重试。');
    return `http://127.0.0.1:${this.port}`;
  }
  getState(): ServiceState { return { ...this.state }; }
  private setState(status: ServiceState['status'], message?: string) {
    const safeMessage = message?.replaceAll(this.token, '[redacted]');
    if (this.state.status === status && this.state.message === safeMessage) return;
    this.state = { status, revision: this.state.revision + 1, ...(safeMessage ? { message: safeMessage } : {}) };
    this.options.onState?.(this.getState());
  }
  private log(message: string) {
    const sanitized = message.replaceAll(this.token, '[redacted]');
    try {
      if (existsSync(this.logPath) && statSync(this.logPath).size > 2 * 1024 * 1024) {
        const previous = `${this.logPath}.1`;
        rmSync(previous, { force: true }); renameSync(this.logPath, previous);
      }
      appendFileSync(this.logPath, `[${new Date().toISOString()}] ${sanitized.slice(0, 16000)}\n`, { mode: 0o600 });
    } catch { /* A full disk must not turn a recoverable service error into an application crash. */ }
  }
  private prepareEngine() {
    prepareSelectedEngine(this.engineSelection);
  }
  private prepareRuntime() {
    if (existsSync(path.join(this.runtimeDirectory, '.desktop-runtime-ready'))) return;
    const stage = `${this.runtimeDirectory}.staging-${process.pid}`;
    mkdirSync(path.dirname(this.runtimeDirectory), { recursive: true, mode: 0o700 });
    rmSync(stage, { recursive: true, force: true });
    mkdirSync(stage, { mode: 0o700 });
    try {
      for (const name of ['python', 'bin', 'site-packages', 'licenses', 'THIRD_PARTY_NOTICES.md', 'manifest.json', 'service-requirements.txt']) {
        cpSync(path.join(this.options.runtime, name), path.join(stage, name), { recursive: true, verbatimSymlinks: true });
      }
      const serviceDirectory = path.dirname(this.options.service);
      mkdirSync(path.join(stage, 'service'));
      for (const entry of readdirSync(serviceDirectory).filter(name => name.endsWith('.py'))) cpSync(path.join(serviceDirectory, entry), path.join(stage, 'service', entry));
      writeFileSync(path.join(stage, '.desktop-runtime-ready'), this.options.version);
      if (existsSync(this.runtimeDirectory)) rmSync(this.runtimeDirectory, { recursive: true, force: true });
      renameSync(stage, this.runtimeDirectory);
    } catch (error) { rmSync(stage, { recursive: true, force: true }); throw error; }
    // Keep previous runtime versions. A detached experiment may still use their
    // Python binary or native libraries after the desktop app is upgraded.
  }
  async start(): Promise<void> {
    if (this.stopping) return;
    this.setState(this.state.status === 'reconnecting' ? 'reconnecting' : 'starting');
    try {
      this.prepareRuntime();
      this.prepareEngine();
      for (let attempt = 0; attempt < 3; attempt++) {
        if (this.stopping) return;
        try { await this.launch(); return; } catch (error) {
          if (this.stopping) return;
          this.log(`Startup attempt ${attempt + 1}: ${String(error)}`);
          if (attempt === 2) throw error;
          await new Promise(resolve => setTimeout(resolve, 500 * (attempt + 1)));
        }
      }
    } catch (error) {
      if (!this.stopping) this.setState('failed', String(error));
      throw error;
    }
  }
  private async launch(): Promise<void> {
    const python = process.platform === 'win32' ? path.join(this.runtimeDirectory, 'python', 'python.exe') : path.join(this.runtimeDirectory, 'python', 'bin', 'python3');
    const uv = path.join(this.runtimeDirectory, 'bin', process.platform === 'win32' ? 'uv.exe' : 'uv');
    if (!existsSync(python) || !existsSync(uv)) throw new Error('缺少内置 Python 运行时。请重新安装应用。');
    this.port = null;
    const child = spawn(python, ['-u', path.join(this.runtimeDirectory, 'service', path.basename(this.options.service))], {
      cwd: this.engine, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env,
        PYTHONPATH: [path.join(this.runtimeDirectory, 'site-packages'), this.engine].join(path.delimiter),
        PYTHONNOUSERSITE: '1', PYTHONDONTWRITEBYTECODE: '1', PYTHONUNBUFFERED: '1',
        BIOMAKER_DESKTOP_TOKEN: this.token, BIOMAKER_DESKTOP_HOME: this.options.data,
        BIOMAKER_DESKTOP_ENGINE_ROOT: this.engine, BIOMAKER_DESKTOP_PARENT_PID: String(process.pid),
        BIOMAKER_WEBUI_UV: uv, BIOMAKER_DESKTOP_VERSION: this.options.version,
        UV_PYTHON_DOWNLOADS: 'never',
      },
    });
    this.child = child;
    let didBecomeReady = false;
    await new Promise<void>((resolve, reject) => {
      let settled = false;
      let checkingHealth = false;
      let output = '';
      const timeout = setTimeout(() => fail(new Error('BioMaker 本地服务启动超时。')), 30_000);
      const fail = (error: Error) => {
        if (settled) return;
        settled = true; clearTimeout(timeout);
        if (this.child === child) this.port = null;
        child.kill();
        setTimeout(() => { if (child.exitCode === null) child.kill('SIGKILL'); }, 2000).unref();
        reject(error);
      };
      child.stdout?.on('data', (buffer: Buffer) => {
        output += buffer.toString('utf8');
        if (output.length > 64 * 1024) output = output.slice(-32 * 1024);
        let newline: number;
        while ((newline = output.indexOf('\n')) !== -1) {
          const line = output.slice(0, newline); output = output.slice(newline + 1);
          try {
            const status = JSON.parse(line);
            if (!settled && !checkingHealth && status.type === 'ready' && Number.isInteger(status.port) && status.port > 0 && status.port < 65536) {
              checkingHealth = true;
              // The announced socket must answer authenticated health before renderer access.
              void this.waitForHealth(`http://127.0.0.1:${status.port}`, child).then(() => {
                if (settled) return;
                if (this.stopping || this.child !== child) { fail(new Error('Service startup was cancelled')); return; }
                settled = true; didBecomeReady = true; clearTimeout(timeout);
                this.port = status.port;
                this.setState('ready');
                resolve();
              }).catch(error => fail(error));
            } else this.log(line);
          } catch { this.log(line); }
        }
      });
      child.stderr?.on('data', (buffer: Buffer) => this.log(buffer.toString('utf8')));
      child.once('error', error => fail(new Error(`无法启动本地服务：${error.message}`)));
      child.once('exit', (code, signal) => {
        const current = this.child === child;
        if (current) { this.child = null; this.port = null; }
        if (!settled) fail(new Error(`BioMaker 服务退出（${code ?? signal}）。诊断日志：${this.logPath}`));
        else if (current && didBecomeReady && !this.stopping) this.recover();
      });
    });
  }
  private async waitForHealth(origin: string, child: ChildProcess): Promise<void> {
    let lastError: unknown;
    for (let attempt = 0; attempt < 30 && !this.stopping && this.child === child; attempt++) {
      try {
        const response = await fetch(`${origin}/api/health`, { headers: { Authorization: `Bearer ${this.token}` }, redirect: 'error', signal: AbortSignal.timeout(1000) });
        if (!response.ok) throw new Error(`Health check failed (${response.status})`);
        return;
      } catch (error) { lastError = error; }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw lastError ?? new Error('Service startup was cancelled');
  }
  /** A transport failure is distinct from a valid HTTP error response. Probe
   * authenticated health before replacing a possibly still healthy service. */
  checkConnection(): Promise<void> {
    if (this.probing) return this.probing;
    if (this.stopping || this.state.status !== 'ready' || !this.child) return Promise.resolve();
    const child = this.child;
    const origin = this.origin;
    this.setState('reconnecting');
    this.probing = (async () => {
      try {
        const response = await fetch(`${origin}/api/health`, { headers: { Authorization: `Bearer ${this.token}` }, redirect: 'error', signal: AbortSignal.timeout(1000) });
        if (!response.ok) throw new Error(`Health check failed (${response.status})`);
        if (!this.stopping && this.child === child) this.setState('ready');
      } catch (error) {
        if (this.stopping || this.child !== child) return;
        this.log(`Connection health check failed: ${String(error)}`);
        child.kill();
        setTimeout(() => { if (this.child === child) child.kill('SIGKILL'); }, 2000).unref();
      }
    })().finally(() => { this.probing = null; });
    return this.probing;
  }
  private recover() {
    if (this.stopping) return;
    this.setState('reconnecting');
    if (this.restarting) return;
    if (this.restartCount >= 3) {
      const message = 'BioMaker 本地服务已停止。请重新启动应用；运行中的实验由独立进程管理。';
      this.setState('failed', message);
      this.options.onFailure(message); return;
    }
    this.restartCount++;
    this.restarting = new Promise<void>(resolve => setTimeout(resolve, 1000 * this.restartCount))
      .then(() => this.start()).catch(error => { if (!this.stopping) this.options.onFailure(String(error)); })
      .finally(() => {
        this.restarting = null;
        if (!this.stopping && this.state.status === 'reconnecting' && !this.child) this.recover();
      });
  }
  async stop(): Promise<void> {
    this.stopping = true;
    const child = this.child;
    this.port = null;
    if (!child) return;
    const hasExited = () => child.exitCode !== null || child.signalCode !== null;
    if (hasExited()) { if (this.child === child) this.child = null; return; }
    // Stop only our HTTP sidecar. The existing experiment supervisor is detached
    // and deliberately survives closing/quitting the desktop client.
    await new Promise<void>((resolve, reject) => {
      let settled = false;
      let signalFailed = false;
      let gracefulTimeout: NodeJS.Timeout | undefined;
      let forcedTimeout: NodeJS.Timeout | undefined;
      const finish = (error?: Error) => {
        if (settled) return;
        settled = true;
        clearTimeout(gracefulTimeout); clearTimeout(forcedTimeout);
        child.off('exit', onExit); child.off('error', onError);
        if (error) reject(error); else resolve();
      };
      const onExit = () => finish();
      // Kill errors can be emitted as events or thrown. Keep the listener while
      // awaiting exit; a failed signal never counts as confirmed termination.
      const onError = () => { signalFailed = true; };
      const signal = (name: NodeJS.Signals) => {
        try { if (!child.kill(name)) signalFailed = true; }
        catch { signalFailed = true; }
        if (hasExited()) finish();
      };
      child.once('exit', onExit);
      child.on('error', onError);
      gracefulTimeout = setTimeout(() => {
        if (hasExited()) { finish(); return; }
        forcedTimeout = setTimeout(() => {
          if (hasExited()) finish();
          else finish(new Error(signalFailed ? '未能终止 BioMaker 本地服务。' : '无法确认 BioMaker 本地服务已退出。'));
        }, 2000);
        signal('SIGKILL');
      }, 4000);
      signal('SIGTERM');
    });
    // Rejection deliberately retains ownership so the controller cannot start
    // a replacement while this process may still be alive.
    if (this.child === child) this.child = null;
  }
}
