import { spawn, type ChildProcess } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { appendFileSync, cpSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';

export interface SidecarOptions { runtime: string; service: string; data: string; version: string; onFailure: (message: string) => void }
export class Sidecar {
  readonly token = randomBytes(32).toString('hex');
  readonly engine: string;
  readonly runtimeDirectory: string;
  private child: ChildProcess | null = null;
  private port: number | null = null;
  private stopping = false;
  private restarting: Promise<void> | null = null;
  private restartCount = 0;
  private logPath: string;
  constructor(private options: SidecarOptions) {
    let hash = 'development';
    const manifest = path.join(options.runtime, 'manifest.json');
    if (existsSync(manifest)) hash = JSON.parse(readFileSync(manifest, 'utf8')).engineSha256.slice(0, 12);
    this.engine = path.join(options.data, 'engines', `${options.version}-${hash}`);
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
    if (existsSync(path.join(this.engine, '.desktop-engine-ready'))) return;
    const source = path.join(this.options.runtime, 'engine');
    if (!existsSync(source)) throw new Error('缺少 BioMaker 引擎。请重新安装应用。开发环境请先运行 npm run prepare:runtime。');
    mkdirSync(path.dirname(this.engine), { recursive: true, mode: 0o700 });
    const stage = `${this.engine}.staging-${process.pid}`;
    rmSync(stage, { recursive: true, force: true });
    cpSync(source, stage, { recursive: true });
    writeFileSync(path.join(stage, '.desktop-engine-ready'), this.options.version);
    if (existsSync(this.engine)) rmSync(this.engine, { recursive: true, force: true });
    renameSync(stage, this.engine);
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
    this.prepareRuntime();
    this.prepareEngine();
    for (let attempt = 0; attempt < 3; attempt++) {
      if (this.stopping) return;
      try { await this.launch(); return; } catch (error) {
        this.log(`Startup attempt ${attempt + 1}: ${String(error)}`);
        if (attempt === 2) throw error;
        await new Promise(resolve => setTimeout(resolve, 500 * (attempt + 1)));
      }
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
      let output = '';
      const timeout = setTimeout(() => fail(new Error('BioMaker 本地服务启动超时。')), 30_000);
      const fail = (error: Error) => {
        if (settled) return;
        settled = true; clearTimeout(timeout); child.kill();
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
            if (!settled && status.type === 'ready' && Number.isInteger(status.port) && status.port > 0 && status.port < 65536) {
              this.port = status.port;
              // The announced socket must answer authenticated health before renderer access.
              void this.waitForHealth().then(() => {
                if (settled) return;
                settled = true; didBecomeReady = true; clearTimeout(timeout); resolve();
              }).catch(error => fail(error));
            } else this.log(line);
          } catch { this.log(line); }
        }
      });
      child.stderr?.on('data', (buffer: Buffer) => this.log(buffer.toString('utf8')));
      child.once('error', error => fail(new Error(`无法启动本地服务：${error.message}`)));
      child.once('exit', (code, signal) => {
        if (this.child === child) { this.child = null; this.port = null; }
        if (!settled) fail(new Error(`BioMaker 服务退出（${code ?? signal}）。诊断日志：${this.logPath}`));
        else if (didBecomeReady && !this.stopping) this.recover();
      });
    });
  }
  private async waitForHealth(): Promise<void> {
    let lastError: unknown;
    for (let attempt = 0; attempt < 30 && !this.stopping; attempt++) {
      try {
        const response = await fetch(`${this.origin}/api/health`, { headers: { Authorization: `Bearer ${this.token}` }, signal: AbortSignal.timeout(1000) });
        if (!response.ok) throw new Error(`Health check failed (${response.status})`);
        return;
      } catch (error) { lastError = error; }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw lastError ?? new Error('Service startup was cancelled');
  }
  private recover() {
    if (this.restarting || this.restartCount >= 3) {
      this.options.onFailure('BioMaker 本地服务已停止。请重新启动应用；运行中的实验由独立进程管理。'); return;
    }
    this.restartCount++;
    this.restarting = new Promise<void>(resolve => setTimeout(resolve, 1000 * this.restartCount))
      .then(() => this.start()).catch(error => this.options.onFailure(String(error)))
      .finally(() => { this.restarting = null; });
  }
  async stop(): Promise<void> {
    this.stopping = true;
    const child = this.child;
    this.port = null;
    if (!child || child.exitCode !== null) return;
    // Stop only our HTTP sidecar. The existing experiment supervisor is detached
    // and deliberately survives closing/quitting the desktop client.
    await new Promise<void>(resolve => {
      const timeout = setTimeout(() => { child.kill('SIGKILL'); resolve(); }, 4000);
      child.once('exit', () => { clearTimeout(timeout); resolve(); });
      child.kill('SIGTERM');
    });
    this.child = null;
  }
}
