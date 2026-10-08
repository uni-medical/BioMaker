import type { ServiceState } from './sidecar';

export type ServiceFailureCode = 'SERVICE_CONSTRUCTION_FAILED' | 'SERVICE_START_FAILED' | 'SERVICE_FAILED' | 'SERVICE_STOP_FAILED';
export type ServiceEventCode = ServiceFailureCode | 'SERVICE_STARTING' | 'SERVICE_READY' | 'SERVICE_RECONNECTING' | 'SERVICE_RETRY_REQUESTED' | 'SERVICE_STOPPED';
export interface ServiceControllerState extends ServiceState { code?: ServiceFailureCode }
export interface ServiceDiagnosticEvent { code: ServiceEventCode; status: ServiceState['status']; timestamp: string }
export interface ServiceProcess {
  start(): Promise<void>;
  stop(): Promise<void>;
  getState(): ServiceState;
}
export interface ServiceCallbacks {
  onState: (state: ServiceState) => void;
  onFailure: (message: string) => void;
}
export interface ServiceControllerOptions<T extends ServiceProcess> {
  factory: (callbacks: ServiceCallbacks) => T;
  onState?: (state: ServiceControllerState) => void;
  onFailure?: (code: ServiceFailureCode) => void;
  now?: () => Date;
}
interface Generation { id: number; sourceRevision: number; launching: boolean; failed: boolean }

/** Owns process replacement without replacing the app or its renderer. Sidecar
 * revisions belong to one instance; only this controller's revision is public. */
export class ServiceController<T extends ServiceProcess> {
  private state: ServiceControllerState = { status: 'starting', revision: 0 };
  private diagnostics: ServiceDiagnosticEvent[] = [];
  private instance: T | null = null;
  private instances = new Set<T>();
  private stops = new WeakMap<T, Promise<void>>();
  private generation = 0;
  private started = false;
  private stopping = false;
  private operation: Promise<ServiceControllerState> | null = null;
  private retrying = false;
  private shutdown: Promise<void> | null = null;

  constructor(private readonly options: ServiceControllerOptions<T>) {}

  getState(): ServiceControllerState { return { ...this.state }; }
  getService(): T | null { return this.stopping ? null : this.instance; }
  getDiagnostics(): ServiceDiagnosticEvent[] { return this.diagnostics.map(event => ({ ...event })); }

  start(): Promise<ServiceControllerState> {
    if (this.stopping) return Promise.resolve(this.getState());
    if (this.operation) return this.operation;
    if (this.started) return Promise.resolve(this.getState());
    this.started = true;
    return this.replace(false);
  }

  retry(): Promise<ServiceControllerState> {
    if (this.stopping) return Promise.resolve(this.getState());
    if (this.operation && this.retrying) return this.operation;
    if (this.state.status !== 'failed') return Promise.resolve(this.getState());
    if (this.operation) return this.operation;
    return this.replace(true);
  }

  private active(generation: Generation): boolean {
    return !this.stopping && this.generation === generation.id;
  }

  private record(code: ServiceEventCode) {
    this.diagnostics.push({ code, status: this.state.status, timestamp: (this.options.now?.() ?? new Date()).toISOString() });
    if (this.diagnostics.length > 50) this.diagnostics.splice(0, this.diagnostics.length - 50);
  }

  private publish(status: ServiceState['status'], event: ServiceEventCode, code?: ServiceFailureCode, force = false) {
    if (!force && this.state.status === status && this.state.code === code) return;
    this.state = { status, revision: this.state.revision + 1, ...(status === 'failed' && code ? { code } : {}) };
    this.record(event);
    // Observers cannot turn an otherwise successful service transition into a
    // startup failure. Each observer receives a detached state snapshot.
    try { this.options.onState?.(this.getState()); } catch { /* UI observer only. */ }
  }

  private fail(generation: Generation, code: ServiceFailureCode) {
    if (!this.active(generation)) return;
    generation.failed = true;
    const changed = this.state.status !== 'failed' || this.state.code !== code;
    this.publish('failed', code, code);
    if (changed && this.active(generation)) {
      try { this.options.onFailure?.(code); } catch { /* Notification observer only. */ }
    }
  }

  private accept(generation: Generation, snapshot: ServiceState) {
    if (!this.active(generation) || generation.failed || !Number.isSafeInteger(snapshot.revision) || snapshot.revision <= generation.sourceRevision) return;
    generation.sourceRevision = snapshot.revision;
    if (snapshot.status === 'failed') this.fail(generation, generation.launching ? 'SERVICE_START_FAILED' : 'SERVICE_FAILED');
    else {
      const events = { starting: 'SERVICE_STARTING', ready: 'SERVICE_READY', reconnecting: 'SERVICE_RECONNECTING' } as const;
      this.publish(snapshot.status, events[snapshot.status]);
    }
  }

  private replace(retry: boolean): Promise<ServiceControllerState> {
    const previous = this.instance;
    const generation: Generation = { id: ++this.generation, sourceRevision: -1, launching: true, failed: false };
    this.instance = null;
    // Defer construction until the shared promise is assigned, including when
    // the factory calls back synchronously or an observer requests shutdown.
    const operation = Promise.resolve().then(() => this.launch(generation, previous)).finally(() => {
      if (this.operation === operation) { this.operation = null; this.retrying = false; }
    });
    this.operation = operation;
    this.retrying = retry;
    if (retry) this.record('SERVICE_RETRY_REQUESTED');
    this.publish('starting', 'SERVICE_STARTING', undefined, true);
    return operation;
  }

  private async launch(generation: Generation, previous: T | null): Promise<ServiceControllerState> {
    if (!this.active(generation)) return this.getState();
    if (previous) {
      try { await this.stopInstance(previous); }
      catch {
        // A failed stop leaves ownership with the old instance. Never create a
        // second service until a later explicit retry has stopped it safely.
        if (this.active(generation)) this.instance = previous;
        this.fail(generation, 'SERVICE_STOP_FAILED');
        return this.getState();
      }
    }
    if (!this.active(generation)) return this.getState();
    let service: T;
    try {
      service = this.options.factory({
        onState: state => this.accept(generation, state),
        // Raw exception messages and paths never enter state or diagnostics.
        onFailure: _message => this.fail(generation, generation.launching ? 'SERVICE_START_FAILED' : 'SERVICE_FAILED'),
      });
    } catch {
      this.fail(generation, 'SERVICE_CONSTRUCTION_FAILED');
      return this.getState();
    }
    this.instances.add(service);
    if (!this.active(generation)) {
      try { await this.stopInstance(service); } catch { this.record('SERVICE_STOP_FAILED'); }
      return this.getState();
    }
    this.instance = service;
    try {
      await service.start();
      if (this.active(generation)) this.accept(generation, service.getState());
    } catch { this.fail(generation, 'SERVICE_START_FAILED'); }
    finally { generation.launching = false; }
    return this.getState();
  }

  private stopInstance(service: T): Promise<void> {
    const existing = this.stops.get(service);
    if (existing) return existing;
    const stopped = Promise.resolve().then(() => service.stop()).then(() => {
      this.instances.delete(service);
      // Keep the completed promise until this controller is discarded: even a
      // concurrent retry/quit snapshot must not call stop twice on an instance.
    }, error => {
      this.stops.delete(service);
      throw error;
    });
    this.stops.set(service, stopped);
    return stopped;
  }

  stop(): Promise<void> {
    if (this.shutdown) return this.shutdown;
    this.stopping = true;
    this.generation++;
    this.instance = null;
    const operation = this.operation;
    const stopKnown = async () => {
      await Promise.all([...this.instances].map(async service => {
        try { await this.stopInstance(service); } catch { this.record('SERVICE_STOP_FAILED'); }
      }));
    };
    this.shutdown = (async () => {
      await stopKnown();
      // A factory may have requested stop through a synchronous callback. Wait
      // for that launch to relinquish its instance before declaring shutdown.
      await operation;
      await stopKnown();
      this.record('SERVICE_STOPPED');
    })();
    return this.shutdown;
  }
}
