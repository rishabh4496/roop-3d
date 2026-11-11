/**
 * A WebSocket client that survives backend restarts, laptop sleep and flaky links.
 *
 *  - Reconnects forever with exponential backoff (`baseMs * factor^attempt`, capped at `maxMs`) and +-`jitter` so many tabs do not
 *    stampede a server that just came back. The delay resets as soon as a connection opens.
 *  - Retries at once when the browser comes back online or the tab becomes visible again (timers are throttled in background tabs).
 *  - Optional stale watchdog: an open socket that has delivered nothing for `staleMs` is closed and re-opened, so a half-dead
 *    TCP connection can never leave the UI frozen on old data.
 *  - Reports `onOpen(reconnected)` so the owner can re-fetch whatever it missed while disconnected.
 */
export type SocketState = 'connecting' | 'online' | 'reconnecting' | 'offline';
type SocketLike = Pick<WebSocket, 'close' | 'onopen' | 'onmessage' | 'onclose' | 'onerror'>;

export interface ResilientSocketOptions {
  url: string | (() => string);
  onMessage: (data: unknown) => void;
  onState?: (state: SocketState, info: { attempt: number; nextDelayMs: number | null }) => void;
  /** Called every time the socket opens; `reconnected` is false for the first connection. */
  onOpen?: (reconnected: boolean) => void;
  makeSocket?: (url: string) => SocketLike;
  baseMs?: number;
  maxMs?: number;
  factor?: number;
  /** Fraction of the delay randomised in both directions (0 disables). */
  jitter?: number;
  staleMs?: number;
  random?: () => number;
}

export function backoffDelay(attempt: number, { baseMs = 500, maxMs = 15_000, factor = 2, jitter = 0.25 }: Pick<ResilientSocketOptions, 'baseMs' | 'maxMs' | 'factor' | 'jitter'> = {}, random: () => number = Math.random): number {
  const raw = Math.min(maxMs, baseMs * factor ** Math.max(0, attempt));
  const spread = raw * jitter * (random() * 2 - 1);
  return Math.max(0, Math.round(Math.min(maxMs, raw + spread)));
}

export class ResilientSocket {
  private socket: SocketLike | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private staleTimer: ReturnType<typeof setTimeout> | null = null;
  private stopped = true;
  private attempt = 0;
  private everOpened = false;
  private state: SocketState = 'offline';

  constructor(private readonly options: ResilientSocketOptions) {}

  get current(): SocketState { return this.state; }

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    if (typeof window !== 'undefined') {
      window.addEventListener('online', this.wake);
      document.addEventListener('visibilitychange', this.onVisibility);
    }
    this.open();
  }

  stop(): void {
    this.stopped = true;
    if (typeof window !== 'undefined') {
      window.removeEventListener('online', this.wake);
      document.removeEventListener('visibilitychange', this.onVisibility);
    }
    this.clearTimers();
    const socket = this.socket;
    this.socket = null;
    if (socket) { socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null; try { socket.close(); } catch { /* already closed */ } }
    this.setState('offline', null);
  }

  /** Drop the current connection (if any) and reconnect immediately. */
  reconnectNow(): void {
    if (this.stopped) return;
    this.clearTimers();
    const socket = this.socket;
    this.socket = null;
    if (socket) { socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null; try { socket.close(); } catch { /* already closed */ } }
    this.attempt = 0;
    this.open();
  }

  private setState(state: SocketState, nextDelayMs: number | null): void {
    this.state = state;
    this.options.onState?.(state, { attempt: this.attempt, nextDelayMs });
  }

  private clearTimers(): void {
    if (this.retryTimer) clearTimeout(this.retryTimer);
    if (this.staleTimer) clearTimeout(this.staleTimer);
    this.retryTimer = this.staleTimer = null;
  }

  private armWatchdog(): void {
    const { staleMs } = this.options;
    if (!staleMs) return;
    if (this.staleTimer) clearTimeout(this.staleTimer);
    this.staleTimer = setTimeout(() => { if (!this.stopped && this.socket) this.dropAndRetry(); }, staleMs);
  }

  private open(): void {
    if (this.stopped) return;
    this.setState(this.everOpened || this.attempt > 0 ? 'reconnecting' : 'connecting', null);
    let socket: SocketLike;
    try {
      const url = typeof this.options.url === 'function' ? this.options.url() : this.options.url;
      socket = (this.options.makeSocket ?? (value => new WebSocket(value)))(url);
    } catch { this.schedule(); return; }
    this.socket = socket;
    socket.onopen = () => {
      if (this.stopped || this.socket !== socket) return;
      const reconnected = this.everOpened;
      this.everOpened = true;
      this.attempt = 0;
      this.setState('online', null);
      this.armWatchdog();
      try { this.options.onOpen?.(reconnected); } catch (error) { console.warn('[socket] onOpen handler failed', error); }
    };
    socket.onmessage = event => {
      if (this.stopped || this.socket !== socket) return;
      this.armWatchdog();
      let data: unknown;
      try { data = JSON.parse(String(event.data)); } catch { return; }   // a malformed frame is dropped, never fatal
      try { this.options.onMessage(data); } catch (error) { console.warn('[socket] message handler failed', error); }
    };
    socket.onerror = () => { try { socket.close(); } catch { /* the close handler schedules the retry */ } };
    socket.onclose = () => { if (this.socket === socket) this.dropAndRetry(); };
  }

  private dropAndRetry(): void {
    const socket = this.socket;
    this.socket = null;
    if (socket) { socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null; try { socket.close(); } catch { /* already closed */ } }
    if (this.staleTimer) clearTimeout(this.staleTimer);
    this.staleTimer = null;
    this.schedule();
  }

  private schedule(): void {
    if (this.stopped || this.retryTimer) return;
    const delay = backoffDelay(this.attempt, this.options, this.options.random);
    this.attempt += 1;
    this.setState('reconnecting', delay);
    this.retryTimer = setTimeout(() => { this.retryTimer = null; this.open(); }, delay);
  }

  /** The browser says the network is back: do not wait out the rest of the backoff. */
  private wake = (): void => {
    if (this.stopped || this.state === 'online') return;
    if (this.retryTimer) { clearTimeout(this.retryTimer); this.retryTimer = null; this.open(); }
  };

  private onVisibility = (): void => { if (typeof document !== 'undefined' && document.visibilityState === 'visible') this.wake(); };
}
