import type { Event } from './contracts';
type Socket = Pick<WebSocket, 'close' | 'onopen' | 'onmessage' | 'onclose' | 'onerror'>;
export class EventConnection {
  private socket: Socket | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  private attempts = 0;
  private sequence = 0;
  constructor(private url: string, private receive: (event: Event) => void,
    private status: (state: string) => void, private makeSocket: (url: string) => Socket = url => new WebSocket(url)) {}
  start(): void {
    if (this.stopped) return;
    this.status(this.attempts ? 'reconnecting' : 'connecting');
    try { this.socket = this.makeSocket(this.url); } catch { this.reconnect(); return; }
    this.socket.onopen = () => { if (!this.stopped) { this.attempts = 0; this.status('online'); } };
    this.socket.onmessage = event => {
      if (this.stopped) return;
      try {
        const value = JSON.parse(String(event.data)) as Event;
        if (typeof value.type !== 'string') return;
        if (value.sequence && this.sequence && value.sequence > this.sequence + 1) this.receive({ type: 'stream.gap' });
        if (value.sequence) this.sequence = value.sequence;
        this.receive(value);
      } catch { this.receive({ type: 'stream.gap' }); }
    };
    this.socket.onerror = () => this.socket?.close();
    this.socket.onclose = () => this.reconnect();
  }
  private reconnect(): void {
    if (this.stopped || this.timer) return;
    this.status('reconnecting');
    this.timer = setTimeout(() => { this.timer = null; this.start(); }, Math.min(10000, 500 * 2 ** this.attempts++));
  }
  stop(): void { this.stopped = true; if (this.timer) clearTimeout(this.timer); this.timer = null; this.socket?.close(); this.socket = null; this.status('offline'); }
}
