/**
 * Binary live-frame client for `/ws/v1/preview/stream` (see roop_3d/async_pipeline/live.py for the wire format).
 *
 * Each WebSocket binary message is one JPEG behind a 16-byte little-endian header:
 *   u32 frame index · u32 media timestamp (ms) · u16 stage-1/2/3 latency (0.1 ms units) · u16 flags
 * The JPEG is decoded off the main thread by `createImageBitmap` and handed to the caller as an `ImageBitmap`; nothing
 * is base64-encoded, parsed as JSON or stored in React state.  If frames arrive faster than they can be decoded the
 * newest one wins and older ones are dropped (counted), so a slow tab never builds a backlog.
 */
export const HEADER_BYTES = 16;
export const PROTOCOL_VERSION = 1;

export type FrameSource = 'job' | 'preview';
export type FrameHeader = {
  frameIndex: number;
  timestampS: number;
  /** Wall latency of pipeline stages 1 (detect/pose), 2 (swap/mesh) and 3 (enhance/blend) for this frame, ms. */
  stageMs: [number, number, number];
  jpeg: boolean;
  /** Frames were discarded server-side just before this one (the renderer outran the encoder or a slow client). */
  dropped: boolean;
  last: boolean;
  cached: boolean;
  source: FrameSource;
};

export function parseHeader(buffer: ArrayBuffer): FrameHeader {
  if (buffer.byteLength <= HEADER_BYTES) throw new Error('Frame message has no payload');
  const view = new DataView(buffer, 0, HEADER_BYTES);
  const flags = view.getUint16(14, true);
  if ((flags & 0xf) !== PROTOCOL_VERSION) throw new Error(`Unsupported live-frame protocol version ${flags & 0xf}`);
  return {
    frameIndex: view.getUint32(0, true),
    timestampS: view.getUint32(4, true) / 1000,
    stageMs: [view.getUint16(8, true) / 10, view.getUint16(10, true) / 10, view.getUint16(12, true) / 10],
    jpeg: !!(flags & (1 << 4)),
    dropped: !!(flags & (1 << 5)),
    last: !!(flags & (1 << 6)),
    cached: !!(flags & (1 << 7)),
    source: ((flags >> 8) & 3) === 1 ? 'preview' : 'job',
  };
}

export type ControlMessage = { type: string; id?: number; [key: string]: unknown };
export type LiveStreamStats = {
  received: number; decoded: number; droppedLate: number; droppedServer: number; fps: number; lastHeader: FrameHeader | null; decodeMs: number;
};
type SocketLike = Pick<WebSocket, 'close' | 'send' | 'readyState'> & {
  binaryType: BinaryType; onopen: ((e: Event) => void) | null; onmessage: ((e: MessageEvent) => void) | null;
  onclose: ((e: CloseEvent) => void) | null; onerror: ((e: Event) => void) | null;
};

export type LiveStreamOptions = {
  url?: string;
  onFrame: (bitmap: ImageBitmap, header: FrameHeader) => void;
  onControl?: (message: ControlMessage) => void;
  onStatus?: (status: 'connecting' | 'online' | 'reconnecting' | 'offline') => void;
  makeSocket?: (url: string) => SocketLike;
  decode?: (blob: Blob) => Promise<ImageBitmap>;
  now?: () => number;
};

/** True where the zero-copy path exists (a browser, not jsdom). */
export const liveStreamSupported = () =>
  typeof WebSocket !== 'undefined' && typeof createImageBitmap === 'function';

const defaultUrl = () => `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/v1/preview/stream`;

export class LiveFrameStream {
  private socket: SocketLike | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private attempts = 0;
  private stopped = true;
  private decoding = false;
  private queued: { data: ArrayBuffer; header: FrameHeader } | null = null;
  private nextId = 1;
  private recent: number[] = [];
  readonly stats: LiveStreamStats = { received: 0, decoded: 0, droppedLate: 0, droppedServer: 0, fps: 0, lastHeader: null, decodeMs: 0 };

  constructor(private options: LiveStreamOptions) {}

  start(): void {
    if (!this.stopped) return;
    this.stopped = false;
    this.connect();
  }

  private connect(): void {
    if (this.stopped) return;
    this.options.onStatus?.(this.attempts ? 'reconnecting' : 'connecting');
    const url = this.options.url ?? defaultUrl();
    try { this.socket = this.options.makeSocket ? this.options.makeSocket(url) : (new WebSocket(url) as unknown as SocketLike); }
    catch { this.reconnect(); return; }
    this.socket.binaryType = 'arraybuffer';
    this.socket.onopen = () => { if (!this.stopped) { this.attempts = 0; this.options.onStatus?.('online'); } };
    this.socket.onmessage = event => this.receive(event.data);
    this.socket.onerror = () => this.socket?.close();
    this.socket.onclose = () => this.reconnect();
  }

  private reconnect(): void {
    if (this.stopped || this.timer) return;
    this.options.onStatus?.('reconnecting');
    this.timer = setTimeout(() => { this.timer = null; this.connect(); }, Math.min(10000, 500 * 2 ** this.attempts++));
  }

  private receive(data: unknown): void {
    if (this.stopped) return;
    if (typeof data === 'string') {
      try { this.options.onControl?.(JSON.parse(data) as ControlMessage); } catch { /* malformed control text is ignored */ }
      return;
    }
    if (!(data instanceof ArrayBuffer)) return;
    let header: FrameHeader;
    try { header = parseHeader(data); } catch { return; }
    this.stats.received += 1;
    if (header.dropped) this.stats.droppedServer += 1;
    if (this.decoding) {                                   // latest wins: never build a decode backlog
      if (this.queued) this.stats.droppedLate += 1;
      this.queued = { data, header };
      return;
    }
    void this.decode(data, header);
  }

  private async decode(data: ArrayBuffer, header: FrameHeader): Promise<void> {
    this.decoding = true;
    const clock = this.options.now ?? (() => performance.now());
    try {
      const began = clock();
      const blob = new Blob([new Uint8Array(data, HEADER_BYTES)], { type: 'image/jpeg' });
      const bitmap = await (this.options.decode ?? createImageBitmap)(blob);
      this.stats.decodeMs = clock() - began;
      if (this.stopped) { bitmap.close?.(); return; }
      this.stats.decoded += 1;
      this.stats.lastHeader = header;
      const stamp = clock();
      this.recent.push(stamp);
      while (this.recent.length && stamp - this.recent[0] > 1000) this.recent.shift();
      this.stats.fps = this.recent.length;
      this.options.onFrame(bitmap, header);
    } catch { /* an undecodable frame is skipped; the next one replaces it */ }
    finally {
      this.decoding = false;
      const next = this.queued;
      this.queued = null;
      if (next && !this.stopped) void this.decode(next.data, next.header);
    }
  }

  /** Ask for one interactive frame of a configuration previously POSTed to `/api/preview`; returns the request id. */
  render(config: string, frameIndex: number): number | null {
    if (!this.socket || this.socket.readyState !== 1) return null;
    const id = this.nextId++;
    this.socket.send(JSON.stringify({ op: 'render', id, config, frame_index: frameIndex }));
    return id;
  }

  stop(): void {
    this.stopped = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    this.queued = null;
    this.socket?.close();
    this.socket = null;
    this.options.onStatus?.('offline');
  }
}
