// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { HEADER_BYTES, LiveFrameStream, parseHeader, type ControlMessage } from './liveStream';
import { LiveStreamCanvas } from './components/LiveStreamCanvas';
import { saturationVerdict, summarizePipeline } from './components/diagnostics/pipelineMetrics';

/** Build a server-format message: 16-byte little-endian header + payload. */
function message(frame: number, { stage = [123, 805, 400], flags = 0x11, payload = [0xff, 0xd8, 0xff, 0xd9], timestamp = 1500 } = {}): ArrayBuffer {
  const buffer = new ArrayBuffer(HEADER_BYTES + payload.length);
  const view = new DataView(buffer);
  view.setUint32(0, frame, true); view.setUint32(4, timestamp, true);
  stage.forEach((value, index) => view.setUint16(8 + index * 2, value, true));
  view.setUint16(14, flags, true);
  new Uint8Array(buffer, HEADER_BYTES).set(payload);
  return buffer;
}

class FakeSocket {
  static last: FakeSocket;
  binaryType = ''; readyState = 1; sent: string[] = [];
  onopen: ((e: Event) => void) | null = null; onmessage: ((e: MessageEvent) => void) | null = null;
  onclose: ((e: CloseEvent) => void) | null = null; onerror: ((e: Event) => void) | null = null;
  constructor(readonly url: string) { FakeSocket.last = this; }
  send(value: string) { this.sent.push(value); }
  close() { this.readyState = 3; this.onclose?.({} as CloseEvent); }
  push(data: ArrayBuffer | string) { this.onmessage?.({ data } as MessageEvent); }
}
const fakeBitmap = () => ({ width: 640, height: 360, close: vi.fn() }) as unknown as ImageBitmap;

describe('live frame wire format', () => {
  it('parses the 16-byte header: index, timestamp, three stage latencies, flags', () => {
    const header = parseHeader(message(4_000_000_000, { flags: 0x11 | 0x20 | 0x40 | 0x80 | (1 << 8) }));
    expect(header.frameIndex).toBe(4_000_000_000);
    expect(header.timestampS).toBe(1.5);
    expect(header.stageMs).toEqual([12.3, 80.5, 40]);
    expect(header).toMatchObject({ jpeg: true, dropped: true, last: true, cached: true, source: 'preview' });
  });
  it('rejects empty payloads and foreign protocol versions instead of drawing garbage', () => {
    expect(() => parseHeader(new ArrayBuffer(16))).toThrow(/no payload/);
    expect(() => parseHeader(message(1, { flags: 0x12 }))).toThrow(/version/);
  });
});

describe('LiveFrameStream', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const make = (extra: Partial<ConstructorParameters<typeof LiveFrameStream>[0]> = {}) => {
    const frames: Array<{ bitmap: ImageBitmap; frameIndex: number }> = [];
    const controls: ControlMessage[] = [];
    const releases: Array<() => void> = [];
    const stream = new LiveFrameStream({
      url: 'ws://test/stream', makeSocket: url => new FakeSocket(url) as never,
      decode: () => new Promise<ImageBitmap>(resolve => releases.push(() => resolve(fakeBitmap()))),
      onFrame: (bitmap, header) => frames.push({ bitmap, frameIndex: header.frameIndex }),
      onControl: value => controls.push(value), now: () => 0, ...extra,
    });
    stream.start();
    return { stream, frames, controls, releases };
  };

  it('requests arraybuffers and decodes only the JPEG after the header', async () => {
    const seen: Blob[] = [];
    const { frames } = make({ decode: async blob => { seen.push(blob); return fakeBitmap(); } });
    FakeSocket.last.push(message(7, { payload: [1, 2, 3, 4, 5] }));
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeSocket.last.binaryType).toBe('arraybuffer');
    expect(frames.map(frame => frame.frameIndex)).toEqual([7]);
    expect(seen[0].size).toBe(5);
    expect(seen[0].type).toBe('image/jpeg');
  });

  it('keeps only the newest frame while one is still decoding (no backlog)', async () => {
    const { stream, frames, releases } = make();
    const socket = FakeSocket.last;
    socket.push(message(1)); socket.push(message(2)); socket.push(message(3)); socket.push(message(4));
    releases[0]();
    await vi.advanceTimersByTimeAsync(0);
    releases[1]();
    await vi.advanceTimersByTimeAsync(0);
    expect(frames.map(frame => frame.frameIndex)).toEqual([1, 4]);
    expect(stream.stats).toMatchObject({ received: 4, decoded: 2, droppedLate: 2 });
  });

  it('routes JSON text to onControl and ignores malformed text and wrong-version frames', async () => {
    const { frames, controls } = make({ decode: async () => fakeBitmap() });
    const socket = FakeSocket.last;
    socket.push('{"type":"ready","protocol":1}'); socket.push('{broken'); socket.push(message(9, { flags: 0x13 }));
    await vi.advanceTimersByTimeAsync(0);
    expect(controls).toEqual([{ type: 'ready', protocol: 1 }]);
    expect(frames).toHaveLength(0);
  });

  it('sends interactive render requests and reconnects with backoff after a drop', async () => {
    const { stream } = make({ decode: async () => fakeBitmap() });
    const first = FakeSocket.last;
    expect(stream.render('abcdef0123456789', 12)).toBe(1);
    expect(JSON.parse(first.sent[0])).toEqual({ op: 'render', id: 1, config: 'abcdef0123456789', frame_index: 12 });
    first.close();
    expect(FakeSocket.last).toBe(first);
    await vi.advanceTimersByTimeAsync(500);
    expect(FakeSocket.last).not.toBe(first);
    stream.stop();
    const stopped = FakeSocket.last;
    await vi.advanceTimersByTimeAsync(20000);
    expect(FakeSocket.last).toBe(stopped);
  });

  it('closes a bitmap that finishes decoding after stop() instead of leaking it', async () => {
    const bitmap = fakeBitmap();
    let release: () => void = () => {};
    const frames: number[] = [];
    const stream = new LiveFrameStream({
      url: 'ws://test/stream', makeSocket: url => new FakeSocket(url) as never,
      decode: () => new Promise<ImageBitmap>(resolve => { release = () => resolve(bitmap); }),
      onFrame: (_bitmap, header) => frames.push(header.frameIndex), now: () => 0,
    });
    stream.start();
    FakeSocket.last.push(message(1));
    stream.stop();
    release();
    await vi.advanceTimersByTimeAsync(0);
    expect(frames).toEqual([]);
    expect(bitmap.close).toHaveBeenCalledTimes(1);
  });
});

describe('LiveStreamCanvas', () => {
  let host: HTMLDivElement, root: Root;
  beforeEach(() => { (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true; host = document.createElement('div'); document.body.append(host); root = createRoot(host); });
  afterEach(() => { act(() => root.unmount()); host.remove(); vi.unstubAllGlobals(); vi.useRealTimers(); });

  it('does nothing where createImageBitmap does not exist (jsdom): no socket, hidden canvas', () => {
    const sockets: unknown[] = [];
    vi.stubGlobal('WebSocket', class { constructor() { sockets.push(this); } });
    act(() => root.render(<LiveStreamCanvas/>));
    expect(sockets).toHaveLength(0);
    expect(host.querySelector<HTMLElement>('[data-testid=live-stream]')!.dataset.live).toBe('false');
    expect(host.querySelector('canvas')).not.toBeNull();
  });

  it('draws frames with zero React re-renders and hides itself when the stream goes quiet', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'requestAnimationFrame', 'cancelAnimationFrame', 'performance'] });
    const transfer = vi.fn();
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({ transferFromImageBitmap: transfer })) as never;
    vi.stubGlobal('createImageBitmap', async () => fakeBitmap());
    vi.stubGlobal('WebSocket', FakeSocket);
    let renders = 0;
    const Probe = () => { renders += 1; return <LiveStreamCanvas/>; };
    await act(async () => root.render(<Probe/>));
    const rendersAfterMount = renders;
    const socket = FakeSocket.last;
    const live = () => host.querySelector<HTMLElement>('[data-testid=live-stream]')!.dataset.live;
    for (let frame = 0; frame < 30; frame += 1) {
      await act(async () => { socket.push(message(frame)); await vi.advanceTimersByTimeAsync(34); });
    }
    expect(transfer.mock.calls.length).toBeGreaterThan(5);          // frames reached the canvas through the zero-copy path
    expect(live()).toBe('true');
    expect(host.querySelector('.live-stream-readout')!.textContent).toMatch(/640×360 · detect 12 · swap 81 · blend 40 ms/);
    expect(renders).toBe(rendersAfterMount);                         // ...and React never re-rendered for any of them
    await act(async () => { await vi.advanceTimersByTimeAsync(3000); });
    expect(live()).toBe('false');
  });
});

describe('saturation panel model', () => {
  const metrics = {
    hardware: { latest: { gpu_util_pct: 93, cpu_util_pct: 17, pcie_tx_mb_s: 503.8, pcie_rx_mb_s: 318.3, vram_used_mib: 7054, vram_total_mib: 12282, encoder_util_pct: 0, decoder_util_pct: 1 }, window: {}, backend: 'nvml' },
    live_stream: { backend: 'nvjpeg', mean_encode_ms: 1.1, subscribers: 1, dropped: 2 },
    pipelines: [{ id: 'old', running: false, stages: [] }, {
      id: 'render-1', running: true, fps: 3.9, frames_out: 40, bottleneck: 'swap',
      overlap_wall: { ge2: 0.86, ge3: 0.61 }, pools: [{ backend: 'cuda_pinned', count: 14, in_flight: 9 }],
      stages: [
        { name: 'track', busy_pct: 28.2, mean_ms: 90.5, cuda_mean_ms: 90.3, starved_pct: 0.7, blocked_pct: 58.6, cuda_stream: true, queue_out: { depth: 6, capacity: 8, high_water: 8 } },
        { name: 'swap', busy_pct: 85.2, mean_ms: 273.4, cuda_mean_ms: null, starved_pct: 13.7, blocked_pct: 0, cuda_stream: true, queue_out: { depth: 1, capacity: 8, high_water: 2 } },
      ],
    }],
  };
  it('prefers the running pipeline and reports queue fill, latencies and overlap as measured', () => {
    const summary = summarizePipeline(metrics)!;
    expect(summary).toMatchObject({ gpu: 93, cpu: 17, pcieTx: 503.8, pcieRx: 318.3, fps: 3.9, overlap2: 86, overlap3: 61, running: true });
    expect(summary.stages.map((stage: { name: string; fill: number | null; bottleneck: boolean; cudaMs: number | null }) => [stage.name, stage.fill, stage.bottleneck, stage.cudaMs])).toEqual([['track', 0.75, false, 90.3], ['swap', 0.125, true, null]]);
    expect(saturationVerdict(summary)).toEqual({ level: 'ok', text: 'GPU saturated (93%)' });
  });
  it('never invents a number the server did not report', () => {
    const summary = summarizePipeline({ hardware: { latest: {}, window: {} }, pipelines: [] })!;
    expect(summary.gpu).toBeNull();
    expect(summary.pcieTx).toBeNull();
    expect(saturationVerdict(summary).level).toBe('idle');
    expect(summarizePipeline(null)).toBeNull();
  });
  it('names the limiting stage when the GPU is underfed', () => {
    const summary = summarizePipeline({ ...metrics, hardware: { latest: { gpu_util_pct: 41 }, window: {} } })!;
    expect(saturationVerdict(summary)).toEqual({ level: 'warn', text: 'GPU at 41% · limited by swap' });
  });
});
