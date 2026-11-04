// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import PipelinePill from './components/hud/PipelinePill';
import { EventConnection } from './events';
import { installResourceLedger, type ResourceLedgerApi } from './utils/resourceLedger';
import { PreviewURL } from './state';
import { downloadBlob } from './livePreview';

/**
 * Stage 18: the ledger must count exactly, and the real application code paths that create object URLs and sockets must
 * release every one of them: replacing a preview 200 times keeps one URL, 100 start/stop cycles leave no socket, 40
 * mount/unmount cycles of the HUD pill leave no socket, a programmatic download revokes its URL.
 */
(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;

class FakeSocket extends EventTarget {
  static all: FakeSocket[] = [];
  static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
  readyState = 0;
  onopen: ((e: Event) => void) | null = null; onclose: ((e: Event) => void) | null = null;
  onmessage: ((e: MessageEvent) => void) | null = null; onerror: ((e: Event) => void) | null = null;
  constructor(public url: string | URL, public protocols?: string | string[]) { super(); FakeSocket.all.push(this); queueMicrotask(() => { this.readyState = 1; this.onopen?.(new Event('open')); }); }
  send() {}
  close() {
    if (this.readyState === 3) return;
    this.readyState = 3;
    const event = new Event('close');
    this.dispatchEvent(event);
    this.onclose?.(event);
  }
}

let ledger: ResourceLedgerApi;
let counter = 0;
const originals = { create: URL.createObjectURL, revoke: URL.revokeObjectURL, socket: globalThis.WebSocket };

beforeEach(() => {
  FakeSocket.all = []; counter = 0;
  URL.createObjectURL = () => `blob:test/${++counter}`;
  URL.revokeObjectURL = () => {};
  (globalThis as any).WebSocket = FakeSocket;
  ledger = installResourceLedger(window as any);
});
afterEach(() => {
  ledger.uninstall();
  URL.createObjectURL = originals.create; URL.revokeObjectURL = originals.revoke; (globalThis as any).WebSocket = originals.socket;
  vi.useRealTimers();
});

describe('resource ledger accounting', () => {
  it('counts object URLs created, revoked and live, and ignores revoking an unknown url', () => {
    const a = URL.createObjectURL(new Blob(['a'])), b = URL.createObjectURL(new Blob(['b']));
    URL.revokeObjectURL(a); URL.revokeObjectURL('blob:unknown'); URL.revokeObjectURL(a);
    expect(ledger.snapshot().objectUrls).toMatchObject({ created: 2, revoked: 1, live: 1 });
    URL.revokeObjectURL(b);
    expect(ledger.snapshot().objectUrls.live).toBe(0);
  });

  it('counts live WebSockets by path and forwards instanceof / handlers unchanged', () => {
    const socket = new WebSocket('ws://127.0.0.1:8000/ws/events');
    new WebSocket('ws://127.0.0.1:8000/ws/telemetry');
    expect(socket).toBeInstanceOf(FakeSocket);
    expect(ledger.snapshot().webSockets).toMatchObject({ opened: 2, live: 2, liveByPath: { '/ws/events': 1, '/ws/telemetry': 1 } });
    socket.close(); socket.close();
    expect(ledger.snapshot().webSockets).toMatchObject({ closed: 1, live: 1 });
  });

  it('is idempotent to install and restores every patched function on uninstall', () => {
    expect(installResourceLedger(window as any)).toBe(ledger);
    ledger.uninstall();
    expect(globalThis.WebSocket).toBe(FakeSocket);
    expect(window.__R3D_RESOURCES__).toBeUndefined();
    ledger = installResourceLedger(window as any);
  });
});

describe('application code releases what it acquires', () => {
  it('PreviewURL keeps at most one live object URL across 200 replacements', () => {
    const preview = new PreviewURL(URL.createObjectURL, URL.revokeObjectURL);
    for (let i = 0; i < 200; i++) { preview.replace(new Blob([String(i)])); expect(ledger.snapshot().objectUrls.live).toBeLessThanOrEqual(1); }
    preview.clear();
    expect(ledger.snapshot().objectUrls).toMatchObject({ created: 200, revoked: 200, live: 0 });
  });

  it('downloadBlob revokes its url after the click has been handled', () => {
    vi.useFakeTimers();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});   // jsdom cannot navigate to a blob: url
    for (let i = 0; i < 25; i++) downloadBlob(new Blob(['x']), `p${i}.png`);
    expect(ledger.snapshot().objectUrls.live).toBe(25);          // pending: the browser may still be reading them
    vi.advanceTimersByTime(10_001);
    expect(ledger.snapshot().objectUrls.live).toBe(0);
  });

  it('EventConnection leaves no socket and no pending reconnect after 100 start/stop cycles', () => {
    vi.useFakeTimers();
    for (let i = 0; i < 100; i++) {
      const connection = new EventConnection('ws://127.0.0.1:8000/ws/events', () => {}, () => {});
      connection.start();
      connection.stop();
    }
    vi.advanceTimersByTime(60_000);                                // a leaked reconnect timer would open sockets now
    expect(ledger.snapshot().webSockets).toMatchObject({ opened: 100, live: 0 });
    expect(vi.getTimerCount()).toBe(0);
  });

  it('EventConnection reconnects with backoff and still holds exactly one socket', () => {
    vi.useFakeTimers();
    const connection = new EventConnection('ws://127.0.0.1:8000/ws/events', () => {}, () => {});
    connection.start();
    for (let i = 0; i < 6; i++) { FakeSocket.all[FakeSocket.all.length - 1].close(); vi.advanceTimersByTime(11_000); }
    expect(ledger.snapshot().webSockets.live).toBe(1);
    connection.stop();
    expect(ledger.snapshot().webSockets.live).toBe(0);
  });

  it('the HUD pill closes its telemetry socket and retry timer on every unmount (40 mount cycles)', async () => {
    vi.useFakeTimers();
    for (let i = 0; i < 40; i++) {
      const host = document.createElement('div');
      document.body.appendChild(host);
      const root = createRoot(host);
      await act(async () => { root.render(<PipelinePill />); });
      expect(ledger.snapshot().webSockets.live).toBe(1);
      await act(async () => { root.unmount(); });
      host.remove();
      expect(ledger.snapshot().webSockets.live).toBe(0);
    }
    vi.advanceTimersByTime(30_000);                                // an orphaned retry timer would reconnect now
    expect(ledger.snapshot().webSockets).toMatchObject({ opened: 40, live: 0 });
  });
});
