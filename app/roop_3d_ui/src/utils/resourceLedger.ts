/**
 * Stage 18 browser resource ledger: counts the three things a long-running page can leak without any visible symptom -
 * object URLs (each pins a Blob, i.e. a decoded preview frame), WebSockets (each pins a connection and its handlers) and
 * WebGL contexts (browsers hard-cap them at ~16 and silently kill the oldest) - plus JS heap where the browser exposes it.
 *
 * It is diagnostics only: it wraps the platform functions, forwards every call unchanged, and keeps O(live resources)
 * state. It is installed when the page is opened with `?r3d-diag=1`, when `localStorage['roop3d-diagnostics'] === '1'`,
 * or by a test calling `installResourceLedger()`. Nothing is sent anywhere; the numbers are read through
 * `window.__R3D_RESOURCES__.snapshot()` (the Playwright soak does exactly that).
 */

export interface ResourceSnapshot {
  objectUrls: { created: number; revoked: number; live: number; oldestLiveAgeMs: number };
  webSockets: { opened: number; closed: number; live: number; liveByPath: Record<string, number> };
  webgl: { created: number; live: number; lost: number; detached: number };
  heap: { usedMB: number; totalMB: number; limitMB: number } | null;
  takenAt: number;
}

export interface ResourceLedgerApi {
  snapshot(): ResourceSnapshot;
  reset(): void;
  uninstall(): void;
}

interface UrlEntry { at: number }
interface SocketEntry { path: string; socket: WebSocket }
interface GlEntry { canvas: WeakRef<HTMLCanvasElement>; gl: WeakRef<WebGLRenderingContext | WebGL2RenderingContext>; lostByApp: boolean }

declare global {
  interface Window { __R3D_RESOURCES__?: ResourceLedgerApi }
}

const MB = 1024 * 1024;

export function installResourceLedger(target: Window & typeof globalThis = window): ResourceLedgerApi {
  if (target.__R3D_RESOURCES__) return target.__R3D_RESOURCES__;

  const urls = new Map<string, UrlEntry>();
  let urlsCreated = 0, urlsRevoked = 0;
  const sockets = new Set<SocketEntry>();
  let socketsOpened = 0, socketsClosed = 0;
  const contexts: GlEntry[] = [];
  let glCreated = 0;

  // ---- object URLs
  const realCreate = target.URL.createObjectURL, realRevoke = target.URL.revokeObjectURL;
  target.URL.createObjectURL = function patchedCreate(this: unknown, object: Blob | MediaSource) {
    const url = realCreate.call(target.URL, object);
    urls.set(url, { at: Date.now() });
    urlsCreated++;
    return url;
  };
  target.URL.revokeObjectURL = function patchedRevoke(this: unknown, url: string) {
    if (urls.delete(url)) urlsRevoked++;
    return realRevoke.call(target.URL, url);
  };

  // ---- WebSockets: subclass so `instanceof WebSocket`, readyState constants and event handlers behave as before.
  const RealWebSocket = target.WebSocket;
  class TrackedWebSocket extends RealWebSocket {
    constructor(url: string | URL, protocols?: string | string[]) {
      super(url, protocols);
      let path = String(url);
      try { path = new URL(String(url), target.location?.href).pathname; } catch { /* keep the raw string */ }
      const entry: SocketEntry = { path, socket: this };
      sockets.add(entry);
      socketsOpened++;
      this.addEventListener('close', () => { if (sockets.delete(entry)) socketsClosed++; });
    }
  }
  target.WebSocket = TrackedWebSocket as unknown as typeof WebSocket;

  // ---- WebGL contexts
  const proto = target.HTMLCanvasElement?.prototype;
  const realGetContext = proto?.getContext;
  if (proto && realGetContext) {
    proto.getContext = function patchedGetContext(this: HTMLCanvasElement, kind: string, ...rest: unknown[]) {
      const already = contexts.some(entry => entry.canvas.deref() === this);
      const context = (realGetContext as (...args: unknown[]) => unknown).call(this, kind, ...rest);
      if (context && !already && /^(experimental-)?webgl/.test(kind)) {
        contexts.push({ canvas: new WeakRef(this), gl: new WeakRef(context as WebGLRenderingContext), lostByApp: false });
        glCreated++;
      }
      return context;
    } as typeof proto.getContext;
  }

  const snapshot = (): ResourceSnapshot => {
    const now = Date.now();
    const liveByPath: Record<string, number> = {};
    for (const entry of sockets) liveByPath[entry.path] = (liveByPath[entry.path] ?? 0) + 1;
    let oldest = 0;
    for (const entry of urls.values()) oldest = Math.max(oldest, now - entry.at);
    let live = 0, lost = 0, detached = 0;
    for (let index = contexts.length - 1; index >= 0; index--) {
      const canvas = contexts[index].canvas.deref(), gl = contexts[index].gl.deref();
      if (!canvas || !gl) { contexts.splice(index, 1); continue; }           // collected by the browser: not live
      if (gl.isContextLost()) { lost++; continue; }
      live++;
      if (!canvas.isConnected) detached++;                                    // alive but no longer in the document
    }
    const memory = (target.performance as Performance & { memory?: { usedJSHeapSize: number; totalJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
    return {
      objectUrls: { created: urlsCreated, revoked: urlsRevoked, live: urls.size, oldestLiveAgeMs: oldest },
      webSockets: { opened: socketsOpened, closed: socketsClosed, live: sockets.size, liveByPath },
      webgl: { created: glCreated, live, lost, detached },
      heap: memory ? { usedMB: memory.usedJSHeapSize / MB, totalMB: memory.totalJSHeapSize / MB, limitMB: memory.jsHeapSizeLimit / MB } : null,
      takenAt: now,
    };
  };

  const api: ResourceLedgerApi = {
    snapshot,
    reset() { urls.clear(); sockets.clear(); contexts.length = 0; urlsCreated = urlsRevoked = socketsOpened = socketsClosed = glCreated = 0; },
    uninstall() {
      target.URL.createObjectURL = realCreate;
      target.URL.revokeObjectURL = realRevoke;
      target.WebSocket = RealWebSocket;
      if (proto && realGetContext) proto.getContext = realGetContext;
      delete target.__R3D_RESOURCES__;
    },
  };
  target.__R3D_RESOURCES__ = api;
  return api;
}

/** Install only when diagnostics were asked for; a no-op (and zero overhead) otherwise. */
export function installResourceLedgerIfRequested(target: Window & typeof globalThis = window): ResourceLedgerApi | null {
  try {
    const wanted = /[?&]r3d-diag=1\b/.test(target.location?.search ?? '') || target.localStorage?.getItem('roop3d-diagnostics') === '1';
    return wanted ? installResourceLedger(target) : null;
  } catch {
    return null;
  }
}
