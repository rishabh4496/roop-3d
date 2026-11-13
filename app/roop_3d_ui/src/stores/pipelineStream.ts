import { createAtomicStore, useStoreSelector } from './atomicStore';
import { ingestTrackingFrame } from './trackingStream';
import { ResilientSocket, type SocketState } from '../utils/resilientSocket';

/** Status of one model transfer, as `/ws/v1/pipeline/status` reports it (`model_download_progress` frames). */
export type TransferStatus = 'queued' | 'downloading' | 'verifying' | 'compiling' | 'ready' | 'failed' | 'cancelled';
export interface ModelProgress {
  modelId: string;
  status: TransferStatus;
  downloadedBytes: number;
  totalBytes: number;
  /** 0-100. */
  percent: number;
  /** Megabits per second over the last few seconds (the server's `speed_mbps`). */
  speedMbps: number;
  /** null while the rate is unknown; never a made-up 0. */
  etaSeconds: number | null;
  mirror: string | null;
  batchId: string | null;
  detail: string | null;
  receivedAt: number;
}

const STATUSES: readonly TransferStatus[] = ['queued', 'downloading', 'verifying', 'compiling', 'ready', 'failed', 'cancelled'];
export const ACTIVE_TRANSFER: readonly TransferStatus[] = ['queued', 'downloading', 'verifying', 'compiling'];
const finite = (value: unknown, fallback = 0): number => (typeof value === 'number' && Number.isFinite(value) ? value : fallback);
const text = (value: unknown): string | null => (typeof value === 'string' && value ? value : null);

/**
 * Validates one `model_download_progress` frame into a fully-defaulted record, or returns null when it is unusable.
 * Every numeric field is clamped, the status is one of the known values, and no property access can throw later.
 */
export function parseProgressFrame(raw: unknown, now = Date.now()): ModelProgress | null {
  if (!raw || typeof raw !== 'object') return null;
  const frame = raw as Record<string, unknown>;
  if (frame.type !== 'model_download_progress') return null;
  const modelId = text(frame.model_id);
  if (!modelId) return null;
  const downloaded = Math.max(0, finite(frame.downloaded_bytes, finite(frame.bytes_downloaded)));
  const total = Math.max(0, finite(frame.total_bytes));
  const derived = total > 0 ? (downloaded / total) * 100 : 0;
  const eta = frame.eta_seconds == null ? null : finite(frame.eta_seconds, NaN);
  return {
    modelId,
    status: (STATUSES as readonly unknown[]).includes(frame.status) ? frame.status as TransferStatus : 'downloading',
    downloadedBytes: downloaded, totalBytes: total,
    percent: Math.min(100, Math.max(0, finite(frame.percent, derived))),
    speedMbps: Math.max(0, finite(frame.speed_mbps)),
    etaSeconds: eta == null || Number.isNaN(eta) || eta < 0 ? null : eta,
    mirror: text(frame.mirror_used), batchId: text(frame.batch_id), detail: text(frame.detail), receivedAt: now,
  };
}

/** A TensorRT engine build in progress (`trt_compilation_progress`). Elapsed time and VRAM are measured; there is no percentage because TensorRT reports none. */
export interface TrtBuild {
  model: string;
  precision: string | null;
  precisionLabel: string | null;
  /** Optimisation-profile shape such as "4x3x512x512"; null when the build path does not know it. */
  optShape: string | null;
  elapsedSeconds: number;
  vramGb: number | null;
  /** Whether `vramGb` is this process's own memory or the whole device. */
  vramScope: 'process' | 'device' | null;
  receivedAt: number;
}
/**
 * ONNX Runtime holds Python's GIL for the whole TensorRT build, so the server cannot send anything while it runs: the page gets the
 * start frame (with elapsed 0) and, much later, the finish frame. The strip therefore counts time itself from the last frame it got,
 * and VRAM is shown only while a frame that carried it is this fresh.
 */
export const TRT_VRAM_FRESH_MS = 2500;
/** A build still "running" after this long is not believed (a lost finish frame must not leave the strip up for ever). */
export const TRT_BUILD_MAX_S = 15 * 60;
/** Seconds into the build right now: the last server-measured elapsed time plus the time since that frame arrived. */
export const trtElapsedSeconds = (build: TrtBuild, now: number): number => build.elapsedSeconds + Math.max(0, now - build.receivedAt) / 1000;

/** Validates one `trt_compilation_progress` frame. `finished` clears the build; anything malformed is ignored. */
export function parseTrtFrame(raw: unknown, now = Date.now()): { finished: true } | { finished: false; build: TrtBuild } | null {
  if (!raw || typeof raw !== 'object') return null;
  const frame = raw as Record<string, unknown>;
  if (frame.type !== 'trt_compilation_progress') return null;
  if (frame.status === 'finished') return { finished: true };
  const model = text(frame.model);
  if (!model) return null;
  const vram = frame.vram_gb == null ? null : finite(frame.vram_gb, NaN);
  return { finished: false, build: {
    model, precision: text(frame.precision), precisionLabel: text(frame.precision_label), optShape: text(frame.opt_shape),
    elapsedSeconds: Math.max(0, finite(frame.elapsed_seconds)),
    vramGb: vram == null || Number.isNaN(vram) || vram < 0 ? null : vram,
    vramScope: frame.vram_scope === 'process' || frame.vram_scope === 'device' ? frame.vram_scope : null,
    receivedAt: now } };
}

interface StreamState {
  connection: SocketState;
  /** Increments after every reconnect: readers that cached server state re-fetch when it changes. */
  resyncs: number;
  lastFrameAt: number;
  transfers: Record<string, ModelProgress>;
  trtBuild: TrtBuild | null;
}
export const pipelineStream = createAtomicStore<StreamState>({ connection: 'offline', resyncs: 0, lastFrameAt: 0, transfers: {}, trtBuild: null });
export const useTrtBuild = (): TrtBuild | null => useStoreSelector(pipelineStream, state => state.trtBuild);
export const usePipelineConnection = (): SocketState => useStoreSelector(pipelineStream, state => state.connection);
export const useTransfers = (): Record<string, ModelProgress> => useStoreSelector(pipelineStream, state => state.transfers);
export const useStreamResyncs = (): number => useStoreSelector(pipelineStream, state => state.resyncs);

export function ingestFrame(raw: unknown, now = Date.now()): void {
  if (!raw || typeof raw !== 'object') return;
  if (ingestTrackingFrame(raw, now)) {
    pipelineStream.set({ lastFrameAt: now });
    return;
  }
  const trt = parseTrtFrame(raw, now);
  if (trt) {
    pipelineStream.set({ lastFrameAt: now, trtBuild: trt.finished ? null : trt.build });
    return;
  }
  const progress = parseProgressFrame(raw, now);
  pipelineStream.set(state => ({
    lastFrameAt: now,
    ...(progress ? { transfers: { ...state.transfers, [progress.modelId]: progress } } : {}),
  }));
}

/** Forget a finished transfer (its row now reads from the manifest). */
export function clearTransfer(modelId: string): void {
  pipelineStream.set(state => {
    if (!(modelId in state.transfers)) return {};
    const { [modelId]: _removed, ...rest } = state.transfers;
    return { transfers: rest };
  });
}

const socketUrl = (): string => `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/v1/pipeline/status`;

let socket: ResilientSocket | null = null;
let holders = 0;

/**
 * Shared connection to `/ws/v1/pipeline/status`. Reference counted: the first holder opens it, the last one closes it.
 * A backend restart is invisible to the UI apart from the `connection` indicator: the socket retries with exponential backoff and,
 * once it is open again, `resyncs` increments so readers re-fetch what they may have missed. The server sends no heartbeat while
 * idle, so the stale-link watchdog is off unless a caller passes `staleMs`.
 */
export function acquirePipelineStream(options: { makeSocket?: ConstructorParameters<typeof ResilientSocket>[0]['makeSocket']; staleMs?: number; random?: () => number } = {}): () => void {
  holders += 1;
  if (!socket) {
    socket = new ResilientSocket({
      url: socketUrl, onMessage: ingestFrame, staleMs: options.staleMs, makeSocket: options.makeSocket, random: options.random,
      onState: onConnectionState,
      onOpen: reconnected => { if (reconnected) pipelineStream.set(value => ({ resyncs: value.resyncs + 1 })); },
    });
    socket.start();
  }
  let released = false;
  return () => {
    if (released) return;
    released = true;
    holders = Math.max(0, holders - 1);
    if (holders === 0 && socket) { socket.stop(); socket = null; pipelineStream.set({ connection: 'offline', trtBuild: null }); }
  };
}

/** A build can only be believed while the socket is up: if the server went away mid-build, the build went with it. */
export function onConnectionState(state: SocketState): void {
  pipelineStream.set(state === 'offline' || state === 'reconnecting' ? { connection: state, trtBuild: null } : { connection: state });
}

/** Test hook: drop the shared socket and all state. */
export function resetPipelineStream(): void {
  socket?.stop(); socket = null; holders = 0;
  pipelineStream.set({ connection: 'offline', resyncs: 0, lastFrameAt: 0, transfers: {}, trtBuild: null });
}
