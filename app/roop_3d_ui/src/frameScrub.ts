import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { RUNTIME_SETTINGS_CHANGED } from './runtimeSettings';

/**
 * Timeline scrubbing client for `GET /api/v1/media/{info,frame,filmstrip}`.
 *
 * `FrameScrubber` is the "latest frame wins" queue: exactly one frame request is ever in flight, and when it lands the
 * newest wanted frame is requested next. Dragging the playhead therefore updates the canvas as fast as the server can
 * answer, with no pile of stale requests (the old preview aborted and re-armed a 100 ms timer on every move, so a
 * continuous drag never produced a frame). Face detection is asked for only once the playhead has settled, and the
 * server answers that from its decoded-frame cache.
 */
export interface FrameFace {
  /** Left to right rank in the frame: the `faceRank` face extraction takes. */
  face_index: number;
  track_id: string | null;
  /** Native-pixel box [x1, y1, x2, y2]. */
  bbox: [number, number, number, number];
  /** The same box as fractions of the frame, which is what the letterbox mapping wants. */
  bbox_norm: [number, number, number, number];
  kps_5: [number, number][];
  /** Legacy frame (pitch + = looking down). */
  pose: { yaw: number; pitch: number; roll: number };
  /** Standard frame (pitch + = looking up). */
  pose3d: { yaw: number; pitch: number; roll: number };
  pose_cell?: string;
  confidence: number | null;
  /** Stage 12: frame-local id, persistent person cluster (rank-preserved: unchanged when the left-to-right order swaps). */
  face_id?: string;
  persistent_face_id?: number;
  cluster_id?: number | null;
  is_target?: boolean;
  target_number?: number | null;
  detection_tier?: 'primary' | 'profile_tolerant';
}

export interface FrameMeta {
  frame_index: number;
  requested_index: number | null;
  timestamp_sec: number;
  /** Native frame size, the coordinate space of `faces[].bbox` and `kps_5`. */
  width: number;
  height: number;
  /** Size of the JPEG that was sent (it may be shrunk). */
  image_width: number;
  image_height: number;
  total_frames: number;
  fps: number;
  cache: 'hit' | 'miss';
  decode_ms: number;
  faces?: FrameFace[];
  unsupported?: string[];
}

export interface MediaInfo {
  width: number; height: number; fps: number; fps_rational: string; total_frames: number; duration_sec: number;
  rotation_degrees: number; frame_count_declared: boolean;
}

export interface FilmstripFrame { slot: number; frame_index: number; requested_frame_index: number; timestamp_sec: number; data_url: string }
export interface Filmstrip { count: number; thumb_width: number; thumb_height: number; total_frames: number; duration_sec: number; frames: FilmstripFrame[] }

async function failure(response: Response, fallback: string): Promise<Error> {
  let message = `${fallback} (${response.status})`;
  try {
    const body = await response.json();
    if (typeof body?.error?.message === 'string') message = body.error.message;
  } catch { /* keep the status-based message when the body is not JSON */ }
  return new Error(message);
}

export async function fetchFrame(path: string, frameIndex: number, options: { detect?: boolean; signal?: AbortSignal; maxSize?: number } = {}): Promise<{ blob: Blob; meta: FrameMeta }> {
  const params = new URLSearchParams({ path, frame_index: String(frameIndex) });
  if (options.detect) params.set('detect_faces', 'true');
  if (options.maxSize) params.set('max_size', String(options.maxSize));
  const response = await fetch(`/api/v1/media/frame?${params}`, { signal: options.signal, cache: 'no-store' });
  if (!response.ok) throw await failure(response, 'Frame preview failed');
  const header = response.headers.get('X-Frame-Meta');
  if (!header) throw new Error('Frame response carried no metadata');
  return { blob: await response.blob(), meta: JSON.parse(header) as FrameMeta };
}

export async function fetchMediaInfo(path: string, signal?: AbortSignal): Promise<MediaInfo> {
  const response = await fetch(`/api/v1/media/info?${new URLSearchParams({ path })}`, { signal, cache: 'no-store' });
  if (!response.ok) throw await failure(response, 'Media info failed');
  const body = await response.json();
  if (!Number.isFinite(body?.total_frames) || !Number.isFinite(body?.width)) throw new Error('Unexpected media info response');
  return body as MediaInfo;
}

export async function fetchFilmstrip(path: string, count: number, signal?: AbortSignal): Promise<Filmstrip> {
  const response = await fetch(`/api/v1/media/filmstrip?${new URLSearchParams({ path, count: String(count) })}`, { signal, cache: 'no-store' });
  if (!response.ok) throw await failure(response, 'Filmstrip failed');
  const body = await response.json();
  if (!Array.isArray(body?.frames)) throw new Error('Unexpected filmstrip response');
  return body as Filmstrip;
}

// ------------------------------------------------------------------------------------------------ scrubber
export interface ScrubShown { index: number; url: string; meta: FrameMeta; faces: FrameFace[] | null }
export interface ScrubState { shown: ScrubShown | null; loading: boolean; detecting: boolean; error: string }

type Entry = { url: string; meta: FrameMeta; faces: FrameFace[] | null };
export interface ScrubberOptions {
  /** Playhead idle time after which faces are requested for the wanted frame. */
  settleMs?: number;
  cacheSize?: number;
  /** Runs the queue pump; defaults to requestAnimationFrame so requests align with the display. */
  schedule?: (run: () => void) => void;
  makeUrl?: (blob: Blob) => string;
  revokeUrl?: (url: string) => void;
}

const EMPTY: ScrubState = { shown: null, loading: false, detecting: false, error: '' };

export class FrameScrubber {
  private path = '';
  private wanted = 0;
  private enabled = true;
  private detect = true;
  private settled = true;
  private inFlight: AbortController | null = null;
  private generation = 0;
  private pumpQueued = false;
  private settleTimer: ReturnType<typeof setTimeout> | undefined;
  private cache = new Map<number, Entry>();
  private state: ScrubState = EMPTY;
  private listeners = new Set<() => void>();
  readonly requests: { index: number; detect: boolean }[] = [];

  constructor(private fetcher: typeof fetchFrame = fetchFrame, private options: ScrubberOptions = {}) {}

  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; };
  getState = () => this.state;

  private set(patch: Partial<ScrubState>) {
    const next = { ...this.state, ...patch };
    if (next.shown === this.state.shown && next.loading === this.state.loading && next.detecting === this.state.detecting && next.error === this.state.error) return;
    this.state = next;
    this.listeners.forEach(listener => listener());
  }

  setTarget(path: string, index: number, enabled: boolean, detect = true) {
    if (path !== this.path) { this.reset(); this.path = path; }
    this.enabled = enabled; this.detect = detect;
    if (!path || !enabled) { clearTimeout(this.settleTimer); return; }
    if (index !== this.wanted || this.state.shown === null) {
      this.wanted = index;
      this.settled = false;
      clearTimeout(this.settleTimer);
      this.settleTimer = setTimeout(() => { this.settled = true; this.queuePump(); }, this.options.settleMs ?? 120);
    }
    this.queuePump();
  }

  retry() { this.set({ error: '' }); this.queuePump(); }

  invalidate() {
    const path = this.path, index = this.wanted, enabled = this.enabled, detect = this.detect;
    this.reset();
    this.setTarget(path, index, enabled, detect);
  }

  /** Aborts the request, frees every cached frame and forgets the clip. */
  reset() {
    this.generation++;
    this.inFlight?.abort(); this.inFlight = null;
    clearTimeout(this.settleTimer);
    const revoke = this.options.revokeUrl ?? ((url: string) => URL.revokeObjectURL(url));
    for (const entry of this.cache.values()) revoke(entry.url);
    this.cache.clear();
    this.path = ''; this.wanted = 0; this.settled = true; this.pumpQueued = false;
    this.set({ ...EMPTY });
  }

  private queuePump() {
    if (this.pumpQueued) return;
    this.pumpQueued = true;
    const run = () => { this.pumpQueued = false; this.pump(); };
    (this.options.schedule ?? (typeof requestAnimationFrame === 'function' ? requestAnimationFrame : (fn: () => void) => setTimeout(fn, 0)))(run);
  }

  private show(index: number, entry: Entry) {
    const shown = this.state.shown;
    if (shown && shown.index === index && shown.url === entry.url && shown.faces === entry.faces) return;
    this.set({ shown: { index, url: entry.url, meta: entry.meta, faces: entry.faces } });
  }

  private pump() {
    if (!this.enabled || !this.path || this.inFlight) return;
    const index = this.wanted;
    const cached = this.cache.get(index);
    if (cached) {
      this.cache.delete(index); this.cache.set(index, cached);   // most recently used
      this.show(index, cached);
      if (!this.settled || cached.faces !== null || !this.detect) { this.set({ loading: false, detecting: false }); return; }
    }
    if (this.state.error && this.state.shown?.index === index) return;   // do not hammer a frame that failed; retry() or a new frame resumes
    this.start(index, this.detect && this.settled);
  }

  private start(index: number, detect: boolean) {
    const controller = new AbortController();
    this.inFlight = controller;
    const generation = this.generation, path = this.path;
    this.requests.push({ index, detect });
    this.set({ loading: true, detecting: detect, error: '' });
    this.fetcher(path, index, { detect, signal: controller.signal }).then(({ blob, meta }) => {
      if (generation !== this.generation) return;
      this.inFlight = null;
      const prior = this.cache.get(index);
      const makeUrl = this.options.makeUrl ?? ((value: Blob) => URL.createObjectURL(value));
      const entry: Entry = { url: prior?.url ?? makeUrl(blob), meta, faces: detect ? meta.faces ?? [] : prior?.faces ?? null };
      this.cache.delete(index); this.cache.set(index, entry);
      this.trim();
      this.show(index, entry);
      this.set({ loading: false, detecting: false });
      this.queuePump();                                   // the wanted frame may have moved while this one was in flight
    }, (error: Error) => {
      if (generation !== this.generation) return;
      this.inFlight = null;
      if (error?.name === 'AbortError') return;
      this.set({ loading: false, detecting: false, error: error?.message || 'Frame preview failed' });
    });
  }

  private trim() {
    const limit = this.options.cacheSize ?? 24, revoke = this.options.revokeUrl ?? ((url: string) => URL.revokeObjectURL(url));
    for (const [index, entry] of this.cache) {
      if (this.cache.size <= limit) break;
      if (entry.url === this.state.shown?.url || index === this.wanted) continue;
      this.cache.delete(index); revoke(entry.url);
    }
  }
}

export interface ScrubOptions {
  path: string;
  time: number;
  fps: number;
  /** Last valid frame index (total_frames - 1). */
  maxIndex: number;
  /** False while the video element is playing: no frames are requested. */
  enabled: boolean;
  /** Ask for the faces of the frame once the playhead settles. */
  detect?: boolean;
}

export function useScrubFrame({ path, time, fps, maxIndex, enabled, detect = true }: ScrubOptions) {
  const ref = useRef<FrameScrubber | null>(null);
  if (ref.current === null) ref.current = new FrameScrubber();
  const scrubber = ref.current;
  const state = useSyncExternalStore(scrubber.subscribe, scrubber.getState, scrubber.getState);
  const index = Math.max(0, Math.min(Math.max(0, maxIndex), Math.round(time * Math.max(1, fps))));
  useEffect(() => { scrubber.setTarget(path, index, enabled, detect); }, [scrubber, path, index, enabled, detect]);
  useEffect(() => () => scrubber.reset(), [scrubber]);
  useEffect(() => {
    const refresh = () => scrubber.invalidate();
    window.addEventListener(RUNTIME_SETTINGS_CHANGED, refresh);
    window.addEventListener('target-angle-scan-completed', refresh);
    return () => { window.removeEventListener(RUNTIME_SETTINGS_CHANGED, refresh); window.removeEventListener('target-angle-scan-completed', refresh); };
  }, [scrubber]);
  const shown = state.shown;
  return {
    source: shown?.url ?? '', error: state.error, loading: state.loading, detecting: state.detecting,
    shown, meta: shown?.meta ?? null,
    /** True when the frame on screen is the one under the playhead (not a progressive earlier frame). */
    current: !!shown && shown.index === index,
    /** Faces of the frame on screen, or null until detection has run for it. */
    faces: shown?.faces ?? null, index, retry: () => scrubber.retry(),
  };
}

// ------------------------------------------------------------------------------------------------ info / filmstrip
type Fetched<T> = { status: 'idle' | 'loading' | 'ready' | 'error'; data: T | null; error: string };
const cached = new Map<string, unknown>();

function useFetched<T>(key: string, enabled: boolean, load: (signal: AbortSignal) => Promise<T>): Fetched<T> {
  const [state, setState] = useState<Fetched<T> & { key: string }>({ key: '', status: 'idle', data: null, error: '' });
  useEffect(() => {
    if (!enabled) { setState({ key: '', status: 'idle', data: null, error: '' }); return; }
    if (cached.has(key)) { setState({ key, status: 'ready', data: cached.get(key) as T, error: '' }); return; }
    const controller = new AbortController();
    setState({ key, status: 'loading', data: null, error: '' });
    load(controller.signal).then(data => {
      if (cached.size >= 16) cached.delete(cached.keys().next().value as string);
      cached.set(key, data);
      if (!controller.signal.aborted) setState({ key, status: 'ready', data, error: '' });
    }, (error: Error) => { if (error.name !== 'AbortError') setState({ key, status: 'error', data: null, error: error.message }); });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);
  return state.key === key || !enabled ? state : { status: enabled ? 'loading' : 'idle', data: null, error: '' };
}

/** Width, height, fps, total frames and duration, fetched the moment a target is loaded. */
export function useMediaInfo(path: string) {
  return useFetched<MediaInfo>(`info|${path}`, !!path, signal => fetchMediaInfo(path, signal));
}

/** `count` evenly spaced thumbnails in one request. */
export function useFilmstrip(path: string, count: number) {
  return useFetched<Filmstrip>(`strip|${path}|${count}`, !!path, signal => fetchFilmstrip(path, count, signal));
}

/** Zoomed timelines want more thumbnails: 16 minimum, one per ~120 px, in steps of 8 so the count (and the cache) is stable. */
export function filmstripCount(trackWidth: number): number {
  return Math.max(16, Math.min(64, Math.ceil(trackWidth / 120 / 8) * 8));
}
