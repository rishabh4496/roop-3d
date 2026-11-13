import { createAtomicStore, useStoreSelector } from './atomicStore';

/** One `tracking.telemetry` frame from `/ws/v1/pipeline/status`, fully defaulted. `null` always means "not measured", never zero. */
export interface TrackingSample {
  schemaVersion: 1 | 2;
  jobId: string | null;
  frameIndex: number;
  status: 'tracked' | 'coasting' | 'lost';
  reason: string;
  confidence: number | null;
  yawDeg: number | null;
  pitchDeg: number | null;
  rollDeg: number | null;
  quat: [number, number, number, number] | null;
  translationPx: [number, number] | null;
  translationM: [number, number, number] | null;
  scalePxPerM: number | null;
  reprojRmsPx: number | null;
  poseSource: string | null;
  sphericalBin: string | null;
  jitterPx: number | null;
  cutoffHz: number | null;
  stability: number | null;
  shotCut: boolean;
  cutReason: string | null;
  shotCuts: number;
  flash: boolean | null;
  maskFlow: number | null;
  selfOccluded: number | null;
  receivedAt: number;
}

/** Samples kept for the stability graph (about 16 s at the server's 15 Hz telemetry throttle). */
export const TRACKING_HISTORY = 240;

const STATUSES = ['tracked', 'coasting', 'lost'] as const;
const num = (value: unknown): number | null => (typeof value === 'number' && Number.isFinite(value) ? value : null);
const text = (value: unknown): string | null => (typeof value === 'string' && value ? value : null);
const tuple = (value: unknown, length: number): number[] | null => {
  if (!Array.isArray(value) || value.length !== length) return null;
  const numbers = value.map(num);
  return numbers.every(item => item !== null) ? numbers as number[] : null;
};
const unit = (value: unknown): number | null => { const n = num(value); return n === null ? null : Math.min(1, Math.max(0, n)); };
const nonNegative = (value: unknown): number | null => { const n = num(value); return n === null || n < 0 ? null : n; };

/** Validates one frame; returns null for anything that is not a usable `tracking.telemetry` frame. Never throws. */
export function parseTrackingFrame(raw: unknown, now = Date.now()): TrackingSample | null {
  if (!raw || typeof raw !== 'object') return null;
  const frame = raw as Record<string, unknown>;
  if (frame.type !== 'tracking.telemetry') return null;
  const index = num(frame.frame_index);
  if (index === null || index < 0) return null;
  const status = (STATUSES as readonly unknown[]).includes(frame.status) ? frame.status as TrackingSample['status'] : 'lost';
  return {
    schemaVersion: frame.schema_version === 2 ? 2 : 1,
    jobId: text(frame.job_id), frameIndex: Math.floor(index), status, reason: text(frame.reason) ?? '',
    confidence: unit(frame.confidence),
    yawDeg: num(frame.yaw_deg), pitchDeg: num(frame.pitch_deg), rollDeg: num(frame.roll_deg),
    quat: tuple(frame.quat_wxyz, 4) as TrackingSample['quat'],
    translationPx: tuple(frame.translation_px, 2) as TrackingSample['translationPx'],
    translationM: tuple(frame.translation_m, 3) as TrackingSample['translationM'],
    scalePxPerM: nonNegative(frame.pose_scale_px_per_m),
    reprojRmsPx: nonNegative(frame.reproj_rms_px),
    poseSource: text(frame.pose_source), sphericalBin: text(frame.spherical_bin),
    jitterPx: nonNegative(frame.jitter_px), cutoffHz: nonNegative(frame.one_euro_cutoff_hz), stability: unit(frame.stability_score),
    shotCut: frame.shot_cut === true, cutReason: text(frame.cut_reason),
    shotCuts: Math.max(0, Math.floor(num(frame.shot_cuts) ?? 0)),
    flash: typeof frame.scene_flash === 'boolean' ? frame.scene_flash : null,
    maskFlow: unit(frame.mask_flow_confidence), selfOccluded: unit(frame.self_occluded_fraction),
    receivedAt: now,
  };
}

interface TrackingState {
  latest: TrackingSample | null;
  history: TrackingSample[];
  /** The run that produced `history`; a different job or a frame index going backwards starts a new history. */
  jobId: string | null;
  ended: boolean;
}
const INITIAL: TrackingState = { latest: null, history: [], jobId: null, ended: false };
export const trackingStream = createAtomicStore<TrackingState>(INITIAL);
export const useTrackingLatest = (): TrackingSample | null => useStoreSelector(trackingStream, state => state.latest);
export const useTrackingHistory = (): TrackingSample[] => useStoreSelector(trackingStream, state => state.history);
export const useTrackingEnded = (): boolean => useStoreSelector(trackingStream, state => state.ended);

/** Appends a validated sample; a new job, or a frame index that moves backwards (rewind, restart), begins a fresh history. */
export function ingestTrackingFrame(raw: unknown, now = Date.now()): boolean {
  const sample = parseTrackingFrame(raw, now);
  if (!sample) return false;
  const ended = (raw as Record<string, unknown>).state === 'ended';
  trackingStream.set(state => {
    const last = state.history[state.history.length - 1];
    const fresh = !last || sample.jobId !== state.jobId || sample.frameIndex < last.frameIndex;
    const base = fresh ? [] : state.history;
    // The server resends the latest frame on every reconnect; an identical index is replaced, not duplicated.
    const kept = base.length && base[base.length - 1].frameIndex === sample.frameIndex ? base.slice(0, -1) : base;
    const history = [...kept, sample].slice(-TRACKING_HISTORY);
    return { latest: sample, history, jobId: sample.jobId, ended };
  });
  return true;
}

export const resetTrackingStream = () => trackingStream.reset();

export interface GraphGeometry {
  jitter: string;
  stability: string;
  /** x positions (0..width) of shot cuts and flashes. */
  cuts: number[];
  flashes: number[];
  /** Largest jitter shown, in px (the left axis). */
  jitterMax: number;
}

/**
 * SVG point lists for the stability graph. Samples where a value was not measured break the line (they are skipped, not drawn as zero),
 * jitter is scaled to its own maximum (at least 1 px so a perfectly still head does not look noisy), stability is fixed to 0..1.
 */
export function buildGraphGeometry(samples: readonly TrackingSample[], width: number, height: number): GraphGeometry {
  const n = samples.length;
  const x = (index: number) => (n <= 1 ? width / 2 : (index / (n - 1)) * width);
  const jitterMax = Math.max(1, ...samples.map(sample => sample.jitterPx ?? 0));
  const line = (pick: (sample: TrackingSample) => number | null, scale: (value: number) => number) => {
    const segments: string[] = [];
    let open = '';
    samples.forEach((sample, index) => {
      const value = pick(sample);
      if (value === null) { if (open) segments.push(open); open = ''; return; }
      open += `${open ? ' ' : ''}${x(index).toFixed(1)},${scale(value).toFixed(1)}`;
    });
    if (open) segments.push(open);
    return segments.join(' | ');
  };
  const cuts: number[] = []; const flashes: number[] = [];
  samples.forEach((sample, index) => {
    if (sample.shotCut) cuts.push(Number(x(index).toFixed(1)));
    if (sample.flash) flashes.push(Number(x(index).toFixed(1)));
  });
  return {
    jitter: line(sample => sample.jitterPx, value => height - (Math.min(value, jitterMax) / jitterMax) * height),
    stability: line(sample => sample.stability, value => height - value * height),
    cuts, flashes, jitterMax,
  };
}
