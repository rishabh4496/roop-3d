/**
 * View model for the 3-step run-bar pill. Pure functions of the server's `pipeline.status` frames
 * (`/ws/telemetry`), so everything the pill shows is unit-testable without a socket or a clock.
 */

/** Server stage ids mapped 1:1 onto the three user-facing steps. */
export const STEPS = [
  { id: 'STAGE_1_SPARSE_PRESCAN', number: 1, label: 'Pre-Scan' },
  { id: 'STAGE_2_ACTIVE_PREPASS', number: 2, label: 'Range Lock' },
  { id: 'STAGE_3_MAIN_SWAP', number: 3, label: 'Render & Mux' },
];

const ENGINE_BANNERS = {
  COMPILING_TRT: { kind: 'compiling', title: 'Compiling TensorRT engine' },
  RUNNING_CANARY: { kind: 'canary', title: 'Running engine canary' },
};

/** @typedef {{ type?: string, job_id?: string|null, stage?: string|null, stages?: Record<string,string>, stage_progress_pct?: number|null,
 *   skipped_frames_pct?: number|null, fps?: number, eta_seconds?: number|null, engine_state?: string, engine_label?: string|null,
 *   engine_model?: string|null, engine_precision?: string|null, engine_since?: number|null }} PipelineFrame */

/** @param {unknown} value @returns {value is PipelineFrame} */
export const isPipelineFrame = value => !!value && typeof value === 'object' && /** @type {any} */ (value).type === 'pipeline.status';

/** Seconds since `since` (epoch seconds, as the server sends `engine_since`); never negative, null when unknown. */
export function elapsedSeconds(since, nowMs) {
  return typeof since === 'number' && Number.isFinite(since) ? Math.max(0, Math.floor(nowMs / 1000 - since)) : null;
}

/** "1:05", "42s": compact enough for a pill. */
export function formatElapsed(seconds) {
  if (seconds === null || seconds === undefined) return '';
  return seconds >= 60 ? `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}` : `${seconds}s`;
}

const percent = value => (typeof value === 'number' && Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : null);

/**
 * @param {PipelineFrame | null} frame latest frame, or null before the first one arrives
 * @param {number} nowMs
 */
export function buildPillModel(frame, nowMs = Date.now()) {
  const stages = frame?.stages ?? {};
  const steps = STEPS.map(step => {
    const raw = String(stages[step.id] ?? 'PENDING').toUpperCase();
    // RESTORED = satisfied from a saved checkpoint; it counts as done.
    const status = raw === 'RESTORED' ? 'done' : raw.toLowerCase();
    const active = frame?.stage === step.id;
    return { ...step, status, active, progressPct: active && status === 'running' ? percent(frame?.stage_progress_pct) : null };
  });
  const render = steps[2];
  // The render step reaching 100% while still running is the mux/finalize window: every frame is out, the file isn't published yet.
  const muxing = render.status === 'running' && (percent(frame?.stage_progress_pct) ?? 0) >= 100;
  const banner = ENGINE_BANNERS[frame?.engine_state ?? 'IDLE'] ?? null;
  const elapsed = elapsedSeconds(frame?.engine_since, nowMs);
  const parts = [frame?.engine_label, frame?.engine_model, frame?.engine_precision].filter(Boolean);
  return {
    connected: frame !== null,
    idle: !steps.some(step => step.status === 'running') && !banner,
    steps,
    muxing,
    fps: typeof frame?.fps === 'number' ? frame.fps : 0,
    skippedFramesPct: percent(frame?.skipped_frames_pct),
    etaSeconds: typeof frame?.eta_seconds === 'number' ? frame.eta_seconds : null,
    banner: banner && {
      kind: banner.kind,
      title: banner.title,
      label: frame?.engine_label ?? null,
      model: frame?.engine_model ?? null,
      precision: frame?.engine_precision ?? null,
      elapsedSeconds: elapsed,
      text: [banner.title, ...parts, elapsed === null ? '' : formatElapsed(elapsed)].filter(Boolean).join(' · '),
    },
  };
}
