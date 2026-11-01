import type { Telemetry } from '../contracts';
import { createAtomicStore, shallowEqual, useStoreSelector } from './atomicStore';

/** A `pipeline.status` / `job.progress` frame as streamed by `/ws/telemetry` (fields the UI actually reads). */
export interface LiveFrame {
  type?: string;
  stage?: string | null;
  fps?: number;
  stage_progress_pct?: number | null;
  eta_seconds?: number | null;
  vram_available_gb?: number | null;
  stage_timings_ms?: Record<string, number>;
  [key: string]: unknown;
}

export interface TelemetryState {
  /** Latest `system.telemetry` hardware snapshot (GPUs, free VRAM, CPU count). */
  hardware: Telemetry | null;
  /** Latest `pipeline.status` frame. */
  pipeline: LiveFrame | null;
  /** Latest `job.progress` frame. */
  progress: LiveFrame | null;
  /** Whether a telemetry socket is currently connected. */
  online: boolean;
}

const INITIAL: TelemetryState = { hardware: null, pipeline: null, progress: null, online: false };

/**
 * Stage 11: the single sink for `/ws/telemetry` and `/api/v1/ws/events` hardware ticks. Nothing here is React state in
 * `App`, so a 30 Hz stream re-renders only the components that subscribe to a slice (`PipelinePill`, the Diagnostics
 * HUD, the Processing GPU readout), never the parameter panels, the timeline filmstrip or the Face Manager.
 */
export const telemetryStore = createAtomicStore<TelemetryState>(INITIAL);

export const publishHardware = (hardware: Telemetry | null) => telemetryStore.set({ hardware });
export const publishPipeline = (pipeline: LiveFrame) => telemetryStore.set({ pipeline });
export const publishProgress = (progress: LiveFrame) => telemetryStore.set({ progress });
export const publishOnline = (online: boolean) => telemetryStore.set({ online });
export const resetTelemetry = () => telemetryStore.reset();

/** Subscribe the calling component to one slice of telemetry. Re-renders only when `selector`'s result changes. */
export function useTelemetry<S>(selector: (state: TelemetryState) => S, isEqual: (a: S, b: S) => boolean = Object.is): S {
  return useStoreSelector(telemetryStore, selector, isEqual);
}
export { shallowEqual };
