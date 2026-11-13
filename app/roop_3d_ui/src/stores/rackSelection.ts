import { createAtomicStore, useStoreSelector } from './atomicStore';

/**
 * Engine & Model Rack choices that are not already part of `WorkstationParameters`.
 *
 * `swapper` is sent as the job's `swapper_model_id` and `detector` mirrors the persisted global `detector_model_id`; both reach the render
 * engine. Choices the engine has no stage or setting for (frame upscaler, landmarker, cranial silhouette, per-person models) are not kept at
 * all: the rack lists them as unsupported instead of remembering a selection that changes nothing.
 */
export type RackState = {
  swapper: string | null;
  detector: string;
  /** Parameters the Batch screen merges into every queued job (only fields the user touched). */
  batchParameters: Record<string, unknown>;
};
const KEY = 'roop3d-model-rack';
const DEFAULTS: RackState = { swapper: null, detector: 'scrfd', batchParameters: {} };

function load(): RackState {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const value = JSON.parse(raw);
    if (!value || typeof value !== 'object') return DEFAULTS;
    // Only the known keys: entries written by earlier versions (upscaler, landmarker, per-person pins…) are dropped on the next save.
    return {
      swapper: typeof value.swapper === 'string' ? value.swapper : null,
      detector: typeof value.detector === 'string' ? value.detector : DEFAULTS.detector,
      batchParameters: value.batchParameters && typeof value.batchParameters === 'object' ? value.batchParameters : {},
    };
  } catch { return DEFAULTS; }
}

export const rackStore = createAtomicStore<RackState>(load());
rackStore.subscribe(() => { try { window.localStorage.setItem(KEY, JSON.stringify(rackStore.get())); } catch { /* storage may be blocked; the rack still works in memory */ } });

export const setRack = (patch: Partial<RackState>) => rackStore.set(patch);
export const useRack = <S,>(selector: (state: RackState) => S) => useStoreSelector(rackStore, selector);
