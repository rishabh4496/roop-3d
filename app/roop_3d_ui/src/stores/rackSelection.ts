import { createAtomicStore, useStoreSelector } from './atomicStore';
import { stripRemovedParameters } from '../true3d';

/**
 * Engine & Model Rack choices that are not already part of `WorkstationParameters`.
 *
 * `detector` mirrors the persisted global `detector_model_id`. The head topology engine, shader mode and camera live in the
 * workspace parameters. Choices the engine has no stage or setting for are not kept at all. Entries written by earlier
 * versions (the removed swapper choice, upscaler, landmarker ...) are dropped on load, and removed keys inside saved batch
 * parameters are stripped.
 */
export type RackState = {
  detector: string;
  /** Parameters the Batch screen merges into every queued job (only fields the user touched). */
  batchParameters: Record<string, unknown>;
};
const KEY = 'roop3d-model-rack';
const DEFAULTS: RackState = { detector: 'scrfd', batchParameters: {} };

function load(): RackState {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const value = JSON.parse(raw);
    if (!value || typeof value !== 'object') return DEFAULTS;
    return {
      detector: typeof value.detector === 'string' ? value.detector : DEFAULTS.detector,
      batchParameters: value.batchParameters && typeof value.batchParameters === 'object' && !Array.isArray(value.batchParameters)
        ? stripRemovedParameters(value.batchParameters) : {},
    };
  } catch { return DEFAULTS; }
}

export const rackStore = createAtomicStore<RackState>(load());
rackStore.subscribe(() => { try { window.localStorage.setItem(KEY, JSON.stringify(rackStore.get())); } catch { /* storage may be blocked; the rack still works in memory */ } });

export const setRack = (patch: Partial<RackState>) => rackStore.set(patch);
export const useRack = <S,>(selector: (state: RackState) => S) => useStoreSelector(rackStore, selector);
