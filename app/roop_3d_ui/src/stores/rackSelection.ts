import { createAtomicStore, useStoreSelector } from './atomicStore';

/** Models a single detected person is pinned to. Either field may be unset (= use the scene-wide choice). */
export type PersonModels = { swapper?: string; enhancer?: string };

/**
 * Engine & Model Rack choices that are not already part of `WorkstationParameters`.
 *
 * `swapper` is sent as the job's `swapper_model_id` (the one model knob the render engine has). The others
 * (landmarker, upscaler, cranial silhouette, per-person overrides) have no render-engine setting yet: they are
 * remembered here so the choice survives a reload, and the Rack labels them "saved preference" instead of pretending they apply.
 * The detector mirrors the persisted global detector_model_id for frame analysis and rendering.
 */
export type RackState = {
  swapper: string | null;
  detector: string; landmarker: string; upscaler: string; cranial: boolean;
  perPerson: Record<string, PersonModels>;
  /** Parameters the Batch screen merges into every queued job (only fields the user touched). */
  batchParameters: Record<string, unknown>;
};
const KEY = 'roop3d-model-rack';
const DEFAULTS: RackState = { swapper: null, detector: 'scrfd', landmarker: 'dense_3d', upscaler: 'none', cranial: true, perPerson: {}, batchParameters: {} };

function load(): RackState {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULTS;
    const value = JSON.parse(raw);
    return value && typeof value === 'object' ? { ...DEFAULTS, ...value } : DEFAULTS;
  } catch { return DEFAULTS; }
}

export const rackStore = createAtomicStore<RackState>(load());
rackStore.subscribe(() => { try { window.localStorage.setItem(KEY, JSON.stringify(rackStore.get())); } catch { /* storage may be blocked; the rack still works in memory */ } });

export const setRack = (patch: Partial<RackState>) => rackStore.set(patch);
export const useRack = <S,>(selector: (state: RackState) => S) => useStoreSelector(rackStore, selector);

export function setPersonModels(personKey: string, patch: PersonModels) {
  rackStore.set(state => {
    const next = { ...state.perPerson[personKey], ...patch };
    (Object.keys(next) as (keyof PersonModels)[]).forEach(key => { if (!next[key]) delete next[key]; });
    const perPerson = { ...state.perPerson };
    if (Object.keys(next).length) perPerson[personKey] = next; else delete perPerson[personKey];
    return { perPerson };
  });
}
