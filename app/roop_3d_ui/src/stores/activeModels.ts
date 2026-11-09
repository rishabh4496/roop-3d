import { useEffect } from 'react';
import { api } from '../api';
import { createAtomicStore, useStoreSelector } from './atomicStore';
import { RUNTIME_SETTINGS_CHANGED } from '../runtimeSettings';

/** One stage's loaded model, from `GET /models/active`. `null` = no session of that stage is in memory. */
export type ActiveModel = {
  stage: 'detector' | 'swapper' | 'enhancer'; model_id: string; model_path: string; filename: string;
  provider: string; precision: string; sha256: string | null; loaded_at: number; switches: number;
} | null;
export type ActiveModels = {
  selected: { swapper_model: string; detector_model: string; enhancer_model: string };
  active: { detector: ActiveModel; swapper: ActiveModel; enhancer: ActiveModel };
};
type State = { value: ActiveModels | null; error: string };

export const activeModelStore = createAtomicStore<State>({ value: null, error: '' });
export const useActiveModels = <S,>(selector: (state: State) => S) => useStoreSelector(activeModelStore, selector);

let inflight: Promise<void> | null = null;
export function refreshActiveModels(): Promise<void> {
  if (inflight) return inflight;
  inflight = api<ActiveModels>('/models/active')
    .then(value => {
      if (!value || typeof value !== 'object' || !value.active || typeof value.active !== 'object') throw new Error('unexpected response from /models/active');
      activeModelStore.set({ value, error: '' });
    })
    .catch(reason => activeModelStore.set({ error: reason instanceof Error ? reason.message : 'Could not read the loaded models.' }))
    .finally(() => { inflight = null; });
  return inflight;
}

/** Keeps the read-back current: on mount, after every settings change, and while the tab is visible. */
export function useActiveModelPolling(intervalMs = 4000) {
  useEffect(() => {
    void refreshActiveModels();
    const refresh = () => { if (typeof document === 'undefined' || document.visibilityState !== 'hidden') void refreshActiveModels(); };
    const timer = window.setInterval(refresh, intervalMs);
    window.addEventListener(RUNTIME_SETTINGS_CHANGED, refresh);
    return () => { window.clearInterval(timer); window.removeEventListener(RUNTIME_SETTINGS_CHANGED, refresh); };
  }, [intervalMs]);
}
