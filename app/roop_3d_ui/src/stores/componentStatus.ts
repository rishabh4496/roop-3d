import { useEffect } from 'react';
import { api } from '../api';
import type { ComponentStatus } from '../contracts';
import { RUNTIME_SETTINGS_CHANGED } from '../runtimeSettings';
import { createAtomicStore, useStoreSelector } from './atomicStore';

type State = { value: ComponentStatus | null; error: string; loading: boolean };
export const componentStatusStore = createAtomicStore<State>({ value: null, error: '', loading: false });
export const useComponentStatus = <S,>(selector: (state: State) => S) => useStoreSelector(componentStatusStore, selector);
let inflight: Promise<void> | null = null;

export function refreshComponentStatus(): Promise<void> {
  if (inflight) return inflight;
  componentStatusStore.set({ loading: true });
  inflight = api<ComponentStatus>('/models/components')
    .then(value => {
      if (value?.schema_version !== 1 || !Array.isArray(value.components) || !value.renderer) throw new Error('Unexpected response from /models/components');
      componentStatusStore.set({ value, error: '' });
    })
    .catch(reason => componentStatusStore.set({ error: reason instanceof Error ? reason.message : 'Could not read component diagnostics.' }))
    .finally(() => { inflight = null; componentStatusStore.set({ loading: false }); });
  return inflight;
}

/** Shared read-only request, refreshed while visible and after a saved runtime change. */
export function useComponentStatusPolling(intervalMs = 10000) {
  useEffect(() => {
    void refreshComponentStatus();
    const refresh = () => { if (document.visibilityState !== 'hidden') void refreshComponentStatus(); };
    const timer = window.setInterval(refresh, intervalMs);
    window.addEventListener(RUNTIME_SETTINGS_CHANGED, refresh);
    return () => { window.clearInterval(timer); window.removeEventListener(RUNTIME_SETTINGS_CHANGED, refresh); };
  }, [intervalMs]);
}
