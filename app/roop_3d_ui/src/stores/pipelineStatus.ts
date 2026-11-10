import { useEffect } from 'react';
import { api } from '../api';
import { RUNTIME_SETTINGS_CHANGED } from '../runtimeSettings';
import { createAtomicStore, useStoreSelector } from './atomicStore';
import { pushToast } from './toasts';
import type { ActiveModel } from './activeModels';

export type SwitchFailure = { id: number; stage: 'detector' | 'swapper' | 'enhancer'; model_id: string; code: string; reason: string; at: number };
/** `GET /api/pipeline/status`: what each stage has loaded, the precision those sessions really run, and failed switches. */
export type PipelineStatus = {
  pipeline: { engine_state: string; engine_label: string | null; engine_model: string | null; engine_precision: string | null; engine_batch_profile?: number[] | null };
  selected: { swapper_model: string; detector_model: string; enhancer_model: string };
  active: { detector: ActiveModel; swapper: ActiveModel; enhancer: ActiveModel };
  precision: { label: string | null; stage: string | null; stages: Record<string, string | null>; policy: string };
  failures: SwitchFailure[];
};
type State = { value: PipelineStatus | null; error: string };

export const pipelineStatusStore = createAtomicStore<State>({ value: null, error: '' });
/** Failures already raised as toasts, keyed by id + time (ids restart when the server restarts, the time does not repeat). */
const raised = new Set<string>();
export const forgetRaisedFailures = () => raised.clear();
export const usePipelineStatus = <S,>(selector: (state: State) => S) => useStoreSelector(pipelineStatusStore, selector);

export const STAGE_LABEL: Record<SwitchFailure['stage'], string> = { detector: 'Detector', swapper: 'Swapper', enhancer: 'Enhancer' };

/** Toast text for a failed switch: the backend's exact code and reason, and which model is still (or not) in use. */
export function failureToast(failure: SwitchFailure, active: ActiveModel): { title: string; text: string } {
  const still = active ? `Still using ${active.model_id} (${active.filename}).` : 'No model of this stage is loaded.';
  return { title: `${STAGE_LABEL[failure.stage]} did not switch to ${failure.model_id}`, text: `${failure.reason} [${failure.code}] ${still}` };
}

let inflight: Promise<void> | null = null;
export function refreshPipelineStatus(): Promise<void> {
  if (inflight) return inflight;
  inflight = api<PipelineStatus>('/pipeline/status')
    .then(value => {
      if (!value || typeof value !== 'object' || !value.active || typeof value.active !== 'object') throw new Error('unexpected response from /pipeline/status');
      const failures = Array.isArray(value.failures) ? value.failures : [];
      // Each failure is raised once; a failure that happened before this page loaded is raised on first load too.
      for (const failure of failures) {
        const key = `${failure.id}:${failure.at}`;
        if (raised.has(key)) continue;
        raised.add(key);
        const message = failureToast(failure, value.active[failure.stage] ?? null);
        pushToast({ kind: 'error', ...message });
      }
      pipelineStatusStore.set({ value: { ...value, failures }, error: '' });
    })
    .catch(reason => pipelineStatusStore.set({ error: reason instanceof Error ? reason.message : 'Could not read the engine status.' }))
    .finally(() => { inflight = null; });
  return inflight;
}

/** Keeps the engine status current: on mount, after every settings change, and while the tab is visible. */
export function usePipelineStatusPolling(intervalMs = 3000) {
  useEffect(() => {
    void refreshPipelineStatus();
    const refresh = () => { if (typeof document === 'undefined' || document.visibilityState !== 'hidden') void refreshPipelineStatus(); };
    const timer = window.setInterval(refresh, intervalMs);
    window.addEventListener(RUNTIME_SETTINGS_CHANGED, refresh);
    return () => { window.clearInterval(timer); window.removeEventListener(RUNTIME_SETTINGS_CHANGED, refresh); };
  }, [intervalMs]);
}
