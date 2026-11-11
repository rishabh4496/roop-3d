import { useEffect, useState } from 'react';
import { api } from '../../api';
import { parseEnvironments, type StudioEnvironment } from './studioTools';

export type EnvironmentsState = { state: 'idle' | 'loading' | 'ready' | 'error'; environments: StudioEnvironment[]; error: string };

let cache: StudioEnvironment[] | null = null;
let pending: Promise<StudioEnvironment[]> | null = null;

/** The five built-in CC0 studio lighting probes (`GET /rendering/studio-environments`), fetched once per page load. */
export function loadStudioEnvironments(signal?: AbortSignal): Promise<StudioEnvironment[]> {
  if (cache) return Promise.resolve(cache);
  pending ??= api<unknown>('/rendering/studio-environments', { signal })
    .then(raw => { const list = parseEnvironments(raw); if (!list.length) throw new Error('The service listed no usable lighting environments.'); cache = list; return list; })
    .finally(() => { pending = null; });
  return pending;
}

export const resetStudioEnvironmentCache = (): void => { cache = null; pending = null; };

/** Loads lazily: nothing is fetched until `enabled` is true (the studio-lighting panel is opened). */
export function useStudioEnvironments(enabled: boolean): EnvironmentsState & { retry: () => void } {
  const [result, setResult] = useState<EnvironmentsState>({ state: cache ? 'ready' : 'idle', environments: cache ?? [], error: '' });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!enabled || (cache && attempt === 0)) { if (cache) setResult({ state: 'ready', environments: cache, error: '' }); return undefined; }
    let live = true;
    setResult(previous => ({ ...previous, state: 'loading', error: '' }));
    loadStudioEnvironments()
      .then(environments => { if (live) setResult({ state: 'ready', environments, error: '' }); })
      .catch((error: Error) => { if (live) setResult({ state: 'error', environments: [], error: error.message }); });
    return () => { live = false; };
  }, [enabled, attempt]);
  return { ...result, retry: () => { resetStudioEnvironmentCache(); setAttempt(value => value + 1); } };
}
