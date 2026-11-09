import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, post } from '../../api';
import type { Coverage, FaceSet, FaceSetView } from '../../contracts';
import AngularCoverageRadar from '../three/AngularCoverageRadar';

const PoseGeometryViewer = lazy(() => import('../three/PoseGeometryViewer'));

type Load<T> = { state: 'idle' | 'loading' | 'ready' | 'error'; value: T | null; error: string };
const IDLE = { state: 'idle', value: null, error: '' } as const;

/** Fetch `path` (null = nothing); a late answer for an earlier FaceSet or head model is dropped. `revision` forces a reload. */
function useFetched<T>(path: string | null, revision = 0): Load<T> & { reload: () => void } {
  const [result, setResult] = useState<Load<T>>(IDLE);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!path) { setResult(IDLE); return undefined; }
    const controller = new AbortController();
    let live = true;
    setResult({ state: 'loading', value: null, error: '' });
    api<T>(path, { signal: controller.signal })
      .then(value => { if (live) setResult({ state: 'ready', value, error: '' }); })
      .catch((error: Error) => { if (live && error.name !== 'AbortError') setResult({ state: 'error', value: null, error: error.message }); });
    return () => { live = false; controller.abort(); };
  }, [path, revision, attempt]);
  return { ...result, reload: useCallback(() => setAttempt(value => value + 1), []) };
}

/**
 * Inspect the fitted anatomical face and observed source-view coverage.
 * Cached reads are inexpensive; reconstruction runs through an explicit button.
 */
export default function FacesetInspector3D({ sources }: { sources: Pick<FaceSet, 'faceset_id' | 'name'>[] }) {
  const [chosen, setChosen] = useState('');
  const id = sources.some(item => item.faceset_id === chosen) ? chosen : sources[0]?.faceset_id ?? '';
  const [revision, setRevision] = useState(0);
  const [build, setBuild] = useState<{ busy: boolean; error: string }>({ busy: false, error: '' });
  const live = useRef(id); live.current = id;
  useEffect(() => setBuild({ busy: false, error: '' }), [id]);

  const views = useFetched<FaceSetView[]>(id ? `/facesets/${encodeURIComponent(id)}/views` : null);
  const meshState = useFetched<Coverage>(id ? `/faceset/3d-state?faceset_id=${encodeURIComponent(id)}` : null, revision);
  const surface = meshState.value?.mesh_full_3d;
  // Stored view pitch is legacy (+ = down); the radar is in the standard frame (+ = up).
  const coverage = useMemo(() => ({ views: (views.value ?? []).map(view => ({ yaw: view.yaw, pitch: -view.pitch })) }), [views.value]);

  const runBuild = async () => {
    const forId = id;
    setBuild({ busy: true, error: '' });
    try {
      await post<Coverage>('/faceset/reconstruct-3d', { faceset_id: forId });
      if (live.current === forId) { setBuild({ busy: false, error: '' }); setRevision(value => value + 1); }
    } catch (error) { if (live.current === forId) setBuild({ busy: false, error: (error as Error).message }); }
  };

  if (!sources.length) return <p className="muted" data-testid="fsz-inspector-empty">Import or build a FaceSet to inspect it in 3D.</p>;
  return (
    <section className="fsz-inspector-3d" aria-label="3D FaceSet inspector" data-testid="fsz-inspector-3d">
      <div className="fsz-inspector-head">
        <label>Inspect .fsz in 3D
          <select aria-label="FaceSet to inspect in 3D" value={id} onChange={event => setChosen(event.target.value)}>
            {sources.map(item => <option key={item.faceset_id} value={item.faceset_id}>{item.name}</option>)}
          </select>
        </label>
        <span className="status-chip">Anatomical facial surface</span>
      </div>
      <div className="fsz-inspector-body">
        <div className="fsz-inspector-viewer">
          {surface && meshState.value
            ? <><Suspense fallback={<div className="skeleton-tile wide" aria-label="Loading 3D viewer" />}><PoseGeometryViewer coverage={meshState.value} track={null} mesh={null} /></Suspense>
                {surface.reconstruction_level !== 'three_tier_facial_surface' && <p role="status" className="identity-warn">Fitted facial model unavailable. This is a canonical preview.{surface.backbone?.reason ? ` ${surface.backbone.reason}` : ''}</p>}
                <button type="button" className="button small" disabled={build.busy} onClick={() => void runBuild()}>{build.busy ? 'Reconstructing…' : 'Refresh facial reconstruction'}</button>
                {build.error && <p role="alert" className="identity-warn">{build.error}</p>}</>
            : <div className="fsz-inspector-placeholder" role="status">
                {meshState.state === 'loading' ? 'Loading…' : <>
                  {meshState.state === 'error' ? meshState.error : 'No fitted facial surface for this FaceSet yet.'}{' '}
                  <button type="button" className="button small accent" disabled={build.busy} onClick={() => void runBuild()}>{build.busy ? 'Reconstructing…' : 'Reconstruct facial surface'}</button>
                  {meshState.state === 'error' && <button type="button" className="button small" onClick={meshState.reload}>Retry</button>}
                </>}
                {build.error && <p role="alert" className="identity-warn">{build.error}</p>}
              </div>}
        </div>
        <div className="fsz-inspector-radar">
          {views.value?.some(view => view.is_synthetic) && <small className="identity-warn">Includes generated pose references. Hidden detail and low-confidence profile identities are inferred.</small>}
          {views.state === 'error' ? <div className="error-retry" role="status"><span>{views.error}</span><button type="button" className="button small" onClick={views.reload}>Retry</button></div>
            : <AngularCoverageRadar coverage={views.state === 'ready' ? coverage : null} name={sources.find(item => item.faceset_id === id)?.name} />}
        </div>
      </div>
    </section>
  );
}
