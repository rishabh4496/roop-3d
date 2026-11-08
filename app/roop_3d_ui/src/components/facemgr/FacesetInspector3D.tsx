import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, post } from '../../api';
import type { FaceSet, FaceSetView, IdentityAssetStatus, Reconstruction } from '../../contracts';
import AngularCoverageRadar from '../three/AngularCoverageRadar';
import { readTemplate, writeTemplate, type HeadTemplate } from '../studio/FaceSetDetail';

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
    setResult({ state: 'loading', value: null, error: '' });
    api<T>(path, { signal: controller.signal })
      .then(value => setResult({ state: 'ready', value, error: '' }))
      .catch((error: Error) => { if (error.name !== 'AbortError') setResult({ state: 'error', value: null, error: error.message }); });
    return () => controller.abort();
  }, [path, revision, attempt]);
  return { ...result, reload: useCallback(() => setAttempt(value => value + 1), []) };
}

/**
 * Face Manager's 3D `.fsz` inspector: the stored FaceSet's views on the angular coverage radar, and its identity asset as a textured 3D head
 * (the stitched UV albedo mapped onto the Canonical or MakeHuman CC0 head). Reading never builds anything; building is an explicit button.
 */
export default function FacesetInspector3D({ sources }: { sources: Pick<FaceSet, 'faceset_id' | 'name'>[] }) {
  const [chosen, setChosen] = useState('');
  const id = sources.some(item => item.faceset_id === chosen) ? chosen : sources[0]?.faceset_id ?? '';
  const [template, setTemplate] = useState<HeadTemplate>(readTemplate);
  const [revision, setRevision] = useState(0);
  const [build, setBuild] = useState<{ busy: boolean; error: string }>({ busy: false, error: '' });
  const live = useRef(id); live.current = id;
  const query = template === 'makehuman' ? '?template=makehuman' : '';
  useEffect(() => setBuild({ busy: false, error: '' }), [id, template]);

  const views = useFetched<FaceSetView[]>(id ? `/facesets/${encodeURIComponent(id)}/views` : null);
  const status = useFetched<IdentityAssetStatus>(id ? `/facesets/${encodeURIComponent(id)}/identity-asset${query}` : null, revision);
  const built = status.value?.state === 'built';
  const meshState = useFetched<Reconstruction>(built && id ? `/facesets/${encodeURIComponent(id)}/identity-asset/mesh${query}` : null, revision);
  // Stored view pitch is legacy (+ = down); the radar is in the standard frame (+ = up).
  const coverage = useMemo(() => ({ views: (views.value ?? []).map(view => ({ yaw: view.yaw, pitch: -view.pitch })) }), [views.value]);

  const runBuild = async () => {
    const forId = id;
    setBuild({ busy: true, error: '' });
    try {
      await post<IdentityAssetStatus>(`/facesets/${encodeURIComponent(forId)}/identity-asset${query}`, {});
      if (live.current === forId) { setBuild({ busy: false, error: '' }); setRevision(value => value + 1); }
    } catch (error) { if (live.current === forId) setBuild({ busy: false, error: (error as Error).message }); }
  };
  const pick = (value: HeadTemplate) => { writeTemplate(value); setTemplate(value); };

  if (!sources.length) return <p className="muted" data-testid="fsz-inspector-empty">Import or build a FaceSet to inspect it in 3D.</p>;
  return (
    <section className="fsz-inspector-3d" aria-label="3D FaceSet inspector" data-testid="fsz-inspector-3d">
      <div className="fsz-inspector-head">
        <label>Inspect .fsz in 3D
          <select aria-label="FaceSet to inspect in 3D" value={id} onChange={event => setChosen(event.target.value)}>
            {sources.map(item => <option key={item.faceset_id} value={item.faceset_id}>{item.name}</option>)}
          </select>
        </label>
        <div className="head-template" role="radiogroup" aria-label="Head model">
          {([['canonical', 'Canonical head'], ['makehuman', 'MakeHuman CC0 head']] as const).map(([value, label]) => (
            <button key={value} type="button" role="radio" aria-checked={template === value} className={`head-template-btn ${template === value ? 'active' : ''}`} onClick={() => pick(value)}>{label}</button>
          ))}
        </div>
      </div>
      <div className="fsz-inspector-body">
        <div className="fsz-inspector-viewer">
          {built && meshState.value
            ? <Suspense fallback={<div className="skeleton-tile wide" aria-label="Loading 3D viewer" />}><PoseGeometryViewer coverage={null} track={null} mesh={meshState.value} /></Suspense>
            : <div className="fsz-inspector-placeholder" role="status">
                {status.state === 'loading' || meshState.state === 'loading' ? 'Loading…'
                  : status.state === 'error' ? <>{status.error} <button type="button" className="button small" onClick={status.reload}>Retry</button></>
                  : meshState.state === 'error' ? <>{meshState.error} <button type="button" className="button small" onClick={meshState.reload}>Retry</button></>
                  : <>No 3D {template === 'makehuman' ? 'MakeHuman' : 'canonical'} head for this FaceSet yet.{' '}
                      <button type="button" className="button small accent" disabled={build.busy} onClick={() => void runBuild()}>{build.busy ? 'Building…' : 'Build 3D head'}</button></>}
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
