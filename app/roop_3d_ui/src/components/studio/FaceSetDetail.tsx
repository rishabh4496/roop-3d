import { useCallback, useEffect, useRef, useState } from 'react';
import { api, post } from '../../api';
import type { FaceSet, FaceSetView, IdentityAssetStatus, Reconstruction } from '../../contracts';
import VirtualList from '../common/VirtualList';
import { startFacesetDrag } from '../../utils/facesetDrag';
import { describeCoverage, parseRequest } from '../guide/coverageGuide';
import { READINESS, SOURCE_PHOTO_RULES } from '../guide/helpContent';

/** Yaw columns (image left → right) and pitch rows used by the pose coverage indicator. Standard frame: + pitch = up. */
export const YAW_BINS = [
  { id: 'far_left', label: '−75°', min: -90, max: -60 }, { id: 'left', label: '−45°', min: -60, max: -25 },
  { id: 'front', label: '0°', min: -25, max: 25 }, { id: 'right', label: '+45°', min: 25, max: 60 }, { id: 'far_right', label: '+75°', min: 60, max: 91 },
] as const;
export const PITCH_BINS = [{ id: 'up', label: 'Up', min: 10, max: 91 }, { id: 'level', label: 'Level', min: -10, max: 10 }, { id: 'down', label: 'Down', min: -91, max: -10 }] as const;

/** `/facesets/{id}/views` returns the stored (legacy, + = looking down) pitch; the indicator shows the standard frame. */
export function coverageGrid(views: FaceSetView[]) {
  const grid = PITCH_BINS.map(() => YAW_BINS.map(() => 0));
  for (const view of views) {
    const pitch = -view.pitch;
    const column = YAW_BINS.findIndex(bin => view.yaw >= bin.min && view.yaw < bin.max);
    const row = PITCH_BINS.findIndex(bin => pitch >= bin.min && pitch < bin.max);
    if (column >= 0 && row >= 0) grid[row][column] += 1;
  }
  const covered = grid.flat().filter(count => count > 0).length;
  return { grid, covered, total: YAW_BINS.length * PITCH_BINS.length };
}

/** Response shape checks: a malformed answer becomes a visible error with Retry, never a crash. */
export const isViewList = (value: unknown): boolean => Array.isArray(value) && value.every(item => item && typeof item === 'object' && Number.isFinite((item as FaceSetView).index) && Number.isFinite((item as FaceSetView).yaw) && Number.isFinite((item as FaceSetView).pitch));
export const isAssetStatus = (value: unknown): boolean => {
  const item = value as IdentityAssetStatus | null;
  return !!item && typeof item === 'object' && (item.state === 'built' || item.state === 'not_built') && !!item.coverage && typeof item.coverage.readiness === 'string'
    && Array.isArray(item.coverage.requests) && (item.state === 'not_built' || (!!item.asset && Number.isFinite(item.asset.vertex_count)));
};

/** Which head the 3D identity asset is built on. `canonical` is what pose_3d / reenactment_3d renders reuse; `makehuman` is the open CC0 head+neck template (inspection / export). */
export type HeadTemplate = 'canonical' | 'makehuman';
const TEMPLATE_KEY = 'roop3d-head-template';
export const readTemplate = (): HeadTemplate => { try { return localStorage.getItem(TEMPLATE_KEY) === 'makehuman' ? 'makehuman' : 'canonical'; } catch { return 'canonical'; } };
export const writeTemplate = (value: HeadTemplate) => { try { localStorage.setItem(TEMPLATE_KEY, value); } catch { /* the choice just is not remembered */ } };
type Load<T> = { state: 'idle' | 'loading' | 'ready' | 'error'; value: T | null; error: string };
const IDLE = { state: 'idle', value: null, error: '' } as const;

/** Fetch `path` whenever it changes; aborts the previous request so a slow answer for an old FaceSet never lands. */
function useResource<T>(path: string | null, valid: (value: unknown) => boolean, revision = 0): [Load<T>, () => void] {
  const [result, setResult] = useState<Load<T>>(IDLE);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!path) { setResult(IDLE); return undefined; }
    const controller = new AbortController();
    let live = true;   // abort covers fetch; this also drops any late answer for a FaceSet that is no longer shown
    setResult({ state: 'loading', value: null, error: '' });
    api<T>(path, { signal: controller.signal })
      .then(value => { if (live) setResult(valid(value) ? { state: 'ready', value, error: '' } : { state: 'error', value: null, error: 'The local service returned an unexpected response.' }); })
      .catch((error: Error) => { if (live && error.name !== 'AbortError') setResult({ state: 'error', value: null, error: error.message }); });
    return () => { live = false; controller.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `valid` is a pure shape check
  }, [path, retry, revision]);
  return [result, useCallback(() => setRetry(value => value + 1), [])];
}

function Skeleton({ rows = 3 }: { rows?: number }) {
  return <div className="skeleton-stack" aria-hidden="true">{Array.from({ length: rows }, (_, i) => <span key={i} className="skeleton-line" style={{ width: `${92 - i * 14}%` }} />)}</div>;
}

export function ErrorRetry({ message, onRetry }: { message: string; onRetry: () => void }) {
  return <div className="error-retry" role="status" aria-live="polite"><span>{message}</span><button type="button" className="button small" onClick={onRetry}>Retry</button></div>;
}

export interface FaceSetDetailProps {
  faceset: FaceSet;
  /** Label of the selected target track, or null when none is selected (then assignment is disabled with a reason). */
  selectedTrackLabel: string | null;
  assignedHere: boolean;
  onAssign: () => void;
  sources: FaceSet[];
  identitySourceB: string;
  onIdentitySourceB: (id: string) => void;
  /** Show the built identity mesh in the inspector's 3D viewer (null hides it). */
  onShowMesh: (mesh: Reconstruction | null) => void;
  meshShown: boolean;
}

export default function FaceSetDetail({ faceset, selectedTrackLabel, assignedHere, onAssign, sources, identitySourceB, onIdentitySourceB, onShowMesh, meshShown }: FaceSetDetailProps) {
  const id = faceset.faceset_id;
  const [views, retryViews] = useResource<FaceSetView[]>(`/facesets/${id}/views`, isViewList);
  const [assetRevision, setAssetRevision] = useState(0);
  const [template, setTemplateState] = useState<HeadTemplate>(readTemplate);
  const query = template === 'makehuman' ? '?template=makehuman' : '';
  const [status, retryStatus] = useResource<IdentityAssetStatus>(`/facesets/${id}/identity-asset${query}`, isAssetStatus, assetRevision);
  const [build, setBuild] = useState<{ busy: boolean; started: number; error: string }>({ busy: false, started: 0, error: '' });
  const [elapsed, setElapsed] = useState(0);
  const [meshBusy, setMeshBusy] = useState(false);
  const live = useRef(id);
  live.current = id;
  useEffect(() => { setBuild({ busy: false, started: 0, error: '' }); setMeshBusy(false); }, [id]);
  useEffect(() => {
    if (!build.busy) return undefined;
    const timer = setInterval(() => setElapsed(Math.round((Date.now() - build.started) / 1000)), 500);
    return () => clearInterval(timer);
  }, [build.busy, build.started]);

  const runBuild = async () => {
    const forId = id;
    setElapsed(0); setBuild({ busy: true, started: Date.now(), error: '' });
    try {
      await post<IdentityAssetStatus>(`/facesets/${forId}/identity-asset${query}`, {});
      if (live.current !== forId) return;
      setBuild({ busy: false, started: 0, error: '' }); setAssetRevision(value => value + 1);
    } catch (error) {
      if (live.current === forId) setBuild({ busy: false, started: 0, error: (error as Error).message });
    }
  };
  const showMesh = async () => {
    if (meshShown) { onShowMesh(null); return; }
    const forId = id; setMeshBusy(true);
    try {
      const mesh = await api<Reconstruction>(`/facesets/${forId}/identity-asset/mesh${query}`);
      if (live.current === forId) onShowMesh(mesh);
    } catch (error) {
      if (live.current === forId) setBuild(value => ({ ...value, error: (error as Error).message }));
    } finally { if (live.current === forId) setMeshBusy(false); }
  };

  const list = views.value ?? [];
  const coverage = coverageGrid(list);
  const asset = status.value?.asset ?? null;
  const readiness = status.value?.coverage.readiness;
  const reading = describeCoverage(coverage.grid);
  const readinessHelp = readiness ? READINESS[readiness] : undefined;

  return (
    <section className="fs-detail" aria-label={`FaceSet ${faceset.name}`}>
      <div className="fs-detail-head" draggable onDragStart={event => startFacesetDrag(event, id, faceset.name)} title="Drag onto a target track to assign">
        <img className="fs-detail-thumb" src={`/api/v1/facesets/${id}/thumb`} alt={`${faceset.name} source identity preview`} />
        <div className="fs-detail-meta">
          <strong>{faceset.name}</strong>
          <small>{faceset.item_count} stored view{faceset.item_count === 1 ? '' : 's'} · local</small>
          <button type="button" className={`button small ${assignedHere ? '' : 'accent'}`} disabled={!selectedTrackLabel || assignedHere} onClick={onAssign}
            title={selectedTrackLabel ? `Swap ${selectedTrackLabel} with ${faceset.name}` : 'Select a target track first (click a face in the video or a track in the Target panel)'}>
            {assignedHere ? `Assigned to ${selectedTrackLabel}` : selectedTrackLabel ? `Assign to ${selectedTrackLabel}` : 'Select a target track to assign'}
          </button>
        </div>
      </div>

      <div className="fs-subhead"><span>POSE COVERAGE</span><span className="muted">{views.state === 'ready' ? `${coverage.covered}/${coverage.total} cells` : ''}</span></div>
      {views.state === 'loading' && <Skeleton rows={2} />}
      {views.state === 'error' && <ErrorRetry message={views.error} onRetry={retryViews} />}
      {views.state === 'ready' && (
        <div className="coverage-grid" role="table" aria-label="Pose coverage, yaw by pitch">
          <div role="row" className="coverage-row header"><span role="columnheader" />{YAW_BINS.map(bin => <span role="columnheader" key={bin.id}>{bin.label}</span>)}</div>
          {PITCH_BINS.map((pitch, row) => (
            <div role="row" className="coverage-row" key={pitch.id}>
              <span role="rowheader">{pitch.label}</span>
              {YAW_BINS.map((yaw, column) => {
                const count = coverage.grid[row][column];
                return <span role="cell" key={yaw.id} className={`coverage-cell ${count === 0 ? 'empty' : count < 2 ? 'sparse' : 'dense'}`} title={`${pitch.label} · yaw ${yaw.label}: ${count} view${count === 1 ? '' : 's'}`} aria-label={`${pitch.label}, yaw ${yaw.label}: ${count === 0 ? 'no views' : `${count} view${count === 1 ? '' : 's'}`}`}>{count || '–'}</span>;
              })}
            </div>
          ))}
        </div>
      )}
      {views.state === 'ready' && <div className="coverage-reading" data-testid="coverage-reading">
        <p role="status">{reading.text}</p>
        <p className="muted coverage-legend">– no photo at this angle · <b>1</b> one photo (sparse) · <b>2+</b> well covered. Left and right are as seen in the picture.</p>
        <details className="source-checklist"><summary>Source photo checklist</summary>
          <ul>{SOURCE_PHOTO_RULES.map(rule => <li key={rule.id}><strong>{rule.rule}.</strong> {rule.detail}</li>)}</ul></details>
      </div>}

      <div className="fs-subhead"><span>MULTI-ANGLE VIEWS</span><span className="muted">{views.state === 'ready' ? list.length : ''}</span></div>
      {views.state === 'loading' && <div className="view-gallery-skeleton" aria-hidden="true">{Array.from({ length: 6 }, (_, i) => <span key={i} className="skeleton-tile" />)}</div>}
      {views.state === 'ready' && (list.length
        ? <VirtualList items={list} itemKey={view => view.index} estimateHeight={92} minColumnWidth={70} gap={6} maxHeight={220} aria-label={`${faceset.name} views`} className="view-gallery" renderItem={view => (
            <figure className="view-tile">
              <img loading="lazy" decoding="async" src={`/api/v1/facesets/${id}/views/${view.index}/image?size=128`} alt={`View ${view.index + 1}: yaw ${view.yaw.toFixed(0)}°, pitch ${(-view.pitch).toFixed(0)}°`} />
              <figcaption>{view.yaw >= 0 ? '+' : ''}{view.yaw.toFixed(0)}° / {(-view.pitch) >= 0 ? '+' : ''}{(-view.pitch).toFixed(0)}°</figcaption>
            </figure>)} />
        : <p className="muted fs-empty">This FaceSet has no readable views.</p>)}

      <div className="fs-subhead"><span>3D IDENTITY ASSET</span>{asset && <span className="status-chip ok">BUILT</span>}{status.value?.state === 'not_built' && <span className="status-chip">NOT BUILT</span>}</div>
      <div className="head-template" role="radiogroup" aria-label="Head model">
        {([['canonical', 'Canonical head', 'The built-in template that 3D render modes reuse'], ['makehuman', 'MakeHuman CC0 head', 'Open CC0 head + neck, textured with this .fsz\'s stitched UV albedo. For inspection and export.']] as const).map(([value, label, hint]) => (
          <button key={value} type="button" role="radio" aria-checked={template === value} className={`head-template-btn ${template === value ? 'active' : ''}`} title={hint}
            onClick={() => { if (template === value) return; writeTemplate(value); setTemplateState(value); if (meshShown) onShowMesh(null); setBuild({ busy: false, started: 0, error: '' }); }}>{label}</button>
        ))}
      </div>
      {status.state === 'loading' && <Skeleton rows={3} />}
      {status.state === 'error' && <ErrorRetry message={status.error} onRetry={retryStatus} />}
      {status.value && (
        <div className="asset-card">
          <p className="asset-readiness">Coverage: <strong>{readinessHelp?.label ?? readiness?.replaceAll('_', ' ')}</strong> · {status.value.coverage.usable_count}/{status.value.coverage.view_count} usable views
            {status.value.coverage.yaw_range_deg && <> · yaw {status.value.coverage.yaw_range_deg[0].toFixed(0)}° … {status.value.coverage.yaw_range_deg[1].toFixed(0)}°</>}</p>
          {asset ? (
            <dl className="asset-facts">
              <div><dt>Level</dt><dd>{asset.reconstruction_level}</dd></div>
              <div><dt>Landmarks</dt><dd>{asset.landmark_layout}</dd></div>
              <div><dt>Mesh</dt><dd>{asset.vertex_count.toLocaleString()} v · {asset.face_count.toLocaleString()} f</dd></div>
              <div><dt>Views used</dt><dd>{asset.sources_used}/{asset.sources_total}</dd></div>
            </dl>
          ) : <p className="muted">Not built yet. Building runs locally on the CPU from this FaceSet's own views; 3D render modes build it on first use otherwise.</p>}
          {readinessHelp && <p className="muted asset-meaning">{readinessHelp.meaning} <em>{readinessHelp.next}</em></p>}
          {status.value.coverage.requests.length > 0 && <details className="asset-requests" open={readiness !== 'full_coverage'}><summary>{status.value.coverage.requests.length} suggested additional angle{status.value.coverage.requests.length === 1 ? '' : 's'}</summary><ul>{status.value.coverage.requests.map(item => { const request = parseRequest(item); return <li key={item}>{request.level !== 'info' && <span className={`status-chip request-${request.level}`}>{request.level}</span>} {request.text}</li>; })}</ul></details>}
          {asset && asset.missing_information.length > 0 && <details className="asset-requests"><summary>What this asset does not contain</summary><ul>{asset.missing_information.map(item => <li key={item}>{item}</li>)}{asset.limitations.map(item => <li key={item}>{item}</li>)}</ul></details>}
          <div className="asset-actions">
            {!asset && <button type="button" className="button small accent" disabled={build.busy} onClick={() => void runBuild()}>{build.busy ? `Building… ${elapsed}s` : 'Build 3D identity asset'}</button>}
            {asset && <button type="button" className="button small" aria-pressed={meshShown} disabled={meshBusy} onClick={() => void showMesh()}>{meshBusy ? 'Loading mesh…' : meshShown ? 'Hide 3D asset' : 'View 3D asset'}</button>}
            {build.busy && <span className="build-progress" role="status">Reconstructing locally (fit, texture, validation). This usually takes seconds.</span>}
          </div>
          {build.error && <ErrorRetry message={build.error} onRetry={() => { setBuild(value => ({ ...value, error: '' })); retryStatus(); }} />}
        </div>
      )}

      <label className="fs-blend">Identity B (animated Slerp)
        <select aria-label="Identity B for animated Slerp" value={identitySourceB} onChange={event => onIdentitySourceB(event.target.value)}>
          <option value="">No identity morph</option>
          {sources.filter(source => source.faceset_id !== id).map(source => <option key={source.faceset_id} value={source.faceset_id}>{source.name}</option>)}
        </select>
      </label>
    </section>
  );
}
