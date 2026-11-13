import { useMemo } from 'react';
import { buildGraphGeometry, useTrackingEnded, useTrackingHistory, useTrackingLatest, type TrackingSample } from '../../stores/trackingStream';
import './trackingTelemetry.css';

const GRAPH_W = 240;
const GRAPH_H = 56;

const deg = (value: number | null) => (value === null ? 'n/a' : `${value >= 0 ? '+' : ''}${value.toFixed(1)}°`);
const fixed = (value: number | null, digits: number, unit = '') => (value === null ? 'n/a' : `${value.toFixed(digits)}${unit}`);

const SOURCE_LABEL: Record<string, string> = {
  pnp_dense: 'Dense-landmark PnP',
  pnp_5pt: '5-point PnP',
  geometric_5pt: '5-point geometric estimate',
  weak_perspective: 'Weak-perspective mesh fit',
};

function position(sample: TrackingSample): { value: string; title: string } {
  if (sample.translationM) {
    const [x, y, z] = sample.translationM;
    return { value: `${x.toFixed(3)}, ${y.toFixed(3)}, ${z.toFixed(3)} m`, title: 'Metric head translation t in the camera frame (x right, y down, z away).' };
  }
  if (sample.translationPx) {
    const [x, y] = sample.translationPx;
    const scale = sample.scalePxPerM === null ? '' : ` · ${sample.scalePxPerM.toFixed(0)} px/m`;
    return { value: `${x.toFixed(0)}, ${y.toFixed(0)} px${scale}`, title: 'Head centre in the image. Depth is encoded in the scale, it is not a metric distance for this pose source.' };
  }
  return { value: 'n/a', title: 'This pose source did not report a translation.' };
}

/** Why the swap is or is not following the head right now, in words the backend's `reason` code stands behind. */
function statusLine(sample: TrackingSample): string {
  if (sample.status === 'tracked') return 'Following the head';
  if (sample.status === 'coasting') return 'Predicting the pose (no fresh measurement)';
  return 'Head not followed';
}

function Badge({ label, value, title, testId }: { label: string; value: string; title?: string; testId: string }) {
  return (
    <li className="trk-badge" title={title} data-testid={testId}>
      <span className="trk-badge-label">{label}</span>
      <b className="trk-badge-value">{value}</b>
    </li>
  );
}

/**
 * Live head pose and temporal stability from the server's `tracking.telemetry` frames: yaw/pitch/roll and the 6DoF pose (rotation as a
 * quaternion, translation, where the pose came from, reprojection error, which reference-angle bin it falls in), then a graph of
 * landmark jitter and a 0..1 stability score with shot cuts and camera flashes marked where the filters were flushed.
 * Every number the backend did not measure reads "n/a"; nothing is drawn as zero.
 */
export default function TrackingTelemetryPanel() {
  const latest = useTrackingLatest();
  const history = useTrackingHistory();
  const ended = useTrackingEnded();
  const graph = useMemo(() => buildGraphGeometry(history, GRAPH_W, GRAPH_H), [history]);

  if (!latest) {
    return (
      <section className="trk-panel" aria-label="Head pose and temporal stability" data-testid="tracking-telemetry" data-state="idle">
        <span className="hud-label">HEAD POSE · 6DoF</span>
        <p className="trk-empty muted" role="status">No tracking frames yet. They stream while a 3D render or preview is running.</p>
      </section>
    );
  }
  const where = position(latest);
  const source = latest.poseSource ? SOURCE_LABEL[latest.poseSource] ?? latest.poseSource : null;
  const quat = latest.quat ? latest.quat.map(value => value.toFixed(3)).join(', ') : 'n/a';
  const stabilityPct = latest.stability === null ? 'n/a' : `${Math.round(latest.stability * 100)}%`;
  return (
    <section className="trk-panel" aria-label="Head pose and temporal stability" data-testid="tracking-telemetry" data-state={ended ? 'ended' : latest.status}>
      <header className="trk-head">
        <span className="hud-label">HEAD POSE · 6DoF</span>
        <span className={`trk-status ${latest.status}`} data-testid="trk-status" title={latest.reason}>
          {ended ? 'Run ended' : statusLine(latest)}
        </span>
      </header>
      <ul className="trk-badges" aria-label="Pose">
        <Badge testId="trk-yaw" label="Yaw" value={deg(latest.yawDeg)} title="Left/right head turn. ±90° is a full profile." />
        <Badge testId="trk-pitch" label="Pitch" value={deg(latest.pitchDeg)} title="Positive looks up." />
        <Badge testId="trk-roll" label="Roll" value={deg(latest.rollDeg)} title="Head tilt toward a shoulder." />
        <Badge testId="trk-quat" label="q (w,x,y,z)" value={quat} title="Unit rotation quaternion (w ≥ 0). The temporal filter slerps in this space." />
        <Badge testId="trk-position" label="t" value={where.value} title={where.title} />
        <Badge testId="trk-source" label="Source" value={source ?? 'n/a'} title="How the pose was solved." />
        <Badge testId="trk-bin" label="Ref. bin" value={latest.sphericalBin ?? 'n/a'} title="The reference-angle bin of the FaceSet this pose draws on." />
        <Badge testId="trk-reproj" label="Reproj." value={fixed(latest.reprojRmsPx, 2, ' px')} title="RMS landmark reprojection error of the pose fit." />
        <Badge testId="trk-occluded" label="Self-occluded" value={latest.selfOccluded === null ? 'n/a' : `${Math.round(latest.selfOccluded * 100)}%`}
          title="Share of the face mesh hidden by the head itself from this camera." />
      </ul>
      {latest.status !== 'tracked' && latest.reason && <p className="trk-reason" role="status" data-testid="trk-reason">{latest.reason}</p>}
      <figure className="trk-graph" data-testid="trk-graph">
        <svg viewBox={`0 0 ${GRAPH_W} ${GRAPH_H}`} role="img" preserveAspectRatio="none"
          aria-label={`Temporal stability over the last ${history.length} frames. Stability ${stabilityPct}, jitter ${fixed(latest.jitterPx, 2, ' px')}. ${latest.shotCuts} shot cuts.`}>
          {graph.cuts.map(x => <line key={`c${x}`} className="trk-mark cut" x1={x} x2={x} y1={0} y2={GRAPH_H} data-testid="trk-cut-mark" />)}
          {graph.flashes.map(x => <line key={`f${x}`} className="trk-mark flash" x1={x} x2={x} y1={0} y2={GRAPH_H} data-testid="trk-flash-mark" />)}
          {graph.stability.split(' | ').filter(Boolean).map((points, index) => <polyline key={`s${index}`} className="trk-line stability" points={points} fill="none" />)}
          {graph.jitter.split(' | ').filter(Boolean).map((points, index) => <polyline key={`j${index}`} className="trk-line jitter" points={points} fill="none" />)}
        </svg>
        <figcaption>
          <span className="trk-key stability">Stability {stabilityPct}</span>
          <span className="trk-key jitter">Jitter {fixed(latest.jitterPx, 2, ' px')} <small>(scale 0–{graph.jitterMax.toFixed(1)} px)</small></span>
          <span className="trk-key">Filter cutoff {fixed(latest.cutoffHz, 2, ' Hz')}</span>
          <span className="trk-key">Mask flow lock {latest.maskFlow === null ? 'n/a' : `${Math.round(latest.maskFlow * 100)}%`}</span>
          <span className="trk-key" data-testid="trk-cuts">Shot cuts {latest.shotCuts}{latest.flash ? ' · flash' : ''}</span>
        </figcaption>
      </figure>
    </section>
  );
}
