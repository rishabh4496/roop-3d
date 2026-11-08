import { useId, useMemo } from 'react';
import {
  CELL_COUNT, YAW_BINS, cellCounts, cellPitchRange, cellPolygon, cellYawRange, inRange, polarPoint, summarizeRadar,
  type CoverageViews, type PoseDeg,
} from '../../utils/coverageRadar';

export interface AngularCoverageRadarProps {
  /** The FaceSet coverage payload (`views` / `observed_pose_points`); null while none is loaded. */
  coverage: CoverageViews | null;
  /** Standard-frame poses of the target the FaceSet will be swapped onto (red when the FaceSet cannot serve them). */
  targetPoses?: PoseDeg[];
  /** The pose of the face currently selected / under the playhead. */
  livePose?: PoseDeg | null;
  /** Name shown in the accessible label. */
  name?: string;
  compact?: boolean;
}

const R = 1.0, PAD = 0.2;
const path = (points: { x: number; y: number }[]) => `M${points.map(point => `${point.x.toFixed(4)},${point.y.toFixed(4)}`).join('L')}Z`;
const signed = (value: number) => `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value)}`;
const NO_POSES: PoseDeg[] = [];

/**
 * Angular Coverage Radar: a polar heat-map of the yaw (-90°..+90°) x pitch (-60°..+60°) grid. The centre is a frontal head;
 * distance from the centre is how far the head is turned or tilted; cells are filled by how many stored views fall in them.
 * Empty profile cells are outlined in the danger colour and listed, and target poses the FaceSet cannot serve are marked, so a
 * missing angle is visible before a render starts.
 */
export default function AngularCoverageRadar({ coverage, targetPoses = NO_POSES, livePose = null, name, compact = false }: AngularCoverageRadarProps) {
  const id = useId();
  const counts = useMemo(() => cellCounts(coverage), [coverage]);
  const summary = useMemo(() => summarizeRadar(counts, targetPoses), [counts, targetPoses]);
  const polygons = useMemo(() => Array.from({ length: CELL_COUNT }, (_, index) => ({ index, column: index % YAW_BINS, row: Math.floor(index / YAW_BINS) }))
    .map(cell => ({ ...cell, d: path(cellPolygon(cell.column, cell.row)) })), []);
  const maxCount = Math.max(1, ...counts);
  const profileGap = (column: number) => summary.profiles.some(band => (band.id === 'left_profile' ? column < 2 : column >= YAW_BINS - 2));
  const uncovered = new Set(summary.uncoveredTargets.map(item => item.pose));
  const label = `Angular coverage radar${name ? ` for ${name}` : ''}: ${summary.filled} of ${summary.total} yaw and pitch cells have views.${summary.profiles.length ? ` Missing ${summary.profiles.map(band => band.label.toLowerCase()).join(' and ')}.` : ''}`;
  const live = livePose && inRange(livePose) ? polarPoint(livePose.yaw, livePose.pitch) : null;

  return (
    <section className={`coverage-radar ${compact ? 'compact' : ''}`} aria-label="Angular coverage radar" data-testid="coverage-radar">
      <div className="coverage-radar-head">
        {!compact && <strong>Angular coverage</strong>}
        <span className={`tiny-badge radar-badge ${summary.ready ? 'ok' : 'warn'}`}>{summary.samples ? `${summary.percent}% · ${summary.filled}/${summary.total}` : 'no views'}</span>
      </div>
      <svg className="coverage-radar-svg" viewBox={`${-R - PAD} ${-R - PAD} ${2 * (R + PAD)} ${2 * (R + PAD)}`} role="img" aria-label={label} aria-describedby={`${id}-summary`}>
        <defs>
          <pattern id={`${id}-hatch`} width="0.07" height="0.07" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="0.07" stroke="var(--danger)" strokeWidth="0.018" opacity="0.7" /></pattern>
        </defs>
        <circle r={R} className="radar-disc" />
        {polygons.map(({ index, column, row, d }) => {
          const count = counts[index], [y0, y1] = cellYawRange(column), [p0, p1] = cellPitchRange(row);
          const gap = count === 0 && profileGap(column);
          return (
            <path key={index} d={d} className={`radar-cell ${count ? 'filled' : 'empty'} ${gap ? 'profile-gap' : ''}`}
              style={count ? { fillOpacity: 0.28 + 0.62 * (count / maxCount) } : undefined} fill={gap ? `url(#${id}-hatch)` : undefined}
              data-count={count} data-cell={index}>
              <title>{`Yaw ${signed(y0)}…${signed(y1)}°, pitch ${signed(p0)}…${signed(p1)}°: ${count} view${count === 1 ? '' : 's'}${gap ? ' · missing profile angle' : ''}`}</title>
            </path>
          );
        })}
        {[0.5, 1].map(radius => <circle key={radius} r={radius * R} className="radar-ring" />)}
        <line x1={-R} y1="0" x2={R} y2="0" className="radar-spoke" /><line x1="0" y1={-R} x2="0" y2={R} className="radar-spoke" />
        <text x={R + 0.04} y="0.04" className="radar-label">+90°</text>
        <text x={-R - 0.04} y="0.04" className="radar-label" textAnchor="end">−90°</text>
        <text x="0" y={-R - 0.05} className="radar-label" textAnchor="middle">pitch +60°</text>
        <text x="0" y={R + 0.12} className="radar-label" textAnchor="middle">pitch −60°</text>
        {targetPoses.filter(inRange).map((pose, index) => {
          const point = polarPoint(pose.yaw, pose.pitch), bad = uncovered.has(pose);
          return <circle key={index} cx={point.x} cy={point.y} r={0.03} className={`radar-target ${bad ? 'uncovered' : ''}`}><title>{`Target ${signed(Math.round(pose.yaw))}° yaw, ${signed(Math.round(pose.pitch))}° pitch${bad ? ' · no nearby FaceSet view' : ''}`}</title></circle>;
        })}
        {live && <g className="radar-live" transform={`translate(${live.x} ${live.y})`}><circle r={0.06} /><line x1="-0.1" x2="0.1" y1="0" y2="0" /><line y1="-0.1" y2="0.1" x1="0" x2="0" /></g>}
      </svg>
      <p id={`${id}-summary`} className={`coverage-radar-note ${summary.ready ? 'ok' : 'warn'}`} role="status">
        {summary.samples === 0 ? 'Load a FaceSet to see which head angles it covers.' : summary.ready ? 'All profile angles are covered and the target poses have a nearby view.' : summary.message}
      </p>
      {!compact && summary.samples > 0 && (
        <ul className="coverage-gap-list" aria-label="Coverage gaps">
          {summary.gaps.filter(gap => gap.state !== 'ok').map(gap => (
            <li key={gap.id} className={`gap-${gap.state} ${gap.yaw.profile ? 'profile' : ''}`}>{gap.label}<small>{gap.state === 'missing' ? 'missing' : `${gap.covered}/${gap.total} cells`}</small></li>
          ))}
        </ul>
      )}
      <div className="coverage-radar-legend" aria-hidden="true">
        <span><i className="swatch filled" />Views</span><span><i className="swatch empty" />Empty</span><span><i className="swatch gap" />Missing profile</span>
        {targetPoses.length > 0 && <span><i className="swatch target" />Target pose</span>}
      </div>
    </section>
  );
}
