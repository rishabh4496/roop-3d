import { Fragment } from 'react';
import { angleCropURL, type AngleBucket, type TargetLock } from '../../api';
import { ANGLE_CELLS, PITCH_ROWS, YAW_COLUMNS, angleLabel, faceColor } from '../../utils/faceIdentity';

export type HudState = 'capturing' | 'complete' | 'partial' | 'idle' | 'failed';

/** What the capture HUD says, derived only from the server's coverage and scan fields (nothing is rounded up). */
export function hudSummary(lock: TargetLock): { state: HudState; title: string; detail: string; captured: number; fraction: number } {
  const captured = lock.coverage?.captured ?? 0, total = lock.coverage?.total ?? ANGLE_CELLS;
  const scan = lock.scan ?? { state: 'idle' as const };
  const fraction = total > 0 ? Math.min(1, captured / total) : 0;
  const frames = scan.requested ? `scanned ${Math.min(scan.sampled ?? 0, scan.requested)} / ${scan.requested} frames` : '';
  if (scan.state === 'running') {
    return { state: 'capturing', title: 'Capturing Target Angles…', detail: [`${captured}/${total} angles`, frames].filter(Boolean).join(' · '), captured, fraction };
  }
  if (scan.state === 'error') return { state: 'failed', title: 'Angle scan failed', detail: scan.message ?? 'The background scan stopped with an error.', captured, fraction };
  if (captured >= total) {
    const full = lock.coverage?.quality_locked ?? total;
    return { state: 'complete', title: 'Angles Locked (100% Coverage)',
      detail: full >= total ? `${total}/${total} angles at full quality` : `${total}/${total} angles; ${total - full} still low quality`, captured, fraction };
  }
  if (scan.state === 'done') {
    const reason = scan.exhaustive ? `the whole video was scanned; the other ${total - captured} angles do not occur in it`
      : scan.skipped ? `${scan.skipped} frames could not be read, so absent angles are unconfirmed` : 'the scan stopped early, so absent angles are unconfirmed';
    return { state: 'partial', title: `${captured}/${total} angles locked (${Math.round(fraction * 100)}% coverage)`, detail: reason, captured, fraction };
  }
  return { state: 'idle', title: scan.state === 'skipped' ? 'Angle scan skipped' : 'Angles not captured yet',
    detail: `${captured}/${total} angles from the frames seen so far`, captured, fraction };
}

const cellOf = (lock: TargetLock, yaw: number, pitch: number): AngleBucket | undefined =>
  lock.coverage?.buckets?.find(item => item.bucket[0] === yaw && item.bucket[1] === pitch);
export const cropVersion = (item: AngleBucket) => `${item.frame_index}-${Math.round(item.quality)}`;

/** Circular progress of captured angles; the centre shows the count. */
function Ring({ fraction, count, total, color, done }: { fraction: number; count: number; total: number; color: string; done: boolean }) {
  const radius = 17, circumference = 2 * Math.PI * radius;
  return (
    <svg className={`angle-ring ${done ? 'done' : ''}`} viewBox="0 0 44 44" width="44" height="44" role="img" aria-label={`${count} of ${total} angles captured`}>
      <circle cx="22" cy="22" r={radius} fill="none" className="angle-ring-track" strokeWidth="4" />
      <circle cx="22" cy="22" r={radius} fill="none" stroke={done ? '#34d399' : color} strokeWidth="4" strokeLinecap="round"
        strokeDasharray={circumference} strokeDashoffset={circumference * (1 - fraction)} transform="rotate(-90 22 22)" className="angle-ring-arc" />
      <text x="22" y="26" textAnchor="middle" className="angle-ring-text">{count}</text>
    </svg>
  );
}

/**
 * The live multi-angle capture progress of one locked face: a progress ring, the 5 x 3 angle grid (yaw: L profile … R profile,
 * pitch: up / level / down) whose cells fill with the crops the background harvester keeps as it scans the video, and a
 * Re-Scan button. A cell's image is replaced (and pops in again) whenever a better face for that angle is found.
 */
export default function AngleCaptureHud({ lock, onRescan }: { lock: TargetLock; onRescan?: (lockId: string) => void }) {
  const summary = hudSummary(lock);
  const color = faceColor(lock.face_id);
  const total = lock.coverage?.total ?? ANGLE_CELLS;
  const running = summary.state === 'capturing';
  return (
    <div className={`angle-hud ${summary.state}`} data-testid="angle-hud" data-state={summary.state} style={{ ['--face-color' as string]: color }}>
      <div className="angle-hud-head">
        <Ring fraction={summary.fraction} count={summary.captured} total={total} color={color} done={summary.state === 'complete'} />
        <div className="angle-hud-text" role="status" aria-live="polite">
          <strong className={`angle-hud-title ${summary.state}`}>{summary.state === 'complete' && <span aria-hidden="true">✓ </span>}{summary.title}</strong>
          <small>{summary.detail}</small>
        </div>
      </div>
      <div className="angle-grid" role="grid" aria-label={`Angle coverage of face ${lock.face_id ?? ''}`}>
        <span className="angle-corner" aria-hidden="true" />
        {YAW_COLUMNS.map(column => <span key={column.bin} className="angle-col-label" role="columnheader" aria-label={column.label} title={`${column.label}: head turned toward the ${column.bin < 2 ? 'image left' : column.bin > 2 ? 'image right' : 'camera'}`}>{column.short}</span>)}
        {PITCH_ROWS.map(row => <Fragment key={row.bin}><span className="angle-row-label" role="rowheader">{row.label}</span>{YAW_COLUMNS.map(column => {
          const item = cellOf(lock, column.bin, row.bin);
          const label = angleLabel(column.bin, row.bin);
          return (
            <span key={`${column.bin}-${row.bin}`} role="gridcell" className={`angle-cell ${item ? 'filled' : 'empty'} ${item && !item.quality_locked ? 'weak' : ''}`}
              data-bucket={`${column.bin},${row.bin}`} data-captured={item ? 'true' : 'false'}
              title={item ? `${label}: frame ${item.frame_index}, yaw ${item.yaw.toFixed(0)}°, pitch ${item.pitch.toFixed(0)}°${item.quality_locked ? '' : ' (low quality)'}` : `${label}: not captured yet`}
              aria-label={item ? `${label} captured` : `${label} not captured`}>
              {item?.has_crop ? <img key={cropVersion(item)} src={angleCropURL(lock.lock_id, item.bucket, cropVersion(item))} alt="" loading="lazy" className="angle-crop" />
                : item ? <i aria-hidden="true">●</i> : <i aria-hidden="true" className="angle-empty-dot" />}
            </span>
          );
        })}</Fragment>)}
      </div>
      <div className="angle-hud-foot">
        <button type="button" className="button small" disabled={running || !onRescan} onClick={event => { event.stopPropagation(); onRescan?.(lock.lock_id); }}
          title={running ? 'A scan is already running' : 'Forget the harvested angles and scan the whole video again (the anchor face and angles you added by hand are kept)'}>
          {running ? 'Scanning…' : 'Re-Scan All Angles'}
        </button>
      </div>
    </div>
  );
}
