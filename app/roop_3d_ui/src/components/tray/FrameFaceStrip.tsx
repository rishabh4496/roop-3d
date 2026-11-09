import { useEffect, useRef, useState, type CSSProperties, type FocusEvent } from 'react';
import { createPortal } from 'react-dom';
import type { FrameMeta } from '../../frameScrub';
import { angleCropURL, type TargetLock } from '../../api';
import type { FaceSet } from '../../contracts';
import { cropRect, type MatchedFrameFace } from '../../utils/frameFaces';
import { ANGLE_CELLS, angleLabel, faceColor, faceName } from '../../utils/faceIdentity';
import AngleCaptureHud, { cropVersion } from './AngleCaptureHud';
import './faceBank.css';

const THUMB = 72;
const NO_CROPS: Record<string, string> = {};
const NO_FACES: MatchedFrameFace[] = [];
const NO_LOCKS: TargetLock[] = [];
const NO_SOURCES: FaceSet[] = [];
const NO_ASSIGNMENTS: Record<string, string> = {};

/** Crops each detected face out of the frame on screen. Canvas-less environments (tests) get empty thumbnails. */
function useFaceCrops(imageUrl: string, meta: FrameMeta | null, faces: MatchedFrameFace[]): Record<string, string> {
  const [crops, setCrops] = useState<Record<string, string>>(NO_CROPS);
  useEffect(() => {
    setCrops(NO_CROPS);
    if (!imageUrl || !meta || !faces.length || typeof document === 'undefined') return;
    let active = true;
    const image = new Image();
    image.onload = () => {
      if (!active) return;
      const scale = image.naturalWidth / meta.width, next: Record<string, string> = {};
      try {
        for (const match of faces) {
          const rect = cropRect(match.face.bbox, meta.width, meta.height);
          const canvas = document.createElement('canvas');
          canvas.width = canvas.height = THUMB * 2;     // 2x: crisp on high-DPI screens
          const ctx = canvas.getContext('2d');
          if (!ctx) return;
          ctx.drawImage(image, rect.x * scale, rect.y * scale, rect.w * scale, rect.h * scale, 0, 0, canvas.width, canvas.height);
          next[match.id] = canvas.toDataURL('image/jpeg', 0.85);
        }
        setCrops(next);
      } catch { /* a tainted or unsupported canvas leaves the cards without thumbnails */ }
    };
    image.src = imageUrl;
    return () => { active = false; };
  }, [imageUrl, meta, faces]);
  return crops;
}

const signed = (value: number) => `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(0)}`;

/** Closed padlock (drawn, so it needs no emoji font). */
function Padlock({ locked }: { locked: boolean }) {
  return (
    <svg className={`face-lock-icon ${locked ? 'locked' : ''}`} viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
      <rect x="3" y="7" width="10" height="7" rx="1.6" fill="currentColor" />
      <path d={locked ? 'M5 7V5a3 3 0 0 1 6 0v2' : 'M5 7V5a3 3 0 0 1 5.6-1.5'} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

/** Every crop the harvester kept for a locked face, so the user can check that all of them are the same person. */
function AngleInspector({ lock, anchor }: { lock: TargetLock; anchor: HTMLElement | null }) {
  const buckets = [...(lock.coverage?.buckets ?? [])].filter(item => item.has_crop)
    .sort((a, b) => (b.bucket[1] - a.bucket[1]) || (a.bucket[0] - b.bucket[0]));
  const scores = buckets.filter(item => item.source === 'auto').map(item => item.match_score);
  const rect = anchor?.getBoundingClientRect();
  const width = Math.min(720, (typeof window === 'undefined' ? 720 : window.innerWidth) - 16);
  // Anchored above the card, and never taller than the room above it (the bank sits at the bottom of the screen).
  const style: CSSProperties = rect ? {
    left: Math.max(8, Math.min(rect.left, window.innerWidth - width - 8)), bottom: Math.max(8, window.innerHeight - rect.top + 8), width,
    maxHeight: Math.max(160, rect.top - 16), ['--face-color' as string]: faceColor(lock.face_id),
  } : { ['--face-color' as string]: faceColor(lock.face_id) };
  return createPortal(
    <div className="angle-inspector" role="tooltip" id={`angle-inspector-${lock.lock_id}`} data-testid="angle-inspector" style={style}>
      <div className="angle-inspector-head">
        <strong>{faceName(lock.face_id)} · {buckets.length} captured angle{buckets.length === 1 ? '' : 's'}</strong>
        <small>
          {scores.length
            ? `Every crop was assigned to ${faceName(lock.face_id)} by the identity tracker. Measured similarity to the reference bank: ${Math.min(...scores).toFixed(2)} – ${Math.max(...scores).toFixed(2)} (accepted only above 0.42).`
            : `Only the anchor crop so far; harvested angles appear here and are accepted into ${faceName(lock.face_id)} only above 0.42 similarity.`}
        </small>
      </div>
      <div className="angle-inspector-grid">
        {buckets.map(item => (
          <figure key={`${item.bucket[0]}-${item.bucket[1]}`} className={`angle-inspector-item ${item.source}`}>
            <img src={angleCropURL(lock.lock_id, item.bucket, cropVersion(item))} alt={`${faceName(lock.face_id)} at ${angleLabel(item.bucket[0], item.bucket[1])}`} />
            <figcaption>
              <b>{angleLabel(item.bucket[0], item.bucket[1])}</b>
              <span>frame {item.frame_index} · Y {signed(item.yaw)}° P {signed(item.pitch)}°</span>
              <span>{item.source === 'anchor' ? 'anchor (your click)' : item.source === 'user' ? 'added by you' : `match ${item.match_score.toFixed(2)}`}</span>
            </figcaption>
          </figure>
        ))}
        {!buckets.length && <p className="muted">No crops yet.</p>}
      </div>
    </div>,
    document.body);
}

export interface FrameFaceStripProps {
  imageUrl: string;
  meta: FrameMeta | null;
  /** Null until detection has run for the frame on screen. */
  faces: MatchedFrameFace[] | null;
  detecting: boolean;
  /** Id of the face being locked (its card shows progress). */
  lockingId: string | null;
  /** Click on a card: toggles the face's Target Lock (the parent locks an unlocked face and unlocks a locked one). */
  onLock: (face: MatchedFrameFace) => void;
  disabled?: boolean;
  /** Locked targets of the video; a card shows the capture HUD of the lock whose `face_id` is the face's persistent id. */
  locks?: TargetLock[];
  /** Source identities (.fsz FaceSets) a locked face can be bound to. */
  sources?: FaceSet[];
  /** Source assignments by track / cluster id, as the face selection store keeps them. */
  assignments?: Record<string, string>;
  onAssign?: (trackKey: string, facesetId: string | null, faceId: number | null) => void;
  onUnlock?: (lockId: string) => void;
  onRescan?: (lockId: string) => void;
}

function SourcePicker({ faceId, trackKey, locked, assignedId, sources, onAssign }: {
  faceId: number | null | undefined; trackKey: string | null; locked: boolean; assignedId: string; sources: FaceSet[];
  onAssign?: FrameFaceStripProps['onAssign'];
}) {
  const enabled = locked && !!trackKey && !!onAssign && sources.length > 0;
  const reason = !onAssign ? '' : !locked ? 'Lock this face first, then bind a source identity to it' : !sources.length ? 'Create or import a FaceSet (.fsz) to bind' : '';
  return (
    <label className="face-bank-source" title={reason || `Swap ${faceName(faceId)} with this source identity`}>
      <span>Source</span>
      <select aria-label={`Source identity for ${faceName(faceId)}`} value={assignedId} disabled={!enabled}
        onClick={event => event.stopPropagation()} onChange={event => trackKey && onAssign?.(trackKey, event.target.value || null, faceId ?? null)}>
        <option value="">Keep original</option>
        {sources.map(source => <option key={source.faceset_id} value={source.faceset_id}>{source.name}</option>)}
      </select>
    </label>
  );
}

/** Hover / keyboard-focus inspection of a locked face's captured angles. */
function useInspect() {
  const [open, setOpen] = useState(false);
  const shell = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', close);
    return () => window.removeEventListener('keydown', close);
  }, [open]);
  return { open, shell, bind: {
    onMouseEnter: () => setOpen(true), onMouseLeave: () => setOpen(false),
    onFocus: () => setOpen(true), onBlur: (event: FocusEvent) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false); },
  } };
}

function LockedBankItem({ lock, onUnlock, onRescan, sources, assignedId, onAssign, trackKey, thumb }: {
  lock: TargetLock; onUnlock?: (lockId: string) => void; onRescan?: (lockId: string) => void; sources: FaceSet[]; assignedId: string;
  onAssign?: FrameFaceStripProps['onAssign']; trackKey: string | null; thumb: string | null;
}) {
  const inspect = useInspect();
  const color = faceColor(lock.face_id);
  return (
    <div ref={inspect.shell} className="face-bank-item locked off-frame" data-face-id={lock.face_id} style={{ ['--face-color' as string]: color }} {...inspect.bind}
      aria-describedby={inspect.open ? `angle-inspector-${lock.lock_id}` : undefined}>
      <div className="frame-face-main">
        <span className="frame-face-thumb">{thumb ? <img src={thumb} alt={`${faceName(lock.face_id)} frontal crop`} /> : <i aria-hidden="true">◎</i>}</span>
        <span className="frame-face-meta">
          <span className="face-id-pill">{faceName(lock.face_id)}</span>
          <small>Not in this frame</small>
        </span>
        <button type="button" className="face-lock-toggle locked" aria-pressed="true" aria-label={`Unlock ${faceName(lock.face_id)}`} title="Unlock this target" onClick={() => onUnlock?.(lock.lock_id)}><Padlock locked /></button>
      </div>
      <SourcePicker faceId={lock.face_id} trackKey={trackKey} locked assignedId={assignedId} sources={sources} onAssign={onAssign} />
      <AngleCaptureHud lock={lock} onRescan={onRescan} />
      {inspect.open && <AngleInspector lock={lock} anchor={inspect.shell.current} />}
    </div>
  );
}

function BankItem({ match, crop, lock, busy, disabled, onLock, sources, assignedId, onAssign, onRescan }: {
  match: MatchedFrameFace; crop: string | undefined; lock: TargetLock | undefined; busy: boolean; disabled: boolean;
  onLock: FrameFaceStripProps['onLock']; sources: FaceSet[]; assignedId: string; onAssign?: FrameFaceStripProps['onAssign']; onRescan?: (lockId: string) => void;
}) {
  const inspect = useInspect();
  const pose = match.face.pose3d;
  const faceId = match.face.persistent_face_id;
  const color = faceColor(faceId);
  const name = faceId != null ? `face ${faceId}` : `face ${match.face.face_index}`;
  return (
    <div ref={inspect.shell} className={`face-bank-item ${match.locked ? 'locked' : ''} ${busy ? 'busy' : ''}`} data-face-id={faceId} style={{ ['--face-color' as string]: color }}
      {...(lock ? inspect.bind : {})} aria-describedby={lock && inspect.open ? `angle-inspector-${lock.lock_id}` : undefined}>
      <button type="button" className={`frame-face-card ${match.locked ? 'locked' : ''} ${busy ? 'busy' : ''}`}
        aria-pressed={match.locked} disabled={disabled || busy} data-face-rank={match.face.face_index}
        title={match.locked ? `${name} is a locked target: click to unlock it` : `Lock ${name} as a target (its angles are then captured automatically)`}
        onClick={() => onLock(match)}>
        <span className="frame-face-thumb">{crop ? <img src={crop} alt={`Face ${faceId ?? match.face.face_index} crop`} /> : <i aria-hidden="true">◎</i>}</span>
        <span className="frame-face-meta">
          <strong className="face-id-pill">{name}</strong>
          <span className="frame-face-pose">Y {signed(pose.yaw)}° P {signed(pose.pitch)}°</span>
          {match.face.confidence != null && <small>{Math.round(match.face.confidence * 100)}%</small>}
        </span>
        <span className={`face-lock-toggle ${match.locked ? 'locked' : ''}`} aria-hidden="true"><Padlock locked={match.locked} /></span>
        {match.locked && <span className="frame-face-badge">TARGET LOCKED</span>}
        {busy && <span className="frame-face-badge pending">Locking…</span>}
      </button>
      <SourcePicker faceId={faceId} trackKey={match.clusterId} locked={match.locked} assignedId={assignedId} sources={sources} onAssign={onAssign} />
      {lock && <AngleCaptureHud lock={lock} onRescan={onRescan} />}
      {lock && inspect.open && <AngleInspector lock={lock} anchor={inspect.shell.current} />}
    </div>
  );
}

/**
 * The Target Face Bank, right above the timeline. Each person of the frame on screen is a card with a colour-coded persistent
 * `FACE n` pill (the same colour as its box on the canvas) and a padlock: click to toggle its Target Lock, then bind a source
 * identity to it. A locked face also shows its live angle-capture HUD, and hovering it lists every captured crop. Locked
 * faces that are not in this frame stay in the bank, so their capture progress and source binding are never lost.
 */
export default function FrameFaceStrip({ imageUrl, meta, faces, detecting, lockingId, onLock, disabled = false,
  locks = NO_LOCKS, sources = NO_SOURCES, assignments = NO_ASSIGNMENTS, onAssign, onUnlock, onRescan }: FrameFaceStripProps) {
  const crops = useFaceCrops(imageUrl, meta, faces ?? NO_FACES);
  const status = detecting ? 'Detecting faces…' : faces === null ? '' : faces.length ? `${faces.length} in frame ${meta?.frame_index ?? ''}` : 'No faces in this frame';
  const lockOfFace = (id: number | null | undefined) => (id == null ? undefined : locks.find(item => item.face_id === id));
  const inFrame = new Set((faces ?? NO_FACES).map(match => match.face.persistent_face_id).filter((id): id is number => id != null));
  const offFrame = locks.filter(item => item.face_id != null && !inFrame.has(item.face_id));
  const frontal = (lock: TargetLock) => {
    const cell = lock.coverage?.buckets?.find(item => item.bucket[0] === 2 && item.bucket[1] === 1 && item.has_crop) ?? lock.coverage?.buckets?.find(item => item.has_crop);
    return cell ? angleCropURL(lock.lock_id, cell.bucket, cropVersion(cell)) : null;
  };
  const locked = locks.length;
  return (
    <section className="frame-face-strip" aria-label="Detected faces in frame" data-testid="frame-face-strip">
      <div className="frame-face-strip-head">
        <span className="eyebrow">TARGET FACE BANK</span>
        <span className="muted">Click a face to lock or unlock it{locked ? ` · ${locked} locked · ${ANGLE_CELLS} angles each` : ''}</span>
        <span className="frame-face-strip-status" role="status">{status}</span>
      </div>
      <div className="frame-face-strip-cards">
        {faces && faces.length > 0 ? faces.map(match => {
          const lock = lockOfFace(match.face.persistent_face_id);
          return <BankItem key={match.id} match={match} crop={crops[match.id]} lock={lock} busy={lockingId === match.id} disabled={disabled}
            onLock={onLock} sources={sources} assignedId={(match.clusterId && assignments[match.clusterId]) || ''} onAssign={onAssign} onRescan={onRescan} />;
        }) : <p className="muted frame-face-empty">{detecting ? 'Looking for faces in this frame…' : faces === null ? 'Faces appear here when the playhead rests on a frame.' : 'Scrub to a frame with a clear, visible face.'}</p>}
        {offFrame.map(lock => <LockedBankItem key={lock.lock_id} lock={lock} onUnlock={onUnlock} onRescan={onRescan} sources={sources}
          assignedId={(lock.track_id && assignments[lock.track_id]) || ''} trackKey={lock.track_id ?? null} onAssign={onAssign} thumb={frontal(lock)} />)}
      </div>
    </section>
  );
}
