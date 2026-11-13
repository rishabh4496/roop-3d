import { useEffect, useRef } from 'react';
import { poseCellName, type Pose } from '../../utils/poseCells';
import type { ExtractFromFrameResult } from '../../api';

export type FaceDestination = ExtractFromFrameResult['destination'];
/** Stage 12 target actions, shown when the host provides `onTargetAction`. */
export type TargetAction = 'lock_primary' | 'add_angle' | 'bind_source' | 'exclude' | 'bind_specific' | 'toggle_ultramax' | 'toggle_pose_lock' | 'solo_mask';
/** Extra detail for an action: the FaceSet a `bind_specific` action maps this person to. */
export type TargetActionExtra = { facesetId?: string };
export type FaceMenuTarget = {
  /** Viewport coordinates of the right-click / button anchor. */
  x: number;
  y: number;
  /** Left-to-right position of the face in the frame: the server's `face_index`. */
  faceRank: number;
  faceLabel?: string;
  /** Standard-frame pose of that face when known. */
  pose: Pose | null;
  norm_cx?: number;
  norm_cy?: number;
  /** The face's identity across frames (`faceFlags.faceKey`): what the per-face toggles act on. */
  faceKey?: string;
};

/** A FaceSet the menu can bind the right-clicked person to. */
export type BindableSource = { id: string; name: string };

type Props = {
  target: FaceMenuTarget;
  /** FaceSet that "Add to .fsz Faceset" extends (the active source), or null when none is selected. */
  activeSourceName: string | null;
  busy?: boolean;
  onChoose: (destination: FaceDestination, target: FaceMenuTarget) => void;
  onClose: () => void;
  /** Stage 12: lock / reference-bank / bind / exclude actions, plus the per-face canvas toggles. */
  onTargetAction?: (action: TargetAction, target: FaceMenuTarget, extra?: TargetActionExtra) => void;
  /** The target lock on this video ("Target #1"), when one exists: enables "Add This Angle". */
  targetLabel?: string | null;
  /** FaceSets this person can be bound to by name. */
  bindSources?: BindableSource[];
  /** Current state of the per-face toggles, shown as checkmarks. */
  ultramaxOn?: boolean;
  poseLocked?: boolean;
  maskSolo?: boolean;
};

const sign = (value: number) => `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(0)}°`;

/** The three "Use Face From Frame" actions for one detected face. */
export default function FaceContextMenu({ target, activeSourceName, busy = false, onChoose, onClose, onTargetAction, targetLabel = null, bindSources = [], ultramaxOn = false, poseLocked = false, maskSolo = false }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const cell = target.pose ? poseCellName(target.pose) : null;
  const poseText = target.pose ? `Yaw ${sign(target.pose.yaw)}, Pitch ${sign(target.pose.pitch)}` : 'pose unknown';

  useEffect(() => {
    const items = () => [...(ref.current?.querySelectorAll<HTMLButtonElement>('[role^="menuitem"]:not(:disabled)') ?? [])];
    items()[0]?.focus();
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); onClose(); return; }
      if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
      const all = items();
      if (!all.length) return;
      event.preventDefault();
      const at = all.indexOf(document.activeElement as HTMLButtonElement);
      all[(at + (event.key === 'ArrowDown' ? 1 : -1) + all.length) % all.length].focus();
    };
    const outside = (event: PointerEvent) => { if (!ref.current?.contains(event.target as Node)) onClose(); };
    document.addEventListener('keydown', key);
    document.addEventListener('pointerdown', outside, true);
    return () => { document.removeEventListener('keydown', key); document.removeEventListener('pointerdown', outside, true); };
  }, [onClose]);

  const choose = (destination: FaceDestination) => { onChoose(destination, target); };
  const act = (action: TargetAction, extra?: TargetActionExtra) => { onTargetAction?.(action, target, extra); };
  // Keep the menu on screen near the bottom/right edges.
  const left = Math.max(4, Math.min(target.x, (typeof window === 'undefined' ? 1280 : window.innerWidth) - 330));
  const top = Math.max(4, Math.min(target.y, (typeof window === 'undefined' ? 800 : window.innerHeight) - 260));

  return (
    <div ref={ref} className="face-context-menu" role="menu" aria-label={`Use face ${target.faceRank + 1} from this frame`} style={{ position: 'fixed', left, top, maxHeight: `calc(100vh - ${top}px - 8px)`, overflowY: 'auto' }}
      onContextMenu={event => event.preventDefault()}>
      <div className="face-menu-title">{target.faceLabel ?? `Face ${target.faceRank + 1}`} in this frame{cell ? ` · ${cell.replace('_', ' / ')}` : ''}</div>
      <button type="button" role="menuitem" disabled={busy} onClick={() => choose('source_gallery')}>
        <span aria-hidden="true">🧑</span> Use as Active Source Face
      </button>
      <button type="button" role="menuitem" disabled={busy || !activeSourceName} onClick={() => choose('active_faceset')}
        title={activeSourceName ? `Extends “${activeSourceName}” at the matching pose cell` : 'Select or create a source FaceSet first'}>
        <span aria-hidden="true">🗂</span> Add to .fsz Faceset (Auto-Pose Cell [{poseText}]{cell ? ` → ${cell}` : ''})
        {!activeSourceName && <small>Select a source FaceSet first</small>}
      </button>
      <button type="button" role="menuitem" disabled={busy} onClick={() => choose('target_cluster')}>
        <span aria-hidden="true">📌</span> Lock as Target Identity Cluster
      </button>
      {onTargetAction && <>
        <div className="face-menu-title" role="separator">Target lock</div>
        <button type="button" role="menuitem" disabled={busy} onClick={() => act('lock_primary')}>
          <span aria-hidden="true">🎯</span> Lock as Primary Target
        </button>
        <button type="button" role="menuitem" disabled={busy || !targetLabel} onClick={() => act('add_angle')}
          title={targetLabel ? `Adds this pose (${poseText}) to ${targetLabel}'s reference bank` : 'Lock a primary target first'}>
          <span aria-hidden="true">➕</span> Add This Angle to Target Reference Bank
          {!targetLabel && <small>Lock a primary target first</small>}
        </button>
        <button type="button" role="menuitem" disabled={busy || !activeSourceName} onClick={() => act('bind_source')}
          title={activeSourceName ? `Maps this person to “${activeSourceName}”` : 'Select or create a source FaceSet first'}>
          <span aria-hidden="true">🔗</span> Bind Specific Source / .fsz to This Person
          {!activeSourceName && <small>Select a source FaceSet first</small>}
        </button>
        <button type="button" role="menuitem" disabled={busy} onClick={() => act('exclude')}>
          <span aria-hidden="true">🚫</span> Exclude / Ignore This Face in Current Segment
        </button>
        {bindSources.length > 0 && <>
          <div className="face-menu-title" role="separator">Bind .fsz identity to this face</div>
          {bindSources.slice(0, 6).map(source => (
            <button key={source.id} type="button" role="menuitem" disabled={busy} onClick={() => act('bind_specific', { facesetId: source.id })}>
              <span aria-hidden="true">🧬</span> Bind “{source.name}”
            </button>
          ))}
        </>}
        <div className="face-menu-title" role="separator">This face on the canvas</div>
        <button type="button" role="menuitemcheckbox" aria-checked={ultramaxOn} onClick={() => act('toggle_ultramax')}
          title="The render engine applies one enhancer to every swapped face, so this turns UltraMax on (or off) for the whole swap">
          <span aria-hidden="true">{ultramaxOn ? '✅' : '✨'}</span> UltraMax enhancer {ultramaxOn ? 'on' : 'off'}<small>applies to all swapped faces</small>
        </button>
        <button type="button" role="menuitemcheckbox" aria-checked={poseLocked} disabled={!target.faceKey} onClick={() => act('toggle_pose_lock')}
          title="Hold this face's wireframe, gimbal and gaze at its current 6DoF pose; the canvas overlay only, the render is not affected">
          <span aria-hidden="true">{poseLocked ? '🔒' : '🔓'}</span> {poseLocked ? 'Unlock' : 'Lock'} 6DoF pose filter<small>canvas overlay</small>
        </button>
        <button type="button" role="menuitemcheckbox" aria-checked={maskSolo} disabled={!target.faceKey} onClick={() => act('solo_mask')}
          title="Show only this face's mask and mesh, dim the others, and open the matte view">
          <span aria-hidden="true">◐</span> {maskSolo ? 'Show all face masks' : 'Solo this face mask'}
        </button>
      </>}
    </div>
  );
}
