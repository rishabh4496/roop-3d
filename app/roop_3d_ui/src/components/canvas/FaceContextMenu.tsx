import { useEffect, useRef } from 'react';
import { poseCellName, type Pose } from '../../utils/poseCells';
import type { ExtractFromFrameResult } from '../../api';

export type FaceDestination = ExtractFromFrameResult['destination'];
export type FaceMenuTarget = {
  /** Viewport coordinates of the right-click / button anchor. */
  x: number;
  y: number;
  /** Left-to-right position of the face in the frame: the server's `face_index`. */
  faceRank: number;
  /** Standard-frame pose of that face when known. */
  pose: Pose | null;
  norm_cx?: number;
  norm_cy?: number;
};

type Props = {
  target: FaceMenuTarget;
  /** FaceSet that "Add to .fsz Faceset" extends (the active source), or null when none is selected. */
  activeSourceName: string | null;
  busy?: boolean;
  onChoose: (destination: FaceDestination, target: FaceMenuTarget) => void;
  onClose: () => void;
};

const sign = (value: number) => `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(0)}°`;

/** The three "Use Face From Frame" actions for one detected face. */
export default function FaceContextMenu({ target, activeSourceName, busy = false, onChoose, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const cell = target.pose ? poseCellName(target.pose) : null;
  const poseText = target.pose ? `Yaw ${sign(target.pose.yaw)}, Pitch ${sign(target.pose.pitch)}` : 'pose unknown';

  useEffect(() => {
    const items = () => [...(ref.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)') ?? [])];
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
  // Keep the menu on screen near the bottom/right edges.
  const left = Math.max(4, Math.min(target.x, (typeof window === 'undefined' ? 1280 : window.innerWidth) - 330));
  const top = Math.max(4, Math.min(target.y, (typeof window === 'undefined' ? 800 : window.innerHeight) - 150));

  return (
    <div ref={ref} className="face-context-menu" role="menu" aria-label={`Use face ${target.faceRank + 1} from this frame`} style={{ position: 'fixed', left, top }}
      onContextMenu={event => event.preventDefault()}>
      <div className="face-menu-title">Face {target.faceRank + 1} in this frame{cell ? ` · ${cell.replace('_', ' / ')}` : ''}</div>
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
    </div>
  );
}
