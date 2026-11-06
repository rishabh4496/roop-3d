import { useEffect, useState } from 'react';
import type { FrameMeta } from '../../frameScrub';
import { cropRect, type MatchedFrameFace } from '../../utils/frameFaces';

const THUMB = 72;
const NO_CROPS: Record<string, string> = {};
const NO_FACES: MatchedFrameFace[] = [];

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

export interface FrameFaceStripProps {
  imageUrl: string;
  meta: FrameMeta | null;
  /** Null until detection has run for the frame on screen. */
  faces: MatchedFrameFace[] | null;
  detecting: boolean;
  /** Id of the face being locked (its card shows progress). */
  lockingId: string | null;
  onLock: (face: MatchedFrameFace) => void;
  disabled?: boolean;
}

/**
 * The faces detected in the frame on screen, as cards right above the timeline: click one to make it the target, exactly as
 * clicking its box on the canvas does. A locked face is the current target.
 */
export default function FrameFaceStrip({ imageUrl, meta, faces, detecting, lockingId, onLock, disabled = false }: FrameFaceStripProps) {
  const crops = useFaceCrops(imageUrl, meta, faces ?? NO_FACES);
  const status = detecting ? 'Detecting faces…' : faces === null ? '' : faces.length ? `${faces.length} in frame ${meta?.frame_index ?? ''}` : 'No faces in this frame';
  return (
    <section className="frame-face-strip" aria-label="Detected faces in frame" data-testid="frame-face-strip">
      <div className="frame-face-strip-head">
        <span className="eyebrow">DETECTED FACES IN FRAME</span>
        <span className="muted">Click to target</span>
        <span className="frame-face-strip-status" role="status">{status}</span>
      </div>
      <div className="frame-face-strip-cards">
        {faces && faces.length > 0 ? faces.map(match => {
          const pose = match.face.pose3d, busy = lockingId === match.id;
          return (
            <button key={match.id} type="button" className={`frame-face-card ${match.locked ? 'locked' : ''} ${busy ? 'busy' : ''}`}
              aria-pressed={match.locked} disabled={disabled || busy} data-face-rank={match.face.face_index}
              title={match.locked ? `Face #${match.face.face_index + 1} is the current target` : `Make face #${match.face.face_index + 1} the target`}
              onClick={() => onLock(match)}>
              <span className="frame-face-thumb">{crops[match.id] ? <img src={crops[match.id]} alt={`Face ${match.face.face_index + 1} crop`} /> : <i aria-hidden="true">◎</i>}</span>
              <span className="frame-face-meta">
                <strong>#{match.face.face_index + 1}</strong>
                <span className="frame-face-pose">Y {signed(pose.yaw)}° P {signed(pose.pitch)}°</span>
                {match.face.confidence != null && <small>{Math.round(match.face.confidence * 100)}%</small>}
              </span>
              {match.locked && <span className="frame-face-badge">TARGET LOCKED</span>}
              {busy && <span className="frame-face-badge pending">Locking…</span>}
            </button>
          );
        }) : <p className="muted frame-face-empty">{detecting ? 'Looking for faces in this frame…' : faces === null ? 'Faces appear here when the playhead rests on a frame.' : 'Scrub to a frame with a clear, visible face.'}</p>}
      </div>
    </section>
  );
}
