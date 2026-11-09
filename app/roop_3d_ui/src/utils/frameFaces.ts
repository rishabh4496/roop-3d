import { useMemo } from 'react';
import type { DetectedFace } from '../contracts';
import type { FrameFace, FrameMeta } from '../frameScrub';
import { useFaceSelectionStore } from '../useFaceSelectionStore';

/**
 * Faces detected in the frame on screen ("candidates") and the target faces already in the selection store.
 *
 * The store knows a track by one static box (where it first appeared), so it cannot draw a face that moves; the frame
 * detection can. A candidate that overlaps a stored face is that face (same id, so selection, role, source mapping and
 * the context menu keep working) drawn at its real position in this frame; the rest are free candidates that locking turns
 * into a track.
 */
export interface MatchedFrameFace {
  face: FrameFace;
  /** The stored face's id when matched, else `frame:<frame>:<rank>`. */
  id: string;
  clusterId: string | null;
  stored: DetectedFace | null;
  /** The selected target: the active face, or another face of the active track. */
  locked: boolean;
  /** Normalised 5-point landmarks (the overlay's coordinate space). */
  landmarks5: [number, number][];
}

export const MATCH_IOU = 0.3;

export function iou(a: readonly number[], b: readonly number[]): number {
  const width = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), height = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (width <= 0 || height <= 0) return 0;
  const inter = width * height, union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

export function matchFrameFaces(
  faces: FrameFace[], meta: Pick<FrameMeta, 'frame_index' | 'width' | 'height'>, stored: DetectedFace[],
  activeSelection: string | null, minIou = MATCH_IOU,
): MatchedFrameFace[] {
  const pairs: { frame: number; stored: number; score: number }[] = [];
  faces.forEach((face, f) => stored.forEach((candidate, s) => {
    if (face.track_id && typeof face.track_id === 'string') {
      if (face.track_id === candidate.clusterId) pairs.push({ frame: f, stored: s, score: 2 });
      return;
    }
    const score = iou(face.bbox_norm, candidate.bbox);
    if (score >= minIou) pairs.push({ frame: f, stored: s, score });
  }));
  pairs.sort((a, b) => b.score - a.score);
  const takenFrame = new Set<number>(), takenStored = new Set<number>(), match = new Map<number, DetectedFace>();
  for (const pair of pairs) {
    if (takenFrame.has(pair.frame) || takenStored.has(pair.stored)) continue;
    takenFrame.add(pair.frame); takenStored.add(pair.stored); match.set(pair.frame, stored[pair.stored]);
  }
  const active = activeSelection ? stored.find(face => face.id === activeSelection) ?? null : null;
  return faces.map((face, index) => {
    const found = match.get(index) ?? null;
    return {
      face, stored: found,
      id: found?.id ?? face.track_id ?? `frame:${meta.frame_index}:${face.face_index}`,
      clusterId: found?.clusterId ?? face.track_id ?? null,
      locked: !!face.is_target || (!!found && (found.id === activeSelection || (!!active && found.clusterId === active.clusterId))),
      landmarks5: face.kps_5.map(([x, y]) => [x / meta.width, y / meta.height] as [number, number]),
    };
  });
}

/** The matched faces of the frame on screen, kept current as the selection changes. */
export function useMatchedFrameFaces(faces: FrameFace[] | null, meta: FrameMeta | null, fps: number): MatchedFrameFace[] {
  const detected = useFaceSelectionStore(state => state.detectedFaces);
  const clusters = useFaceSelectionStore(state => state.targetClusters);
  const activeSelection = useFaceSelectionStore(state => state.activeSelection);
  const getFacesAtTimestamp = useFaceSelectionStore(state => state.getFacesAtTimestamp);
  return useMemo(() => {
    if (!faces || !meta) return [];
    const stored = detected.length ? getFacesAtTimestamp(meta.timestamp_sec, fps) : [];
    return matchFrameFaces(faces, meta, stored, activeSelection);
    // `clusters` re-runs the match when a track is split or removed, which changes what `getFacesAtTimestamp` returns.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [faces, meta, detected, clusters, activeSelection, fps, getFacesAtTimestamp]);
}

/** A square-ish crop rectangle (native pixels) around a face with some margin, clamped to the frame. */
export function cropRect(bbox: readonly number[], width: number, height: number, margin = 0.3): { x: number; y: number; w: number; h: number } {
  const bw = bbox[2] - bbox[0], bh = bbox[3] - bbox[1];
  const side = Math.max(bw, bh) * (1 + margin * 2);
  const cx = (bbox[0] + bbox[2]) / 2, cy = (bbox[1] + bbox[3]) / 2;
  const w = Math.min(side, width), h = Math.min(side, height);
  const x = Math.max(0, Math.min(width - w, cx - w / 2)), y = Math.max(0, Math.min(height - h, cy - h / 2));
  return { x, y, w, h };
}

/** The server face (from `/media/detect-frame-faces`) that is the same face as the clicked candidate. */
export function bestOverlap<T extends { bbox: number[] }>(face: FrameFace, candidates: T[]): T | null {
  let best: T | null = null, bestScore = 0;
  for (const candidate of candidates) {
    const score = iou(face.bbox_norm, candidate.bbox);
    if (score > bestScore) { best = candidate; bestScore = score; }
  }
  return best;
}
