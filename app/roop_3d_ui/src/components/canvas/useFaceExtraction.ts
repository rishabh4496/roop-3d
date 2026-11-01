import { useCallback, useEffect, useRef, useState } from 'react';
import { extractFaceFromFrame, type ExtractFromFrameResult } from '../../api';
import type { FaceSet } from '../../contracts';
import type { FaceDestination, FaceMenuTarget } from './FaceContextMenu';

export type Toast = { kind: 'ok' | 'error'; text: string };

type Options = {
  target: string;
  frameIndex: number;
  /** Active source FaceSet id: the FaceSet "Add to .fsz Faceset" extends. */
  activeSourceId: string | null;
  /** The new/extended FaceSet, so the source gallery strip updates immediately. */
  onSourceExtracted?: (faceset: FaceSet) => void;
  /** Re-render the swapped preview (the FaceSet or identity changed under an unchanged payload). */
  onRefreshPreview?: () => void;
  /** Test seam; defaults to POST /api/faces/extract_from_frame. */
  extract?: typeof extractFaceFromFrame;
  toastMs?: number;
};

const describe = (result: ExtractFromFrameResult): string => {
  const cell = result.face.pose_cell.replace('_', ' / ');
  if (result.destination === 'source_gallery') return `Active source face set: ${result.active_source?.name ?? 'new FaceSet'} (${cell})`;
  if (result.destination === 'active_faceset') {
    return `Added to ${result.faceset?.name ?? 'FaceSet'} in cell ${result.inserted?.pose_cell.replace('_', ' / ') ?? cell} (${result.faceset?.item_count ?? '?'} faces)`;
  }
  return result.cluster?.created ? `Locked as target identity #${(result.cluster.rank ?? 0) + 1}` : `Refined target identity #${(result.cluster?.rank ?? 0) + 1}`;
};

/** Dispatches `POST /api/faces/extract_from_frame` for the chosen destination and reports the outcome as a toast. */
export function useFaceExtraction({ target, frameIndex, activeSourceId, onSourceExtracted, onRefreshPreview, extract = extractFaceFromFrame, toastMs = 4000 }: Options) {
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  const show = useCallback((next: Toast) => {
    setToast(next);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setToast(null), toastMs);
  }, [toastMs]);

  const run = useCallback(async (destination: FaceDestination, menu: Pick<FaceMenuTarget, 'faceRank'>) => {
    if (!target || busy) return null;
    setBusy(true);
    try {
      const result = await extract({
        target_media_path: target, frame_index: frameIndex, face_index: menu.faceRank, destination,
        ...(destination === 'active_faceset' && activeSourceId ? { faceset_name: activeSourceId } : {}),
      });
      if (result.active_source) onSourceExtracted?.({ faceset_id: result.active_source.faceset_id, name: result.active_source.name, item_count: 1 });
      else if (result.faceset) onSourceExtracted?.({ faceset_id: result.faceset.faceset_id, name: result.faceset.name, item_count: result.faceset.item_count });
      if (destination !== 'target_cluster') onRefreshPreview?.();
      show({ kind: 'ok', text: describe(result) });
      return result;
    } catch (error) {
      show({ kind: 'error', text: (error as Error).message || 'Could not extract a face from this frame.' });
      return null;
    } finally {
      setBusy(false);
    }
  }, [target, busy, extract, frameIndex, activeSourceId, onSourceExtracted, onRefreshPreview, show]);

  return { run, busy, toast, dismiss: () => setToast(null) };
}
