import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { Reconstruction } from '../../contracts';
import { prepareFromBuffers, prepareHeadMesh } from './headMesh';

export type PreparedHead = NonNullable<ReturnType<typeof prepareHeadMesh>>;
export type HeadMeshState = {
  head: PreparedHead | null;
  /** `makehuman`: the dense CC0 head fitted to the active FaceSet; `canonical`: the built-in low-poly head; null while nothing is loaded. */
  kind: 'makehuman' | 'canonical' | null;
  loading: boolean;
  /** Why the dense MakeHuman topology is not shown (no FaceSet, not built yet, ...). Empty when it is. */
  note: string;
};
const NONE: HeadMeshState = { head: null, kind: null, loading: false, note: '' };

const dense = new Map<string, Promise<PreparedHead>>();
let canonical: Promise<PreparedHead | null> | null = null;

/** The built-in low-poly head, loaded on demand so the 3D library stays out of the main bundle. */
function loadCanonical(): Promise<PreparedHead | null> {
  canonical ??= import('../three/PoseGeometryViewer').then(({ createCanonicalHeadGeometry }) => {
    const geometry = createCanonicalHeadGeometry();
    const head = prepareFromBuffers(geometry.getAttribute('position').array, geometry.getIndex()!.array);
    geometry.dispose();
    return head;
  }).catch(() => null);
  return canonical;
}

/** Forget cached meshes (a FaceSet's identity asset was rebuilt, or tests). */
export function clearHeadMeshCache() { dense.clear(); canonical = null; }

/**
 * The head topology for the canvas wireframe overlay. With an active FaceSet it asks for that FaceSet's MakeHuman identity asset
 * (`GET /facesets/{id}/identity-asset/mesh?template=makehuman`, which never builds anything); when that is missing it falls back to the
 * canonical head and says why, so the overlay is never silently different from what was asked for.
 */
export function useHeadMesh(facesetId: string | null | undefined, enabled: boolean): HeadMeshState {
  const [state, setState] = useState<HeadMeshState>(NONE);
  useEffect(() => {
    if (!enabled) { setState(NONE); return undefined; }
    let live = true;
    setState(previous => ({ ...previous, loading: true }));
    const fallback = async (note: string) => {
      const head = await loadCanonical();
      if (live) setState({ head, kind: head ? 'canonical' : null, loading: false, note });
    };
    if (!facesetId) { void fallback('Select a source FaceSet to see its MakeHuman topology.'); return () => { live = false; }; }
    let pending = dense.get(facesetId);
    if (!pending) {
      pending = api<Reconstruction>(`/facesets/${encodeURIComponent(facesetId)}/identity-asset/mesh?template=makehuman`)
        .then(mesh => { const head = prepareHeadMesh(mesh); if (!head) throw new Error('empty mesh'); return head; });
      dense.set(facesetId, pending);
      pending.catch(() => { if (dense.get(facesetId) === pending) dense.delete(facesetId); });
    }
    pending.then(head => { if (live) setState({ head, kind: 'makehuman', loading: false, note: '' }); })
      .catch(() => fallback('MakeHuman topology is not built for this FaceSet (build it from the FaceSet panel): showing the canonical head.'));
    return () => { live = false; };
  }, [facesetId, enabled]);
  return state;
}
