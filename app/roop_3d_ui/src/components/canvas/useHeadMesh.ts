import { useEffect, useState } from 'react';
import { api } from '../../api';
import type { Coverage } from '../../contracts';
import { prepareFromBuffers, prepareHeadMesh } from './headMesh';

export type PreparedHead = NonNullable<ReturnType<typeof prepareHeadMesh>>;
export type HeadMeshState = {
  head: PreparedHead | null;
  /** The fitted anatomical face, an explicitly labelled orientation proxy, or nothing. */
  kind: 'dense_face' | 'canonical' | null;
  loading: boolean;
  /** Why the fitted anatomical face is not shown. */
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
 * Use the same fitted anatomical surface as the FaceSet reconstruction viewer.
 * Reading a cached reconstruction never builds a legacy full-head identity asset.
 */
export function useHeadMesh(facesetId: string | null | undefined, enabled: boolean, coverage?: Coverage | null): HeadMeshState {
  const [state, setState] = useState<HeadMeshState>(NONE);
  useEffect(() => {
    if (!enabled) { setState(NONE); return undefined; }
    let live = true;
    setState(previous => ({ ...previous, loading: true }));
    const fallback = async (note: string) => {
      const head = await loadCanonical();
      if (live) setState({ head, kind: head ? 'canonical' : null, loading: false, note });
    };
    if (!facesetId) { void fallback('Select a source FaceSet to see its fitted facial surface.'); return () => { live = false; }; }
    const prepare = (payload: Coverage) => {
      const mesh = payload.mesh_full_3d;
      if (mesh?.reconstruction_level !== 'three_tier_facial_surface') throw new Error('Dense face is unavailable');
      const head = prepareHeadMesh({ ...mesh, faces: mesh.indices, coordinate_system: 'mesh' });
      if (!head) throw new Error('empty facial surface');
      return head;
    };
    if (coverage !== undefined) {
      try {
        if (!coverage || coverage.faceset_id !== facesetId) throw new Error('Reconstruction is not ready');
        setState({ head: prepare(coverage), kind: 'dense_face', loading: false, note: '' });
      } catch {
        void fallback('Fitted facial surface is unavailable. This wireframe is a canonical orientation preview.');
      }
      return () => { live = false; };
    }
    let pending = dense.get(facesetId);
    if (!pending) {
      pending = api<Coverage>(`/faceset/3d-state?faceset_id=${encodeURIComponent(facesetId)}`)
        .then(prepare);
      dense.set(facesetId, pending);
      pending.catch(() => { if (dense.get(facesetId) === pending) dense.delete(facesetId); });
    }
    pending.then(head => { if (live) setState({ head, kind: 'dense_face', loading: false, note: '' }); })
      .catch(() => fallback('Fitted facial surface is unavailable. Refresh the FaceSet reconstruction; this wireframe is a canonical orientation preview.'));
    return () => { live = false; };
  }, [facesetId, enabled, coverage]);
  return state;
}
