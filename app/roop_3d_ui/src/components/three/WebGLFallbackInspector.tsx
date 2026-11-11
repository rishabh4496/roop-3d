import { useMemo } from 'react';
import type { Reconstruction } from '../../contracts';

const MAX_EDGES = 5000;

/** Front-view outline of a mesh as SVG path data, in a 0..100 box (y down). Pure so it can be tested without a DOM. */
export function frontProjection(mesh: Reconstruction | null | undefined, maxEdges = MAX_EDGES): { path: string; edges: number; bounds: { width: number; height: number; depth: number } | null } {
  const vertices = Array.isArray(mesh?.vertices) ? mesh.vertices : [];
  const faces = Array.isArray(mesh?.faces) ? mesh.faces : [];
  const valid = (v: unknown): v is number[] => Array.isArray(v) && v.length >= 3 && v.every(n => Number.isFinite(n));
  if (!vertices.length || !faces.length) return { path: '', edges: 0, bounds: null };
  const flipY = mesh?.coordinate_system === 'opengl';          // mesh and opencv frames already have +y down, like SVG
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const vertex of vertices) {
    if (!valid(vertex)) continue;
    minX = Math.min(minX, vertex[0]); maxX = Math.max(maxX, vertex[0]);
    minY = Math.min(minY, vertex[1]); maxY = Math.max(maxY, vertex[1]);
    minZ = Math.min(minZ, vertex[2]); maxZ = Math.max(maxZ, vertex[2]);
  }
  if (!Number.isFinite(minX)) return { path: '', edges: 0, bounds: null };
  const span = Math.max(maxX - minX, maxY - minY, 1e-9);
  const point = (index: number): string | null => {
    const vertex = vertices[index];
    if (!valid(vertex)) return null;
    const x = 5 + ((vertex[0] - minX) / span) * 90;
    const y = 5 + (((flipY ? maxY - vertex[1] : vertex[1] - minY)) / span) * 90;
    return `${x.toFixed(2)} ${y.toFixed(2)}`;
  };
  const stride = Math.max(1, Math.ceil((faces.length * 3) / maxEdges));
  const segments: string[] = [];
  for (let i = 0; i < faces.length; i += stride) {
    const face = faces[i];
    if (!Array.isArray(face) || face.length < 3) continue;
    const [a, b, c] = [point(face[0]), point(face[1]), point(face[2])];
    if (a && b && c) segments.push(`M${a}L${b}L${c}Z`);
  }
  return { path: segments.join(''), edges: segments.length * 3, bounds: { width: maxX - minX, height: maxY - minY, depth: maxZ - minZ } };
}

/**
 * What the 3D viewport shows when WebGL is missing, was lost for good, or the user chose the lightweight inspector: the
 * mesh's numbers and a flat front outline drawn with SVG (no GPU involved). It reads the same `Reconstruction` the viewport does.
 */
export default function WebGLFallbackInspector({ mesh, reason, onRetry, onUse3D, label = 'Lightweight mesh inspector' }: {
  mesh: Reconstruction | null | undefined; reason: string; onRetry?: () => void; onUse3D?: () => void; label?: string;
}) {
  const outline = useMemo(() => frontProjection(mesh), [mesh]);
  const vertexCount = Array.isArray(mesh?.vertices) ? mesh.vertices.length : 0;
  const faceCount = Array.isArray(mesh?.faces) ? mesh.faces.length : 0;
  const unit = mesh?.coordinate_system === 'mesh' ? 'm' : '';
  return (
    <section className="webgl-fallback" role="region" aria-label={label} data-testid="webgl-fallback">
      <div className="webgl-fallback-head">
        <strong>{label}</strong>
        <span className="status-chip" title="The 3D viewport is not drawing">no 3D drawing</span>
      </div>
      <p className="muted" role="status">{reason || 'The 3D viewport is not available.'}</p>
      <div className="webgl-fallback-body">
        {outline.path
          ? <svg viewBox="0 0 100 100" role="img" aria-label="Front outline of the mesh" className="webgl-fallback-outline" preserveAspectRatio="xMidYMid meet">
              <path d={outline.path} fill="none" stroke="currentColor" strokeWidth="0.15" vectorEffect="non-scaling-stroke" />
            </svg>
          : <p className="muted">No mesh is loaded yet.</p>}
        <dl className="webgl-fallback-stats">
          <div><dt>Vertices</dt><dd>{vertexCount.toLocaleString()}</dd></div>
          <div><dt>Triangles</dt><dd>{faceCount.toLocaleString()}</dd></div>
          {outline.bounds && <div><dt>Size (W × H × D)</dt><dd>{[outline.bounds.width, outline.bounds.height, outline.bounds.depth].map(v => v.toFixed(3)).join(' × ')} {unit}</dd></div>}
          <div><dt>Textured</dt><dd>{mesh?.texture ? 'yes' : 'no'}</dd></div>
          {mesh?.provenance && <div><dt>Provenance</dt><dd>{String(mesh.provenance).replaceAll('_', ' ')}</dd></div>}
        </dl>
      </div>
      <div className="webgl-fallback-actions">
        {onRetry && <button type="button" className="button small" onClick={onRetry}>Restart 3D view</button>}
        {onUse3D && <button type="button" className="button small" onClick={onUse3D}>Use the 3D viewport</button>}
      </div>
    </section>
  );
}
