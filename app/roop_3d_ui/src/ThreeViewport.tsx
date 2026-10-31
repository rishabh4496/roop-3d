import React from 'react';
import type { Coverage, Reconstruction, Track } from './contracts';
import PoseGeometryViewer from './components/three/PoseGeometryViewer';

export { default as PoseGeometryViewer } from './components/three/PoseGeometryViewer';

export function ThreeViewport({
  coverage,
  track,
  mesh,
}: {
  coverage: Coverage | null;
  track: Track | null;
  mesh: Reconstruction | null;
}) {
  return <PoseGeometryViewer coverage={coverage} track={track} mesh={mesh} />;
}

export function PoseCoverageInspector({
  coverage,
  track,
  mesh,
}: {
  coverage: Coverage | null;
  track: Track | null;
  mesh: Reconstruction | null;
}) {
  return (
    <section className="pose-inspector">
      <div className="section-title">
        <span>POSE & GEOMETRY</span>
        <span className="tiny-badge">{mesh?.method === 'flame' ? 'FLAME MESH' : 'ESTIMATED 2.5D'}</span>
      </div>
      <PoseGeometryViewer coverage={coverage} track={track} mesh={mesh} />
      <p className="muted compact">
        {mesh
          ? `Imported ${mesh.method === 'flame' ? 'FLAME 3D' : 'approximate 2.5D'} mesh · user supplied provenance`
          : 'Canonical low-poly 3D head with live tracking slerp + FaceSet orientation sphere.'}
      </p>
      {mesh?.texture && (
        <details>
          <summary>UV texture</summary>
          <img className="uv-texture" src={mesh.texture} alt="Actual reconstruction UV texture" />
        </details>
      )}
      <div className="pose-legend">
        <span><i className="dot mint" />Observed views {coverage?.views.length ?? 0}</span>
        <span><i className="dot amber" />Sparse</span>
        <span><i className="dot dim" />Unobserved blind spots</span>
      </div>
      {coverage && (
        <div className="pose-bins">
          {Object.entries(coverage.bins).map(([bin, count]) => (
            <span key={bin}>
              {bin.replaceAll('_', ' ')} <strong>{count}</strong>
            </span>
          ))}
        </div>
      )}
    </section>
  );
}
