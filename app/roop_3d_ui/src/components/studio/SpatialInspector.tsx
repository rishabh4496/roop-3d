import React, { useState } from 'react';
import type { Coverage, Reconstruction, Track } from '../../contracts';
import type { WorkstationParameters } from '../workstation/pipeline';
import PoseGeometryViewer from '../three/PoseGeometryViewer';
import LightProbeSphere from '../three/LightProbeSphere';
import FlameMeshViewer from '../three/FlameMeshViewer';

export type SpatialTab = 'pose' | 'lighting' | 'flame';

export interface SpatialInspectorProps {
  coverage: Coverage | null;
  track: Track | null;
  mesh: Reconstruction | null;
  parameters?: WorkstationParameters;
  setParameters?: (params: WorkstationParameters) => void;
  defaultTab?: SpatialTab;
}

export function SpatialInspector({
  coverage,
  track,
  mesh,
  parameters,
  setParameters,
  defaultTab = 'pose',
}: SpatialInspectorProps) {
  const [activeTab, setActiveTab] = useState<SpatialTab>(defaultTab);

  const isFlame = mesh?.method === 'flame';

  return (
    <section className="pose-inspector spatial-inspector-host" role="region" aria-label="3D Spatial Inspector">
      <div className="section-title">
        <span>3D SPATIAL STUDIO</span>
        <span className="tiny-badge">
          {isFlame ? 'FLAME 3D MESH' : mesh ? '2.5D GEOMETRY' : 'CANONICAL HEAD'}
        </span>
      </div>

      <div className="spatial-nav-tabs" role="tablist" aria-label="Spatial Inspection Modes">
        <button
          type="button"
          role="tab"
          id="spatial-tab-pose"
          className={`spatial-nav-btn ${activeTab === 'pose' ? 'active' : ''}`}
          aria-selected={activeTab === 'pose'}
          aria-controls="spatial-panel-pose"
          onClick={() => setActiveTab('pose')}
        >
          <span>Head Pose</span>
        </button>

        <button
          type="button"
          role="tab"
          id="spatial-tab-lighting"
          className={`spatial-nav-btn ${activeTab === 'lighting' ? 'active' : ''}`}
          aria-selected={activeTab === 'lighting'}
          aria-controls="spatial-panel-lighting"
          onClick={() => setActiveTab('lighting')}
        >
          <span>Light Probe</span>
        </button>

        <button
          type="button"
          role="tab"
          id="spatial-tab-flame"
          className={`spatial-nav-btn ${activeTab === 'flame' ? 'active' : ''}`}
          aria-selected={activeTab === 'flame'}
          aria-controls="spatial-panel-flame"
          onClick={() => setActiveTab('flame')}
        >
          <span>FLAME Mesh</span>
        </button>
      </div>

      {/* Tab Panels: keep mounted to prevent WebGL context destruction */}
      <div
        role="tabpanel"
        id="spatial-panel-pose"
        aria-labelledby="spatial-tab-pose"
        hidden={activeTab !== 'pose'}
        style={{ display: activeTab === 'pose' ? 'block' : 'none' }}
      >
        <PoseGeometryViewer coverage={coverage} track={track} mesh={mesh} />
      </div>

      <div
        role="tabpanel"
        id="spatial-panel-lighting"
        aria-labelledby="spatial-tab-lighting"
        hidden={activeTab !== 'lighting'}
        style={{ display: activeTab === 'lighting' ? 'block' : 'none' }}
      >
        <LightProbeSphere parameters={parameters} setParameters={setParameters} />
      </div>

      <div
        role="tabpanel"
        id="spatial-panel-flame"
        aria-labelledby="spatial-tab-flame"
        hidden={activeTab !== 'flame'}
        style={{ display: activeTab === 'flame' ? 'block' : 'none' }}
      >
        <FlameMeshViewer mesh={mesh} poseStrength={parameters?.pose_strength ?? 0.35} />
      </div>

      <p className="muted compact" style={{ marginTop: 10 }}>
        {mesh
          ? `Imported ${isFlame ? 'FLAME 3D' : 'approximate 2.5D'} topology · user supplied provenance`
          : 'Canonical low-poly 3D head with live tracking slerp + FaceSet orientation sphere.'}
      </p>

      {mesh?.texture && (
        <details style={{ marginTop: 6 }}>
          <summary>UV texture</summary>
          <img className="uv-texture" src={mesh.texture} alt="Actual reconstruction UV texture" />
        </details>
      )}

      {coverage && (
        <>
          <div className="pose-legend">
            <span><i className="dot mint" />Observed views {coverage.views?.length ?? 0}</span>
            <span><i className="dot amber" />Sparse</span>
            <span><i className="dot dim" />Unobserved blind spots</span>
          </div>

          <div className="pose-bins">
            {Object.entries(coverage.bins ?? {}).map(([bin, count]) => (
              <span key={bin}>
                {bin.replaceAll('_', ' ')} <strong>{count}</strong>
              </span>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

export default SpatialInspector;
