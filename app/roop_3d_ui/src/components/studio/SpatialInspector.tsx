import React, { useMemo, useState } from 'react';
import type { Coverage, Reconstruction, ReconstructionStatus, Track } from '../../contracts';
import type { WorkstationParameters } from '../workstation/pipeline';
import { viewerLighting, viewerMorph } from '../three/viewerRigs';
import LightProbeSphere from '../three/LightProbeSphere';
import FlameMeshViewer from '../three/FlameMeshViewer';
import CoverageRadarPanel from '../three/CoverageRadarPanel';
import IdentityMorphPanel from '../three/IdentityMorphPanel';
import PoseGeometryViewer from '../three/PoseGeometryViewer';

export type SpatialTab = 'pose' | 'coverage' | 'morph' | 'lighting' | 'flame';
const TABS: { id: SpatialTab; label: string }[] = [
  { id: 'pose', label: 'Head Pose' }, { id: 'coverage', label: 'Coverage' }, { id: 'morph', label: 'Morph' },
  { id: 'lighting', label: 'Light Probe' }, { id: 'flame', label: 'FLAME Mesh' },
];

export interface SpatialInspectorProps {
  coverage: Coverage | null;
  track: Track | null;
  mesh: Reconstruction | null;
  reconstruction?: ReconstructionStatus | null;
  parameters?: WorkstationParameters;
  setParameters?: (params: WorkstationParameters) => void;
  defaultTab?: SpatialTab;
}

export function SpatialInspector({
  coverage,
  track,
  mesh,
  reconstruction,
  parameters,
  setParameters,
  defaultTab = 'pose',
}: SpatialInspectorProps) {
  const [activeTab, setActiveTab] = useState<SpatialTab>(defaultTab);
  const [morphPreview, setMorphPreview] = useState(false);

  const isFlame = mesh?.method === 'flame';
  // The viewer's lights follow the probe and its geometry ghost / hairline ring follow the morph controls; memoised so the 3D scene only updates on a real change.
  const lighting = useMemo(() => viewerLighting(parameters), [parameters]);
  const morph = useMemo(() => viewerMorph(parameters, morphPreview || activeTab === 'morph'), [parameters, morphPreview, activeTab]);

  return (
    <section className="pose-inspector spatial-inspector-host" role="region" aria-label="3D Spatial Inspector">
      <div className="section-title">
        <span>3D SPATIAL STUDIO</span>
        <span className="tiny-badge">
          {isFlame ? 'FLAME 3D MESH' : mesh ? '2.5D GEOMETRY' : 'CANONICAL HEAD'}
        </span>
      </div>

      <div className="spatial-nav-tabs" role="tablist" aria-label="Spatial Inspection Modes">
        {TABS.map(tab => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            id={`spatial-tab-${tab.id}`}
            className={`spatial-nav-btn ${activeTab === tab.id ? 'active' : ''}`}
            aria-selected={activeTab === tab.id}
            aria-controls={`spatial-panel-${tab.id}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Panels: keep mounted to prevent WebGL context destruction */}
      <div
        role="tabpanel"
        id="spatial-panel-pose"
        aria-labelledby="spatial-tab-pose"
        hidden={activeTab !== 'pose' && activeTab !== 'morph' && activeTab !== 'lighting'}
        style={{ display: activeTab === 'pose' || activeTab === 'morph' || activeTab === 'lighting' ? 'block' : 'none' }}
      >
        <PoseGeometryViewer coverage={coverage} track={track} mesh={mesh} reconstruction={reconstruction}
          renderMode={parameters ? parameters.render_mode ?? 'standard' : undefined} lighting={lighting} morph={morph} />
      </div>

      <div
        role="tabpanel"
        id="spatial-panel-coverage"
        aria-labelledby="spatial-tab-coverage"
        hidden={activeTab !== 'coverage'}
        style={{ display: activeTab === 'coverage' ? 'block' : 'none' }}
      >
        <CoverageRadarPanel coverage={coverage} track={track} />
      </div>

      <div
        role="tabpanel"
        id="spatial-panel-morph"
        aria-labelledby="spatial-tab-morph"
        hidden={activeTab !== 'morph'}
        style={{ display: activeTab === 'morph' ? 'block' : 'none' }}
      >
        {parameters && setParameters
          ? <IdentityMorphPanel parameters={parameters} setParameters={setParameters} preview={morphPreview} onPreview={setMorphPreview} />
          : <p className="muted">Open a project to edit the identity morph.</p>}
      </div>

      <div
        role="tabpanel"
        id="spatial-panel-lighting"
        aria-labelledby="spatial-tab-lighting"
        hidden={activeTab !== 'lighting'}
        style={{ display: activeTab === 'lighting' ? 'block' : 'none' }}
      >
        <LightProbeSphere parameters={parameters} setParameters={setParameters} />
        {parameters && !parameters.relight && <p className="muted compact">3D relighting is off: the head above keeps a neutral light. Turn on “3D SH relighting” to preview and render with this light.</p>}
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
