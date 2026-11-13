import React from 'react';
import type { Coverage, Reconstruction, Track } from './contracts';
import type { WorkstationParameters } from './components/workstation/pipeline';
import type { RenderModeId } from './components/workstation/renderModes';
import type { ReconstructionStatus } from './contracts';
import PoseGeometryViewer from './components/three/PoseGeometryViewer';
import LightProbeSphere from './components/three/LightProbeSphere';
import FlameMeshViewer from './components/three/FlameMeshViewer';
import SpatialInspector from './components/studio/SpatialInspector';

// Consumers load this whole module lazily (`lazy(() => import('./ThreeViewport'))`), so these are plain components.
export { PoseGeometryViewer, LightProbeSphere, FlameMeshViewer, SpatialInspector };

export function ThreeViewport({
  coverage,
  track,
  mesh,
  reconstruction,
  renderMode,
}: {
  coverage: Coverage | null;
  track: Track | null;
  mesh: Reconstruction | null;
  reconstruction?: ReconstructionStatus | null;
  renderMode?: RenderModeId;
}) {
  return <PoseGeometryViewer coverage={coverage} track={track} mesh={mesh} reconstruction={reconstruction} renderMode={renderMode} />;
}

export function PoseCoverageInspector({
  coverage,
  track,
  mesh,
  reconstruction,
  parameters,
  setParameters,
}: {
  coverage: Coverage | null;
  track: Track | null;
  mesh: Reconstruction | null;
  reconstruction?: ReconstructionStatus | null;
  parameters?: WorkstationParameters;
  setParameters?: (params: WorkstationParameters) => void;
}) {
  return (
    <SpatialInspector
      coverage={coverage}
      track={track}
      mesh={mesh}
      reconstruction={reconstruction}
      parameters={parameters}
      setParameters={setParameters}
    />
  );
}
