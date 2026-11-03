import React from 'react';
import type { Coverage, Reconstruction, Track } from './contracts';
import type { WorkstationParameters } from './components/workstation/pipeline';
import PoseGeometryViewer from './components/three/PoseGeometryViewer';
import LightProbeSphere from './components/three/LightProbeSphere';
import FlameMeshViewer from './components/three/FlameMeshViewer';
import SpatialInspector from './components/studio/SpatialInspector';

export { default as PoseGeometryViewer } from './components/three/PoseGeometryViewer';
export { default as LightProbeSphere } from './components/three/LightProbeSphere';
export { default as FlameMeshViewer } from './components/three/FlameMeshViewer';
export { default as SpatialInspector } from './components/studio/SpatialInspector';

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
  parameters,
  setParameters,
}: {
  coverage: Coverage | null;
  track: Track | null;
  mesh: Reconstruction | null;
  parameters?: WorkstationParameters;
  setParameters?: (params: WorkstationParameters) => void;
}) {
  return (
    <SpatialInspector
      coverage={coverage}
      track={track}
      mesh={mesh}
      parameters={parameters}
      setParameters={setParameters}
    />
  );
}
