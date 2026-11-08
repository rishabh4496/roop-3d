import { useMemo } from 'react';
import type { Coverage, Track } from '../../contracts';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import { targetPoseSamples } from '../../utils/coverageRadar';
import AngularCoverageRadar from './AngularCoverageRadar';

/** The coverage radar for the active source FaceSet, with the selected target's poses drawn on it (red where the FaceSet has no nearby view). */
export default function CoverageRadarPanel({ coverage, track, compact = false }: { coverage: Coverage | null; track: Track | null; compact?: boolean }) {
  const detectedFaces = useFaceSelectionStore(state => state.detectedFaces);
  const activeSelection = useFaceSelectionStore(state => state.activeSelection);
  const targetPoses = useMemo(() => targetPoseSamples(track?.pose_range, (detectedFaces ?? []).filter(face => !track || face.clusterId === track.track_id).slice(0, 400)), [track, detectedFaces]);
  const livePose = useMemo(() => {
    const face = activeSelection ? detectedFaces?.find(item => item.id === activeSelection) : null;
    return face?.pose ? { yaw: face.pose.yaw ?? 0, pitch: -(face.pose.pitch ?? 0) } : null;   // stored pose is legacy (+ = down)
  }, [activeSelection, detectedFaces]);
  return <AngularCoverageRadar coverage={coverage} targetPoses={targetPoses} livePose={livePose} name={coverage?.faceset_id} compact={compact} />;
}
