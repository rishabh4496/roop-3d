import React, { useMemo } from 'react';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import TargetFaceCard from './TargetFaceCard';

/**
 * Horizontal face selection tray positioned directly below the canvas.
 * Renders target identity cards and synchronizes live visibility during timeline scrubbing.
 * @param {{
 *   currentTime?: number;
 *   sources?: import('../../contracts').FaceSet[];
 *   fps?: number;
 *   onSeekTime?: (time: number) => void;
 *   onAssignSource?: (clusterId: string, sourceId: string | null) => void;
 * }} props
 */
export default function FaceSelectionTray({
  currentTime = 0,
  sources = [],
  fps = 30,
  onSeekTime,
  onAssignSource,
}) {
  const targetClusters = useFaceSelectionStore(s => s.targetClusters);
  const assignments = useFaceSelectionStore(s => s.assignments);
  const activeSelection = useFaceSelectionStore(s => s.activeSelection);
  const detectedFaces = useFaceSelectionStore(s => s.detectedFaces);

  const setActiveSelection = useFaceSelectionStore(s => s.setActiveSelection);
  const assignSource = useFaceSelectionStore(s => s.assignSource);
  const setClusterMode = useFaceSelectionStore(s => s.setClusterMode);
  const getFacesAtTimestamp = useFaceSelectionStore(s => s.getFacesAtTimestamp);

  // Determine which clusters are visible in the currently scrubbed frame/timestamp
  const activeClusterIds = useMemo(() => {
    if (!detectedFaces || detectedFaces.length === 0) return new Set();
    const currentFaces = getFacesAtTimestamp(currentTime, fps);
    return new Set(currentFaces.map(f => f.clusterId));
  }, [detectedFaces, currentTime, fps, getFacesAtTimestamp]);

  const handleSelectCluster = clusterId => {
    // Find representative face or first face for this cluster
    const cluster = targetClusters.find(c => c.id === clusterId);
    if (!cluster) return;

    const repFace = detectedFaces.find(f => f.id === cluster.representativeFaceId) ||
      detectedFaces.find(f => f.clusterId === clusterId);

    if (repFace) {
      setActiveSelection(repFace.id);
      if (repFace.timestamp !== undefined && onSeekTime) {
        onSeekTime(repFace.timestamp);
      }
    }
  };

  const handleAssign = (clusterId, facesetId) => {
    assignSource(clusterId, facesetId);
    if (onAssignSource) {
      onAssignSource(clusterId, facesetId);
    }
  };

  if (!targetClusters || targetClusters.length === 0) {
    return (
      <div className="face-selection-tray empty">
        <div className="tray-empty-hint">
          <span className="tray-symbol">◎</span>
          <span>No target faces detected yet. Inspect target faces to identify identities in the video.</span>
        </div>
      </div>
    );
  }

  return (
    <div className="face-selection-tray">
      <div className="tray-header">
        <div className="tray-title-group">
          <span className="eyebrow">TARGET IDENTITIES</span>
          <span className="tray-count-badge">{targetClusters.length} detected</span>
        </div>
        <div className="tray-scrub-status">
          <span className="muted">
            {activeClusterIds.size > 0
              ? `${activeClusterIds.size} visible in current frame`
              : 'No identities in current frame'}
          </span>
        </div>
      </div>

      <div className="tray-scroll-container">
        {targetClusters.map(cluster => {
          const isActive = activeClusterIds.has(cluster.id);
          const isSelected =
            cluster.faceIds.includes(activeSelection) ||
            cluster.representativeFaceId === activeSelection;

          return (
            <TargetFaceCard
              key={cluster.id}
              cluster={cluster}
              sources={sources}
              assignedSourceId={assignments[cluster.id]}
              isActive={isActive}
              isSelected={isSelected}
              onSelect={handleSelectCluster}
              onAssignSource={handleAssign}
              onSetMode={setClusterMode}
              onSeek={onSeekTime}
            />
          );
        })}
      </div>
    </div>
  );
}
