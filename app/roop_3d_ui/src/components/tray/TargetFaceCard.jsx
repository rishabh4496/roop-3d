import React, { useState } from 'react';
import { FACE_ROLES } from '../canvas/FaceOverlayCanvas';
import { facesetDropProps } from '../../utils/facesetDrag';

/**
 * Target identity card in the horizontal face selection tray.
 * @param {{
 *   cluster: import('../../contracts').TargetCluster;
 *   sources?: import('../../contracts').FaceSet[];
 *   assignedSourceId?: string;
 *   isActive?: boolean;
 *   isSelected?: boolean;
 *   onSelect?: (clusterId: string) => void;
 *   onAssignSource?: (clusterId: string, sourceId: string | null) => void;
 *   onSetMode?: (clusterId: string, mode: 'swap' | 'ignore' | 'exclude') => void;
 *   onSeek?: (time: number) => void;
 * }} props
 */
export default function TargetFaceCard({
  cluster,
  sources = [],
  assignedSourceId,
  isActive = false,
  isSelected = false,
  onSelect,
  onAssignSource,
  onSetMode,
  onSeek,
}) {
  const currentMode = cluster?.mode || 'swap';
  const [dropOver, setDropOver] = useState(false);
  // Stage 7: drop a FaceSet from the identity panel onto this card (swap targets only, like the select below).
  const dropProps = currentMode === 'swap' && onAssignSource ? facesetDropProps(id => onAssignSource(cluster.id, id), setDropOver) : {};
  const assignedSource = sources.find(s => s.faceset_id === assignedSourceId);

  const yaw = cluster?.avgPose?.yaw ?? 0;
  const pitch = cluster?.avgPose?.pitch ?? 0;
  const yawFormatted = `${yaw >= 0 ? '+' : ''}${yaw.toFixed(1)}°`;
  const pitchFormatted = `${pitch >= 0 ? '+' : ''}${pitch.toFixed(1)}°`;

  const handleCardClick = () => {
    if (onSelect) onSelect(cluster.id);
  };

  return (
    <div
      className={`target-face-card ${isActive ? 'active-in-frame' : ''} ${isSelected ? 'selected' : ''} ${dropOver ? 'drop-over' : ''}`}
      onClick={handleCardClick}
      {...dropProps}
      data-cluster-id={cluster.id}
    >
      {/* On-screen badge indicator */}
      {isActive && <div className="card-live-indicator" title="Visible in current frame">● VISIBLE</div>}
      <span className={`face-role-badge role-${currentMode}`} style={{ color: FACE_ROLES[currentMode].color }}>{FACE_ROLES[currentMode].label}</span>

      <div className="card-top-row">
        {/* Thumbnail Crop */}
        <div className="card-thumb-container">
          {cluster.thumbnailUrl ? (
            <img
              src={cluster.thumbnailUrl}
              alt={cluster.name || cluster.id}
              className="card-thumb"
              loading="lazy"
            />
          ) : (
            <div className="card-thumb-fallback">◎</div>
          )}
          <span className="card-frame-badge">{cluster.frameCount}f</span>
        </div>

        {/* Identity & Pose Meta */}
        <div className="card-meta">
          <div className="card-title-row">
            <strong className="card-name">{cluster.name || cluster.id.replace('track_', 'Face ')}</strong>
            <span className="card-cluster-id">{cluster.id}</span>
          </div>

          <div className="card-pose-stats">
            <span title="Average Yaw">Yaw <strong>{yawFormatted}</strong></span>
            <span className="pose-sep">·</span>
            <span title="Average Pitch">Pitch <strong>{pitchFormatted}</strong></span>
          </div>

          <div className="card-frame-count">
            <span>{cluster.frameCount} detected frames</span>
          </div>
        </div>
      </div>

      {/* Assigned Source Face Indicator & Picker */}
      <div className="card-assignment-row" onClick={e => e.stopPropagation()}>
        <span className="assignment-label">Source</span>
        <select
          className={`assignment-select ${assignedSourceId ? 'has-source' : ''}`}
          aria-label={`Source face assignment for ${cluster.id}`}
          value={assignedSourceId || ''}
          disabled={currentMode !== 'swap'}
          onChange={e => onAssignSource(cluster.id, e.target.value || null)}
        >
          <option value="">Keep original</option>
          {sources.map(src => (
            <option key={src.faceset_id} value={src.faceset_id}>
              {src.name} ({src.item_count}v)
            </option>
          ))}
        </select>
      </div>

      {/* "Swap / Ignore / Exclude" Toggle Switch */}
      <div className="card-mode-toggle" onClick={e => e.stopPropagation()}>
        <button
          type="button"
          className={`mode-btn swap ${currentMode === 'swap' ? 'active' : ''}`}
          onClick={() => onSetMode(cluster.id, 'swap')}
          title="Swap this target face with source identity"
        >
          Swap
        </button>
        <button
          type="button"
          className={`mode-btn ignore ${currentMode === 'ignore' ? 'active' : ''}`}
          onClick={() => onSetMode(cluster.id, 'ignore')}
          title="Ignore / Pass-through original face"
        >
          Ignore
        </button>
        <button
          type="button"
          className={`mode-btn exclude ${currentMode === 'exclude' ? 'active' : ''}`}
          onClick={() => onSetMode(cluster.id, 'exclude')}
          title="Mask / Exclude this face from processing"
        >
          Exclude
        </button>
      </div>
    </div>
  );
}
