import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import { api } from '../../api';
import { drawPoseHud, facePose, LANDMARK_COLORS } from './poseHud';
import { leftToRightRank } from '../../utils/poseCells';
import { reducedMotion } from '../../utils/motionPrefs';
export const FACE_ROLES = { swap: { label: 'Swap Target', color: '#91e6c9' }, ignore: { label: 'Reference Guide', color: '#60a5fa' }, exclude: { label: 'Exclude (Never Swap)', color: '#fb7185' } };

import { calculateLetterbox, mapBoxToPixels, mapPointToPixels } from '../../utils/letterbox';
import { carriesFaceset, droppedFaceset } from '../../utils/facesetDrag';
export { calculateLetterbox, mapBoxToPixels, mapPointToPixels };

/**
 * Canvas Interactive Overlay for target face detection and selection.
 * @param {{
 *   videoRef?: React.RefObject<HTMLVideoElement | null>;
 *   mediaDimensions?: { width: number; height: number };
 *   currentTime?: number;
 *   sources?: import('../../contracts').FaceSet[];
 *   fps?: number;
 *   onAssignSource?: (clusterId: string, sourceId: string | null) => void;
 *   onExtractFaces?: () => void;
 *   detecting?: boolean;
 *   showOverlays?: boolean;
 *   showLandmarks?: boolean;
 *   showPose?: boolean;
 *   interactive?: boolean;
 *   frameFaces?: import('../../utils/frameFaces').MatchedFrameFace[] | null;
 *   frameMeta?: import('../../frameScrub').FrameMeta | null;
 *   onLockFace?: (face: import('../../utils/frameFaces').MatchedFrameFace) => void;
 *   lockingId?: string | null;
 *   onFaceContextMenu?: (menu: { face: any; faceRank: number; pose: { yaw: number; pitch: number; roll: number } | null; x: number; y: number }) => void;
 * }} props
 */
export default function FaceOverlayCanvas({
  videoRef,
  mediaDimensions,
  currentTime = 0,
  sources = [],
  fps = 30,
  onAssignSource,
  onExtractFaces,
  detecting = false,
  showOverlays = true,
  showLandmarks = true,
  showPose = true,
  interactive = true,
  onFaceContextMenu,
  frameFaces = null,
  frameMeta = null,
  onLockFace,
  lockingId = null,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  const detectedFaces = useFaceSelectionStore(s => s.detectedFaces);
  const targetClusters = useFaceSelectionStore(s => s.targetClusters);
  const assignments = useFaceSelectionStore(s => s.assignments);
  const activeSelection = useFaceSelectionStore(s => s.activeSelection);
  const hoveredFaceId = useFaceSelectionStore(s => s.hoveredFaceId);

  const setActiveSelection = useFaceSelectionStore(s => s.setActiveSelection);
  const setHoveredFace = useFaceSelectionStore(s => s.setHoveredFace);
  const assignSource = useFaceSelectionStore(s => s.assignSource);
  const setClusterMode = useFaceSelectionStore(s => s.setClusterMode);
  const getFacesAtTimestamp = useFaceSelectionStore(s => s.getFacesAtTimestamp);

  // Popover state
  const [popover, setPopover] = useState(null); // { x, y, w, h, face, cluster }

  // Measure letterbox dimensions
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  const updateSize = useCallback(() => {
    const parent = containerRef.current?.parentElement;
    if (parent) {
      const rect = parent.getBoundingClientRect();
      setContainerSize({ width: rect.width, height: rect.height });
    }
  }, []);

  useEffect(() => {
    updateSize();
    const handleResize = () => updateSize();
    window.addEventListener('resize', handleResize);
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(updateSize) : null;
    if (ro && containerRef.current?.parentElement) {
      ro.observe(containerRef.current.parentElement);
    }
    return () => {
      window.removeEventListener('resize', handleResize);
      ro?.disconnect();
    };
  }, [updateSize]);

  // Media natural dimensions
  const mediaW = frameMeta?.width || videoRef?.current?.videoWidth || mediaDimensions?.width || 1920;
  const mediaH = frameMeta?.height || videoRef?.current?.videoHeight || mediaDimensions?.height || 1080;

  const letterbox = useMemo(() => {
    return calculateLetterbox(containerSize.width, containerSize.height, mediaW, mediaH);
  }, [containerSize.width, containerSize.height, mediaW, mediaH]);

  // Faces visible on current frame/timestamp
  const visibleFaces = useMemo(() => {
    if (!detectedFaces || detectedFaces.length === 0) return [];
    return getFacesAtTimestamp(currentTime, fps);
  }, [detectedFaces, currentTime, fps, getFacesAtTimestamp]);

  // Per-frame detection: when the server has detected this exact frame's faces, those (at their real position) replace the
  // track boxes, which only know where a track first appeared. A detection that overlaps a stored face IS that face.
  const frameMode = Array.isArray(frameFaces) && !!frameMeta;
  const overlayFaces = useMemo(() => (frameMode ? frameFaces.map(match => toOverlayFace(match, frameMeta)) : []), [frameMode, frameFaces, frameMeta]);
  const interactiveFaces = frameMode ? overlayFaces : visibleFaces;

  // Close popover when active face is not in current frame
  useEffect(() => {
    if (popover && !interactiveFaces.some(f => f.id === popover.face.id)) {
      setPopover(null);
    }
  }, [interactiveFaces, popover]);

  // Draw overlay on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || containerSize.width === 0 || containerSize.height === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(containerSize.width * dpr);
    canvas.height = Math.round(containerSize.height * dpr);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, containerSize.width, containerSize.height);

    if (frameMode) {
      drawFrameFaces(ctx, overlayFaces, { letterbox, hoveredFaceId, lockingId, showLandmarks, showPose, bounds: containerSize });
      ctx.restore();
      return;
    }

    for (const face of visibleFaces) {
      const coords = mapBoxToPixels(face.bbox, letterbox);
      const isSelected = activeSelection === face.id;
      const isHovered = hoveredFaceId === face.id;
      const cluster = targetClusters.find(c => c.id === face.clusterId);
      const assignedSourceId = assignments[face.clusterId];
      const assignedSource = sources.find(s => s.faceset_id === assignedSourceId);

      const r = 4; // corner radius

      if (isHovered) {
        // --- Hovered: Smooth neon border with identity tag and confidence score ---
        ctx.save();
        ctx.shadowColor = '#00f5d4';
        ctx.shadowBlur = 14;
        ctx.strokeStyle = '#00f5d4';
        ctx.lineWidth = 2.5;

        // Draw rounded rectangle
        drawRoundedRect(ctx, coords.x, coords.y, coords.w, coords.h, r);
        ctx.stroke();

        // Neon fill hint
        ctx.fillStyle = 'rgba(0, 245, 212, 0.08)';
        ctx.fill();
        ctx.restore();

        // Identity tag & confidence badge
        const tagText = `${cluster?.name || face.clusterId} · ${Math.round((face.confidence ?? 0.95) * 100)}%`;
        ctx.save();
        ctx.font = '600 11px Inter, system-ui, sans-serif';
        const textMetrics = ctx.measureText(tagText);
        const tagW = textMetrics.width + 16;
        const tagH = 22;
        const tagX = Math.max(letterbox.offsetX, coords.x);
        const tagY = coords.y - tagH - 4 >= letterbox.offsetY ? coords.y - tagH - 4 : coords.y + 4;

        // Badge background
        ctx.fillStyle = 'rgba(10, 16, 18, 0.92)';
        ctx.strokeStyle = '#00f5d4';
        ctx.lineWidth = 1;
        drawRoundedRect(ctx, tagX, tagY, tagW, tagH, 3);
        ctx.fill();
        ctx.stroke();

        // Badge text
        ctx.fillStyle = '#00f5d4';
        ctx.fillText(tagText, tagX + 8, tagY + 15);
        ctx.restore();

      } else if (isSelected) {
        // --- Selected: Active accent bounding box with 5-point ArcFace landmark dots ---
        ctx.save();
        ctx.shadowColor = '#ffb703';
        ctx.shadowBlur = 12;
        ctx.strokeStyle = '#ffb703';
        ctx.lineWidth = 2.5;

        drawRoundedRect(ctx, coords.x, coords.y, coords.w, coords.h, r);
        ctx.stroke();

        // Accent corner brackets for tactical workstation feel
        const bracketLen = Math.min(14, coords.w / 4, coords.h / 4);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 3;
        drawCornerBrackets(ctx, coords.x, coords.y, coords.w, coords.h, bracketLen);

        ctx.restore();

        // 5-point ArcFace landmark dots
        const landmarks = face.landmarks5 || [
          [face.bbox[0] + (face.bbox[2] - face.bbox[0]) * 0.35, face.bbox[1] + (face.bbox[3] - face.bbox[1]) * 0.38],
          [face.bbox[0] + (face.bbox[2] - face.bbox[0]) * 0.65, face.bbox[1] + (face.bbox[3] - face.bbox[1]) * 0.38],
          [face.bbox[0] + (face.bbox[2] - face.bbox[0]) * 0.50, face.bbox[1] + (face.bbox[3] - face.bbox[1]) * 0.58],
          [face.bbox[0] + (face.bbox[2] - face.bbox[0]) * 0.38, face.bbox[1] + (face.bbox[3] - face.bbox[1]) * 0.75],
          [face.bbox[0] + (face.bbox[2] - face.bbox[0]) * 0.62, face.bbox[1] + (face.bbox[3] - face.bbox[1]) * 0.75],
        ];

        ctx.save();
        for (const pt of landmarks) {
          const lPixel = mapPointToPixels(pt, letterbox);
          // Glowing halo
          ctx.beginPath();
          ctx.arc(lPixel.x, lPixel.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#ffb703';
          ctx.shadowColor = '#ffb703';
          ctx.shadowBlur = 8;
          ctx.fill();

          // White center dot
          ctx.beginPath();
          ctx.arc(lPixel.x, lPixel.y, 1.8, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.shadowBlur = 0;
          ctx.fill();
        }
        ctx.restore();

        // Selection badge
        const selLabel = `${cluster?.name || face.clusterId}${assignedSource ? ` → ${assignedSource.name}` : ''}`;
        ctx.save();
        ctx.font = '600 10px Inter, system-ui, sans-serif';
        const metrics = ctx.measureText(selLabel);
        const bW = metrics.width + 14;
        const bH = 20;
        const bX = Math.max(letterbox.offsetX, coords.x);
        const bY = coords.y - bH - 4 >= letterbox.offsetY ? coords.y - bH - 4 : coords.y + 4;

        ctx.fillStyle = 'rgba(23, 30, 32, 0.95)';
        ctx.strokeStyle = '#ffb703';
        ctx.lineWidth = 1;
        drawRoundedRect(ctx, bX, bY, bW, bH, 3);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#ffb703';
        ctx.fillText(selLabel, bX + 7, bY + 14);
        ctx.restore();

      } else {
        // --- Normal / Inactive: Clean subtle workstation box ---
        ctx.save();
        ctx.strokeStyle = 'rgba(145, 230, 201, 0.45)';
        ctx.lineWidth = 1.5;
        drawRoundedRect(ctx, coords.x, coords.y, coords.w, coords.h, r);
        ctx.stroke();

        // Corner ticks
        const tick = Math.min(8, coords.w / 5, coords.h / 5);
        ctx.strokeStyle = 'rgba(145, 230, 201, 0.8)';
        ctx.lineWidth = 2;
        drawCornerBrackets(ctx, coords.x, coords.y, coords.w, coords.h, tick);

        // Tiny identity label
        ctx.font = '500 9px monospace';
        ctx.fillStyle = 'rgba(145, 230, 201, 0.75)';
        const label = cluster?.name || face.clusterId.replace('track_', 'Face ');
        ctx.fillText(label, coords.x + 4, coords.y + 12);
        ctx.restore();
      }
      const role = FACE_ROLES[cluster?.mode ?? 'swap'];
      ctx.save(); ctx.font = '600 10px system-ui';
      const badgeWidth = Math.min(containerSize.width, ctx.measureText(role.label).width + 12);
      const badgeX = Math.max(0, Math.min(coords.x, containerSize.width - badgeWidth));
      const badgeY = Math.min(containerSize.height - 18, coords.y + coords.h + 2);
      ctx.fillStyle = '#101719'; ctx.fillRect(badgeX, badgeY, badgeWidth, 18);
      ctx.fillStyle = role.color; ctx.fillText(role.label, badgeX + 6, badgeY + 12); ctx.restore();
    }

    // 3D pose & landmark HUD: 5-point ArcFace dots and a live yaw/pitch/roll badge on every detected face.
    drawPoseHud(ctx, visibleFaces, {
      mapPoint: point => mapPointToPixels(point, letterbox),
      mapBox: box => mapBoxToPixels(box, letterbox),
      bounds: { offsetX: letterbox.offsetX, offsetY: letterbox.offsetY, width: letterbox.displayW, height: letterbox.displayH },
      landmarks: showLandmarks,
      pose: showPose,
    });

    ctx.restore();
  }, [
    visibleFaces,
    containerSize,
    letterbox,
    activeSelection,
    hoveredFaceId,
    targetClusters,
    assignments,
    sources,
    showLandmarks,
    showPose,
    frameMode,
    overlayFaces,
    lockingId,
  ]);

  // Mouse move: Hit testing for hover
  const handleMouseMove = useCallback(
    e => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let hit = null;
      for (const face of interactiveFaces) {
        const coords = mapBoxToPixels(face.bbox, letterbox);
        if (
          mouseX >= coords.x - 4 &&
          mouseX <= coords.x + coords.w + 4 &&
          mouseY >= coords.y - 4 &&
          mouseY <= coords.y + coords.h + 4
        ) {
          hit = face;
          break;
        }
      }

      if (hit) {
        setHoveredFace(hit.id);
        canvas.style.cursor = 'pointer';
      } else {
        setHoveredFace(null);
        canvas.style.cursor = 'default';
      }
    },
    [interactiveFaces, letterbox, setHoveredFace]
  );

  const handleMouseLeave = useCallback(() => {
    setHoveredFace(null);
    if (canvasRef.current) {
      canvasRef.current.style.cursor = 'default';
    }
  }, [setHoveredFace]);

  // Click: Select and open target-to-source face assignment popover
  const handleClick = useCallback(
    e => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let hit = null;
      for (const face of interactiveFaces) {
        const coords = mapBoxToPixels(face.bbox, letterbox);
        if (
          mouseX >= coords.x - 4 &&
          mouseX <= coords.x + coords.w + 4 &&
          mouseY >= coords.y - 4 &&
          mouseY <= coords.y + coords.h + 4
        ) {
          hit = face;
          break;
        }
      }

      if (hit?.candidate) {
        // A detected face that is not a target yet: clicking it locks it (the server creates its track).
        setPopover(null);
        onLockFace?.(frameFaces.find(match => match.id === hit.id));
        return;
      }

      if (hit) {
        setActiveSelection(hit.id);
        const coords = mapBoxToPixels(hit.bbox, letterbox);
        const cluster = targetClusters.find(c => c.id === hit.clusterId);
        setPopover({
          x: coords.x,
          y: coords.y,
          w: coords.w,
          h: coords.h,
          face: hit,
          cluster,
        });
      } else {
        setPopover(null);
      }
    },
    [interactiveFaces, letterbox, targetClusters, setActiveSelection, onLockFace, frameFaces]
  );

  // Close popover on Escape
  useEffect(() => {
    const onKey = e => {
      if (e.key === 'Escape') setPopover(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const handleSelectSource = (clusterId, facesetId) => {
    assignSource(clusterId, facesetId || null);
    if (onAssignSource) {
      onAssignSource(clusterId, facesetId || null);
    }
  };

  if (!showOverlays) return null;

  return (
    <div
      ref={containerRef}
      className="face-overlay-container"
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 5,
        overflow: 'hidden',
      }}
    >
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        onContextMenu={event => {
          event.preventDefault();
          const rect = event.currentTarget.getBoundingClientRect();
          const hit = interactiveFaces.find(face => {
            const box = mapBoxToPixels(face.bbox, letterbox);
            const x = event.clientX - rect.left, y = event.clientY - rect.top;
            return x >= box.x - 4 && x <= box.x + box.w + 4 && y >= box.y - 4 && y <= box.y + box.h + 4;
          });
          if (hit && onFaceContextMenu) {
            setActiveSelection(hit.id);
            setPopover(null);
            onFaceContextMenu({ face: hit, faceRank: leftToRightRank(interactiveFaces, hit), pose: facePose(hit), x: event.clientX, y: event.clientY });
          } else {
            handleClick(event);
          }
        }}
        onDragOver={event => {
          // Stage 7: a FaceSet dragged from the identity panel can be dropped straight onto a face in the video.
          if (!carriesFaceset(event)) return;
          const rect = event.currentTarget.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
          if (interactiveFaces.some(face => { if (face.candidate) return false; const box = mapBoxToPixels(face.bbox, letterbox); return x >= box.x - 4 && x <= box.x + box.w + 4 && y >= box.y - 4 && y <= box.y + box.h + 4; })) {
            event.preventDefault(); event.dataTransfer.dropEffect = 'copy';
          }
        }}
        onDrop={event => {
          const facesetId = droppedFaceset(event);
          if (!facesetId) return;
          const rect = event.currentTarget.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
          const hit = interactiveFaces.find(face => { if (face.candidate) return false; const box = mapBoxToPixels(face.bbox, letterbox); return x >= box.x - 4 && x <= box.x + box.w + 4 && y >= box.y - 4 && y <= box.y + box.h + 4; });
          if (!hit) return;
          event.preventDefault();
          setActiveSelection(hit.id);
          handleSelectSource(hit.clusterId, facesetId);
        }}
        tabIndex={0}
        aria-label="Detected face bounding boxes"
        onKeyDown={event => { if (event.key !== 'ContextMenu' && !(event.shiftKey && event.key === 'F10')) return; const face = interactiveFaces.find(face => face.id === activeSelection) ?? interactiveFaces[0]; if (!face) return; event.preventDefault(); const coords = mapBoxToPixels(face.bbox, letterbox); setPopover({ ...coords, face, cluster: targetClusters.find(cluster => cluster.id === face.clusterId) }); }}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
          pointerEvents: interactive ? 'auto' : 'none',
        }}
      />

      {/* Quick target face selector strip on top of preview box */}
      {!frameMode && interactiveFaces.length > 1 && (
        <div className="preview-face-selector-strip" role="toolbar" aria-label="Select target face from preview">
          <span className="strip-title">🎯 TARGET FACE:</span>
          {interactiveFaces.map((f, i) => {
            const cluster = targetClusters.find(c => c.id === f.clusterId);
            const isSel = activeSelection === f.id;
            const assigned = sources.find(s => s.faceset_id === assignments[f.clusterId]);
            return (
              <button
                key={f.id}
                type="button"
                className={`preview-face-pill ${isSel ? 'active' : ''}`}
                onClick={e => {
                  e.stopPropagation();
                  setActiveSelection(f.id);
                  const coords = mapBoxToPixels(f.bbox, letterbox);
                  setPopover({
                    x: coords.x,
                    y: coords.y,
                    w: coords.w,
                    h: coords.h,
                    face: f,
                    cluster,
                  });
                  if (typeof document !== 'undefined') {
                    const cardEl = document.querySelector(`[data-cluster-id="${f.clusterId}"]`);
                    cardEl?.scrollIntoView?.({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'nearest', inline: 'center' });
                  }
                }}
                aria-pressed={isSel}
              >
                <span className="pill-status" />
                <span className="pill-name">{cluster?.name || `Face ${i + 1}`}</span>
                {assigned && <span className="pill-mapping">→ {assigned.name}</span>}
                {isSel && <span className="pill-check">✓</span>}
              </button>
            );
          })}
        </div>
      )}

      {/* Floating 1-click face detection prompt when no faces detected on current preview frame */}
      {interactiveFaces.length === 0 && onExtractFaces && (
        <div className="preview-detect-prompt">
          <button
            type="button"
            className={`preview-detect-btn ${detecting ? 'detecting' : ''}`}
            onClick={e => {
              e.stopPropagation();
              onExtractFaces();
            }}
            disabled={detecting}
            title="Extract and select target faces directly from this preview frame"
          >
            <span className="detect-icon">🎯</span>
            <span>{detecting ? 'Detecting faces in frame…' : 'Use Face from this Frame'}</span>
          </button>
        </div>
      )}

      {/* Context Action: Fast target-to-source face assignment popover */}
      {popover && (
        <FaceAssignmentPopover
          popover={{ ...popover, cluster: targetClusters.find(cluster => cluster.id === popover.face.clusterId) }}
          containerWidth={containerSize.width}
          containerHeight={containerSize.height}
          sources={sources}
          assignedSourceId={assignments[popover.face.clusterId]}
          onSelectSource={facesetId => handleSelectSource(popover.face.clusterId, facesetId)}
          onSetMode={mode => setClusterMode(popover.face.clusterId, mode)}
          onRecenter={() => { useFaceSelectionStore.getState().focusPose(popover.face.id); setPopover(null); }}
          onSplit={() => { useFaceSelectionStore.getState().splitCluster(popover.face.clusterId, Math.round(currentTime * fps)); setPopover(null); }}
          onSourceView={index => useFaceSelectionStore.getState().setSourceView(popover.face.clusterId, index)}
          canSplit={Math.round(currentTime * fps) > (targetClusters.find(cluster => cluster.id === popover.face.clusterId)?.startFrame ?? 0)}
          onClose={() => setPopover(null)}
        />
      )}
    </div>
  );
}

/**
 * Fast assignment popover directly anchored to the bounding box.
 */
function FaceAssignmentPopover({
  popover,
  containerWidth,
  containerHeight,
  sources,
  assignedSourceId,
  onSelectSource,
  onSetMode,
  onClose,
  onRecenter, onSplit, onSourceView, canSplit,
}) {
  const [views, setViews] = useState([]), [error, setError] = useState('');
  const menu = useRef(null);
  useEffect(() => { menu.current?.querySelector('button')?.focus(); }, []);
  useEffect(() => { setViews([]); setError(''); if (!assignedSourceId) return; const controller = new AbortController(); void api(`/facesets/${encodeURIComponent(assignedSourceId)}/views`, { signal: controller.signal }).then(setViews).catch(error => { if (error.name !== 'AbortError') setError(error.message); }); return () => controller.abort(); }, [assignedSourceId]);
  const popoverW = 260;
  const popoverH = 240;

  // Compute anchored position: prefer bottom of box, flip to top if needed
  let left = Math.max(12, Math.min(popover.x, containerWidth - popoverW - 12));
  let top = popover.y + popover.h + 8;
  if (top + popoverH > containerHeight - 10) {
    top = Math.max(10, popover.y - popoverH - 8);
  }

  const cluster = popover.cluster;
  const currentMode = cluster?.mode || 'swap';

  return (
    <div
      ref={menu} role="menu" aria-label="Face context menu"
      onKeyDown={event => { if (!['ArrowDown', 'ArrowUp'].includes(event.key)) return; event.preventDefault(); const items = [...menu.current.querySelectorAll('button:not(:disabled),select:not(:disabled)')]; const index = items.indexOf(document.activeElement); items[(index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length]?.focus(); }}
      className="face-assignment-popover"
      style={{
        position: 'absolute',
        left: `${left}px`,
        top: `${top}px`,
        width: `${popoverW}px`,
        maxHeight: `${popoverH}px`,
        pointerEvents: 'auto',
      }}
      onClick={e => e.stopPropagation()}
    >
      <div className="popover-header">
        <div>
          <span className="popover-eyebrow">TARGET IDENTITY</span>
          <strong>{cluster?.name || popover.face.clusterId}</strong>
        </div>
        <button className="popover-close" onClick={onClose} aria-label="Close popover">
          ×
        </button>
      </div>

      {/* Mode switch */}
      <div className="popover-mode-switch">
        {['swap', 'ignore', 'exclude'].map(m => (
          <button
            key={m}
            className={`popover-mode-btn ${currentMode === m ? 'active ' + m : ''}`}
            onClick={() => onSetMode(m)}
          >
            {FACE_ROLES[m].label}
          </button>
        ))}
      </div>
      <div className="face-context-actions"><button className="button small" role="menuitem" onClick={onRecenter}>Re-center pose camera</button><button className="button small" role="menuitem" disabled={!canSplit} onClick={onSplit}>Split cluster from current frame</button></div>
      <label className="source-view-choice">Single source face<select aria-label="Single source face" disabled={currentMode !== 'swap' || !assignedSourceId || !views.length} value={popover.cluster?.sourceViewIndex ?? ''} onChange={event => onSourceView(event.target.value === '' ? null : Number(event.target.value))}><option value="">Auto-select FaceSet view</option>{views.map(view => <option key={view.index} value={view.index}>Face {view.index} · yaw {view.yaw.toFixed(1)}°</option>)}</select></label>{error && <small role="alert">{error}</small>}

      <div className="popover-source-heading">
        <span>ASSIGN SOURCE IDENTITY</span>
        {assignedSourceId ? <span className="assigned-tag">Mapped</span> : <span className="muted">None</span>}
      </div>

      <div className="popover-source-list">
        <button
          className={`popover-source-item ${!assignedSourceId ? 'active' : ''}`}
          onClick={() => onSelectSource(null)}
        >
          <span className="source-avatar-mini">⊘</span>
          <div className="source-meta">
            <strong>Keep original</strong>
            <small>Do not modify this face</small>
          </div>
          {!assignedSourceId && <span className="check-mark">✓</span>}
        </button>

        {sources.map((src, idx) => {
          const isSelected = assignedSourceId === src.faceset_id;
          return (
            <button
              key={src.faceset_id}
              disabled={currentMode !== 'swap'}
              className={`popover-source-item ${isSelected ? 'active' : ''}`}
              onClick={() => onSelectSource(src.faceset_id)}
            >
              <span className="source-avatar-mini">{(idx + 1).toString().padStart(2, '0')}</span>
              <div className="source-meta">
                <strong>{src.name}</strong>
                <small>{src.item_count} views</small>
              </div>
              {isSelected && <span className="check-mark">✓</span>}
            </button>
          );
        })}

        {sources.length === 0 && (
          <p className="popover-empty-notice">No .fsz FaceSets loaded. Add a source identity to map.</p>
        )}
      </div>
    </div>
  );
}

// Helpers for canvas geometry
function drawRoundedRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawCornerBrackets(ctx, x, y, width, height, length) {
  ctx.beginPath();
  // Top-left
  ctx.moveTo(x, y + length);
  ctx.lineTo(x, y);
  ctx.lineTo(x + length, y);
  // Top-right
  ctx.moveTo(x + width - length, y);
  ctx.lineTo(x + width, y);
  ctx.lineTo(x + width, y + length);
  // Bottom-right
  ctx.moveTo(x + width, y + height - length);
  ctx.lineTo(x + width, y + height);
  ctx.lineTo(x + width - length, y + height);
  // Bottom-left
  ctx.moveTo(x + length, y + height);
  ctx.lineTo(x, y + height);
  ctx.lineTo(x, y + height - length);
  ctx.stroke();
}

/** A detected frame face as the overlay's face shape: normalised box and landmarks, both pose frames. */
function toOverlayFace(match, meta) {
  const { face } = match;
  return {
    id: match.id, clusterId: match.clusterId, frameIndex: meta.frame_index, bbox: face.bbox_norm, landmarks5: match.landmarks5,
    pose: face.pose, pose3d: face.pose3d, confidence: face.confidence ?? undefined,
    candidate: !match.stored, locked: match.locked, rank: face.face_index,
  };
}

const FRAME_CYAN = '#22d3ee', FRAME_LOCKED = '#34d399';
const signed = value => `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(0)}`;

function pill(ctx, text, x, y, { fill, stroke, color, font = '600 10px ui-monospace, Menlo, Consolas, monospace' }) {
  ctx.save();
  ctx.font = font;
  const width = ctx.measureText(text).width + 10, height = 16;
  ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = 1;
  drawRoundedRect(ctx, x, y, width, height, 3); ctx.fill(); ctx.stroke();
  ctx.fillStyle = color; ctx.fillText(text, x + 5, y + 11.5);
  ctx.restore();
  return width;
}

/**
 * Per-frame face boxes. Unselected: dashed cyan with its `#n` badge and a yaw / pitch pill. Hovered: glow and the 5-point
 * landmarks. Selected target: solid emerald with a TARGET LOCKED badge. `#n` is the left-to-right rank face extraction uses.
 */
function drawFrameFaces(ctx, faces, { letterbox, hoveredFaceId, lockingId, showLandmarks, showPose, bounds }) {
  for (const face of faces) {
    const box = mapBoxToPixels(face.bbox, letterbox);
    const locked = face.locked, hovered = hoveredFaceId === face.id && !locked, locking = lockingId === face.id;
    const accent = locked ? FRAME_LOCKED : FRAME_CYAN;

    ctx.save();
    ctx.strokeStyle = accent;
    ctx.lineWidth = locked || hovered ? 2.5 : 1.5;
    if (locked || hovered) { ctx.shadowColor = accent; ctx.shadowBlur = hovered ? 16 : 12; }
    ctx.setLineDash(locked || hovered ? [] : [6, 4]);
    drawRoundedRect(ctx, box.x, box.y, box.w, box.h, 4);
    ctx.stroke();
    if (locked || hovered) { ctx.shadowBlur = 0; ctx.fillStyle = locked ? 'rgba(52, 211, 153, 0.08)' : 'rgba(34, 211, 238, 0.10)'; ctx.fill(); }
    ctx.restore();

    // `#n` badge, then TARGET LOCKED / Locking… beside it; above the box unless that leaves the frame.
    const top = box.y - 20 >= letterbox.offsetY ? box.y - 20 : box.y + 4;
    const badge = pill(ctx, `#${face.rank + 1}`, box.x, top, { fill: 'rgba(8, 14, 16, 0.92)', stroke: accent, color: accent });
    if (locked) pill(ctx, 'TARGET LOCKED', box.x + badge + 4, top, { fill: FRAME_LOCKED, stroke: FRAME_LOCKED, color: '#04130d' });
    else if (locking) pill(ctx, 'Locking…', box.x + badge + 4, top, { fill: 'rgba(8, 14, 16, 0.92)', stroke: accent, color: '#e8f1f2' });

    if (showPose) {
      const attitude = facePose(face);
      if (attitude) {
        const text = `Y ${signed(attitude.yaw)}° P ${signed(attitude.pitch)}°`;
        ctx.save(); ctx.font = '600 10px ui-monospace, Menlo, Consolas, monospace';
        const width = ctx.measureText(text).width + 10;
        ctx.restore();
        const left = Math.max(letterbox.offsetX, Math.min(box.x + box.w - width, letterbox.offsetX + letterbox.displayW - width));
        const below = Math.min(bounds.height - 18, box.y + box.h + 3);
        pill(ctx, text, left, below, { fill: 'rgba(8, 14, 16, 0.88)', stroke: 'rgba(232, 241, 242, 0.25)', color: '#e8f1f2' });
      }
    }

    if ((hovered || locked || showLandmarks) && Array.isArray(face.landmarks5)) {
      ctx.save();
      face.landmarks5.forEach((point, index) => {
        const { x, y } = mapPointToPixels(point, letterbox);
        ctx.beginPath(); ctx.arc(x, y, hovered || locked ? 3.5 : 2.5, 0, Math.PI * 2);
        ctx.fillStyle = LANDMARK_COLORS[index]; ctx.fill();
        ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(0,0,0,0.65)'; ctx.stroke();
      });
      ctx.restore();
    }
  }
}
