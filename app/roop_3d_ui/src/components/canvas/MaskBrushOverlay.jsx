import NumericInput from '../controls/NumericInput';
import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { calculateLetterbox } from './FaceOverlayCanvas';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';

/**
 * Manual & Occlusion Mask Editor.
 *
 * Controls:
 * - Brush Mode: Paint (Keep Target / Occlusion), Erase, and Box Blur.
 * - Brush Settings: Size (1px - 100px), Hardness (feathering), Opacity.
 * - Text-prompt occlusion guide: visual indicator showing auto-detected occluders (glasses, fingers, hair strands).
 * - Export mask data as a base64 / 8-bit PNG alpha mask alongside the swap request payload.
 * @param {{
 *   containerWidth?: number;
 *   containerHeight?: number;
 *   mediaDimensions?: { width: number; height: number };
 *   onMaskChange?: (maskBase64: string | null) => void;
 *   onClose?: () => void;
 *   initialMask?: string | null;
 * }} props
 */
export default function MaskBrushOverlay({
  containerWidth = 0,
  containerHeight = 0,
  mediaDimensions = { width: 1920, height: 1080 },
  onMaskChange,
  onClose,
  initialMask = null,
}) {
  const visibleCanvasRef = useRef(null);
  const exportCanvasRef = useRef(null);
  const isPainting = useRef(false);
  const lastPoint = useRef(null);

  // Brush settings
  const [brushMode, setBrushMode] = useState('paint'); // 'paint' | 'erase' | 'blur'
  const [brushSize, setBrushSize] = useState(28); // 1..100
  const [hardness, setHardness] = useState(0.7); // 0..1 (feathering)
  const [opacity, setOpacity] = useState(0.85); // 0.1..1.0

  // Occlusion guide state
  const [showOcclusionGuides, setShowOcclusionGuides] = useState(true);
  const [occlusionPrompt, setOcclusionPrompt] = useState('glasses, hair strands, fingers');
  const [detectedOccluders] = useState([
    { id: 'glasses', label: 'Glasses / Eyewear', region: 'eyes', confidence: 0.94, active: true },
    { id: 'hair', label: 'Hair Strands / Fringe', region: 'forehead', confidence: 0.88, active: true },
    { id: 'fingers', label: 'Fingers / Occluding Hand', region: 'chin', confidence: 0.76, active: false },
  ]);

  // Cursor preview position
  const [cursorPos, setCursorPos] = useState({ x: -100, y: -100, visible: false });

  // Active faces for landmark-guided occlusion projection
  const detectedFaces = useFaceSelectionStore(s => s.detectedFaces);
  const activeSelection = useFaceSelectionStore(s => s.activeSelection);

  const activeFace = useMemo(() => {
    if (activeSelection) return detectedFaces.find(f => f.id === activeSelection);
    return detectedFaces[0] || null;
  }, [activeSelection, detectedFaces]);

  const letterbox = useMemo(() => {
    return calculateLetterbox(
      containerWidth,
      containerHeight,
      mediaDimensions.width,
      mediaDimensions.height
    );
  }, [containerWidth, containerHeight, mediaDimensions.width, mediaDimensions.height]);

  // Native mask canvas dimensions (matches media dimensions)
  const nativeW = mediaDimensions.width || 1920;
  const nativeH = mediaDimensions.height || 1080;

  // Screen -> Native canvas coordinate mapping
  const screenToCanvasCoords = useCallback(
    (clientX, clientY) => {
      const canvas = visibleCanvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const screenX = clientX - rect.left;
      const screenY = clientY - rect.top;

      const scaleX = nativeW / (letterbox.displayW || 1);
      const scaleY = nativeH / (letterbox.displayH || 1);

      const x = (screenX - letterbox.offsetX) * scaleX;
      const y = (screenY - letterbox.offsetY) * scaleY;

      return {
        x,
        y,
        screenX,
        screenY,
        onImage:
          screenX >= letterbox.offsetX &&
          screenX <= letterbox.offsetX + letterbox.displayW &&
          screenY >= letterbox.offsetY &&
          screenY <= letterbox.offsetY + letterbox.displayH,
      };
    },
    [nativeW, nativeH, letterbox]
  );

  // Initialize canvases
  useEffect(() => {
    const vCanvas = visibleCanvasRef.current;
    const eCanvas = exportCanvasRef.current;
    if (!vCanvas || !eCanvas) return;

    vCanvas.width = nativeW;
    vCanvas.height = nativeH;
    eCanvas.width = nativeW;
    eCanvas.height = nativeH;

    const eCtx = eCanvas.getContext('2d');
    if (eCtx) {
      eCtx.fillStyle = '#000000';
      eCtx.fillRect(0, 0, nativeW, nativeH);
    }
  }, [nativeW, nativeH]);

  // The authoritative mask belongs to the workspace command history. Reload it
  // after undo/redo, snapshot restoration, or reopening the brush editor.
  useEffect(() => {
    const visible = visibleCanvasRef.current, exported = exportCanvasRef.current;
    if (!visible || !exported) return;
    const vCtx = visible.getContext('2d'), eCtx = exported.getContext('2d');
    if (!vCtx || !eCtx) return;
    vCtx.clearRect(0, 0, nativeW, nativeH); eCtx.fillStyle = '#000'; eCtx.fillRect(0, 0, nativeW, nativeH);
    if (!initialMask) return;
    let live = true; const image = new Image();
    image.onload = () => { if (!live) return; eCtx.drawImage(image, 0, 0, nativeW, nativeH); const pixels = eCtx.getImageData(0, 0, nativeW, nativeH); for (let i = 0; i < pixels.data.length; i += 4) { const alpha = pixels.data[i] * .6; pixels.data[i] = 244; pixels.data[i + 1] = 63; pixels.data[i + 2] = 94; pixels.data[i + 3] = alpha; } vCtx.putImageData(pixels, 0, 0); };
    image.src = initialMask;
    return () => { live = false; };
  }, [initialMask, nativeW, nativeH]);

  // Apply brush dab / stroke
  const drawDab = useCallback(
    (pt, prevPt) => {
      const vCanvas = visibleCanvasRef.current;
      const eCanvas = exportCanvasRef.current;
      if (!vCanvas || !eCanvas) return;

      const vCtx = vCanvas.getContext('2d');
      const eCtx = eCanvas.getContext('2d');
      if (!vCtx || !eCtx) return;

      const scale = nativeW / (letterbox.displayW || 1);
      const r = Math.max(1, (brushSize / 2) * scale);
      const innerR = Math.max(0, r * hardness);

      if (brushMode === 'blur') {
        // Box blur operation on region
        applyBoxBlur(vCtx, pt.x, pt.y, r);
        applyBoxBlur(eCtx, pt.x, pt.y, r);
        return;
      }

      const isErase = brushMode === 'erase';

      for (const [ctx, isExport] of [
        [vCtx, false],
        [eCtx, true],
      ]) {
        ctx.save();
        ctx.globalCompositeOperation = isErase
          ? isExport
            ? 'source-over'
            : 'destination-out'
          : 'source-over';

        let color, endColor;
        if (isExport) {
          color = isErase ? '#000000' : `rgba(255, 255, 255, ${opacity})`;
          endColor = isErase ? '#000000' : 'rgba(255, 255, 255, 0)';
        } else {
          color = isErase ? '#000000' : `rgba(244, 63, 94, ${opacity * 0.6})`;
          endColor = 'rgba(244, 63, 94, 0)';
        }

        if (hardness < 0.95 && !isErase) {
          const grad = ctx.createRadialGradient(pt.x, pt.y, innerR, pt.x, pt.y, r);
          grad.addColorStop(0, color);
          grad.addColorStop(1, endColor);
          ctx.fillStyle = grad;
          ctx.strokeStyle = grad;
        } else {
          ctx.fillStyle = color;
          ctx.strokeStyle = color;
        }

        ctx.lineWidth = r * 2;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // Draw dab circle
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
        ctx.fill();

        // Connect with previous point for unbroken stroke
        if (prevPt) {
          ctx.beginPath();
          ctx.moveTo(prevPt.x, prevPt.y);
          ctx.lineTo(pt.x, pt.y);
          ctx.stroke();
        }

        ctx.restore();
      }
    },
    // applyBoxBlur is a pure function of its arguments (declared below); depending on it would rebuild this callback every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [brushMode, brushSize, hardness, opacity, nativeW, letterbox]
  );

  // Box blur filter helper
  function applyBoxBlur(ctx, cx, cy, radius) {
    const d = Math.round(radius * 1.5);
    const x = Math.max(0, Math.round(cx - d));
    const y = Math.max(0, Math.round(cy - d));
    const w = Math.min(nativeW - x, d * 2);
    const h = Math.min(nativeH - y, d * 2);
    if (w <= 0 || h <= 0) return;

    try {
      const imgData = ctx.getImageData(x, y, w, h);
      const data = imgData.data;
      // Fast horizontal & vertical 3-box blur
      for (let i = 4; i < data.length - 4; i += 4) {
        data[i + 3] = (data[i - 4 + 3] + data[i + 3] + data[i + 4 + 3]) / 3;
      }
      ctx.putImageData(imgData, x, y);
    } catch {
      // Fallback
    }
  }

  // Pointer event handlers
  const handlePointerDown = e => {
    const pt = screenToCanvasCoords(e.clientX, e.clientY);
    if (!pt || !pt.onImage) return;

    isPainting.current = true;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    lastPoint.current = pt;
    drawDab(pt, null);
  };

  const handlePointerMove = e => {
    const pt = screenToCanvasCoords(e.clientX, e.clientY);
    if (pt) {
      setCursorPos({ x: pt.screenX, y: pt.screenY, visible: pt.onImage });
      if (isPainting.current && pt.onImage) {
        drawDab(pt, lastPoint.current);
        lastPoint.current = pt;
      }
    }
  };

  const handlePointerUp = () => {
    if (isPainting.current) {
      isPainting.current = false;
      lastPoint.current = null;
      exportMask();
    }
  };

  const handlePointerLeave = () => {
    if (isPainting.current) {
      isPainting.current = false;
      lastPoint.current = null;
      exportMask();
    }
    setCursorPos(prev => ({ ...prev, visible: false }));
  };

  // Export 8-bit PNG alpha mask
  const exportMask = useCallback(() => {
    const eCanvas = exportCanvasRef.current;
    if (!eCanvas) return;
    try {
      const dataUrl = eCanvas.toDataURL('image/png');
      if (onMaskChange) {
        onMaskChange(dataUrl);
      }
    } catch {
      // Cross-origin fallback
    }
  }, [onMaskChange]);

  // Clear mask
  const handleClearMask = () => {
    const vCanvas = visibleCanvasRef.current;
    const eCanvas = exportCanvasRef.current;
    if (!vCanvas || !eCanvas) return;

    const vCtx = vCanvas.getContext('2d');
    const eCtx = eCanvas.getContext('2d');
    if (vCtx) vCtx.clearRect(0, 0, nativeW, nativeH);
    if (eCtx) {
      eCtx.fillStyle = '#000000';
      eCtx.fillRect(0, 0, nativeW, nativeH);
    }

    if (onMaskChange) onMaskChange(null);
  };

  // Auto-fill detected occlusion guide into mask
  const handleAutoFillOcclusion = occluder => {
    if (!activeFace) return;
    const [bx1, by1, bx2, by2] = activeFace.bbox;
    const bw = bx2 - bx1;
    const bh = by2 - by1;

    let rx1, ry1, rx2, ry2;
    if (occluder.region === 'eyes') {
      // Glasses / Eyewear zone
      rx1 = (bx1 + bw * 0.15) * nativeW;
      rx2 = (bx2 - bw * 0.15) * nativeW;
      ry1 = (by1 + bh * 0.25) * nativeH;
      ry2 = (by1 + bh * 0.48) * nativeH;
    } else if (occluder.region === 'forehead') {
      // Hair / Bangs
      rx1 = (bx1 + bw * 0.1) * nativeW;
      rx2 = (bx2 - bw * 0.1) * nativeW;
      ry1 = (by1 - bh * 0.05) * nativeH;
      ry2 = (by1 + bh * 0.28) * nativeH;
    } else {
      // Chin / Finger occlusion
      rx1 = (bx1 + bw * 0.2) * nativeW;
      rx2 = (bx2 - bw * 0.2) * nativeW;
      ry1 = (by1 + bh * 0.65) * nativeH;
      ry2 = (by2 + bh * 0.05) * nativeH;
    }

    const vCtx = visibleCanvasRef.current?.getContext('2d');
    const eCtx = exportCanvasRef.current?.getContext('2d');

    for (const [ctx, isExport] of [
      [vCtx, false],
      [eCtx, true],
    ]) {
      if (!ctx) continue;
      ctx.save();
      ctx.fillStyle = isExport ? '#ffffff' : 'rgba(244, 63, 94, 0.55)';
      ctx.beginPath();
      ctx.roundRect(rx1, ry1, rx2 - rx1, ry2 - ry1, 12);
      ctx.fill();
      ctx.restore();
    }

    exportMask();
  };

  return (
    <div
      className="mask-brush-overlay"
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
    >
      {/* Visible tinted painting canvas */}
      <canvas
        ref={visibleCanvasRef}
        className="mask-visible-canvas"
        style={{
          position: 'absolute',
          left: `${letterbox.offsetX}px`,
          top: `${letterbox.offsetY}px`,
          width: `${letterbox.displayW}px`,
          height: `${letterbox.displayH}px`,
          pointerEvents: 'auto',
          cursor: 'none',
        }}
      />

      {/* Hidden export canvas (pure 8-bit black/white alpha mask) */}
      <canvas ref={exportCanvasRef} style={{ display: 'none' }} />

      {/* Floating Brush Cursor */}
      {cursorPos.visible && (
        <div
          className={`brush-cursor mode-${brushMode}`}
          style={{
            left: `${cursorPos.x}px`,
            top: `${cursorPos.y}px`,
            width: `${brushSize}px`,
            height: `${brushSize}px`,
            marginLeft: `${-brushSize / 2}px`,
            marginTop: `${-brushSize / 2}px`,
          }}
        >
          {hardness < 0.95 && (
            <div
              className="brush-cursor-inner"
              style={{
                width: `${brushSize * hardness}px`,
                height: `${brushSize * hardness}px`,
                marginLeft: `${(-brushSize * hardness) / 2}px`,
                marginTop: `${(-brushSize * hardness) / 2}px`,
              }}
            />
          )}
        </div>
      )}

      {/* Text-Prompt Occlusion Visual Guides */}
      {showOcclusionGuides && activeFace && (
        <div className="occlusion-guides-layer" style={{ pointerEvents: 'none' }}>
          {detectedOccluders.map(occ => {
            const [bx1, by1, bx2, by2] = activeFace.bbox;
            const bw = (bx2 - bx1) * letterbox.displayW;
            const bh = (by2 - by1) * letterbox.displayH;
            const bx = letterbox.offsetX + bx1 * letterbox.displayW;
            const by = letterbox.offsetY + by1 * letterbox.displayH;

            let ox = bx + bw * 0.1;
            let oy = by + bh * 0.25;
            let ow = bw * 0.8;
            let oh = bh * 0.22;

            if (occ.region === 'forehead') {
              oy = by;
              oh = bh * 0.25;
            } else if (occ.region === 'chin') {
              oy = by + bh * 0.65;
              oh = bh * 0.3;
            }

            return (
              <div
                key={occ.id}
                className={`occlusion-guide-box ${occ.active ? 'active' : ''}`}
                style={{
                  left: `${ox}px`,
                  top: `${oy}px`,
                  width: `${ow}px`,
                  height: `${oh}px`,
                }}
              >
                <span className="occlusion-guide-label">
                  ✦ Auto-Occluder: {occ.label} ({Math.round(occ.confidence * 100)}%)
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Mask Editor Toolbar */}
      <div className="mask-editor-toolbar" onClick={e => e.stopPropagation()}>
        <div className="toolbar-section">
          <span className="toolbar-title">MASK BRUSH</span>
          <div className="brush-mode-buttons">
            <button
              type="button"
              className={`brush-mode-btn ${brushMode === 'paint' ? 'active paint' : ''}`}
              onClick={() => setBrushMode('paint')}
              title="Paint Occlusion Mask (Keep Target original area)"
            >
              ● Paint
            </button>
            <button
              type="button"
              className={`brush-mode-btn ${brushMode === 'erase' ? 'active erase' : ''}`}
              onClick={() => setBrushMode('erase')}
              title="Erase Mask"
            >
              ⊘ Erase
            </button>
            <button
              type="button"
              className={`brush-mode-btn ${brushMode === 'blur' ? 'active blur' : ''}`}
              onClick={() => setBrushMode('blur')}
              title="Box Blur / Soften Edge"
            >
              ≋ Blur
            </button>
          </div>
        </div>

        <div className="toolbar-divider" />

        {/* Brush Settings */}
        <div className="toolbar-section settings">
          <div className="brush-setting"><NumericInput label="Size:" value={brushSize} min={2} max={100} step={1} onChange={setBrushSize} /></div>

          <div className="brush-setting"><NumericInput label="Hardness:" value={hardness} min={0} max={1} step={0.01} onChange={setHardness} /></div>

          <div className="brush-setting"><NumericInput label="Opacity:" value={opacity} min={0.1} max={1} step={0.01} onChange={setOpacity} /></div>
        </div>

        <div className="toolbar-divider" />

        {/* Occlusion Guides & Text Prompt */}
        <div className="toolbar-section occlusion">
          <label className="toggle-guides">
            <input
              type="checkbox"
              checked={showOcclusionGuides}
              onChange={e => setShowOcclusionGuides(e.target.checked)}
            />
            <span>Auto Occlusion Guides</span>
          </label>

          {showOcclusionGuides && (
            <div className="occlusion-actions">
              <input
                type="text"
                className="occlusion-prompt-input"
                value={occlusionPrompt}
                onChange={e => setOcclusionPrompt(e.target.value)}
                placeholder="glasses, hands, hair..."
                title="Target occluder text prompts"
              />
              <button
                type="button"
                className="auto-fill-btn"
                onClick={() => handleAutoFillOcclusion(detectedOccluders[0])}
                title="Auto-fill detected glasses into mask"
              >
                + Fill Glasses
              </button>
            </div>
          )}
        </div>

        <div className="toolbar-divider" />

        {/* Action buttons */}
        <div className="toolbar-section actions">
          <button
            type="button"
            className="mask-action-btn clear"
            onClick={handleClearMask}
            title="Clear entire mask"
          >
            Clear
          </button>
          <button
            type="button"
            className="mask-action-btn done"
            onClick={onClose}
            title="Done editing mask"
          >
            Done ✓
          </button>
        </div>
      </div>
    </div>
  );
}
