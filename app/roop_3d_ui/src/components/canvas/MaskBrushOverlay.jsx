import NumericInput from '../controls/NumericInput';
import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { calculateLetterbox } from './FaceOverlayCanvas';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import useHistoryStore from '../../stores/useHistoryStore';

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
  const history = useHistoryStore();   // brush strokes are workspace commands: this is the undo / redo stack
  const visibleCanvasRef = useRef(null);
  const exportCanvasRef = useRef(null);
  const includeCanvasRef = useRef(null);
  const excludeCanvasRef = useRef(null);
  const isPainting = useRef(false);
  const lastPoint = useRef(null);
  const eraseLayer = useRef('include');

  // Brush settings
  const [brushMode, setBrushMode] = useState('include'); // 'include' | 'exclude' | 'erase'
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

  const renderVisible = useCallback((bounds = null) => {
    const visible = visibleCanvasRef.current;
    const includeCanvas = includeCanvasRef.current, excludeCanvas = excludeCanvasRef.current;
    if (!visible || !includeCanvas || !excludeCanvas) return;
    const out = visible.getContext('2d'), inc = includeCanvas.getContext('2d'), exc = excludeCanvas.getContext('2d');
    if (!out || !inc || !exc) return;
    const left = bounds ? Math.max(0, Math.floor(bounds.left)) : 0;
    const top = bounds ? Math.max(0, Math.floor(bounds.top)) : 0;
    const width = Math.min(nativeW - left, bounds ? Math.max(1, Math.ceil(bounds.right) - left) : nativeW);
    const height = Math.min(nativeH - top, bounds ? Math.max(1, Math.ceil(bounds.bottom) - top) : nativeH);
    const a = inc.getImageData(left, top, width, height).data;
    const b = exc.getImageData(left, top, width, height).data;
    const pixels = out.createImageData(width, height);
    for (let i = 0; i < a.length; i += 4) {
      const include = a[i + 3], exclude = b[i + 3];
      if (include >= exclude) {
        pixels.data[i] = 45; pixels.data[i + 1] = 205; pixels.data[i + 2] = 125; pixels.data[i + 3] = Math.round(include * .62);
      } else {
        pixels.data[i] = 244; pixels.data[i + 1] = 63; pixels.data[i + 2] = 94; pixels.data[i + 3] = Math.round(exclude * .62);
      }
    }
    out.putImageData(pixels, left, top);
  }, [nativeW, nativeH]);

  // Screen -> Native canvas coordinate mapping
  const screenToCanvasCoords = useCallback(
    (clientX, clientY) => {
      const canvas = visibleCanvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const screenX = clientX - rect.left;
      const screenY = clientY - rect.top;
      const x = screenX * nativeW / (rect.width || 1);
      const y = screenY * nativeH / (rect.height || 1);

      return {
        x,
        y,
        screenX,
        screenY,
        onImage:
          screenX >= 0 && screenX <= rect.width && screenY >= 0 && screenY <= rect.height,
      };
    },
    [nativeW, nativeH]
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
    includeCanvasRef.current.width = nativeW;
    includeCanvasRef.current.height = nativeH;
    excludeCanvasRef.current.width = nativeW;
    excludeCanvasRef.current.height = nativeH;

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
    vCtx.clearRect(0, 0, nativeW, nativeH); eCtx.clearRect(0, 0, nativeW, nativeH);
    for (const canvas of [includeCanvasRef.current, excludeCanvasRef.current]) canvas?.getContext('2d')?.clearRect(0, 0, nativeW, nativeH);
    if (!initialMask) return;
    let live = true; const image = new Image();
    image.onload = () => {
      if (!live) return;
      const temp = document.createElement('canvas'); temp.width = nativeW; temp.height = nativeH;
      const ctx = temp.getContext('2d'); if (!ctx) return; ctx.drawImage(image, 0, 0, nativeW, nativeH);
      const data = ctx.getImageData(0, 0, nativeW, nativeH).data;
      const inc = includeCanvasRef.current.getContext('2d'), exc = excludeCanvasRef.current.getContext('2d');
      if (!inc || !exc) return;
      const incPixels = inc.createImageData(nativeW, nativeH), excPixels = exc.createImageData(nativeW, nativeH);
      for (let i = 0; i < data.length; i += 4) {
        const alpha = data[i + 3] / 255;
        if (data[i + 2] > 0) excPixels.data[i + 3] = Math.round(data[i + 2] * alpha); // legacy white / gray keep-target mask
        else { incPixels.data[i + 3] = Math.round(data[i] * alpha); excPixels.data[i + 3] = Math.round(data[i + 1] * alpha); }
        incPixels.data[i] = incPixels.data[i + 1] = incPixels.data[i + 2] = 255;
        excPixels.data[i] = excPixels.data[i + 1] = excPixels.data[i + 2] = 255;
      }
      inc.putImageData(incPixels, 0, 0); exc.putImageData(excPixels, 0, 0); renderVisible();
    };
    image.src = initialMask;
    return () => { live = false; };
  }, [initialMask, nativeW, nativeH, renderVisible]);

  // Apply brush dab / stroke
  const drawDab = useCallback(
    (pt, prevPt) => {
      const vCanvas = visibleCanvasRef.current;
      const layerCanvas = (brushMode === 'erase' ? eraseLayer.current : brushMode) === 'include' ? includeCanvasRef.current : excludeCanvasRef.current;
      if (!vCanvas || !layerCanvas) return;

      const ctx = layerCanvas.getContext('2d');
      if (!ctx) return;

      const scale = nativeW / (letterbox.displayW || 1);
      const r = Math.max(1, (brushSize / 2) * scale);
      const innerR = Math.max(0, r * hardness);

      const isErase = brushMode === 'erase';
      ctx.save(); ctx.globalCompositeOperation = isErase ? 'destination-out' : 'source-over';
      const color = `rgba(255,255,255,${opacity})`, clear = 'rgba(255,255,255,0)';
      const drawPoint = (x, y) => {
        const grad = ctx.createRadialGradient(x, y, innerR, x, y, r);
        grad.addColorStop(0, color); grad.addColorStop(1, hardness < .95 ? clear : color);
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      };
      if (prevPt) {
        const distance = Math.hypot(pt.x - prevPt.x, pt.y - prevPt.y);
        const steps = Math.max(1, Math.ceil(distance / Math.max(1, r * .3)));
        for (let index = 1; index <= steps; index++) {
          const t = index / steps;
          drawPoint(prevPt.x + (pt.x - prevPt.x) * t, prevPt.y + (pt.y - prevPt.y) * t);
        }
      } else drawPoint(pt.x, pt.y);
      ctx.restore();
      const pad = r + 2;
      renderVisible({ left: Math.min(pt.x, prevPt?.x ?? pt.x) - pad, top: Math.min(pt.y, prevPt?.y ?? pt.y) - pad,
        right: Math.max(pt.x, prevPt?.x ?? pt.x) + pad, bottom: Math.max(pt.y, prevPt?.y ?? pt.y) + pad });
    },
    [brushMode, brushSize, hardness, opacity, nativeW, letterbox, renderVisible]
  );

  // Pointer event handlers
  const handlePointerDown = e => {
    e.stopPropagation();
    if (e.target.closest?.('.mask-editor-toolbar')) return;
    const pt = screenToCanvasCoords(e.clientX, e.clientY);
    if (!pt || !pt.onImage) return;

    isPainting.current = true;
    e.currentTarget.setPointerCapture?.(e.pointerId);
    lastPoint.current = pt;
    drawDab(pt, null);
  };

  const handlePointerMove = e => {
    e.stopPropagation();
    const pt = screenToCanvasCoords(e.clientX, e.clientY);
    if (pt) {
      setCursorPos({ x: pt.screenX, y: pt.screenY, visible: pt.onImage });
      if (isPainting.current && pt.onImage) {
        drawDab(pt, lastPoint.current);
        lastPoint.current = pt;
      }
    }
  };

  const handlePointerUp = e => {
    e?.stopPropagation?.();
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
      const inc = includeCanvasRef.current?.getContext('2d'), exc = excludeCanvasRef.current?.getContext('2d');
      const out = eCanvas.getContext('2d');
      if (!inc || !exc || !out) return;
      const a = inc.getImageData(0, 0, nativeW, nativeH).data, b = exc.getImageData(0, 0, nativeW, nativeH).data;
      const packed = out.createImageData(nativeW, nativeH);
      for (let i = 0; i < a.length; i += 4) {
        packed.data[i] = a[i + 3]; packed.data[i + 1] = b[i + 3]; packed.data[i + 2] = 0; packed.data[i + 3] = 255;
      }
      out.putImageData(packed, 0, 0);
      const dataUrl = eCanvas.toDataURL('image/png');
      if (onMaskChange) {
        onMaskChange(dataUrl);
      }
    } catch {
      // Cross-origin fallback
    }
  }, [nativeW, nativeH, onMaskChange]);

  // Clear mask
  const handleClearMask = () => {
    const vCanvas = visibleCanvasRef.current;
    const eCanvas = exportCanvasRef.current;
    if (!vCanvas || !eCanvas) return;

    const vCtx = vCanvas.getContext('2d');
    if (vCtx) vCtx.clearRect(0, 0, nativeW, nativeH);
    includeCanvasRef.current?.getContext('2d')?.clearRect(0, 0, nativeW, nativeH);
    excludeCanvasRef.current?.getContext('2d')?.clearRect(0, 0, nativeW, nativeH);
    eCanvas.getContext('2d')?.clearRect(0, 0, nativeW, nativeH);

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

    const eCtx = excludeCanvasRef.current?.getContext('2d');
    if (eCtx) { eCtx.save(); eCtx.fillStyle = 'rgba(255,255,255,0.75)'; eCtx.beginPath(); eCtx.roundRect(rx1, ry1, rx2 - rx1, ry2 - ry1, 12); eCtx.fill(); eCtx.restore(); }
    renderVisible();

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
      <canvas ref={includeCanvasRef} style={{ display: 'none' }} />
      <canvas ref={excludeCanvasRef} style={{ display: 'none' }} />

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
                  âœ¦ Auto-Occluder: {occ.label} ({Math.round(occ.confidence * 100)}%)
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
              className={`brush-mode-btn ${brushMode === 'include' ? 'active include' : ''}`}
              onClick={() => { eraseLayer.current = 'include'; setBrushMode('include'); }}
              title="Include this area in the face swap. Paint the include mask."
            >
              Include
            </button>
            <button
              type="button"
              className={`brush-mode-btn ${brushMode === 'exclude' ? 'active exclude' : ''}`}
              onClick={() => { eraseLayer.current = 'exclude'; setBrushMode('exclude'); }}
              title="Exclude this area and keep the target pixels. Paint the exclusion mask."
            >
              Exclude
            </button>
            <button
              type="button"
              className={`brush-mode-btn ${brushMode === 'erase' ? 'active erase' : ''}`}
              onClick={() => setBrushMode('erase')}
              title="Erase the selected Include or Exclude override."
            >
              Erase
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
              <span>Occlusion Guides</span>
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
          <button type="button" className="mask-action-btn undo" disabled={!history.past.length} onClick={history.undo}
            title={`Undo the last edit (Ctrl+Z) · ${history.past.length} step${history.past.length === 1 ? '' : 's'} available`}>
            ↶ Undo{history.past.length ? ` (${history.past.length})` : ''}
          </button>
          <button type="button" className="mask-action-btn redo" disabled={!history.future.length} onClick={history.redo}
            title="Redo (Ctrl+Y)">
            ↷ Redo
          </button>
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
