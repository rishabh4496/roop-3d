import NumericInput from '../controls/NumericInput';
import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { calculateLetterbox } from './FaceOverlayCanvas';

/**
 * Advanced Workstation Comparison Canvas.
 * Supports 4 modes:
 * a) "Wipe": Interactive split-screen slider (A/B comparison between original target and swapped result).
 *    Supports both horizontal and vertical split dragging.
 * b) "Diff Map": Real-time pixel difference shader highlighting swapped regions and artifacts.
 * c) "Alpha Blend": Opacity fader between original and result.
 * d) "Magnifier Loupe": 3.5x circular zoom lens tracking the mouse cursor with sub-pixel crispness.
 * @param {{
 *   target?: string;
 *   output?: string;
 *   targetMediaRef?: any;
 *   outputMediaRef?: any;
 *   sourcePreviewUrl?: string;
 *   outputPreviewUrl?: string;
 *   mode?: 'wipe' | 'diff' | 'blend' | 'loupe';
 *   onModeChange?: (mode: 'wipe' | 'diff' | 'blend' | 'loupe') => void;
 *   containerWidth?: number;
 *   containerHeight?: number;
 *   mediaDimensions?: { width: number; height: number };
 * }} props
 */
export default function WorkstationCanvas({
  target = '',
  output = '',
  targetMediaRef,
  outputMediaRef,
  sourcePreviewUrl = '',
  outputPreviewUrl = '',
  mode = 'wipe', // 'wipe' | 'diff' | 'blend' | 'loupe'
  onModeChange,
  containerWidth = 0,
  containerHeight = 0,
  mediaDimensions = { width: 1920, height: 1080 },
}) {
  const containerRef = useRef(null);
  const diffCanvasRef = useRef(null);
  const loupeCanvasRef = useRef(null);

  // Wipe settings: split ratio (0..1), orientation ('horizontal' | 'vertical')
  const [split, setSplit] = useState(0.5);
  const [wipeOrientation, setWipeOrientation] = useState('horizontal'); // 'horizontal' | 'vertical'
  const isDraggingWipe = useRef(false);

  // Alpha blend opacity (0..1)
  const [blendOpacity, setBlendOpacity] = useState(0.5);

  // Diff map sensitivity boost (1x..8x)
  const [diffBoost, setDiffBoost] = useState(3.0);

  // Magnifier Loupe: cursor position [x, y], zoom factor 3.5x
  const [mousePos, setMousePos] = useState({ x: -1000, y: -1000, active: false });
  const zoomFactor = 3.5;
  const loupeRadius = 90; // 180px diameter lens

  const letterbox = useMemo(() => {
    return calculateLetterbox(
      containerWidth,
      containerHeight,
      mediaDimensions.width,
      mediaDimensions.height
    );
  }, [containerWidth, containerHeight, mediaDimensions.width, mediaDimensions.height]);

  // Pointer move inside container
  const handlePointerMove = useCallback(
    e => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      setMousePos({ x, y, active: true });

      // Handle wipe dragging
      if (isDraggingWipe.current) {
        if (wipeOrientation === 'horizontal') {
          const ratio = Math.max(0.01, Math.min(0.99, x / rect.width));
          setSplit(ratio);
        } else {
          const ratio = Math.max(0.01, Math.min(0.99, y / rect.height));
          setSplit(ratio);
        }
      }
    },
    [wipeOrientation]
  );

  const handlePointerDown = useCallback(
    e => {
      if (mode !== 'wipe') return;
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Check proximity to divider line
      const threshold = 18;
      if (wipeOrientation === 'horizontal') {
        const dividerX = split * rect.width;
        if (Math.abs(x - dividerX) < threshold) {
          isDraggingWipe.current = true;
          e.preventDefault();
        }
      } else {
        const dividerY = split * rect.height;
        if (Math.abs(y - dividerY) < threshold) {
          isDraggingWipe.current = true;
          e.preventDefault();
        }
      }
    },
    [mode, split, wipeOrientation]
  );

  const handlePointerUp = useCallback(() => {
    isDraggingWipe.current = false;
  }, []);

  const handlePointerLeave = useCallback(() => {
    isDraggingWipe.current = false;
    setMousePos(prev => ({ ...prev, active: false }));
  }, []);

  // Compute Diff Map
  useEffect(() => {
    if (mode !== 'diff' || !output) return;
    const canvas = diffCanvasRef.current;
    if (!canvas || containerWidth === 0 || containerHeight === 0) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const targetMedia = targetMediaRef?.current;
    const outputMedia = outputMediaRef?.current;

    // Offscreen render to compare
    const offW = Math.min(640, containerWidth);
    const offH = Math.min(360, containerHeight);

    const offA = document.createElement('canvas');
    offA.width = offW;
    offA.height = offH;
    const ctxA = offA.getContext('2d');

    const offB = document.createElement('canvas');
    offB.width = offW;
    offB.height = offH;
    const ctxB = offB.getContext('2d');

    if (!ctxA || !ctxB) return;

    try {
      if (targetMedia && targetMedia.readyState >= 2) {
        ctxA.drawImage(targetMedia, 0, 0, offW, offH);
      } else if (sourcePreviewUrl) {
        const imgA = new Image();
        imgA.src = sourcePreviewUrl;
        if (imgA.complete) ctxA.drawImage(imgA, 0, 0, offW, offH);
      }

      if (outputMedia && outputMedia.readyState >= 2) {
        ctxB.drawImage(outputMedia, 0, 0, offW, offH);
      } else if (outputPreviewUrl) {
        const imgB = new Image();
        imgB.src = outputPreviewUrl;
        if (imgB.complete) ctxB.drawImage(imgB, 0, 0, offW, offH);
      }

      const imgDataA = ctxA.getImageData(0, 0, offW, offH);
      const imgDataB = ctxB.getImageData(0, 0, offW, offH);
      const diffData = ctx.createImageData(offW, offH);

      const dA = imgDataA.data;
      const dB = imgDataB.data;
      const out = diffData.data;

      for (let i = 0; i < dA.length; i += 4) {
        const dr = Math.abs(dA[i] - dB[i]) * diffBoost;
        const dg = Math.abs(dA[i + 1] - dB[i + 1]) * diffBoost;
        const db = Math.abs(dA[i + 2] - dB[i + 2]) * diffBoost;
        const delta = (dr + dg + db) / 3;

        if (delta < 2) {
          // Unmodified area: dark slate
          out[i] = 15;
          out[i + 1] = 23;
          out[i + 2] = 26;
          out[i + 3] = 210;
        } else {
          // Heatmap: cyan (#00f5d4) to bright magenta (#f43f5e)
          const norm = Math.min(1.0, delta / 128);
          out[i] = Math.round(norm * 244 + (1 - norm) * 0);
          out[i + 1] = Math.round((1 - norm) * 245 + norm * 63);
          out[i + 2] = Math.round(norm * 94 + (1 - norm) * 212);
          out[i + 3] = Math.min(255, Math.round(180 + norm * 75));
        }
      }

      canvas.width = offW;
      canvas.height = offH;
      ctx.putImageData(diffData, 0, 0);
    } catch {
      // Cross-origin fallback
    }
  }, [
    mode,
    output,
    diffBoost,
    containerWidth,
    containerHeight,
    targetMediaRef,
    outputMediaRef,
    sourcePreviewUrl,
    outputPreviewUrl,
  ]);

  // Compute Magnifier Loupe
  useEffect(() => {
    if (mode !== 'loupe' || !mousePos.active) return;
    const canvas = loupeCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = loupeRadius * 2;
    canvas.width = size;
    canvas.height = size;

    ctx.clearRect(0, 0, size, size);

    // Circular clip path for lens
    ctx.save();
    ctx.beginPath();
    ctx.arc(loupeRadius, loupeRadius, loupeRadius - 2, 0, Math.PI * 2);
    ctx.clip();

    // Source coordinates centered at cursor
    const sourceX = mousePos.x - size / (zoomFactor * 2);
    const sourceY = mousePos.y - size / (zoomFactor * 2);
    const sourceW = size / zoomFactor;
    const sourceH = size / zoomFactor;

    // Draw magnified output (or target if output unavailable)
    const activeMedia = outputMediaRef?.current || targetMediaRef?.current;
    if (activeMedia && activeMedia.readyState >= 2) {
      // Map container coords to media
      const scaleX = activeMedia.videoWidth / (letterbox.displayW || 1);
      const scaleY = activeMedia.videoHeight / (letterbox.displayH || 1);
      const sx = (sourceX - letterbox.offsetX) * scaleX;
      const sy = (sourceY - letterbox.offsetY) * scaleY;
      const sw = sourceW * scaleX;
      const sh = sourceH * scaleY;
      try {
        ctx.drawImage(activeMedia, sx, sy, sw, sh, 0, 0, size, size);
      } catch {
        // Fallback
      }
    } else {
      ctx.fillStyle = '#111719';
      ctx.fillRect(0, 0, size, size);
    }

    // Glass reflection vignette
    const grad = ctx.createRadialGradient(
      loupeRadius * 0.7,
      loupeRadius * 0.7,
      loupeRadius * 0.2,
      loupeRadius,
      loupeRadius,
      loupeRadius
    );
    grad.addColorStop(0, 'rgba(255, 255, 255, 0.15)');
    grad.addColorStop(0.7, 'rgba(0, 0, 0, 0)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0.45)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);

    // Micro crosshair
    ctx.strokeStyle = 'rgba(0, 245, 212, 0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(loupeRadius - 8, loupeRadius);
    ctx.lineTo(loupeRadius + 8, loupeRadius);
    ctx.moveTo(loupeRadius, loupeRadius - 8);
    ctx.lineTo(loupeRadius, loupeRadius + 8);
    ctx.stroke();

    ctx.restore();

    // Outer lens ring
    ctx.beginPath();
    ctx.arc(loupeRadius, loupeRadius, loupeRadius - 2, 0, Math.PI * 2);
    ctx.strokeStyle = '#00f5d4';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }, [mode, mousePos, letterbox, targetMediaRef, outputMediaRef]);

  // Compute Wipe Clip Path
  const wipeClipStyle = useMemo(() => {
    if (mode !== 'wipe') return undefined;
    const p = (split * 100).toFixed(2);
    if (wipeOrientation === 'horizontal') {
      return { clipPath: `inset(0 0 0 ${p}%)` };
    }
    return { clipPath: `inset(${p}% 0 0 0)` };
  }, [mode, split, wipeOrientation]);

  return (
    <div
      ref={containerRef}
      className={`workstation-canvas-container mode-${mode}`}
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerLeave}
    >
      {/* 4-Mode Toggle Bar */}
      <div className="workstation-preview-bar">
        <div className="preview-mode-segmented">
          {[
            { id: 'wipe', label: 'Wipe (A/B)', icon: '◫' },
            { id: 'diff', label: 'Diff Map', icon: '◐' },
            { id: 'blend', label: 'Alpha Blend', icon: '◩' },
            { id: 'loupe', label: 'Loupe (3.5x)', icon: '⊕' },
          ].map(m => (
            <button
              key={m.id}
              type="button"
              className={`preview-mode-btn ${mode === m.id ? 'active' : ''}`}
              onClick={() => onModeChange && onModeChange(m.id)}
              disabled={m.id !== 'wipe' && !output}
              title={m.id !== 'wipe' && !output ? 'Available when swapped output is loaded' : m.label}
            >
              <span className="mode-icon">{m.icon}</span>
              <span>{m.label}</span>
            </button>
          ))}
        </div>

        {/* Mode specific controls */}
        {mode === 'wipe' && (
          <div className="wipe-aux-controls">
            <button
              type="button"
              className="wipe-orient-btn"
              onClick={() => setWipeOrientation(o => (o === 'horizontal' ? 'vertical' : 'horizontal'))}
              title={`Switch to ${wipeOrientation === 'horizontal' ? 'Vertical (Top/Bottom)' : 'Horizontal (Left/Right)'} Split`}
            >
              {wipeOrientation === 'horizontal' ? '↔ Horizontal' : '↕ Vertical'}
            </button>
            <span className="wipe-percent-label">{Math.round(split * 100)}%</span>
          </div>
        )}

        {mode === 'diff' && (
          <div className="diff-controls">
            <NumericInput label="Difference boost" value={diffBoost} min={1} max={8} step={.1} onChange={setDiffBoost} />
            <input
              type="range"
              min="1"
              max="8"
              step="0.5"
              value={diffBoost}
              onChange={e => setDiffBoost(parseFloat(e.target.value))}
              className="diff-boost-slider"
              title="Difference Sensitivity Boost"
            />
            <span className="control-val">{diffBoost.toFixed(1)}x</span>
          </div>
        )}

        {mode === 'blend' && (
          <div className="blend-controls">
            <NumericInput label="Comparison mix" value={blendOpacity} min={0} max={1} step={.01} onChange={setBlendOpacity} />
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={blendOpacity}
              onChange={e => setBlendOpacity(parseFloat(e.target.value))}
              className="blend-slider"
              title="Original (0%) to Swapped Result (100%)"
            />
            <span className="control-val">{Math.round(blendOpacity * 100)}%</span>
          </div>
        )}

        {mode === 'loupe' && (
          <div className="loupe-controls">
            <span className="tiny-badge">3.5x SUB-PIXEL LENS</span>
          </div>
        )}
      </div>

      {/* Wipe Divider Line and Handle */}
      {mode === 'wipe' && output && (
        <div
          className={`wipe-divider ${wipeOrientation}`}
          style={
            wipeOrientation === 'horizontal'
              ? { left: `${split * 100}%` }
              : { top: `${split * 100}%` }
          }
        >
          <div className="wipe-handle">
            <span>{wipeOrientation === 'horizontal' ? '◀ ▶' : '▲ ▼'}</span>
          </div>
          <span className="wipe-tag left">{wipeOrientation === 'horizontal' ? 'ORIGINAL' : 'TOP'}</span>
          <span className="wipe-tag right">{wipeOrientation === 'horizontal' ? 'SWAPPED' : 'BOTTOM'}</span>
        </div>
      )}

      {/* Diff Map Layer */}
      {mode === 'diff' && output && (
        <div className="diff-canvas-layer">
          <canvas ref={diffCanvasRef} className="diff-canvas" />
          <div className="diff-legend">
            <span><i className="dot diff-cyan" />Minor Shift</span>
            <span><i className="dot diff-magenta" />Major Swap Difference</span>
          </div>
        </div>
      )}

      {/* Alpha Blend Output Layer */}
      {mode === 'blend' && output && (
        <div
          className="alpha-blend-layer"
          style={{ opacity: blendOpacity }}
        />
      )}

      {/* Magnifier Loupe Lens */}
      {mode === 'loupe' && mousePos.active && (
        <div
          className="magnifier-loupe"
          style={{
            left: `${mousePos.x - loupeRadius}px`,
            top: `${mousePos.y - loupeRadius}px`,
            width: `${loupeRadius * 2}px`,
            height: `${loupeRadius * 2}px`,
          }}
        >
          <canvas ref={loupeCanvasRef} className="loupe-canvas" />
          <span className="loupe-badge">3.5x</span>
        </div>
      )}
    </div>
  );
}
