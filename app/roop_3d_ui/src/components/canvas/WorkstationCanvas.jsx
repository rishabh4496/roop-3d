import NumericInput from '../controls/NumericInput';
import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { calculateLetterbox } from '../../utils/letterbox';
import WebGLViewport from './WebGLViewport';
import MaskBrushOverlay from './MaskBrushOverlay';
import { COLORMAPS, DIFF_NOISE_FLOOR, LOUPE_ZOOM, crossfadeValue, webgl2Available } from './webglModel';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';

// The shader constants are the single source of truth; these re-exports keep the 2D-path API (and its tests) stable.
export { DIFF_NOISE_FLOOR, LOUPE_ZOOM, crossfadeValue };

export const MODES = [
  { id: 'wipe', label: 'Wipe (A/B)', icon: '◫' },
  { id: 'crossfade', label: 'Crossfade', icon: '◑' },
  { id: 'blend', label: 'Alpha Blend', icon: '◉' },
  { id: 'diff', label: 'Diff Map', icon: '◐' },
  { id: 'loupe', label: 'Loupe (3.5x)', icon: '⊕' },
];
const CANVAS_MODES = [
  { id: 'result', label: 'Result', icon: '◉' },
  { id: 'wipe', label: 'Split Wipe', legacyTitle: 'Split Wipe · Wipe (A/B)', icon: '↔' },
  ...MODES.filter(mode => mode.id !== 'wipe'),
  { id: 'matte', label: 'Mask Matte Solo', icon: '▧' },
];

const LOUPE_RADIUS = 90;

/**
 * Pointer â†’ media pixel coordinates with sub-pixel precision.
 * `stage` is the on-screen rectangle (client px) the media is drawn into; the result may be fractional.
 */
export function clientToMedia(stage, media, clientX, clientY) {
  if (!stage.width || !stage.height) return null;
  const u = (clientX - stage.left) / stage.width, v = (clientY - stage.top) / stage.height;
  if (u < 0 || u > 1 || v < 0 || v > 1) return null;
  return { x: u * media.width, y: v * media.height, u, v };
}

/** Per-pixel change heat-map data (RGBA) between two same-size RGBA buffers. */
export function diffHeatmap(a, b, boost, out) {
  for (let i = 0; i < a.length; i += 4) {
    const raw = (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])) / 3;
    if (raw < DIFF_NOISE_FLOOR) {
      out[i] = 15; out[i + 1] = 23; out[i + 2] = 26; out[i + 3] = 210;   // unchanged: dark slate
    } else {
      const norm = Math.min(1, (raw * boost) / 128);                     // cyan (minor) -> magenta (major)
      out[i] = Math.round(norm * 244); out[i + 1] = Math.round((1 - norm) * 245 + norm * 63);
      out[i + 2] = Math.round(norm * 94 + (1 - norm) * 212); out[i + 3] = Math.min(255, Math.round(180 + norm * 75));
    }
  }
  return out;
}

function useLoadedImage(url) {
  const [image, setImage] = useState(null);
  useEffect(() => {
    if (!url) { setImage(null); return undefined; }
    let live = true;
    const element = new Image();
    element.onload = () => { if (live) setImage(element); };
    element.onerror = () => { if (live) setImage(null); };
    element.src = url;
    return () => { live = false; };
  }, [url]);
  return image;
}

/**
 * Workstation comparison canvas. Modes:
 *  - Wipe: draggable split (horizontal / vertical) between original and swapped.
 *  - Crossfade: persistent Aâ†”B fade that keeps running while you scrub frames or edit settings.
 *  - Alpha Blend: fixed mix between the two.
 *  - Diff Map: heat-map of exactly which pixels changed (noise floor excluded), boostable.
 *  - Loupe: 3.5Ã— lens that shows individual pixels under the cursor.
 * With `originalUrl` / `swappedUrl` it composites the real frames aligned to the letterboxed media rectangle; a live
 * sub-pixel media coordinate readout follows the pointer.
 *
 * @param {{
 *   target?: string; output?: string; originalUrl?: string; swappedUrl?: string;
 *   sourcePreviewUrl?: string; outputPreviewUrl?: string; targetMediaRef?: any; outputMediaRef?: any;
 *   mode?: 'result' | 'wipe' | 'crossfade' | 'diff' | 'blend' | 'matte' | 'loupe'; onModeChange?: (mode: string) => void;
 *   containerWidth?: number; containerHeight?: number; mediaDimensions?: { width: number; height: number };
 *   fullBleed?: boolean; now?: () => number;
 *   onContextMenuPoint?: (clientX: number, clientY: number) => void;
 *   currentTime?: number; fps?: number; webgl?: boolean;
 *   maskEditing?: boolean; maskBase64?: string | null; onMaskChange?: (mask: string | null) => void;
 *   onMaskEditToggle?: (enabled: boolean) => void;
 * }} props
 */
export default function WorkstationCanvas({
  output = '',
  originalUrl,
  swappedUrl,
  sourcePreviewUrl = '',
  outputPreviewUrl = '',
  mode = 'wipe',
  onModeChange,
  containerWidth = 0,
  containerHeight = 0,
  mediaDimensions = { width: 1920, height: 1080 },
  fullBleed = false,
  now = () => performance.now(),
  onContextMenuPoint,
  currentTime = 0,
  fps = 30,
  webgl = true,
  maskEditing = false,
  maskBase64 = null,
  onMaskChange,
  onMaskEditToggle,
}) {
  const containerRef = useRef(null);
  const diffCanvasRef = useRef(null);
  const loupeCanvasRef = useRef(null);
  const aUrl = originalUrl ?? sourcePreviewUrl, bUrl = swappedUrl ?? outputPreviewUrl;
  const imageA = useLoadedImage(aUrl), imageB = useLoadedImage(bUrl);
  const hasOutput = Boolean(output || bUrl);

  const [split, setSplit] = useState(0.5);
  const [wipeOrientation, setWipeOrientation] = useState('horizontal');
  const isDraggingWipe = useRef(false);
  const [blendOpacity, setBlendOpacity] = useState(0.5);
  const [diffBoost, setDiffBoost] = useState(3.0);
  const [fadePlaying, setFadePlaying] = useState(() => !(typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches));
  const [fadeSeconds, setFadeSeconds] = useState(2);
  const [fade, setFade] = useState(0);
  const [mousePos, setMousePos] = useState({ x: -1000, y: -1000, active: false });
  const [coords, setCoords] = useState(null);
  // Stage 11: WebGL2 shader path (native-resolution textures, GPU diff / loupe / wipe / gizmo). Falls back to the DOM + 2D path.
  const [gl2] = useState(webgl2Available);
  const [glFailed, setGlFailed] = useState(false);
  const [colormap, setColormap] = useState('inferno');
  const [bicubic, setBicubic] = useState(false);
  const [showGizmo, setShowGizmo] = useState(true);
  const [loupeHeld, setLoupeHeld] = useState(false);
  const activeMode = loupeHeld ? 'loupe' : mode;
  const getFacesAtTimestamp = useFaceSelectionStore(state => state.getFacesAtTimestamp);
  const detectedFaces = useFaceSelectionStore(state => state.detectedFaces);
  const faces = useMemo(() => (detectedFaces?.length ? getFacesAtTimestamp(currentTime, fps) : []), [detectedFaces, currentTime, fps, getFacesAtTimestamp]);

  const letterbox = useMemo(
    () => calculateLetterbox(containerWidth, containerHeight, mediaDimensions.width, mediaDimensions.height),
    [containerWidth, containerHeight, mediaDimensions.width, mediaDimensions.height],
  );
  // Where the media is drawn, relative to the container.
  const stage = fullBleed
    ? { left: letterbox.offsetX, top: letterbox.offsetY, width: letterbox.displayW, height: letterbox.displayH }
    : { left: 0, top: 0, width: containerWidth, height: containerHeight };

  const useGl = webgl && gl2 && !glFailed && fullBleed && !maskEditing && Boolean(imageA) && stage.width > 0 && stage.height > 0;

  useEffect(() => {
    const down = event => {
      if ((event.code === 'KeyZ' || event.key?.toLowerCase() === 'z')
          && !['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target?.tagName)) {
        event.preventDefault(); setLoupeHeld(true);
      }
    };
    const up = event => { if (event.code === 'KeyZ' || event.key?.toLowerCase() === 'z') setLoupeHeld(false); };
    window.addEventListener('keydown', down); window.addEventListener('keyup', up);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); };
  }, []);

  // Persistent crossfade: keeps cycling until paused, independent of which frame is showing. (The shader path animates its own uniform.)
  useEffect(() => {
    if (useGl || activeMode !== 'crossfade' || !fadePlaying || typeof requestAnimationFrame !== 'function') return undefined;
    let frame = 0;
    const started = now();
    const tick = () => { setFade(crossfadeValue(now() - started, fadeSeconds)); frame = requestAnimationFrame(tick); };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [useGl, activeMode, fadePlaying, fadeSeconds, now]);

  const handlePointerMove = useCallback(event => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    setMousePos({ x, y, active: true });
    setCoords(clientToMedia({ left: rect.left + stage.left, top: rect.top + stage.top, width: stage.width, height: stage.height }, mediaDimensions, event.clientX, event.clientY));
    if (isDraggingWipe.current) {
      const ratio = wipeOrientation === 'horizontal' ? (x - stage.left) / (stage.width || rect.width) : (y - stage.top) / (stage.height || rect.height);
      setSplit(Math.max(0.01, Math.min(0.99, ratio)));
    }
  }, [wipeOrientation, stage.left, stage.top, stage.width, stage.height, mediaDimensions]);

  const handlePointerDown = useCallback(event => {
    if (activeMode !== 'wipe') return;
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    const threshold = 18;
    const divider = wipeOrientation === 'horizontal' ? stage.left + split * (stage.width || rect.width) : stage.top + split * (stage.height || rect.height);
    if (Math.abs((wipeOrientation === 'horizontal' ? x : y) - divider) < threshold) {
      isDraggingWipe.current = true;
      event.preventDefault();
    }
  }, [activeMode, split, wipeOrientation, stage.left, stage.top, stage.width, stage.height]);

  const handlePointerUp = useCallback(() => { isDraggingWipe.current = false; }, []);
  const handlePointerLeave = useCallback(() => {
    isDraggingWipe.current = false;
    setMousePos(previous => ({ ...previous, active: false }));
    setCoords(null);
  }, []);

  // Diff map: both frames resampled to the same grid, compared per pixel.
  useEffect(() => {
    if (useGl || activeMode !== 'diff' || !hasOutput) return;
    const canvas = diffCanvasRef.current;
    if (!canvas || !imageA || !imageB) return;
    const width = Math.min(960, mediaDimensions.width), height = Math.max(1, Math.round(width * mediaDimensions.height / mediaDimensions.width));
    const draw = image => {
      const layer = document.createElement('canvas');
      layer.width = width; layer.height = height;
      const context = layer.getContext('2d', { willReadFrequently: true });
      if (!context) return null;
      context.drawImage(image, 0, 0, width, height);
      return context.getImageData(0, 0, width, height).data;
    };
    const target = canvas.getContext('2d');
    if (!target) return;
    try {
      const a = draw(imageA), b = draw(imageB);
      if (!a || !b) return;
      const out = target.createImageData(width, height);
      diffHeatmap(a, b, diffBoost, out.data);
      canvas.width = width; canvas.height = height;
      target.putImageData(out, 0, 0);
    } catch { /* tainted canvas (cross-origin media): leave the previous heat-map in place */ }
  }, [useGl, activeMode, hasOutput, imageA, imageB, diffBoost, mediaDimensions.width, mediaDimensions.height]);

  // Loupe: 3.5x of the swapped frame (or the original when no swap exists), nearest-neighbour so pixels stay visible.
  useEffect(() => {
    if (useGl || activeMode !== 'loupe' || !mousePos.active) return;
    const canvas = loupeCanvasRef.current, image = imageB ?? imageA;
    if (!canvas || !image || !stage.width || !stage.height) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    const size = LOUPE_RADIUS * 2;
    canvas.width = size; canvas.height = size;
    context.imageSmoothingEnabled = false;
    const scaleX = image.naturalWidth / stage.width, scaleY = image.naturalHeight / stage.height;
    const sw = (size / LOUPE_ZOOM) * scaleX, sh = (size / LOUPE_ZOOM) * scaleY;
    context.save();
    context.beginPath();
    context.arc(LOUPE_RADIUS, LOUPE_RADIUS, LOUPE_RADIUS - 2, 0, Math.PI * 2);
    context.clip();
    context.drawImage(image, (mousePos.x - stage.left) * scaleX - sw / 2, (mousePos.y - stage.top) * scaleY - sh / 2, sw, sh, 0, 0, size, size);
    context.restore();
    context.beginPath();
    context.arc(LOUPE_RADIUS, LOUPE_RADIUS, LOUPE_RADIUS - 2, 0, Math.PI * 2);
    context.strokeStyle = '#00f5d4';
    context.lineWidth = 2.5;
    context.stroke();
  }, [useGl, activeMode, mousePos, imageA, imageB, stage.left, stage.top, stage.width, stage.height]);

  const overlayStyle = useMemo(() => {
    if (activeMode === 'wipe') {
      const p = (split * 100).toFixed(2);
      return { clipPath: wipeOrientation === 'horizontal' ? `inset(0 0 0 ${p}%)` : `inset(${p}% 0 0 0)` };
    }
    if (activeMode === 'crossfade') return { opacity: fade };
    if (activeMode === 'blend') return { opacity: blendOpacity };
    return undefined;
  }, [activeMode, split, wipeOrientation, fade, blendOpacity]);

  const stageStyle = fullBleed ? { left: stage.left, top: stage.top, width: stage.width, height: stage.height } : undefined;

  return (
    <div ref={containerRef} className={`workstation-canvas-container mode-${activeMode} ${maskEditing ? 'mask-editing' : ''} ${fullBleed ? 'full-bleed' : ''}`}
      onPointerMove={handlePointerMove} onPointerDown={handlePointerDown} onPointerUp={handlePointerUp} onPointerLeave={handlePointerLeave}
      onContextMenu={event => { if (onContextMenuPoint) { event.preventDefault(); onContextMenuPoint(event.clientX, event.clientY); } }}>
      <div className="workstation-preview-bar">
        <div className="preview-mode-segmented" role="group" aria-label="Comparison mode">
          {CANVAS_MODES.map(m => (
            <button key={m.id} type="button" className={`preview-mode-btn ${activeMode === m.id ? 'active' : ''}`} aria-pressed={activeMode === m.id}
              onClick={() => onModeChange && onModeChange(m.id)} disabled={m.id !== 'wipe' && !hasOutput}
              title={m.id !== 'wipe' && !hasOutput ? 'Available once a swapped frame exists' : (m.legacyTitle ?? m.label)}>
              <span className="mode-icon">{m.icon}</span><span>{m.label}</span>
            </button>
          ))}
        </div>

        {activeMode === 'wipe' && (
          <div className="wipe-aux-controls">
            <button type="button" className="wipe-orient-btn" onClick={() => setWipeOrientation(o => (o === 'horizontal' ? 'vertical' : 'horizontal'))}
              title={`Switch to ${wipeOrientation === 'horizontal' ? 'Vertical (Top/Bottom)' : 'Horizontal (Left/Right)'} Split`}>
              {wipeOrientation === 'horizontal' ? 'â†” Horizontal' : 'â†• Vertical'}
            </button>
            <span className="wipe-percent-label">{Math.round(split * 100)}%</span>
          </div>
        )}

        {activeMode === 'crossfade' && (
          <div className="crossfade-controls">
            <button type="button" className="crossfade-toggle" aria-pressed={fadePlaying} onClick={() => setFadePlaying(playing => !playing)}
              title="The crossfade keeps cycling while you scrub frames or change settings">{fadePlaying ? 'â¸ Pause' : 'â–¶ Play'}</button>
            <NumericInput label="Fade period (s)" value={fadeSeconds} min={0.5} max={10} step={0.1} onChange={value => setFadeSeconds(value ?? 2)} />
            <span className="control-val">{Math.round(fade * 100)}% swapped</span>
          </div>
        )}

        {activeMode === 'diff' && (
          <div className="diff-controls">
            <NumericInput label="Difference boost" value={diffBoost} min={1} max={8} step={.1} onChange={setDiffBoost} />
            {useGl && <select aria-label="Diff colormap" className="diff-colormap" value={colormap} onChange={event => setColormap(event.target.value)}>{COLORMAPS.map(name => <option key={name} value={name}>{name[0].toUpperCase() + name.slice(1)}</option>)}</select>}
            <input type="range" min="1" max="8" step="0.5" value={diffBoost} onChange={e => setDiffBoost(parseFloat(e.target.value))}
              className="diff-boost-slider" title="Difference Sensitivity Boost" />
            <span className="control-val">{diffBoost.toFixed(1)}x</span>
          </div>
        )}

        {activeMode === 'blend' && (
          <div className="blend-controls">
            <NumericInput label="Comparison mix" value={blendOpacity} min={0} max={1} step={.01} onChange={setBlendOpacity} />
            <input type="range" min="0" max="1" step="0.01" value={blendOpacity} onChange={e => setBlendOpacity(parseFloat(e.target.value))}
              className="blend-slider" title="Original (0%) to Swapped Result (100%)" />
            <span className="control-val">{Math.round(blendOpacity * 100)}%</span>
          </div>
        )}

        {activeMode === 'loupe' && <div className="loupe-controls"><span className="tiny-badge">3.5x SUB-PIXEL LENS Â· Hold Z</span>
          {useGl && <span className="loupe-sampling" role="group" aria-label="Loupe sampling">
            <button type="button" className={`preview-mode-btn ${bicubic ? '' : 'active'}`} aria-pressed={!bicubic} onClick={() => setBicubic(false)} title="Hardware bilinear sampling">Bilinear</button>
            <button type="button" className={`preview-mode-btn ${bicubic ? 'active' : ''}`} aria-pressed={bicubic} onClick={() => setBicubic(true)} title="Catmull-Rom bicubic sampling (sharper at 3.5x)">Bicubic</button>
          </span>}</div>}
        {useGl && <div className="gizmo-controls">
          <button type="button" className={`preview-mode-btn ${showGizmo ? 'active' : ''}`} aria-pressed={showGizmo} onClick={() => setShowGizmo(value => !value)} title="3D pose axes over each face's nose bridge">âŒ– Pose gizmo</button>
          {showGizmo && <span className="gizmo-legend" aria-label="Gizmo axes"><i style={{ background: '#ef4444' }}/>Pitch <i style={{ background: '#22c55e' }}/>Yaw <i style={{ background: '#3b82f6' }}/>Roll</span>}
        </div>}
      </div>

      {/* The two real frames, aligned to the letterboxed media rectangle. */}
      {useGl && (
        <div className="compare-stage webgl-stage" style={stageStyle}>
          <WebGLViewport imageA={imageA} imageB={bUrl ? imageB : null} mode={activeMode} split={split} vertical={wipeOrientation === 'vertical'} mix={activeMode === 'blend' ? blendOpacity : fade}
            gain={diffBoost} colormap={colormap} bicubic={bicubic} faces={faces} showGizmo={showGizmo} width={stage.width} height={stage.height}
            pointer={{ x: mousePos.x - stage.left, y: mousePos.y - stage.top, active: mousePos.active }}
            fadePlaying={fadePlaying} fadeSeconds={fadeSeconds} now={now} onFade={setFade} onUnsupported={() => setGlFailed(true)} />
        </div>
      )}
      {!useGl && (aUrl || bUrl) && (
        <div className="compare-stage" style={stageStyle}>
          {aUrl && <img className="compare-a" src={aUrl} alt="Original frame" draggable={false} />}
          {bUrl && activeMode !== 'diff' && activeMode !== 'matte' && !maskEditing && <img className="compare-b" src={bUrl} alt="Swapped frame" draggable={false} style={overlayStyle} />}
          {bUrl && activeMode === 'matte' && !maskEditing && <img className="compare-b matte-image" src={bUrl} alt="Composite mask matte" draggable={false} />}
          {bUrl && activeMode === 'result' && !maskEditing && <img className="compare-b" src={bUrl} alt="Swapped result" draggable={false} />}
        </div>
      )}

      {activeMode === 'wipe' && hasOutput && (
        <div className={`wipe-divider ${wipeOrientation}`}
          style={wipeOrientation === 'horizontal'
            ? { left: fullBleed ? stage.left + split * stage.width : `${split * 100}%` }
            : { top: fullBleed ? stage.top + split * stage.height : `${split * 100}%` }}>
          <div className="wipe-handle"><span>{wipeOrientation === 'horizontal' ? 'â—€ â–¶' : 'â–² â–¼'}</span></div>
          <span className="wipe-tag left">{wipeOrientation === 'horizontal' ? 'ORIGINAL' : 'TOP'}</span>
          <span className="wipe-tag right">{wipeOrientation === 'horizontal' ? 'SWAPPED' : 'BOTTOM'}</span>
        </div>
      )}

      {activeMode === 'diff' && hasOutput && (
        <div className="diff-canvas-layer" style={stageStyle}>
          {!useGl && <canvas ref={diffCanvasRef} className="diff-canvas" />}
          <div className="diff-legend">
            <span><i className="dot diff-cyan" />Minor Shift</span>
            <span><i className="dot diff-magenta" />Major Swap Difference</span>
          </div>
        </div>
      )}

      {activeMode === 'blend' && hasOutput && !useGl && <div className="alpha-blend-layer" style={{ opacity: blendOpacity }} />}

      {activeMode === 'loupe' && mousePos.active && useGl && <span className="loupe-badge gl-loupe-badge" style={{ left: mousePos.x + LOUPE_RADIUS - 24, top: mousePos.y + LOUPE_RADIUS - 6 }}>3.5x Â· {bicubic ? 'bicubic' : 'bilinear'}</span>}
      {activeMode === 'loupe' && mousePos.active && !useGl && (
        <div className="magnifier-loupe" style={{ left: `${mousePos.x - LOUPE_RADIUS}px`, top: `${mousePos.y - LOUPE_RADIUS}px`, width: `${LOUPE_RADIUS * 2}px`, height: `${LOUPE_RADIUS * 2}px` }}>
          <canvas ref={loupeCanvasRef} className="loupe-canvas" />
          <span className="loupe-badge">3.5x</span>
        </div>
      )}

      {coords && <div className="coord-readout" aria-live="off">x {coords.x.toFixed(2)} Â· y {coords.y.toFixed(2)} px</div>}
      {onMaskEditToggle && <button type="button" className="canvas-mask-toggle" aria-pressed={maskEditing}
        onClick={() => onMaskEditToggle(!maskEditing)} title="Paint include and exclude overrides on this frame">{maskEditing ? 'Done Mask' : 'Mask Brush'}</button>}
      {maskEditing && <MaskBrushOverlay containerWidth={containerWidth} containerHeight={containerHeight}
        mediaDimensions={mediaDimensions} onMaskChange={onMaskChange} initialMask={maskBase64}
        onClose={() => onMaskEditToggle?.(false)} />}
    </div>
  );
}
