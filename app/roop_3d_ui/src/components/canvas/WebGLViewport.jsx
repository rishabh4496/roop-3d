import { useEffect, useRef } from 'react';
import {
  COLORMAPS, FRAGMENT_SHADER, GIZMO_FRAGMENT_SHADER, GIZMO_VERTEX_SHADER, LOUPE_RADIUS_PX, LOUPE_ZOOM, MODE_IDS, VERTEX_SHADER,
  buildGizmoGeometry, colormapLut, crossfadeValue, gizmoFaces,
} from './webglModel';

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) === false) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed: ${log}`);
  }
  return shader;
}
function link(gl, vertex, fragment) {
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, vertex));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, fragment));
  gl.linkProgram(program);
  if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) throw new Error(`Program link failed: ${gl.getProgramInfoLog(program)}`);
  return program;
}

/** Native-resolution texture: the image is uploaded at its own size, only downscaled if the GPU cannot hold it. */
function uploadTexture(gl, texture, image, maxSize) {
  let source = image, width = ('naturalWidth' in image ? image.naturalWidth : 0) || image.width, height = ('naturalHeight' in image ? image.naturalHeight : 0) || image.height;
  if (Math.max(width, height) > maxSize) {
    const scale = maxSize / Math.max(width, height), layer = document.createElement('canvas');
    layer.width = Math.max(1, Math.floor(width * scale)); layer.height = Math.max(1, Math.floor(height * scale));
    layer.getContext('2d').drawImage(image, 0, 0, layer.width, layer.height);
    source = layer; width = layer.width; height = layer.height;
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, source);
  gl.generateMipmap(gl.TEXTURE_2D);                               // crisp when a 4K frame is shown below 1:1
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return { width, height };
}

function createResources(gl) {
  const main = link(gl, VERTEX_SHADER, FRAGMENT_SHADER), gizmo = link(gl, GIZMO_VERTEX_SHADER, GIZMO_FRAGMENT_SHADER);
  const uniforms = {};
  for (const name of ['uOriginal', 'uSwapped', 'uColormap', 'uOriginalSize', 'uSwappedSize', 'uStage', 'uMode', 'uSplit', 'uVertical', 'uMix', 'uGain', 'uPointer', 'uLoupeRadius', 'uLoupeZoom', 'uBicubic', 'uHasSwapped']) uniforms[name] = gl.getUniformLocation(main, name);
  const quad = gl.createVertexArray(), quadBuffer = gl.createBuffer();
  gl.bindVertexArray(quad);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(main, 'aPosition');
  gl.enableVertexAttribArray(position); gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  const arrows = gl.createVertexArray(), arrowBuffer = gl.createBuffer();
  gl.bindVertexArray(arrows);
  gl.bindBuffer(gl.ARRAY_BUFFER, arrowBuffer);
  const gp = gl.getAttribLocation(gizmo, 'aPosition'), gc = gl.getAttribLocation(gizmo, 'aColor');
  gl.enableVertexAttribArray(gp); gl.vertexAttribPointer(gp, 2, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(gc); gl.vertexAttribPointer(gc, 4, gl.FLOAT, false, 24, 8);
  gl.bindVertexArray(null);
  const textures = { original: gl.createTexture(), swapped: gl.createTexture() }, luts = {};
  for (const name of COLORMAPS) {
    const lut = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, lut);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, colormapLut(name));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    luts[name] = lut;
  }
  return { main, gizmo, uniforms, quad, arrows, arrowBuffer, textures, luts, sizes: { original: [1, 1], swapped: [1, 1] }, maxTexture: gl.getParameter(gl.MAX_TEXTURE_SIZE) || 4096, maxViewport: (gl.getParameter(gl.MAX_VIEWPORT_DIMS) || [8192])[0] };
}

/**
 * Hardware-accelerated comparison canvas (WebGL2). One fragment shader renders every inspection mode from two
 * native-resolution textures: split wipe, alpha crossfade / blend, diff heat map (Inferno / Viridis), and a 3.5x loupe
 * with a bilinear / bicubic sampling toggle. A second tiny program draws the live yaw / pitch / roll axis gizmo over each
 * face's nose bridge. It draws only when something changed (coalesced to one frame), and calls `onUnsupported` when
 * WebGL2 or the shaders are unavailable so the caller can fall back to the DOM / 2D path.
 *
 * @param {{
 *   imageA: HTMLImageElement | ImageBitmap | null; imageB: HTMLImageElement | ImageBitmap | null;
 *   mode: 'wipe'|'crossfade'|'blend'|'diff'|'loupe'; split?: number; vertical?: boolean; mix?: number; gain?: number;
 *   colormap?: 'inferno'|'viridis'; bicubic?: boolean; pointer?: { x: number; y: number; active: boolean } | null;
 *   faces?: any[]; showGizmo?: boolean; width: number; height: number; pixelRatio?: number; preserveDrawingBuffer?: boolean;
 *   fadePlaying?: boolean; fadeSeconds?: number; now?: () => number; onFade?: (value: number) => void;
 *   onUnsupported?: (reason: string) => void; onReady?: (info: { maxTexture: number }) => void;
 * }} props
 */
export default function WebGLViewport(props) {
  const canvas = useRef(null), state = useRef(null), latest = useRef(props), frame = useRef(0), fade = useRef(0), dirty = useRef({ a: true, b: true });
  latest.current = props;

  const draw = () => {
    frame.current = 0;
    const gl = state.current?.gl, res = state.current?.res, p = latest.current;
    if (!gl || !res || gl.isContextLost?.()) return;
    const dpr = Math.min(p.pixelRatio ?? (window.devicePixelRatio || 1), res.maxViewport / Math.max(1, p.width, p.height));
    const w = Math.max(1, Math.round(p.width * dpr)), h = Math.max(1, Math.round(p.height * dpr));
    if (canvas.current.width !== w || canvas.current.height !== h) { canvas.current.width = w; canvas.current.height = h; }
    if (dirty.current.a && p.imageA) { res.sizes.original = Object.values(uploadTexture(gl, res.textures.original, p.imageA, res.maxTexture)); dirty.current.a = false; }
    if (dirty.current.b && p.imageB) { res.sizes.swapped = Object.values(uploadTexture(gl, res.textures.swapped, p.imageB, res.maxTexture)); dirty.current.b = false; }
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND);
    gl.useProgram(res.main);
    const u = res.uniforms, mode = MODE_IDS[p.mode] ?? 0, pointer = p.pointer;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, res.textures.original); gl.uniform1i(u.uOriginal, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, p.imageB ? res.textures.swapped : res.textures.original); gl.uniform1i(u.uSwapped, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, res.luts[p.colormap] ?? res.luts.inferno); gl.uniform1i(u.uColormap, 2);
    gl.uniform2f(u.uOriginalSize, res.sizes.original[0], res.sizes.original[1]);
    gl.uniform2f(u.uSwappedSize, res.sizes.swapped[0], res.sizes.swapped[1]);
    gl.uniform2f(u.uStage, w, h);
    gl.uniform1i(u.uMode, mode);
    gl.uniform1f(u.uSplit, p.split ?? 0.5);
    gl.uniform1i(u.uVertical, p.vertical ? 1 : 0);
    gl.uniform1f(u.uMix, p.mode === 'crossfade' && p.fadePlaying ? fade.current : (p.mix ?? 0.5));
    gl.uniform1f(u.uGain, p.gain ?? 3);
    gl.uniform2f(u.uPointer, (pointer?.x ?? -1e4) * dpr, (pointer?.y ?? -1e4) * dpr);
    gl.uniform1f(u.uLoupeRadius, pointer?.active ? LOUPE_RADIUS_PX * dpr : 0);
    gl.uniform1f(u.uLoupeZoom, LOUPE_ZOOM);
    gl.uniform1i(u.uBicubic, p.bicubic ? 1 : 0);
    gl.uniform1i(u.uHasSwapped, p.imageB ? 1 : 0);
    gl.bindVertexArray(res.quad);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    if (p.showGizmo && p.faces?.length) {
      const geometry = buildGizmoGeometry(gizmoFaces(p.faces), w, h, { thickness: 2.5 * dpr });
      if (geometry.length) {
        gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(res.gizmo);
        gl.bindVertexArray(res.arrows);
        gl.bindBuffer(gl.ARRAY_BUFFER, res.arrowBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, geometry, gl.DYNAMIC_DRAW);
        gl.drawArrays(gl.TRIANGLES, 0, geometry.length / 6);
        gl.disable(gl.BLEND);
      }
    }
    gl.bindVertexArray(null);
  };
  const schedule = () => { if (!frame.current && state.current) frame.current = requestAnimationFrame(draw); };

  // Context lifetime: created once, rebuilt after a GPU reset, released on unmount.
  useEffect(() => {
    const element = canvas.current;
    const init = () => {
      let gl = null;
      // preserveDrawingBuffer is opt-in: it lets the canvas be read back (snapshots, pixel-exact tests) at a small compositing cost.
      try { gl = element.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: !!latest.current.preserveDrawingBuffer }); } catch { gl = null; }
      if (!gl) { latest.current.onUnsupported?.('WebGL2 is not available in this browser.'); return; }
      try {
        state.current = { gl, res: createResources(gl) };
        dirty.current = { a: true, b: true };
        latest.current.onReady?.({ maxTexture: state.current.res.maxTexture });
        schedule();
      } catch (error) { state.current = null; latest.current.onUnsupported?.(error.message); }
    };
    const lost = event => { event.preventDefault(); state.current = null; };
    const restored = () => init();
    element.addEventListener('webglcontextlost', lost); element.addEventListener('webglcontextrestored', restored);
    init();
    return () => {
      element.removeEventListener('webglcontextlost', lost); element.removeEventListener('webglcontextrestored', restored);
      if (frame.current) cancelAnimationFrame(frame.current);
      state.current?.gl.getExtension?.('WEBGL_lose_context')?.loseContext();
      state.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- one context per canvas element; callbacks are read through `latest`
  }, []);

  useEffect(() => { dirty.current.a = true; schedule(); }, [props.imageA]);   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { dirty.current.b = true; schedule(); }, [props.imageB]);   // eslint-disable-line react-hooks/exhaustive-deps
  // Any visual parameter change redraws once (coalesced); nothing runs while the view is idle.
  useEffect(() => { schedule(); });

  // Persistent crossfade: advances the uniform directly, so a running fade never re-renders React.
  useEffect(() => {
    if (props.mode !== 'crossfade' || !props.fadePlaying) return undefined;
    const clock = props.now ?? (() => performance.now()), started = clock();
    let loop = 0, reported = 0;
    const tick = () => {
      fade.current = crossfadeValue(clock() - started, latest.current.fadeSeconds ?? 2);
      const t = clock();
      if (t - reported > 100) { reported = t; latest.current.onFade?.(fade.current); }
      schedule(); loop = requestAnimationFrame(tick);
    };
    loop = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(loop);
  }, [props.mode, props.fadePlaying, props.now]);                              // eslint-disable-line react-hooks/exhaustive-deps

  return <canvas ref={canvas} className="webgl-viewport" data-testid="webgl-viewport" data-mode={props.mode} role="img"
    aria-label={`Comparison canvas, ${props.mode} mode`} style={{ width: props.width, height: props.height, display: 'block', position: 'absolute', inset: 0 }} />;
}
