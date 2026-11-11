import { Canvas, useThree, type CanvasProps } from '@react-three/fiber';
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import WorkstationErrorBoundary from '../common/WorkstationErrorBoundary';
import { detectWebGL, liteInspectorRequested, setLiteInspector } from '../../utils/webglSupport';
import '../modelhub/modelHub.css';

export type ContextPhase = 'ok' | 'lost' | 'failed';
/** After a context loss the browser has this long to hand the context back before the canvas is rebuilt. */
export const RESTORE_TIMEOUT_MS = 4000;
/** Rebuilds before the viewport gives up on 3D and shows the lightweight inspector. */
export const MAX_REBUILDS = 2;

/**
 * Lives inside the `<Canvas>`: listens to the canvas element for `webglcontextlost` / `webglcontextrestored`.
 * `preventDefault()` on the loss event is what allows the browser to restore the context at all; three.js then re-uploads every
 * geometry, texture and program by itself on the next frame, so a restore only needs a repaint.
 */
export function WebGLRecovery({ onPhase, timeoutMs = RESTORE_TIMEOUT_MS }: { onPhase: (phase: ContextPhase) => void; timeoutMs?: number }) {
  const gl = useThree(state => state.gl);
  const invalidate = useThree(state => state.invalidate);
  useEffect(() => {
    const element = gl?.domElement;
    if (!element || typeof element.addEventListener !== 'function') return undefined;   // a test double without a real canvas
    let timer: ReturnType<typeof setTimeout> | undefined;
    const lost = (event: Event) => {
      event.preventDefault();
      onPhase('lost');
      clearTimeout(timer);
      timer = setTimeout(() => onPhase('failed'), timeoutMs);
    };
    const restored = () => { clearTimeout(timer); onPhase('ok'); invalidate(); };
    element.addEventListener('webglcontextlost', lost);
    element.addEventListener('webglcontextrestored', restored);
    return () => { clearTimeout(timer); element.removeEventListener('webglcontextlost', lost); element.removeEventListener('webglcontextrestored', restored); };
  }, [gl, invalidate, onPhase, timeoutMs]);
  return null;
}

export interface GuardedCanvasProps extends Omit<CanvasProps, 'fallback'> {
  /** Shown instead of the canvas when WebGL is unavailable, cannot be recovered, or the lightweight inspector is chosen. Receives the reason. */
  fallback: (context: { reason: string; retry?: () => void; use3D?: () => void }) => ReactNode;
  /** Name for messages (`3D viewport`). */
  label?: string;
  /** Changing this value rebuilds the canvas from scratch (a different FaceSet, a different mesh source). */
  resetKey?: unknown;
}

/**
 * A react-three-fiber `<Canvas>` that cannot take the page down.
 *  - no WebGL at all -> `fallback` (with the browser's reason);
 *  - the context is lost and not restored in time -> the canvas is rebuilt (up to {@link MAX_REBUILDS} times), then `fallback`;
 *  - an error thrown while the canvas is created or drawn -> the boundary shows `fallback`, and "Restart" rebuilds it;
 *  - a software renderer (SwiftShader / llvmpipe) is flagged with a notice and an opt-in switch to the lightweight inspector;
 *  - the render loop is paused while the canvas is scrolled out of view or the tab is hidden, so inactive tabs cost nothing.
 */
export default function GuardedCanvas({ fallback, label = '3D viewport', resetKey, children, frameloop, ...canvasProps }: GuardedCanvasProps) {
  const report = useMemo(() => detectWebGL(), []);
  const [lite, setLite] = useState(() => liteInspectorRequested());
  const [phase, setPhase] = useState<ContextPhase>('ok');
  const [rebuilds, setRebuilds] = useState(0);
  const [generation, setGeneration] = useState(0);
  const [visible, setVisible] = useState(true);
  const host = useRef<HTMLDivElement>(null);

  const lastKey = useRef(resetKey);
  useEffect(() => {
    if (Object.is(lastKey.current, resetKey)) return;      // not on mount: the first canvas is already fresh
    lastKey.current = resetKey;
    setPhase('ok'); setRebuilds(0); setGeneration(value => value + 1);
  }, [resetKey]);

  // A canvas that stays dead gets rebuilt (new context) a limited number of times.
  useEffect(() => {
    if (phase !== 'failed') return;
    if (rebuilds < MAX_REBUILDS) { setRebuilds(value => value + 1); setGeneration(value => value + 1); setPhase('ok'); }
  }, [phase, rebuilds]);

  useEffect(() => {
    const element = host.current;
    if (!element || typeof IntersectionObserver === 'undefined') return undefined;
    const observer = new IntersectionObserver(entries => { const last = entries.at(-1); if (last) setVisible(last.isIntersecting); });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const onPhase = useCallback((next: ContextPhase) => setPhase(next), []);
  const restart = useCallback(() => { setPhase('ok'); setRebuilds(0); setGeneration(value => value + 1); }, []);
  const chooseLite = useCallback((on: boolean) => { setLiteInspector(on); setLite(on); }, []);

  const gaveUp = phase === 'failed' && rebuilds >= MAX_REBUILDS;
  if (!report.available) return <>{fallback({ reason: report.reason })}</>;
  if (lite) return <>{fallback({ reason: 'Lightweight inspector selected. The 3D viewport is off, so no GPU is used.', use3D: () => chooseLite(false) })}</>;
  if (gaveUp) return <>{fallback({ reason: 'The graphics context was lost and could not be restored. Your work is not affected.', retry: restart })}</>;

  return (
    <WorkstationErrorBoundary name={label} resetKeys={[resetKey, generation]}
      fallback={({ error, reset }) => <>{fallback({ reason: `${label} could not start: ${error.message}`, retry: () => { restart(); reset(); } })}</>}>
      <div ref={host} className="guarded-canvas" data-webgl-phase={phase} data-webgl-software={report.software ? 'true' : 'false'} style={{ position: 'relative', width: '100%', height: '100%' }}>
        <Canvas key={generation} frameloop={visible && phase === 'ok' ? frameloop ?? 'always' : 'never'} {...canvasProps}>
          <WebGLRecovery onPhase={onPhase} />
          {children}
        </Canvas>
        {phase === 'lost' && <div className="pose-viewer-overlay loading" role="status" data-testid="webgl-lost">Graphics context lost. Recovering the 3D view…</div>}
        {report.software && (
          <div className="webgl-software-note" role="status" data-testid="webgl-software">
            <span>Software WebGL ({report.renderer || 'CPU renderer'}): 3D may be slow.</span>
            <button type="button" className="button small" onClick={() => chooseLite(true)}>Use lightweight inspector</button>
          </div>
        )}
      </div>
    </WorkstationErrorBoundary>
  );
}
