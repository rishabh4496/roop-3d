import { useEffect, useRef, useState } from 'react';
import { api, post } from '../../api';
import { LiveFrameStream } from '../../liveStream';
import { useRack } from '../../stores/rackSelection';
import WebGLViewport from '../canvas/WebGLViewport';

type Status = { state: string; error: string | null; frames: number; capture_to_publish_p95_ms: number | null; virtual_camera_available: boolean; screen_available: boolean; spout_available: boolean };

function StudioCanvas() {
  const [frame, setFrame] = useState<ImageBitmap | null>(null);
  const current = useRef<ImageBitmap | null>(null);
  useEffect(() => {
    const stream = new LiveFrameStream({ onFrame: bitmap => {
      const previous = current.current;
      current.current = bitmap;
      setFrame(bitmap);
      // Texture upload occurs after React commits the frame.
      requestAnimationFrame(() => previous?.close());
    } });
    stream.start();
    return () => { stream.stop(); current.current?.close(); current.current = null; };
  }, []);
  return frame ? <WebGLViewport imageA={frame} imageB={null} width={640} height={480} mode="blend" mix={0} /> : <p>Waiting for live frames…</p>;
}

export default function LiveStudio({ sourceId }: { sourceId: string }) {
  const swapperId = useRack(state => state.swapper);
  const [renderMode, setRenderMode] = useState('standard');
  const [open, setOpen] = useState(false), [status, setStatus] = useState<Status | null>(null);
  const [input, setInput] = useState('webcam'), [device, setDevice] = useState(0), [fps, setFps] = useState(30);
  const [virtualCamera, setVirtualCamera] = useState(false), [spout, setSpout] = useState(false);
  const [error, setError] = useState(''), [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    const refresh = () => void api<Status>('/studio', { signal: controller.signal }).then(setStatus).catch(e => { if (!controller.signal.aborted) setError(String(e)); });
    refresh();
    const timer = setInterval(refresh, 1000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [open]);
  const running = status?.state === 'starting' || status?.state === 'running';
  const action = async () => {
    setBusy(true); setError('');
    try { setStatus(await post<Status>(running ? '/studio/stop' : '/studio/start', running ? {} : { source_faceset_id: sourceId, swapper_model_id: swapperId, render_mode: renderMode, input, device, fps, virtual_camera: virtualCamera, spout })); }
    catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  };
  return <>
    <button className="button small" aria-pressed={open} onClick={() => setOpen(!open)}>Live Studio Mode</button>
    {open && <section className="studio-live-panel" aria-label="Live Studio Mode" style={{ position: 'fixed', top: 64, right: 16, zIndex: 100, width: 680, maxWidth: '95vw', maxHeight: '85vh', overflow: 'auto', padding: 20, background: 'var(--panel, #181b22)', border: '1px solid var(--border, #555)' }}>
      <h2>Live Studio</h2>
      <fieldset disabled={running || busy}>
        <label>Live input <select value={input} onChange={e => setInput(e.target.value)}><option value="webcam">Webcam</option><option value="screen" disabled={!status?.screen_available}>Screen</option><option disabled>NDI — receiver unavailable</option></select></label>
        <label>Render mode <select value={renderMode} onChange={e => setRenderMode(e.target.value)}><option value="standard">Identity swap</option><option value="pose_3d">3D pose</option><option value="reenactment_3d">3D reenactment</option></select></label>
        <label>Device index <input type="number" min={0} max={32} value={device} onChange={e => setDevice(Number(e.target.value))} /></label>
        <label>Frame rate <select value={fps} onChange={e => setFps(Number(e.target.value))}><option value={30}>30 FPS</option><option value={60}>60 FPS</option></select></label>
        <label><input type="checkbox" checked={virtualCamera} disabled={!status?.virtual_camera_available} onChange={e => setVirtualCamera(e.target.checked)} />OBS / virtual camera</label>
        <label><input type="checkbox" checked={spout} disabled={!status?.spout_available} onChange={e => setSpout(e.target.checked)} />Spout2</label>
      </fieldset>
      <button className="button" disabled={busy || (!running && !sourceId)} onClick={() => void action()}>{running ? 'Stop live studio' : 'Start live studio'}</button>
      <button className="button" onClick={() => setOpen(false)}>Hide controls</button>
      {!sourceId && <p>Select a source FaceSet first.</p>}
      <p role="status">{status?.state ?? 'Connecting'} · {status?.frames ?? 0} frames{status?.capture_to_publish_p95_ms != null ? ` · processing p95 ${status.capture_to_publish_p95_ms.toFixed(1)} ms` : ''}</p>
      <p>Host-frame output · browser texture upload · 35 ms target requires hardware validation.</p>
      {(error || status?.error) && <p role="alert">{error || status?.error}</p>}
      {running && <div style={{ position: "relative", width: 640, height: 480 }}><StudioCanvas /></div>}
    </section>}
  </>;
}
