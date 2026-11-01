import { useEffect, useMemo, useState } from 'react';
import { EventConnection } from '../../events';
import { buildPillModel, isPipelineFrame, formatElapsed } from './pillModel';
import { publishPipeline } from '../../stores/telemetryStore';

export const TELEMETRY_PATH = '/ws/telemetry';
const telemetryUrl = () => `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${TELEMETRY_PATH}`;

/**
 * Live 3-step run bar: Pre-Scan, Range Lock, Render & Mux, with skipped-frame share, fps, a Muxing state, and a
 * prominent banner while a TensorRT engine is compiling or its canary is running. Subscribes to `/ws/telemetry`
 * itself (reconnecting with backoff) and renders only what the server has actually measured.
 *
 * @param {{ url?: string, makeSocket?: (url: string) => any, now?: () => number, initialFrame?: any }} props
 */
export default function PipelinePill({ url, makeSocket, now = Date.now, initialFrame = null }) {
  const [frame, setFrame] = useState(initialFrame);
  const [connection, setConnection] = useState('connecting');
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const link = new EventConnection(url ?? telemetryUrl(), event => { if (isPipelineFrame(event)) { setFrame(event); publishPipeline(event); } }, setConnection, makeSocket);
    link.start();
    return () => link.stop();
  }, [url, makeSocket]);

  const compiling = frame?.engine_state && frame.engine_state !== 'IDLE';
  useEffect(() => {
    if (!compiling) return undefined;
    const timer = setInterval(() => setTick(value => value + 1), 1000); // keeps "elapsed" moving between frames
    return () => clearInterval(timer);
  }, [compiling]);

  // eslint-disable-next-line react-hooks/exhaustive-deps -- `tick` is the clock: it must recompute the elapsed seconds
  const model = useMemo(() => buildPillModel(frame, now()), [frame, now, tick]);
  const state = model.banner ? model.banner.kind : model.muxing ? 'muxing' : model.idle ? 'idle' : 'running';

  return (
    <section className={`run-pill ${state}`} data-tour="run-pill" data-state={state} data-connection={connection} aria-label="Pipeline run status">
      {model.banner && (
        <div className="compile-banner" role="alert">
          <strong>{model.banner.title}</strong>
          {model.banner.label && <span className="engine-label">{model.banner.label}</span>}
          {model.banner.model && <span className="engine-model">{model.banner.model}</span>}
          {model.banner.precision && <span className="engine-precision">{model.banner.precision}</span>}
          {model.banner.elapsedSeconds !== null && <time className="engine-elapsed">{formatElapsed(model.banner.elapsedSeconds)} elapsed</time>}
        </div>
      )}
      <ol className="run-steps">
        {model.steps.map(step => (
          <li key={step.id} className={`run-step ${step.status} ${step.active ? 'active' : ''}`} data-status={step.status} aria-current={step.active ? 'step' : undefined}>
            <span className="step-index" aria-hidden="true">{step.number}</span>
            <span className="step-label">Step {step.number} · {step.label}</span>
            <small className="step-state">
              {step.number === 3 && model.muxing ? 'Muxing…' : step.status === 'running' && step.progressPct !== null ? `${step.progressPct.toFixed(0)}%` : step.status}
            </small>
            {step.progressPct !== null && <progress max="100" value={step.progressPct} aria-label={`${step.label} progress`} />}
          </li>
        ))}
      </ol>
      <dl className="run-metrics">
        <div><dt>Skipped</dt><dd data-testid="skipped-pct">{model.skippedFramesPct === null ? '—' : `${model.skippedFramesPct.toFixed(1)}%`}</dd></div>
        <div><dt>FPS</dt><dd data-testid="fps">{model.fps > 0 ? model.fps.toFixed(1) : '—'}</dd></div>
        {model.muxing && <div><dt>Status</dt><dd className="muxing" data-testid="muxing">Muxing</dd></div>}
        {model.etaSeconds !== null && !model.muxing && <div><dt>ETA</dt><dd>{formatElapsed(Math.round(model.etaSeconds))}</dd></div>}
      </dl>
      <span className={`run-link ${connection}`} title={`/ws/telemetry: ${connection}`} role="img" aria-label={`Telemetry ${connection}`} />
    </section>
  );
}
