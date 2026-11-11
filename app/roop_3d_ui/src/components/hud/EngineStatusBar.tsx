import { useEffect, useState } from 'react';
import { TRT_BUILD_MAX_S, TRT_VRAM_FRESH_MS, trtElapsedSeconds, useTrtBuild } from '../../stores/pipelineStream';
import { STAGE_LABEL, usePipelineStatus, usePipelineStatusPolling, type SwitchFailure } from '../../stores/pipelineStatus';
import type { ActiveModel } from '../../stores/activeModels';
import './engineStatus.css';

type Stage = SwitchFailure['stage'];
const STAGES: Stage[] = ['detector', 'swapper', 'enhancer'];

function StageChip({ stage, active, selected, failure }: { stage: Stage; active: ActiveModel; selected: string | undefined; failure: SwitchFailure | undefined }) {
  const label = STAGE_LABEL[stage];
  const verified = !!active?.sha256;
  const title = [
    active ? `${active.filename}\n${active.provider.replace('ExecutionProvider', '')} · ${active.precision}\n${active.model_path}` : `Nothing is loaded for this stage yet${selected ? ` (selected: ${selected}; it loads on the next preview or render)` : ''}.`,
    active?.sha256 ? `SHA-256 ${active.sha256}` : '',
    failure ? `Could not switch to ${failure.model_id}: ${failure.reason} [${failure.code}]` : '',
  ].filter(Boolean).join('\n');
  return (
    <li className={`engine-chip ${active ? 'loaded' : 'idle'} ${failure ? 'failed' : ''}`} data-testid={`engine-${stage}`} data-stage={stage} title={title}>
      <span className="engine-chip-label">{label}:</span>
      <b className="engine-chip-value">{active ? active.model_id : '—'}</b>
      {stage === 'enhancer' && active && <span className={`engine-chip-verify ${verified ? 'ok' : 'warn'}`}>{verified ? '• Verified' : '• Unverified'}</span>}
      {failure && <span className="engine-chip-fail" role="img" aria-label={`Switch to ${failure.model_id} failed`}>⚠ {failure.model_id} failed</span>}
    </li>
  );
}

/**
 * Shown while TensorRT compiles an engine (a first-use build takes 15-60 s inside one blocking native call). It is fed by the
 * server's `trt_compilation_progress` frames. ONNX Runtime holds the GIL while TensorRT builds, so the server is silent for the whole
 * build: this strip counts the elapsed time itself from the start frame, shows VRAM only while a frame that carried it is fresh (the
 * terminal has the live figure), and clears on the finish frame, a dropped socket, or a 15 minute ceiling. It sits outside `.engine-bar` on purpose: that strip is hidden on narrow windows and this signal must not be.
 */
function TrtBuildStrip() {
  const build = useTrtBuild();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!build) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [build]);
  if (!build) return null;
  const elapsed = trtElapsedSeconds(build, now);
  if (elapsed > TRT_BUILD_MAX_S) return null;
  const vramFresh = build.vramGb != null && now - build.receivedAt <= TRT_VRAM_FRESH_MS;
  const vram = vramFresh && build.vramGb != null ? `${build.vramGb.toFixed(1)} GB${build.vramScope === 'device' ? ' (device)' : ''}` : 'n/a';
  return (
    <div className="trt-build" role="status" data-testid="trt-build"
      title="TensorRT is building an engine for this model. It happens once per model and precision; later runs load the cached engine.">
      <span className="trt-build-spinner" aria-hidden="true" />
      <span>Compiling TensorRT engine <b>{build.model}</b> ({build.precisionLabel ?? build.precision ?? 'precision n/a'} · opt={build.optShape ?? 'n/a'})</span>
      <span className="trt-build-clock" aria-hidden="true" data-testid="trt-build-clock">{elapsed.toFixed(1)} s · VRAM {vram}</span>
      <small className="trt-build-note" data-testid="trt-build-note">
        {vramFresh ? 'First use only; this can take 15-60 s.' : 'First use only (15-60 s). The server is busy compiling, so the timer runs here and live VRAM is in the terminal.'}
      </small>
    </div>
  );
}

/**
 * The engines actually running, read back from `GET /api/pipeline/status` (never the dropdown selection): the detector, swapper
 * and enhancer model each stage has loaded, whether the enhancer's weight file was integrity-checked, and the execution
 * provider + precision the sessions were really built with. A model that failed to load shows its name with a warning here
 * and raises a toast with the backend's exact reason; the previous model is reported as such, not hidden.
 */
export default function EngineStatusBar() {
  usePipelineStatusPolling();
  const value = usePipelineStatus(state => state.value);
  const error = usePipelineStatus(state => state.error);
  const failures = value?.failures ?? [];
  const failureOf = (stage: Stage) => failures.find(item => item.stage === stage);
  return (
    <>
    <section className="engine-bar" aria-label="Active engines" data-testid="engine-status-bar" data-state={value ? 'live' : error ? 'offline' : 'loading'}>
      <ul className="engine-chips">
        {STAGES.map(stage => <StageChip key={stage} stage={stage} active={value?.active?.[stage] ?? null} failure={failureOf(stage)}
          selected={value?.selected?.[`${stage}_model` as const]} />)}
        <li className={`engine-chip precision ${value?.precision?.label ? 'loaded' : 'idle'}`} data-testid="engine-precision"
          title={value?.precision?.label ? Object.entries(value.precision.stages).map(([stage, text]) => `${STAGE_LABEL[stage as Stage] ?? stage}: ${text}`).join('\n') : `No session is loaded yet. Configured precision policy: ${value?.precision?.policy ?? 'auto'}.`}>
          <span className="engine-chip-label">Precision:</span>
          <b className="engine-chip-value">{value?.precision?.label ?? '—'}</b>
        </li>
      </ul>
      {!value && error && <small className="engine-bar-note" role="status">Engine status unavailable: {error}</small>}
    </section>
    <TrtBuildStrip />
    </>
  );
}
