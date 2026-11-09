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
  );
}
