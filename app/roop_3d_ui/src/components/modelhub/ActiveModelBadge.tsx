import { useActiveModelPolling, useActiveModels, type ActiveModel } from '../../stores/activeModels';

/** Stage names the engine reports: the detector plus whatever the true-3D renderer loads (head regressor, shader networks ...). */
const STAGE_LABELS: Record<string, string> = { detector: 'Detector', head: 'Head model', shader: 'Shader' };
const labelOf = (stage: string) => STAGE_LABELS[stage] ?? stage.replaceAll('_', ' ');

/**
 * A loaded session of another model means the new choice has not been used yet; it loads on the next preview / render.
 * Sessions are read back under the stable catalog id.
 */
function isPending(active: ActiveModel, selected: string | undefined): boolean {
  if (!active || !selected || selected === 'off' || selected === 'auto') return false;
  return active.model_id !== selected;
}

function Row({ label, stage, active, selected }: { label: string; stage: string; active: ActiveModel; selected: string | undefined }) {
  const wanted = selected && selected !== 'off' && selected !== 'auto' ? selected : null;
  const pending = isPending(active, selected);
  return (
    <li className={`active-model-row ${active ? 'loaded' : 'idle'}`} data-testid={`active-model-${stage}`} data-stage={stage}>
      <span className="active-model-stage">{label}</span>
      {active ? (
        <>
          <code className="active-model-file" title={active.model_path}>{active.filename}</code>
          <span className="active-model-meta">{active.provider.replace('ExecutionProvider', '')} · {active.precision}</span>
          {pending && <span className="active-model-pending" role="status">Selected {wanted}: loads on the next preview or render</span>}
        </>
      ) : (
        <span className="active-model-none">{wanted ? `Selected ${wanted}; nothing loaded yet` : 'Nothing loaded'}</span>
      )}
    </li>
  );
}

/** Live read-back of loaded CPU/GPU sessions (`GET /models/active`), not a VRAM residency claim. */
export default function ActiveModelBadge({ selected }: { selected?: Partial<Record<string, string>> }) {
  useActiveModelPolling();
  const value = useActiveModels(state => state.value);
  const error = useActiveModels(state => state.error);
  // The detector row is always shown; other stages appear as the renderer reports them.
  const stages = ['detector', ...Object.keys(value?.active ?? {}).filter(stage => stage !== 'detector')];
  return (
    <section className="rack-section active-models" aria-label="Loaded models" data-testid="active-models">
      <h4>Loaded model sessions</h4>
      <ul className="active-model-list">
        {stages.map(stage => (
          <Row key={stage} label={labelOf(stage)} stage={stage} active={value?.active?.[stage] ?? null} selected={selected?.[stage] ?? (value?.selected as Record<string, string | undefined> | undefined)?.[`${stage}_model`]} />
        ))}
      </ul>
      {error && !value && <small className="muted">{error}</small>}
    </section>
  );
}
