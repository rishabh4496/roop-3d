import { useActiveModelPolling, useActiveModels, type ActiveModel } from '../../stores/activeModels';

type Stage = 'detector' | 'swapper' | 'enhancer';
const STAGES: { stage: Stage; label: string }[] = [
  { stage: 'detector', label: 'Detector' }, { stage: 'swapper', label: 'Swapper' }, { stage: 'enhancer', label: 'Enhancer' },
];

/** Model ids the registry uses for the rack's enhancer ids, so "selected" and "loaded" compare like with like. */
const POLICY_MODEL: Record<string, string> = { gpen: 'gpen_bfr_512', gfpgan: 'gfpgan_v14', codeformer: 'codeformer', ultramax: 'ultramax' };

function Row({ label, stage, active, selected }: { label: string; stage: Stage; active: ActiveModel; selected: string | undefined }) {
  const wanted = stage === 'enhancer' && selected ? POLICY_MODEL[selected] ?? selected : selected;
  // A loaded session of another model means the new choice has not been used yet; it loads on the next preview / render.
  const pending = !!active && !!wanted && wanted !== 'off' && wanted !== 'auto' && active.model_id !== wanted && !(stage === 'enhancer' && active.model_id === 'codeformer' && wanted === 'ultramax');
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
        <span className="active-model-none">{selected && selected !== 'off' ? `Selected ${selected}; nothing loaded yet` : 'Nothing loaded'}</span>
      )}
    </li>
  );
}

/** Live read-back of the exact weight file each stage currently has in VRAM (`GET /models/active`). */
export default function ActiveModelBadge({ selected }: { selected?: Partial<Record<Stage, string>> }) {
  useActiveModelPolling();
  const value = useActiveModels(state => state.value);
  const error = useActiveModels(state => state.error);
  return (
    <section className="rack-section active-models" aria-label="Loaded models" data-testid="active-models">
      <h4>Loaded in VRAM</h4>
      <ul className="active-model-list">
        {STAGES.map(({ stage, label }) => (
          <Row key={stage} label={label} stage={stage} active={value?.active?.[stage] ?? null} selected={selected?.[stage] ?? value?.selected?.[`${stage}_model` as const]} />
        ))}
      </ul>
      {error && !value && <small className="muted">{error}</small>}
    </section>
  );
}
