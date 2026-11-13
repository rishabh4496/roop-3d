import { useActiveModelPolling, useActiveModels, type ActiveModel } from '../../stores/activeModels';
import { RESTORER_LABELS, restorerFileMatches, restorerWeights, type RestorerModel } from '../workstation/pipeline';

type Stage = 'detector' | 'swapper' | 'enhancer';
const STAGES: { stage: Stage; label: string }[] = [
  { stage: 'detector', label: 'Detector' }, { stage: 'swapper', label: 'Swapper' }, { stage: 'enhancer', label: 'Enhancer' },
];

const isRestorer = (value: string | undefined): value is RestorerModel => value !== undefined && Object.hasOwn(RESTORER_LABELS, value);

/** What the user selected, as it reads next to the loaded weight: a restorer by name and file, any other stage by its model id. */
function selectedText(stage: Stage, selected: string | undefined): string | null {
  if (!selected || selected === 'off' || selected === 'auto') return null;
  return stage === 'enhancer' && isRestorer(selected) ? `${RESTORER_LABELS[selected]} (${restorerWeights(selected)})` : selected;
}

/**
 * A loaded session of another model means the new choice has not been used yet; it loads on the next preview / render. Detector and swapper
 * sessions are read back under the stable catalog id. Enhancers are read back under their registry hash id, so the loaded weight file decides
 * (CodeFormer and UltraMax load different artifacts).
 */
function isPending(stage: Stage, active: ActiveModel, selected: string | undefined): boolean {
  if (!active || !selected || selected === 'off' || selected === 'auto') return false;
  if (stage === 'enhancer') return isRestorer(selected) && !restorerFileMatches(active.filename, selected);
  return active.model_id !== selected;
}

function Row({ label, stage, active, selected }: { label: string; stage: Stage; active: ActiveModel; selected: string | undefined }) {
  const wanted = selectedText(stage, selected);
  const pending = isPending(stage, active, selected);
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
