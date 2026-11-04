import { ENGINE_STATUS, recommendEngine, type EngineLike } from './helpContent';

/**
 * Explains the engine list in the Inspector: which engine Roop 3D recommends here and why, what the chosen
 * engine's status means, and why the unavailable ones cannot be selected. Never recommends an unselectable engine.
 */
export default function EngineExplainer({ engines, selectedId, onSelect }: { engines: EngineLike[]; selectedId: string; onSelect: (id: string) => void }) {
  const { engine: recommended, reason } = recommendEngine(engines);
  const selected = engines.find(item => item.engine_id === selectedId);
  const status = selected ? ENGINE_STATUS[selected.status] : undefined;
  return (
    <div className="engine-explainer" data-testid="engine-explainer">
      {recommended
        ? <p><span className="status-chip ok">Recommended</span> <strong>{recommended.name}</strong>. {reason}
          {recommended.engine_id !== selectedId && <> <button type="button" className="button small" onClick={() => onSelect(recommended.engine_id)}>Use recommended engine</button></>}</p>
        : <p role="status" className="identity-warn">{reason}</p>}
      {selected && <p className="muted"><strong>{selected.name}:</strong> {status ? `${status.label}. ${status.meaning}` : selected.status}{selected.purpose ? ` ${selected.purpose}` : ''}</p>}
      {engines.some(item => !item.selectable) && <details><summary>Why can’t I select the other engines?</summary>
        <ul>{engines.filter(item => !item.selectable).map(item => <li key={item.engine_id}><strong>{item.name}</strong>: {(ENGINE_STATUS[item.status]?.label ?? item.status)}. {item.evidence ?? ''}</li>)}</ul></details>}
    </div>
  );
}
