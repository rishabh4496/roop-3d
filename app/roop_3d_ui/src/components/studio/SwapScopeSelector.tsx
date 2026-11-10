import type { WorkstationParameters } from '../workstation/pipeline';
import { SWAP_SCOPES, skullBlendOf, swapScopeApplies, swapScopeOf, withSkullBlend, withSwapScope, type SwapScope } from './swapScope';

export interface SwapScopeSelectorProps {
  parameters: WorkstationParameters;
  setParameters: (value: WorkstationParameters) => void;
}

/** Which part of the target's head the source geometry replaces. Real compositor option (`CompositeOptions.swap_scope`). */
export default function SwapScopeSelector({ parameters, setParameters }: SwapScopeSelectorProps) {
  const scope = swapScopeOf(parameters);
  const share = Math.round(skullBlendOf(parameters) * 100);
  const applies = swapScopeApplies(parameters);
  return (
    <div className="swap-scope influence-mode" role="group" aria-labelledby="swap-scope-label" data-testid="swap-scope">
      <span id="swap-scope-label" className="influence-mode-title">Swap scope</span>
      <div className="mode-cards" role="radiogroup" aria-labelledby="swap-scope-label">
        {SWAP_SCOPES.map((item, index) => (
          <button key={item.id} type="button" role="radio" aria-checked={scope === item.id} className={`mode-card ${scope === item.id ? 'active' : ''}`}
            title={item.detail} onClick={() => setParameters(withSwapScope(parameters, item.id as SwapScope))}>
            <b>{item.label}{index === 0 && <small className="muted"> default</small>}</b>
            <small className="muted"> {item.detail}</small>
          </button>
        ))}
      </div>
      {scope === 'adaptive_skull' && (
        <label className="inspector-field">Skull blend <output aria-live="off">β source {share}% · β target {100 - share}%</output>
          <input type="range" aria-label="Source skull share" min={0} max={100} step={1} value={share}
            aria-valuetext={`${share} percent source skull, ${100 - share} percent target skull`}
            onChange={event => setParameters(withSkullBlend(parameters, Number(event.target.value) / 100))} />
          <span className="protection-scale" aria-hidden="true"><small>0 target skull</small><small>100 source skull</small></span>
        </label>)}
      {!applies && (
        <small className="identity-warn" data-testid="swap-scope-applies">
          The swap scope shapes the head silhouette, so it takes effect in a 3D render mode. The plain 2D swap only replaces the inside of the face.
        </small>)}
    </div>
  );
}
