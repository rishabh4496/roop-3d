import type { WorkstationParameters } from '../workstation/pipeline';
import { PrecisionRangeSlider } from '../ui-layouts';

export interface IdentityMorphPanelProps {
  parameters: WorkstationParameters;
  setParameters: (value: WorkstationParameters) => void;
  /** Whether the 3D viewer shows the target-geometry ghost and the hairline ring while these controls are edited. */
  preview: boolean;
  onPreview: (value: boolean) => void;
}

export const INFLUENCE_MODES = [
  { id: 'retarget', label: '3D Pose-Guided Neural Swap', hint: 'The neural swapper generates the skin; the 3D pose only guides it. No source geometry is projected.' },
  { id: 'mesh_blend', label: '3D Geometry + Jaw Reshape Weight', hint: 'Projects the swap onto a 3D head volume; the weight below sets how much of it comes from the source mesh.' },
] as const;

/**
 * Live 3D identity controls.
 *
 * - **3D Influence Mode** is the engine's real `pose_mode`: `retarget` (pose-guided neural swap) or `mesh_blend` (3D geometry).
 * - **Jaw Reshape Weight** (Jawline / Cranial Volume Blend) is the engine's mesh blend, `pose_strength` in `mesh_blend` mode: the head volume
 *   the swap is projected onto goes from the target's estimated 2.5D proxy (0%) to the source's fitted 3D mesh (100%). It is applied to the
 *   whole head volume, which includes the jaw and cranium. The forehead hairline inset moved to the Masking step with the other matte controls.
 */
export default function IdentityMorphPanel({ parameters, setParameters, preview, onPreview }: IdentityMorphPanelProps) {
  const meshBlend = parameters.pose_mode !== 'retarget';
  const blend = Math.round((parameters.pose_strength ?? 0.35) * 100);
  const set = (patch: Partial<WorkstationParameters>) => setParameters({ ...parameters, ...patch });
  return (
    <section className="identity-morph" aria-label="3D identity morph and influence" data-testid="identity-morph">
      <div className="identity-morph-head"><strong>3D influence &amp; jaw reshape</strong>
        <label className="inspector-toggle compact"><input type="checkbox" checked={preview} onChange={event => onPreview(event.target.checked)} aria-label="Show morph overlay in the 3D viewer" />Show in 3D view</label>
      </div>

      <div className="influence-mode">
        <span id="influence-mode-label" className="influence-mode-title">3D Influence Mode</span>
        <div className="mode-cards" role="radiogroup" aria-labelledby="influence-mode-label">
          {INFLUENCE_MODES.map(mode => (
            <button key={mode.id} type="button" role="radio" aria-checked={(mode.id === 'mesh_blend') === meshBlend} className={`mode-card ${(mode.id === 'mesh_blend') === meshBlend ? 'active' : ''}`}
              title={mode.hint} onClick={() => set({ pose_mode: mode.id })}>
              <b>{mode.label}</b><small className="muted"> {mode.hint}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="morph-field">
        <PrecisionRangeSlider label="Jaw Reshape Weight" ariaLabel="Jawline and cranial volume blend" min={0} max={100} step={1} unit="%" disabled={!meshBlend}
          value={blend} defaultValue={35} onChange={value => set({ pose_strength: value / 100 })}
          title={meshBlend ? undefined : 'Geometry mode is off'} />
        <span className="morph-scale" aria-hidden="true"><small>0% target geometry</small><small>100% source 3D mesh</small></span>
        {!meshBlend && <small className="muted">Used only in “3D Geometry + Jaw Reshape Weight” mode.</small>}
      </div>
      <div className="morph-actions">
        <button type="button" className="button small" disabled={blend === 35} onClick={() => set({ pose_strength: 0.35 })}>Reset weight</button>
      </div>
    </section>
  );
}
