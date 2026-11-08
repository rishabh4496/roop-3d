import type { WorkstationParameters } from '../workstation/pipeline';
import { HAIRLINE_RANGE_PX } from './viewerRigs';

export interface IdentityMorphPanelProps {
  parameters: WorkstationParameters;
  setParameters: (value: WorkstationParameters) => void;
  /** Whether the 3D viewer shows the target-geometry ghost and the hairline ring while these controls are edited. */
  preview: boolean;
  onPreview: (value: boolean) => void;
}

/**
 * Live 3D identity morph + alignment controls.
 *
 * - **Jawline / Cranial Volume Blend** is the engine's real mesh blend (`pose_strength` in `mesh_blend` mode): the head volume the
 *   swap is projected onto goes from the target's estimated 2.5D proxy (0%) to the source's fitted 3D mesh (100%). It is applied to the
 *   whole head volume, which includes the jaw and cranium.
 * - **Forehead & Hairline Boundary Offset** (`hairline_offset_px`) moves the upper edge of the swapped region: + grows the swap upward over
 *   the forehead, - pulls it down. It only touches the forehead band of the face matte.
 */
export default function IdentityMorphPanel({ parameters, setParameters, preview, onPreview }: IdentityMorphPanelProps) {
  const meshBlend = parameters.pose_mode !== 'retarget';
  const blend = Math.round((parameters.pose_strength ?? 0.35) * 100);
  const offset = parameters.hairline_offset_px ?? 0;
  const set = (patch: Partial<WorkstationParameters>) => setParameters({ ...parameters, ...patch });
  const setOffset = (value: number) => {
    const { hairline_offset_px: _old, ...rest } = parameters;
    setParameters(value === 0 ? rest : { ...rest, hairline_offset_px: value });   // omitted at 0 so untouched requests stay byte-identical
  };
  return (
    <section className="identity-morph" aria-label="3D identity morph and alignment" data-testid="identity-morph">
      <div className="identity-morph-head"><strong>Identity morph &amp; alignment</strong>
        <label className="inspector-toggle compact"><input type="checkbox" checked={preview} onChange={event => onPreview(event.target.checked)} aria-label="Show morph overlay in the 3D viewer" />Show in 3D view</label>
      </div>

      <label className="morph-field">
        <span>Jawline / Cranial Volume Blend <output aria-live="off">{meshBlend ? `${blend}%` : 'off'}</output></span>
        <input type="range" min={0} max={100} step={1} value={blend} disabled={!meshBlend} aria-label="Jawline and cranial volume blend"
          aria-valuetext={meshBlend ? `${blend}% source mesh geometry` : 'Mesh blending is off'}
          onChange={event => set({ pose_strength: Number(event.target.value) / 100 })} />
        <span className="morph-scale" aria-hidden="true"><small>0% target geometry</small><small>100% source 3D mesh</small></span>
        {!meshBlend && <small className="identity-warn">Pose correction is set to “reference retarget”, which does not use the head volume.{' '}
          <button type="button" className="button small" onClick={() => set({ pose_mode: 'mesh_blend' })}>Use mesh blend</button></small>}
      </label>

      <label className="morph-field">
        <span>Forehead &amp; Hairline Boundary Offset <output aria-live="off">{offset > 0 ? '+' : ''}{offset} px</output></span>
        <input type="range" min={-HAIRLINE_RANGE_PX} max={HAIRLINE_RANGE_PX} step={1} value={offset} aria-label="Forehead and hairline boundary offset"
          aria-valuetext={`${offset > 0 ? 'plus ' : offset < 0 ? 'minus ' : ''}${Math.abs(offset)} pixels`}
          onChange={event => setOffset(Number(event.target.value))} />
        <span className="morph-scale" aria-hidden="true"><small>− pull down</small><small>+ grow upward</small></span>
      </label>
      <div className="morph-actions">
        <button type="button" className="button small" disabled={offset === 0 && blend === 35} onClick={() => { const { hairline_offset_px: _old, ...rest } = parameters; setParameters({ ...rest, pose_strength: 0.35 }); }}>Reset morph</button>
      </div>
    </section>
  );
}
