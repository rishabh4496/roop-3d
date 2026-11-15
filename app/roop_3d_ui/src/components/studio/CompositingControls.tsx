import type { WorkstationParameters } from '../workstation/pipeline';

export interface CompositingControlsProps {
  /** Current values (the workspace parameters, or the Batch defaults); only touched fields are present. */
  params: Partial<WorkstationParameters>;
  onParams: (patch: Partial<WorkstationParameters>) => void;
}

/** Stage 4 uses mandatory semantic protection and a fixed four-band compositor. */
export default function CompositingControls({ params, onParams }: CompositingControlsProps) {
  const restoration = params.restoration_weight ?? 0.35;
  return (
    <div className="compositing-controls" role="group" aria-label="Occlusion strength and seam blending" data-testid="compositing-controls">
      <label className="inspector-field">XSeg occlusion strength <output>100%</output>
        <input type="range" aria-label="XSeg occlusion strength" min={0} max={100} value={100} disabled readOnly />
        <small className="muted" data-testid="xseg-strength-note">BiSeNet, XSeg and the face occluder run on every face. Hair, glasses, hands and background cannot be opted into the replacement.</small>
      </label>
      <label className="inspector-field">Seam blending
        <select aria-label="Seam blending" value="multi_band" disabled>
          <option value="multi_band">Multi-band pyramid</option>
        </select>
        <small className="muted" data-testid="blend-mode-note">Distance-transform boundary feathering and four-level Laplacian blending. Other fusion modes are not supported by this renderer.</small>
      </label>
      <label className="inspector-field">Pyramid bands <output>4</output>
        <input type="range" aria-label="Pyramid bands" min={4} max={6} value={4} disabled readOnly />
        <small className="muted" data-testid="pyramid-note">The volumetric compositor uses four bands; manual band selection is not supported.</small>
      </label>
      <label className="inspector-field">UV skin restoration <output>{Math.round(restoration * 100)}%</output>
        <input type="range" aria-label="UV skin restoration" min={0} max={100} step={1} value={Math.round(restoration * 100)}
          onChange={event => onParams({ restoration_weight: Number(event.target.value) / 100 })} />
        <small className="muted">Skin-detail blend in the canonical UV atlas, before 3D projection. Zero disables restoration; source identity and low-frequency colour remain preserved.</small>
      </label>
    </div>
  );
}
