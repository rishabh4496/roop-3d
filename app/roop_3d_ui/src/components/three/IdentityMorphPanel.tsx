import type { WorkstationParameters } from '../workstation/pipeline';

export interface IdentityMorphPanelProps {
  parameters: WorkstationParameters;
  setParameters: (value: WorkstationParameters) => void;
  /** Whether the 3D viewer shows the hairline ring while the matte controls are edited. */
  preview: boolean;
  onPreview: (value: boolean) => void;
}

/**
 * Live 3D identity overlay. The head is always the source identity's full 3D mesh (there is no blend with a target proxy), so the
 * only control left here is the viewer overlay that shows the forehead hairline ring edited in the Masking step. Semantic shape
 * sliders (jaw width, chin projection ...) live with the topology controls.
 */
export default function IdentityMorphPanel({ preview, onPreview }: IdentityMorphPanelProps) {
  return (
    <section className="identity-morph" aria-label="3D identity overlay" data-testid="identity-morph">
      <div className="identity-morph-head"><strong>3D identity overlay</strong>
        <label className="inspector-toggle compact"><input type="checkbox" checked={preview} onChange={event => onPreview(event.target.checked)} aria-label="Show morph overlay in the 3D viewer" />Show in 3D view</label>
      </div>
      <small className="muted">The render uses the source identity's full 3D head. The overlay marks the hairline boundary set in Masking &amp; occlusion.</small>
    </section>
  );
}
