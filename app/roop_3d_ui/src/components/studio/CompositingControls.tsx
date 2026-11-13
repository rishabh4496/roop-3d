import { useSelectedEngine } from '../../stores/engineSelection';
import type { BlendMode, WorkstationParameters } from '../workstation/pipeline';
import {
  BLEND_MODES, PHOTOREAL_ENGINE, PYRAMID_LEVEL_RANGE, blendModeOf, frequencyBlockedReason, pyramidInactiveReason, pyramidLevelsOf,
  xsegInactiveReason, xsegStrengthOf,
} from './compositing';

export interface CompositingControlsProps {
  /** Current values (the workspace parameters, or the Batch defaults); only touched fields are present. */
  params: Partial<WorkstationParameters>;
  onParams: (patch: Partial<WorkstationParameters>) => void;
}

/**
 * The compositor's occlusion strength and seam fusion. Each control is a real `WorkstationParameters` field that reaches
 * `CompositeOptions` (`xseg_occlusion_strength` -> `OcclusionAnalyzer.analyze`, `blend_mode` / `pyramid_levels` -> `fuse_crop` / `fuse_warped`),
 * and a control the current selection cannot act on is disabled with the reason instead of being sent and ignored.
 */
export default function CompositingControls({ params, onParams }: CompositingControlsProps) {
  const engine = useSelectedEngine();
  const strength = Math.round(xsegStrengthOf(params) * 100);
  const strengthReason = xsegInactiveReason(params);
  const mode = blendModeOf(params);
  const levels = pyramidLevelsOf(params);
  const levelsReason = pyramidInactiveReason(params);
  return (
    <div className="compositing-controls" role="group" aria-label="Occlusion strength and seam blending" data-testid="compositing-controls">
      <label className="inspector-field">XSeg occlusion strength <output aria-live="off">{strength}%</output>
        <input type="range" aria-label="XSeg occlusion strength" min={0} max={100} step={1} value={strength} disabled={!!strengthReason}
          aria-valuetext={`${strength} percent`} title={strengthReason ?? undefined}
          onChange={event => onParams({ xseg_occlusion_strength: Number(event.target.value) / 100 })} />
        <small className={strengthReason ? 'identity-warn' : 'muted'} data-testid="xseg-strength-note">
          {strengthReason ?? 'How much the XSeg and face-occluder networks cut hands, hair strands, glasses and props out of the swap. 0 ignores the networks; the face parser and your manual mask always apply in full.'}
        </small>
      </label>

      <label className="inspector-field">Seam blending
        <select aria-label="Seam blending" value={mode} onChange={event => onParams({ blend_mode: event.target.value as BlendMode })}>
          {BLEND_MODES.map(item => {
            const blocked = item.id === 'frequency_separated' ? frequencyBlockedReason(params) : null;
            return <option key={item.id} value={item.id} disabled={!!blocked} title={blocked ?? item.hint}>{item.label}{blocked ? ' (unavailable)' : ''}</option>;
          })}
        </select>
        <small className="muted" data-testid="blend-mode-note">{BLEND_MODES.find(item => item.id === mode)?.hint}</small>
        {frequencyBlockedReason(params) && <small className="muted">Frequency-separated: {frequencyBlockedReason(params)}</small>}
      </label>

      <label className="inspector-field">Pyramid bands <output aria-live="off">{levels}</output>
        <input type="range" aria-label="Pyramid bands" min={PYRAMID_LEVEL_RANGE[0]} max={PYRAMID_LEVEL_RANGE[1]} step={1} value={levels} disabled={!!levelsReason}
          title={levelsReason ?? undefined} onChange={event => onParams({ pyramid_levels: Number(event.target.value) })} />
        <small className={levelsReason ? 'identity-warn' : 'muted'} data-testid="pyramid-note">
          {levelsReason ?? 'More bands widen the low-frequency cross-fade across the seam. Engine default 4.'}
        </small>
      </label>
      {engine === PHOTOREAL_ENGINE && (
        <small className="identity-warn" data-testid="photoreal-blend-note">
          The Photoreal engine fuses with its own 4–6 band Laplacian pyramid: it follows the band count but ignores the seam blending mode.
        </small>)}
    </div>
  );
}
