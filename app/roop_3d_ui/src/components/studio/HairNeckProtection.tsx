import type { WorkstationParameters } from '../workstation/pipeline';
import {
  HAIRLINE_INSET_RANGE_PX, JAW_FEATHER_RANGE_PX, hairlineInset, matteProtectionApplies, protectionTouched, protectionValue,
  resetProtection, withHairlineInset, withProtection, type ProtectionKey,
} from './protection';

export interface HairNeckProtectionProps {
  parameters: WorkstationParameters;
  setParameters: (value: WorkstationParameters) => void;
  /** The enhancer that will really run; the pore slider only has an effect after UltraMax. */
  enhancer: string;
}

const TOGGLES: readonly { key: Extract<ProtectionKey, 'neck_protection' | 'hair_matting' | 'preserve_ears_glasses'>; label: string; label2?: string; hint: string }[] = [
  { key: 'neck_protection', label: 'Strict Neck & Throat Protection', label2: 'Clamp to 3D Jawline',
    hint: 'Keeps neck and clothing out of the swap and cuts the matte along the jaw ridge.' },
  { key: 'hair_matting', label: 'Hair-Strand Guided Matting', label2: 'Prevent Hairline Smudge',
    hint: 'Steers the forehead matte by the target\'s own hair strands instead of blurring across them.' },
  { key: 'preserve_ears_glasses', label: 'Preserve Target Ears & Eyeglasses',
    hint: 'Ears and glasses stay the target\'s. Off lets the swap cover them.' },
];

/** Hair & neck protection controls for the Masking & occlusion step. Each one is a real compositor option (`CompositeOptions`). */
export default function HairNeckProtection({ parameters, setParameters, enhancer }: HairNeckProtectionProps) {
  const inset = hairlineInset(parameters);
  const jaw = protectionValue(parameters, 'jaw_feather_px');
  const pores = Math.round(protectionValue(parameters, 'skin_texture_retention') * 100);
  const matteApplies = matteProtectionApplies(parameters);
  const poresApply = enhancer === 'ultramax';
  const sign = (value: number) => (value > 0 ? '+' : '');
  return (
    <div className="hair-neck-protection" role="group" aria-label="Hair and neck protection" data-testid="hair-neck-protection">
      <div className="protection-head"><strong>Hair &amp; neck protection</strong>
        <button type="button" className="button small" disabled={!protectionTouched(parameters)} onClick={() => setParameters(resetProtection(parameters))}>Reset</button>
      </div>
      {TOGGLES.map(item => (
        <label key={item.key} className="inspector-toggle" title={item.hint}>
          <input type="checkbox" aria-label={item.label} checked={protectionValue(parameters, item.key)}
            onChange={event => setParameters(withProtection(parameters, item.key, event.target.checked))} />
          <span>{item.label}{item.label2 && <small className="muted"> {item.label2}</small>}</span>
        </label>
      ))}
      {!matteApplies && (
        <small className="identity-warn" data-testid="protection-scope">
          The jaw clamp, hairline matting and jaw feather shape the 3D-projected matte: they take effect in a 3D render mode or with XSeg + Smart Occluder on.
          The plain 2D swap uses the face-parser matte only.
        </small>)}

      <label className="inspector-field">Forehead Hairline Inset <output aria-live="off">{sign(inset)}{inset} px</output>
        <input type="range" aria-label="Forehead hairline inset" min={-HAIRLINE_INSET_RANGE_PX} max={HAIRLINE_INSET_RANGE_PX} step={1} value={inset}
          aria-valuetext={`${inset > 0 ? 'plus ' : inset < 0 ? 'minus ' : ''}${Math.abs(inset)} pixels`}
          onChange={event => setParameters(withHairlineInset(parameters, Number(event.target.value)))} />
        <span className="protection-scale" aria-hidden="true"><small>− grow up onto the forehead</small><small>+ pull down off the hair</small></span>
      </label>

      <label className="inspector-field">Jaw-to-Neck Feather Falloff <output aria-live="off">{jaw} px</output>
        <input type="range" aria-label="Jaw-to-neck feather falloff" min={JAW_FEATHER_RANGE_PX[0]} max={JAW_FEATHER_RANGE_PX[1]} step={1} value={jaw}
          disabled={!protectionValue(parameters, 'neck_protection')}
          onChange={event => setParameters(withProtection(parameters, 'jaw_feather_px', Number(event.target.value)))} />
        <small className="muted">{protectionValue(parameters, 'neck_protection') ? 'How softly the swap fades out inward along the jaw. Engine default 6 px.' : 'Needs Strict Neck & Throat Protection.'}</small>
      </label>

      <label className="inspector-field">Real Skin Micro-Texture &amp; Pore Retention <output aria-live="off">{pores}%</output>
        <input type="range" aria-label="Skin micro-texture and pore retention" min={0} max={100} step={1} value={pores} disabled={!poresApply}
          onChange={event => setParameters(withProtection(parameters, 'skin_texture_retention', Number(event.target.value) / 100))} />
        <small className={poresApply ? 'muted' : 'identity-warn'}>
          {poresApply ? 'Share of the target\'s own pore and specular detail laid back over UltraMax skin. Engine default 25%.'
            : 'Applies to the UltraMax enhancer only: pick UltraMax High-Detail in Face restoration.'}
        </small>
      </label>
    </div>
  );
}
