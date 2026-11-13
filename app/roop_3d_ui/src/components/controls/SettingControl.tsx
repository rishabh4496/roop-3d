import { useId } from 'react';
import { renderProbe } from '../../utils/renderProbe';
import NumericInput from './NumericInput';
import { recommendedText, type SettingSpec } from './settingsRegistry';

type Props = {
  spec: SettingSpec;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  /** Why the control is unavailable right now (shown in the help and as the disabled tooltip). */
  disabledReason?: string;
  /** Overrides `spec.label` (a setting can mean something different in another mode). */
  label?: string;
};

/**
 * One self-documenting setting: drag-to-scrub label with inline expression math, a slider, one-click Reset to
 * default, and hover/focus help that says what you will see change and what range usually looks right.
 */
export default function SettingControl({ spec, value, onChange, disabled = false, disabledReason, label }: Props) {
  const tipId = useId();
  renderProbe('SettingControl');
  const name = label ?? spec.label;
  const unavailable = disabled;
  const reason = disabledReason;
  const atDefault = Math.abs(value - spec.default) < spec.step / 2;
  return (
    <div className={`setting-control ${unavailable ? 'unavailable' : ''}`} data-setting={spec.id}>
      <NumericInput label={name} value={value} min={spec.min} max={spec.max} step={spec.step} disabled={unavailable} title={reason}
        onChange={next => onChange(next ?? spec.default)} />
      <input type="range" className="setting-slider" aria-label={`${name} slider`} min={spec.min} max={spec.max} step={spec.step}
        value={value} disabled={unavailable} title={reason} onChange={event => onChange(Number(event.target.value))} />
      <button type="button" className="reset-setting" aria-label={`Reset ${name} to default (${spec.default})`} title={`Reset to ${spec.default}${spec.unit ?? ''}`}
        disabled={unavailable || atDefault} onClick={() => onChange(spec.default)}>↺</button>
      <span className="setting-help">
        <button type="button" className="help-badge" aria-label={`About ${name}`} aria-describedby={tipId}>?</button>
        <span role="tooltip" id={tipId} className="setting-tooltip">
          <strong>{name}</strong>
          <span className="tip-effect">{spec.effect}</span>
          <span className="tip-range">Recommended: {recommendedText(spec)} · Default: {spec.default}{spec.unit ? ` ${spec.unit}` : ''} · Range: {spec.min}–{spec.max}</span>
          {reason && <em className="tip-warning">{reason}</em>}
        </span>
      </span>
    </div>
  );
}
