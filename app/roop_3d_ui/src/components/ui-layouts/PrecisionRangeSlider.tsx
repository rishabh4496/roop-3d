import { useId, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { clamp, cn, decimalsOf } from '../../lib/utils';
import { formatNumber } from './MotionNumber';

/**
 * Scrubbable numeric range slider for dense 3D parameters (gaze, jaw, light azimuth, blend strength).
 *
 *  - Drag the track, or drag the numeric readout sideways to scrub (hold Shift for 10x finer control).
 *  - Keyboard: arrows = 1 step, Shift+arrows or PageUp/PageDown = 10 steps, Home/End = min/max, Escape/Backspace = reset.
 *  - Double-click the track or the readout to reset to `defaultValue`.
 *  - The fill is live, and starts at zero when the range straddles it (signed angles fill outward from the centre).
 *  - With `sides`, a Link toggle splits the control into independent Left / Right tracks (face symmetry). Linked, one track drives both.
 */
interface BaseProps {
  label: string;
  /** Accessible name when it should be longer than the visible label. Defaults to `label`. */
  ariaLabel?: string;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  /** Fraction digits shown. Default: derived from `step`. */
  decimals?: number;
  disabled?: boolean;
  /** Native tooltip for the whole control (e.g. why it is disabled). */
  title?: string;
  className?: string;
}

export interface SingleSliderProps extends BaseProps {
  value: number;
  onChange: (value: number) => void;
  /** Double-click resets to this. Default: 0 if inside the range, else `min`. */
  defaultValue?: number;
  sides?: undefined;
}

export interface Sides { left: number; right: number }

export interface SplitSliderProps extends BaseProps {
  sides: Sides;
  onSidesChange: (sides: Sides) => void;
  defaultSides?: Sides;
  /** Symmetry lock. Linked = both sides move together. Default true. */
  linked?: boolean;
  onLinkedChange?: (linked: boolean) => void;
  defaultLinked?: boolean;
  value?: undefined;
}

export type PrecisionRangeSliderProps = SingleSliderProps | SplitSliderProps;

/** Snaps to the step grid anchored at `min`, clamps, and removes float noise (0.1 + 0.2). */
export function snapToStep(value: number, min: number, max: number, step: number): number {
  const clamped = clamp(value, min, max);
  if (!(step > 0)) return clamped;
  const snapped = Math.round((clamped - min) / step) * step + min;
  return clamp(Number(snapped.toFixed(Math.max(decimalsOf(step), decimalsOf(min)) + 2)), min, max);
}

/** Fill segment as percentages of the track: from the zero line when the range straddles it, else from the minimum. */
export function fillSegment(value: number, min: number, max: number): { start: number; size: number } {
  const span = max - min || 1;
  const pct = ((clamp(value, min, max) - min) / span) * 100;
  const origin = min < 0 && max > 0 ? ((0 - min) / span) * 100 : 0;
  return { start: Math.min(origin, pct), size: Math.abs(pct - origin) };
}

interface TrackProps extends BaseProps {
  value: number;
  onChange: (value: number) => void;
  resetTo: number;
  tag?: string;
}

function Track({ label, ariaLabel, tag, min, max, step = 1, unit, decimals, disabled, value, onChange, resetTo }: TrackProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const scrub = useRef<{ x: number; start: number } | null>(null);
  const places = decimals ?? decimalsOf(step);
  const text = formatNumber(value, { decimals: places, suffix: unit ?? '' });
  const { start, size } = fillSegment(value, min, max);
  const pct = ((clamp(value, min, max) - min) / (max - min || 1)) * 100;
  // Only real changes reach the parent: dragging within one step must not fire a stream of identical updates (a native <input type=range> behaves the same).
  const commit = (next: number) => { const snapped = snapToStep(next, min, max, step); if (snapped !== value) onChange(snapped); };
  const name = tag ? `${ariaLabel ?? label} ${tag}` : (ariaLabel ?? label);

  const fromClientX = (clientX: number) => {
    const box = trackRef.current?.getBoundingClientRect();
    if (!box || box.width <= 0) return;
    commit(min + ((clientX - box.left) / box.width) * (max - min));
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    const big = event.shiftKey ? 10 : 1;
    let next: number | undefined;
    if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = value + step * big;
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = value - step * big;
    else if (event.key === 'PageUp') next = value + step * 10;
    else if (event.key === 'PageDown') next = value - step * 10;
    else if (event.key === 'Home') next = min;
    else if (event.key === 'End') next = max;
    else if (event.key === 'Escape' || event.key === 'Backspace') next = resetTo;
    if (next === undefined) return;
    event.preventDefault();
    commit(next);
  };

  const onTrackDown = (event: PointerEvent<HTMLDivElement>) => {
    if (disabled || event.button !== 0) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    scrub.current = { x: event.clientX, start: value };
    fromClientX(event.clientX);
  };
  const onTrackMove = (event: PointerEvent<HTMLDivElement>) => { if (scrub.current) fromClientX(event.clientX); };
  const onTrackUp = (event: PointerEvent<HTMLDivElement>) => { scrub.current = null; event.currentTarget.releasePointerCapture?.(event.pointerId); };

  // Scrubbing the readout is relative: 1 px = range/300, so the whole range takes ~300px and Shift gives 10x finer control.
  const onReadoutDown = (event: PointerEvent<HTMLSpanElement>) => {
    if (disabled || event.button !== 0) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    scrub.current = { x: event.clientX, start: value };
  };
  const onReadoutMove = (event: PointerEvent<HTMLSpanElement>) => {
    const s = scrub.current; if (!s) return;
    const perPx = (max - min) / 300 / (event.shiftKey ? 10 : 1);
    commit(s.start + (event.clientX - s.x) * perPx);
  };
  const onReadoutUp = (event: PointerEvent<HTMLSpanElement>) => { scrub.current = null; event.currentTarget.releasePointerCapture?.(event.pointerId); };

  return (
    <div className="ul-slider__row" data-disabled={disabled ? 'true' : 'false'}>
      {tag && <span className="ul-slider__tag" aria-hidden="true">{tag.slice(0, 1)}</span>}
      <div
        ref={trackRef} className="ul-slider__track" data-testid="slider-track"
        onPointerDown={onTrackDown} onPointerMove={onTrackMove} onPointerUp={onTrackUp} onPointerCancel={() => { scrub.current = null; }}
        onDoubleClick={() => { if (!disabled) commit(resetTo); }}
      >
        <span className="ul-slider__fill" style={{ left: `${start}%`, width: `${size}%` }} data-testid="slider-fill" />
        <span
          className="ul-slider__thumb" style={{ left: `${pct}%` }}
          role="slider" tabIndex={disabled ? -1 : 0} aria-label={name} aria-disabled={disabled || undefined}
          aria-valuemin={min} aria-valuemax={max} aria-valuenow={value} aria-valuetext={text}
          onKeyDown={onKeyDown}
        />
      </div>
      <span
        className="ul-slider__readout ul-number" title="Drag to scrub (Shift = fine). Double-click to reset."
        onPointerDown={onReadoutDown} onPointerMove={onReadoutMove} onPointerUp={onReadoutUp} onPointerCancel={() => { scrub.current = null; }}
        onDoubleClick={() => { if (!disabled) commit(resetTo); }} aria-hidden="true"
      >{text}</span>
    </div>
  );
}

export function PrecisionRangeSlider(props: PrecisionRangeSliderProps) {
  const groupId = useId();
  const { label, className, min, max, disabled } = props;
  const { title } = props;
  const base: BaseProps = { label, ariaLabel: props.ariaLabel, min, max, step: props.step, unit: props.unit, decimals: props.decimals, disabled };
  const fallback = min <= 0 && max >= 0 ? 0 : min;
  const [innerLinked, setInnerLinked] = useState(props.sides ? props.defaultLinked ?? true : true);

  if (!props.sides) {
    const resetTo = props.defaultValue ?? fallback;
    return (
      <div className={cn('ul-slider', className)} title={title}>
        <span className="ul-slider__label" id={groupId}>{label}</span>
        <Track {...base} value={props.value} onChange={props.onChange} resetTo={resetTo} />
      </div>
    );
  }

  const { sides, onSidesChange } = props;
  const linked = props.linked ?? innerLinked;
  const defaults = props.defaultSides ?? { left: fallback, right: fallback };
  const setLinked = (next: boolean) => {
    if (props.linked === undefined) setInnerLinked(next);
    props.onLinkedChange?.(next);
    // Re-linking collapses the two sides onto their mean, so the single track shows the truth.
    if (next && sides.left !== sides.right) {
      const mean = snapToStep((sides.left + sides.right) / 2, min, max, props.step ?? 1);
      onSidesChange({ left: mean, right: mean });
    }
  };

  return (
    <div className={cn('ul-slider ul-slider--split', className)} title={title} data-linked={linked ? 'true' : 'false'}>
      <div className="ul-slider__head">
        <span className="ul-slider__label" id={groupId}>{label}</span>
        <button
          type="button" className="ul-slider__link" aria-pressed={linked} disabled={disabled}
          aria-label={`Link left and right ${label}`} title={linked ? 'Left and right move together. Click to split.' : 'Left and right are independent. Click to link.'}
          onClick={() => setLinked(!linked)}
        >{linked ? 'L = R' : 'L | R'}</button>
      </div>
      {linked ? (
        <Track {...base} value={sides.left} resetTo={defaults.left} onChange={v => onSidesChange({ left: v, right: v })} />
      ) : (
        <>
          <Track {...base} tag="Left" value={sides.left} resetTo={defaults.left} onChange={v => onSidesChange({ ...sides, left: v })} />
          <Track {...base} tag="Right" value={sides.right} resetTo={defaults.right} onChange={v => onSidesChange({ ...sides, right: v })} />
        </>
      )}
    </div>
  );
}

export default PrecisionRangeSlider;
