import { useId } from 'react';
import { useMotionPref, type MotionPref } from '../../utils/motionPrefs';

export interface ZoomControl { value: number; bump: (delta: number) => void; reset: () => void }

/**
 * Display & accessibility preferences. "Reduce motion" stops the interface animations and the 3D viewers' idle
 * rotation and eased camera; "Follow system" tracks the operating-system setting live. The interface size control is
 * the same zoom as the header buttons and Ctrl + / − / 0.
 */
export default function AccessibilityControls({ zoom }: { zoom?: ZoomControl }) {
  const [pref, reduced, setPref] = useMotionPref();
  const id = useId();
  return (
    <section className="a11y-controls" aria-label="Display and accessibility" data-testid="a11y-controls">
      <div className="a11y-row">
        <label htmlFor={`${id}-motion`}>Reduce motion</label>
        <select id={`${id}-motion`} aria-describedby={`${id}-motion-help`} value={pref} onChange={event => setPref(event.target.value as MotionPref)}>
          <option value="system">Follow my system setting</option><option value="reduce">Always reduce</option><option value="full">Never reduce</option>
        </select>
      </div>
      <p id={`${id}-motion-help`} className="muted">Motion is currently <b>{reduced ? 'reduced' : 'on'}</b>. Reduced motion removes pulsing indicators, fades, the idle turntable in the 3D viewers and smooth scrolling. Video playback is not affected.</p>
      {zoom && <div className="a11y-row" role="group" aria-label="Interface size">
        <span>Interface size</span>
        <span className="a11y-zoom">
          <button type="button" className="button small" aria-label="Smaller interface" onClick={() => zoom.bump(-0.05)}>−</button>
          <output aria-live="polite">{Math.round(zoom.value * 100)}%</output>
          <button type="button" className="button small" aria-label="Larger interface" onClick={() => zoom.bump(0.05)}>+</button>
          <button type="button" className="button small" onClick={zoom.reset}>Reset</button>
        </span>
      </div>}
    </section>
  );
}
