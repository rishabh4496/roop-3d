import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { TOUR_STEPS, markTourSeen, type TourStep } from './tourSteps';
import { useDialogA11y } from '../controls/useDialogA11y';

type Box = { top: number; left: number; width: number; height: number };

/** Small diagram per step. Decorative: the same information is in the text. */
function Illustration({ id }: { id: TourStep['id'] }) {
  const common = { width: 132, height: 84, viewBox: '0 0 132 84', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, 'aria-hidden': true } as const;
  if (id === 'source') return <svg {...common}><rect x="6" y="12" width="40" height="52" rx="5"/><circle cx="26" cy="32" r="9"/><path d="M12 58c3-10 11-14 14-14s11 4 14 14"/><path d="M54 38h20m-7-7 7 7-7 7"/><rect x="82" y="12" width="44" height="52" rx="5" strokeDasharray="3 3"/><path d="M92 28h24M92 38h24M92 48h14"/></svg>;
  if (id === 'target') return <svg {...common}><rect x="6" y="8" width="120" height="68" rx="5"/><rect x="22" y="22" width="34" height="42" rx="4"/><circle cx="33" cy="38" r="1.8" fill="currentColor"/><circle cx="46" cy="38" r="1.8" fill="currentColor"/><circle cx="39.5" cy="46" r="1.8" fill="currentColor"/><circle cx="34" cy="54" r="1.8" fill="currentColor"/><circle cx="45" cy="54" r="1.8" fill="currentColor"/><rect x="72" y="26" width="40" height="38" rx="4" strokeDasharray="3 3"/><text x="62" y="18" fontSize="8" fill="currentColor" stroke="none">Y P R</text></svg>;
  if (id === 'preview') return <svg {...common}><rect x="4" y="30" width="22" height="24" rx="4"/><rect x="32" y="30" width="22" height="24" rx="4"/><rect x="60" y="30" width="22" height="24" rx="4"/><rect x="88" y="30" width="22" height="24" rx="4"/><path d="M26 42h6M54 42h6M82 42h6"/><path d="M71 14v14m-6-6 6 6 6-6" /><rect x="22" y="66" width="88" height="6" rx="3"/><circle cx="66" cy="69" r="5" fill="currentColor"/></svg>;
  return <svg {...common}><rect x="4" y="24" width="32" height="36" rx="5"/><rect x="50" y="24" width="32" height="36" rx="5"/><rect x="96" y="24" width="32" height="36" rx="5"/><path d="M36 42h14m-5-5 5 5-5 5M82 42h14m-5-5 5 5-5 5"/><circle cx="20" cy="42" r="3" fill="currentColor"/><circle cx="66" cy="42" r="3" fill="currentColor"/><circle cx="112" cy="42" r="3" fill="currentColor"/></svg>;
}

function measure(selector: string): Box | null {
  const element = document.querySelector<HTMLElement>(selector);
  if (!element) return null;
  const rect = element.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return null;
  return { top: rect.top, left: rect.left, width: rect.width, height: rect.height };
}

export default function QuickTour({ open, onClose, onNavigate }: { open: boolean; onClose: () => void; onNavigate: (route: string) => void }) {
  const [index, setIndex] = useState(0);
  const [box, setBox] = useState<Box | null>(null);
  const next = useRef<HTMLButtonElement>(null);
  const navigate = useRef(onNavigate);
  navigate.current = onNavigate;
  const step = TOUR_STEPS[index];
  const last = index === TOUR_STEPS.length - 1;

  const close = useCallback(() => { markTourSeen(); setIndex(0); onClose(); }, [onClose]);
  const card = useRef<HTMLElement>(null);
  useDialogA11y(card, close, open, '[data-tour-next]');

  // Move to the step's screen, then (re)measure its highlight target.
  useEffect(() => {
    if (!open) return;
    navigate.current(step.route);
  }, [open, step]);

  useLayoutEffect(() => {
    if (!open) return;
    const update = () => setBox(measure(step.target));
    update();
    // The target mounts after navigation, so look again shortly and on layout changes.
    const timer = window.setTimeout(update, 120);
    window.addEventListener('resize', update);
    window.addEventListener('scroll', update, true);
    return () => { window.clearTimeout(timer); window.removeEventListener('resize', update); window.removeEventListener('scroll', update, true); };
  }, [open, step]);

  useEffect(() => {
    if (!open) return;
    // Escape, focus and the Tab trap come from useDialogA11y; here only the arrow keys step through the tour.
    const onKey = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement | null)?.closest?.('input,select,textarea')) return;
      if (event.key === 'ArrowRight') setIndex(value => Math.min(TOUR_STEPS.length - 1, value + 1));
      else if (event.key === 'ArrowLeft') setIndex(value => Math.max(0, value - 1));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  if (!open) return null;
  const pad = 6;
  return <div className="tour-layer" data-testid="quick-tour">
    {box && <div className="tour-spotlight" data-testid="tour-spotlight" style={{ top: box.top - pad, left: box.left - pad, width: box.width + pad * 2, height: box.height + pad * 2 }} />}
    {!box && <div className="tour-dim" />}
    <section ref={card} className="tour-card" role="dialog" aria-modal="true" aria-labelledby="tour-title" data-step={step.id}>
      <header>
        <span className="eyebrow">QUICK TOUR · STEP {index + 1} OF {TOUR_STEPS.length}</span>
        <button className="icon-button" aria-label="Close tour" onClick={close}>×</button>
      </header>
      <div className="tour-illustration"><Illustration id={step.id} /></div>
      <h2 id="tour-title">{step.title}</h2>
      <p className="muted">{step.summary}</p>
      <ul>{step.points.map(point => <li key={point}>{point}</li>)}</ul>
      {!box && <small className="muted tour-note">This part of the workspace is not on screen yet; the steps still apply once a target video is loaded.</small>}
      <ol className="tour-dots" aria-label="Tour progress">{TOUR_STEPS.map((item, position) => <li key={item.id}><button aria-label={`Go to step ${position + 1}: ${item.title}`} aria-current={position === index ? 'step' : undefined} className={position === index ? 'active' : ''} onClick={() => setIndex(position)} /></li>)}</ol>
      <footer>
        <button className="button" disabled={index === 0} onClick={() => setIndex(index - 1)}>Back</button>
        <button className="button" onClick={close}>Skip</button>
        <button ref={next} data-tour-next className="button accent" onClick={() => (last ? close() : setIndex(index + 1))}>{last ? 'Done' : 'Next'}</button>
      </footer>
    </section>
  </div>;
}
