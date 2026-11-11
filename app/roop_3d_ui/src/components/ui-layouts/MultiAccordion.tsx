import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { cn } from '../../lib/utils';

/**
 * Multi-expand accordion with a height animation, for dense parameter groups. Any number of sections can be open at once.
 * Keyboard: Up/Down move between headers, Home/End jump to the first/last, Enter/Space toggle (native button behaviour).
 * Works uncontrolled (`defaultOpen`) or controlled (`open` + `onOpenChange`).
 */
export interface AccordionItem {
  id: string;
  title: ReactNode;
  content: ReactNode;
  /** Small right-aligned status text, e.g. a value summary or a count. */
  badge?: ReactNode;
  disabled?: boolean;
}

export interface MultiAccordionProps {
  items: AccordionItem[];
  defaultOpen?: string[];
  open?: string[];
  onOpenChange?: (open: string[]) => void;
  className?: string;
  'aria-label'?: string;
}

export function MultiAccordion({ items, defaultOpen = [], open, onOpenChange, className, 'aria-label': ariaLabel }: MultiAccordionProps) {
  const base = useId();
  const reduce = useReducedMotion();
  const [inner, setInner] = useState<string[]>(defaultOpen);
  const current = open ?? inner;
  const headers = useRef<(HTMLButtonElement | null)[]>([]);

  const toggle = (id: string) => {
    const next = current.includes(id) ? current.filter(x => x !== id) : [...current, id];
    if (open === undefined) setInner(next);
    onOpenChange?.(next);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const enabled = items.map((item, i) => (item.disabled ? -1 : i)).filter(i => i >= 0);
    if (!enabled.length) return;
    const at = enabled.indexOf(index);
    let target: number | undefined;
    if (event.key === 'ArrowDown') target = enabled[(at + 1) % enabled.length];
    else if (event.key === 'ArrowUp') target = enabled[(at - 1 + enabled.length) % enabled.length];
    else if (event.key === 'Home') target = enabled[0];
    else if (event.key === 'End') target = enabled[enabled.length - 1];
    if (target === undefined) return;
    event.preventDefault();
    headers.current[target]?.focus();
  };

  return (
    <div className={cn('ul-accordion', className)} role="group" aria-label={ariaLabel}>
      {items.map((item, index) => {
        const isOpen = current.includes(item.id);
        const headerId = `${base}-h-${item.id}`, panelId = `${base}-p-${item.id}`;
        return (
          <section key={item.id} className="ul-accordion__item" data-open={isOpen ? 'true' : 'false'}>
            <h3 className="ul-accordion__heading">
              <button
                ref={node => { headers.current[index] = node; }}
                id={headerId} type="button" className="ul-accordion__trigger"
                aria-expanded={isOpen} aria-controls={panelId} disabled={item.disabled}
                onClick={() => toggle(item.id)} onKeyDown={event => onKeyDown(event, index)}
              >
                <span className="ul-accordion__title">{item.title}</span>
                {item.badge != null && <span className="ul-accordion__badge ul-number">{item.badge}</span>}
                <svg className="ul-accordion__chevron" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
              </button>
            </h3>
            <AnimatePresence initial={false}>
              {isOpen && (
                <motion.div
                  key="panel" id={panelId} role="region" aria-labelledby={headerId} className="ul-accordion__panel"
                  initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
                  transition={reduce ? { duration: 0 } : { height: { type: 'spring', stiffness: 420, damping: 38 }, opacity: { duration: 0.18 } }}
                >
                  <div className="ul-accordion__content">{item.content}</div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        );
      })}
    </div>
  );
}

export default MultiAccordion;
