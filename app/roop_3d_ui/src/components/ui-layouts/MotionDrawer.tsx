import { useId, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { useDialogA11y } from '../controls/useDialogA11y';
import { cn } from '../../lib/utils';

/**
 * Slide-over drawer on a spring. Reuses the workstation's `useDialogA11y` contract (focus moves in, Tab wraps, Esc closes,
 * focus returns to the opener) and portals to <body> because animated screen wrappers create a containing block for `position: fixed`.
 * Reduced motion swaps the slide for an instant fade.
 */
export type DrawerSide = 'right' | 'left' | 'bottom';

export interface MotionDrawerProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  side?: DrawerSide;
  /** CSS width (left/right) or height (bottom). */
  size?: string;
  children?: ReactNode;
  footer?: ReactNode;
  className?: string;
}

const offscreen: Record<DrawerSide, { x?: string; y?: string }> = { right: { x: '100%' }, left: { x: '-100%' }, bottom: { y: '100%' } };

function DrawerPanel({ onClose, title, description, side, size, children, footer, className }: Omit<MotionDrawerProps, 'open'> & { side: DrawerSide }) {
  const ref = useRef<HTMLElement>(null);
  const reduce = useReducedMotion();
  const titleId = useId(), descId = useId();
  useDialogA11y(ref, onClose);
  const sizeStyle = side === 'bottom' ? { height: size } : { width: size };
  return (
    <motion.div
      className="ul-drawer-backdrop" data-testid="drawer-backdrop"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: reduce ? 0 : 0.2 }}
      onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}
    >
      <motion.section
        ref={ref} role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={description ? descId : undefined}
        className={cn('ul-drawer', `ul-drawer--${side}`, className)} style={sizeStyle}
        initial={reduce ? { opacity: 0 } : offscreen[side]} animate={reduce ? { opacity: 1 } : { x: 0, y: 0 }} exit={reduce ? { opacity: 0 } : offscreen[side]}
        transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 380, damping: 38, mass: 0.9 }}
      >
        <header className="ul-drawer__head">
          <div>
            <h2 id={titleId} className="ul-drawer__title">{title}</h2>
            {description && <p id={descId} className="ul-drawer__desc">{description}</p>}
          </div>
          <button type="button" className="ul-drawer__close" onClick={onClose} aria-label={`Close ${title}`}>
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
          </button>
        </header>
        <div className="ul-drawer__body">{children}</div>
        {footer && <footer className="ul-drawer__foot">{footer}</footer>}
      </motion.section>
    </motion.div>
  );
}

export function MotionDrawer({ open, side = 'right', size = side === 'bottom' ? '55vh' : 'min(420px, 92vw)', ...rest }: MotionDrawerProps) {
  if (typeof document === 'undefined') return null;
  return createPortal(
    <AnimatePresence>{open && <DrawerPanel key="drawer" side={side} size={size} {...rest} />}</AnimatePresence>,
    document.body,
  );
}

export default MotionDrawer;
