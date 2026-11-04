import { useRef, type CSSProperties, type ReactNode } from 'react';
import { useDialogA11y } from '../controls/useDialogA11y';

/**
 * The dialog surface only (no backdrop), for dialogs that already render their own backdrop: labelled, modal, focus
 * moved in and trapped, Escape closes, focus restored to the opener. Clicks inside do not reach the backdrop.
 */
export default function DialogPanel({ onClose, label, labelledBy, className, style, children, initialFocus, testId }: {
  onClose: () => void; label?: string; labelledBy?: string; className?: string; style?: CSSProperties; children: ReactNode; initialFocus?: string; testId?: string;
}) {
  const ref = useRef<HTMLElement>(null);
  useDialogA11y(ref, onClose, true, initialFocus);
  return <section ref={ref} role="dialog" aria-modal="true" aria-label={label} aria-labelledby={labelledBy} className={className} style={style} data-testid={testId} onClick={event => event.stopPropagation()}>{children}</section>;
}
