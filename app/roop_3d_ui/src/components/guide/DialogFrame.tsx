import { useId, useRef, type ReactNode } from 'react';
import { useDialogA11y } from '../controls/useDialogA11y';

/**
 * The one accessible modal shell: labelled `role="dialog"`, focus moved in and trapped, Escape closes, focus
 * restored to the opener, and a click on the dim backdrop closes. Used by the guide, welcome and theme dialogs.
 */
export default function DialogFrame({ title, eyebrow, onClose, children, className = '', testId, initialFocus, describedBy }: {
  title: string; eyebrow?: string; onClose: () => void; children: ReactNode; className?: string; testId?: string; initialFocus?: string; describedBy?: string;
}) {
  const ref = useRef<HTMLElement>(null);
  const titleId = useId();
  useDialogA11y(ref, onClose, true, initialFocus);
  return (
    <div className="modal-backdrop" data-testid={testId ? `${testId}-backdrop` : undefined} onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
      <section ref={ref} role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={describedBy} className={`project-modal ${className}`} data-testid={testId}>
        <div className="panel-heading">
          <div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2 id={titleId}>{title}</h2></div>
          <button type="button" className="icon-button" aria-label={`Close ${title}`} onClick={onClose}>×</button>
        </div>
        {children}
      </section>
    </div>
  );
}
