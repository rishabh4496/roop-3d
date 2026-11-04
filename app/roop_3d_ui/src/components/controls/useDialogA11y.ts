import { useEffect, useRef, type RefObject } from 'react';

const FOCUSABLE = 'a[href],button:not(:disabled),input:not(:disabled):not([type=hidden]),select:not(:disabled),textarea:not(:disabled),summary,[tabindex]:not([tabindex="-1"])';

/** Visible, enabled, tabbable controls inside `root`, in DOM order. Exported so tests can assert the trap's contract. */
export function focusableIn(root: HTMLElement | null): HTMLElement[] {
  if (!root) return [];
  return [...root.querySelectorAll<HTMLElement>(FOCUSABLE)].filter(element => {
    if (element.closest('[hidden],[inert]')) return false;
    const style = getComputedStyle(element);
    return style.visibility !== 'hidden' && style.display !== 'none' && (element.offsetWidth > 0 || element.offsetHeight > 0 || element.getClientRects().length > 0);
  });
}

/**
 * One keyboard contract for every dialog in the workstation:
 *  - focus moves into the dialog when it opens (first control, or `initialFocus` when given);
 *  - Tab / Shift+Tab wrap inside it, so a keyboard or screen-reader user cannot wander onto the page behind;
 *  - Escape closes it (unless the key was already handled by a control inside, for example a closing combobox);
 *  - focus returns to whatever opened it.
 * The dialog element itself must carry `role="dialog"`, `aria-modal="true"` and an accessible name.
 */
export function useDialogA11y(ref: RefObject<HTMLElement | null>, onClose: () => void, active = true, initialFocus?: string) {
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    if (!active) return;
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const root = ref.current;
    const target = (initialFocus && root?.querySelector<HTMLElement>(initialFocus)) || focusableIn(root)[0] || root;
    if (root && !root.hasAttribute('tabindex')) root.setAttribute('tabindex', '-1');
    target?.focus({ preventScroll: true });
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented) { event.preventDefault(); close.current(); return; }
      if (event.key !== 'Tab') return;
      const items = focusableIn(ref.current);
      if (!items.length) { event.preventDefault(); ref.current?.focus(); return; }
      const first = items[0], last = items[items.length - 1], current = document.activeElement;
      if (!ref.current?.contains(current)) { event.preventDefault(); first.focus(); }
      else if (event.shiftKey && (current === first || current === ref.current)) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && current === last) { event.preventDefault(); first.focus(); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      if (opener && opener.isConnected) opener.focus({ preventScroll: true });
    };
  }, [active, ref, initialFocus]);
}
