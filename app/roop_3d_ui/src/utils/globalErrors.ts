import { pushToast } from '../stores/toasts';

/** A rejection that only means "this request was cancelled" (navigation, a newer request, unmount). Never worth showing. */
export const isBenignRejection = (reason: unknown): boolean => {
  const name = (reason as { name?: string } | null)?.name;
  const message = String((reason as { message?: string } | null)?.message ?? reason ?? '');
  return name === 'AbortError' || /aborted|ResizeObserver loop/i.test(message);
};

const RECENT = new Map<string, number>();
const REPEAT_WINDOW_MS = 10_000;

/**
 * Last-resort net for promise rejections nobody handled. Real call sites should still `.catch`; this makes sure that a missed one
 * becomes one readable toast (de-duplicated for 10 s) instead of a silent failure, and that the console shows a single warning
 * rather than an "Uncaught (in promise)" error. Returns the uninstall function.
 */
export function installGlobalErrorHandlers(target: Window = window, now: () => number = () => Date.now()): () => void {
  const onRejection = (event: PromiseRejectionEvent) => {
    if (isBenignRejection(event.reason)) { event.preventDefault(); return; }
    event.preventDefault();
    const message = event.reason instanceof Error ? event.reason.message : String(event.reason ?? 'Unknown error');
    console.warn('[workstation] unhandled promise rejection', event.reason);
    const at = now();
    if (at - (RECENT.get(message) ?? -Infinity) < REPEAT_WINDOW_MS) return;
    RECENT.set(message, at);
    if (RECENT.size > 50) RECENT.clear();
    pushToast({ kind: 'error', title: 'Something went wrong', text: message.slice(0, 240) });
  };
  target.addEventListener('unhandledrejection', onRejection);
  return () => target.removeEventListener('unhandledrejection', onRejection);
}
