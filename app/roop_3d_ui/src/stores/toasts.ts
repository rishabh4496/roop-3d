import { createAtomicStore, useStoreSelector } from './atomicStore';

export type AppToast = { id: number; kind: 'error' | 'ok' | 'info'; title?: string; text: string };
type State = { items: AppToast[] };

/** App-wide transient messages. Errors stay until dismissed or 12 s pass; the newest five are kept. */
export const toastStore = createAtomicStore<State>({ items: [] });
export const useToasts = () => useStoreSelector(toastStore, state => state.items);

let nextId = 1;
const timers = new Map<number, ReturnType<typeof setTimeout>>();

export function dismissToast(id: number): void {
  clearTimeout(timers.get(id)); timers.delete(id);
  toastStore.set(state => ({ items: state.items.filter(item => item.id !== id) }));
}

export function pushToast(toast: Omit<AppToast, 'id'>, ms = toast.kind === 'error' ? 12000 : 4500): number {
  const id = nextId++;
  toastStore.set(state => ({ items: [...state.items, { ...toast, id }].slice(-5) }));
  if (typeof window !== 'undefined' && ms > 0) timers.set(id, setTimeout(() => dismissToast(id), ms));
  return id;
}

export function clearToasts(): void {
  timers.forEach(timer => clearTimeout(timer)); timers.clear();
  toastStore.set({ items: [] });
}
