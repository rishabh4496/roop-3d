import { useRef, useSyncExternalStore } from 'react';

/**
 * Tiny atomic subscription store (Stage 11).
 *
 * High-frequency state (`/ws/telemetry` frames, the playhead timecode) must not live in `App`'s React state: every
 * tick would re-render the parameter panels, the timeline filmstrip and the Face Manager. Writers call `set()`;
 * readers either subscribe *transiently* (`subscribeSelector`, no React render) or through `useStoreSelector`, which
 * re-renders only the one component whose selected slice actually changed.
 */
export interface AtomicStore<T> {
  get: () => T;
  /** Shallow-merge a patch (or replace with the updater's result). Listeners fire only when something changed. */
  set: (patch: Partial<T> | ((state: T) => Partial<T>)) => void;
  subscribe: (listener: () => void) => () => void;
  /** Transient subscription: `listener` runs outside React whenever the selected slice changes. */
  subscribeSelector: <S>(selector: (state: T) => S, listener: (value: S, previous: S) => void, isEqual?: (a: S, b: S) => boolean) => () => void;
  reset: () => void;
}

export function createAtomicStore<T extends object>(initial: T): AtomicStore<T> {
  let state = initial;
  const listeners = new Set<() => void>();
  const get = () => state;
  const set: AtomicStore<T>['set'] = patch => {
    const resolved = typeof patch === 'function' ? patch(state) : patch;
    const keys = Object.keys(resolved) as (keyof T)[];
    if (!keys.some(key => !Object.is(state[key], resolved[key]))) return;
    state = { ...state, ...resolved };
    for (const listener of [...listeners]) listener();
  };
  const subscribe = (listener: () => void) => { listeners.add(listener); return () => { listeners.delete(listener); }; };
  const subscribeSelector: AtomicStore<T>['subscribeSelector'] = (selector, listener, isEqual = Object.is) => {
    let previous = selector(state);
    return subscribe(() => {
      const next = selector(state);
      if (isEqual(previous, next)) return;
      const before = previous; previous = next; listener(next, before);
    });
  };
  const reset = () => { state = initial; for (const listener of [...listeners]) listener(); };
  return { get, set, subscribe, subscribeSelector, reset };
}

/** Re-render the calling component only when `selector(state)` changes (by `isEqual`, default `Object.is`). */
export function useStoreSelector<T, S>(store: Pick<AtomicStore<T>, 'get' | 'subscribe'>, selector: (state: T) => S, isEqual: (a: S, b: S) => boolean = Object.is): S {
  const cache = useRef<{ state: T; selector: (state: T) => S; value: S } | null>(null);
  const getSnapshot = (): S => {
    const state = store.get(), cached = cache.current;
    if (cached && cached.state === state && cached.selector === selector) return cached.value;
    const value = selector(state);
    if (cached && isEqual(cached.value, value)) { cache.current = { state, selector, value: cached.value }; return cached.value; }
    cache.current = { state, selector, value };
    return value;
  };
  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot);
}

export const shallowEqual = <S,>(a: S, b: S): boolean => {
  if (Object.is(a, b)) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || !a || !b) return false;
  const left = a as Record<string, unknown>, right = b as Record<string, unknown>, keys = Object.keys(left);
  return keys.length === Object.keys(right).length && keys.every(key => Object.is(left[key], right[key]));
};
