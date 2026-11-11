import React, { useEffect, useState } from 'react';

export const SCREENS = [
  ['home', 'Home'], ['faceswap', 'Face Swap'], ['batch', 'Batch Matrix'],
  ['processing', 'Processing'], ['facemgr', 'Face Manager'], ['extras', 'Editor'],
  ['gallery', 'Outputs'], ['history', 'History'], ['settings', 'Settings'],
];
/**
 * Stage 20 deep links. Each is a stable, shareable URL that opens the same screen as an existing route, so nothing that links to the
 * old names (the rail, the tour, saved bookmarks) breaks. The URL the person typed or clicked is kept in the address bar.
 */
export const ROUTE_ALIASES = { '#/studio': '#/faceswap', '#/identity-forge': '#/facemgr', '#/queue': '#/processing' };
/** Deep links that open a slide-over drawer on top of the Studio instead of a screen of their own. */
export const DRAWER_ROUTES = { '#/models': 'models', '#/diagnostics': 'diagnostics' };
const cleanHash = hash => String(hash || '').split('?')[0].replace(/\/$/, '');
export const isDeepLink = hash => { const route = cleanHash(hash); return route in ROUTE_ALIASES || route in DRAWER_ROUTES; };
/** 'models' | 'diagnostics' | null: which drawer the hash asks for. */
export const drawerOf = hash => DRAWER_ROUTES[cleanHash(hash)] ?? null;
/**
 * Stage 23: the last screen survives a reload. Pinokio tab switches can reload the page with no hash at all; without this
 * every switch dropped the person back on the Studio. Only an *empty* hash falls back to it, so a typed or clicked link always wins.
 */
export const LAST_ROUTE_KEY = 'roop3d-last-route';
export function readStoredRoute(storage = typeof localStorage === 'undefined' ? null : localStorage) {
  try {
    const stored = storage?.getItem(LAST_ROUTE_KEY);
    return stored && SCREENS.some(([id]) => stored === `#/${id}`) ? stored : null;
  } catch { return null; }
}
export function storeRoute(route, storage = typeof localStorage === 'undefined' ? null : localStorage) {
  try { if (SCREENS.some(([id]) => route === `#/${id}`)) storage?.setItem(LAST_ROUTE_KEY, route); } catch { /* storage is optional */ }
}
export const isEmptyHash = hash => ['', '#', '#/'].includes(String(hash || ''));
export function normalizeRoute(hash) {
  const route = cleanHash(hash);
  if (route === '#' || route === '' || route === '#/workstation') return '#/faceswap';
  if (route in ROUTE_ALIASES) return ROUTE_ALIASES[route];
  if (route in DRAWER_ROUTES) return '#/faceswap';
  return SCREENS.some(([id]) => route === `#/${id}`) ? route : '#/home';
}
export function useHashRoute() {
  const [route, setRoute] = useState(() => (typeof location !== 'undefined' && isEmptyHash(location.hash) && readStoredRoute()) || normalizeRoute(typeof location === 'undefined' ? '' : location.hash));
  useEffect(() => {
    const update = () => {
      const next = (isEmptyHash(location.hash) && readStoredRoute()) || normalizeRoute(location.hash);
      // Stage 20 deep links keep their own URL (#/studio stays #/studio); only legacy and unknown hashes are rewritten.
      if (location.hash !== next && !isDeepLink(location.hash)) history.replaceState(null, '', `${location.pathname}${location.search}${next}`);
      storeRoute(next);
      setRoute(next);
    };
    update(); window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  return route;
}
/** Which drawer (if any) the current hash opens. Follows hashchange like `useHashRoute`. */
export function useHashDrawer() {
  const [drawer, setDrawer] = useState(() => drawerOf(typeof location === 'undefined' ? '' : location.hash));
  useEffect(() => {
    const update = () => setDrawer(drawerOf(location.hash));
    update(); window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  return drawer;
}
export function Navigation({ route }) {
  return <nav className="screen-nav" aria-label="Workstation screens">{SCREENS.map(([id, label]) =>
    <a key={id} href={`#/${id}`} className={route === `#/${id}` ? 'active' : ''} aria-current={route === `#/${id}` ? 'page' : undefined}>{label}</a>)}</nav>;
}
// Hidden screens stay mounted: local controls, scroll, selections and drafts survive navigation.
export default function AppRouter({ route, screen, children }) {
  return <div hidden={route !== `#/${screen}`} inert={route !== `#/${screen}` ? true : undefined} data-screen={screen}>{children}</div>;
}
