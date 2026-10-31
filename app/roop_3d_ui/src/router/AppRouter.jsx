import React, { useEffect, useState } from 'react';

export const SCREENS = [
  ['home', 'Dashboard'], ['faceswap', 'Workstation'], ['batch', 'Batch Matrix'],
  ['processing', 'Processing'], ['facemgr', 'Face Manager'], ['extras', 'Extras'],
  ['gallery', 'Gallery'], ['history', 'History'], ['settings', 'Settings'],
];
export function normalizeRoute(hash) {
  const route = String(hash || '').split('?')[0].replace(/\/$/, '');
  if (route === '#' || route === '' || route === '#/workstation') return '#/faceswap';
  return SCREENS.some(([id]) => route === `#/${id}`) ? route : '#/home';
}
export function useHashRoute() {
  const [route, setRoute] = useState(() => normalizeRoute(typeof location === 'undefined' ? '' : location.hash));
  useEffect(() => {
    const update = () => {
      const next = normalizeRoute(location.hash);
      if (location.hash !== next) history.replaceState(null, '', `${location.pathname}${location.search}${next}`);
      setRoute(next);
    };
    update(); window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  return route;
}
export function Navigation({ route }) {
  return <nav className="screen-nav" aria-label="Workstation screens">{SCREENS.map(([id, label]) =>
    <a key={id} href={`#/${id}`} className={route === `#/${id}` ? 'active' : ''} aria-current={route === `#/${id}` ? 'page' : undefined}>{label}</a>)}</nav>;
}
// Hidden screens stay mounted: local controls, scroll, selections and drafts survive navigation.
export default function AppRouter({ route, screen, children }) {
  return <div hidden={route !== `#/${screen}`} inert={route !== `#/${screen}` ? true : undefined} data-screen={screen}>{children}</div>;
}
