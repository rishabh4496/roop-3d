import { useEffect, useState, type ComponentType } from 'react';
import { SCREENS } from '../router/AppRouter';
import { LayoutGroup, motion, spring } from '../motion';
import { IconActivity, IconCube, IconFolder, IconHome, IconLayers, IconClock, IconSidebar, IconSliders, IconSwap, IconUsers, IconWand } from './icons';

const ICONS: Record<string, ComponentType<{ size?: number }>> = {
  home: IconHome, faceswap: IconSwap, batch: IconLayers, processing: IconActivity, facemgr: IconUsers,
  extras: IconWand, gallery: IconFolder, history: IconClock, settings: IconSliders,
};
const KEY = 'roop3d-rail-collapsed';
const read = () => { try { const stored = localStorage.getItem(KEY); if (stored != null) return stored === '1'; } catch { /* default below */ } return typeof window !== 'undefined' && window.innerWidth < 1400; };

/** Vertical navigation rail. Keeps the `screen-nav` class and "Workstation screens" label that the router tests and e2e rely on. */
export default function Rail({ route }: { route: string }) {
  const [collapsed, setCollapsed] = useState(read);
  const toggle = () => setCollapsed(value => { const next = !value; try { localStorage.setItem(KEY, next ? '1' : '0'); } catch { /* per-session */ } return next; });
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if ((event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === 'b') { event.preventDefault(); toggle(); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return (
    <aside className={`rail ${collapsed ? 'collapsed' : ''}`}>
      <a className="rail-brand brand" href="#/home" aria-label="Roop 3D home">
        <span className="rail-logo"><IconCube size={20} /></span>
        <span className="rail-brand-text"><strong>Roop 3D</strong><small>Local workstation</small></span>
      </a>
      <nav className="screen-nav rail-nav" aria-label="Workstation screens">
        <LayoutGroup id="rail">
          {SCREENS.map(([id, label]: string[]) => {
            const Icon = ICONS[id] ?? IconHome, active = route === `#/${id}`;
            return (
              <a key={id} href={`#/${id}`} className={`rail-link ${active ? 'active' : ''}`} aria-current={active ? 'page' : undefined} title={collapsed ? label : undefined}>
                {active && <motion.span layoutId="rail-active" className="rail-active" transition={spring.snappy} />}
                <span className="rail-icon"><Icon /></span>
                <span className="rail-label">{label}</span>
              </a>
            );
          })}
        </LayoutGroup>
      </nav>
      <button type="button" className="rail-collapse" onClick={toggle} aria-pressed={collapsed} aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'} title="Collapse navigation (Ctrl B)">
        <span className="rail-icon"><IconSidebar /></span><span className="rail-label">Collapse</span>
      </button>
    </aside>
  );
}
