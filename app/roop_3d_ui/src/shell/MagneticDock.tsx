import { useRef, type ComponentType } from 'react';
import { motion, useMotionValue, useReducedMotion, useSpring, useTransform, type MotionValue } from 'framer-motion';
import { IconActivity, IconGauge, IconLibrary, IconSearch, IconSwap, IconUsers } from './icons';
import { cn } from '../lib/utils';

/**
 * Floating bottom dock with magnetic icon scaling (items swell as the pointer nears them, like a macOS dock). The scale is driven by a
 * motion value and springs, so hovering re-renders nothing. Every item is a real link to a Stage 20 deep link (`#/studio`,
 * `#/identity-forge`, `#/queue`, `#/models`, `#/diagnostics`), so it works with the keyboard, middle-click and "copy link address";
 * Models and Runtime report open as drawers over the Studio. Reduced motion keeps the dock but turns the magnification off.
 */
export interface DockItemDef { id: string; label: string; hash: string; icon: ComponentType<{ size?: number }> }

export const DOCK_ITEMS: readonly DockItemDef[] = [
  { id: 'studio', label: 'Studio', hash: '#/studio', icon: IconSwap },
  { id: 'identity-forge', label: 'Identity Forge', hash: '#/identity-forge', icon: IconUsers },
  { id: 'queue', label: 'Batch Queue', hash: '#/queue', icon: IconActivity },
  { id: 'models', label: 'Model Hub', hash: '#/models', icon: IconLibrary },
  { id: 'diagnostics', label: 'Runtime Report', hash: '#/diagnostics', icon: IconGauge },
];

/** Which dock item is current: a drawer wins over the screen behind it. */
export function activeDockItem(route: string, drawer: string | null): string | null {
  if (drawer) return drawer;
  if (route === '#/faceswap') return 'studio';
  if (route === '#/facemgr') return 'identity-forge';
  if (route === '#/processing') return 'queue';
  return null;
}

/** Magnification for a pointer `distance` px from an item's centre: 1.55x under it, easing back to 1x by 140px away. */
export function magnetScale(distance: number): number {
  const d = Math.min(Math.abs(distance), 140);
  return 1 + 0.55 * (1 - d / 140) ** 2;
}

const FAR = -9999; // "pointer is not over the dock": far enough that every item sits at 1x

function DockItem({ item, mouseX, magnify, active }: { item: DockItemDef; mouseX: MotionValue<number>; magnify: boolean; active: boolean }) {
  const ref = useRef<HTMLAnchorElement>(null);
  const raw = useTransform(mouseX, x => {
    const box = ref.current?.getBoundingClientRect();
    return box ? magnetScale(x - (box.left + box.width / 2)) : 1;
  });
  const scale = useSpring(raw, { stiffness: 380, damping: 26, mass: 0.6 });
  const Icon = item.icon;
  return (
    <motion.a
      ref={ref} href={item.hash} className={cn('dock-item', active && 'active')} data-dock={item.id}
      aria-label={item.label} aria-current={active ? 'page' : undefined}
      style={magnify ? { scale } : undefined}
    >
      <Icon size={20} />
      <span className="dock-tip" aria-hidden="true">{item.label}</span>
    </motion.a>
  );
}

export default function MagneticDock({ route, drawer, onPalette }: { route: string; drawer: string | null; onPalette: () => void }) {
  const mouseX = useMotionValue(FAR);
  const reduce = useReducedMotion();
  const current = activeDockItem(route, drawer);
  return (
    <nav
      className="dock" aria-label="Quick dock" data-testid="magnetic-dock"
      onPointerMove={event => { if (event.pointerType === 'mouse') mouseX.set(event.clientX); }}
      onPointerLeave={() => mouseX.set(FAR)}
    >
      {DOCK_ITEMS.map(item => <DockItem key={item.id} item={item} mouseX={mouseX} magnify={!reduce} active={current === item.id} />)}
      <span className="dock-sep" aria-hidden="true" />
      <button type="button" className="dock-item dock-palette" aria-haspopup="dialog" aria-label="Open command palette" title="Search screens and actions (Ctrl K / ⌘K)" onClick={onPalette}>
        <IconSearch size={20} /><span className="dock-tip" aria-hidden="true">Search · ⌘K</span>
      </button>
    </nav>
  );
}
