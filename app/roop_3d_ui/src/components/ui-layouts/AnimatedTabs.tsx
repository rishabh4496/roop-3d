import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { LayoutGroup, motion, useReducedMotion } from 'framer-motion';
import { cn } from '../../lib/utils';

/**
 * Pill tabs whose active indicator is a shared-layout element: it springs from the old tab to the new one instead of cutting.
 * WAI-ARIA tabs pattern: roving tabindex, Left/Right (and Up/Down) move + activate, Home/End jump. Panels are optional,
 * so the same component also works as a pure segmented control (omit `content`).
 */
export interface TabItem {
  id: string;
  label: ReactNode;
  content?: ReactNode;
  disabled?: boolean;
}

export interface AnimatedTabsProps {
  tabs: TabItem[];
  defaultValue?: string;
  value?: string;
  onValueChange?: (id: string) => void;
  className?: string;
  panelClassName?: string;
  'aria-label'?: string;
}

export function AnimatedTabs({ tabs, defaultValue, value, onValueChange, className, panelClassName, 'aria-label': ariaLabel }: AnimatedTabsProps) {
  const base = useId();
  const reduce = useReducedMotion();
  const firstEnabled = tabs.find(tab => !tab.disabled)?.id ?? tabs[0]?.id;
  const [inner, setInner] = useState(defaultValue ?? firstEnabled);
  const active = value ?? inner;
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});

  const select = (id: string) => {
    if (value === undefined) setInner(id);
    onValueChange?.(id);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, id: string) => {
    const enabled = tabs.filter(tab => !tab.disabled).map(tab => tab.id);
    const at = enabled.indexOf(id);
    let target: string | undefined;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') target = enabled[(at + 1) % enabled.length];
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') target = enabled[(at - 1 + enabled.length) % enabled.length];
    else if (event.key === 'Home') target = enabled[0];
    else if (event.key === 'End') target = enabled[enabled.length - 1];
    if (target === undefined) return;
    event.preventDefault();
    select(target);
    refs.current[target]?.focus();
  };

  const activeTab = tabs.find(tab => tab.id === active);
  return (
    <div className={cn('ul-tabs', className)}>
      <LayoutGroup id={`${base}-tabs`}>
        <div className="ul-tabs__list" role="tablist" aria-label={ariaLabel}>
          {tabs.map(tab => {
            const selected = tab.id === active;
            return (
              <button
                key={tab.id} ref={node => { refs.current[tab.id] = node; }}
                id={`${base}-t-${tab.id}`} type="button" role="tab" className="ul-tabs__tab"
                aria-selected={selected} aria-controls={tab.content !== undefined ? `${base}-p-${tab.id}` : undefined}
                tabIndex={selected ? 0 : -1} disabled={tab.disabled} data-active={selected ? 'true' : 'false'}
                onClick={() => select(tab.id)} onKeyDown={event => onKeyDown(event, tab.id)}
              >
                {selected && (
                  <motion.span
                    layoutId="ul-tab-pill" className="ul-tabs__pill" aria-hidden="true"
                    transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 520, damping: 38, mass: 0.7 }}
                  />
                )}
                <span className="ul-tabs__label">{tab.label}</span>
              </button>
            );
          })}
        </div>
      </LayoutGroup>
      {activeTab?.content !== undefined && (
        <div key={activeTab.id} id={`${base}-p-${activeTab.id}`} role="tabpanel" aria-labelledby={`${base}-t-${activeTab.id}`} tabIndex={0} className={cn('ul-tabs__panel', panelClassName)}>
          {activeTab.content}
        </div>
      )}
    </div>
  );
}

export default AnimatedTabs;
