import { useRef, type KeyboardEvent } from 'react';

/**
 * Tabs for the Studio's bottom dock. The dock carries lanes, trim + export, the frame strip, the face tray and the curve editor.
 * Showing them all at once needs ~1500px, so each tool gets its own pane that fits the dock without an inner scrollbar.
 * The panes themselves stay mounted (CSS hides the inactive ones) so scrubbing state, drafts and portals survive a tab switch.
 */
export const DOCK_TABS = [
  { id: 'timeline', label: 'Timeline', hint: 'Track lanes, transport and the keyframe timeline' },
  { id: 'range', label: 'Range & export', hint: 'In / Out trim and the export preset' },
  { id: 'frames', label: 'Frames', hint: 'Frame navigator: thumbnails, chapters and speed' },
  { id: 'faces', label: 'Faces', hint: 'Faces found in the current frame' },
  { id: 'curves', label: 'Curves', hint: 'Per-parameter keyframe curves' },
] as const;

export type DockTabId = typeof DOCK_TABS[number]['id'];

export function DockTabs({ active, onChange }: { active: DockTabId; onChange: (id: DockTabId) => void }) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const move = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const last = DOCK_TABS.length - 1;
    const next = event.key === 'ArrowRight' ? (index + 1) % DOCK_TABS.length : event.key === 'ArrowLeft' ? (index - 1 + DOCK_TABS.length) % DOCK_TABS.length
      : event.key === 'Home' ? 0 : event.key === 'End' ? last : -1;
    if (next < 0) return;
    event.preventDefault();
    const tab = DOCK_TABS[next];
    onChange(tab.id);
    refs.current[tab.id]?.focus();
  };
  return <div className="dock-tabs" role="tablist" aria-label="Timeline tools" data-testid="dock-tabs">
    {DOCK_TABS.map((tab, index) => <button key={tab.id} ref={node => { refs.current[tab.id] = node; }} type="button" role="tab" id={`dock-tab-${tab.id}`}
      aria-selected={active === tab.id} tabIndex={active === tab.id ? 0 : -1} title={tab.hint} data-testid={`dock-tab-${tab.id}`}
      className={`dock-tab${active === tab.id ? ' active' : ''}`} onClick={() => onChange(tab.id)} onKeyDown={event => move(event, index)}>{tab.label}</button>)}
  </div>;
}
