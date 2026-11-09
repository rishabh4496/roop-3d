import type { ReactNode } from 'react';
import { IconHelp, IconLibrary, IconSearch, IconMinus, IconMoon, IconPalette, IconPlus, IconSave, IconSun } from './icons';

export interface TopBarProps {
  title: string;
  projectName: string; onProjectName: (value: string) => void; dirty: boolean;
  connection: string;
  pill?: ReactNode; liveStudio?: ReactNode;
  zoom: number; onZoom: (delta: number) => void; onZoomReset: () => void;
  themeMode: 'dark' | 'light'; onToggleMode: () => void; onThemes: () => void;
  tourFresh: boolean; onTour: () => void; onGuide: () => void;
  layoutMode: 'studio' | 'classic'; onLayout: () => void;
  projectsDisabled: boolean; projectsReason?: string | null; busy: boolean; canSave: boolean;
  onPalette: () => void;
  onProjects: () => void; onSave: () => void; onSaveCopy: () => void;
}

/** Slim top bar: where you are, the project, run/hardware status and workspace tools. One row, no stacked strips. */
export default function TopBar(p: TopBarProps) {
  const online = p.connection === 'online';
  return (
    <header className="app-header topbar">
      <h1 className="topbar-title">{p.title}</h1>
      <div className="project-name"><input aria-label="Project name" maxLength={128} value={p.projectName} onChange={event => p.onProjectName(event.target.value)} /><span className={p.dirty ? 'dirty' : ''}>{p.dirty ? 'UNSAVED' : 'SAVED'}</span></div>
      <span className={`local-status ${online ? 'online' : ''}`} role="status" aria-label={online ? 'Local service connected' : `Local service ${p.connection}`} title={online ? 'Connected to the local service' : `Local service: ${p.connection}`}><i className={`dot ${online ? 'mint' : 'amber'}`} />{!online && p.connection}</span>
      <span className="topbar-spacer" />
      {p.pill}
      {p.liveStudio}
      <button type="button" className="palette-trigger" aria-haspopup="dialog" aria-label="Open command palette" title="Search screens and actions (Ctrl K)" onClick={p.onPalette}><IconSearch size={15} /><span>Search</span><kbd>Ctrl K</kbd></button>
      <div className="topbar-group">
        <button type="button" className="button small layout-toggle" aria-pressed={p.layoutMode === 'studio'} title="Switch between the four-panel studio and the classic stacked workstation" onClick={p.onLayout}>{p.layoutMode === 'studio' ? 'Studio layout' : 'Classic layout'}</button>
      </div>
      <div role="group" aria-label="UI zoom" title="UI Zoom (Ctrl + / − / 0)" className="ui-zoom-group">
        <button type="button" onClick={() => p.onZoom(-0.05)} title="Zoom out (Ctrl −)" aria-label="Zoom out" className="zoom-btn"><IconMinus size={14} /></button>
        <button type="button" onClick={p.onZoomReset} title="Reset zoom (Ctrl 0)" aria-label={`Reset zoom, ${Math.round(p.zoom * 100)}%`} className="zoom-reset-btn">{Math.round(p.zoom * 100)}%</button>
        <button type="button" onClick={() => p.onZoom(0.05)} title="Zoom in (Ctrl +)" aria-label="Zoom in" className="zoom-btn"><IconPlus size={14} /></button>
      </div>
      <div className="topbar-group">
        <button type="button" className="icon-button" aria-label="Open theme gallery" title="Themes" onClick={p.onThemes}><IconPalette /></button>
        <button type="button" className="icon-button" aria-label={`Switch to ${p.themeMode === 'dark' ? 'light' : 'dark'} theme`} title={p.themeMode === 'dark' ? 'Light mode' : 'Dark mode'} onClick={p.onToggleMode}>{p.themeMode === 'dark' ? <IconSun /> : <IconMoon />}</button>
        <button type="button" className={`button small tour-button ${p.tourFresh ? 'fresh' : ''}`} data-testid="header-tour-button" aria-haspopup="dialog" title="Four-step guide to the workstation" onClick={p.onTour}>How to Use</button>
        <button type="button" className="icon-button guide-button" data-testid="header-guide-button" aria-haspopup="dialog" aria-label="Guide" title="Step-by-step guide, explanations and recovery help" onClick={p.onGuide}><IconHelp /></button>
      </div>
      <div className="topbar-group">
        <button type="button" className="button small" disabled={p.projectsDisabled || p.busy} title={p.projectsReason ?? 'Project library'} onClick={p.onProjects}><IconLibrary size={15} /> Projects</button>
        <button type="button" className="button small accent" disabled={p.projectsDisabled || !p.canSave || p.busy} onClick={p.onSave}><IconSave size={15} /> Save project</button>
        <button type="button" className="button small" disabled={!p.canSave || p.busy} onClick={p.onSaveCopy}>Save copy</button>
      </div>
    </header>
  );
}
