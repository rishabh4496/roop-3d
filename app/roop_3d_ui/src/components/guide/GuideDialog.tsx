import { useId, useRef, useState, type KeyboardEvent } from 'react';
import type { Capabilities } from '../../contracts';
import DialogFrame from './DialogFrame';
import WorkflowChecklist from './WorkflowChecklist';
import AccessibilityControls from './AccessibilityControls';
import { useExportProfiles } from '../export/exportOptions';
import {
  ENGINE_STATUS, FACESET_STEPS, HARDWARE_PROFILES, JOB_STATE_HELP, PRESET_GUIDANCE, PRESET_LIMITS, PROVIDER_TERMS, QUEUE_GUIDE, READINESS,
  RECOVERY_STEPS, SOURCE_PHOTO_RULES, TARGET_FACE_TUTORIAL, VIEW_REASONS, type HardwareProfileId, type WorkflowStepId,
} from './helpContent';
import type { WorkflowProgress } from './workflowState';
import { KEY_GROUPS, KEY_LABELS, DEFAULT_KEYS } from '../controls/Hotkeys';

export const GUIDE_TABS = [
  { id: 'workflow', label: 'Workflow' },
  { id: 'target', label: 'Target faces' },
  { id: 'facesets', label: 'FaceSets & photos' },
  { id: 'engine', label: 'Engines & hardware' },
  { id: 'presets', label: 'Render presets' },
  { id: 'queue', label: 'Queue & recovery' },
  { id: 'access', label: 'Display & keyboard' },
] as const;
export type GuideTab = typeof GUIDE_TABS[number]['id'];

function PresetTable() {
  const { profiles, error, retry } = useExportProfiles();
  if (error) return <p role="status" className="muted">The presets could not be loaded ({error}). <button type="button" className="link-button" onClick={retry}>Retry</button></p>;
  if (!profiles) return <p className="muted">Loading presets…</p>;
  return (
    <table className="guide-table" aria-label="Export presets">
      <thead><tr><th scope="col">Preset</th><th scope="col">What it is</th><th scope="col">Use it for</th><th scope="col">Size</th></tr></thead>
      <tbody>{profiles.map(item => (
        <tr key={item.id}>
          <th scope="row">{item.label}<small>{item.container === 'sequence' ? 'image sequence' : item.container} · {item.container_bit_depth}-bit{item.lossless ? ' · lossless' : ''}</small></th>
          <td>{item.description}</td><td>{PRESET_GUIDANCE[item.id]?.when ?? ''}</td><td>{PRESET_GUIDANCE[item.id]?.size ?? ''}</td>
        </tr>))}</tbody>
    </table>
  );
}

export default function GuideDialog({ onClose, progress, onAction, onTour, caps, initialTab = 'workflow' }: {
  onClose: () => void; progress: WorkflowProgress; onAction: (id: WorkflowStepId) => void; onTour: () => void; caps: Capabilities | null; initialTab?: GuideTab;
}) {
  const [tab, setTab] = useState<GuideTab>(initialTab);
  const base = useId();
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const resolved = (caps?.workstation.resolved_hardware_profile ?? '') as string;
  const profile = (['auto', 'cpu', '6gb', '12gb'] as HardwareProfileId[]).includes(resolved as HardwareProfileId) ? (resolved as HardwareProfileId) : null;

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const index = GUIDE_TABS.findIndex(item => item.id === tab);
    const move = event.key === 'ArrowRight' || event.key === 'ArrowDown' ? 1 : event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : event.key === 'Home' ? -index : event.key === 'End' ? GUIDE_TABS.length - 1 - index : 0;
    if (!move) return;
    event.preventDefault();
    const next = (index + move + GUIDE_TABS.length) % GUIDE_TABS.length;
    setTab(GUIDE_TABS[next].id); refs.current[next]?.focus();
  };

  return (
    <DialogFrame title="Guide" eyebrow="HOW TO USE ROOP 3D" onClose={onClose} className="guide-dialog" testId="guide-dialog" initialFocus='[role="tab"][aria-selected="true"]'>
      <div role="tablist" aria-label="Guide topics" className="guide-tabs" onKeyDown={onKeyDown}>
        {GUIDE_TABS.map((item, index) => (
          <button key={item.id} ref={element => { refs.current[index] = element; }} type="button" role="tab" id={`${base}-tab-${item.id}`} aria-selected={tab === item.id} aria-controls={`${base}-panel-${item.id}`} tabIndex={tab === item.id ? 0 : -1}
            className={tab === item.id ? 'active' : ''} onClick={() => setTab(item.id)}>{item.label}</button>))}
      </div>
      <div role="tabpanel" id={`${base}-panel-${tab}`} aria-labelledby={`${base}-tab-${tab}`} className="guide-panel" tabIndex={0}>
        {tab === 'workflow' && <>
          <p className="muted">Everything runs on this computer. Follow the steps in order; the list updates as you work. <button type="button" className="link-button" onClick={onTour}>Take the 4-step visual tour</button></p>
          <WorkflowChecklist progress={progress} onAction={id => { onAction(id); onClose(); }} />
        </>}
        {tab === 'target' && <>
          <h3>{TARGET_FACE_TUTORIAL.title}</h3>
          <ol className="guide-steps">{TARGET_FACE_TUTORIAL.steps.map(item => <li key={item}>{item}</li>)}</ol>
          <h4>Good to know</h4>
          <ul>{TARGET_FACE_TUTORIAL.tips.map(item => <li key={item}>{item}</li>)}</ul>
        </>}
        {tab === 'facesets' && <>
          <h3>Preparing a FaceSet</h3>
          <ol className="guide-steps">{FACESET_STEPS.map(item => <li key={item}>{item}</li>)}</ol>
          <h3>Source photo checklist</h3>
          <p className="muted">These are the checks Roop 3D really applies to each photo. A photo that fails is kept in the FaceSet but not used for the 3D build.</p>
          <ul className="guide-rules">{SOURCE_PHOTO_RULES.map(item => <li key={item.id}><strong>{item.rule}.</strong> {item.detail}</li>)}</ul>
          <h3>What the coverage label means</h3>
          <dl className="guide-dl">{Object.entries(READINESS).map(([key, item]) => <div key={key}><dt>{item.label}</dt><dd>{item.meaning} <em>{item.next}</em></dd></div>)}</dl>
          <details><summary>Why a photo was not used</summary><dl className="guide-dl">{Object.entries(VIEW_REASONS).map(([key, text]) => <div key={key}><dt><code>{key}</code></dt><dd>{text}</dd></div>)}</dl></details>
        </>}
        {tab === 'engine' && <>
          <h3>Engines</h3>
          <p>An engine is the method that swaps the faces. In Inspector → Models &amp; engine the list shows what each engine can do on this computer, and which one Roop 3D recommends. An engine that cannot run here is shown but cannot be selected.</p>
          <dl className="guide-dl">{Object.entries(ENGINE_STATUS).map(([key, item]) => <div key={key}><dt>{item.label}</dt><dd>{item.meaning}</dd></div>)}</dl>
          <h3>Hardware profile{profile ? <> · <span className="status-chip ok">this computer: {profile}</span></> : null}</h3>
          <p>The profile tells Roop 3D how much graphics memory it may use. Leave it on Auto unless you have a reason; change it in Settings → Workstation Preferences → Compute.</p>
          <dl className="guide-dl">{(Object.keys(HARDWARE_PROFILES) as HardwareProfileId[]).map(key => { const item = HARDWARE_PROFILES[key]; return <div key={key} className={key === profile ? 'current' : ''}><dt>{item.label}{key === profile && <span className="status-chip ok"> detected</span>}</dt><dd>{item.summary} {item.behaviour.join(' ')} <em>Needs: {item.needs}</em></dd></div>; })}</dl>
          <h4>Reading the hardware pill</h4>
          <dl className="guide-dl">{PROVIDER_TERMS.map(item => <div key={item.term}><dt>{item.term}</dt><dd>{item.meaning}</dd></div>)}</dl>
        </>}
        {tab === 'presets' && <>
          <h3>Render presets</h3>
          <p>An export preset decides the file format and quality of the finished video. Pick it next to the trim bar, before you press Queue render.</p>
          <PresetTable />
          <p className="muted">{PRESET_LIMITS}</p>
        </>}
        {tab === 'queue' && <>
          <h3>The queue and checkpoints</h3>
          <ul>{QUEUE_GUIDE.map(item => <li key={item}>{item}</li>)}</ul>
          <h4>Job states</h4>
          <table className="guide-table" aria-label="Job states"><thead><tr><th scope="col">State</th><th scope="col">Meaning</th><th scope="col">What to do</th></tr></thead>
            <tbody>{Object.values(JOB_STATE_HELP).map(item => <tr key={item.label}><th scope="row">{item.label}</th><td>{item.meaning}</td><td>{item.action}</td></tr>)}</tbody></table>
          <h3>If something goes wrong</h3>
          {RECOVERY_STEPS.map(item => <details key={item.problem}><summary>{item.problem}</summary><ol>{item.steps.map(step => <li key={step}>{step}</li>)}</ol></details>)}
        </>}
        {tab === 'access' && <>
          <h3>Display and motion</h3>
          <AccessibilityControls />
          <h3>Keyboard</h3>
          <p>Everything can be reached with Tab and Shift+Tab. Dialogs keep focus inside until you press Esc. Shortcuts are paused while you type in a field or have a dialog open. Any of them can be changed or switched off in Settings → Workstation Preferences → Keyboard shortcut remapper: focus a shortcut and press a new key combination, or press Backspace to leave it unassigned.</p>
          {KEY_GROUPS.map(group => <section key={group.title} className="guide-keys"><h4>{group.title}</h4><dl className="guide-dl keys">{group.actions.map(action => <div key={action}><dt><kbd>{DEFAULT_KEYS[action]}</kbd></dt><dd>{KEY_LABELS[action]}</dd></div>)}</dl></section>)}
          <p className="muted">Page zoom: Ctrl + / Ctrl − / Ctrl 0 (or the − 100% + buttons in the header). The layout reflows instead of overflowing.</p>
        </>}
      </div>
      <div className="guide-footer"><button type="button" className="button" onClick={onClose}>Close guide</button></div>
    </DialogFrame>
  );
}
