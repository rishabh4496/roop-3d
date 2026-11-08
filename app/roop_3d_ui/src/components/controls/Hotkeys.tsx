import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';

/**
 * Central, remappable hotkey manager (Stage 26). Every action the workstation reacts to is declared here with a default
 * chord, a label and a group; components bind behaviour with `useShortcutActions({ action: handler })`. Remaps are
 * conflict-checked, persisted to localStorage and shared between tabs, and survive route changes because the provider
 * lives above the router.
 */
export const DEFAULT_KEYS: Record<string, string> = {
  play: 'Space', previous: 'ArrowLeft', next: 'ArrowRight', markIn: 'I', markOut: 'O', jumpIn: 'Shift+I', jumpOut: 'Shift+O', resetRange: 'Alt+R',
  toggleWipe: 'W', toggleDiff: 'D',
  buffer1: '1', buffer2: '2', buffer3: '3', buffer4: '4', buffer5: '5', buffer6: '6', buffer7: '7', buffer8: '8',
  solo: 'S', bypass: 'B', sweep: 'Shift+P',
  extractFace: 'E',
  snapshotA: 'Shift+A', snapshotB: 'Shift+B', snapshotC: 'Shift+C', snapshotD: 'Shift+D',
  floating: 'Shift+F', pip: 'Shift+G',
  save: 'Ctrl+S', undo: 'Ctrl+Z', redo: 'Ctrl+Y',
};
export const KEY_LABELS: Record<string, string> = {
  play: 'Play / pause', previous: 'Previous frame', next: 'Next frame', markIn: 'Mark trim in', markOut: 'Mark trim out', jumpIn: 'Jump to trim in', jumpOut: 'Jump to trim out', resetRange: 'Reset trim range',
  toggleWipe: 'Toggle split wipe', toggleDiff: 'Toggle diff map',
  // One key per node of the preview pipeline (components/workstation/pipeline.ts), in that order.
  buffer1: 'Solo buffer 1 (Raw Target)', buffer2: 'Solo buffer 2 (6DoF 3D Mesh & Pose)', buffer3: 'Solo buffer 3 (Canonical UV Projection)', buffer4: 'Solo buffer 4 (SH 3D Relighting & Normals)', buffer5: 'Solo buffer 5 (Base 3D Swap)', buffer6: 'Solo buffer 6 (UltraMax High-Res)', buffer7: 'Solo buffer 7 (XSeg + Hair Occlusion Matte)', buffer8: 'Solo buffer 8 (Final Composite)',
  solo: 'Solo selected pipeline node', bypass: 'Bypass selected pipeline node', sweep: 'Parameter sweep',
  extractFace: 'Extract face from frame',
  snapshotA: 'Capture snapshot A', snapshotB: 'Capture snapshot B', snapshotC: 'Capture snapshot C', snapshotD: 'Capture snapshot D',
  floating: 'Floating preview', pip: 'Picture-in-picture preview',
  save: 'Save project', undo: 'Undo edit', redo: 'Redo edit',
};
/** Display grouping for the remapper UI; every action in DEFAULT_KEYS appears in exactly one group. */
export const KEY_GROUPS: { title: string; actions: string[] }[] = [
  { title: 'Transport & range', actions: ['play', 'previous', 'next', 'markIn', 'markOut', 'jumpIn', 'jumpOut', 'resetRange'] },
  { title: 'Inspection canvas', actions: ['toggleWipe', 'toggleDiff', 'buffer1', 'buffer2', 'buffer3', 'buffer4', 'buffer5', 'buffer6', 'buffer7', 'buffer8', 'solo', 'bypass', 'sweep'] },
  { title: 'Faces & snapshots', actions: ['extractFace', 'snapshotA', 'snapshotB', 'snapshotC', 'snapshotD'] },
  { title: 'Workspace', actions: ['floating', 'pip', 'save', 'undo', 'redo'] },
];
export const STORAGE = 'roop3d-hotkeys-v1';
const CHORD = /^(?:(?:Ctrl|Alt|Shift|Meta)\+)*(?:[A-Z0-9]|Space|ArrowLeft|ArrowRight|F(?:[1-9]|1[0-2]))$/;

export function keyChord(event: Pick<KeyboardEvent, 'key' | 'ctrlKey' | 'altKey' | 'shiftKey' | 'metaKey'>) {
  if (['Control', 'Alt', 'Shift', 'Meta'].includes(event.key)) return '';
  const key = event.key === ' ' ? 'Space' : event.key.length === 1 ? event.key.toUpperCase() : event.key;
  return [event.ctrlKey && 'Ctrl', event.altKey && 'Alt', event.shiftKey && 'Shift', event.metaKey && 'Meta', key].filter(Boolean).join('+');
}
/** The other action already using `chord` (an empty chord, i.e. "unassigned", never conflicts). */
export function keyConflict(keys: Record<string, string>, action: string, chord: string) { return chord ? Object.keys(keys).find(key => key !== action && keys[key] === chord) : undefined; }

/**
 * Normalise a stored or edited keymap. Entries that are present must be a valid, unique chord (or '' = unassigned),
 * otherwise the whole map falls back to the defaults. Actions that are *absent* (a map saved before they existed) are
 * filled from the defaults when that chord is free and left unassigned when the user already took it, so adding new
 * actions never discards a user's existing remaps.
 */
export function validKeys(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object') return { ...DEFAULT_KEYS };
  const stored = value as Record<string, unknown>, actions = Object.keys(DEFAULT_KEYS), keys: Record<string, string> = {}, taken = new Set<string>();
  for (const action of actions) {
    if (!(action in stored)) continue;
    const chord = stored[action];
    if (typeof chord !== 'string' || (chord !== '' && (!CHORD.test(chord) || taken.has(chord)))) return { ...DEFAULT_KEYS };
    keys[action] = chord; if (chord) taken.add(chord);
  }
  for (const action of actions) {
    if (action in keys) continue;
    const chord = DEFAULT_KEYS[action];
    keys[action] = taken.has(chord) ? '' : chord; if (keys[action]) taken.add(chord);
  }
  return keys;
}

const readStored = () => { try { return validKeys(JSON.parse(localStorage.getItem(STORAGE) ?? 'null')); } catch { return { ...DEFAULT_KEYS }; } };
const KeyContext = createContext({ keys: DEFAULT_KEYS, update: (_action: string, _chord: string) => {}, reset: () => {} });
export function HotkeyProvider({ children }: { children: ReactNode }) {
  const [keys, setKeys] = useState(readStored);
  useEffect(() => { try { localStorage.setItem(STORAGE, JSON.stringify(keys)); } catch { /* Session remains usable when storage is full. */ } }, [keys]);
  // Another tab (or window) remapped a key: adopt it so every open workstation agrees.
  useEffect(() => {
    const sync = (event: StorageEvent) => { if (event.key === STORAGE) setKeys(readStored()); };
    window.addEventListener('storage', sync); return () => window.removeEventListener('storage', sync);
  }, []);
  return <KeyContext.Provider value={{ keys, update: (action, chord) => setKeys(previous => keyConflict(previous, action, chord) ? previous : validKeys({ ...previous, [action]: chord })), reset: () => setKeys({ ...DEFAULT_KEYS }) }}>{children}</KeyContext.Provider>;
}
export const useHotkeys = () => useContext(KeyContext);
export const shortcutLabel = (keys: Record<string, string>, action: string) => keys[action] || 'unassigned';

export function useShortcutActions(actions: Record<string, (() => void) | undefined>, enabled = true) {
  const { keys } = useHotkeys();
  useEffect(() => {
    if (!enabled) return;
    const handle = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat || event.isComposing || (event.target as HTMLElement)?.closest?.('input,textarea,select,[contenteditable=true],[role=dialog]') || document.querySelector('[aria-modal=true]')) return;
      const chord = keyChord(event);
      const action = chord ? Object.keys(keys).find(action => keys[action] === chord) : undefined;
      if (action && actions[action]) { event.preventDefault(); actions[action]?.(); }
    };
    window.addEventListener('keydown', handle); return () => window.removeEventListener('keydown', handle);
  }, [keys, actions, enabled]);
}

export function HotkeyEditor() {
  const { keys, update, reset } = useHotkeys(), [query, setQuery] = useState(''), [error, setError] = useState('');
  const matches = (action: string) => `${KEY_LABELS[action]} ${keys[action]}`.toLowerCase().includes(query.toLowerCase());
  const row = (action: string) => <label className="hotkey-row" key={action}>{KEY_LABELS[action]}<input aria-label={`Shortcut for ${KEY_LABELS[action]}`} readOnly value={keys[action]} placeholder="Unassigned" onKeyDown={event => {
    if (event.key === 'Tab') return; event.preventDefault();
    if ((event.key === 'Backspace' || event.key === 'Delete') && !event.ctrlKey && !event.altKey && !event.shiftKey && !event.metaKey) { setError(''); update(action, ''); return; }
    const chord = keyChord(event.nativeEvent); if (!chord) return;
    const conflict = keyConflict(keys, action, chord);
    if (conflict) setError(`${chord} is already assigned to ${KEY_LABELS[conflict]}`);
    else if (validKeys({ ...keys, [action]: chord })[action] !== chord) setError('Use a letter, digit, arrow, Space or F1–F12 with optional modifiers');
    else { setError(''); update(action, chord); }
  }} /></label>;
  return <section className="hotkey-editor"><h2>Keyboard shortcut remapper</h2><input aria-label="Search shortcuts" placeholder="Search actions or keys" value={query} onChange={e => setQuery(e.target.value)} />
    <p className="muted">Focus a shortcut and press a new key combination (Backspace clears it). Conflicts are rejected. Shortcuts pause while typing or using a dialog.</p>
    {KEY_GROUPS.map(group => { const shown = group.actions.filter(matches); return shown.length ? <fieldset className="hotkey-group" key={group.title}><legend>{group.title}</legend>{shown.map(row)}</fieldset> : null; })}
    {error && <p role="alert">{error}</p>}<button className="button" onClick={() => { reset(); setError(''); }}>Restore default shortcuts</button></section>;
}
