import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
export const DEFAULT_KEYS: Record<string, string> = { play: 'Space', previous: 'ArrowLeft', next: 'ArrowRight', markIn: 'I', markOut: 'O', solo: 'S', bypass: 'B', sweep: 'Shift+P', floating: 'Shift+F', save: 'Ctrl+S' };
export const KEY_LABELS: Record<string, string> = { play: 'Play / pause', previous: 'Previous frame', next: 'Next frame', markIn: 'Mark trim in', markOut: 'Mark trim out', solo: 'Solo selected pipeline node', bypass: 'Bypass selected pipeline node', sweep: 'Parameter sweep', floating: 'Floating preview', save: 'Save project' };
const STORAGE = 'roop3d-hotkeys-v1';
export function keyChord(event: Pick<KeyboardEvent, 'key' | 'ctrlKey' | 'altKey' | 'shiftKey' | 'metaKey'>) {
  if (['Control', 'Alt', 'Shift', 'Meta'].includes(event.key)) return '';
  const key = event.key === ' ' ? 'Space' : event.key.length === 1 ? event.key.toUpperCase() : event.key;
  return [event.ctrlKey && 'Ctrl', event.altKey && 'Alt', event.shiftKey && 'Shift', event.metaKey && 'Meta', key].filter(Boolean).join('+');
}
export function keyConflict(keys: Record<string, string>, action: string, chord: string) { return Object.keys(keys).find(key => key !== action && keys[key] === chord); }
export function validKeys(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object') return { ...DEFAULT_KEYS };
  const keys = Object.fromEntries(Object.keys(DEFAULT_KEYS).map(key => [key, (value as Record<string, unknown>)[key]]));
  if (Object.values(keys).some(value => typeof value !== 'string' || !/^(?:(?:Ctrl|Alt|Shift|Meta)\+)*(?:[A-Z0-9]|Space|ArrowLeft|ArrowRight|F(?:[1-9]|1[0-2]))$/.test(value as string)) || new Set(Object.values(keys)).size !== Object.keys(keys).length) return { ...DEFAULT_KEYS };
  return keys as Record<string, string>;
}
const KeyContext = createContext({ keys: DEFAULT_KEYS, update: (_action: string, _chord: string) => {}, reset: () => {} });
export function HotkeyProvider({ children }: { children: ReactNode }) {
  const [keys, setKeys] = useState(() => { try { return validKeys(JSON.parse(localStorage.getItem(STORAGE) ?? 'null')); } catch { return { ...DEFAULT_KEYS }; } });
  useEffect(() => { try { localStorage.setItem(STORAGE, JSON.stringify(keys)); } catch { /* Session remains usable when storage is full. */ } }, [keys]);
  return <KeyContext.Provider value={{ keys, update: (action, chord) => setKeys(previous => keyConflict(previous, action, chord) ? previous : validKeys({ ...previous, [action]: chord })), reset: () => setKeys({ ...DEFAULT_KEYS }) }}>{children}</KeyContext.Provider>;
}
export const useHotkeys = () => useContext(KeyContext);
export function useShortcutActions(actions: Record<string, (() => void) | undefined>, enabled = true) {
  const { keys } = useHotkeys();
  useEffect(() => {
    if (!enabled) return;
    const handle = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat || event.isComposing || (event.target as HTMLElement)?.closest?.('input,textarea,select,[contenteditable=true],[role=dialog]') || document.querySelector('[aria-modal=true]')) return;
      const action = Object.keys(keys).find(action => keys[action] === keyChord(event));
      if (action && actions[action]) { event.preventDefault(); actions[action]?.(); }
    };
    window.addEventListener('keydown', handle); return () => window.removeEventListener('keydown', handle);
  }, [keys, actions, enabled]);
}
export function HotkeyEditor() {
  const { keys, update, reset } = useHotkeys(), [query, setQuery] = useState(''), [error, setError] = useState('');
  return <section className="hotkey-editor"><h2>Keyboard shortcut remapper</h2><input aria-label="Search shortcuts" placeholder="Search actions or keys" value={query} onChange={e => setQuery(e.target.value)} />
    <p className="muted">Focus a shortcut and press a new key combination. Shortcuts pause while typing or using a dialog.</p>
    {Object.keys(keys).filter(action => `${KEY_LABELS[action]} ${keys[action]}`.toLowerCase().includes(query.toLowerCase())).map(action => <label className="hotkey-row" key={action}>{KEY_LABELS[action]}<input aria-label={`Shortcut for ${KEY_LABELS[action]}`} readOnly value={keys[action]} onKeyDown={event => {
      if (event.key === 'Tab') return; event.preventDefault();
      const chord = keyChord(event.nativeEvent); if (!chord) return;
      const conflict = keyConflict(keys, action, chord);
      if (conflict) setError(`${chord} is already assigned to ${KEY_LABELS[conflict]}`);
      else if (validKeys({ ...keys, [action]: chord })[action] !== chord) setError('Use a letter, digit, arrow, Space or F1–F12 with optional modifiers');
      else { setError(''); update(action, chord); }
    }} /></label>)}{error && <p role="alert">{error}</p>}<button className="button" onClick={() => { reset(); setError(''); }}>Restore default shortcuts</button></section>;
}
