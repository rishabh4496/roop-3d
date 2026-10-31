import { useSyncExternalStore } from 'react';

const listeners = new Set();
const LIMIT = 120;
const MEMORY_LIMIT = 64 * 1024 * 1024;
let serial = 0;
let state;
const emit = (patch) => { state = { ...state, ...patch }; listeners.forEach(listener => listener()); };

/** Commands retain reversible edits, never execute application code from persisted text. */
export function executeCommand(command) {
  if (state.replaying) { command.redo(); return; }
  command.redo();
  let past = [...state.past];
  const now = Date.now(), previous = past.at(-1);
  const next = { ...command, id: ++serial, at: now, bytes: command.bytes ?? 1024 };
  if (command.mergeKey && previous?.mergeKey === command.mergeKey && now - previous.at < 350 && !state.future.length) {
    past[past.length - 1] = { ...next, id: previous.id, undo: previous.undo, bytes: Math.max(next.bytes, previous.bytes) };
  } else past.push(next);
  let bytes = past.reduce((sum, item) => sum + item.bytes, 0);
  while (past.length > 1 && (past.length > LIMIT || bytes > MEMORY_LIMIT)) bytes -= past.shift().bytes;
  emit({ past, future: [] });
}
function travel(direction) {
  const source = direction === 'undo' ? state.past : state.future;
  const command = source.at(-1); if (!command) return;
  emit({ replaying: true });
  try {
    command[direction]();
    emit(direction === 'undo' ? { past: state.past.slice(0, -1), future: [...state.future, command] }
      : { past: [...state.past, command], future: state.future.slice(0, -1) });
  } finally { emit({ replaying: false }); }
}
export function replaceCurves(curves) { emit({ curves }); }
export function editCurves(curves, label = 'Edit parameter curve', mergeKey) {
  const before = state.curves, after = structuredClone(curves);
  if (JSON.stringify(before) === JSON.stringify(after)) return;
  executeCommand({ label, mergeKey, undo: () => replaceCurves(before), redo: () => replaceCurves(after) });
}
const actions = {
  execute: executeCommand, undo: () => travel('undo'), redo: () => travel('redo'), editCurves,
  reset: () => emit({ past: [], future: [], curves: {}, slots: { A: null, B: null, C: null, D: null }, previewSlot: null, comparison: null }),
  captureSlot: (slot, snapshot) => {
    if (!['A', 'B', 'C', 'D'].includes(slot) || !snapshot?.image?.startsWith('data:image/')) throw new Error('A rendered frame is required for a snapshot');
    emit({ slots: { ...state.slots, [slot]: { ...snapshot, workspace: structuredClone(snapshot.workspace) } }, previewSlot: slot, comparison: null });
  },
  selectSlot: slot => { if (state.slots[slot]) emit({ previewSlot: slot, comparison: null }); },
  compare: (left, right) => { if (left !== right && state.slots[left] && state.slots[right]) emit({ comparison: [left, right], previewSlot: null }); },
  live: () => emit({ previewSlot: null, comparison: null }),
};
state = { past: [], future: [], replaying: false, curves: {}, slots: { A: null, B: null, C: null, D: null }, previewSlot: null, comparison: null, ...actions };
export function useHistoryStore(selector = value => value) {
  return useSyncExternalStore(callback => { listeners.add(callback); return () => listeners.delete(callback); }, () => selector(state), () => selector(state));
}
useHistoryStore.getState = () => state;
useHistoryStore.subscribe = listener => { listeners.add(listener); return () => listeners.delete(listener); };
export default useHistoryStore;
