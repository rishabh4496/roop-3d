import React from 'react';
import useHistoryStore from '../../stores/useHistoryStore';
import Modal from '../controls/Modal';

/** @param {any} props */
export default function HistoryDrawer({ onClose }) {
  const history = useHistoryStore();
  return <Modal title="Non-destructive edit history" close={onClose}><p className="muted">Undo parameters, assignments, roles, masks and curve edits. New edits discard the redo branch. History is scoped to the current target and bounded to 120 commands / 64 MiB.</p><div className="history-actions"><button className="button" disabled={!history.past.length} onClick={history.undo}>Undo</button><button className="button" disabled={!history.future.length} onClick={history.redo}>Redo</button></div><ol className="history-list"><li><button className="button" disabled={!history.past.length} onClick={() => { const count = history.past.length; for (let i = 0; i < count; i++) useHistoryStore.getState().undo(); }}>Original workspace</button></li>{history.past.map((command, index) => <li key={command.id}><button className="button" aria-current={index === history.past.length - 1 ? 'step' : undefined} onClick={() => { const count = history.past.length - index - 1; for (let i = 0; i < count; i++) useHistoryStore.getState().undo(); }}>{command.label}</button></li>)}{[...history.future].reverse().map((command, index) => <li key={command.id} className="redo-entry"><button className="button" onClick={() => { for (let i = 0; i <= index; i++) useHistoryStore.getState().redo(); }}>{command.label} · undone</button></li>)}</ol></Modal>;
}
