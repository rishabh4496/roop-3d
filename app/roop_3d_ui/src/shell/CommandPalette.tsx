import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, spring } from '../motion';
import { useDialogA11y } from '../components/controls/useDialogA11y';
import { IconSearch } from './icons';

export interface Command {
  id: string;
  label: string;
  group: string;
  hint?: string;
  keywords?: string;
  disabled?: boolean;
  run: () => void;
}

/** Ctrl K launcher: jump to any screen or run a workspace action by name. Keyboard first, a11y-contract dialog. */
export default function CommandPalette({ open, onClose, commands }: { open: boolean; onClose: () => void; commands: Command[] }) {
  const [query, setQuery] = useState('');
  const [index, setIndex] = useState(0);
  const panel = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);
  useDialogA11y(panel, onClose, open, 'input');

  useEffect(() => { if (open) { setQuery(''); setIndex(0); } }, [open]);
  const results = useMemo(() => {
    const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    return commands.filter(command => !command.disabled && words.every(word => `${command.label} ${command.group} ${command.keywords ?? ''}`.toLowerCase().includes(word)));
  }, [commands, query]);
  useEffect(() => { setIndex(0); }, [query]);
  useEffect(() => { list.current?.querySelector('[aria-selected="true"]')?.scrollIntoView?.({ block: 'nearest' }); }, [index, results]);

  const run = (command: Command | undefined) => { if (!command) return; onClose(); window.setTimeout(command.run, 0); };
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'ArrowDown') { event.preventDefault(); setIndex(value => Math.min(results.length - 1, value + 1)); }
    else if (event.key === 'ArrowUp') { event.preventDefault(); setIndex(value => Math.max(0, value - 1)); }
    else if (event.key === 'Enter') { event.preventDefault(); run(results[index]); }
  };

  let lastGroup = '';
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="palette-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }} onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
          <motion.div ref={panel} className="palette" role="dialog" aria-modal="true" aria-label="Command palette" onKeyDown={onKeyDown}
            initial={{ opacity: 0, y: -14, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -8, scale: 0.985 }} transition={spring.snappy}>
            <div className="palette-search">
              <IconSearch />
              <input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search screens and actions…" aria-label="Search commands" role="combobox" aria-expanded="true" aria-controls="palette-list" aria-activedescendant={results[index] ? `palette-${results[index].id}` : undefined} />
              <kbd>Esc</kbd>
            </div>
            <ul id="palette-list" className="palette-list" role="listbox" aria-label="Commands" ref={list}>
              {results.map((command, position) => {
                const header = command.group !== lastGroup ? command.group : null;
                lastGroup = command.group;
                return (
                  <li key={command.id} role="presentation">
                    {header && <div className="palette-group">{header}</div>}
                    <div id={`palette-${command.id}`} role="option" aria-selected={position === index} className={`palette-item ${position === index ? 'active' : ''}`}
                      onMouseMove={() => setIndex(position)} onClick={() => run(command)}>
                      <span>{command.label}</span>{command.hint && <kbd>{command.hint}</kbd>}
                    </div>
                  </li>
                );
              })}
              {!results.length && <li className="palette-empty" role="presentation">No matching command</li>}
            </ul>
            <div className="palette-foot"><span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>Enter</kbd> run</span><span><kbd>Ctrl</kbd><kbd>K</kbd> toggle</span></div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
