import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { Reorder, useDragControls, useReducedMotion } from 'framer-motion';
import { cn } from '../../lib/utils';

/**
 * Spring-animated drag-to-reorder list (ui-layouts "Drag Items", on framer-motion's `Reorder`).
 *
 *  - Neighbours slide out of the way on a spring while one item is dragged by its grip.
 *  - `onReorder` follows the drag live (for local state); `onCommit` fires once, when the item is dropped or moved with the
 *    keyboard, with the final order: that is the place to call the server, not on every hover swap.
 *  - Keyboard: focus the grip and press Alt+Up / Alt+Down (or use the move buttons). The result is announced politely.
 */
export function moveItem<T>(items: readonly T[], from: number, to: number): T[] {
  if (from === to || from < 0 || from >= items.length) return [...items];
  const target = Math.max(0, Math.min(items.length - 1, to));
  const next = [...items];
  const [item] = next.splice(from, 1);
  next.splice(target, 0, item);
  return next;
}

export const sameOrder = <T,>(a: readonly T[], b: readonly T[], key: (item: T) => string) => a.length === b.length && a.every((item, index) => key(item) === key(b[index]));

export interface DragRowContext {
  index: number;
  count: number;
  dragging: boolean;
  /** The grip and the two move buttons. Place them wherever the row wants them. */
  controls: ReactNode;
}
export interface DragItemsProps<T> {
  items: readonly T[];
  getKey: (item: T) => string;
  /** Human name of one item, for the screen-reader announcement and button labels. */
  getLabel?: (item: T) => string;
  renderItem: (item: T, context: DragRowContext) => ReactNode;
  onReorder?: (items: T[]) => void;
  onCommit?: (items: T[]) => void;
  axis?: 'x' | 'y';
  label: string;
  disabled?: boolean;
  className?: string;
  itemClassName?: string;
}

interface RowProps<T> {
  item: T; index: number; count: number; axis: 'x' | 'y'; label: string; disabled: boolean; className?: string;
  render: DragItemsProps<T>['renderItem']; move: (from: number, to: number) => void; setDragging: (value: boolean) => void; commit: () => void;
}

function DragRow<T>({ item, index, count, axis, label, disabled, className, render, move, setDragging, commit }: RowProps<T>) {
  const controls = useDragControls();
  const reduce = useReducedMotion();
  const [dragging, setLocalDragging] = useState(false);
  const back = axis === 'y' ? 'up' : 'left', forward = axis === 'y' ? 'down' : 'right';
  const key = (event: KeyboardEvent) => {
    if (!event.altKey || disabled) return;
    const delta = event.key === (axis === 'y' ? 'ArrowUp' : 'ArrowLeft') ? -1 : event.key === (axis === 'y' ? 'ArrowDown' : 'ArrowRight') ? 1 : 0;
    if (!delta) return;
    event.preventDefault();
    move(index, index + delta);
  };
  const controlsNode = (
    <span className="ul-drag__controls">
      <button type="button" className="ul-drag__grip" disabled={disabled} aria-label={`Reorder ${label}. Press Alt and the arrow keys to move it, or drag.`}
        onPointerDown={event => { if (!disabled) { event.preventDefault(); controls.start(event); } }} onKeyDown={key}><span aria-hidden>⠿</span></button>
      <button type="button" className="ul-drag__move" disabled={disabled || index === 0} aria-label={`Move ${label} ${back}`} onClick={() => move(index, index - 1)}><span aria-hidden>{axis === 'y' ? '▲' : '◀'}</span></button>
      <button type="button" className="ul-drag__move" disabled={disabled || index === count - 1} aria-label={`Move ${label} ${forward}`} onClick={() => move(index, index + 1)}><span aria-hidden>{axis === 'y' ? '▼' : '▶'}</span></button>
    </span>
  );
  return (
    <Reorder.Item value={item} as="li" dragListener={false} dragControls={controls} className={cn('ul-drag__item', dragging && 'is-dragging', className)}
      transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 520, damping: 38 }}
      whileDrag={reduce ? undefined : { scale: 1.015, zIndex: 3 }}
      onDragStart={() => { setLocalDragging(true); setDragging(true); }}
      onDragEnd={() => { setLocalDragging(false); setDragging(false); commit(); }}>
      {render(item, { index, count, dragging, controls: controlsNode })}
    </Reorder.Item>
  );
}

export function DragItems<T>({ items, getKey, getLabel, renderItem, onReorder, onCommit, axis = 'y', label, disabled = false, className, itemClassName }: DragItemsProps<T>) {
  const [order, setOrder] = useState<T[]>(() => [...items]);
  const [announcement, setAnnouncement] = useState('');
  const dragging = useRef(false);
  const latest = useRef(order);
  latest.current = order;
  const lastCommitted = useRef<T[]>([...items]);

  // Follow the parent's list (a refresh, a new job) unless a drag is in progress.
  useEffect(() => {
    if (dragging.current) return;
    setOrder(current => (sameOrder(current, items, getKey) ? current : [...items]));
    lastCommitted.current = [...items];
  }, [items, getKey]);

  const labelOf = (item: T) => getLabel?.(item) ?? getKey(item);
  const apply = (next: T[]) => { setOrder(next); latest.current = next; onReorder?.(next); };
  const commit = (next = latest.current) => {
    if (sameOrder(next, lastCommitted.current, getKey)) return;   // dropped where it started: nothing to save
    lastCommitted.current = [...next];
    onCommit?.([...next]);
  };
  const move = (from: number, to: number) => {
    if (disabled) return;
    const next = moveItem(latest.current, from, to);
    if (sameOrder(next, latest.current, getKey)) return;
    const item = latest.current[from];
    apply(next);
    setAnnouncement(`${labelOf(item)} moved to position ${next.findIndex(entry => getKey(entry) === getKey(item)) + 1} of ${next.length}.`);
    commit(next);
  };

  return (
    <>
      <Reorder.Group as="ol" axis={axis} values={order} onReorder={next => { if (!disabled) apply(next); }} className={cn('ul-drag', axis === 'x' && 'ul-drag--x', className)} aria-label={label}>
        {order.map((item, index) => (
          <DragRow key={getKey(item)} item={item} index={index} count={order.length} axis={axis} label={labelOf(item)} disabled={disabled} className={itemClassName}
            render={renderItem} move={move} setDragging={value => { dragging.current = value; }} commit={() => commit()} />
        ))}
      </Reorder.Group>
      <span className="sr-only" role="status" aria-live="polite">{announcement}</span>
    </>
  );
}
