import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react';

/**
 * Windowed list / grid (Stage 11). Thousands of extracted face crops, pose-grid thumbnails, history cards or rendered
 * outputs would otherwise mount thousands of DOM nodes and decode thousands of <img>s; this keeps only the rows that
 * intersect the viewport (plus `overscan`) mounted, so scrolling stays at 60 FPS with a constant DOM footprint.
 *
 * - Rows are laid out with a CSS grid (`columns` fixed, or derived from `minColumnWidth`), so single-column lists and
 *   responsive thumbnail grids use the same component.
 * - Row heights start at `estimateHeight` and are replaced by the measured height, so cards of different heights are
 *   fine and the scroll extent corrects itself as rows are first seen.
 * - It scrolls inside its own box when `maxHeight` is set, otherwise it follows the nearest scrollable ancestor (or the
 *   window), so it drops into existing page layouts unchanged.
 */

export interface WindowInput { rowCount: number; heights: ReadonlyMap<number, number>; estimate: number; viewTop: number; viewBottom: number; overscan: number }
export interface WindowResult { first: number; last: number; offset: number; total: number }

/** Which rows [first, last] must be mounted for the visible band, where the first mounted row starts, and the full extent. */
export function windowRange({ rowCount, heights, estimate, viewTop, viewBottom, overscan }: WindowInput): WindowResult {
  if (rowCount <= 0) return { first: 0, last: -1, offset: 0, total: 0 };
  let cursor = 0, first = -1, last = rowCount - 1, firstOffset = 0;
  const starts: number[] = new Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    starts[row] = cursor;
    const height = heights.get(row) ?? estimate;
    if (first < 0 && cursor + height > viewTop) first = row;
    if (cursor <= viewBottom) last = row;
    cursor += height;
  }
  if (first < 0) first = rowCount - 1;
  first = Math.max(0, first - overscan);
  last = Math.min(rowCount - 1, Math.max(first, last) + overscan);
  firstOffset = starts[first];
  return { first, last, offset: firstOffset, total: cursor };
}

export const columnsFor = (width: number, minColumnWidth: number, gap: number) => Math.max(1, Math.floor((width + gap) / (minColumnWidth + gap)));

const SCROLLABLE = /(auto|scroll|overlay)/;
function scrollParent(node: HTMLElement | null): HTMLElement | null {
  for (let element = node?.parentElement ?? null; element; element = element.parentElement) {
    if (element === document.body || element === document.documentElement) return null;   // the window itself scrolls
    if (SCROLLABLE.test(getComputedStyle(element).overflowY)) return element;
  }
  return null;
}

export interface VirtualListProps<T> {
  items: readonly T[];
  itemKey: (item: T, index: number) => string | number;
  renderItem: (item: T, index: number) => ReactNode;
  /** Expected height of one row in px, gap included. Measured heights replace it once rows have been seen. */
  estimateHeight: number;
  /** Fixed column count; otherwise derived from `minColumnWidth`. Default 1 (a plain list). */
  columns?: number;
  minColumnWidth?: number;
  gap?: number;
  overscan?: number;
  /** Scroll inside this box (px or CSS length). Omit to follow the page / nearest scrollable ancestor. */
  maxHeight?: number | string;
  className?: string;
  style?: CSSProperties;
  role?: string;
  'aria-label'?: string;
  /** Viewport height / width assumed where layout reports 0 (jsdom, hidden screens): enough to mount the first rows. */
  fallbackViewportHeight?: number;
  fallbackWidth?: number;
  empty?: ReactNode;
}

export default function VirtualList<T>({ items, itemKey, renderItem, estimateHeight, columns, minColumnWidth, gap = 8, overscan = 3, maxHeight, className = '', style, role = 'list', 'aria-label': ariaLabel, fallbackViewportHeight = 720, fallbackWidth = 960, empty }: VirtualListProps<T>) {
  const outer = useRef<HTMLDivElement>(null), spacer = useRef<HTMLDivElement>(null);
  const heights = useRef(new Map<number, number>());
  const [layout, setLayout] = useState({ columns: columns ?? 1, first: 0, last: Math.max(0, Math.ceil(fallbackViewportHeight / Math.max(1, estimateHeight)) + overscan), offset: 0, total: 0 });
  const count = items.length;

  const measure = useCallback(() => {
    const box = outer.current, inner = spacer.current;
    if (!box || !inner) return;
    const innerRect = inner.getBoundingClientRect();
    const width = innerRect.width || box.clientWidth || fallbackWidth;
    const cols = columns ?? (minColumnWidth ? columnsFor(width, minColumnWidth, gap) : 1);
    const rowCount = Math.ceil(count / cols);
    const scroller = maxHeight !== undefined ? box : scrollParent(box);
    let top = 0, bottom = typeof window === 'undefined' ? 0 : window.innerHeight;
    if (scroller) { const rect = scroller.getBoundingClientRect(); top = rect.top; bottom = rect.bottom; }
    if (!(bottom - top > 0)) bottom = top + fallbackViewportHeight;                 // layout-less environment
    const viewTop = Math.max(0, top - innerRect.top), viewBottom = Math.max(viewTop, bottom - innerRect.top);
    const next = windowRange({ rowCount, heights: heights.current, estimate: estimateHeight, viewTop, viewBottom, overscan });
    setLayout(previous => previous.columns === cols && previous.first === next.first && previous.last === next.last && previous.offset === next.offset && previous.total === next.total ? previous : { columns: cols, ...next });
  }, [columns, minColumnWidth, gap, count, estimateHeight, overscan, maxHeight, fallbackViewportHeight, fallbackWidth]);

  // Re-window whenever the data or geometry changes, and on every scroll / resize of whatever scrolls us.
  useLayoutEffect(() => { measure(); }, [measure]);
  useEffect(() => {
    const box = outer.current; if (!box) return undefined;
    const scroller = maxHeight !== undefined ? box : scrollParent(box);
    const target: HTMLElement | Window = scroller ?? window;
    let frame = 0;
    const schedule = () => { if (frame) return; frame = requestAnimationFrame(() => { frame = 0; measure(); }); };
    target.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule);
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(schedule) : null;
    observer?.observe(box);
    return () => { target.removeEventListener('scroll', schedule); window.removeEventListener('resize', schedule); observer?.disconnect(); if (frame) cancelAnimationFrame(frame); };
  }, [measure, maxHeight]);

  // Replace estimates with measured row heights so variable-height cards keep a truthful scroll extent.
  const rows = useRef(new Map<number, HTMLElement>());
  useLayoutEffect(() => {
    let changed = false;
    for (const [row, element] of rows.current) {
      const height = element.offsetHeight;
      if (height > 0 && Math.abs((heights.current.get(row) ?? estimateHeight) - height) > 0.5) { heights.current.set(row, height); changed = true; }
    }
    if (changed) measure();
  });
  useEffect(() => { heights.current.clear(); }, [layout.columns, estimateHeight]);   // a new column count changes every row's height

  const slice = useMemo(() => {
    const out: { row: number; entries: { item: T; index: number }[] }[] = [];
    for (let row = layout.first; row <= layout.last; row++) {
      const entries: { item: T; index: number }[] = [];
      for (let column = 0; column < layout.columns; column++) { const index = row * layout.columns + column; if (index < count) entries.push({ item: items[index], index }); }
      if (entries.length) out.push({ row, entries });
    }
    return out;
  }, [items, count, layout.first, layout.last, layout.columns]);

  if (!count) return <>{empty ?? null}</>;
  return (
    <div ref={outer} className={`virtual-list ${className}`} role={role} aria-label={ariaLabel}
      style={{ position: 'relative', ...(maxHeight !== undefined ? { overflowY: 'auto', maxHeight } : {}), ...style }} data-virtual-count={count} data-virtual-mounted={slice.reduce((n, r) => n + r.entries.length, 0)}>
      <div ref={spacer} className="virtual-spacer" style={{ position: 'relative', height: Math.max(layout.total, 0) || undefined, width: '100%' }}>
        <div className="virtual-window" style={{ position: 'absolute', left: 0, right: 0, top: 0, transform: `translateY(${layout.offset}px)` }}>
          {slice.map(({ row, entries }) => (
            <div key={row} role="presentation" data-row={row} ref={element => { if (element) rows.current.set(row, element); else rows.current.delete(row); }}
              style={{ display: 'grid', gridTemplateColumns: `repeat(${layout.columns}, minmax(0, 1fr))`, gap, paddingBottom: gap, alignItems: 'start' }}>
              {entries.map(({ item, index }) => <div key={itemKey(item, index)} role="listitem" aria-setsize={count} aria-posinset={index + 1} className="virtual-item" style={{ minWidth: 0 }}>{renderItem(item, index)}</div>)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
