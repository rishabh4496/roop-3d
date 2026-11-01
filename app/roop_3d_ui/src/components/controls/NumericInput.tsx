import { useEffect, useRef, useState } from 'react';

/**
 * Bounded recursive descent; never execute user text as JavaScript.
 *
 * A leading `+`, `*` or `/` is *relative to the field's current value*: with the value at 0.4, `*1.25` → 0.5,
 * `/2` → 0.2 and `+0.05` → 0.45. A leading `-` stays a negative literal (angles, offsets); type `+-0.1` to subtract.
 * `×`, `÷` and `−` are accepted as `*`, `/` and `-`.
 */
export function evaluateExpression(text: string, current = 0): number {
  if (!text.trim() || text.length > 128) throw new Error('Enter a numeric expression');
  let source = text.trim().replace(/×/g, '*').replace(/÷/g, '/').replace(/−/g, '-');
  if (/^[+*/]/.test(source)) source = `(${Number.isFinite(current) ? current : 0})${source[0]}(${source.slice(1)})`;
  const tokens = source.match(/(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|[()+*/-]/gi) ?? [];
  if (tokens.join('') !== source.replace(/\s/g, '')) throw new Error('Use numbers, parentheses and + − × /');
  let at = 0;
  function primary(): number {
    const token = tokens[at++];
    if (token === '+') return primary();
    if (token === '-') return -primary();
    if (token === '(') { const value = sum(); if (tokens[at++] !== ')') throw new Error('Missing closing parenthesis'); return value; }
    if (!token || !/^[\d.]/.test(token)) throw new Error('Incomplete expression');
    return Number(token);
  }
  function product(): number { let value = primary(); while (tokens[at] === '*' || tokens[at] === '/') { const op = tokens[at++], rhs = primary(); value = op === '*' ? value * rhs : value / rhs; } return value; }
  function sum(): number { let value = product(); while (tokens[at] === '+' || tokens[at] === '-') { const op = tokens[at++], rhs = product(); value = op === '+' ? value + rhs : value - rhs; } return value; }
  const result = sum();
  if (at !== tokens.length || !Number.isFinite(result)) throw new Error('Expression must produce a finite number');
  return result;
}
export function constrain(value: number, min = -Infinity, max = Infinity, integer = false) {
  return Math.min(max, Math.max(min, integer ? Math.round(value) : Number(value.toPrecision(12))));
}
export function scrubDelta(pixels: number, step: number, shift = false, alt = false) { return pixels * step * (shift ? 10 : 1) * (alt ? .1 : 1); }

type Props = { label: string; value: number | null; onChange: (value: number | null) => void; id?: string; min?: number; max?: number; step?: number; integer?: boolean; disabled?: boolean; nullable?: boolean; placeholder?: string; title?: string };
export default function NumericInput({ label, value, onChange, id, min, max, step = 1, integer = false, disabled = false, nullable = false, placeholder, title }: Props) {
  const [draft, setDraft] = useState(String(value ?? '')), [error, setError] = useState('');
  const drag = useRef<{ x: number; value: number } | null>(null);
  const inputId = id ?? `numeric-${label.replace(/\W/g, '-')}`;
  useEffect(() => { setDraft(String(value ?? '')); }, [value]);
  const commit = () => {
    if (disabled) return;
    try { const next = !draft.trim() && nullable ? null : constrain(evaluateExpression(draft, value ?? 0), min, max, integer); setError(''); setDraft(String(next ?? '')); if (next !== value) onChange(next); }
    catch (e) { setError((e as Error).message); }
  };
  const change = (v: number) => { onChange(constrain(v, min, max, integer)); setError(''); };
  // Instant evaluation: while the draft is an expression (not a plain number), show what Enter will commit.
  const preview = (() => {
    const text = draft.trim();
    if (!text || /^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(text) || error) return null;
    try { return constrain(evaluateExpression(text, value ?? 0), min, max, integer); } catch { return null; }
  })();
  return <div className="numeric-control">
    <label htmlFor={inputId} className="scrub-label" title="Drag to scrub · Shift 10× · Alt 0.1×" tabIndex={disabled ? -1 : 0}
      onPointerDown={event => { if (disabled || event.button !== 0) return; event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); drag.current = { x: event.clientX, value: value ?? 0 }; }}
      onPointerMove={event => { if (drag.current) change(drag.current.value + scrubDelta(event.clientX - drag.current.x, step, event.shiftKey, event.altKey)); }}
      onPointerUp={event => { drag.current = null; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
      onPointerCancel={() => { drag.current = null; }} onLostPointerCapture={() => { drag.current = null; }}
      onKeyDown={event => { if (!disabled && ['ArrowLeft', 'ArrowRight'].includes(event.key)) { event.preventDefault(); change((value ?? 0) + scrubDelta(event.key === 'ArrowRight' ? 1 : -1, step, event.shiftKey, event.altKey)); } }}>{label}</label>
    <input id={inputId} aria-label={label} title={title} disabled={disabled} inputMode="decimal" value={draft} placeholder={placeholder} aria-invalid={!!error} aria-describedby={error ? `${inputId}-error` : undefined}
      onChange={event => { setDraft(event.target.value); setError(''); }} onBlur={commit} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); commit(); } if (event.key === 'Escape') { event.preventDefault(); setDraft(String(value ?? '')); setError(''); } }} />
    {preview !== null && <small className="expression-preview" data-testid="expression-preview" aria-live="polite">= {preview}</small>}
    {error && <small role="alert" id={`${inputId}-error`}>{error}</small>}
  </div>;
}
