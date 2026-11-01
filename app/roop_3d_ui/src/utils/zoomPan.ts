/**
 * Shared zoom/pan maths for the preview stage and timeline.
 * Ported from roop-ultimate.
 */

export const ZOOM_MIN = 0.1;
export const ZOOM_MAX = 16.0;

export function uiScale(el: HTMLElement | null): number {
  if (!el) return 1;
  const w = el.offsetWidth;
  if (!w) return 1;
  const s = el.getBoundingClientRect().width / w;
  return Number.isFinite(s) && s > 0.01 ? s : 1;
}

export function stageSize(stage: HTMLElement | null): { w: number; h: number } {
  if (!stage) return { w: 0, h: 0 };
  const r = stage.getBoundingClientRect ? stage.getBoundingClientRect() : { width: (stage as any).clientWidth || 0, height: (stage as any).clientHeight || 0 };
  const s = uiScale(stage);
  return { w: r.width / s, h: r.height / s };
}

export function contentSize(el: HTMLElement | null): { w: number; h: number } {
  if (!el) return { w: 0, h: 0 };
  const r = el.getBoundingClientRect ? el.getBoundingClientRect() : { width: el.offsetWidth || 0, height: el.offsetHeight || 0 };
  const s = uiScale(el);
  return { w: r.width / s, h: r.height / s };
}

export function panBounds(
  stage: HTMLElement | null,
  zoom: number,
  content?: HTMLElement | null
): { x: number; y: number } {
  if (!stage || zoom <= 1) return { x: 0, y: 0 };
  const s = stageSize(stage);
  const c = content ? contentSize(content) : s;
  return {
    x: Math.max(0, (c.w * zoom - s.w) / 2),
    y: Math.max(0, (c.h * zoom - s.h) / 2),
  };
}

export function clampPan(
  pan: { x: number; y: number },
  zoom: number,
  stage: HTMLElement | null,
  content?: HTMLElement | null
): { x: number; y: number } {
  const b = panBounds(stage, zoom, content);
  return {
    x: Math.max(-b.x, Math.min(b.x, pan.x)),
    y: Math.max(-b.y, Math.min(b.y, pan.y)),
  };
}

export function panAnchoredAt(
  client: { x: number; y: number },
  stage: HTMLElement | null,
  prevZoom: number,
  nextZoom: number,
  pan: { x: number; y: number }
): { x: number; y: number } {
  if (!prevZoom) return pan;
  const rect = stage?.getBoundingClientRect ? stage.getBoundingClientRect() : { left: 0, top: 0, width: (stage as any)?.clientWidth || 800, height: (stage as any)?.clientHeight || 600 };
  const s = stage ? uiScale(stage) : 1;
  const sx = (client.x - (rect.left + rect.width / 2)) / s;
  const sy = (client.y - (rect.top + rect.height / 2)) / s;
  const k = nextZoom / prevZoom;
  return { x: sx - (sx - pan.x) * k, y: sy - (sy - pan.y) * k };
}

export function panCenteringAt(
  client: { x: number; y: number },
  stage: HTMLElement | null,
  zoom: number,
  content?: HTMLElement | null
): { x: number; y: number } {
  if (!stage) return { x: 0, y: 0 };
  const rect = stage?.getBoundingClientRect ? stage.getBoundingClientRect() : { left: 0, top: 0, width: (stage as any)?.clientWidth || 800, height: (stage as any)?.clientHeight || 600 };
  const s = uiScale(stage);
  const u = {
    x: (client.x - (rect.left + rect.width / 2)) / s,
    y: (client.y - (rect.top + rect.height / 2)) / s,
  };
  return clampPan({ x: -u.x * zoom, y: -u.y * zoom }, zoom, stage, content);
}

export function wheelZoom(
  deltaY: number,
  zoom: number,
  max = ZOOM_MAX,
  min = ZOOM_MIN,
  step = 0.15
): number | null {
  const raw = deltaY < 0 ? zoom * (1 + step) : zoom / (1 + step);
  const next = Math.min(Math.max(raw, min), max);
  return Math.abs(next - zoom) < 1e-4 ? null : next;
}
