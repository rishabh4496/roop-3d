/**
 * Render-count probe for performance gates. When `globalThis.__R3D_RENDER_PROBE__` is an object (set by the Vitest harness
 * or a Playwright `addInitScript`), every probed component increments `probe[name]` each time its function body runs.
 * With no sink it is a single property read, so shipping it in production code is free.
 */
type Sink = Record<string, number>;
export function renderProbe(name: string): void {
  const sink = (globalThis as unknown as { __R3D_RENDER_PROBE__?: Sink }).__R3D_RENDER_PROBE__;
  if (sink) sink[name] = (sink[name] ?? 0) + 1;
}
