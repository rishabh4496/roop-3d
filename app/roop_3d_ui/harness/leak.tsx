/**
 * Stage 18 harness (used only by e2e/resourceLeaks.e2e.ts): mounts and unmounts the real WebGL2 comparison canvas many
 * times under the resource ledger, and can deliberately leak one context so the spec can prove the ledger sees leaks.
 */
import { createRoot } from 'react-dom/client';
import WebGLViewport from '../src/components/canvas/WebGLViewport';
import { installResourceLedger } from '../src/utils/resourceLedger';

installResourceLedger();

function synthetic(width: number, height: number, color: string): Promise<HTMLImageElement> {
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d')!; ctx.fillStyle = color; ctx.fillRect(0, 0, width, height);
  return new Promise(resolve => { const image = new Image(); image.onload = () => resolve(image); image.src = canvas.toDataURL('image/png'); });
}
const frames = (n: number) => new Promise<void>(resolve => { const step = (left: number) => (left ? requestAnimationFrame(() => step(left - 1)) : resolve()); step(n); });

async function main() {
  const a = await synthetic(256, 144, '#c80000'), b = await synthetic(256, 144, '#00c800');
  (window as any).__cycle = async (count: number) => {
    for (let i = 0; i < count; i++) {
      const host = document.createElement('div');
      host.style.cssText = 'position:relative;width:320px;height:180px';
      document.body.appendChild(host);
      const root = createRoot(host);
      root.render(<WebGLViewport imageA={a} imageB={b} width={320} height={180} pixelRatio={1} mode="wipe" />);
      await frames(2);
      root.unmount();
      host.remove();
    }
  };
  /** A context that is created and then abandoned: removed from the page without being released. */
  (window as any).__leakOne = () => {
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    canvas.getContext('webgl2');
    (window as any).__leaked = canvas;     // keep it reachable so GC cannot hide the leak
    canvas.remove();
  };
  (window as any).__releaseLeak = () => {
    const canvas = (window as any).__leaked as HTMLCanvasElement;
    (canvas.getContext('webgl2') as WebGL2RenderingContext).getExtension('WEBGL_lose_context')?.loseContext();
    (window as any).__leaked = null;
  };
  (window as any).__ready = true;
}
void main();
