/**
 * Real-browser harness for the WebGL2 comparison canvas (used only by the Playwright specs in `e2e/`).
 * It mounts <WebGLViewport> with synthetic, exactly-known images and exposes a tiny control surface on `window.__gl`
 * so a spec can change the mode and read back pixels from the real GPU (or SwiftShader) output.
 */
import { createRoot } from 'react-dom/client';
import { useState } from 'react';
import WebGLViewport from '../src/components/canvas/WebGLViewport';

type Props = Record<string, unknown>;
const WIDTH = 640, HEIGHT = 360;

/** Left half pure red, right half pure green; optional exact-pixel patch so differences are known to the unit. */
function synthetic(width: number, height: number, paint: (ctx: CanvasRenderingContext2D) => void): Promise<HTMLImageElement> {
  const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d')!; ctx.imageSmoothingEnabled = false; paint(ctx);
  return new Promise(resolve => { const image = new Image(); image.onload = () => resolve(image); image.src = canvas.toDataURL('image/png'); });
}

function Harness({ images }: { images: { a: HTMLImageElement; b: HTMLImageElement } }) {
  const [props, setProps] = useState<Props>({ mode: 'wipe' });
  (window as any).__setProps = (next: Props) => setProps(previous => ({ ...previous, ...next }));
  return <div style={{ position: 'relative', width: WIDTH, height: HEIGHT }}>
    <WebGLViewport imageA={images.a} imageB={images.b} width={WIDTH} height={HEIGHT} pixelRatio={1} preserveDrawingBuffer mode="wipe"
      onUnsupported={reason => { (window as any).__glUnsupported = reason; }} onReady={info => { (window as any).__glReady = info; }} {...props as any} />
  </div>;
}

async function main() {
  const bigW = Number(new URLSearchParams(location.search).get('w') ?? 256), bigH = Number(new URLSearchParams(location.search).get('h') ?? 144);
  const a = await synthetic(bigW, bigH, ctx => { ctx.fillStyle = '#c80000'; ctx.fillRect(0, 0, bigW / 2, bigH); ctx.fillStyle = '#00c800'; ctx.fillRect(bigW / 2, 0, bigW / 2, bigH); });
  // B equals A except a bright block in the lower-right quarter, so wipe/diff outputs have a known unchanged and changed region.
  const b = await synthetic(bigW, bigH, ctx => {
    ctx.fillStyle = '#c80000'; ctx.fillRect(0, 0, bigW / 2, bigH); ctx.fillStyle = '#00c800'; ctx.fillRect(bigW / 2, 0, bigW / 2, bigH);
    ctx.fillStyle = '#ffffff'; ctx.fillRect(bigW * 0.75, bigH * 0.5, bigW * 0.25, bigH * 0.5);
  });
  (window as any).__native = { a: [a.naturalWidth, a.naturalHeight], b: [b.naturalWidth, b.naturalHeight] };
  createRoot(document.getElementById('root')!).render(<Harness images={{ a, b }} />);
  (window as any).__readPixel = (x: number, y: number) => {
    const canvas = document.querySelector<HTMLCanvasElement>('[data-testid="webgl-viewport"]')!, gl = canvas.getContext('webgl2')!, out = new Uint8Array(4);
    gl.readPixels(x, canvas.height - 1 - y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, out);       // top-left origin like the page
    return [...out];
  };
  (window as any).__canvasSize = () => { const c = document.querySelector<HTMLCanvasElement>('[data-testid="webgl-viewport"]')!; return [c.width, c.height]; };
  (window as any).__ready = true;
}
void main();
