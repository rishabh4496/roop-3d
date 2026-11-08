import { useEffect, useRef } from 'react';
import { LiveFrameStream, liveStreamSupported, type FrameHeader } from '../liveStream';

/** The canvas hides itself again if no frame has arrived for this long (render finished, or the stream dropped). */
export const LIVE_STALE_MS = 2500;

/**
 * Live render monitor.  Frames arrive over a binary WebSocket, are decoded by `createImageBitmap` and drawn straight
 * onto a canvas with `bitmaprenderer` (a transfer, not a copy; `2d` is the fallback).  Nothing here touches React state
 * per frame: the canvas, the readout and the visibility are all driven imperatively from refs, so a 30 fps stream
 * causes zero component renders and zero DOM reconciliation.
 */
export function LiveStreamCanvas({ enabled = true, onFirstFrame }: { enabled?: boolean; onFirstFrame?: () => void }) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const readout = useRef<HTMLSpanElement | null>(null);
  const wrapper = useRef<HTMLDivElement | null>(null);
  const firstFrame = useRef(onFirstFrame);
  firstFrame.current = onFirstFrame;      // a new callback identity must never restart the stream

  useEffect(() => {
    const element = canvas.current, wrap = wrapper.current;
    if (!enabled || !element || !wrap || !liveStreamSupported()) return;
    let context: ImageBitmapRenderingContext | CanvasRenderingContext2D | null = null;
    let pending: { bitmap: ImageBitmap; header: FrameHeader } | null = null;
    let raf = 0, lastFrameAt = 0, lastReadout = 0, announced = false, hideTimer: ReturnType<typeof setTimeout> | undefined;

    const draw = () => {
      raf = 0;
      const frame = pending;
      pending = null;
      if (!frame) return;
      const { bitmap, header } = frame;
      const { width, height } = bitmap;            // a transferred bitmap is detached (0×0) afterwards
      if (element.width !== width || element.height !== height) { element.width = width; element.height = height; context = null; }
      context ??= element.getContext('bitmaprenderer') ?? element.getContext('2d');
      if (context && 'transferFromImageBitmap' in context) context.transferFromImageBitmap(bitmap);   // zero-copy hand-over
      else if (context) { (context as CanvasRenderingContext2D).drawImage(bitmap, 0, 0); bitmap.close(); }
      lastFrameAt = performance.now();
      wrap.dataset.live = 'true';
      if (!announced) { announced = true; firstFrame.current?.(); }
      if (readout.current && (!lastReadout || lastFrameAt - lastReadout > 250)) {                         // text immediately, then at 4 Hz, not per frame
        lastReadout = lastFrameAt;
        const [s1, s2, s3] = header.stageMs;
        readout.current.textContent = `${stream.stats.fps} fps · ${width}×${height} · detect ${s1.toFixed(0)} · swap ${s2.toFixed(0)} · blend ${s3.toFixed(0)} ms`;
      }
      clearTimeout(hideTimer);
      hideTimer = setTimeout(() => { if (performance.now() - lastFrameAt >= LIVE_STALE_MS) wrap.dataset.live = 'false'; }, LIVE_STALE_MS);
    };

    const stream = new LiveFrameStream({
      onFrame: (bitmap, header) => {
        if (pending) pending.bitmap.close();            // a newer frame arrived before the repaint: show only the newest
        pending = { bitmap, header };
        if (!raf) raf = requestAnimationFrame(draw);
      },
    });
    stream.start();
    return () => {
      stream.stop();
      cancelAnimationFrame(raf);
      clearTimeout(hideTimer);
      pending?.bitmap.close();
      pending = null;
      wrap.dataset.live = 'false';
    };
  }, [enabled]);

  return (
    <div ref={wrapper} className="live-stream" data-testid="live-stream" data-live="false" aria-live="off">
      <canvas ref={canvas} className="live-stream-canvas" aria-label="Live render stream"/>
      <span ref={readout} className="live-stream-readout" role="status"/>
    </div>
  );
}
