/**
 * Harness for e2e/liveStream.e2e.ts: mounts the real LiveStreamCanvas and counts how many times React renders it,
 * so the spec can prove a binary frame stream causes no re-renders.
 */
import { createRoot } from 'react-dom/client';
import { LiveStreamCanvas } from '../src/components/LiveStreamCanvas';
import '../src/styles.css';

(window as unknown as { __liveRenders: number }).__liveRenders = 0;
function Probe() {
  (window as unknown as { __liveRenders: number }).__liveRenders += 1;
  return <LiveStreamCanvas/>;
}
createRoot(document.getElementById('root')!).render(<Probe/>);
