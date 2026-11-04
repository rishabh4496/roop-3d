import type { WorkflowProgress } from './workflowState';
import type { WorkflowStepId } from './helpContent';

/**
 * One line under the navigation: what to do next, and a single button that does it. When the action cannot run yet
 * (the service is still starting, or a task is in progress) the button is `aria-disabled` rather than `disabled`, so it
 * stays focusable and pressing it explains why instead of doing nothing.
 */
export default function NextStepBar({ progress, onAction, onGuide, blockedReason = null, busy = false }: { progress: WorkflowProgress; onAction: (id: WorkflowStepId) => void; onGuide: () => void; blockedReason?: string | null; busy?: boolean }) {
  const next = progress.next;
  const blocked = !!blockedReason || busy;
  return (
    <div className="next-step-bar" role="region" aria-label="Next step" data-testid="next-step-bar">
      <span className="eyebrow">{next ? `STEP ${progress.nextNumber} OF ${progress.total}` : 'ALL STEPS DONE'}</span>
      {next ? <>
        <strong>{next.title}</strong>
        <span className="muted next-step-goal">{next.goal}</span>
        <button type="button" className="button small accent" aria-disabled={blocked || undefined} title={blockedReason ?? (busy ? 'Another task is running' : undefined)}
          onClick={() => { if (!busy) onAction(next.id); }}>{next.action}</button>
      </> : <strong>Every step is done. Your video is rendered or queued.</strong>}
      <button type="button" className="button small" onClick={onGuide}>Open guide</button>
    </div>
  );
}
