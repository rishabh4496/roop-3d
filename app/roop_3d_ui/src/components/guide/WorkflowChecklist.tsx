import type { WorkflowProgress } from './workflowState';
import type { WorkflowStepId } from './helpContent';

const MARK: Record<string, string> = { done: '✓', current: '→', todo: '' };
const STATE_WORD: Record<string, string> = { done: 'done', current: 'next step', todo: 'not started' };

/**
 * The workflow as a live checklist. Every state is derived from real application state (see workflowState.ts), so a
 * new user always sees what is finished, what to do next and what "finished" looks like for each step.
 */
export default function WorkflowChecklist({ progress, onAction, compact = false }: { progress: WorkflowProgress; onAction?: (id: WorkflowStepId) => void; compact?: boolean }) {
  return (
    <div className="workflow-checklist" data-testid="workflow-checklist">
      <p className="workflow-count" role="status">{progress.doneCount} of {progress.total} steps done{progress.next ? ` · next: ${progress.next.title}` : ' · all steps done'}</p>
      <ol className={compact ? 'compact' : ''} aria-label="Local video workflow">
        {progress.steps.map((step, index) => (
          <li key={step.id} className={`workflow-step ${step.state}`} data-step={step.id} aria-current={step.state === 'current' ? 'step' : undefined}>
            <span className="workflow-mark" aria-hidden="true">{MARK[step.state] || index + 1}</span>
            <div className="workflow-body">
              <strong>{step.title}{step.id === 'project' && <small className="muted"> · recommended</small>}<span className="sr-only"> ({STATE_WORD[step.state]})</span></strong>
              <span className="muted">{step.goal}</span>
              {!compact && step.state !== 'done' && <>
                <ul>{step.how.map(item => <li key={item}>{item}</li>)}</ul>
                <small className="workflow-check"><b>You are done when:</b> {step.check}</small>
              </>}
              {onAction && step.state !== 'done' && <button type="button" className={`button small ${step.state === 'current' ? 'accent' : ''}`} onClick={() => onAction(step.id)}>{step.action}</button>}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
