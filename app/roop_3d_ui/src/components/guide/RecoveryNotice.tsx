import { explainError } from './errorGuide';

/**
 * The error banner body: what failed, the first thing to try, and (expandable) the cause and every step. Unknown
 * errors still show their message untouched, never a made-up explanation. Compact on purpose: the banner sits above
 * the workspace, so it must not push the preview off screen.
 */
export default function RecoveryNotice({ message, code, requestId, onGuide }: { message: string; code?: string | null; requestId?: string | null; onGuide?: () => void }) {
  const guide = explainError({ code, message });
  return (
    <div className="recovery" data-testid="recovery-notice">
      <strong className="recovery-message">{guide ? guide.title : 'Something went wrong'}</strong>
      <span className="recovery-detail">{message}{code ? <small> (code {code}{requestId ? `, request ${requestId}` : ''})</small> : null}</span>
      {guide && <>
        <span className="recovery-next"><b>Try this first:</b> {guide.steps[0]}</span>
        <details className="recovery-steps">
          <summary>What happened, and every step</summary>
          <p>{guide.cause}</p>
          <ol>{guide.steps.map(step => <li key={step}>{step}</li>)}</ol>
          {onGuide && <button type="button" className="link-button" onClick={onGuide}>Open the recovery guide</button>}
        </details>
      </>}
    </div>
  );
}
