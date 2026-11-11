import { Component, type ErrorInfo, type ReactNode } from 'react';

export interface BoundaryContext { error: Error; reset: () => void; attempt: number }

export interface WorkstationErrorBoundaryProps {
  /** Short name shown in the message and the log (`3D viewport`, `Workstation`). */
  name: string;
  children: ReactNode;
  /** Replaces the default message. Receives the error and a `reset` that remounts the children. */
  fallback?: (context: BoundaryContext) => ReactNode;
  /** When any of these values changes the boundary clears itself (a different FaceSet or mesh deserves a fresh try). */
  resetKeys?: readonly unknown[];
  onError?: (error: Error, info: ErrorInfo) => void;
  /** `app` renders a full-page recovery screen; `panel` a compact inline one. */
  variant?: 'app' | 'panel';
}

interface State { error: Error | null; attempt: number }

const sameKeys = (a: readonly unknown[] = [], b: readonly unknown[] = []): boolean => a.length === b.length && a.every((value, index) => Object.is(value, b[index]));

/**
 * One boundary for the whole workstation and for the 3D viewport. A render error (including the ones react-three-fiber
 * re-throws when a WebGL context cannot be created) never blanks the page: the user gets a message, a retry that remounts
 * the subtree, and, for the app variant, a reload. Errors are logged with `console.warn` so they stay visible in diagnostics.
 */
export default class WorkstationErrorBoundary extends Component<WorkstationErrorBoundaryProps, State> {
  state: State = { error: null, attempt: 0 };
  static getDerivedStateFromError(error: unknown): Partial<State> { return { error: error instanceof Error ? error : new Error(String(error)) }; }
  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.warn(`[workstation] ${this.props.name} failed`, error, info.componentStack);
    try { this.props.onError?.(error, info); } catch { /* a faulty reporter must not mask the original error */ }
  }
  componentDidUpdate(previous: WorkstationErrorBoundaryProps): void {
    if (this.state.error && !sameKeys(previous.resetKeys, this.props.resetKeys)) this.reset();
  }
  reset = (): void => this.setState(state => ({ error: null, attempt: state.attempt + 1 }));
  render(): ReactNode {
    const { error, attempt } = this.state;
    if (!error) return <div className="workstation-boundary" style={{ display: 'contents' }} key={attempt}>{this.props.children}</div>;
    if (this.props.fallback) return this.props.fallback({ error, reset: this.reset, attempt });
    const app = this.props.variant === 'app';
    return (
      <div className={`error-retry workstation-boundary-error ${app ? 'app' : 'panel-error'}`} role="alert" data-testid="workstation-error">
        <strong>{this.props.name} stopped unexpectedly.</strong>
        <span>{error.message || 'An unknown error occurred.'}</span>
        <span className="muted">Your project, queue and FaceSets are stored on disk and are not affected.</span>
        <span>
          <button type="button" className="button small" onClick={this.reset}>Try again</button>
          {app && <button type="button" className="button small accent" onClick={() => window.location.reload()}>Reload workstation</button>}
        </span>
      </div>
    );
  }
}
