import { Component, type ErrorInfo, type ReactNode } from 'react';

/**
 * Keeps one studio panel's failure local: the panel shows the error and a Retry button (which remounts it), while the
 * preview, timeline and the other panels keep working. Nothing is retried automatically.
 */
export default class PanelBoundary extends Component<{ name: string; children: ReactNode }, { error: Error | null; attempt: number }> {
  state = { error: null as Error | null, attempt: 0 };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.warn(`[studio] ${this.props.name} panel failed`, error, info.componentStack); }
  render() {
    if (this.state.error) {
      return <div className="error-retry panel-error" role="alert"><span>{this.props.name} could not be displayed: {this.state.error.message}</span>
        <button type="button" className="button small" onClick={() => this.setState(state => ({ error: null, attempt: state.attempt + 1 }))}>Retry</button></div>;
    }
    return <div className="panel-boundary" key={this.state.attempt}>{this.props.children}</div>;
  }
}
