import { dismissToast, useToasts } from '../../stores/toasts';
import './toastHost.css';

/** Renders the app-wide toasts. Errors are `role="alert"` (announced at once); the rest are polite status messages. */
export default function ToastHost() {
  const items = useToasts();
  if (!items.length) return null;
  return (
    <div className="toast-host" data-testid="toast-host">
      {items.map(item => (
        <div key={item.id} className={`toast ${item.kind}`} role={item.kind === 'error' ? 'alert' : 'status'}>
          <div className="toast-body">
            {item.title && <strong>{item.title}</strong>}
            <span>{item.text}</span>
          </div>
          <button type="button" className="toast-close" aria-label="Dismiss message" onClick={() => dismissToast(item.id)}>×</button>
        </div>
      ))}
    </div>
  );
}
