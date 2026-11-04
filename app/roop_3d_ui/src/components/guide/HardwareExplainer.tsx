import type { Capabilities } from '../../contracts';
import { HARDWARE_PROFILES, PROVIDER_TERMS, type HardwareProfileId } from './helpContent';

const isProfile = (value: unknown): value is HardwareProfileId => value === 'auto' || value === 'cpu' || value === '6gb' || value === '12gb';

/**
 * What the active hardware profile means for this computer, in words: what Roop 3D will use, what that changes,
 * and how to read the CPU / TRT / VRAM readout. The profile names come from the server, the text from helpContent.
 */
export default function HardwareExplainer({ caps, configured }: { caps: Capabilities | null; configured?: string }) {
  const resolved = caps?.workstation.resolved_hardware_profile;
  const active = isProfile(resolved) ? resolved : isProfile(configured) ? configured : null;
  if (!active) return <p className="muted hardware-explainer" data-testid="hardware-explainer">The hardware profile is not known yet. It appears once the local service has probed this computer.</p>;
  const info = HARDWARE_PROFILES[active];
  const automatic = (configured ?? 'auto') === 'auto';
  return (
    <div className="hardware-explainer" data-testid="hardware-explainer">
      <p><strong>{info.label}</strong>{automatic && active !== 'auto' ? ' (chosen automatically)' : ''}. {info.summary}</p>
      <details><summary>What this changes</summary>
        <ul>{info.behaviour.map(item => <li key={item}>{item}</li>)}</ul>
        <p className="muted">Needs: {info.needs}</p>
        <p className="muted">Reading the hardware pill in the header:</p>
        <dl className="guide-dl">{PROVIDER_TERMS.map(item => <div key={item.term}><dt>{item.term}</dt><dd>{item.meaning}</dd></div>)}</dl>
      </details>
    </div>
  );
}
