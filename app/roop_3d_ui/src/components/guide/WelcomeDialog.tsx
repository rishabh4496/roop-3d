import type { Capabilities } from '../../contracts';
import DialogFrame from './DialogFrame';

/**
 * Shown once, on the first launch, over the Home screen. It tells a new person in four sentences what the app is,
 * checks the two things that stop a first render (models, GPU), and offers three clear ways forward.
 */
export default function WelcomeDialog({ caps, onGuide, onTour, onStart, onClose }: { caps: Capabilities | null; onGuide: () => void; onTour: () => void; onStart: () => void; onClose: () => void }) {
  const verified = caps ? caps.models.filter(model => model.state === 'VERIFIED_LOCAL').length : null;
  const profile = caps?.workstation.resolved_hardware_profile;
  return (
    <DialogFrame title="Welcome to Roop 3D" eyebrow="FIRST RUN" onClose={onClose} className="welcome-dialog" testId="welcome-dialog" describedBy="welcome-lead" initialFocus="[data-welcome-primary]">
      <p id="welcome-lead" className="welcome-lead">Roop 3D replaces the faces in a video you already have, using photos of the person you want to put in. Everything runs on this computer: your videos, photos and results are never uploaded.</p>
      <ol className="welcome-steps" aria-label="The workflow in short">
        <li><strong>Import</strong> your video.</li><li><strong>Add</strong> photos of the new face (a FaceSet).</li><li><strong>Inspect</strong> the video and choose who becomes whom.</li><li><strong>Check</strong> the live preview, then <strong>render</strong>.</li>
      </ol>
      <div className="welcome-status" role="group" aria-label="This computer">
        <span className={`status-chip ${verified ? 'ok' : ''}`}>{caps === null ? 'Checking this computer…' : verified ? `${verified} model file${verified === 1 ? '' : 's'} verified` : 'No model files verified yet'}</span>
        {profile && <span className="status-chip ok">Hardware profile: {profile}</span>}
        {caps !== null && !verified && <p className="muted">Roop 3D works offline, so its model files must be on disk. Open Settings → Environment &amp; Offline Manager and press Verify Offline Health to see what is missing, or read docs/OFFLINE_MODELS.md.</p>}
      </div>
      <div className="welcome-actions">
        <button type="button" className="button accent" data-welcome-primary onClick={onStart}>Start a guided project</button>
        <button type="button" className="button" onClick={onTour}>Take the quick tour</button>
        <button type="button" className="button" onClick={onGuide}>Open the full guide</button>
        <button type="button" className="button" onClick={onClose}>Skip for now</button>
      </div>
      <p className="muted">You can reopen all of this from <b>Guide</b> in the header at any time.</p>
    </DialogFrame>
  );
}
