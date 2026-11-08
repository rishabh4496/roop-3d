import type { FaceEditor, WorkstationParameters } from '../workstation/pipeline';

export const FACE_EDIT_CONTROLS: { key: keyof FaceEditor; label: string; min: number; max: number; step: number }[] = [
  { key: 'head_pitch', label: 'Head Pitch Offset', min: -30, max: 30, step: 1 },
  { key: 'head_yaw', label: 'Head Yaw Offset', min: -45, max: 45, step: 1 },
  { key: 'head_roll', label: 'Head Roll Offset', min: -30, max: 30, step: 1 },
  { key: 'gaze_x', label: 'Eyeball Gaze X', min: -1, max: 1, step: .01 },
  { key: 'gaze_y', label: 'Eyeball Gaze Y', min: -1, max: 1, step: .01 },
  { key: 'blink_left', label: 'Eye Blink Left', min: 0, max: 1, step: .01 },
  { key: 'blink_right', label: 'Eye Blink Right', min: 0, max: 1, step: .01 },
  { key: 'eyebrow_tilt', label: 'Eyebrow Tilt', min: -1, max: 1, step: .01 },
  { key: 'eyebrow_raise', label: 'Eyebrow Raise', min: -1, max: 1, step: .01 },
  { key: 'mouth_smile', label: 'Mouth Smile', min: -1, max: 1, step: .01 },
  { key: 'mouth_pout', label: 'Mouth Pout', min: 0, max: 1, step: .01 },
  { key: 'mouth_open', label: 'Mouth Open Ratio', min: 0, max: 1, step: .01 },
  { key: 'jaw_width', label: 'Jaw Width 3D Morph', min: -1, max: 1, step: .01 },
  { key: 'chin_length', label: 'Chin Length 3D Morph', min: -1, max: 1, step: .01 },
];

export default function FaceExpressionEditor({ parameters, setParameters, trackId }: {
  parameters: WorkstationParameters; setParameters: (value: WorkstationParameters) => void; trackId: string | null;
}) {
  const edits = parameters.face_expression_overrides ?? {};
  const edit = trackId ? edits[trackId] ?? {} : {};
  const update = (key: keyof FaceEditor, value: number) => {
    if (!trackId) return;
    const next = { ...edit, [key]: value };
    if (value === 0) delete next[key];
    const all = { ...edits };
    if (Object.keys(next).length) all[trackId] = next; else delete all[trackId];
    setParameters({ ...parameters, face_expression_overrides: all });
  };
  return <details className="inspector-subsection" open>
    <summary>Micro-expression &amp; 3D Face Editor<small>{trackId ?? 'Select a target face'}</small></summary>
    <label className="inspector-field">Expression restore factor <output>{(parameters.expression_restore_factor ?? 0).toFixed(2)}</output>
      <input type="range" aria-label="Expression restore factor" min={0} max={1} step={.01} value={parameters.expression_restore_factor ?? 0}
        onChange={e => setParameters({ ...parameters, expression_restore_factor: Number(e.target.value) })} /></label>
    <small className="muted">Restores target expression before UltraMax. Blinks below EAR 0.16 close fully. Requires local LivePortrait, eye and 68-point landmark models.</small>
    <fieldset disabled={!trackId} aria-label="Per-face 3D expression and pose">
      {FACE_EDIT_CONTROLS.map(control => <label key={control.key} className="inspector-field">{control.label}<output>{(edit[control.key] ?? 0).toFixed(control.step === 1 ? 0 : 2)}</output>
        <input type="range" aria-label={control.label} min={control.min} max={control.max} step={control.step} value={edit[control.key] ?? 0}
          onChange={e => update(control.key, Number(e.target.value))} /></label>)}
      <button type="button" className="button small" disabled={!Object.keys(edit).length} onClick={() => {
        if (!trackId) return;
        const next = { ...edits }; delete next[trackId]; setParameters({ ...parameters, face_expression_overrides: next });
      }}>Reset selected face</button>
    </fieldset>
  </details>;
}
