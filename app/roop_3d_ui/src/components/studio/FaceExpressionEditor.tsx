import { useState } from 'react';
import type { FaceEditor, WorkstationParameters } from '../workstation/pipeline';
import { ARKIT52_GROUPS, arkitLabel } from './arkit52';
import { PrecisionRangeSlider } from '../ui-layouts';

type FaceEditKey = Exclude<keyof FaceEditor, 'arkit_blendshapes'>;

export const FACE_EDIT_CONTROLS: { key: FaceEditKey; label: string; min: number; max: number; step: number }[] = [
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
  const [arkitOpen, setArkitOpen] = useState(false);
  const edits = parameters.face_expression_overrides ?? {};
  const edit = trackId ? edits[trackId] ?? {} : {};
  // The two blink channels are a left/right pair: they share one symmetric slider that can be split (see PrecisionRangeSlider `sides`).
  const [blinkLinked, setBlinkLinked] = useState(true);
  const updateMany = (patch: Partial<Record<FaceEditKey, number>>) => {
    if (!trackId) return;
    const next: Record<string, unknown> = { ...edit, ...patch };
    for (const key of Object.keys(patch)) if (patch[key as FaceEditKey] === 0) delete next[key];
    const all = { ...edits };
    if (Object.keys(next).length) all[trackId] = next as typeof edit; else delete all[trackId];
    setParameters({ ...parameters, face_expression_overrides: all });
  };
  const update = (key: FaceEditKey, value: number) => updateMany({ [key]: value });
  const updateArkit = (name: string, value: number) => {
    if (!trackId) return;
    const arkit = { ...(edit.arkit_blendshapes ?? {}) };
    if (value === 0) delete arkit[name]; else arkit[name] = value;
    const next = { ...edit };
    if (Object.keys(arkit).length) next.arkit_blendshapes = arkit;
    else delete next.arkit_blendshapes;
    const all = { ...edits };
    if (Object.keys(next).length) all[trackId] = next; else delete all[trackId];
    setParameters({ ...parameters, face_expression_overrides: all });
  };
  return <details className="inspector-subsection" open>
    <summary>Micro-expression &amp; 3D Face Editor<small>{trackId ?? 'Select a target face'}</small></summary>
    <PrecisionRangeSlider className="inspector-field" label="Expression restore factor" min={0} max={1} step={.01} value={parameters.expression_restore_factor ?? 0}
      onChange={value => setParameters({ ...parameters, expression_restore_factor: value })} />
    <small className="muted">Restores target expression before UltraMax. Blinks below EAR 0.16 close fully. Requires local LivePortrait, eye and 68-point landmark models.</small>
    <fieldset disabled={!trackId} aria-label="Per-face 3D expression and pose">
      {FACE_EDIT_CONTROLS.map(control => {
        if (control.key === 'blink_right') return null; // rendered together with blink_left
        const unit = control.step === 1 ? '°' : undefined; // the head offsets are whole degrees; the rest are unitless 0-1 / -1..1 weights
        if (control.key === 'blink_left') {
          const sides = { left: edit.blink_left ?? 0, right: edit.blink_right ?? 0 };
          return <PrecisionRangeSlider key="blink" className="inspector-field" label="Eye Blink" min={control.min} max={control.max} step={control.step} disabled={!trackId}
            sides={sides} linked={blinkLinked && sides.left === sides.right} onLinkedChange={setBlinkLinked}
            onSidesChange={next => updateMany({ blink_left: next.left, blink_right: next.right })} />;
        }
        return <PrecisionRangeSlider key={control.key} className="inspector-field" label={control.label} min={control.min} max={control.max} step={control.step} unit={unit}
          disabled={!trackId} value={edit[control.key] ?? 0} onChange={value => update(control.key, value)} />;
      })}
      <details className="arkit-slider-panel" onToggle={event => setArkitOpen((event.currentTarget as HTMLDetailsElement).open)}>
        <summary>ARKit 52 expression sliders<small>Standard blendshape controls · 0–1</small></summary>
        {arkitOpen && <div className="arkit-slider-list">
          {ARKIT52_GROUPS.map(group => <section key={group.label} aria-label={`ARKit ${group.label}`}>
            <h4>{group.label}</h4>
            {group.names.map(name => <label key={name} className="inspector-field">{arkitLabel(name)}<output>{(edit.arkit_blendshapes?.[name] ?? 0).toFixed(2)}</output>
              <input type="range" aria-label={`ARKit ${arkitLabel(name)}`} min={0} max={1} step={.01}
                value={edit.arkit_blendshapes?.[name] ?? 0} onChange={e => updateArkit(name, Number(e.target.value))} /></label>)}
          </section>)}
        </div>}
        <small className="muted">Standard controls feed the local expression rig. Video tracking can map them to the 3DMM basis, jaw and gaze; paired local samples can calibrate that bridge.</small>
      </details>
      <button type="button" className="button small" disabled={!Object.keys(edit).length} onClick={() => {
        if (!trackId) return;
        const next = { ...edits }; delete next[trackId]; setParameters({ ...parameters, face_expression_overrides: next });
      }}>Reset selected face</button>
    </fieldset>
  </details>;
}
