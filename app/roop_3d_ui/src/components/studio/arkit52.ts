/** Public ARKit blendshape order; values remain sparse in per-face overrides. */
export const ARKIT52_NAMES = [
  'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
  'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
  'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
  'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
  'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
  'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight',
  'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
  'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight', 'tongueOut',
] as const;

export const arkitLabel = (name: string) => name.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/^./, first => first.toUpperCase());

export const ARKIT52_GROUPS = [
  { label: 'Brows', names: ARKIT52_NAMES.filter(name => name.startsWith('brow')) },
  { label: 'Eyes and gaze', names: ARKIT52_NAMES.filter(name => name.startsWith('eye')) },
  { label: 'Jaw and cheeks', names: ARKIT52_NAMES.filter(name => name.startsWith('jaw') || name.startsWith('cheek')) },
  { label: 'Mouth', names: ARKIT52_NAMES.filter(name => name.startsWith('mouth')) },
  { label: 'Nose and tongue', names: ARKIT52_NAMES.filter(name => name.startsWith('nose') || name === 'tongueOut') },
];
