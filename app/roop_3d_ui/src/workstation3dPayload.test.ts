import { describe, expect, it } from 'vitest';
import { buildLivePayload } from './livePreview';
import { DEFAULT_PARAMETERS } from './components/workstation/pipeline';
import type { JobRequest } from './contracts';

const REQUEST = { target_path: 'local.mp4', inspection_id: 'a'.repeat(32), face_mappings: [{ track_id: 'track_0000', faceset_id: 'alice' }], processing: 'local', remote_consent: false, trim_start_seconds: null, trim_end_seconds: null } as unknown as JobRequest;
const build = (parameters = DEFAULT_PARAMETERS) => buildLivePayload({ request: REQUEST, time: 1, fps: 25, parameters, maskBase64: null, restorerAvailable: false, solo: null })!;

describe('opt-in 3D workstation controls in the live preview payload', () => {
  it('omits the hairline offset and ambient fill until they are touched, so earlier previews keep their cache keys', () => {
    const payload = JSON.parse(JSON.stringify(build()));
    expect('hairline_offset_px' in payload).toBe(false);
    expect('ambient_fill' in payload).toBe(false);
    expect('hairline_offset_px' in JSON.parse(JSON.stringify(build({ ...DEFAULT_PARAMETERS, hairline_offset_px: 0 })))).toBe(false);   // 0 is "not set"
  });

  it('sends the hairline offset once set, and the ambient fill only while relighting', () => {
    expect(build({ ...DEFAULT_PARAMETERS, hairline_offset_px: -12 }).hairline_offset_px).toBe(-12);
    expect(build({ ...DEFAULT_PARAMETERS, ambient_fill: 0.6 }).ambient_fill).toBeUndefined();                                      // relighting is off
    const relit = build({ ...DEFAULT_PARAMETERS, relight: true, lighting_mode: 'manual', ambient_fill: 0.6 });
    expect(relit).toMatchObject({ relight: true, lighting_mode: 'manual', ambient_fill: 0.6 });
  });
});
