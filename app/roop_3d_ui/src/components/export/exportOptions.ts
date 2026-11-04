import { useCallback, useEffect, useState } from 'react';
import { api } from '../../api';
import type { ExportOptions } from '../../contracts';

export interface ExportProfileInfo {
  id: string; label: string; description: string; container: string; encoder: string; codec_family: string;
  pix_fmt: string | null; container_bit_depth: number; lossless: boolean; rgb_native: boolean; hdr_passthrough: boolean;
  audio_default: string; limits: string[];
  verification?: { state: string; reason?: string };
}

/**
 * Drop every Stage 15 field that equals the server default, so "nothing chosen" serialises exactly like the
 * original `DEFAULT_EXPORT` whether the options came from the picker or from a project the server echoed back
 * with its defaults filled in. Keeps the UNSAVED badge honest.
 */
export function normalizeExport(value: ExportOptions): ExportOptions {
  const { profile, hdr_policy, subtitles, sequence_format, ...base } = value;
  const next: ExportOptions = { ...base };
  if (profile && profile !== 'standard_mp4') next.profile = profile;
  if (hdr_policy && hdr_policy !== 'tonemap_sdr') next.hdr_policy = hdr_policy;
  if (subtitles && subtitles !== 'copy') next.subtitles = subtitles;
  if (profile === 'image_sequence' && sequence_format && sequence_format !== 'png') next.sequence_format = sequence_format;
  return next;
}

/** The server's presets (what each can carry, from `media/export_profiles.py`). Loaded once; failure is a visible, retryable state. */
export function useExportProfiles() {
  const [state, setState] = useState<{ profiles: ExportProfileInfo[] | null; error: string; attempt: number }>({ profiles: null, error: '', attempt: 0 });
  useEffect(() => {
    const controller = new AbortController();
    api<{ profiles: ExportProfileInfo[] }>('/media/export-profiles', { signal: controller.signal })
      .then(body => setState(value => (Array.isArray(body?.profiles) ? { ...value, profiles: body.profiles, error: '' } : { ...value, error: 'unexpected response from the local service' })))
      .catch((error: Error) => { if (error.name !== 'AbortError') setState(value => ({ ...value, profiles: null, error: error.message })); });
    return () => controller.abort();
  }, [state.attempt]);
  const retry = useCallback(() => setState(value => ({ ...value, error: '', attempt: value.attempt + 1 })), []);
  return { profiles: state.profiles, error: state.error, retry };
}
