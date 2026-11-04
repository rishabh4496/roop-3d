import { useId } from 'react';
import type { ExportOptions } from '../../contracts';
import { PRESET_GUIDANCE, PRESET_LIMITS } from '../guide/helpContent';
import { normalizeExport, useExportProfiles } from './exportOptions';

/**
 * "Export preset" selector with a plain-language explanation of the chosen preset. The encoder facts come from the
 * server (`GET /media/export-profiles`), the "use it for" advice from helpContent. Choosing the standard preset
 * clears the field, so a project that never touched it serialises exactly as before.
 */
export default function ExportPresetBar({ value, onChange, disabled = false }: { value: ExportOptions; onChange: (value: ExportOptions) => void; disabled?: boolean }) {
  const { profiles, error, retry } = useExportProfiles();
  const id = useId();
  const current = value.profile ?? 'standard_mp4';
  const info = profiles?.find(item => item.id === current);
  const guidance = PRESET_GUIDANCE[current];
  const locked = disabled || !value.baked_video;
  const change = (patch: Partial<ExportOptions>) => onChange(normalizeExport({ ...value, ...patch }));
  return (
    <section className="export-preset" aria-label="Export options" data-testid="export-preset">
      <div className="export-preset-row">
        <label htmlFor={`${id}-preset`}><span className="eyebrow">EXPORT PRESET</span></label>
        <select id={`${id}-preset`} aria-label="Export preset" aria-describedby={`${id}-help`} value={current} disabled={locked || !profiles}
          onChange={event => change({ profile: event.target.value as ExportOptions['profile'] })}>
          {(profiles ?? [{ id: 'standard_mp4', label: 'Standard MP4' }]).map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
        {guidance && <span className="status-chip" title="Typical output file size relative to the other presets">{guidance.size} files</span>}
      </div>
      <div id={`${id}-help`} className="export-preset-help muted">
        {!value.baked_video ? 'Presets apply to the baked video. Turn on “Baked video” under Editor → VFX compositing exports to use one.'
          : error ? <span role="status">Presets could not be loaded: {error} <button type="button" className="link-button" onClick={retry}>Retry</button>. The standard MP4 is used.</span>
          : !profiles ? 'Loading presets…'
          : <>
            <span>{info?.description}</span>
            {guidance && <span> <b>Use it for:</b> {guidance.when}</span>}
            <span> {info ? `${info.container === 'sequence' ? 'Image sequence' : info.container} · ${info.container_bit_depth}-bit${info.lossless ? ' · lossless' : ''}.` : ''}</span>
          </>}
      </div>
      <details className="export-preset-more">
        <summary>More export options</summary>
        <div className="export-preset-grid">
          <label>If the video is HDR
            <select aria-label="HDR policy" disabled={locked} value={value.hdr_policy ?? 'tonemap_sdr'} onChange={event => change({ hdr_policy: event.target.value as ExportOptions['hdr_policy'] })}>
              <option value="tonemap_sdr">Convert to SDR (default)</option><option value="reject">Stop with an error</option>
            </select></label>
          <label>Subtitles in the source
            <select aria-label="Subtitle policy" disabled={locked} value={value.subtitles ?? 'copy'} onChange={event => change({ subtitles: event.target.value as ExportOptions['subtitles'] })}>
              <option value="copy">Copy them if the container allows (default)</option><option value="require">Require them (fail if they cannot be kept)</option><option value="drop">Drop them</option>
            </select></label>
          {current === 'image_sequence' && <label>Image format
            <select aria-label="Image sequence format" disabled={locked} value={value.sequence_format ?? 'png'} onChange={event => change({ sequence_format: event.target.value as ExportOptions['sequence_format'] })}>
              <option value="png">PNG</option><option value="tiff">TIFF</option><option value="dpx">DPX</option>
            </select></label>}
        </div>
        <p className="muted">{PRESET_LIMITS}</p>
      </details>
    </section>
  );
}
