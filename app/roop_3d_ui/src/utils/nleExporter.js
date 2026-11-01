/** Frame ranges are half-open; URLs point to local media, never API streaming URLs. */
const integer = (n, name) => { if (!Number.isSafeInteger(n) || n < 0) throw new Error(`Invalid ${name}`); return n; };
const xml = value => String(value).replace(/[<>&"']/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' }[c]));
// Intentionally matches control characters: it strips them from values written into line-oriented NLE files.
// eslint-disable-next-line no-control-regex
const line = value => String(value).replace(/[\r\n\x00-\x1f]/g, ' ');
export function frameRate(value) {
  const parts = String(value).split('/');
  if (parts.length > 2 || parts.some(p => !/^\d+$/.test(p))) throw new Error('Supply a rational frame rate');
  const numerator = Number(parts[0]), denominator = Number(parts[1] ?? 1);
  if (!Number.isSafeInteger(numerator) || !Number.isSafeInteger(denominator) || numerator <= 0 || denominator <= 0 || numerator / denominator > 240) throw new Error('Invalid frame rate');
  return { numerator, denominator, rate: numerator / denominator };
}
export function buildTimeline(plan) {
  const fps = frameRate(plan.frameRate), total = integer(plan.frameCount, 'frame count');
  if (!total || !plan.sourceUrl) throw new Error('Source and frame count are required');
  if (plan.vfr !== false) throw new Error('Frame-accurate NLE export requires verified CFR media; conform VFR first');
  const mediaOffset = integer(plan.mediaOffset ?? 0, 'media offset');
  const ranges = (plan.ranges ?? []).map(([a, b]) => [integer(a, 'range start'), integer(b, 'range end')]);
  if (ranges.some(([a, b]) => b <= a || b > total || a < mediaOffset)) throw new Error('Swapped ranges exceed media bounds');
  const cuts = [...new Set([0, total, ...(plan.sceneCuts ?? []).map(n => integer(n, 'scene cut')), ...ranges.flat()])].sort((a, b) => a - b);
  if (cuts.some(n => n > total)) throw new Error('Scene cut exceeds media length');
  const tracks = ['Original source', 'Swapped video', 'Face alpha matte'].map((name, i) => ({ name: `V${i + 1} ${name}`, clips: [] }));
  for (let i = 0; i < cuts.length - 1; i++) {
    const start = cuts[i], end = cuts[i + 1];
    tracks[0].clips.push({ start, end, mediaIn: start, url: plan.sourceUrl, name: 'Source' });
    if (ranges.some(([a, b]) => a <= start && end <= b)) {
      if (plan.swappedUrl) tracks[1].clips.push({ start, end, mediaIn: start - mediaOffset, url: plan.swappedUrl, name: 'Swap' });
      if (plan.alphaUrl) tracks[2].clips.push({ start, end, mediaIn: start - mediaOffset, url: plan.alphaUrl, name: 'Alpha matte' });
    }
  }
  return { ...plan, fps, total, tracks, name: plan.name || 'Roop 3D composite' };
}
export function exportFCPXML(plan) {
  const t = buildTimeline(plan);
  const seconds = frames => `${BigInt(frames) * BigInt(t.fps.denominator)}/${t.fps.numerator}s`;
  const urls = [...new Set(t.tracks.flatMap(track => track.clips.map(clip => clip.url)))];
  const ref = url => `r${urls.indexOf(url) + 2}`;
  const resources = `<format id="r1" frameDuration="${t.fps.denominator}/${t.fps.numerator}s" width="${integer(plan.width ?? 1920, 'width')}" height="${integer(plan.height ?? 1080, 'height')}"/>` + urls.map(url => `<asset id="${ref(url)}" name="${xml(url.split('/').at(-1))}" start="0s" duration="${seconds(t.total)}" hasVideo="1" format="r1"><media-rep kind="original-media" src="${xml(url)}"/></asset>`).join('');
  // Connected clips anchor to the source clip at the same boundary. Their offset
  // is in the parent's source time coordinate, preserving trims and exact frames.
  const spine = t.tracks[0].clips.map(source => {
    const connected = t.tracks.slice(1).flatMap((track, index) => track.clips.filter(c => c.start === source.start).map(c => `<asset-clip name="${xml(c.name)}" ref="${ref(c.url)}" lane="${index + 1}" offset="${seconds(c.start)}" start="${seconds(c.mediaIn)}" duration="${seconds(c.end - c.start)}"/>`)).join('');
    return `<asset-clip name="Source" ref="${ref(source.url)}" offset="${seconds(source.start)}" start="${seconds(source.mediaIn)}" duration="${seconds(source.end - source.start)}">${connected}</asset-clip>`;
  }).join('');
  return `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n<fcpxml version="1.10"><resources>${resources}</resources><library><event name="Roop 3D"><project name="${xml(t.name)}"><sequence format="r1" duration="${seconds(t.total)}" tcStart="0s" tcFormat="NDF"><spine>${spine}</spine></sequence></project></event></library></fcpxml>\n`;
}
export function exportOTIO(plan) {
  const t = buildTimeline(plan);
  const rt = value => ({ OTIO_SCHEMA: 'RationalTime.1', value, rate: t.fps.rate });
  const range = (start, duration) => ({ OTIO_SCHEMA: 'TimeRange.1', start_time: rt(start), duration: rt(duration) });
  const children = t.tracks.map(track => {
    let cursor = 0; const clips = [];
    for (const clip of track.clips) {
      if (clip.start > cursor) clips.push({ OTIO_SCHEMA: 'Gap.1', source_range: range(0, clip.start - cursor) });
      clips.push({ OTIO_SCHEMA: 'Clip.2', name: clip.name, source_range: range(clip.mediaIn, clip.end - clip.start),
        media_references: { DEFAULT_MEDIA: { OTIO_SCHEMA: 'ExternalReference.1', target_url: clip.url } }, active_media_reference_key: 'DEFAULT_MEDIA',
        metadata: { roop3d: { frame_rate: `${t.fps.numerator}/${t.fps.denominator}`, role: track.name } } });
      cursor = clip.end;
    }
    if (cursor < t.total) clips.push({ OTIO_SCHEMA: 'Gap.1', source_range: range(0, t.total - cursor) });
    return { OTIO_SCHEMA: 'Track.1', name: track.name, kind: 'Video', children: clips };
  });
  return JSON.stringify({ OTIO_SCHEMA: 'Timeline.1', name: t.name, global_start_time: rt(0), metadata: { roop3d: { alpha_compositing: 'Set V3 as V2 track matte in the receiving NLE' } }, tracks: { OTIO_SCHEMA: 'Stack.1', children } }, null, 2) + '\n';
}
export function frameTimecode(frame, rate, dropFrame = false) {
  integer(frame, 'timecode frame');
  const fps = frameRate(rate), nominal = Math.round(fps.rate);
  if (dropFrame) {
    if (!([30000, 60000].includes(fps.numerator) && fps.denominator === 1001)) throw new Error('Drop frame supports 30000/1001 or 60000/1001');
    const drop = nominal === 30 ? 2 : 4, per10 = nominal * 600 - drop * 9, perMinute = nominal * 60 - drop;
    const ten = Math.floor(frame / per10), remainder = frame % per10;
    frame += drop * 9 * ten + (remainder >= drop ? drop * Math.floor((remainder - drop) / perMinute) : 0);
  }
  const h = Math.floor(frame / (nominal * 3600));
  if (h >= 24) throw new Error('CMX timecode exceeds 24 hours');
  return [h, Math.floor(frame / (nominal * 60)) % 60, Math.floor(frame / nominal) % 60, frame % nominal].map(n => String(n).padStart(2, '0')).join(':');
}
/** CMX 3600 is single-track: return a file per track rather than flattening layers. */
export function exportEDL(plan, { dropFrame = false } = {}) {
  const t = buildTimeline(plan), nominal = Math.round(t.fps.rate);
  if (![24, 25, 30, 50, 60].includes(nominal) || !(Number.isInteger(t.fps.rate) || [24000/1001,30000/1001,60000/1001].includes(t.fps.rate))) throw new Error('Unsupported CMX edit rate');
  const time = frame => frameTimecode(frame, plan.frameRate, dropFrame);
  return Object.fromEntries(t.tracks.map((track, trackIndex) => {
    if (track.clips.length > 999) throw new Error('CMX supports at most 999 events per track');
    const events = track.clips.map((c, i) => `${String(i + 1).padStart(3, '0')}  R${String(trackIndex + 1).padStart(7, '0')} V     C        ${time(c.mediaIn)} ${time(c.mediaIn + c.end - c.start)} ${time(c.start)} ${time(c.end)}\n* FROM CLIP NAME: ${line(c.url)}\n* ${line(c.name)}`).join('\n\n');
    return [`V${trackIndex + 1}.edl`, `TITLE: ${line(t.name)} ${track.name}\nFCM: ${dropFrame ? 'DROP FRAME' : 'NON-DROP FRAME'}\n* FRAME RATE: ${t.fps.numerator}/${t.fps.denominator}\n\n${events}\n`];
  }));
}
export function downloadText(filename, text, mime = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
