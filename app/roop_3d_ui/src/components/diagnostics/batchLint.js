// Pre-flight linting for batch submission: one request per distinct target video, advisory unless a check FAILS.

export const SKIP_KEY = 'roop3d_skip_linter';

/**
 * The lint request for one job payload. Mappings are reduced to the string pairs the endpoint accepts.
 * @param {any} payload @param {Record<string, unknown>} [rackParameters] @param {string | null} [swapperId]
 */
export function lintRequestFor(payload, rackParameters = {}, swapperId = null) {
  const mappings = (payload.face_mappings ?? []).map(item => ({ track_id: String(item.track_id ?? ''), faceset_id: String(item.faceset_id ?? '') }));
  const parameters = { ...(payload.workstation_parameters ?? {}), ...rackParameters };
  return {
    target_path: payload.target_path, inspection_id: payload.inspection_id ?? null, face_mappings: mappings,
    workstation_parameters: Object.keys(parameters).length ? parameters : null,
    swapper_model_id: swapperId || payload.swapper_model_id || null,
    export_passes: payload.export?.passes ?? [],
  };
}

const name = path => String(path).split(/[\\/]/).pop();

/**
 * Lint every distinct target once. Returns `failures` (block), `warnings` (proceed, tell the user) and `unavailable`
 * (the linter itself could not be reached: never a reason to refuse the batch).
 * @param {(url: string, body: any) => Promise<any>} post
 * @param {any[]} payloads
 * @param {{ rackParameters?: Record<string, unknown>, swapperId?: string | null }} [options]
 */
export async function lintBatch(post, payloads, { rackParameters = {}, swapperId = null } = {}) {
  const seen = new Set(), failures = [], warnings = [];
  let unavailable = false;
  for (const payload of payloads) {
    if (!payload?.target_path || seen.has(payload.target_path)) continue;
    seen.add(payload.target_path);
    let report;
    try {
      report = await post('/system/preflight-lint', lintRequestFor(payload, rackParameters, swapperId));
    } catch {
      unavailable = true;
      continue;
    }
    for (const check of report?.checks ?? []) {
      if (check.status === 'FAIL') failures.push({ target: name(payload.target_path), id: check.id, name: check.name, message: check.message });
      else if (check.status === 'WARN') warnings.push({ target: name(payload.target_path), id: check.id, name: check.name, message: check.message });
    }
  }
  return { targets: seen.size, failures, warnings, unavailable };
}

/** One sentence per outcome for the batch status line. */
export function describeLint(result) {
  const list = items => items.map(item => `${item.name} (${item.target})`).join('; ');
  return {
    block: result.failures.length ? `Pre-flight blocked the batch: ${list(result.failures)}. ${result.failures[0].message}` : '',
    note: [
      result.warnings.length ? `${result.warnings.length} pre-flight warning(s): ${list(result.warnings)}.` : '',
      result.unavailable ? 'Pre-flight lint was unavailable for some targets.' : '',
    ].filter(Boolean).join(' '),
  };
}
