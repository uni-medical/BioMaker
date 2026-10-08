import { readResponse } from './response';

const uncertain = () => new Error('[provider_settings:uncertain] The save result could not be confirmed. Review the saved settings before saving again.');

/** A failed settings transport may happen after the atomic save has committed.
 * Do not retry it or expose its raw error. Explicit HTTP refusals retain their
 * existing codes so a rejected, uncommitted save can be corrected and retried. */
export async function saveProviderSettings(send: () => Promise<Response>): Promise<unknown> {
  let response: Response;
  try { response = await send(); }
  catch { throw uncertain(); }
  if (!response.ok) return readResponse(response, true);
  try {
    const value = await readResponse(response, true);
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw uncertain();
    const config = value as { settings_revision?: unknown; values?: unknown };
    if (typeof config.settings_revision !== 'string' || !/^[0-9a-f]{64}$/.test(config.settings_revision) ||
        !config.values || typeof config.values !== 'object' || Array.isArray(config.values)) throw uncertain();
    return value;
  } catch { throw uncertain(); }
}
