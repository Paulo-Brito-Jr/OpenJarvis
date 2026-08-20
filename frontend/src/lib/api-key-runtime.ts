const SETTINGS_KEY = 'openjarvis-settings';

let runtimeApiKey = '';

function normalizeApiKey(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

export type LegacyApiKeyInspection =
  | { kind: 'none' }
  | {
      kind: 'found';
      apiKey: string;
      rawSnapshot: string;
      sanitizedSettings: Record<string, unknown>;
    }
  | { kind: 'error' };

/**
 * Inspect the legacy settings blob without mutating it. Startup must first
 * confirm that another safe copy exists (Keychain or process memory) before
 * committing removal of the plaintext field.
 */
export function inspectLegacyApiKeyStorage(): LegacyApiKeyInspection {
  if (typeof localStorage === 'undefined') return { kind: 'none' };

  let raw: string | null;
  try {
    raw = localStorage.getItem(SETTINGS_KEY);
  } catch {
    return { kind: 'error' };
  }
  if (!raw) return { kind: 'none' };

  try {
    const parsed = JSON.parse(raw) as unknown;
    if (
      !parsed
      || Array.isArray(parsed)
      || typeof parsed !== 'object'
    ) {
      return raw.includes('"apiKey"') ? { kind: 'error' } : { kind: 'none' };
    }

    const settings = parsed as Record<string, unknown>;
    if (!Object.prototype.hasOwnProperty.call(settings, 'apiKey')) {
      return { kind: 'none' };
    }

    const sanitizedSettings = { ...settings };
    const apiKey = normalizeApiKey(sanitizedSettings.apiKey);
    delete sanitizedSettings.apiKey;

    return {
      kind: 'found',
      apiKey,
      rawSnapshot: raw,
      sanitizedSettings,
    };
  } catch {
    return raw.includes('"apiKey"') ? { kind: 'error' } : { kind: 'none' };
  }
}

/**
 * Commit removal only after the caller has secured the value elsewhere.
 * Returning false keeps startup fail-closed and leaves the original blob
 * untouched when storage is unavailable.
 */
export function commitLegacyApiKeyMigration(
  inspection: Extract<LegacyApiKeyInspection, { kind: 'found' }>,
): boolean {
  if (typeof localStorage === 'undefined') return false;

  try {
    // Compare-and-swap: never overwrite settings or a replacement key written
    // by another WebView while Keychain IPC was in flight.
    if (localStorage.getItem(SETTINGS_KEY) !== inspection.rawSnapshot) {
      return false;
    }
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify(inspection.sanitizedSettings),
    );
    return true;
  } catch {
    return false;
  }
}

export function getRuntimeApiKey(): string {
  return runtimeApiKey;
}

export function setRuntimeApiKey(apiKey: string): void {
  runtimeApiKey = apiKey.trim();
}
