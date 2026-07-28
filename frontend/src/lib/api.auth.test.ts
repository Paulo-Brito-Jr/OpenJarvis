import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { invokeMock } = vi.hoisted(() => ({
  invokeMock: vi.fn(),
}));

vi.mock('@tauri-apps/api/core', () => ({
  invoke: invokeMock,
}));

// Regression for #266: the frontend must send the local API key as a Bearer
// token on /v1 + /api requests, or `jarvis serve` with a key configured 401s
// every data-plane call. These tests cover the ephemeral runtime source,
// legacy migration, and the headers used by regular and streaming fetches.

const SETTINGS_KEY = 'openjarvis-settings';

// Minimal in-memory localStorage stub so the helpers can run under node
// (no jsdom dependency).
class MemoryStorage {
  private store = new Map<string, string>();
  getItem(k: string): string | null {
    return this.store.has(k) ? (this.store.get(k) as string) : null;
  }
  setItem(k: string, v: string): void {
    this.store.set(k, String(v));
  }
  removeItem(k: string): void {
    this.store.delete(k);
  }
  clear(): void {
    this.store.clear();
  }
}

beforeEach(() => {
  vi.resetModules();
  invokeMock.mockReset();
  vi.stubEnv('VITE_SUPABASE_ANON_KEY', 'test-anon-key');
  (globalThis as unknown as { localStorage: MemoryStorage }).localStorage =
    new MemoryStorage();
  delete (globalThis as unknown as { window?: unknown }).window;
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  (globalThis as unknown as { localStorage?: MemoryStorage }).localStorage =
    undefined;
  delete (globalThis as unknown as { window?: unknown }).window;
});

async function freshApi() {
  // Re-import to pick up the current localStorage stub.
  return await import('./api');
}

function enableTauri() {
  (globalThis as unknown as {
    window: { __TAURI_INTERNALS__: Record<string, never> };
  }).window = { __TAURI_INTERNALS__: {} };
}

describe('getApiKey', () => {
  it('returns empty string when no key is configured', async () => {
    const { getApiKey } = await freshApi();
    expect(getApiKey()).toBe('');
  });

  it('does not read apiKey directly from localStorage', async () => {
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ apiUrl: 'http://x', apiKey: 'sk-local-123' }),
    );
    const { getApiKey } = await freshApi();
    expect(getApiKey()).toBe('');
  });

  it('migrates a legacy key to ephemeral memory and erases the persisted field', async () => {
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ apiUrl: 'http://x', apiKey: 'sk-local-123' }),
    );
    const { getApiKey, initApiKey } = await freshApi();

    const result = await initApiKey();

    expect(result).toEqual({ ok: true, migrated: true });
    expect(getApiKey()).toBe('sk-local-123');
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).toEqual({
      apiUrl: 'http://x',
    });
  });

  it('preserves the only legacy copy when Keychain cannot be read', async () => {
    enableTauri();
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ apiUrl: 'http://x', apiKey: 'legacy-only-secret' }),
    );
    invokeMock.mockRejectedValueOnce(new Error('keychain unavailable'));

    const { getApiKey, initApiKey } = await freshApi();
    const result = await initApiKey();

    expect(result).toMatchObject({
      ok: false,
      code: 'keychain_unavailable',
    });
    expect(getApiKey()).toBe('');
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).toEqual({
      apiUrl: 'http://x',
      apiKey: 'legacy-only-secret',
    });
  });

  it('preserves the only legacy copy when Keychain migration cannot write', async () => {
    enableTauri();
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ apiUrl: 'http://x', apiKey: 'legacy-only-secret' }),
    );
    invokeMock
      .mockResolvedValueOnce(null)
      .mockRejectedValueOnce(new Error('keychain write failed'));

    const { getApiKey, initApiKey } = await freshApi();
    const result = await initApiKey();

    expect(result).toMatchObject({
      ok: false,
      code: 'keychain_write_failed',
    });
    expect(getApiKey()).toBe('');
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).toEqual({
      apiUrl: 'http://x',
      apiKey: 'legacy-only-secret',
    });
    expect(invokeMock.mock.calls.map(([command]) => command)).toEqual([
      'get_local_api_key',
      'save_cloud_key',
    ]);
  });

  it('blocks a conflicting legacy value without deleting either copy', async () => {
    enableTauri();
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ apiUrl: 'http://x', apiKey: 'legacy-secret' }),
    );
    invokeMock.mockResolvedValueOnce('keychain-secret');

    const { getApiKey, initApiKey } = await freshApi();
    const result = await initApiKey();

    expect(result).toMatchObject({
      ok: false,
      code: 'legacy_conflict',
    });
    expect(getApiKey()).toBe('keychain-secret');
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).toEqual({
      apiUrl: 'http://x',
      apiKey: 'legacy-secret',
    });
    expect(invokeMock).toHaveBeenCalledTimes(1);
  });

  it('does not overwrite settings or a replacement key written during migration', async () => {
    enableTauri();
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ theme: 'dark', apiKey: 'legacy-secret' }),
    );
    let keychainValue: string | null = null;
    invokeMock.mockImplementation(async (
      command: string,
      args?: { keyValue?: string },
    ) => {
      if (command === 'get_local_api_key') return keychainValue;
      if (command === 'save_cloud_key') {
        keychainValue = args?.keyValue ?? null;
        localStorage.setItem(
          SETTINGS_KEY,
          JSON.stringify({ theme: 'light', apiKey: 'replacement-secret' }),
        );
        return undefined;
      }
      if (command === 'start_backend') return undefined;
      throw new Error(`Unexpected command: ${command}`);
    });

    const { initApiKey } = await freshApi();
    const result = await initApiKey();

    expect(result).toMatchObject({
      ok: false,
      code: 'legacy_cleanup_failed',
    });
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).toEqual({
      theme: 'light',
      apiKey: 'replacement-secret',
    });
    expect(invokeMock.mock.calls.map(([command]) => command)).toEqual([
      'get_local_api_key',
      'save_cloud_key',
      'get_local_api_key',
    ]);
  });

  it('verifies Keychain, erases legacy plaintext, and restarts the backend', async () => {
    enableTauri();
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ apiUrl: 'http://x', apiKey: 'legacy-secret' }),
    );
    let keychainValue: string | null = null;
    invokeMock.mockImplementation(async (
      command: string,
      args?: { keyValue?: string },
    ) => {
      if (command === 'get_local_api_key') return keychainValue;
      if (command === 'save_cloud_key') {
        keychainValue = args?.keyValue ?? null;
        return undefined;
      }
      if (command === 'start_backend') return undefined;
      throw new Error(`Unexpected command: ${command}`);
    });

    const { getApiKey, initApiKey } = await freshApi();
    const result = await initApiKey();

    expect(result).toEqual({ ok: true, migrated: true });
    expect(getApiKey()).toBe('legacy-secret');
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).toEqual({
      apiUrl: 'http://x',
    });
    expect(invokeMock.mock.calls.map(([command]) => command)).toEqual([
      'get_local_api_key',
      'save_cloud_key',
      'get_local_api_key',
      'start_backend',
    ]);
  });

  it('retries backend start without repeating a completed Keychain migration', async () => {
    enableTauri();
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ apiKey: 'legacy-secret' }),
    );
    let keychainValue: string | null = null;
    let startAttempts = 0;
    let saveAttempts = 0;
    invokeMock.mockImplementation(async (
      command: string,
      args?: { keyValue?: string },
    ) => {
      if (command === 'get_local_api_key') return keychainValue;
      if (command === 'save_cloud_key') {
        saveAttempts += 1;
        keychainValue = args?.keyValue ?? null;
        return undefined;
      }
      if (command === 'start_backend') {
        startAttempts += 1;
        if (startAttempts === 1) throw new Error('backend busy');
        return undefined;
      }
      throw new Error(`Unexpected command: ${command}`);
    });

    const { initApiKey } = await freshApi();
    const firstResult = await initApiKey();
    const retryResult = await initApiKey();

    expect(firstResult).toMatchObject({
      ok: false,
      code: 'backend_restart_failed',
    });
    expect(retryResult).toEqual({ ok: true, migrated: false });
    expect(saveAttempts).toBe(1);
    expect(startAttempts).toBe(2);
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).toEqual({});
  });

  it('keeps the active desktop session key while Keychain is rotated or cleared', async () => {
    enableTauri();
    invokeMock.mockResolvedValue(undefined);

    const {
      getApiKey,
      saveCloudKey,
      setRuntimeApiKey,
    } = await freshApi();
    setRuntimeApiKey('active-session-key');

    await saveCloudKey('OPENJARVIS_API_KEY', 'next-restart-key');
    expect(getApiKey()).toBe('active-session-key');

    await saveCloudKey('OPENJARVIS_API_KEY', '');
    expect(getApiKey()).toBe('active-session-key');
  });
});

describe('authHeaders', () => {
  it('omits Authorization when no key is set (keyless default unchanged)', async () => {
    const { authHeaders } = await freshApi();
    expect(authHeaders()).toEqual({});
  });

  it('adds a Bearer Authorization header when a key is set', async () => {
    const { authHeaders, setRuntimeApiKey } = await freshApi();
    setRuntimeApiKey('sk-local-123');
    expect(authHeaders()).toEqual({ Authorization: 'Bearer sk-local-123' });
  });

  it('merges extra headers alongside Authorization', async () => {
    const { authHeaders, setRuntimeApiKey } = await freshApi();
    setRuntimeApiKey('sk-local-123');
    expect(authHeaders({ 'Content-Type': 'application/json' })).toEqual({
      'Content-Type': 'application/json',
      Authorization: 'Bearer sk-local-123',
    });
  });
});

describe('streaming auth', () => {
  it('uses the ephemeral Bearer token on chat SSE requests', async () => {
    const { setRuntimeApiKey } = await freshApi();
    setRuntimeApiKey('sk-runtime-123');

    const releaseLock = vi.fn();
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      body: {
        getReader: () => ({
          read: vi.fn().mockResolvedValue({ done: true }),
          releaseLock,
        }),
      },
    });
    vi.stubGlobal('fetch', fetchMock);

    const { streamChat } = await import('./sse');
    const stream = streamChat({
      model: 'local-model',
      messages: [{ role: 'user', content: 'hello' }],
      stream: true,
    });
    await stream.next();

    expect(fetchMock).toHaveBeenCalledWith(
      '/v1/chat/completions',
      expect.objectContaining({
        headers: {
          'Content-Type': 'application/json',
          Authorization: 'Bearer sk-runtime-123',
        },
      }),
    );
    expect(releaseLock).toHaveBeenCalledOnce();
  });
});
