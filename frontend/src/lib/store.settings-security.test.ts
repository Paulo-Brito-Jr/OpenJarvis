import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const SETTINGS_KEY = 'openjarvis-settings';

class MemoryStorage {
  private store = new Map<string, string>();

  getItem(key: string): string | null {
    return this.store.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.store.set(key, String(value));
  }

  removeItem(key: string): void {
    this.store.delete(key);
  }
}

beforeEach(() => {
  vi.resetModules();
  vi.stubEnv('VITE_SUPABASE_ANON_KEY', 'test-anon-key');
  (globalThis as unknown as { localStorage: MemoryStorage }).localStorage =
    new MemoryStorage();
});

afterEach(() => {
  vi.unstubAllEnvs();
  (globalThis as unknown as { localStorage?: MemoryStorage }).localStorage =
    undefined;
});

describe('settings API key persistence', () => {
  it('keeps legacy storage untouched until secure startup commits migration', async () => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({
      theme: 'dark',
      apiUrl: 'http://127.0.0.1:8000',
      apiKey: 'legacy-secret',
    }));

    const { useAppStore } = await import('./store');

    expect(useAppStore.getState().settings.apiKey).toBe('');
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).toEqual({
      theme: 'dark',
      apiUrl: 'http://127.0.0.1:8000',
      apiKey: 'legacy-secret',
    });
  });

  it('keeps API key edits in memory when other settings are persisted', async () => {
    const { useAppStore } = await import('./store');

    useAppStore.getState().updateSettings({ apiKey: 'session-secret' });
    useAppStore.getState().updateSettings({ theme: 'dark' });

    expect(useAppStore.getState().settings.apiKey).toBe('session-secret');
    expect(JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}')).not.toHaveProperty('apiKey');
  });
});
