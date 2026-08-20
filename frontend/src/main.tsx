import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router';
import { ErrorBoundary } from './components/ErrorBoundary';
import App from './App';
import {
  initApiBase,
  initApiKey,
  type ApiKeyInitResult,
} from './lib/api';
import { initAnalytics } from './lib/analytics';
import './index.css';

function applyTheme() {
  try {
    const raw = localStorage.getItem('openjarvis-settings');
    const settings = raw ? JSON.parse(raw) : {};
    const theme = settings.theme || 'system';
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    } else if (theme === 'light') {
      document.documentElement.classList.add('light');
      document.documentElement.classList.remove('dark');
    }
  } catch { /* use system default */ }
}

applyTheme();

const root = createRoot(document.getElementById('root')!);

function renderStartupFailure(
  failure: Extract<ApiKeyInitResult, { ok: false }>,
) {
  root.render(
    <main
      role="alert"
      className="min-h-screen flex items-center justify-center p-6"
      style={{
        background: 'var(--color-bg)',
        color: 'var(--color-text)',
      }}
    >
      <section
        className="max-w-lg rounded-xl p-6"
        style={{
          background: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border)',
        }}
      >
        <h1 className="text-lg font-semibold">Secure startup paused</h1>
        <p className="mt-2 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
          {failure.message} No API key value was logged or shown.
        </p>
        <p className="mt-2 text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
          Resolve Keychain or storage access, then retry. OpenJarvis will not
          open the normal interface until secure key initialization succeeds.
        </p>
        <button
          type="button"
          className="mt-4 rounded-lg px-3 py-2 text-sm font-medium"
          style={{ background: 'var(--color-accent)', color: 'white' }}
          onClick={() => void bootstrap()}
        >
          Retry secure startup
        </button>
      </section>
    </main>,
  );
}

function renderApp() {
  // Kick off analytics init in the background — it's never awaited so
  // a slow/failed identity fetch never delays UI render.
  void initAnalytics();

  root.render(
    <StrictMode>
      <ErrorBoundary>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </ErrorBoundary>
    </StrictMode>,
  );
}

let bootstrapInFlight: Promise<void> | null = null;

async function runBootstrap() {
  root.render(
    <main
      role="status"
      className="min-h-screen flex items-center justify-center p-6"
      style={{
        background: 'var(--color-bg)',
        color: 'var(--color-text-secondary)',
      }}
    >
      Securing local API access...
    </main>,
  );

  await initApiBase();
  const apiKeyResult = await initApiKey();
  if (!apiKeyResult.ok) {
    renderStartupFailure(apiKeyResult);
    return;
  }
  renderApp();
}

function bootstrap(): Promise<void> {
  if (bootstrapInFlight) return bootstrapInFlight;
  bootstrapInFlight = runBootstrap().finally(() => {
    bootstrapInFlight = null;
  });
  return bootstrapInFlight;
}

// Fetch the API base URL and hydrate the ephemeral Bearer token before
// rendering. This keeps the first HTTP/SSE request authenticated without ever
// persisting OPENJARVIS_API_KEY in the WebView's localStorage.
void bootstrap();
