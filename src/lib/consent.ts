export type ConsentCategory = 'analytics' | 'marketing';

export interface ConsentState {
  analytics: boolean;
  marketing: boolean;
  updatedAt: string;
}

type ConsentListener = (state: ConsentState) => void;
type Loader = () => void;

const STORAGE_KEY = 'curious_kelly_consent';
const loaders: Record<ConsentCategory, Set<Loader>> = {
  analytics: new Set(),
  marketing: new Set()
};
const listeners = new Set<ConsentListener>();

function getDefaultState(): ConsentState {
  return {
    analytics: false,
    marketing: false,
    updatedAt: new Date().toISOString()
  };
}

export function loadConsent(): ConsentState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return getDefaultState();
    }
    const parsed = JSON.parse(raw) as ConsentState;
    return {
      analytics: Boolean(parsed.analytics),
      marketing: Boolean(parsed.marketing),
      updatedAt: parsed.updatedAt ?? new Date().toISOString()
    };
  } catch {
    return getDefaultState();
  }
}

export function persistConsent(state: ConsentState): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

export function updateConsent(partial: Partial<ConsentState>): ConsentState {
  const current = loadConsent();
  const next: ConsentState = {
    ...current,
    ...partial,
    updatedAt: new Date().toISOString()
  };
  persistConsent(next);
  notifyListeners(next);
  fireLoaders(next);
  return next;
}

export function onConsentChange(listener: ConsentListener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function registerLoader(category: ConsentCategory, loader: Loader): () => void {
  loaders[category].add(loader);
  const state = loadConsent();
  if (state[category]) {
    loader();
  }
  return () => loaders[category].delete(loader);
}

function notifyListeners(state: ConsentState) {
  listeners.forEach((listener) => listener(state));
}

function fireLoaders(state: ConsentState) {
  (Object.keys(loaders) as ConsentCategory[]).forEach((category) => {
    if (state[category]) {
      loaders[category].forEach((loader) => loader());
    }
  });
}

export function exposeConsentApi() {
  const state = loadConsent();
  fireLoaders(state);
  const api = {
    get: () => loadConsent(),
    set: (partial: Partial<ConsentState>) => updateConsent(partial),
    onChange: (listener: ConsentListener) => onConsentChange(listener)
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (window as any).consent = api;
  return api;
}












