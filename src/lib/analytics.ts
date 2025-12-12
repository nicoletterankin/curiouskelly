import type { ConsentState } from './consent';

interface AnalyticsEvent {
  event: string;
  [key: string]: unknown;
}

type Listener = (event: AnalyticsEvent) => void;

const listeners = new Set<Listener>();

function getDataLayer(): unknown[] {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any;
  if (!w.dataLayer) {
    w.dataLayer = [];
  }
  return w.dataLayer as unknown[];
}

export function pushEvent(event: AnalyticsEvent) {
  const dataLayer = getDataLayer();
  dataLayer.push(event);
  listeners.forEach((listener) => listener(event));
}

export function onAnalyticsEvent(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function trackView(locale: string) {
  pushEvent({ event: 'page_view', locale });
}

export function trackLeadSubmitted(locale: string, journey: string) {
  pushEvent({ event: 'lead_submitted', locale, journey });
}

export function trackLeadError(locale: string, message: string) {
  pushEvent({ event: 'lead_error', locale, message });
}

export function trackConsentChange(state: ConsentState) {
  pushEvent({ event: 'consent_changed', consent: state });
}

export function trackLocalePrompt(eventType: 'shown' | 'accepted' | 'dismissed', locale: string) {
  pushEvent({ event: `locale_prompt_${eventType}`, locale });
}

export function trackUnityDemo(eventName: string, payload: Record<string, unknown>) {
  pushEvent({
    event: eventName,
    ...payload
  });
}





