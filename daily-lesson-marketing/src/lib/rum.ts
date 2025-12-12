import { logger } from './logger';

interface RumPayload {
  metric: 'LCP' | 'CLS' | 'INP';
  value: number;
  navigationType: string;
  locale: string;
}

const ENDPOINT = '/api/rum';

function sendToServer(payload: RumPayload) {
  if (import.meta.env.PUBLIC_RUM_ENABLED !== 'true') {
    return;
  }
  fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    keepalive: true
  }).catch((error) => {
    logger.warn('RUM beacon failed', { error });
  });
}

export function initRum(locale: string) {
  if (!('PerformanceObserver' in window)) {
    return;
  }
  const navEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined;
  const navigationType = navEntry?.type ?? 'navigate';

  const metrics: PerformanceObserverInit[] = [
    {
      type: 'largest-contentful-paint',
      buffered: true
    },
    {
      type: 'layout-shift',
      buffered: true
    },
    {
      type: 'event',
      buffered: true,
      durationThreshold: 40
    } as PerformanceObserverInit
  ];

  metrics.forEach((config) => {
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.entryType === 'largest-contentful-paint') {
          sendToServer({ metric: 'LCP', value: entry.startTime, navigationType, locale });
        }
        if (entry.entryType === 'layout-shift') {
          const layoutShift = entry as LayoutShift;
          if (!layoutShift.hadRecentInput) {
            sendToServer({
              metric: 'CLS',
              value: layoutShift.value,
              navigationType,
              locale
            });
          }
        }
        if (entry.entryType === 'event') {
          const inp = (entry as PerformanceEventTiming).interactionId ? (entry as PerformanceEventTiming).duration : undefined;
          if (inp) {
            sendToServer({ metric: 'INP', value: inp, navigationType, locale });
          }
        }
      }
    });
    observer.observe(config);
  });
}












