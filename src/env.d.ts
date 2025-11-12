/// <reference types="astro/client" />

interface Window {
  dataLayer: unknown[];
  turnstile?: {
    render: (
      element: Element,
      options: {
        sitekey: string | null;
        callback: (token: string) => void;
      }
    ) => void;
  };
  grecaptcha?: {
    execute: (siteKey: string, options: { action: string }) => Promise<string>;
  };
  __onRecaptchaLoaded?: () => void;
  __CURIOUS_KELLY__?: {
    locale: string;
    consentRequired: boolean;
    countdownEnd: string;
    analytics: Record<string, string>;
  };
  fbq?: (...args: unknown[]) => void;
  twq?: {
    version: string;
    queue: unknown[];
    exe: (...args: unknown[]) => void;
  } & ((...args: unknown[]) => void);
  ttq?: unknown[];
  clarity?: (...args: unknown[]) => void;
  _tfa?: unknown[];
  $?: typeof import('jquery');
  jQuery?: typeof import('jquery');
}

