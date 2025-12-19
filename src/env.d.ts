/// <reference types="astro/client" />
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly TURNSTILE_SITE_KEY?: string;
  readonly PUBLIC_RECAPTCHA_SITE_KEY?: string;
  readonly PUBLIC_CONSENT_REQUIRED?: string;
  readonly PUBLIC_GTM_ID?: string;
  readonly PUBLIC_GA4_ID?: string;
  readonly PUBLIC_META_PIXEL_ID?: string;
  readonly PUBLIC_TIKTOK_PIXEL_ID?: string;
  readonly PUBLIC_TWITTER_PIXEL_ID?: string;
  readonly PUBLIC_TABOOLA_ACCOUNT_ID?: string;
  readonly PUBLIC_VWO_ID?: string;
  readonly PUBLIC_HOTJAR_ID?: string;
  readonly PUBLIC_CLARITY_ID?: string;
  readonly PUBLIC_RUM_ENABLED?: string;
  readonly PUBLIC_COUNTDOWN_END?: string;
  readonly PUBLIC_COUNTDOWN_TEST_MODE?: string;
  readonly PUBLIC_SITE_URL?: string;
  readonly PUBLIC_UNITY_IFRAME_SRC?: string;
  readonly PUBLIC_UNITY_SAMPLE_JSON?: string;
  readonly PUBLIC_UNITY_SAMPLE_AUDIO?: string;
  readonly PUBLIC_UNITY_SAMPLE_EXPRESSIONS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

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

