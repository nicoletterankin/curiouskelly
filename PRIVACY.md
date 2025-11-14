## Privacy & Data Handling

CuriousKelly.com marketing site is designed around the privacy-first mandates in `CLAUDE.md`.

### Consent Framework

- Consent banner renders on first visit (and whenever storage is cleared) unless `PUBLIC_CONSENT_REQUIRED=false`.
- Choices stored in `localStorage` under `curious_kelly_consent` with timestamp metadata.
- `window.consent` API exposes `get`, `set`, and `onChange` to support live inspectors and future analytics dashboards.
- Analytics/marketing scripts (GTM, GA4, Meta, TikTok, Taboola, VWO, Hotjar, Clarity) register deferred loaders and only execute after consent.
- Revoking consent removes injected `<script>` tags and GTM `<noscript>` iframe placeholders.

### Lead Form Data Flow

1. **Collection**: Required fields – first name, last name, email, phone, country, region; optional marketing opt-in.
2. **Client validation**: Sanitises before submission; prevents requests without necessary fields or valid phone numbers.
3. **Anti-bot**:
   - Cloudflare Turnstile preferred (non-Google, privacy-aligned).
   - Optional Google reCAPTCHA v3 fallback (disabled by default).
   - Tokens verified server-side; rejected requests return `400` with `turnstile_failed` or `recaptcha_failed`.
4. **Server processing** (`functions/handlers/lead.ts`):
   - Hashes email (SHA-256, truncated) for log correlation without storing raw PII.
   - Logs sanitized payload only in development (`.data/leads.log`).
   - Forwards to `CRM_WEBHOOK_URL` if set. Secrets (auth token, CRM URL, captcha secrets) never exposed to the browser.
   - Captcha tokens are stripped prior to CRM forwarding.

### Retention & Storage

- Static site contains no databases; all persisted data lives in customer CRM (external).
- Local `.data/*` logs (mock CRM, lead, rum) exist only in development and are ignored via `.gitignore`.
- Service worker avoids caching responses for `/api/*` (NetworkOnly), preventing sensitive data from entering offline caches.

### Tracking & Cookies

- Cookies used:
  - `localeChoice` (1-year max-age) for locale preference.
  - Turnstile/reCAPTCHA may set their own cookies per their policies.
- No interest-based personalisation; no “learning-style” profiling at runtime.
- RUM endpoint disabled by default; enabling requires conscious opt-in via `PUBLIC_RUM_ENABLED=true`.

### Incident Response & Compliance

- CRM webhook failures return `502 crm_unavailable`; Playwright test ensures errors surface to the user.
- `RUNBOOK.md` documents escalation paths and rollback steps.
- Updates to consent categories or data sharing require prior approval (per `CLAUDE.md`).
- Any new third-party script must be wrapped in the consent loader with clear classification (`analytics` vs `marketing`) and documented cost implications.





