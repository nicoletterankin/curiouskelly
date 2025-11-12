# CuriousKelly.com Jamstack Marketing Site

Privacy-first, multilingual marketing experience for CuriousKelly.com built with Astro, TypeScript, Bootstrap 4.6, and serverless forms.

## Quick Start

1. **Install prerequisites**
   - Node.js ≥ 18.17
   - npm ≥ 9

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Copy environment template**
   ```bash
   cp .env.example .env
   ```
   Fill in keys for Turnstile/recaptcha/analytics as needed.

4. **Run mock CRM (optional)**
   ```bash
   npm run mock:crm
   ```
   Logs requests to `.data/mock-crm.log`. Default URL `http://localhost:8787/api/mock-crm`.

5. **Start local dev server**
   ```bash
   npm run dev
   ```
   Visit `http://localhost:4321`.

## Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Astro dev server (port 4321) |
| `npm run build` | Static build + critical CSS + service worker + sitemap |
| `npm run preview` | Preview built output |
| `npm run lint` | ESLint + Stylelint |
| `npm run typecheck` | `astro check` type safety |
| `npm run test:unit` | Vitest unit suite |
| `npm run test:e2e` | Playwright flows (uses local dev server) |
| `npm run test:lhci` | Lighthouse CI budgets (`tests/lighthouse/lhci.config.cjs`) |
| `npm run mock:crm` | Local JSON logger for CRM webhooks |

Playwright browsers are installed automatically via the `prepare` script.

## Environment Variables

All values live in `.env`. Keys prefixed with `PUBLIC_` are exposed to the browser.

| Key | Description |
| --- | --- |
| `PUBLIC_SITE_URL` | Canonical site URL used in sitemap/canonical tags |
| `PUBLIC_DEFAULT_LOCALE` | Fallback locale code (`en-US`) |
| `PUBLIC_AVAILABLE_LOCALES` | Comma-separated list of locales (`en-US,es-ES,pt-BR`) |
| `PUBLIC_COUNTDOWN_END` | ISO timestamp for “Register for 2026” banner |
| `PUBLIC_CONSENT_REQUIRED` | `true` to gate tags behind consent |
| `PUBLIC_RUM_ENABLED` | Enable `/api/rum` endpoint + client beacons |
| `PUBLIC_GTM_ID`, `PUBLIC_GA4_ID`, `PUBLIC_META_PIXEL_ID`, etc. | Marketing tags; load only when consented |
| `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET_KEY` | Cloudflare Turnstile credentials |
| `PUBLIC_RECAPTCHA_SITE_KEY`, `RECAPTCHA_SECRET_KEY` | Optional reCAPTCHA v3 fallback |
| `CRM_WEBHOOK_URL` | External CRM endpoint (falls back to mock logger in dev) |
| `CRM_AUTH_TOKEN` | Optional bearer token for CRM |
| `CRM_TIMEOUT_MS` | Timeout for CRM forwarding (default 5000) |
| `PUBLIC_CSP_REPORT_URI` | Optional CSP report endpoint |

## Serverless Functions

- Core handlers live in `functions/handlers/`.
- Provider shims:
  - **Vercel**: `api/lead.ts`, `api/rum.ts` (re-export from `functions/vercel/api/*`)
  - **Netlify**: `functions/netlify` (mapped via `netlify.toml`; redirects `/api/*`)
  - **Cloudflare Pages**: `functions/cloudflare/api/*` (build & deploy with `wrangler.toml`)

Both endpoints accept JSON POSTs:
- `/api/lead` validates input, verifies Turnstile/recaptcha, forwards to CRM (if configured), and sanitises logs.
- `/api/rum` captures anonymous performance metrics when `PUBLIC_RUM_ENABLED=true`.

## Consent & Analytics

- Consent banner (`CookieConsent.astro`) stores choices in `localStorage` under `curious_kelly_consent`.
- `window.consent.get()` / `set()` / `onChange()` are exposed for inspectors or debugging.
- Marketing and analytics scripts load only when permitted; toggling consent removes injected tags.
- Data layer events (`page_view`, `lead_submitted`, `lead_error`, etc.) stream through `@lib/analytics`.

## Internationalisation

Locales are defined under `src/lib/i18n/`. To add a new locale:
1. Duplicate a dictionary file (e.g., `en-us.ts` → `de-de.ts`) and translate copy.
2. Register the locale in `src/lib/i18n/index.ts` (`supportedLocales` array).
3. Add localized routes through `getStaticPaths` in `src/pages/[[...slug]].astro`.
4. Provide localized nav URLs (`dictionary.nav`).
5. Update `.env` `PUBLIC_AVAILABLE_LOCALES`.
6. Re-run `npm run build` to regenerate sitemap with updated hreflang tags.

Language switcher preserves the current path, storing user choice in a `localeChoice` cookie. First-visit locale banner suggests the best fit based on `navigator.languages`.

## Forms & Anti-bot

- `<LeadForm>` renders as an Astro Island with static fallback instructions.
- Client validation leverages `libphonenumber-js`, localized error strings, and dynamic country/region lists.
- Anti-bot selection:
  - Default Turnstile when both site + secret keys are present.
  - If Turnstile missing and reCAPTCHA keys provided, reCAPTCHA v3 is used.
  - With neither, the form submits without CAPTCHA (suitable for smoke tests only).
- Successful submissions redirect to `/thank-you/` and emit analytics events; errors surface inline with telemetry.

## Deployment

### Vercel
1. Set `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID` secrets in GitHub.
2. Trigger workflow `Deploy to Vercel` → choose environment (`preview` or `production`).

### Netlify
1. Add `NETLIFY_AUTH_TOKEN`, `NETLIFY_SITE_ID` secrets.
2. Run `Deploy to Netlify` workflow; set “Deploy to production?” to `true` for prod publishes.

### Cloudflare Pages
1. Set `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_PROJECT_NAME`.
2. Trigger `Deploy to Cloudflare Pages`, specifying the branch.

Each workflow builds locally (`npm run build`) to guarantee parity with production artifacts.

## Performance & Monitoring

- Critical CSS via `scripts/extract-critical-css.mjs`.
- Service worker (Workbox) caches static assets, respects form submissions (`NetworkOnly` for `/api/*`).
- Lighthouse CI budgets enforced in `tests/lighthouse/lhci.config.cjs`.
- RUM `/api/rum` endpoint collects Core Web Vitals when enabled. Raw samples stored in `.data/rum.log` during development.

See `PERFORMANCE.md` for detailed budgets, measurement loops, and dashboard integration ideas.

## Privacy & Compliance

- `PRIVACY.md` outlines data flow, consent storage, and webhook handling.
- No runtime language generation; all dictionary content precomputed (EN/ES/PT).
- CRM payloads exclude CAPTCHA tokens and include hashed email to minimise PII exposure in logs.

Refer to `PRIVACY.md` and `RUNBOOK.md` for escalation paths, retention policy, and incident response.

## Troubleshooting

- **Playwright fails locally**: ensure required browsers installed (`npx playwright install --with-deps`), check for conflicting port 4321.
- **intl-tel-input assets missing**: `npm install` must complete to copy CSS; rebuild after install.
- **Turnstile errors**: verify secret matches site key; view console logs for `turnstile_missing_token`.
- **Locale banner not showing**: check `localeChoice` cookie; clear to simulate first visit.
- **CRM webhook timeouts**: adjust `CRM_TIMEOUT_MS` and review `.data/leads.log` for payload snapshots.

## Documentation Index

- `ARCHITECTURE.md` – Component structure, rendering model, deployment targets.
- `PERFORMANCE.md` – Budget definitions, measurement playbooks.
- `PRIVACY.md` – Consent, data retention, webhook hygiene.
- `RUNBOOK.md` – Ops checklist for incident response and release duties.

This site complies with `CLAUDE.md` invariants: multilingual precomputation, privacy gating, no runtime lesson generation, and no browser TTS usage. Pull requests should include links to CI artifacts and Lighthouse summaries.
