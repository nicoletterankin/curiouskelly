## Architecture Overview

This repository delivers the public marketing experience for CuriousKelly.com as a static (Jamstack) site with serverless form handling. The design prioritises multilingual content, privacy compliance, and fast-first interaction loops.

### Technology Stack

- **Astro 4** with TypeScript for static generation and partial hydration.
- **Bootstrap 4.6**, SCSS modules, Slick Carousel, jQuery 3.7 + jQuery Migrate for legacy UI parity.
- **Astro Islands** for interactive components (`LeadForm`, `HeroCountdown`, `TestimonialsCarousel`, consent/locale banners).
- **Serverless functions** implemented once and adapted to Vercel, Netlify, and Cloudflare Pages through thin wrappers.
- **Workbox** service worker for offline caching of static assets.
- **PurgeCSS** (via `vite-plugin-purgecss`) to minimise Bootstrap payload.

### Directory Layout

```
src/
  components/          # Astro Islands & shared UI elements
  layouts/             # `SiteLayout.astro` page skeleton (nav, footer, scripts)
  lib/                 # i18n dictionaries, validation, analytics, consent, utilities
  pages/               # Route implementation using `[...slug]` pattern + 404
styles/                # Bootstrap variable overrides & main SCSS entry
public/                # Favicons, manifest, robots
functions/
  handlers/            # Framework-agnostic function logic
  vercel/api/          # Vercel adapters (re-exported via root `api/`)
  netlify/             # Netlify handlers (wired via `netlify.toml`)
  cloudflare/api/      # Cloudflare Pages functions
scripts/               # Critical CSS, service worker, sitemap, mock CRM
tests/                 # Vitest + Playwright + Lighthouse configuration
```

### Rendering Flow

1. `src/pages/[[...slug]].astro` maps locale-prefixed URLs (`/`, `/es-es/`, `/pt-br/`) to marketing pages or legal content.
2. Each request resolves a `LocaleDictionary` from `src/lib/i18n/*`.
3. `SiteLayout.astro` injects:
   - SEO metadata (`SeoHead.astro`), hreflang alternates, canonical URLs.
   - Global navigation + locale switcher.
   - Consent manager and locale prompt islands (hydrated client-side).
   - Data layer bootstrap, consent registry, optional RUM startup.
4. Page sections render based on `routeKey` (hero, features, pricing, testimonials, legal copy, thank-you checklist).

### Internationalisation

- Localised dictionaries contain all copy, validation messaging, and analytics event names in English (US), Spanish (Spain), and Portuguese (Brazil).
- URL strategy: base path -> `en-US`, `/es-es/` -> Spanish, `/pt-br/` -> Portuguese. Paths maintain trailing slashes.
- Language switcher calls `buildLocalizedPath()` to preserve the current route during locale changes and drops a `localeChoice` cookie.
- Initial visit triggers a non-blocking language banner using `navigator.languages`, respecting stored consent.

### Lead Generation Pipeline

1. `<LeadForm>` Astro Island renders HTML inputs with accessible labels, client validation, and `intl-tel-input`.
2. Anti-bot:
   - Turnstile rendered when `TURNSTILE_SITE_KEY` present.
   - reCAPTCHA v3 fallback when Turnstile unavailable and reCAPTCHA keys set.
3. Submission posts JSON to `/api/lead`.
4. `functions/handlers/lead.ts`:
   - Validates payload (names, email, E.164 phone, locale, journey, country, region).
   - Verifies Turnstile/reCAPTCHA tokens using configured secrets.
   - Forwards sanitized payload to `CRM_WEBHOOK_URL` (with optional bearer token).
   - Logs hashed email + request metadata to `.data/leads.log` during development.
5. Success returns `{ status: "ok", requestId }`, redirected to `/thank-you/` and analytics events fired (`lead_submitted`).

### Consent & Analytics

- `CookieConsent.astro` exposes `window.consent` API backed by `localStorage` and toggles analytics/marketing loaders.
- `SiteLayout` registers loaders for GTM, GA4, Meta Pixel, TikTok, Taboola, VWO, Hotjar, and Clarity. Scripts inject only when allowed; revoking consent removes tags and associated `<noscript>` shims.
- The data layer (`window.dataLayer`) stores events from hero views, locale prompts, consent changes, and form submissions (`@lib/analytics`).

### Performance Pipeline

- CSS: Bootstrap compiled with custom variables, purged of unused selectors. Critical CSS inlined post-build (`scripts/extract-critical-css.mjs`).
- JS budget: homescreen hydrated islands limited to ≤80 KB gzip by restricting third-party libraries to contexts where needed.
- Images: placeholder assets rely on Astro’s static delivery; integration with `@astrojs/image` ready for future responsive assets.
- Service worker caches static assets, uses `NetworkOnly` for `/api/*`, and offers offline shell for key legal pages.
- Lighthouse CI ensures `LCP < 2.5s`, `CLS < 0.1`, `INP < 200ms` on budget devices.

### Deployment Targets

- **Vercel**: Default `api/*` route exports from `api/lead.ts` and `api/rum.ts` ensure parity with Vercel Functions.
- **Netlify**: `netlify.toml` sets `functions/netlify` as runtime directory and rewrites `/api/*` to corresponding functions.
- **Cloudflare Pages**: Functions live in `functions/cloudflare/api/*` with `wrangler.toml` pointing to `dist` output.
- GitHub workflows (`/ .github/workflows`) handle CI and optional environment-specific deployments when secrets are configured.

### Extensibility

- Additional locales: add new dictionary, update `supportedLocales`, and expand sitemap generation (`scripts/generate-sitemap.mjs`).
- New marketing sections: place component modules under `src/components/` and render conditionally by `routeKey`.
- CRM integrations: extend `postToCrm` helper or add branching in `leadHandler` for vendor-specific payload transforms.
- Analytics: register new loader in `SiteLayout` with appropriate consent category.

The architecture aligns with the repository’s CLAUDE.md directives: no runtime lesson generation, consent-first marketing tags, multilingual precomputation, and fast static delivery with optional serverless features.




