## Performance Playbook

### Core Web Vital Budgets

| Metric | Budget | Notes |
| --- | --- | --- |
| LCP | `< 2.5s` (Moto G / Slow 4G) | Hero image uses gradients + text to minimise payload; future media should leverage `@astrojs/image`. |
| CLS | `< 0.1` | Avoid layout shifts; testimonials carousel pauses on focus and uses fixed aspect panels. |
| INP | `< 200ms` | All primary interactions rely on lightweight islands; form validation uses synchronous logic. |
| JS transfer | `≤ 80 KB gzip` (homepage) | PurgeCSS eliminates unused Bootstrap rules; only required libraries hydrate. |

Budgets enforced through `tests/lighthouse/lhci.config.cjs` which runs against the locally built site (`npm run test:lhci`).

### Build Optimisations

- **Critical CSS**: `scripts/extract-critical-css.mjs` inlines top-route CSS post-build using Critters.
- **PurgeCSS**: `vite-plugin-purgecss` keeps Bootstrap foot-print minimal; safelist covers modal/collapse/carousel/intl-tel-input classes.
- **Code splitting**: Astro isolates interactive islands; jQuery/Slick load only in `TestimonialsCarousel`.
- **Lazy hydration**: Countdown, locale prompt, consent manager ship as inline modules with `nonce` to maintain CSP compatibility.
- **Image strategy**: Currently gradient-driven hero; when adding media, use `@astrojs/image` (`serviceEntryPoint: sharp`) for adaptive WebP/AVIF output.

### Runtime Observability

- **Real User Monitoring**: `@lib/rum.initRum` captures LCP/CLS/INP via PerformanceObserver when `PUBLIC_RUM_ENABLED=true`. POSTs to `/api/rum` (disabled by default in production). Development builds log to `.data/rum.log`.
- **Client Logger**: `@lib/logger` exposes `window` events for debug sessions (`?debug=1` query flag).
- **Consent Telemetry**: `trackConsentChange` pushes `consent_changed` events to the data layer for tag manager parity.

### Testing Matrix

| Tool | Command | Purpose |
| --- | --- | --- |
| Vitest | `npm run test:unit` | Validates utility logic (`validation`, countdown utilities). |
| Playwright | `npm run test:e2e` | Runs lead form submission (with mocked API) + locale banner flow in mobile viewports. |
| Lighthouse CI | `npm run test:lhci` | Checks budgets against `/`, `/adults/`, `/privacy/`. |

CI (`.github/workflows/build-test.yml`) executes all three suites on every push/PR. Artifacts (Playwright report, Lighthouse JSON) upload for inspection.

### Performance Regression Process

1. Run `npm run build` locally; inspect `dist/` asset sizes (ensure JS bundles remain under budget).
2. Execute `npm run test:lhci`; review HTML summary and budgets.
3. For new interactive components, profile with Chrome Performance recording to confirm INP stays <200ms.
4. Use `npm run dev -- --host` on mobile hardware for manual smoke tests.
5. If budgets exceeded, document root cause + mitigation in PR summary; do not merge until budgets pass or stakeholders approve.

### Future Enhancements

- Integrate `astro-compress` for automatic HTML compression.
- Add hero illustration pipeline using `@astrojs/image/components` once assets finalised.
- Streamline jQuery usage with native carousel replacement when legacy parity no longer required (would reduce JS weight further).
- Hook RUM endpoint to analytics backend (BigQuery / Reinmaker metrics) when ready; keep storage optional until cost approved.




