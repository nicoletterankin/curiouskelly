## Operations Runbook

### Daily Checks

- Confirm production build (`npm run build`) completes with no errors.
- Review Lighthouse CI output from last merge (budget regressions must be addressed before next deploy).
- Inspect CRM webhook health (200 responses, latency < `CRM_TIMEOUT_MS`).
- Verify consent banner renders in an incognito session and that marketing tags remain blocked until opt-in.

### Deployments

1. **Pre-flight**
   - Ensure `build-test` workflow is green on the target branch.
   - Update `.env` with environment-specific keys (Turnstile, CRM, analytics).
   - Run `npm run build` locally; confirm `dist/` contains sitemap, service worker, and expected assets.
2. **Trigger desired workflow**
   - Vercel: `Deploy to Vercel` workflow (set environment).
   - Netlify: `Deploy to Netlify` (set production flag when releasing).
   - Cloudflare Pages: `Deploy to Cloudflare Pages` (specify branch).
3. **Post-deploy validation**
   - Spot check `/`, `/adults/`, `/thank-you/`, `/privacy/` in each locale.
   - Submit a test lead with mock CRM running; ensure thank-you page loads and request appears in mock log.
   - Review consent opt-in/out to confirm analytics scripts load/unload correctly.

### Incident Playbooks

| Issue | Actions |
| --- | --- |
| **Lead form returns errors** | Check `/api/lead` function logs (provider dashboard). Validate captcha secrets. Reproduce locally with `npm run dev` + `npm run mock:crm`. |
| **CRM webhook offline** | Temporarily set `CRM_WEBHOOK_URL` to mock server; notify stakeholders. Add context to lead log for replay. |
| **Consent banner missing** | Verify `PUBLIC_CONSENT_REQUIRED` env, inspect `localStorage.curious_kelly_consent`. Clear stored value; ensure component in layout. |
| **Performance regressions** | Re-run `npm run test:lhci`; identify largest JS bundles via `dist/assets`. Consider code splitting or lazy loading. |
| **Analytics tags firing without consent** | Confirm script registration occurs within consent loader. Ensure `registerLoader` categories align with `updateConsent` state. |
| **Turnstile/reCAPTCHA downtime** | Disable anti-bot by clearing relevant env keys (temporary). Monitor spam volume; reinstate once service recovers. |

### CRM Mock Server

- Start with `npm run mock:crm` (defaults to port `8787`).
- Used for local development and smoke tests in CI.
- Logs requests to `.data/mock-crm.log` (ignored by git).
- When switching to production CRM, update environment variables; consider leaving mock running for parallel monitoring.

### Access & Secrets

- Secrets managed via deployment platform: Vercel/Netlify/Cloudflare environment variables and GitHub Actions secrets.
- Never commit real keys. `.env.example` contains placeholders only.
- Rotating captcha or CRM secrets requires redeploying functions (rerun `npm run build` and workflow).

### On-Call Checklist

- Bookmark provider dashboards:
  - Vercel Function logs (`/api/lead`, `/api/rum`)
  - Netlify Functions monitor
  - Cloudflare Pages logs (if deployed)
- Maintain contact for CRM vendor; ensure webhook retries (if available) are configured.
- Keep `TURNSTILE_SECRET_KEY` stored in secure vault; same for `RECAPTCHA_SECRET_KEY` fallback.
- Track release notes in repo issues or `docs/roadmaps` for audit.

### Rollback Procedure

1. Identify last known good build (Git tag/commit).
2. Redeploy via workflow using that commit/branch (set environment to `production` where applicable).
3. Clear CDN/cache (Vercel/Netlify/Cloudflare) to avoid stale assets.
4. Announce rollback in team channel and log cause for follow-up.

### Maintenance Windows

- **Monthly**: Update npm dependencies (`npm outdated` → `npm update` or targeted bump). Run `npm run lint`, `npm run test:*`, `npm run build`.
- **Quarterly**: Review Lighthouse budgets and update if product scope changes. Validate new locales or marketing tags follow consent guidelines.
- **Ad-hoc**: Add new CRM integrations or analytics tags only after verifying consent gating and updating documentation (README + PRIVACY).





