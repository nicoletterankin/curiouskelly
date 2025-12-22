## Release Notes — 2025-12-21 — Settings/Billing/Support Truth Pass

### Goal
Remove “broken promises” from the in-app Settings panel and ensure every label maps to **real, working wiring**.

### Scope (what changed)
- **Settings → Support**
  - Replaced “Help & FAQ” with **email-first support** (`hello@curiouskelly.com`).
  - Contact panel now clearly indicates email replies require an email address.
  - Removed FAQ-style claims that implied features we don’t actually expose.

- **Settings → Billing**
  - Replaced “Support Kelly / Sponsor” framing with **Billing** and aligned the CTAs to the **existing Stripe embedded checkout flow**.
  - Unified messaging around the principle:
    - **“Contributing is learning”** — contribute **time**, **talent**, or **tokens**.
    - Tokens = either **BYOK provider API keys** (third-party) or **money via Stripe**.

- **Preferences**
  - Ensured Preferences toggles are **real and wired**:
    - Auto-play (auto-advance) and Captions now stay in sync across both toggle UI and accordion checkbox UI.

- **Copy hardening**
  - Removed/avoided “free”, “unlock”, “unlimited”, and other misleading language from Settings/Billing surfaces.
  - Ensured legal/footer copy references **Lesson of the Day PBC** where appropriate.

### Files changed
- `public/learn.html`
- `public/js/i18n/language-selector.js`

### No new systems
This release only updates **existing UI and wiring**:
- Existing mailto support path
- Existing Supabase insert attempt for contact messages (with mailto fallback)
- Existing Stripe embedded checkout panel (`/api/create-checkout`, `/api/stripe-public`, `/api/create-gift-checkout`, `/api/subscription-status`)
- Existing BYOK UI (local-only keys, provider links)

### Deployment Notes / Risks (verify before announcing)
- **Stripe checkout requires server-side `/api/*` endpoints**.
  - Verify your production host actually serves:
    - `/api/stripe-public`
    - `/api/create-checkout`
    - `/api/create-gift-checkout` (if gifts are enabled)
    - `/api/subscription-status`
  - If deploying as static-only (where `/api/*` is disabled), Billing will render but checkout will not complete.

### Post-deploy verification (quick)
- Open `public/learn.html` in production and verify:
  - Settings → **Support** opens Contact and `mailto:hello@curiouskelly.com` works
  - Settings → **Billing** opens the right-panel checkout and plan switching works
  - Preferences toggles:
    - Auto-play and Captions persist (reload page and confirm)
    - Toggle UI and accordion checkbox UI stay in sync


