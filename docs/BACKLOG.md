# 📋 KELLY BACKLOG

**Last Updated:** December 17, 2025

---

## 🔥 Priority 1: Blocking Launch

### Stripe International Setup
**Status:** Ready for execution  
**Time:** 45-60 minutes  
**Guide:** `docs/billing/STRIPE_BATCH_WORK.md`

- [ ] Create 36 multi-currency prices (EUR, GBP, CAD, AUD, INR, BRL, MXN, PLN)
- [ ] Enable 18 payment methods (iDEAL, SEPA, UPI, Boleto, etc.)
- [ ] Add 36 environment variables to Vercel
- [ ] Test international checkout flow

---

## 🌍 Priority 2: Internationalization

### Phase 1: Foundation ✅ COMPLETE
**Guide:** `docs/i18n/INTERNATIONALIZATION_MASTER_PLAN.md`

- [x] Create `/public/locales/en/` directory with translation files
- [x] Build i18n-core.js translation engine
- [x] Build i18n-kelly.js personality layer
- [x] Build language-selector.js component
- [x] Wire up scripts in learn.html
- [x] Create locales manifest

### Phase 2: Spanish & Portuguese ✅ COMPLETE (UI)
- [x] Translate UI strings to Spanish (6 namespaces)
- [x] Translate UI strings to Portuguese (6 namespaces)
- [ ] Machine translate lesson content (Days 1-365)
- [ ] Human review translations

### Phase 3: Context Awareness ✅ COMPLETE
- [x] Build /api/geo-context.ts (time, season, weather, hemisphere)
- [x] Build Kelly greetings with context awareness
- [x] Add hemisphere-aware seasons (northern/southern)
- [ ] Holiday awareness for major regions (future enhancement)

---

## 📚 Priority 3: Content

### Lesson Translations
- [ ] Spanish: Lessons 1-365
- [ ] Portuguese: Lessons 1-365
- [ ] French: Lessons 1-365 (Feb 2026)
- [ ] German: Lessons 1-365 (Feb 2026)

### Audio Generation
- [ ] Spanish TTS for all lessons
- [ ] Portuguese TTS for all lessons
- [ ] French TTS (Feb 2026)
- [ ] German TTS (Feb 2026)

---

## 💳 Completed Today (Dec 17)

### Payment System Audit & Fixes
- [x] Enabled paywall (`testingMode: false`)
- [x] Fixed Stripe API version
- [x] Added idempotency keys
- [x] Added rate limiting
- [x] Fixed family plan handling
- [x] Fixed error message security
- [x] Created email library
- [x] Added pause/resume subscription endpoints
- [x] Updated webhook to handle more events

### International Pricing Infrastructure
- [x] Built `api/lib/pricing-config.ts` (single source of truth)
- [x] Built `api/geo-pricing.ts` (country detection + localized prices)
- [x] Built `public/js/geo-pricing.js` (frontend integration)
- [x] Updated `api/create-checkout.ts` for multi-currency
- [x] Updated `api/stripe-checkout.ts` for multi-currency
- [x] Updated `api/create-gift-checkout.ts` for multi-currency
- [x] Updated paywall with `data-price` attributes

### Documentation Created
- [x] `docs/billing/PAYMENT_LOGIC_AUDIT.md`
- [x] `docs/billing/INTERNATIONAL_PRICING_PLAN.md`
- [x] `docs/billing/STRIPE_BATCH_WORK.md`
- [x] `docs/i18n/INTERNATIONALIZATION_MASTER_PLAN.md`

### i18n Foundation Built (Dec 17)
- [x] `/public/locales/en/` - 6 complete namespace files
- [x] `/public/locales/es/` - 6 complete namespace files (Spanish)
- [x] `/public/locales/pt/` - 6 complete namespace files (Portuguese)
- [x] `/public/locales/manifest.json` - language configuration
- [x] `/public/js/i18n/i18n-core.js` - translation engine
- [x] `/public/js/i18n/i18n-kelly.js` - personality layer
- [x] `/public/js/i18n/language-selector.js` - dropdown component
- [x] `/api/geo-context.ts` - location/time/season API

---

## 📊 Progress Summary

| Area | Status | Next Action |
|------|--------|-------------|
| **Payments** | 🟢 Code ready | You: Stripe Dashboard work |
| **Geo-Pricing** | 🟢 Code ready | You: Add prices in Stripe |
| **i18n Foundation** | 🟢 Complete | Ready to use |
| **Spanish UI** | 🟢 Complete | All 6 namespaces |
| **Portuguese UI** | 🟢 Complete | All 6 namespaces |
| **Context API** | 🟢 Complete | Season, time, hemisphere aware |
| **Lesson Content Translation** | 🟡 Next phase | Machine translate days 1-365 |

---

## 🏗️ i18n File Structure

```
public/
├── locales/
│   ├── manifest.json          # Language configuration
│   ├── en/
│   │   ├── common.json        # Nav, actions, status
│   │   ├── lessons.json       # Player, phases, completion
│   │   ├── settings.json      # Preferences, account
│   │   ├── paywall.json       # Plans, checkout, gifts
│   │   ├── kelly.json         # Greetings, reactions, personality
│   │   └── onboarding.json    # Welcome flow, age gate
│   ├── es/
│   │   └── (same structure)
│   └── pt/
│       └── (same structure)
└── js/
    └── i18n/
        ├── i18n-core.js       # Translation engine
        ├── i18n-kelly.js      # Context-aware personality
        └── language-selector.js # Dropdown component
```

---

*Updated: December 17, 2025 6:30 PM*
