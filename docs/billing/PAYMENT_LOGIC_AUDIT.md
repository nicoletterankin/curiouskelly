# 💳 Payment Logic Audit - Curious Kelly

**Audit Date:** December 17, 2025  
**Auditor:** Claude

---

## Executive Summary

Your payment model is **"Today is Always Yours"** - a generous freemium approach where the current day's lesson is always accessible without payment. Past lessons and full access require subscription or individual purchase.

---

## 🎯 WHEN Do We Ask for Payment?

### The Core Rule: "Today is Always Yours"

```
┌─────────────────────────────────────────────────────────────┐
│                    ACCESS DECISION TREE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User requests Day X                                         │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ Is Day X     │──── YES ──→ ✅ ALLOW (No payment needed)  │
│  │ TODAY?       │                                           │
│  └──────────────┘                                           │
│         │ NO                                                 │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ Is user      │──── YES ──→ ✅ ALLOW                      │
│  │ subscribed?  │                                           │
│  └──────────────┘                                           │
│         │ NO                                                 │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ Did user buy │──── YES ──→ ✅ ALLOW                      │
│  │ this lesson? │                                           │
│  └──────────────┘                                           │
│         │ NO                                                 │
│         ▼                                                    │
│  🔒 SHOW PAYWALL (after 5-second preview)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Paywall Trigger Points

| Trigger | When | Delay |
|---------|------|-------|
| **Past lesson access** | User navigates to any day before today | 5 seconds (preview first) |
| **Future lesson access** | User tries to access unreleased day | Instant |
| **Journey calendar tap** | User taps a non-today lesson | 5 seconds |
| **Deep link to past day** | URL like `?day=50` | 5 seconds |

### Current Config (from `config.js`)

```javascript
testingMode: true,      // ⚠️ CURRENTLY DISABLED FOR TESTING
disablePaywall: true,   // ⚠️ PAYWALL IS OFF - change before launch!
paywallDelayMs: 5000,   // 5-second preview before paywall

accessModel: {
  todayIsFree: true,           // The core promise
  enablePayPerLesson: true,    // Buy individual lessons ($1.99)
  enableSubscription: true,    // Monthly/annual/lifetime
  emergencyLessonsCount: 40    // Bonus lessons for subscribers
}
```

---

## 💰 WHY Do We Ask for Payment?

### Value Proposition

| What They Get | Price Point |
|---------------|-------------|
| Today's lesson | **Always included** - no payment |
| One past lesson | $1.99 (own forever) |
| All 365 lessons + 40 bonus | $7.99/month, $49.99/year, or $199.99 lifetime |
| Family (6 members) | $99.99/year |

### Business Model Rationale

1. **Low friction entry**: Anyone can learn today without account/payment
2. **Natural scarcity**: Miss today? Pay to catch up
3. **FOMO driver**: "Today's lesson won't be yours tomorrow"
4. **Subscription value**: Archive access makes subscription worth it

---

## 🛠️ HOW Does Payment Work?

### Payment Flow Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        PAYMENT FLOW                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. USER SEES PAYWALL                                            │
│     └─→ Options: Buy lesson ($1.99) | Subscribe ($7.99/mo)       │
│                                                                   │
│  2. USER SELECTS PLAN                                            │
│     └─→ selectPlan('monthly') or purchaseSingleLesson(dayNum)    │
│                                                                   │
│  3. CHECKOUT PANEL OPENS (In-App)                                │
│     └─→ Stripe Embedded Checkout (never leaves Kelly)            │
│     └─→ Rate limited: 10 attempts / 15 min per email             │
│     └─→ Idempotency keys prevent duplicate charges               │
│                                                                   │
│  4. STRIPE PROCESSES PAYMENT                                     │
│     └─→ 7-day trial for subscriptions                            │
│     └─→ Automatic tax calculation                                │
│     └─→ Promo codes supported                                    │
│                                                                   │
│  5. WEBHOOK CONFIRMS                                             │
│     └─→ /api/webhooks/stripe-revenue                             │
│     └─→ Updates user record in Supabase                          │
│     └─→ Records revenue event for analytics                      │
│     └─→ Sends gift email (if applicable)                         │
│                                                                   │
│  6. ACCESS GRANTED                                               │
│     └─→ User returned to lesson                                  │
│     └─→ subscription_status = 'active' or purchase recorded      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### API Endpoints Involved

| Endpoint | Purpose |
|----------|---------|
| `POST /api/stripe-checkout` | Redirect checkout (from pricing page) |
| `POST /api/create-checkout` | Embedded checkout (in-app) |
| `POST /api/create-gift-checkout` | Gift purchases |
| `POST /api/lesson-purchase` | Single lesson purchases |
| `POST /api/webhooks/stripe-revenue` | Stripe event processing |
| `POST /api/create-portal-session` | Billing management |
| `POST /api/cancel-subscription` | Cancel at period end |
| `POST /api/pause-subscription` | Pause for 1-3 months |
| `POST /api/resume-subscription` | Resume paused subscription |

---

## 📍 WHERE Does Payment Happen?

### Payment Touchpoints in the App

| Location | How Payment Appears |
|----------|---------------------|
| **Paywall overlay** | Modal with single-lesson and subscription options |
| **Pricing page** | `/pricing.html` → redirects to `/learn.html?checkout={plan}` |
| **Gifts page** | `/gifts.html` → gift checkout flow |
| **Settings → Account** | Manage subscription, billing portal |
| **Journey calendar** | Tapping locked lesson shows paywall |

### User Journey to Payment

```
Organic Discovery:
  Homepage → Today's Lesson (included) → Love it → 
  Try past lesson → Paywall → Subscribe

Pricing Page:
  Google "Curious Kelly pricing" → /pricing.html →
  Select plan → Embedded checkout → Welcome page

Gift Flow:
  /gifts.html → Select duration → Enter recipient →
  Stripe checkout → Gift email sent
```

---

## 🌍 INTERNATIONAL: What Happens in Other Countries?

### Current Implementation

| Feature | Status | Details |
|---------|--------|---------|
| **Currency** | ❌ USD only | All prices in USD |
| **Tax** | ✅ Automatic | Stripe Tax enabled, calculates VAT/GST |
| **PPP (Purchasing Power Parity)** | ❌ Not implemented | No regional pricing |
| **Language** | ⚠️ Partial | UI in English, lessons in EN/ES/PT |

### How Stripe Tax Works

```javascript
// In checkout creation:
automatic_tax: { enabled: true },
billing_address_collection: 'auto',
```

This means:
- Stripe detects customer's country from billing address
- Automatically calculates and adds applicable taxes (VAT, GST, etc.)
- You see net revenue, Stripe handles tax remittance (if configured)

### What International Users See

1. **Prices**: Always $7.99, $49.99, etc. (USD)
2. **Tax**: Added at checkout based on their country
3. **Payment methods**: Card only (Stripe handles currency conversion)
4. **Total**: USD price + local tax (e.g., $49.99 + €10.50 VAT = €56.49 equivalent)

### Recommended Improvements

| Improvement | Effort | Impact |
|-------------|--------|--------|
| Add EUR/GBP/CAD currencies | Medium | Higher conversion in those regions |
| Add PPP discounts (India, Brazil, etc.) | Medium | 3-5x more signups from developing countries |
| Add local payment methods (iDEAL, Bancontact) | Low | 10-20% lift in EU |
| Multi-language checkout | Low | Already supported by Stripe |

### PPP Pricing (Documented but not implemented)

From `PRICING_STRATEGY_BIBLE.md`:

```
| Region | Currency | Price Adjustment |
|--------|----------|------------------|
| USA | USD | Base price |
| Canada | CAD | +5% |
| UK | GBP | Parity |
| EU | EUR | Parity |
| Australia | AUD | +10% |
| India | INR | -50% (PPP) |
| Brazil | BRL | -40% (PPP) |
```

**Status**: Documented but NOT implemented in code. All checkouts use USD.

---

## ⚠️ CRITICAL ISSUES FOUND

### 1. Paywall is DISABLED

```javascript
// config.js
testingMode: true,
disablePaywall: true,  // ← EVERYONE GETS ALL LESSONS FOR FREE!
```

**Action Required:** Set both to `false` before production launch.

### 2. No Geographic Pricing

All users pay the same USD price regardless of location. A user in India pays the same $49.99 as a user in USA, despite ~4x difference in purchasing power.

### 3. Single Lesson Price Hardcoded

```javascript
// api/lesson-purchase.ts
const DEFAULT_LESSON_PRICE = 199; // $1.99 in cents
```

Not configurable via environment variables.

### 4. Currency Hardcoded

```javascript
currency: 'usd', // Everywhere
```

No multi-currency support despite Stripe supporting 135+ currencies.

---

## 📊 Payment Analytics Tracked

| Event | When Tracked |
|-------|--------------|
| `paywall.shown` | User sees paywall |
| `paywall.dismissed` | User closes paywall without action |
| `checkout.started` | User begins checkout flow |
| `purchase.completed` | Webhook confirms payment |
| `subscription_created` | New subscription started |
| `subscription_cancelled` | Subscription ended |
| `trial_converted` | Trial became paid |
| `payment_failed` | Charge failed |
| `refund_issued` | Refund processed |

---

## 🔒 Security Measures

| Measure | Status |
|---------|--------|
| Server-side price validation | ✅ |
| Webhook signature verification | ✅ |
| Idempotency keys | ✅ |
| Rate limiting | ✅ (10/15min) |
| No stack traces in production | ✅ |
| Token-based auth for portal | ✅ |

---

## 📋 Recommended Actions

### Before Launch (Critical)

- [ ] Set `disablePaywall: false` in `config.js`
- [ ] Set `testingMode: false` in `config.js`
- [ ] Test payment flow end-to-end in production

### Short-term (Week 1-2)

- [ ] Add EUR and GBP as additional currencies
- [ ] Make single lesson price configurable via env var
- [ ] Add payment method options for EU (iDEAL, etc.)

### Medium-term (Month 1)

- [ ] Implement PPP pricing for India, Brazil, etc.
- [ ] Add annual price anchoring on paywall
- [ ] A/B test paywall delay (5s vs 10s vs instant)

---

*Audit completed: December 17, 2025*
