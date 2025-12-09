# 🔒 CURIOUS KELLY — LOCKED PRICING
## Single Source of Truth | Effective December 17, 2025

---

## ⚠️ THIS IS THE CANONICAL PRICING DOCUMENT

**All pricing across the codebase MUST match this document.**  
**Any deviation is a bug.**

---

## 📊 SUBSCRIPTION PRICING

| Plan | Price | Interval | Stripe Amount (cents) |
|------|-------|----------|----------------------|
| **Monthly** | $7.99 | /month | 799 |
| **Annual** | $49.99 | /year | 4999 |
| **Family** | $99.99 | /year | 9999 |
| **Lifetime** | $199.99 | one-time | 19999 |

---

## 🎁 GIFT PRICING

| Gift | Price | Duration | Stripe Amount (cents) |
|------|-------|----------|----------------------|
| **3 Months** | $24.99 | 3 months | 2499 |
| **6 Months** | $39.99 | 6 months | 3999 |
| **12 Months** | $49.99 | 12 months | 4999 |
| **Lifetime Gift** | $149.99 | forever | 14999 |

---

## 🚀 LAUNCH SPECIAL (Dec 17-24, 2025)

| Plan | Launch Price | Regular Price | Discount |
|------|--------------|---------------|----------|
| **Annual** | $39.99 | $49.99 | 20% off |

**Promo Code:** `FOUNDING` or auto-applied during launch week

---

## 💰 VALUE ANCHORS (Use in Marketing)

| Anchor | Calculation |
|--------|-------------|
| **Annual daily cost** | $0.14/day |
| **Annual monthly equiv** | $4.17/month |
| **Annual savings vs monthly** | 48% ($45.89/year) |
| **Family per-person** | $16.67/person/year (at 6 members) |

---

## 🏷️ STRIPE ENVIRONMENT VARIABLES

```env
# Primary Prices (Update these in Stripe Dashboard)
STRIPE_PRICE_MONTHLY=price_monthly_799
STRIPE_PRICE_ANNUAL=price_annual_4999
STRIPE_PRICE_FAMILY=price_family_9999
STRIPE_PRICE_LIFETIME=price_lifetime_19999

# Gift Prices
STRIPE_PRICE_GIFT_3MO=price_gift_3mo_2499
STRIPE_PRICE_GIFT_6MO=price_gift_6mo_3999
STRIPE_PRICE_GIFT_12MO=price_gift_12mo_4999
STRIPE_PRICE_GIFT_LIFETIME=price_gift_lifetime_14999

# Launch Special
STRIPE_PRICE_ANNUAL_FOUNDING=price_annual_founding_3999
```

---

## ✅ CHECKLIST: Files That Need This Pricing

### Critical (Customer-Facing):
- [ ] `public/pricing.html` — Main pricing page
- [ ] `public/gifts.html` — Gift purchase page
- [ ] `daily-lesson-marketing/src/pages/checkout.astro` — Checkout flow
- [ ] `public/index.html` — Homepage pricing mentions
- [ ] `scripts/create-launch-prices.ts` — Stripe price creation

### API/Backend:
- [ ] `api/stripe-checkout.ts` — Checkout handler
- [ ] `.env` files — Price IDs

### Documentation:
- [ ] `docs/billing/PRICING_STRATEGY_BIBLE.md`
- [ ] `CFO_FINANCIAL_ANALYSIS_DEC7_2025.md`
- [ ] `CHRISTMAS_LAUNCH_PLAN.md`

---

## 📋 COPY-PASTE PRICING BLOCKS

### HTML Pricing Card (Monthly)
```html
<div class="price-amount">$7.99</div>
<span class="price-period">/month</span>
```

### HTML Pricing Card (Annual)
```html
<div class="price-amount">$49.99</div>
<span class="price-period">/year</span>
<div class="price-savings">Save 48% ($4.17/mo)</div>
```

### HTML Pricing Card (Family)
```html
<div class="price-amount">$99.99</div>
<span class="price-period">/year</span>
<div class="price-savings">Up to 6 family members</div>
```

### HTML Pricing Card (Lifetime)
```html
<div class="price-amount">$199.99</div>
<span class="price-period">one-time</span>
```

### JavaScript/TypeScript Constants
```typescript
export const PRICING = {
  monthly: { amount: 799, display: '$7.99', interval: 'month' },
  annual: { amount: 4999, display: '$49.99', interval: 'year', savings: '48%' },
  family: { amount: 9999, display: '$99.99', interval: 'year' },
  lifetime: { amount: 19999, display: '$199.99', interval: 'one-time' },
  gift: {
    '3mo': { amount: 2499, display: '$24.99' },
    '6mo': { amount: 3999, display: '$39.99' },
    '12mo': { amount: 4999, display: '$49.99' },
    'lifetime': { amount: 14999, display: '$149.99' },
  }
} as const;
```

---

## 🔐 LOCKED BY

**Date:** December 7, 2025  
**Approved by:** CEO  
**Effective:** December 17, 2025 (Launch Day)

---

**DO NOT MODIFY PRICING WITHOUT UPDATING THIS DOCUMENT FIRST.**


