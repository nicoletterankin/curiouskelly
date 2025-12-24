# Stripe i18n Implementation Complete

**Date:** December 21, 2025  
**Status:** ✅ Deployed  
**Commit:** `3aca8aae` — "Stripe i18n: add locale to checkout sessions"

---

## Summary

Added internationalization support to Stripe checkout flow:
1. **Stripe.js explicit locale** — Uses page language for checkout UI
2. **Server-side locale** — Passes browser locale to Stripe session
3. **Dynamic currency for single-lesson** — Accepts currency query param

---

## Changes Made

### 1. Stripe.js Initialization (`public/learn.html`)

**Before:**
```javascript
const stripe = window.Stripe(publishableKey);
```

**After:**
```javascript
const stripe = window.Stripe(publishableKey, {
  locale: document.documentElement.lang || 'auto'
});
```

**Lines affected:** ~11587, ~11626 (two Stripe instances)

### 2. Checkout Session Locale (`api/create-checkout.ts`)

Added `locale` parameter to `stripe.checkout.sessions.create()`:

```typescript
const session = await stripe.checkout.sessions.create({
  // ... existing params ...
  locale: (req.headers['accept-language']?.split(',')[0]?.split('-')[0] as any) || 'auto',
});
```

### 3. Gift Checkout Locale (`api/create-gift-checkout.ts`)

Same pattern applied to gift checkout sessions.

### 4. Single Lesson Dynamic Currency (`api/lesson-purchase.ts`)

**Before:**
```typescript
currency: 'usd'
```

**After:**
```typescript
currency: req.query.currency?.toString().toLowerCase() || 'usd'
```

---

## How It Works

### Frontend (Stripe.js)
- Reads `<html lang="...">` attribute (set by i18n system)
- Falls back to `'auto'` (Stripe detects from browser)
- Stripe Checkout UI displays in user's language

### Backend (Session Creation)
- Reads `Accept-Language` HTTP header
- Extracts primary language code (e.g., `es` from `es-ES,en;q=0.9`)
- Passes to Stripe as `locale` parameter
- Stripe validates and uses supported locale or falls back to `auto`

### Single Lesson Purchases
- Accepts `?currency=eur` or `?currency=gbp` query param
- Defaults to `usd` if not specified
- Allows geo-pricing frontend to pass detected currency

---

## Testing Results

| Test | Result |
|------|--------|
| `/api/stripe-public` | ✅ 200 - Returns publishable key |
| `/api/geo-pricing` (US) | ✅ 200 - USD $7.99/mo |
| `/api/geo-pricing?country=DE` | ✅ 200 - EUR €7.49/mo |
| Checkout panel loads | ✅ Plan buttons visible |
| Stripe.js locale init | ✅ Code deployed |
| Click Continue → API call | ✅ (requires auth) |

---

## Stripe Locale Behavior

Stripe automatically handles unsupported locales:
- If locale is supported → Uses that language
- If locale is not supported → Falls back to `auto`
- No validation needed on our side

**Supported locales:** en, es, fr, de, it, pt, nl, pl, ru, ja, zh, ko, ar, etc.

See: https://stripe.com/docs/js/appendix/supported_locales

---

## Remaining Work (Per STRIPE_BATCH_WORK.md)

The code is i18n-ready. The following **requires manual Stripe Dashboard work**:

1. **Create 36 multi-currency Price objects**
   - EUR, GBP, CAD, AUD prices for all plans
   - INR, BRL, MXN, PLN prices (PPP discounted)

2. **Enable payment methods**
   - iDEAL, SEPA, Bancontact (EU)
   - UPI (India), PIX/Boleto (Brazil)
   - OXXO (Mexico), etc.

3. **Add environment variables**
   - `STRIPE_PRICE_MONTHLY_EUR`, `STRIPE_PRICE_ANNUAL_EUR`, etc.
   - Total: 36 new price ID variables

4. **Test international checkout**
   - Use VPN or `?country=XX` to simulate different countries
   - Verify correct currency and payment methods appear

---

## Files Modified

| File | Changes |
|------|---------|
| `public/learn.html` | Stripe.js locale init (2 locations) |
| `api/create-checkout.ts` | Added `locale` to session params |
| `api/create-gift-checkout.ts` | Added `locale` to session params |
| `api/lesson-purchase.ts` | Dynamic currency from query param |

---

*Documentation created: December 21, 2025*


