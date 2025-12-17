# Payment Journey & Stripe Fixes - December 17, 2025

## Summary of Critical Fixes Applied

All 6 critical issues identified in the payment audit have been resolved.

---

## ✅ 1. Pricing Mismatch Fixed

**File:** `api/webhooks/stripe-revenue.ts`

**Before:**
```typescript
const PLAN_PRICES = {
  'monthly': { price_cents: 999, mrr_cents: 999 },        // $9.99 ❌
  'annual': { price_cents: 9999, mrr_cents: 833 },        // $99.99 ❌
  'lifetime': { price_cents: 29999, mrr_cents: 0 },       // $299.99 ❌
  // family was missing ❌
};
```

**After:**
```typescript
const PLAN_PRICES = {
  'monthly': { price_cents: 799, mrr_cents: 799 },        // $7.99 ✅
  'annual': { price_cents: 4999, mrr_cents: 417 },        // $49.99 ✅
  'family': { price_cents: 9999, mrr_cents: 833 },        // $99.99 ✅
  'lifetime': { price_cents: 19999, mrr_cents: 0 },       // $199.99 ✅
  'gift_3mo': { price_cents: 2499, mrr_cents: 0 },        // $24.99 ✅
  'gift_6mo': { price_cents: 3999, mrr_cents: 0 },        // $39.99 ✅
  'gift_12mo': { price_cents: 4999, mrr_cents: 0 },       // $49.99 ✅
  'gift_lifetime': { price_cents: 14999, mrr_cents: 0 },  // $149.99 ✅
};
```

**Canonical Pricing Source:** `docs/billing/PRICING_STRATEGY_BIBLE.md`

---

## ✅ 2. Family Plan Mode Fixed

**File:** `api/stripe-checkout.ts`

**Before:** Family was incorrectly grouped with Lifetime as a one-time payment.

**After:** Family is now correctly handled as a recurring subscription with:
- `mode: 'subscription'`
- 7-day trial period
- Proper MRR tracking

---

## ✅ 3. Webhook Audit Completed

**New Documentation:** `docs/billing/WEBHOOK_AUDIT.md`

**Active Endpoint:** `api/webhooks/stripe-revenue.ts` (ONLY this one)

**Deprecated/Unused:**
- `functions/handlers/stripe-webhook.ts` - Template, not deployed
- `daily-lesson-marketing/src/pages/api/stripe-webhook.ts` - Legacy
- `api-disabled/stripe-webhook.ts` - Disabled

**Action Required:** Verify in Stripe Dashboard that only one webhook endpoint is registered.

---

## ✅ 4. Missing Webhook Events Added

**File:** `api/webhooks/stripe-revenue.ts`

Added handlers for:

### `invoice.upcoming`
- Logs renewal reminders with days until due
- Prepares for email integration
- Records to `revenue_events` table

### `customer.updated`
- Syncs email changes to `users` table
- Updates `display_name` if changed
- Logs changed fields for audit

### `charge.dispute.created`
- Logs ALERT level for immediate visibility
- Records dispute details
- Prepares for Slack/email notification

---

## ✅ 5. Success URLs Standardized

| Checkout Type | Success URL | Rationale |
|---------------|-------------|-----------|
| Subscriptions (redirect) | `/welcome.html?session_id={...}` | Onboarding flow |
| Lifetime (redirect) | `/welcome.html?session_id={...}&plan=lifetime` | Onboarding |
| Gifts (redirect) | `/gift-success.html?session_id={...}` | Shows gift code |
| Embedded (in-app) | `/learn.html?checkout=success` | Stay in-app |
| Gift Embedded | `/gift-success.html?session_id={...}` | Fixed: was `/learn.html` |

---

## ✅ 6. Stack Traces Removed from Production

**Files:** 
- `api/stripe-checkout.ts`
- `api/create-checkout.ts`

**Before:**
```typescript
return res.status(500).json({
  error: 'checkout_failed',
  message: error.message,
  stack: error.stack  // ❌ Security risk
});
```

**After:**
```typescript
return res.status(500).json({
  error: 'checkout_failed',
  message: process.env.NODE_ENV === 'development' && error instanceof Error 
    ? error.message 
    : 'An error occurred during checkout. Please try again.'
});
```

---

## Files Modified

1. `api/webhooks/stripe-revenue.ts` - Pricing + new events
2. `api/stripe-checkout.ts` - Family mode + error handling
3. `api/create-checkout.ts` - Error handling
4. `api/create-gift-checkout.ts` - Success URL

## Files Created

1. `docs/billing/WEBHOOK_AUDIT.md` - Webhook documentation
2. `docs/billing/PAYMENT_AUDIT_FIXES_DEC17.md` - This file

---

## Additional Improvements (Also Completed)

All recommended improvements were also implemented:

| # | Improvement | Status | Files |
|---|-------------|--------|-------|
| 7 | **Idempotency Keys** | ✅ | All checkout files |
| 8 | **Stripe API Version** | ✅ | All API files → `2024-11-20.acacia` |
| 9 | **Gift Email Sending** | ✅ | `api/lib/email.ts`, webhook |
| 10 | **Rate Limiting** | ✅ | `api/lib/rate-limit.ts`, checkout files |
| 11 | **Pause/Resume Endpoints** | ✅ | New endpoints created |

### Idempotency Keys
- Added to all checkout creation calls
- Uses 5-minute time windows to allow retries
- Format: `{type}_{email}_{plan}_{timestamp}`

### Stripe API Version
- Standardized all files to `2024-11-20.acacia`
- Previously was inconsistent `2023-10-16` vs `2024-11-20.acacia`

### Gift Email Sending
- Created `api/lib/email.ts` with Resend integration
- Beautiful HTML templates for gift codes
- Auto-sends on checkout.session.completed webhook
- Falls back to console logging if RESEND_API_KEY not set

### Rate Limiting
- Created `api/lib/rate-limit.ts` with in-memory store
- Checkout: 10 attempts per email per 15 minutes
- Gift checkout: 5 attempts per email per 15 minutes
- Returns proper 429 with Retry-After headers

### Subscription Pause/Resume
- Created `api/pause-subscription.ts` - Pause for 1-3 months
- Created `api/resume-subscription.ts` - Resume paused subscription
- Uses Stripe's `pause_collection` feature
- Records events to `revenue_events` table

---

## New Files Created

| File | Purpose |
|------|---------|
| `docs/billing/WEBHOOK_AUDIT.md` | Webhook documentation |
| `docs/billing/PAYMENT_AUDIT_FIXES_DEC17.md` | This changelog |
| `api/lib/email.ts` | Email sending utilities (Resend) |
| `api/lib/rate-limit.ts` | Rate limiting utilities |
| `api/pause-subscription.ts` | Pause subscription endpoint |
| `api/resume-subscription.ts` | Resume subscription endpoint |

---

## New Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `RESEND_API_KEY` | Optional | Gift/reminder emails (falls back to console log) |

---

## Verification Checklist

Before launch, verify:

- [ ] All Stripe price IDs match locked pricing ($7.99, $49.99, $99.99, $199.99)
- [ ] Only ONE webhook endpoint in Stripe Dashboard
- [ ] `STRIPE_WEBHOOK_SECRET` matches signing secret in Stripe
- [ ] Test checkout flow for each plan type
- [ ] Test gift checkout flow
- [ ] Verify welcome.html and gift-success.html pages work

---

*Fixes applied: December 17, 2025*
