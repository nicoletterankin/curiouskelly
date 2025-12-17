# Stripe Webhook Audit & Configuration

**Audit Date:** December 17, 2025  
**Status:** ✅ Audited

---

## Active Webhook Endpoint

### Primary Production Endpoint

**URL:** `https://www.curiouskelly.com/api/webhooks/stripe-revenue`  
**Handler:** `api/webhooks/stripe-revenue.ts`

This is the **ONLY** webhook endpoint that should be registered in Stripe Dashboard.

---

## Events to Subscribe

Register these events in Stripe Dashboard → Webhooks:

### Critical (Required for Core Functionality)

| Event | Purpose | Implemented |
|-------|---------|-------------|
| `checkout.session.completed` | Create user account, grant access, process gifts | ✅ |
| `customer.subscription.created` | Record subscription start, grant trial access | ✅ |
| `customer.subscription.updated` | Handle plan changes, trial conversions | ✅ |
| `customer.subscription.deleted` | Revoke access, trigger win-back | ✅ |
| `invoice.payment_succeeded` | Record payment, extend access | ✅ |
| `invoice.payment_failed` | Trigger dunning emails | ✅ |
| `charge.refunded` | Handle refunds, clawback commissions | ✅ |

### Recommended (For Better UX)

| Event | Purpose | Implemented |
|-------|---------|-------------|
| `invoice.upcoming` | Send renewal reminder emails | ✅ |
| `customer.updated` | Sync customer email/name changes | ✅ |
| `charge.dispute.created` | Alert team of chargebacks | ✅ |

---

## Deprecated/Unused Webhook Handlers

The following files exist but should **NOT** be registered as webhook endpoints:

| File | Reason |
|------|--------|
| `functions/handlers/stripe-webhook.ts` | Platform-agnostic handler template, not deployed |
| `daily-lesson-marketing/src/pages/api/stripe-webhook.ts` | Legacy Astro implementation, not in use |
| `api-disabled/stripe-webhook.ts` | Explicitly disabled |

**Action Required:** Verify in Stripe Dashboard that only `api/webhooks/stripe-revenue` is registered.

---

## Webhook Configuration Checklist

### In Stripe Dashboard

- [ ] Navigate to **Developers → Webhooks**
- [ ] Verify only ONE endpoint is active: `https://www.curiouskelly.com/api/webhooks/stripe-revenue`
- [ ] Click the endpoint → Events → Ensure all events from above are selected
- [ ] Copy the **Signing secret** (starts with `whsec_`)
- [ ] Verify this matches `STRIPE_WEBHOOK_SECRET` in Vercel environment

### In Vercel

- [ ] `STRIPE_WEBHOOK_SECRET` is set and matches Stripe Dashboard
- [ ] Environment variable is set for Production, Preview, and Development

---

## Webhook Handler Features

The primary handler (`api/webhooks/stripe-revenue.ts`) includes:

### Revenue Tracking
- Records all events to `revenue_events` table
- Calculates MRR impact for subscription changes
- Tracks trial starts and conversions

### User Sync
- Updates `users` table with subscription status
- Syncs `stripe_customer_id`, `subscription_tier`, `current_period_end`
- Handles `cancel_at_period_end` flag

### Commission System (Earn to Learn)
- Records commissions for referred users
- Handles clawbacks on refunds
- Updates referrer's `pending_earnings` and `lifetime_earnings`

### Gift Codes
- Generates 12-character alphanumeric codes
- Stores in `gift_codes` table
- Tracks gifter, recipient, and redemption status

---

## Testing Webhooks Locally

```bash
# Install Stripe CLI
# Windows: scoop install stripe
# Mac: brew install stripe/stripe-cli/stripe

# Login to Stripe
stripe login

# Forward webhooks to local server
stripe listen --forward-to localhost:3000/api/webhooks/stripe-revenue

# In another terminal, trigger test events
stripe trigger checkout.session.completed
stripe trigger customer.subscription.created
stripe trigger invoice.payment_succeeded
```

---

## Troubleshooting

### Signature Verification Failed

1. Check that `STRIPE_WEBHOOK_SECRET` matches the signing secret in Stripe Dashboard
2. Ensure the webhook handler receives the raw request body (not parsed JSON)
3. Verify `config.api.bodyParser = false` is set

### Events Not Being Received

1. Check Stripe Dashboard → Webhooks → Your endpoint → Recent attempts
2. Look for failed deliveries and their error messages
3. Verify the endpoint URL is correct and publicly accessible

### Duplicate Events

1. Check if multiple webhook endpoints are registered
2. Implement idempotency using `event.id` as a key
3. Check for retry logic causing duplicate processing

---

## Security Considerations

1. **Always verify signatures** - Never process events without verification
2. **Use HTTPS** - Stripe only sends to HTTPS endpoints
3. **Log sensitively** - Don't log full credit card details
4. **Handle failures gracefully** - Return 200 quickly, process async if needed
5. **Idempotency** - Handle the same event being delivered multiple times

---

## Monitoring & Alerts

Set up alerts for:

- Webhook delivery failures (Stripe Dashboard)
- `charge.dispute.created` events (immediate notification)
- High rate of `invoice.payment_failed` events
- Unexpected event types

---

*Last updated: December 17, 2025*
