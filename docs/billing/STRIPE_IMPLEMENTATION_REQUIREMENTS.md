# 🔧 STRIPE IMPLEMENTATION REQUIREMENTS

## For: Payments/Engineering Team
## From: Revenue Operations
## Priority: HIGH - Required for December 17, 2025 Launch

---

## OVERVIEW

This document specifies exactly what needs to be configured in Stripe to support our pricing strategy. Reference the full `PRICING_STRATEGY_BIBLE.md` for business context.

---

## 1. STRIPE PRODUCTS TO CREATE

### 1.1 Subscription Products

```
┌─────────────────────────────────────────────────────────────────┐
│ PRODUCT: Curious Kelly Annual                                   │
│ ID: prod_curious_kelly_annual                                   │
├─────────────────────────────────────────────────────────────────┤
│ PRICES:                                                         │
│ ├── price_annual_standard                                       │
│ │   └── $99.99 USD / year (recurring)                          │
│ ├── price_annual_launch                                         │
│ │   └── $69.99 USD / year (recurring) - Launch promo           │
│ ├── price_annual_bf                                             │
│ │   └── $59.99 USD / year (recurring) - Black Friday           │
│ └── price_annual_student                                        │
│     └── $49.99 USD / year (recurring) - Student discount        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PRODUCT: Curious Kelly Monthly                                  │
│ ID: prod_curious_kelly_monthly                                  │
├─────────────────────────────────────────────────────────────────┤
│ PRICES:                                                         │
│ ├── price_monthly_standard                                      │
│ │   └── $9.99 USD / month (recurring)                          │
│ └── price_monthly_promo                                         │
│     └── $7.99 USD / month (recurring) - Promotional             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PRODUCT: Curious Kelly Lifetime                                 │
│ ID: prod_curious_kelly_lifetime                                 │
├─────────────────────────────────────────────────────────────────┤
│ PRICES:                                                         │
│ └── price_lifetime                                              │
│     └── $299.99 USD (one-time)                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Gift Products

```
┌─────────────────────────────────────────────────────────────────┐
│ PRODUCT: Curious Kelly Gift - 12 Months                         │
│ ID: prod_gift_12mo                                              │
├─────────────────────────────────────────────────────────────────┤
│ PRICES:                                                         │
│ └── price_gift_12mo                                             │
│     └── $99.99 USD (one-time)                                   │
│     └── Metadata: { "gift_duration_months": 12 }                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PRODUCT: Curious Kelly Gift - 6 Months                          │
│ ID: prod_gift_6mo                                               │
├─────────────────────────────────────────────────────────────────┤
│ PRICES:                                                         │
│ └── price_gift_6mo                                              │
│     └── $59.99 USD (one-time)                                   │
│     └── Metadata: { "gift_duration_months": 6 }                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ PRODUCT: Curious Kelly Gift - 3 Months                          │
│ ID: prod_gift_3mo                                               │
├─────────────────────────────────────────────────────────────────┤
│ PRICES:                                                         │
│ └── price_gift_3mo                                              │
│     └── $34.99 USD (one-time)                                   │
│     └── Metadata: { "gift_duration_months": 3 }                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. COUPONS TO CREATE

### 2.1 Launch Coupons (Create Before Dec 17)

| Coupon ID | Name | Type | Amount | Duration | Redeem By |
|-----------|------|------|--------|----------|-----------|
| `LAUNCH30` | Launch 30% Off | percent_off | 30% | once | Dec 24, 2025 |
| `NEWYEAR25` | New Year 25% Off | percent_off | 25% | once | Jan 7, 2026 |
| `CURIOUS15` | Curious 15% Off | percent_off | 15% | once | None (ongoing) |

### 2.2 Annual Event Coupons (Create Templates)

| Coupon ID | Name | Type | Amount | Duration | Redeem By |
|-----------|------|------|--------|----------|-----------|
| `BLACKFRIDAY` | Black Friday 40% Off | percent_off | 40% | once | Nov 30 (yearly) |
| `CYBERMONDAY` | Cyber Monday 40% Off | percent_off | 40% | once | Dec 3 (yearly) |
| `BACKTOSCHOOL` | Back to School 30% Off | percent_off | 30% | once | Aug 31 (yearly) |

### 2.3 Special Segment Coupons

| Coupon ID | Name | Type | Amount | Duration | Restrictions |
|-----------|------|------|--------|----------|--------------|
| `STUDENT40` | Student Discount | percent_off | 40% | forever | Verified students only |
| `TEACHER50` | Teacher Discount | percent_off | 50% | forever | Verified educators only |
| `LIBRARY100` | Library Free Access | percent_off | 100% | forever | Verified libraries only |

### 2.4 Affiliate Coupon Template

For each approved affiliate, create:

```
Coupon ID: [AFFILIATE_CODE]
Name: [Affiliate Name] Discount
Type: percent_off
Amount: 10-25% (based on tier)
Duration: once
Metadata: {
  "affiliate_id": "[affiliate_user_id]",
  "affiliate_name": "[name]",
  "commission_rate": "0.20"
}
```

---

## 3. PROMOTION CODES

### 3.1 Structure

Each coupon should have a corresponding promotion code:

```
Coupon: LAUNCH30
└── Promotion Code: LAUNCH30
    ├── Code: "LAUNCH30"
    ├── Max Redemptions: null (unlimited)
    ├── First Time Order: true
    ├── Minimum Amount: null
    └── Expires: Dec 24, 2025 23:59:59 UTC
```

### 3.2 Promotion Code Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `first_time_order` | true | Prevent abuse |
| `max_redemptions` | null | No limit unless specified |
| `minimum_amount` | null | No minimum |
| `restrictions.currency` | USD | Primary currency |

---

## 4. CHECKOUT SESSION CONFIGURATION

### 4.1 Required Metadata on Every Session

```javascript
const session = await stripe.checkout.sessions.create({
  // ... other config
  metadata: {
    // Attribution
    source: 'web', // web | ios | android | api
    utm_source: req.query.utm_source || 'direct',
    utm_medium: req.query.utm_medium || 'none',
    utm_campaign: req.query.utm_campaign || 'none',
    
    // Affiliate tracking
    affiliate_code: req.query.ref || null,
    
    // Gift info (if applicable)
    is_gift: 'false', // 'true' | 'false'
    gift_recipient_email: null,
    gift_sender_name: null,
    gift_message: null,
    gift_delivery_date: null,
    
    // User context
    user_age: req.body.age || null,
    user_country: req.headers['cf-ipcountry'] || null,
  },
  
  // Enable promotion codes
  allow_promotion_codes: true,
  
  // Or apply specific code
  discounts: promoCode ? [{
    promotion_code: promoCode
  }] : [],
  
  // Subscription data for recurring
  subscription_data: {
    metadata: {
      // Same metadata for subscription record
    }
  }
});
```

### 4.2 Success/Cancel URLs

```javascript
success_url: 'https://curiouskelly.com/welcome?session_id={CHECKOUT_SESSION_ID}',
cancel_url: 'https://curiouskelly.com/pricing?canceled=true',
```

### 4.3 Customer Creation

```javascript
customer_creation: 'always', // Always create Stripe customer
customer_email: userEmail, // Pre-fill if known
```

---

## 5. WEBHOOK ENDPOINTS

### 5.1 Required Webhooks

Create webhook endpoint: `https://curiouskelly.com/api/stripe-webhook`

Subscribe to these events:

| Event | Priority | Action |
|-------|----------|--------|
| `checkout.session.completed` | CRITICAL | Create user account, grant access |
| `customer.subscription.created` | CRITICAL | Record subscription start |
| `customer.subscription.updated` | HIGH | Handle plan changes |
| `customer.subscription.deleted` | CRITICAL | Revoke access, trigger win-back |
| `invoice.paid` | HIGH | Record payment, extend access |
| `invoice.payment_failed` | HIGH | Send dunning email |
| `invoice.upcoming` | MEDIUM | Send renewal reminder |
| `customer.updated` | LOW | Sync customer data |
| `charge.refunded` | HIGH | Handle refund, adjust access |
| `charge.dispute.created` | HIGH | Alert team, gather evidence |

### 5.2 Webhook Handler Structure

```javascript
// /api/stripe-webhook.ts

import Stripe from 'stripe';

export async function POST(req: Request) {
  const sig = req.headers.get('stripe-signature');
  const body = await req.text();
  
  let event: Stripe.Event;
  
  try {
    event = stripe.webhooks.constructEvent(
      body,
      sig,
      process.env.STRIPE_WEBHOOK_SECRET
    );
  } catch (err) {
    return new Response('Webhook signature verification failed', { status: 400 });
  }
  
  switch (event.type) {
    case 'checkout.session.completed':
      await handleCheckoutComplete(event.data.object);
      break;
    case 'customer.subscription.deleted':
      await handleSubscriptionCanceled(event.data.object);
      break;
    // ... other handlers
  }
  
  return new Response('OK', { status: 200 });
}
```

---

## 6. BILLING PORTAL CONFIGURATION

### 6.1 Enable Customer Portal

In Stripe Dashboard → Settings → Billing → Customer Portal:

| Setting | Value |
|---------|-------|
| Allow customers to update payment methods | ✅ Yes |
| Allow customers to view invoice history | ✅ Yes |
| Allow customers to update subscriptions | ✅ Yes |
| Allow customers to cancel subscriptions | ✅ Yes |
| Allow customers to pause subscriptions | ✅ Yes (max 3 months) |

### 6.2 Cancellation Flow

Configure cancellation reasons:
- Too expensive
- Not using enough
- Found alternative
- Missing features
- Technical issues
- Other

Enable:
- Offer pause instead of cancel
- Offer discount to retain (20% off next 3 months)

### 6.3 Portal Link Generation

```javascript
const portalSession = await stripe.billingPortal.sessions.create({
  customer: customerId,
  return_url: 'https://curiouskelly.com/account',
});

// Redirect user to: portalSession.url
```

---

## 7. TAX CONFIGURATION

### 7.1 Enable Stripe Tax

1. Dashboard → Settings → Tax
2. Enable automatic tax collection
3. Set origin address (company HQ)
4. Enable tax for all products

### 7.2 Tax Behavior

```javascript
// In checkout session
automatic_tax: {
  enabled: true,
},
tax_id_collection: {
  enabled: true, // Allow business tax ID entry
},
```

### 7.3 Tax-Exempt Handling

For verified non-profits and educational institutions:

```javascript
// Update customer
await stripe.customers.update(customerId, {
  tax_exempt: 'exempt', // 'none' | 'exempt' | 'reverse'
});
```

---

## 8. TESTING CHECKLIST

Before launch, test these scenarios:

### 8.1 Purchase Flows

- [ ] New annual subscription (no promo)
- [ ] New annual subscription with promo code
- [ ] New monthly subscription
- [ ] Lifetime purchase
- [ ] Gift purchase (immediate delivery)
- [ ] Gift purchase (future delivery)
- [ ] Student discount verification
- [ ] Affiliate link tracking

### 8.2 Subscription Management

- [ ] Upgrade monthly → annual
- [ ] Downgrade annual → monthly
- [ ] Cancel subscription
- [ ] Pause subscription
- [ ] Resume paused subscription
- [ ] Update payment method
- [ ] Apply promo to existing subscription

### 8.3 Edge Cases

- [ ] Failed payment → dunning emails
- [ ] Refund request → process refund
- [ ] Chargeback → dispute flow
- [ ] Expired promo code → error message
- [ ] Invalid promo code → error message
- [ ] Promo code already used → error message

### 8.4 Webhook Reliability

- [ ] Checkout complete → user created
- [ ] Subscription canceled → access revoked
- [ ] Payment failed → email sent
- [ ] All events logged to database

---

## 9. ENVIRONMENT VARIABLES

Required in production:

```env
# Stripe Keys
STRIPE_SECRET_KEY=sk_live_...
STRIPE_PUBLISHABLE_KEY=pk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Product IDs (after creation)
STRIPE_PRICE_ANNUAL=price_...
STRIPE_PRICE_MONTHLY=price_...
STRIPE_PRICE_LIFETIME=price_...
STRIPE_PRICE_GIFT_12MO=price_...
STRIPE_PRICE_GIFT_6MO=price_...
STRIPE_PRICE_GIFT_3MO=price_...

# Portal
STRIPE_PORTAL_CONFIG=bpc_...
```

---

## 10. LAUNCH READINESS CHECKLIST

### Before December 17, 2025:

- [ ] All products created in Stripe
- [ ] All prices created and tested
- [ ] Launch promo codes created (LAUNCH30, etc.)
- [ ] Webhook endpoint deployed and verified
- [ ] Customer portal configured
- [ ] Tax collection enabled
- [ ] Test transactions completed
- [ ] Refund policy documented
- [ ] Support team trained on billing

### Day of Launch:

- [ ] Verify webhook is receiving events
- [ ] Monitor first 10 transactions manually
- [ ] Check promo codes are working
- [ ] Verify emails are sending
- [ ] Dashboard showing real-time data

---

## CONTACTS

| Role | Responsibility |
|------|----------------|
| Engineering Lead | Stripe integration code |
| DevOps | Webhook deployment, secrets |
| Finance | Reconciliation, reporting |
| Support | Customer billing issues |
| CEO | Pricing decisions, approvals |

---

## QUESTIONS?

Contact: engineering@curiouskelly.com or revenue@curiouskelly.com

---

*Document Version: 1.0*
*Last Updated: November 26, 2025*

