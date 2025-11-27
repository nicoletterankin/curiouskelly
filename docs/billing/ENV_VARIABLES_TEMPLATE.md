# 🔐 ENVIRONMENT VARIABLES FOR BILLING

## Required Variables for Stripe Integration

Copy these to your `.env.local` file or Vercel dashboard.

---

## STRIPE KEYS

```env
# Stripe API Keys (get from https://dashboard.stripe.com/apikeys)
STRIPE_SECRET_KEY=YOUR_STRIPE_SECRET_KEY_HERE
STRIPE_PUBLISHABLE_KEY=YOUR_STRIPE_PUBLISHABLE_KEY_HERE

# Webhook Secret (get from https://dashboard.stripe.com/webhooks)
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## PRICE IDS

After creating products in Stripe, add these:

```env
# Subscription Prices
STRIPE_PRICE_MONTHLY=price_xxxxxxxxxxxxx
STRIPE_PRICE_ANNUAL=price_xxxxxxxxxxxxx

# One-time Prices
STRIPE_PRICE_LIFETIME=price_xxxxxxxxxxxxx

# Gift Prices
STRIPE_PRICE_GIFT_12MO=price_xxxxxxxxxxxxx
STRIPE_PRICE_GIFT_6MO=price_xxxxxxxxxxxxx
STRIPE_PRICE_GIFT_3MO=price_xxxxxxxxxxxxx
```

---

## SITE CONFIGURATION

```env
# Your production URL
PUBLIC_SITE_URL=https://curiouskelly.com
```

---

## HOW TO GET THESE VALUES

### 1. Stripe API Keys

1. Go to https://dashboard.stripe.com/apikeys
2. Copy "Secret key" → `STRIPE_SECRET_KEY`
3. Copy "Publishable key" → `STRIPE_PUBLISHABLE_KEY`

### 2. Webhook Secret

1. Go to https://dashboard.stripe.com/webhooks
2. Click "Add endpoint"
3. URL: `https://curiouskelly.com/api/stripe-webhook`
4. Select events (see STRIPE_IMPLEMENTATION_REQUIREMENTS.md)
5. After creation, click to reveal signing secret
6. Copy → `STRIPE_WEBHOOK_SECRET`

### 3. Price IDs

1. Create products in Stripe Dashboard
2. For each price, copy the Price ID (starts with `price_`)
3. Add to corresponding env variable

---

## VERCEL DASHBOARD

Add these in: Vercel Dashboard → Your Project → Settings → Environment Variables

Set for:

- ✅ Production
- ✅ Preview
- ❌ Development (use test keys locally)

---

## LOCAL DEVELOPMENT

For local testing, use Stripe test keys:

```env
STRIPE_SECRET_KEY=YOUR_TEST_SECRET_KEY_HERE
STRIPE_PUBLISHABLE_KEY=YOUR_TEST_PUBLISHABLE_KEY_HERE
```

Use Stripe CLI for webhook testing:

```bash
stripe listen --forward-to localhost:3000/api/stripe-webhook
```

The CLI will give you a webhook secret for local testing.
