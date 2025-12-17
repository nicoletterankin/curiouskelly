# 🚀 Payment Setup - Do This Now

**Time needed:** ~10 minutes  
**Difficulty:** Easy (just clicking and pasting)

---

## Step 1: Verify Stripe Webhook (2 min)

### Open Stripe Webhooks
👉 **Click this link:** https://dashboard.stripe.com/webhooks

### Check Your Endpoint
Look for an endpoint with URL containing `/api/webhooks/stripe-revenue`

**If it exists:**
1. Click on it
2. Go to "Events to send" section
3. Make sure these events are checked:
   - ✅ `checkout.session.completed`
   - ✅ `customer.subscription.created`
   - ✅ `customer.subscription.updated`
   - ✅ `customer.subscription.deleted`
   - ✅ `invoice.payment_succeeded`
   - ✅ `invoice.payment_failed`
   - ✅ `invoice.upcoming` ← **ADD THIS**
   - ✅ `customer.updated` ← **ADD THIS**
   - ✅ `charge.refunded`
   - ✅ `charge.dispute.created` ← **ADD THIS**

**If it doesn't exist:**
1. Click "+ Add endpoint"
2. Endpoint URL: `https://www.curiouskelly.com/api/webhooks/stripe-revenue`
3. Select all events listed above
4. Click "Add endpoint"

### Copy the Signing Secret
1. Click on your endpoint
2. Click "Reveal" under "Signing secret"
3. Copy the `whsec_...` value
4. You'll need this for Step 3

---

## Step 2: Verify Price IDs (3 min)

### Open Stripe Products
👉 **Click this link:** https://dashboard.stripe.com/products

### Check These Products Exist with Correct Prices

| Product | Price | Type |
|---------|-------|------|
| Kelly+ Monthly | **$7.99**/month | Recurring |
| Kelly+ Annual | **$49.99**/year | Recurring |
| Kelly+ Family | **$99.99**/year | Recurring |
| Kelly+ Lifetime | **$199.99** | One-time |
| Gift - 3 Months | **$24.99** | One-time |
| Gift - 6 Months | **$39.99** | One-time |
| Gift - 12 Months | **$49.99** | One-time |
| Gift - Lifetime | **$149.99** | One-time |

### If Any Are Missing or Wrong
Use the Stripe Wizard at: https://curiouskelly.com/stripe-wizard.html

### Copy All Price IDs
For each product, click on it and copy the Price ID (starts with `price_...`)

You'll need these for Step 3:
- STRIPE_PRICE_MONTHLY = `price_...`
- STRIPE_PRICE_ANNUAL = `price_...`
- STRIPE_PRICE_FAMILY = `price_...`
- STRIPE_PRICE_LIFETIME = `price_...`
- STRIPE_PRICE_GIFT_3MO = `price_...`
- STRIPE_PRICE_GIFT_6MO = `price_...`
- STRIPE_PRICE_GIFT_12MO = `price_...`
- STRIPE_PRICE_GIFT_LIFETIME = `price_...`

---

## Step 3: Update Vercel Environment Variables (3 min)

### Open Vercel Settings
👉 **Click this link:** https://vercel.com/lotd/curiouskelly/settings/environment-variables

### Verify These Variables Are Set

**Required (check all exist):**

| Variable | Value starts with |
|----------|-------------------|
| `STRIPE_SECRET_KEY` | `sk_live_...` |
| `STRIPE_WEBHOOK_SECRET` | `whsec_...` (from Step 1) |
| `STRIPE_PRICE_MONTHLY` | `price_...` |
| `STRIPE_PRICE_ANNUAL` | `price_...` |
| `STRIPE_PRICE_FAMILY` | `price_...` |
| `STRIPE_PRICE_LIFETIME` | `price_...` |
| `STRIPE_PRICE_GIFT_3MO` | `price_...` |
| `STRIPE_PRICE_GIFT_6MO` | `price_...` |
| `STRIPE_PRICE_GIFT_12MO` | `price_...` |
| `STRIPE_PRICE_GIFT_LIFETIME` | `price_...` |

**Optional (for email sending):**

| Variable | Purpose |
|----------|---------|
| `RESEND_API_KEY` | Gift & reminder emails (get from https://resend.com) |

### To Add Missing Variables
1. Click "Add New"
2. Enter the variable name
3. Paste the value
4. Select "Production", "Preview", and "Development"
5. Click "Save"

---

## Step 4: Redeploy (1 min)

### Open Vercel Deployments
👉 **Click this link:** https://vercel.com/lotd/curiouskelly/deployments

### Redeploy Latest
1. Find the most recent deployment
2. Click the "..." menu
3. Click "Redeploy"
4. Wait for deployment to complete (~1-2 min)

---

## Step 5: Quick Test (2 min)

### Test Checkout Page
👉 Visit: https://curiouskelly.com/pricing.html

1. Click on "Start Learning" or any plan
2. Verify Stripe checkout loads
3. Don't complete purchase (unless you want to!)

### Test Webhook
👉 Go to: https://dashboard.stripe.com/webhooks

1. Click on your endpoint
2. Click "Send test webhook"
3. Select `checkout.session.completed`
4. Click "Send test webhook"
5. Verify it shows "200 OK"

---

## ✅ All Done!

Your payment system is now fully configured with:

- ✅ Locked pricing ($7.99, $49.99, $99.99, $199.99)
- ✅ Secure webhook handling
- ✅ Rate limiting protection
- ✅ Gift email automation
- ✅ Subscription pause/resume
- ✅ Renewal reminders

---

## Quick Links Reference

| What | Link |
|------|------|
| Stripe Dashboard | https://dashboard.stripe.com |
| Stripe Webhooks | https://dashboard.stripe.com/webhooks |
| Stripe Products | https://dashboard.stripe.com/products |
| Stripe API Keys | https://dashboard.stripe.com/apikeys |
| Vercel Dashboard | https://vercel.com/lotd/curiouskelly |
| Vercel Env Vars | https://vercel.com/lotd/curiouskelly/settings/environment-variables |
| Resend (for emails) | https://resend.com/api-keys |

---

*Created: December 17, 2025*
