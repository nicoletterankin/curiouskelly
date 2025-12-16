# 🔴 YOUR STRIPE SETUP — DO THIS NOW

**Time needed:** ~15 minutes  
**This must be done before launch.**

---

## 🎮 USE THE FUN WIZARD!

**Open this page and follow the steps:**

### 👉 https://www.curiouskelly.com/stripe-wizard.html

It will guide you through everything and generate your environment variables automatically!

---

## 📍 OR DO IT MANUALLY

1. Open **Stripe Dashboard**: https://dashboard.stripe.com
2. Make sure you're in **LIVE MODE** (toggle in top-left, should say "Live")

---

## STEP 1: Create Products & Prices

Go to **Products** → **+ Add product**

### Product 1: Kelly+ Monthly
- **Name:** `Kelly+ Monthly`
- **Description:** `Monthly subscription to Curious Kelly`
- **Pricing:** `$7.99` / `month` / Recurring
- **After creation:** Copy the Price ID (starts with `price_`)
- **Write it here:** `STRIPE_PRICE_MONTHLY = ____________________`

### Product 2: Kelly+ Annual
- **Name:** `Kelly+ Annual`
- **Description:** `Annual subscription to Curious Kelly - Best Value`
- **Pricing:** `$49.99` / `year` / Recurring
- **Copy Price ID:** `STRIPE_PRICE_ANNUAL = ____________________`

### Product 3: Kelly+ Lifetime
- **Name:** `Kelly+ Lifetime`
- **Description:** `Lifetime access to Curious Kelly`
- **Pricing:** `$199.99` / One time
- **Copy Price ID:** `STRIPE_PRICE_LIFETIME = ____________________`

### Product 4: Kelly+ Family
- **Name:** `Kelly+ Family`
- **Description:** `Family annual subscription - up to 6 members`
- **Pricing:** `$99.99` / `year` / Recurring
- **Copy Price ID:** `STRIPE_PRICE_FAMILY = ____________________`

---

## STEP 2: Create Gift Products

### Gift 1: 3-Month Gift
- **Name:** `Kelly Gift - 3 Months`
- **Pricing:** `$24.99` / One time
- **Copy Price ID:** `STRIPE_PRICE_GIFT_3MO = ____________________`

### Gift 2: 6-Month Gift
- **Name:** `Kelly Gift - 6 Months`
- **Pricing:** `$39.99` / One time
- **Copy Price ID:** `STRIPE_PRICE_GIFT_6MO = ____________________`

### Gift 3: 12-Month Gift
- **Name:** `Kelly Gift - 12 Months`
- **Pricing:** `$49.99` / One time
- **Copy Price ID:** `STRIPE_PRICE_GIFT_12MO = ____________________`

### Gift 4: Lifetime Gift
- **Name:** `Kelly Gift - Lifetime`
- **Pricing:** `$149.99` / One time
- **Copy Price ID:** `STRIPE_PRICE_GIFT_LIFETIME = ____________________`

---

## STEP 3: Create Launch Promo (Optional)

### Product: Kelly+ Annual (Founding)
- **Name:** `Kelly+ Annual - Founding Member`
- **Pricing:** `$39.99` / `year` / Recurring
- **Copy Price ID:** `STRIPE_PRICE_ANNUAL_FOUNDING = ____________________`

OR create a **Coupon**:
- Go to **Products** → **Coupons** → **+ Create coupon**
- **Name:** `FOUNDING`
- **Type:** Percentage off
- **Discount:** `20%`
- **Duration:** Once
- **Redemption limits:** Optional end date Dec 24, 2025

---

## STEP 4: Set Up Webhook

Go to **Developers** → **Webhooks** → **+ Add endpoint**

- **Endpoint URL:** `https://www.curiouskelly.com/api/webhooks/stripe-revenue`
- **Events to listen for:**
  - `checkout.session.completed`
  - `customer.subscription.created`
  - `customer.subscription.updated`
  - `customer.subscription.deleted`
  - `invoice.paid`
  - `invoice.payment_failed`

After creating:
- Click the webhook
- Click **Reveal** under Signing secret
- **Copy it:** `STRIPE_WEBHOOK_SECRET = ____________________`

---

## STEP 5: Get Your Keys

Go to **Developers** → **API keys**

- **Publishable key** (starts with `pk_live_`): 
  `STRIPE_PUBLISHABLE_KEY = ____________________`

- **Secret key** (starts with `sk_live_`):
  `STRIPE_SECRET_KEY = ____________________`

---

## STEP 6: Add to Vercel

Go to **Vercel Dashboard** → **curiouskelly** → **Settings** → **Environment Variables**

Add these (if not already present):

| Variable | Value |
|----------|-------|
| `STRIPE_SECRET_KEY` | sk_live_... |
| `STRIPE_PUBLISHABLE_KEY` | pk_live_... |
| `STRIPE_WEBHOOK_SECRET` | whsec_... |
| `STRIPE_PRICE_MONTHLY` | price_... |
| `STRIPE_PRICE_ANNUAL` | price_... |
| `STRIPE_PRICE_LIFETIME` | price_... |
| `STRIPE_PRICE_FAMILY` | price_... |
| `STRIPE_PRICE_GIFT_3MO` | price_... |
| `STRIPE_PRICE_GIFT_6MO` | price_... |
| `STRIPE_PRICE_GIFT_12MO` | price_... |
| `STRIPE_PRICE_GIFT_LIFETIME` | price_... |

**IMPORTANT:** After adding, click **Redeploy** on the latest deployment to apply changes.

---

## STEP 7: Test (REQUIRED)

After setup, test these URLs:

1. **Regular checkout:**
   https://www.curiouskelly.com/learn.html?checkout=annual
   - Should show Stripe embedded checkout

2. **Gift checkout:**
   https://www.curiouskelly.com/learn.html?gift=12-month
   - Should show gift form then Stripe checkout

3. **Make a real $7.99 test purchase:**
   - Use your own card
   - Verify you receive email confirmation
   - Verify subscription shows in Stripe Dashboard
   - Cancel/refund after testing

---

## ✅ CHECKLIST

- [ ] Created all 4 subscription products in Stripe
- [ ] Created all 4 gift products in Stripe
- [ ] Set up webhook endpoint
- [ ] Added all env vars to Vercel
- [ ] Redeployed Vercel
- [ ] Tested checkout flow
- [ ] Made test purchase
- [ ] Verified webhook received in Stripe

---

## 🚨 IF SOMETHING BREAKS

1. Check Vercel Logs: https://vercel.com/lotd/curiouskelly → Logs
2. Check Stripe Webhook logs: Developers → Webhooks → Click endpoint → Recent events
3. Check browser console on checkout page

---

**Once you complete this, tell me and I'll verify the APIs respond correctly.**

