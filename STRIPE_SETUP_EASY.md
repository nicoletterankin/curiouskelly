# Stripe Setup - Super Easy Guide ✅

**You already have Stripe set up!** Just need to connect it to your code.

---

## Step 1: Get Your Stripe Keys (2 minutes)

1. Go to: https://dashboard.stripe.com
2. Click: **Developers** → **API keys** (left sidebar)
3. Copy these two keys:
   - **Secret key** (starts with `sk_test_` or `sk_live_`)
   - **Publishable key** (starts with `pk_test_` or `pk_live_`) - you might not need this yet

---

## Step 2: Get Your Price IDs (3 minutes)

1. In Stripe dashboard, click: **Products** (left sidebar)
2. You need 4 products. For each one:
   - Click the product name
   - Click the **pricing** tab
   - Copy the **Price ID** (starts with `price_`)

**You need these 4 Price IDs:**
- ✅ Monthly subscription (e.g., $9.99/month)
- ✅ Annual subscription (e.g., $99.99/year)
- ✅ Family plan (e.g., $149.99/year) - optional
- ✅ Gift (one-time payment, e.g., $99.99) - optional

**Don't have all 4?** That's OK! Just create the ones you need:
- Click **+ Add product**
- Set name, price, billing period
- Copy the Price ID

---

## Step 3: Add Keys to Your Project (1 minute)

> **💡 TIP:** See `SECRETS_MASTER_REFERENCE.md` for a complete guide to all secrets, or `SECRETS_QUICK_REFERENCE.md` for a quick cheat sheet!

### Option A: If deploying to Vercel (Easiest)

1. Go to: https://vercel.com/dashboard
2. Click your project → **Settings** → **Environment Variables**
3. Add these one by one (click "Add"):

```
Name: STRIPE_SECRET_KEY
Value: sk_test_xxxxx (paste your secret key)
Environment: Production, Preview, Development (check all)

Name: STRIPE_WEBHOOK_SECRET
Value: whsec_xxxxx (paste your webhook signing secret)
Environment: Production, Preview, Development (check all)

Name: STRIPE_PRICE_MONTHLY
Value: price_xxxxx (paste monthly price ID)
Environment: Production, Preview, Development

Name: STRIPE_PRICE_ANNUAL
Value: price_xxxxx (paste annual price ID)
Environment: Production, Preview, Development

Name: STRIPE_PRICE_FAMILY
Value: price_xxxxx (paste family price ID - optional)
Environment: Production, Preview, Development

Name: STRIPE_PRICE_GIFT
Value: price_xxxxx (paste gift price ID - optional)
Environment: Production, Preview, Development

Name: PUBLIC_SITE_URL
Value: https://curiouskelly.com (or your domain)
Environment: Production, Preview, Development
```

4. Click **Save** after each one

### Option B: If using local `.env` file

**Easiest way:** Run the setup script:
```bash
# Windows PowerShell
.\scripts\setup-secrets.ps1

# Mac/Linux
chmod +x scripts/setup-secrets.sh
./scripts/setup-secrets.sh
```

**Or manually:**
1. Copy `.env.example` to `.env` (or run the script above)
2. Open `.env` file in your project root
3. Add these lines (fill in your actual values):

```bash
STRIPE_SECRET_KEY=sk_test_xxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxx
STRIPE_PRICE_MONTHLY=price_xxxxx
STRIPE_PRICE_ANNUAL=price_xxxxx
STRIPE_PRICE_FAMILY=price_xxxxx
STRIPE_PRICE_GIFT=price_xxxxx
PUBLIC_SITE_URL=https://curiouskelly.com
```

3. Replace `xxxxx` with your actual values
4. Save the file

---

## Step 4: Test It (2 minutes)

### If using Vercel:

1. **Deploy:**
   ```bash
   vercel --prod
   ```
   Or push to GitHub (if connected to Vercel)

2. **Test the endpoint:**
   - Go to: `https://your-site.vercel.app/api/stripe-checkout`
   - Or open `public/index.html` and click a "Buy" button
   - Should redirect to Stripe checkout!

### If testing locally:

1. **Install Vercel CLI** (if not installed):
   ```bash
   npm i -g vercel
   ```

2. **Run local dev server:**
   ```bash
   vercel dev
   ```

3. **Open:** http://localhost:3000
4. **Click:** A "Buy" button
5. **Should see:** Stripe checkout page!

---

## Step 5: Set Up Webhook (Optional - 5 minutes)

**Only needed if you want to track payments automatically**

1. In Stripe dashboard: **Developers** → **Webhooks**
2. Click: **Add endpoint**
3. **Endpoint URL:** `https://curiouskelly.com/api/stripe-webhook`
4. **Events to send:** Select these:
   - `checkout.session.completed`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
5. Click **Add endpoint**
6. Copy the **Signing secret** (starts with `whsec_`)
7. Add to environment variables:
   ```
   Name: STRIPE_WEBHOOK_SECRET
   Value: whsec_xxxxx
   ```

---

## ✅ That's It!

Your billing system is now connected! 

**To verify:**
1. Open your landing page (`public/index.html`)
2. Click "Annual Plan" or "Give as Gift"
3. Enter an email
4. Should redirect to Stripe checkout ✅

---

## Troubleshooting

### "Stripe not configured" error
- ✅ Check environment variables are set
- ✅ Make sure you're using the right environment (production vs preview)
- ✅ Restart your dev server if testing locally

### "Price ID not configured" error
- ✅ Check you added all price IDs to environment variables
- ✅ Make sure price IDs start with `price_`
- ✅ Verify products exist in Stripe dashboard

### Checkout doesn't open
- ✅ Check browser console for errors (F12)
- ✅ Verify `/api/stripe-checkout` endpoint exists
- ✅ Check network tab to see if API call succeeds

### Need help?
- Check `IMPLEMENTATION_COMPLETE.md` for detailed docs
- Stripe dashboard has test mode - use `sk_test_` keys for testing

---

## Quick Checklist

- [ ] Got Stripe secret key (`sk_test_` or `sk_live_`)
- [ ] Got webhook signing secret (`whsec_xxx`)
- [ ] Got Monthly price ID (`price_xxx`)
- [ ] Got Annual price ID (`price_xxx`)
- [ ] Got Family price ID (`price_xxx`) - optional
- [ ] Got Gift price ID (`price_xxx`) - optional
- [ ] Added all to environment variables (Vercel or `.env`)
- [ ] Added `PUBLIC_SITE_URL`
- [ ] Tested checkout flow
- [ ] Webhook set up with 7 events configured

**You're done! 🎉**

