# ☀️ MORNING CHECKLIST — December 18, 2025

> **Time estimate:** 15-20 minutes  
> **Everything you need:** This doc + browser + Stripe dashboard

---

## 🔐 Step 1: Set Environment Variables (5 min)

### Open Vercel Dashboard
👉 **[Vercel Environment Variables](https://vercel.com/lotd/curiouskelly/settings/environment-variables)**

### Add These Variables:

| Variable | Value | How to Get It |
|----------|-------|---------------|
| `CRON_SECRET` | `ck_cron_2025_` + 20 random chars | Generate below |
| `SENDGRID_API_KEY` | `SG.xxxxx` | [SendGrid Dashboard](https://app.sendgrid.com/settings/api_keys) |
| `ADMIN_KEY` | `ck_admin_` + 16 random chars | Generate below |

### Generate Random Secrets:
```
CRON_SECRET:  ck_cron_2025_xxxxxxxxxxxxxxxxxx
ADMIN_KEY:    ck_admin_xxxxxxxxxxxxxxxx
```

Use: https://www.random.org/strings/?num=1&len=20&digits=on&loweralpha=on&unique=on&format=plain

---

## 💳 Step 2: Stripe Configuration (10 min)

### Open the Setup Suite
👉 **[Stripe Setup Suite](https://curiouskelly.com/admin/stripe-setup)**

### Quick Links (open all in tabs):
- [Stripe Dashboard](https://dashboard.stripe.com)
- [Stripe API Keys](https://dashboard.stripe.com/apikeys)
- [Stripe Webhooks](https://dashboard.stripe.com/webhooks)
- [Stripe Products](https://dashboard.stripe.com/products)

### Webhook URL to Configure:
```
https://www.curiouskelly.com/api/webhooks/stripe-revenue
```

### Events to Enable:
- [ ] `checkout.session.completed`
- [ ] `customer.subscription.created`
- [ ] `customer.subscription.updated`
- [ ] `customer.subscription.deleted`
- [ ] `invoice.payment_succeeded`
- [ ] `invoice.payment_failed`
- [ ] `invoice.upcoming`
- [ ] `charge.refunded`
- [ ] `charge.dispute.created`

### Products to Create (if not exists):

| Product | Price | Type |
|---------|-------|------|
| Kelly+ Monthly | $7.99/mo | Recurring |
| Kelly+ Annual | $49.99/yr | Recurring |
| Kelly+ Family | $99.99/yr | Recurring |
| Kelly+ Lifetime | $199.99 | One-time |
| Gift - 3 Months | $24.99 | One-time |
| Gift - 6 Months | $39.99 | One-time |
| Gift - 12 Months | $49.99 | One-time |
| Gift - Lifetime | $149.99 | One-time |

### Copy Price IDs to Vercel:
After creating products, copy each `price_xxx` ID into:
- The Stripe Setup Suite form (auto-exports to .env format)
- Paste into Vercel Environment Variables

---

## ✅ Step 3: Verify Everything (2 min)

### Test Health Endpoint:
👉 **[Operations Health](https://curiouskelly.com/api/health/operations)**

### Check Operations Dashboard:
👉 **[Operations Monitor](https://curiouskelly.com/admin/operations)**

### Test Checkout (use Stripe test mode):
👉 **[Pricing Page](https://curiouskelly.com/pricing)**

---

## 🔄 Step 4: Redeploy (1 min)

After setting env vars, trigger a redeploy:
👉 **[Vercel Deployments](https://vercel.com/lotd/curiouskelly/deployments)** → Redeploy latest

---

## 📧 What Happens Next

After completing this checklist:

| Time | What Happens |
|------|--------------|
| 9 AM | Escalation check runs (no email if no issues) |
| 8 PM | Happy learner digest (celebrates the day) |
| 11 PM | Streak check (logs milestones) |
| Sunday 6 PM | Weekly digest (full summary) |

**You're done.** The system runs itself now. 🚀

---

## 🆘 If Something Breaks

1. Check [Vercel Logs](https://vercel.com/lotd/curiouskelly/logs)
2. Check [Operations Monitor](https://curiouskelly.com/admin/operations)
3. Email arrives at nicoletterankin@gmail.com if it's serious

---

## ✨ Completion Signature

- [ ] Environment variables set
- [ ] Stripe webhook configured
- [ ] Products created with price IDs
- [ ] Health check passes
- [ ] Redeployed

**Date completed:** _______________
