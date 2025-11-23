# 🔑 Credentials Checklist

Use this checklist to track what credentials you have and what you still need.

---

## ✅ Customer.io (Email Automation)

- [x] **Account created:** "Lesson of the Day, PBC"
- [x] **Site ID:** `9ea8fc826910bbd745a3`
- [x] **API Key (Track API):** `d6da47e97fd693615271`
- [ ] **App API Key:** _________________________
  - Get from: https://fly.customer.io/settings/api_credentials
  - Click: "Create App API Key"
  - Name: "The Daily Lesson Website"

### Domain Verification
- [ ] **Domain verified in Customer.io:** curiouskelly.com
- [ ] **DNS records added to Cloudflare**
- [ ] **Test email sent successfully**

---

## ✅ Stripe (Payment Processing)

- [x] **Account created:** "Lessonofthedsy"
- [x] **Test mode enabled**
- [ ] **Publishable Key (Test):** pk_test_________________________
  - Get from: https://dashboard.stripe.com/test/apikeys
- [ ] **Secret Key (Test):** sk_test_________________________
  - Get from: https://dashboard.stripe.com/test/apikeys

### Products Created
- [ ] **Monthly Subscription**
  - Price: $4.99/month
  - Trial: 7 days
  - Price ID: price_________________________
  
- [ ] **Annual Subscription**
  - Price: $49.99/year
  - Trial: 7 days
  - Price ID: price_________________________
  
- [ ] **Gift Subscription**
  - Price: $49.99 one-time
  - No trial
  - Price ID: price_________________________

### Webhook
- [ ] **Webhook endpoint created:** https://curiouskelly.com/api/stripe-webhook
- [ ] **Webhook secret:** whsec_________________________
  - Get from: Stripe Dashboard → Developers → Webhooks

---

## 🔜 Google Analytics (Optional)

- [ ] **Account created**
- [ ] **Property:** "The Daily Lesson"
- [ ] **Measurement ID:** G-________________________
  - Get from: https://analytics.google.com

---

## 🔜 Cloudflare (Already have account)

- [x] **Account ID:** `47ebb2a1adc311cb106acc89720e352c`
- [x] **Domain:** curiouskelly.com
- [ ] **Pages project created:** the-daily-lesson
- [ ] **Environment variables set** (in Cloudflare Pages)

### Turnstile (Anti-bot)
- [ ] **Site Key:** _________________________
- [ ] **Secret Key:** _________________________
  - Get from: Cloudflare Dashboard → Turnstile

---

## 📧 Email Addresses Needed

Set these up through your email provider (Google Workspace, etc.):

- [ ] **support@curiouskelly.com**
  - For customer support tickets
  - Set up auto-reply during setup
  
- [ ] **privacy@curiouskelly.com**
  - For GDPR/CCPA data requests
  - Required by law
  
- [ ] **press@curiouskelly.com**
  - For media inquiries
  - Nice to have, not critical

---

## 📝 .env File Status

- [ ] `.env` file created in `daily-lesson-marketing/`
- [ ] All Customer.io credentials added
- [ ] All Stripe credentials added
- [ ] All Stripe Price IDs added
- [ ] Webhook secret added
- [ ] File added to `.gitignore` (never commit!)

---

## ✅ WHEN EVERYTHING IS CHECKED

You're ready to:
1. Run the site locally: `npm run dev`
2. Test signup flow
3. Test payment processing
4. Test email sending
5. Deploy to production

---

## 🔐 SECURITY REMINDER

- **NEVER** commit the `.env` file to Git
- **NEVER** share credentials in screenshots/Slack/email
- **NEVER** use production keys in test environment
- **ALWAYS** use test keys during development

When ready for production:
1. Create new `.env.production` with live keys
2. Add to Cloudflare Pages environment variables
3. Switch Stripe to live mode
4. Re-test everything with small amounts

---

## 📞 WHERE TO GET HELP

- **Stripe:** https://support.stripe.com
- **Customer.io:** https://customer.io/docs
- **Cloudflare:** https://developers.cloudflare.com

---

**Last Updated:** November 17, 2025
**Status:** Customer.io ✅ | Stripe ⏳ | Analytics ⏳ | Cloudflare ⏳





