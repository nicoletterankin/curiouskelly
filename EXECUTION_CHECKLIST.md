# ✅ Execution Checklist - Threat Mitigation
## Step-by-Step Actions to Execute TODAY

**Date:** December 2024  
**Priority:** 🔴 CRITICAL  
**Time Required:** 4-6 hours

---

## 🎯 PHASE 1: IMMEDIATE ACTIONS (Do First - 2 hours)

### ✅ Task 1: Domain & DNS Setup (30 minutes)

**Status:** ⏳ TODO

**Steps:**
1. [ ] Check if curiouskelly.com domain is owned
   - If yes: Skip to DNS configuration
   - If no: Purchase domain (Namecheap/Google Domains ~$15/year)

2. [ ] Configure DNS Records:
   ```
   Type: A
   Name: @
   Value: [Vercel IP - get from Vercel dashboard]
   TTL: 3600
   
   Type: CNAME
   Name: www
   Value: curiouskelly.com
   TTL: 3600
   ```

3. [ ] Wait for DNS propagation (24-48 hours)
   - Use `nslookup curiouskelly.com` to verify
   - Use `dig curiouskelly.com` to check DNS

4. [ ] Verify domain resolves
   - Visit http://curiouskelly.com (should show placeholder or error initially)

**Success Criteria:** ✅ Domain resolves, DNS configured

---

### ✅ Task 2: Email Infrastructure Setup (30 minutes)

**Status:** ⏳ TODO

**Option A: Cloudflare Email Routing (Recommended - Free)**
1. [ ] Go to cloudflare.com
2. [ ] Add site: curiouskelly.com
3. [ ] Follow wizard to change nameservers at registrar
4. [ ] Email → Email Routing → Get Started
5. [ ] Add destination: your-personal-email@gmail.com
6. [ ] Add route: hello@curiouskelly.com → your-personal-email@gmail.com
7. [ ] Test: Send email TO hello@curiouskelly.com
8. [ ] Verify: Email arrives in your inbox

**Option B: Google Workspace ($6/month)**
1. [ ] Go to workspace.google.com
2. [ ] Get Started → Enter domain: curiouskelly.com
3. [ ] Create admin account
4. [ ] Add MX records to DNS:
   ```
   Priority: 1
   Host: @
   Value: aspmx.l.google.com
   
   Priority: 5
   Host: @
   Value: alt1.aspmx.l.google.com
   
   (Add all Google MX records)
   ```
5. [ ] Create hello@curiouskelly.com account
6. [ ] Test: Send/receive emails

**Success Criteria:** ✅ hello@curiouskelly.com sends/receives emails

---

### ✅ Task 3: Social Media Account Creation (2 hours)

**Status:** ⏳ TODO

**Follow:** `DO_THIS_RIGHT_NOW.md` Hours 3-4

**Accounts to Create:**
1. [ ] Twitter/X (@CuriousKelly)
2. [ ] Instagram (@CuriousKellyAI)
3. [ ] YouTube (@CuriousKelly)
4. [ ] LinkedIn (Curious Kelly PBC)
5. [ ] TikTok (@CuriousKellyAI)
6. [ ] Discord (Curious Kelly Community)

**For Each Account:**
- [ ] Create account
- [ ] Set username/handle
- [ ] Upload profile picture
- [ ] Write bio
- [ ] Add website link
- [ ] Switch to Professional/Creator account
- [ ] Post first announcement

**Success Criteria:** ✅ All 6 accounts created, first post published

---

## 🎯 PHASE 2: CRITICAL SETUP (Do Next - 2 hours)

### ✅ Task 4: Stripe Account Setup (1 hour)

**Status:** ⏳ TODO

**Steps:**
1. [ ] Go to stripe.com
2. [ ] Create account (use hello@curiouskelly.com)
3. [ ] Complete business verification:
   - Business name: Curious Kelly PBC
   - Business type: Public Benefit Corporation
   - Tax ID: [Your EIN or SSN]
   - Address: [Your business address]
4. [ ] Activate account (may take 1-3 days)

5. [ ] Create Products:
   ```
   Product 1: Monthly Subscription
   - Name: Curious Kelly Monthly
   - Price: $9.99/month
   - Billing: Recurring
   - Get Price ID: price_xxxxx
   
   Product 2: Annual Subscription
   - Name: Curious Kelly Annual
   - Price: $99.99/year
   - Billing: Recurring
   - Get Price ID: price_xxxxx
   
   Product 3: Family Plan
   - Name: Curious Kelly Family
   - Price: $149.99/year
   - Billing: Recurring
   - Get Price ID: price_xxxxx
   
   Product 4: Gift Plan
   - Name: Curious Kelly Gift
   - Price: $99.99/year
   - Billing: One-time
   - Get Price ID: price_xxxxx
   ```

6. [ ] Get API Keys:
   - Test Secret Key: sk_test_xxxxx
   - Test Publishable Key: pk_test_xxxxx
   - Live Secret Key: sk_live_xxxxx (after activation)
   - Live Publishable Key: pk_live_xxxxx (after activation)

7. [ ] Configure Webhook:
   - Endpoint: https://curiouskelly.com/api/stripe-webhook
   - Events: payment_intent.succeeded, customer.subscription.created, etc.
   - Get Webhook Secret: whsec_xxxxx

8. [ ] Add to `.env`:
   ```
   STRIPE_SECRET_KEY=sk_test_xxxxx
   STRIPE_PUBLISHABLE_KEY=pk_test_xxxxx
   STRIPE_WEBHOOK_SECRET=whsec_xxxxx
   STRIPE_PRICE_MONTHLY=price_xxxxx
   STRIPE_PRICE_ANNUAL=price_xxxxx
   STRIPE_PRICE_FAMILY=price_xxxxx
   STRIPE_PRICE_GIFT=price_xxxxx
   ```

9. [ ] Test Payment:
   - Use test card: 4242 4242 4242 4242
   - Expiry: Any future date
   - CVC: Any 3 digits
   - Verify payment succeeds

**Success Criteria:** ✅ Stripe account active, products created, test payment works

---

### ✅ Task 5: Cost Monitoring Dashboard (30 minutes)

**Status:** ⏳ TODO

**Steps:**
1. [ ] Create `COST_TRACKER.md`:
   ```
   # Cost Tracker
   
   ## Monthly Budget: $900
   - OpenAI: $500/month
   - ElevenLabs: $200/month
   - Infrastructure: $200/month
   
   ## Current Month Costs:
   - OpenAI: $0
   - ElevenLabs: $0
   - Infrastructure: $0
   - Total: $0
   ```

2. [ ] Set up OpenAI usage alerts:
   - Go to platform.openai.com
   - Settings → Usage Limits
   - Set hard limit: $500/month
   - Set soft limit: $400/month (alert)

3. [ ] Set up ElevenLabs usage alerts:
   - Go to elevenlabs.io
   - Account → Usage
   - Monitor daily usage
   - Set alert at $150/month

4. [ ] Set up infrastructure monitoring:
   - Vercel: Dashboard → Usage
   - Render: Dashboard → Usage
   - Set alerts at $150/month

5. [ ] Create daily cost check:
   - Add to daily routine
   - Update `COST_TRACKER.md` daily
   - Alert if >50% of budget used

**Success Criteria:** ✅ Cost tracking active, alerts configured

---

### ✅ Task 6: Visual Asset Generation (Start - 1 hour)

**Status:** ⏳ TODO

**Priority Order:**
1. [ ] Hero Image: Kelly pointing at calendar (CRITICAL)
2. [ ] Close-up fullscreen image
3. [ ] Full-body panel open image
4. [ ] Remaining 5 images

**Steps:**
1. [ ] Read `CHRISTMAS_GIFT_VISUAL_PROMPTS.md`
2. [ ] Use Priority 1 prompt for hero image
3. [ ] Generate via Midjourney or DALL-E:
   - Midjourney: `/imagine [prompt]`
   - DALL-E: Use OpenAI API or ChatGPT
4. [ ] Download image
5. [ ] Optimize:
   - Resize to 1920×1080 (16:9)
   - Compress (TinyPNG or similar)
   - Format: WebP or optimized PNG
6. [ ] Save to: `public/assets/kelly/hero-image.png`
7. [ ] Update landing page to use image

**Success Criteria:** ✅ Hero image generated, optimized, displayed on landing page

---

## 🎯 PHASE 3: VALIDATION & TESTING (Do After Setup - 2 hours)

### ✅ Task 7: Landing Page Deployment (1 hour)

**Status:** ⏳ TODO

**Steps:**
1. [ ] Verify landing page is ready:
   - Check `public/index.html` exists
   - Verify all links work
   - Check mobile responsiveness
   - Test forms

2. [ ] Deploy to Vercel:
   ```bash
   # If using Vercel CLI
   vercel --prod
   
   # Or connect GitHub repo to Vercel
   # Vercel will auto-deploy
   ```

3. [ ] Configure Custom Domain:
   - Vercel Dashboard → Project → Settings → Domains
   - Add: curiouskelly.com
   - Add: www.curiouskelly.com
   - Verify DNS (may take 24-48 hours)

4. [ ] Configure SSL:
   - Vercel auto-configures SSL
   - Verify: https://curiouskelly.com works

5. [ ] Test Landing Page:
   - [ ] Visit https://curiouskelly.com
   - [ ] Test all links
   - [ ] Test email signup form
   - [ ] Test purchase buttons
   - [ ] Test mobile view
   - [ ] Test desktop view

**Success Criteria:** ✅ Landing page live, all features working, SSL active

---

### ✅ Task 8: Email Service Setup (1 hour)

**Status:** ⏳ TODO

**Steps:**
1. [ ] Choose email service:
   - Option A: SendGrid (recommended)
   - Option B: Mailgun
   - Option C: Resend

2. [ ] Create SendGrid account:
   - Go to sendgrid.com
   - Sign up (free tier: 100 emails/day)
   - Verify email: hello@curiouskelly.com

3. [ ] Verify Domain:
   - SendGrid → Settings → Sender Authentication
   - Add Domain: curiouskelly.com
   - Add DNS records (SPF, DKIM, DMARC)
   - Wait for verification (24-48 hours)

4. [ ] Upload Email Templates:
   - Read `EMAIL_TEMPLATES_CHRISTMAS.md`
   - Create templates in SendGrid
   - Test each template

5. [ ] Configure API Key:
   - SendGrid → Settings → API Keys
   - Create API Key: "Curious Kelly Production"
   - Add to `.env`: `SENDGRID_API_KEY=sg.xxxxx`

6. [ ] Test Email Sending:
   - Send test email from code
   - Verify delivery
   - Check spam folder

**Success Criteria:** ✅ Email service active, templates uploaded, test sends work

---

## 📊 PROGRESS TRACKING

### Today's Progress:
- [ ] Task 1: Domain & DNS Setup
- [ ] Task 2: Email Infrastructure
- [ ] Task 3: Social Media Accounts
- [ ] Task 4: Stripe Account Setup
- [ ] Task 5: Cost Monitoring
- [ ] Task 6: Visual Assets (started)
- [ ] Task 7: Landing Page Deployment
- [ ] Task 8: Email Service Setup

### Completion: 0/8 tasks (0%)

---

## 🚨 BLOCKERS & NOTES

**Current Blockers:**
- None identified

**Notes:**
- DNS propagation takes 24-48 hours (plan accordingly)
- Stripe account activation may take 1-3 days
- Domain verification takes 24-48 hours

---

## ✅ NEXT STEPS

After completing Phase 1-3:
1. Update `DAILY_TASK_TRACKER.md` with progress
2. Move to Week 2 tasks (Integration Testing)
3. Continue content creation
4. Begin quality validation

---

**Status:** 🟢 **READY TO EXECUTE**  
**Start Time:** _____________  
**Target Completion:** _____________  
**Let's execute! 🚀**

