# Curious Kelly - Deployment Setup Guide

## Infrastructure Setup Instructions

This guide covers the manual steps required to set up infrastructure that cannot be automated.

## 1. Domain Setup (curiouskelly.com)

### Purchase Domain
1. Go to Namecheap or GoDaddy
2. Search for `curiouskelly.com`
3. Purchase domain (approximately $12-15/year)

### DNS Configuration
Once domain is purchased, configure these DNS records:

```
Type    Name    Value                           TTL
A       @       [Your server IP or Vercel IP]   Auto
CNAME   www     curiouskelly.com                Auto
MX      @       [Email provider MX records]     Auto
TXT     @       [SPF record for email]          Auto
```

### For Vercel Hosting:
1. Go to vercel.com and create account
2. Import this repository
3. In Vercel dashboard → Settings → Domains
4. Add custom domain: `curiouskelly.com`
5. Follow Vercel's DNS instructions
6. Wait for SSL certificate (automatic)

### For Cloudflare Pages:
1. Go to pages.cloudflare.com
2. Connect GitHub repository
3. Configure build settings:
   - Build command: (none for static)
   - Output directory: `public`
4. Add custom domain in dashboard
5. Update nameservers at domain registrar

## 2. Email Setup (hello@curiouskelly.com)

### Option A: Google Workspace (Recommended)
**Cost: $6/user/month**

1. Go to workspace.google.com
2. Sign up for Google Workspace
3. Add domain: `curiouskelly.com`
4. Verify domain ownership (add TXT record)
5. Configure MX records:
```
Priority  Mail Server
1         ASPMX.L.GOOGLE.COM
5         ALT1.ASPMX.L.GOOGLE.COM
5         ALT2.ASPMX.L.GOOGLE.COM
10        ALT3.ASPMX.L.GOOGLE.COM
10        ALT4.ASPMX.L.GOOGLE.COM
```
6. Create user: `hello@curiouskelly.com`
7. Test send/receive

### Option B: Cloudflare Email Routing (Free)
1. Go to Cloudflare dashboard → Email
2. Enable Email Routing
3. Add destination email (your personal email)
4. Create routing rule: `hello@curiouskelly.com` → your email
5. Verify DNS records added automatically
6. Test with Cloudflare's test tool

### SPF Record (for sending)
Add TXT record:
```
Name: @
Value: v=spf1 include:_spf.google.com ~all
```
(or appropriate for your email provider)

## 3. SendGrid Setup

### Create Account
1. Go to sendgrid.com
2. Sign up (Free tier: 100 emails/day)
3. Verify email address
4. Complete sender authentication

### Domain Authentication
1. Go to Settings → Sender Authentication
2. Authenticate domain: `curiouskelly.com`
3. Add provided DNS records (CNAME records)
4. Wait for verification (can take 24-48 hours)

### Create API Key
1. Go to Settings → API Keys
2. Create API Key with "Full Access"
3. Copy key immediately (shown only once)
4. Save to `.env` file: `SENDGRID_API_KEY=SG.xxxxx`

### Create Dynamic Templates
See `EMAIL_TEMPLATES_CHRISTMAS.md` for all 14 templates.

For each template:
1. Go to Email API → Dynamic Templates
2. Click "Create Dynamic Template"
3. Give it a name (e.g., "Gift Recipient Notification")
4. Create version
5. Use Design Editor or Code Editor
6. Add personalization fields: `{{recipient_name}}`, `{{gift_code}}`, etc.
7. Save and get Template ID
8. Record template ID in `curious-kellly/backend/src/config/email-templates.js`

## 4. Stripe Setup

### Create Account
1. Go to stripe.com
2. Sign up for Stripe account
3. Complete business verification
4. Activate account

### Get API Keys
1. Go to Developers → API keys
2. Copy:
   - Publishable key (starts with `pk_test_` for test mode)
   - Secret key (starts with `sk_test_` for test mode)
3. Save to `.env`:
```
STRIPE_PUBLISHABLE_KEY=pk_test_xxxxx
STRIPE_SECRET_KEY=sk_test_xxxxx
```

### Create Products
Go to Products → Add product

**Product 1: Personal Plan**
- Name: Curious Kelly - Personal Plan
- Description: 365 daily lessons with your personal AI teacher
- Pricing: $199.00 USD
- Billing period: Yearly
- Copy Price ID to `.env`: `PRICE_ID_PERSONAL=price_xxxxx`

**Product 2: Family Plan**
- Name: Curious Kelly - Family Plan
- Description: 365 daily lessons for up to 6 family members
- Pricing: $299.00 USD
- Billing period: Yearly
- Copy Price ID to `.env`: `PRICE_ID_FAMILY=price_xxxxx`

**Product 3: Gift Plan**
- Name: Curious Kelly - Gift Plan
- Description: Give 365 days of learning (starts Jan 1, 2026)
- Pricing: $199.00 USD
- Billing: One-time payment
- Copy Price ID to `.env`: `PRICE_ID_GIFT=price_xxxxx`

### Configure Webhooks
1. Go to Developers → Webhooks
2. Add endpoint: `https://api.curiouskelly.com/webhook`
3. Select events to listen for:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.deleted`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`
4. Copy webhook signing secret to `.env`:
```
STRIPE_WEBHOOK_SECRET=whsec_xxxxx
```

### Test Mode vs Live Mode
- Use test mode for development (API keys start with `pk_test_` and `sk_test_`)
- Use test cards: `4242 4242 4242 4242` (Visa)
- Before launch, switch to live mode
- Update `.env` with live API keys

## 5. Database Setup (PostgreSQL)

### Option A: Heroku Postgres (Free tier available)
1. Create Heroku account
2. Create new app
3. Add Heroku Postgres add-on
4. Get database URL from Settings → Config Vars
5. Copy to `.env`: `DATABASE_URL=postgres://xxxxx`

### Option B: Railway (Free tier available)
1. Go to railway.app
2. Create new project
3. Add PostgreSQL database
4. Copy connection string to `.env`

### Option C: Local PostgreSQL
```bash
# Install PostgreSQL
# Windows: Download from postgresql.org
# Mac: brew install postgresql
# Linux: sudo apt-get install postgresql

# Create database
createdb curious_kelly

# Connection string
DATABASE_URL=postgresql://localhost/curious_kelly
```

### Run Migrations
```bash
cd curious-kellly/backend
npm install
npm run migrate
```

## 6. Environment Variables

Create `curious-kellly/backend/.env` file:

```env
# Server
NODE_ENV=development
PORT=3000
BASE_URL=http://localhost:3000

# Database
DATABASE_URL=postgresql://localhost/curious_kelly

# Stripe
STRIPE_PUBLISHABLE_KEY=pk_test_xxxxx
STRIPE_SECRET_KEY=sk_test_xxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxx
PRICE_ID_PERSONAL=price_xxxxx
PRICE_ID_FAMILY=price_xxxxx
PRICE_ID_GIFT=price_xxxxx

# SendGrid
SENDGRID_API_KEY=SG.xxxxx
FROM_EMAIL=hello@curiouskelly.com
FROM_NAME=Curious Kelly

# Email Template IDs (from SendGrid)
TEMPLATE_WAITLIST=d-xxxxx
TEMPLATE_EARLY_BIRD=d-xxxxx
TEMPLATE_LAST_CHANCE=d-xxxxx
TEMPLATE_GIFT_RECIPIENT=d-xxxxx
TEMPLATE_GIFTER_CONFIRM=d-xxxxx
TEMPLATE_CALENDAR_EXPLORE=d-xxxxx
TEMPLATE_GET_READY=d-xxxxx
TEMPLATE_DAY1=d-xxxxx
TEMPLATE_WELCOME=d-xxxxx
TEMPLATE_DAILY_REMINDER=d-xxxxx
TEMPLATE_STREAK=d-xxxxx
TEMPLATE_WEEK1=d-xxxxx
TEMPLATE_MISSED=d-xxxxx
TEMPLATE_REENGAGE=d-xxxxx

# Frontend URL (for redirects)
FRONTEND_URL=https://curiouskelly.com
```

**IMPORTANT: Never commit `.env` file to git!**
Add to `.gitignore`:
```
.env
.env.local
.env.production
```

## 7. Deployment Checklist

### Before Launch:
- [ ] Domain purchased and DNS configured
- [ ] SSL certificate active (automatic with Vercel/Cloudflare)
- [ ] Email sending verified (test with real email)
- [ ] SendGrid domain authenticated
- [ ] All 14 email templates created and tested
- [ ] Stripe products created
- [ ] Stripe webhooks configured
- [ ] Database created and migrations run
- [ ] All environment variables set
- [ ] Test gift purchase flow end-to-end
- [ ] Switch Stripe to live mode
- [ ] Update `.env` with production values

### Production Environment Variables
For production, set these in your hosting platform (Vercel, Railway, etc.):
- Use live Stripe keys (pk_live_ and sk_live_)
- Use production database URL
- Use production BASE_URL and FRONTEND_URL
- Keep SENDGRID_API_KEY same (works for production)

## 8. Monitoring Setup

### Sentry (Error Tracking)
1. Go to sentry.io
2. Create project
3. Get DSN
4. Add to `.env`: `SENTRY_DSN=https://xxxxx@sentry.io/xxxxx`
5. Install: `npm install @sentry/node`

### Google Analytics
1. Go to analytics.google.com
2. Create property for curiouskelly.com
3. Get Measurement ID (G-XXXXXXXXXX)
4. Add to landing page and lesson player

## 9. Support and Resources

### Documentation
- Vercel Docs: vercel.com/docs
- Cloudflare Pages: developers.cloudflare.com/pages
- SendGrid Docs: docs.sendgrid.com
- Stripe Docs: stripe.com/docs

### Support Contacts
- Domain issues: Contact registrar support
- Email delivery: SendGrid support (support@sendgrid.com)
- Payment issues: Stripe support (support@stripe.com)
- Hosting: Vercel/Cloudflare support

## 10. Testing Checklist

Before going live:
- [ ] Send test email to yourself
- [ ] Make test Stripe purchase
- [ ] Verify webhook receives events
- [ ] Test gift code generation and redemption
- [ ] Test on mobile devices
- [ ] Verify analytics tracking
- [ ] Check all links work
- [ ] Test error pages (404, 500)

## Next Steps

After completing this setup:
1. Run backend locally: `cd curious-kellly/backend && npm start`
2. Test all endpoints
3. Deploy frontend to Vercel/Cloudflare
4. Deploy backend to Railway/Heroku
5. Test production environment
6. Launch! 🚀

































