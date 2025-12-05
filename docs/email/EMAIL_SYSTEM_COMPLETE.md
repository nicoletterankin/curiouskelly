# ✨ Curious Kelly Email System - Complete Guide

> **Status:** Production Ready  
> **Provider:** Resend (resend.com)  
> **Domain:** curiouskelly.com (verified ✅)  
> **From Address:** `Kelly <hello@curiouskelly.com>`

---

## 📧 Email Types & API Endpoints

### 1. Welcome Email (New User Signup)
**Endpoint:** `POST /api/send-welcome-email`  
**Trigger:** Supabase auth webhook on user creation

```bash
curl -X POST https://www.curiouskelly.com/api/send-welcome-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_WELCOME_EMAIL_API_KEY" \
  -d '{"email": "user@example.com", "name": "Alex"}'
```

### 2. Daily Lesson Email
**Endpoint:** `POST /api/send-daily-lesson-email`  
**Trigger:** Cron job (recommended: 7am user timezone)

```bash
# Single email
curl -X POST https://www.curiouskelly.com/api/send-daily-lesson-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_DAILY_EMAIL_API_KEY" \
  -d '{
    "email": "user@example.com",
    "name": "Alex",
    "lessonTitle": "How Money Works",
    "lessonEmoji": "💰",
    "lessonCategory": "Economics",
    "dayNumber": 1
  }'

# Batch emails (up to 100)
curl -X POST https://www.curiouskelly.com/api/send-daily-lesson-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_DAILY_EMAIL_API_KEY" \
  -d '{
    "lessonTitle": "How Money Works",
    "lessonEmoji": "💰",
    "lessonCategory": "Economics",
    "dayNumber": 1,
    "batch": [
      {"email": "user1@example.com", "name": "Alex"},
      {"email": "user2@example.com", "name": "Sam"}
    ]
  }'
```

### 3. Streak Celebration Email
**Endpoint:** `POST /api/send-streak-email`  
**Trigger:** When user hits streak milestone (3, 7, 14, 30, 60, 90, 100, 365 days)

```bash
curl -X POST https://www.curiouskelly.com/api/send-streak-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_WELCOME_EMAIL_API_KEY" \
  -d '{"email": "user@example.com", "name": "Alex", "streakDays": 7}'
```

### 4. Affiliate Welcome Email
**Endpoint:** `POST /api/send-affiliate-email`  
**Trigger:** When affiliate account is approved

```bash
curl -X POST https://www.curiouskelly.com/api/send-affiliate-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_AFFILIATE_EMAIL_API_KEY" \
  -d '{
    "type": "welcome",
    "email": "affiliate@example.com",
    "name": "Partner Name",
    "affiliateCode": "PARTNER123"
  }'
```

### 5. Affiliate Payout Email
**Endpoint:** `POST /api/send-affiliate-email`  
**Trigger:** Monthly payout processed

```bash
curl -X POST https://www.curiouskelly.com/api/send-affiliate-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_AFFILIATE_EMAIL_API_KEY" \
  -d '{
    "type": "payout",
    "email": "affiliate@example.com",
    "name": "Partner Name",
    "amount": "$150.00",
    "referralCount": 5,
    "payoutMethod": "PayPal",
    "period": "November 2025"
  }'
```

### 6. Supabase Auth Webhook (Auto Welcome)
**Endpoint:** `POST /api/supabase-auth-webhook`  
**Trigger:** Automatic on new user signup in Supabase

---

## 🔧 Setup Instructions

### Step 1: Resend Configuration (Already Done ✅)

- **Domain:** curiouskelly.com
- **DKIM:** Verified ✅
- **SPF:** Verified ✅
- **From:** `Kelly <hello@curiouskelly.com>`

### Step 2: Environment Variables in Vercel

Add these to **curiouskelly** project (not lotd):

| Variable | Description | Required |
|----------|-------------|----------|
| `RESEND_API_KEY` | Your Resend API key (starts with `re_`) | ✅ Yes |
| `WELCOME_EMAIL_API_KEY` | Secret for welcome/streak emails | ✅ Yes |
| `DAILY_EMAIL_API_KEY` | Secret for daily lesson emails | ✅ Yes |
| `AFFILIATE_EMAIL_API_KEY` | Secret for affiliate emails | ✅ Yes |
| `SUPABASE_WEBHOOK_SECRET` | Secret for Supabase webhooks | ✅ Yes |

**Generate API keys:**
```bash
# Generate secure random keys
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

### Step 3: Supabase Webhook Setup

1. Go to **Supabase Dashboard** → **Database** → **Webhooks**
2. Click **Create a new webhook**
3. Configure:
   - **Name:** `send-welcome-email`
   - **Table:** `auth.users`
   - **Events:** `INSERT`
   - **Type:** `HTTP Request`
   - **Method:** `POST`
   - **URL:** `https://www.curiouskelly.com/api/supabase-auth-webhook`
   - **HTTP Headers:**
     ```
     x-webhook-secret: YOUR_SUPABASE_WEBHOOK_SECRET
     Content-Type: application/json
     ```
4. Click **Create webhook**

### Step 4: Daily Email Cron Job

Option A: **Vercel Cron** (recommended)
```json
// vercel.json
{
  "crons": [
    {
      "path": "/api/cron/daily-lesson",
      "schedule": "0 12 * * *"  // 12pm UTC = 7am EST
    }
  ]
}
```

Option B: **External Cron Service**
- Use cron-job.org, EasyCron, or GitHub Actions
- Hit `/api/send-daily-lesson-email` with batch of users

---

## 📊 Email Templates Overview

| Email | Subject Line | When Sent |
|-------|--------------|-----------|
| Welcome | "Welcome to curiosity, {name}! 🎉" | User signup |
| Daily Lesson | "{emoji} Today's lesson: {title}" | Daily 7am |
| Streak 3-day | "🔥 3-Day Streak! You're on fire!" | Day 3 |
| Streak 7-day | "⭐ One Week Wonder!" | Day 7 |
| Streak 30-day | "🏆 Monthly Master!" | Day 30 |
| Streak 100-day | "💯 100 Days of Wonder!" | Day 100 |
| Affiliate Welcome | "🤝 Welcome to the Curious Kelly Affiliate Program!" | Approval |
| Affiliate Payout | "💰 Your {amount} affiliate payout is on the way!" | Monthly |
| Re-engagement | "🥺 {name}, I miss our learning adventures!" | 7+ days inactive |

---

## 🎨 Email Design System

### Brand Colors
- **Background:** `#0a0a0b`
- **Card:** `#18181b`
- **Accent (Blue):** `#3b82f6`
- **Gold (Highlights):** `#fbbf24`
- **Text:** `#f4f4f5`
- **Muted:** `#a1a1aa`

### Components
- Kelly avatar header with blue border
- Clean card-based layout
- "Did you know?" fun fact box
- Blue CTA buttons
- Consistent footer with links

### Voice
- Warm, personal, enthusiastic
- First person from Kelly
- Emojis used sparingly but effectively
- Always ends with "Stay curious"

---

## 📈 Analytics & Tracking

Resend provides built-in analytics:
- **Open rates** - Track email opens
- **Click rates** - Track CTA clicks
- **Bounce rates** - Monitor delivery issues
- **Tags** - Filter by email type

### Email Tags
```typescript
EMAIL_TAGS = {
  WELCOME: { name: 'type', value: 'welcome' },
  DAILY_LESSON: { name: 'type', value: 'daily_lesson' },
  STREAK: { name: 'type', value: 'streak' },
  AFFILIATE_WELCOME: { name: 'type', value: 'affiliate_welcome' },
  AFFILIATE_PAYOUT: { name: 'type', value: 'affiliate_payout' },
  RE_ENGAGEMENT: { name: 'type', value: 're_engagement' },
}
```

---

## 🚀 Deployment Checklist

- [ ] Add `RESEND_API_KEY` to Vercel curiouskelly project
- [ ] Add `WELCOME_EMAIL_API_KEY` to Vercel
- [ ] Add `DAILY_EMAIL_API_KEY` to Vercel
- [ ] Add `AFFILIATE_EMAIL_API_KEY` to Vercel
- [ ] Add `SUPABASE_WEBHOOK_SECRET` to Vercel
- [ ] Create Supabase webhook for auth.users INSERT
- [ ] Test welcome email (new signup)
- [ ] Test daily lesson email (manual)
- [ ] Test streak email (manual)
- [ ] Set up daily cron job
- [ ] Monitor Resend dashboard for delivery

---

## 🧪 Testing Commands

```bash
# Test welcome email
curl -X POST https://www.curiouskelly.com/api/send-welcome-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"email": "test@example.com", "name": "Test"}'

# Test daily lesson email
curl -X POST https://www.curiouskelly.com/api/send-daily-lesson-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "email": "test@example.com",
    "name": "Test",
    "lessonTitle": "Test Lesson",
    "lessonEmoji": "🧪",
    "lessonCategory": "Testing",
    "dayNumber": 1
  }'

# Test affiliate welcome
curl -X POST https://www.curiouskelly.com/api/send-affiliate-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "type": "welcome",
    "email": "test@example.com",
    "name": "Test Partner",
    "affiliateCode": "TEST123"
  }'
```

---

## 📞 Support & Contacts

- **Email Issues:** Check Resend dashboard logs
- **Delivery Issues:** Verify SPF/DKIM in DNS
- **API Issues:** Check Vercel function logs
- **Questions:** hello@curiouskelly.com

---

*Last Updated: December 5, 2025*

