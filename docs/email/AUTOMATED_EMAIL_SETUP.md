# 🤖 Automated Email Setup Guide

## Overview

This guide sets up automated emails that trigger on user actions:
- **Welcome email** → When someone signs up
- **Streak emails** → At 7, 14, 30, 60, 100, 365 days
- **Re-engagement** → After 3, 7, 14 days inactive

---

## Step 1: Set Up Resend (5 minutes)

### Why Resend?
- Modern API, great deliverability
- Free tier: 100 emails/day (3,000/month)
- Simple integration
- Beautiful email preview

### Create Account

1. **Go to:** https://resend.com
2. **Click:** "Start Building"
3. **Sign up** with GitHub or email
4. **Verify** your email

### Add Your Domain

1. **Dashboard → Domains → Add Domain**
2. **Enter:** `curiouskelly.com`
3. **Add DNS records** (Resend provides them):
   ```
   Type: TXT
   Name: resend._domainkey
   Value: [Resend provides this]
   
   Type: MX (optional, for receiving)
   Priority: 10
   Value: feedback-smtp.us-east-1.amazonses.com
   ```
4. **Wait 5-15 minutes**
5. **Click:** "Verify DNS Records"

### Get API Key

1. **Dashboard → API Keys → Create API Key**
2. **Name:** "Curious Kelly Production"
3. **Permission:** Full access
4. **Copy the key** (starts with `re_`)

---

## Step 2: Add Environment Variables

### In Vercel Dashboard:

1. **Go to:** vercel.com → Your project
2. **Settings → Environment Variables**
3. **Add:**

| Name | Value |
|------|-------|
| `RESEND_API_KEY` | `re_xxxxxxxxx` (your key) |
| `WELCOME_EMAIL_API_KEY` | Generate a random string (for security) |

**Generate a random key:**
```bash
node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"
```

### In Local Development:

Create/update `.env.local`:
```bash
RESEND_API_KEY=re_xxxxxxxxx
WELCOME_EMAIL_API_KEY=your_generated_key
```

---

## Step 3: Deploy Email Endpoints

The following API endpoints are ready:

| Endpoint | Purpose | Trigger |
|----------|---------|---------|
| `/api/send-welcome-email` | Welcome new users | Signup |
| `/api/send-streak-email` | Celebrate streaks | 7, 30, 100 days |

### Deploy to Vercel:
```bash
git add api/send-welcome-email.ts api/send-streak-email.ts
git commit -m "Add automated email endpoints"
git push origin main
```

Vercel will auto-deploy.

---

## Step 4: Test the Endpoints

### Test Welcome Email:

```bash
curl -X POST https://curiouskelly.com/api/send-welcome-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_WELCOME_EMAIL_API_KEY" \
  -d '{"email": "your@email.com", "name": "Test User"}'
```

**Expected response:**
```json
{"success": true, "message": "Welcome email sent", "id": "..."}
```

### Test Streak Email:

```bash
curl -X POST https://curiouskelly.com/api/send-streak-email \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_WELCOME_EMAIL_API_KEY" \
  -d '{"email": "your@email.com", "name": "Test User", "streak": 7}'
```

---

## Step 5: Connect to Supabase Auth

### Option A: Supabase Edge Function (Recommended)

Create a Supabase Edge Function that triggers on signup:

**File:** `supabase/functions/on-signup/index.ts`

```typescript
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'

serve(async (req) => {
  const { record } = await req.json()
  
  // User just signed up
  const email = record.email
  const name = record.raw_user_meta_data?.full_name || record.email?.split('@')[0]
  
  // Send welcome email
  const response = await fetch('https://curiouskelly.com/api/send-welcome-email', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${Deno.env.get('WELCOME_EMAIL_API_KEY')}`,
    },
    body: JSON.stringify({ email, name }),
  })
  
  return new Response(JSON.stringify({ success: true }), {
    headers: { 'Content-Type': 'application/json' },
  })
})
```

**Set up the trigger:**
1. Go to Supabase Dashboard → Database → Triggers
2. Create new trigger on `auth.users` table
3. Event: INSERT
4. Function: Call edge function

### Option B: Client-Side After Signup

Call the API after successful signup in your frontend:

```javascript
// After Supabase auth.signUp() succeeds
const { data: { user } } = await supabase.auth.signUp({ email, password })

if (user) {
  // Send welcome email
  await fetch('/api/send-welcome-email', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: user.email,
      name: user.user_metadata?.full_name || user.email.split('@')[0]
    })
  })
}
```

### Option C: Supabase Database Webhook

1. **Supabase Dashboard → Database → Webhooks**
2. **Create webhook:**
   - Name: `send-welcome-email`
   - Table: `auth.users`
   - Events: INSERT
   - URL: `https://curiouskelly.com/api/send-welcome-email`
   - Headers: `Authorization: Bearer YOUR_KEY`

---

## Step 6: Set Up Streak Tracking

### Database Table

Create a table to track user streaks:

```sql
CREATE TABLE user_streaks (
  user_id UUID REFERENCES auth.users(id) PRIMARY KEY,
  current_streak INT DEFAULT 0,
  longest_streak INT DEFAULT 0,
  last_lesson_date DATE,
  streak_milestone_sent INT[] DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Cron Job for Streak Emails

Use Vercel Cron or a scheduled function to check and send streak emails:

**File:** `api/cron/check-streaks.ts`

```typescript
import type { VercelRequest, VercelResponse } from '@vercel/node'

const MILESTONE_DAYS = [7, 14, 30, 60, 100, 365]

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Verify cron secret
  if (req.headers.authorization !== `Bearer ${process.env.CRON_SECRET}`) {
    return res.status(401).json({ error: 'Unauthorized' })
  }

  // Query users who hit milestones today
  // For each milestone user, send streak email
  // Update streak_milestone_sent array
  
  return res.status(200).json({ success: true })
}
```

**Add to `vercel.json`:**
```json
{
  "crons": [
    {
      "path": "/api/cron/check-streaks",
      "schedule": "0 8 * * *"
    }
  ]
}
```

---

## Email Preview

### Welcome Email

**Subject:** Welcome to curiosity, [Name]! 🎉

**Preview:**
- Kelly's avatar at top
- Personalized greeting
- What to expect (3 bullet points)
- Big blue CTA button
- Fun fact in highlighted box
- Footer with links

### Streak Email (7 days)

**Subject:** 7 days of curiosity! 🔥

**Preview:**
- Celebration gradient header with 🔥
- "You absolute legend" message
- Motivational content
- "Keep the Streak Alive" CTA
- Fun fact about top percentile

---

## Testing Checklist

- [ ] Resend account created
- [ ] Domain verified in Resend
- [ ] API key added to Vercel
- [ ] Welcome email endpoint deployed
- [ ] Streak email endpoint deployed
- [ ] Test welcome email received
- [ ] Test streak email received
- [ ] Supabase trigger configured
- [ ] Real signup triggers email
- [ ] Emails look good on mobile

---

## Troubleshooting

### "Email not delivered"
→ Check spam folder
→ Verify domain DNS records
→ Check Resend dashboard for errors

### "API returns 401"
→ Verify Authorization header
→ Check WELCOME_EMAIL_API_KEY matches

### "Domain not verified"
→ DNS records can take up to 48 hours
→ Use Resend's "Verify" button to check

### "Email looks broken"
→ Test with different email clients
→ Inline CSS is required for email
→ Check image URLs are absolute

---

## Cost Estimate

| Service | Free Tier | Paid |
|---------|-----------|------|
| Resend | 100/day (3,000/mo) | $20/mo for 50K |
| Vercel | Included | Included |
| Supabase | Included | Included |

**For launch:** Free tier should be plenty (100 signups/day = 3,000/month)

---

## Next Steps After Setup

1. [ ] Monitor delivery rates in Resend dashboard
2. [ ] Set up re-engagement email sequence
3. [ ] A/B test subject lines
4. [ ] Add unsubscribe functionality
5. [ ] Set up email analytics tracking

---

*Welcome emails that feel like a warm hug from Kelly* ✨


