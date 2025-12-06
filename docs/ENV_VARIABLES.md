# Environment Variables - Curious Kelly

## Master List

All environment variables used in production. **Single source of truth.**

---

## Required for Core Functionality

| Variable | Purpose | Where to Get |
|----------|---------|--------------|
| `PUBLIC_SUPABASE_URL` | Database URL | Supabase → Settings → API |
| `SUPABASE_SERVICE_ROLE_KEY` | Admin DB access | Supabase → Settings → API (secret) |
| `PUBLIC_SUPABASE_ANON_KEY` | Public DB access | Supabase → Settings → API |
| `RESEND_API_KEY` | Email sending | Resend → API Keys |
| `CRON_SECRET` | Protects cron jobs | Generate: `openssl rand -hex 32` |

---

## Webhooks & Integrations

| Variable | Purpose | Where to Get |
|----------|---------|--------------|
| `SUPABASE_WEBHOOK_SECRET` | Validates auth webhooks | Supabase → Auth → Hooks → Generate |

**Setup**: When you create a webhook in Supabase, it generates a secret. Copy that secret to Vercel.

---

## Stripe (Payments)

| Variable | Purpose | Where to Get |
|----------|---------|--------------|
| `STRIPE_SECRET_KEY` | Server-side Stripe | Stripe → Developers → API Keys |
| `STRIPE_PUBLISHABLE_KEY` | Client-side Stripe | Stripe → Developers → API Keys |
| `STRIPE_WEBHOOK_SECRET` | Validate Stripe webhooks | Stripe → Webhooks → Signing secret |
| `STRIPE_PRICE_MONTHLY` | Monthly plan price ID | Stripe → Products |
| `STRIPE_PRICE_ANNUAL` | Annual plan price ID | Stripe → Products |
| `STRIPE_PRICE_LIFETIME` | Lifetime plan price ID | Stripe → Products |
| `STRIPE_PRICE_GIFT_1MO` | 1-month gift price ID | Stripe → Products |
| `STRIPE_PRICE_GIFT_6MO` | 6-month gift price ID | Stripe → Products |
| `STRIPE_PRICE_GIFT_3MO` | 3-month gift price ID | Stripe → Products |

---

## ElevenLabs (Voice/AI)

| Variable | Purpose | Where to Get |
|----------|---------|--------------|
| `ELEVENLABS_API_KEY` | TTS and voice | ElevenLabs → Profile → API Key |
| `ELEVENLABS_AGENT_ID` | Conversational AI agent | ElevenLabs → Agents |
| `ELEVENLABS_VOICE_ID` | Kelly's voice | ElevenLabs → Voices |

---

## Other APIs

| Variable | Purpose | Where to Get |
|----------|---------|--------------|
| `ANT_API_KEY` | Anthropic Claude | Anthropic Console |
| `REPLICATE_API_TOKEN` | Lipsync/video | Replicate → Account |

---

## Legacy/Unused (Can Remove)

| Variable | Notes |
|----------|-------|
| `DAILY_EMAIL_API_KEY` | Replaced by CRON_SECRET |
| `WELCOME_EMAIL_API_KEY` | Replaced by webhook |
| `AFFILIATE_EMAIL_API_KEY` | Not yet active |

---

## How to Update a Variable in Vercel

1. Go to Vercel → Project → Settings → Environment Variables
2. Find the variable
3. Click the **three dots** (⋮) on the right
4. Click **Edit**
5. Paste new value
6. Click **Save**
7. **Redeploy** for changes to take effect

---

## How to Add a New Variable

1. Go to Vercel → Project → Settings → Environment Variables
2. Enter Key and Value
3. Select environments (Production, Preview, Development)
4. Click **Save**
5. **Redeploy**

---

## Current Vercel Variables (as of Dec 2024)

From your screenshot:
- ✅ CRON_SECRET
- ✅ SUPABASE_WEBHOOK_SECRET  
- ✅ SUPABASE_SERVICE_ROLE_KEY
- ✅ RESEND_API_KEY
- ✅ ANT_API_KEY
- ✅ ELEVENLABS_AGENT_ID
- ✅ ELEVENLABS_API_KEY
- ✅ STRIPE_WEBHOOK_SECRET
- ✅ STRIPE_SECRET_KEY
- ✅ STRIPE_PRICE_MONTHLY
- ✅ STRIPE_PRICE_ANNUAL
- ✅ STRIPE_PRICE_LIFETIME
- ✅ STRIPE_PRICE_GIFT_1MO
- ✅ STRIPE_PRICE_GIFT_6MO
- ✅ STRIPE_PRICE_GIFT_3MO
- ✅ STRIPE_PUBLISHABLE_KEY
- ✅ PUBLIC_SUPABASE_URL
- ✅ PUBLIC_SUPABASE_ANON_KEY
- ✅ ELEVENLABS_VOICE_ID

