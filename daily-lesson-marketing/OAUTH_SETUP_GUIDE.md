# 🔐 Curious Kelly - OAuth Setup Guide

**Time required:** 15-20 minutes  
**Result:** One-click Google & Apple sign-in for millions of learners

---

## Quick Overview

Your auth system is now production-ready. You need to:
1. Configure Google OAuth in Supabase
2. Configure Apple OAuth in Supabase (optional but recommended)
3. Set environment variables

---

## Step 1: Google OAuth (Primary - Most Users Will Use This)

### 1.1 Create Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing): "Curious Kelly"
3. Navigate to **APIs & Services** → **Credentials**
4. Click **Create Credentials** → **OAuth client ID**
5. Select **Web application**
6. Name it: `Curious Kelly Web`

### 1.2 Configure OAuth Consent Screen

1. Go to **OAuth consent screen**
2. Choose **External** (for all users)
3. Fill in:
   - **App name:** `Curious Kelly`
   - **User support email:** `hello@curiouskelly.com`
   - **App logo:** Upload Kelly logo
   - **App domain:** `curiouskelly.com`
   - **Privacy policy:** `https://curiouskelly.com/privacy.html`
   - **Terms of service:** `https://curiouskelly.com/terms.html`
4. Add scopes: `email`, `profile`, `openid`
5. Save and continue

### 1.3 Set Authorized Redirect URIs

Add these exact URIs:

```
https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback
https://curiouskelly.com/
http://localhost:4321/ (for local development)
```

### 1.4 Copy Credentials

Save these (you'll need them for Supabase):
- **Client ID:** `xxxxxx.apps.googleusercontent.com`
- **Client Secret:** `GOCSPX-xxxxxx`

---

## Step 2: Configure Supabase

### 2.1 Open Supabase Dashboard

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Select project: `tvjalxxsyryjphkforjv`
3. Navigate to **Authentication** → **Providers**

### 2.2 Enable Google Provider

1. Find **Google** in the list
2. Toggle **Enable**
3. Paste your:
   - **Client ID** (from Google)
   - **Client Secret** (from Google)
4. Click **Save**

### 2.3 Configure Redirect URLs

1. Go to **Authentication** → **URL Configuration**
2. Set:
   - **Site URL:** `https://curiouskelly.com`
   - **Redirect URLs:** Add all of these:
     ```
     https://curiouskelly.com/
     https://curiouskelly.com/welcome
     https://curiouskelly.com/checkout
     http://localhost:4321/
     http://localhost:4321/welcome
     ```

---

## Step 3: Apple Sign In (Recommended for iOS Users)

### 3.1 Apple Developer Setup

1. Go to [Apple Developer Portal](https://developer.apple.com/)
2. Navigate to **Certificates, IDs & Profiles**
3. Create a new **App ID** (if not exists):
   - Description: `Curious Kelly`
   - Bundle ID: `com.curiouskelly.web`
   - Enable **Sign In with Apple**

### 3.2 Create Service ID

1. Create new **Service ID**
2. Description: `Curious Kelly Web`
3. Identifier: `com.curiouskelly.web.signin`
4. Enable **Sign In with Apple**
5. Configure:
   - **Domains:** `curiouskelly.com`, `tvjalxxsyryjphkforjv.supabase.co`
   - **Return URLs:** `https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback`

### 3.3 Create Key

1. Create new **Key**
2. Name: `Curious Kelly Sign In`
3. Enable **Sign In with Apple**
4. Download the `.p8` file (you'll need this)

### 3.4 Configure Supabase

1. In Supabase → **Authentication** → **Providers** → **Apple**
2. Toggle **Enable**
3. Fill in:
   - **Service ID:** `com.curiouskelly.web.signin`
   - **Team ID:** (from Apple Developer account top right)
   - **Key ID:** (from the key you created)
   - **Private Key:** (contents of the .p8 file)

---

## Step 4: Environment Variables

Add these to your `.env` file and Vercel/deployment platform:

```bash
# Supabase (REQUIRED)
PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# Stripe (REQUIRED for payments)
STRIPE_SECRET_KEY=sk_live_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
STRIPE_PRICE_MONTHLY=price_xxx
STRIPE_PRICE_ANNUAL=price_xxx
STRIPE_PRICE_GIFT=price_xxx

# Customer.io (for email automation)
CUSTOMER_IO_SITE_ID=xxx
CUSTOMER_IO_API_KEY=xxx
CUSTOMER_IO_APP_API_KEY=xxx

# Site URL
PUBLIC_SITE_URL=https://curiouskelly.com
```

### Get Supabase Keys

1. Go to Supabase Dashboard → **Settings** → **API**
2. Copy:
   - **anon public** key → `PUBLIC_SUPABASE_ANON_KEY`
   - **service_role secret** key → `SUPABASE_SERVICE_ROLE_KEY`

---

## Step 5: Test the Flow

### Local Testing

```bash
cd daily-lesson-marketing
npm install
npm run dev
```

Visit `http://localhost:4321` and try:
1. **Google Sign In** - Should redirect to Google, then back
2. **Guest Mode** - Should instantly show the OS
3. **Checkout** - Should create Stripe session

### Production Testing

1. Deploy to Vercel/Cloudflare
2. Test on `curiouskelly.com`
3. Monitor Supabase Auth logs for errors

---

## Troubleshooting

### "OAuth error: redirect_uri_mismatch"

Your redirect URI in Google doesn't match Supabase. Add:
```
https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback
```

### "Invalid API key"

Check that `PUBLIC_SUPABASE_ANON_KEY` is set correctly. It should start with `eyJ...`

### "User not created in database"

The trigger in `supabase-schema.sql` should auto-create users. Run:
```sql
-- Check if trigger exists
SELECT * FROM pg_trigger WHERE tgname = 'on_auth_user_created';
```

### Apple Sign In not working

1. Verify Service ID domains include Supabase callback URL
2. Check the private key format (must include `-----BEGIN PRIVATE KEY-----`)
3. Ensure Team ID is correct (10-character string)

---

## Security Checklist

- [ ] Never commit `.env` files to git
- [ ] Use `SUPABASE_SERVICE_ROLE_KEY` only server-side (API routes)
- [ ] Enable Supabase RLS (Row Level Security) on all tables
- [ ] Set up Supabase email rate limiting (Authentication → Settings)
- [ ] Review OAuth app permissions (minimal scopes only)

---

## Scale Preparation (AnySphere-level)

Your system is ready for massive traffic:

✅ **Rate Limiting** - 5 checkout requests/min per IP  
✅ **Session Persistence** - Users stay logged in  
✅ **Auto-refresh Tokens** - Sessions don't expire  
✅ **Guest Mode** - Zero friction entry  
✅ **LRU Cache** - Memory-efficient rate limiting  

For >100K concurrent users, consider:
- [ ] Switch rate limiting to Redis/Upstash
- [ ] Add Cloudflare in front of Supabase
- [ ] Enable Supabase connection pooling (pgBouncer)

---

## Files Modified

| File | Purpose |
|------|---------|
| `src/lib/auth.ts` | Production auth library |
| `src/lib/ratelimit.ts` | Scale-ready rate limiting |
| `src/lib/supabase.ts` | Server-side Supabase client |
| `src/pages/index.astro` | Real OAuth integration |
| `public/lesson-player/js/app.js` | OS auth integration |

---

## Need Help?

If you hit any OAuth issues, share:
1. The exact error message
2. Which provider (Google/Apple)
3. Your Supabase project URL

I'll help you debug it step by step.

---

**Your learners are one click away from starting their journey.** 🎉

