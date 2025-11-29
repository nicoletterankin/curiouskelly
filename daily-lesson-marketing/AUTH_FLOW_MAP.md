# 🔐 Curious Kelly - Authentication Flow Map

## Overview

This document maps every URL and redirect in the authentication system so users **never see a 404**.

---

## The Complete Auth Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER CLICKS "Continue with Google"           │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Browser redirects to Google OAuth                              │
│  URL: accounts.google.com/...                                   │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  User logs in / grants permission                               │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Google redirects to Supabase callback                          │
│  URL: tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback         │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Supabase redirects to Site URL with tokens in hash             │
│  URL: curiouskelly.com/#access_token=xxx&refresh_token=xxx      │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  index.astro detects hash, Supabase JS processes tokens         │
│  Session is stored in localStorage                              │
│  URL cleaned to: curiouskelly.com/                              │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  ✅ USER IS LOGGED IN - Kelly OS appears                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supabase URL Configuration

### Site URL (Primary Redirect)

```
https://curiouskelly.com
```

### Redirect URLs (All Allowed Destinations)

| URL                                            | Purpose                          | Required |
| ---------------------------------------------- | -------------------------------- | -------- |
| `https://curiouskelly.com`                     | Production - main site           | ✅       |
| `https://curiouskelly.com/`                    | Production - with trailing slash | ✅       |
| `https://curiouskelly.com/welcome`             | Post-checkout success page       | ✅       |
| `https://curiouskelly.com/auth/callback`       | Explicit auth callback page      | ✅       |
| `http://localhost:4321`                        | Local development                | ✅       |
| `http://localhost:4321/`                       | Local dev with slash             | ✅       |
| `https://curiouskelly-1mv5-lotd.vercel.app/`   | Vercel preview                   | ✅       |
| `https://curiouskelly-1mv5-lotd.vercel.app/**` | Vercel preview wildcard          | ✅       |

### ❌ DELETE These Old URLs

```
https://curiouskelly.com/public/app.html      ← OLD, REMOVE
https://www.curiouskelly.com/public/app.html  ← OLD, REMOVE
http://localhost:5501/public/app.html         ← OLD, REMOVE
http://127.0.0.1:5501/public/app.html         ← OLD, REMOVE
```

---

## Google Cloud Console Configuration

### OAuth Consent Screen

- **App name:** Curious Kelly
- **User support email:** hello@curiouskelly.com
- **App domain:** curiouskelly.com
- **Privacy policy:** https://curiouskelly.com/privacy.html
- **Terms of service:** https://curiouskelly.com/terms.html

### OAuth Client (Web Application)

- **Name:** Curious Kelly Web
- **Client ID:** `207034676667-rk9q47ql261nrumngkni7fl7nmou3cld.apps.googleusercontent.com`

### Authorized Redirect URIs (in Google)

```
https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback
```

**⚠️ IMPORTANT:** Google only needs the Supabase callback URL. The redirect back to your app is handled by Supabase using the Site URL setting.

---

## Error Handling

### If user sees 404:

1. Check Supabase Site URL is `https://curiouskelly.com` (no path)
2. Check the page exists (index.astro handles root)
3. Check Redirect URLs include the destination

### If "redirect_uri_mismatch":

1. Ensure Google has: `https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback`
2. Wait 5 minutes for Google to propagate changes

### If tokens appear in URL but user not logged in:

1. Check Supabase anon key is correct
2. Check `detectSessionInUrl: true` in Supabase client config
3. Check browser console for errors

---

## File Locations

| File                             | Purpose                                      |
| -------------------------------- | -------------------------------------------- |
| `src/pages/index.astro`          | Main entry - handles OAuth callback via hash |
| `src/pages/auth/callback.astro`  | Explicit callback page (fallback)            |
| `src/lib/auth.ts`                | Auth library for programmatic use            |
| `public/lesson-player/js/app.js` | Kelly OS - receives user state               |

---

## Environment Variables

```bash
# Required for auth
PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
PUBLIC_SUPABASE_ANON_KEY=<your-anon-key>

# Required for server-side (webhooks)
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
```

---

## Testing Checklist

- [ ] Visit curiouskelly.com → See login page
- [ ] Click "Continue with Google" → Redirects to Google
- [ ] Complete Google login → Redirects back to curiouskelly.com
- [ ] URL is clean (no tokens visible) → ✅
- [ ] Kelly OS appears → ✅
- [ ] Refresh page → Still logged in → ✅
- [ ] Click "Log Out" → Back to login page → ✅

---

## Quick Fixes

### User stuck on 404 after OAuth:

```
1. Go to Supabase → Authentication → URL Configuration
2. Set Site URL to: https://curiouskelly.com
3. Remove any /public/app.html URLs from Redirect URLs
4. Save changes
5. User tries again
```

### User sees "Access Denied" from Google:

```
1. Go to Google Cloud Console → OAuth consent screen
2. Click "PUBLISH APP" to make it available to all users
3. Or add test users if in testing mode
```


