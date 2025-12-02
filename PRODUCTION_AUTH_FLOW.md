# 🚀 Production Authentication Flow - curiouskelly.com

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION (curiouskelly.com)                     │
│                                                                      │
│  public/index.html          OAuth Providers         public/app.html │
│  (Login Page)              (Google/Apple)          (Kelly OS)       │
│       │                         │                       │           │
│       │  1. User clicks         │                       │           │
│       │     "Google" button     │                       │           │
│       │─────────────────────────►                       │           │
│       │                         │                       │           │
│       │  2. Google OAuth        │                       │           │
│       │     authenticates       │                       │           │
│       │                         │                       │           │
│       │  3. Supabase receives   │                       │           │
│       │     token, redirects    │                       │           │
│       │                         │─────────────────────► │           │
│       │                         │  to /app.html         │           │
│       │                                                 │           │
│       │  4. app.html detects                            │           │
│       │     session via                                 │           │
│       │     detectSessionInUrl                          │           │
│       │                                                 │           │
│       │  5. User sees Kelly OS! ◄───────────────────────│           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Files in Production

| File                    | URL                               | Purpose                       |
| ----------------------- | --------------------------------- | ----------------------------- |
| `public/index.html`     | `curiouskelly.com/`               | Login page with OAuth buttons |
| `public/app.html`       | `curiouskelly.com/app.html`       | Kelly OS (lesson player)      |
| `public/dashboard.html` | `curiouskelly.com/dashboard.html` | User dashboard                |
| `public/js/auth.js`     | -                                 | Auth utilities (imports)      |
| `public/js/api.js`      | -                                 | Supabase data API             |

## Supabase Configuration Required

### Step 1: Go to Supabase Dashboard

https://supabase.com/dashboard/project/tvjalxxsyryjphkforjv/auth/url-configuration

### Step 2: Set Site URL

```
https://curiouskelly.com
```

### Step 3: Add Redirect URLs (ALL of these!)

```
https://curiouskelly.com/app.html
https://www.curiouskelly.com/app.html
https://curiouskelly.com/
https://www.curiouskelly.com/
https://curiouskelly.com/dashboard.html
https://www.curiouskelly.com/dashboard.html
```

### Step 4: For Local Development (optional)

```
http://localhost:4322/app.html
http://localhost:4322/
```

## Code Changes Made (2024-11-27)

### 1. `public/index.html` - Login page

- ✅ OAuth redirects to `/app.html` (not `/public/app.html`)
- ✅ Email magic link redirects to `/app.html`
- ✅ Session check redirects to `/app.html`

### 2. `public/app.html` - Kelly OS

- ✅ Supabase client configured with `detectSessionInUrl: true`
- ✅ Uses PKCE flow for security
- ✅ `init()` function checks session on load
- ✅ Creates user profile if doesn't exist
- ✅ Handles guest mode

## Deployment

The `public/` folder deploys to `curiouskelly.com` via:

- **Cloudflare Pages** (most likely) or
- **Static hosting** configured in the repo

To deploy changes:

1. Commit and push to main branch
2. Cloudflare/hosting auto-deploys

## Troubleshooting

### "404 Not Found" after OAuth

- Check Supabase Redirect URLs include `/app.html`
- Ensure the deployed site has `app.html` at root

### "Session not found" in app.html

- Verify Supabase client has `detectSessionInUrl: true`
- Check browser console for errors
- Clear browser storage and try again

### OAuth button does nothing

- Check browser console for CORS errors
- Verify Supabase URL and key are correct
- Test with a different OAuth provider

## Contact

For issues: hello@curiouskelly.com








