# OAuth Provider Setup Guide for Curious Kelly

This guide will walk you through setting up each OAuth provider for production. Complete these in order of priority.

---

## 🔴 CRITICAL: Supabase URL Configuration

**Before setting up any provider**, verify your Supabase redirect URLs:

1. Go to: https://app.supabase.com/project/tvjalxxsyryjphkforjv/auth/url-configuration
2. Add these to **Redirect URLs**:
   - `https://curiouskelly.com/public/app.html`
   - `https://www.curiouskelly.com/public/app.html`
   - `http://localhost:5501/public/app.html` (for local testing)
   - `http://127.0.0.1:5501/public/app.html` (for local testing)

---

## 1. Google OAuth (Already Working) ✅

**Status**: Should already be configured
**Verification**: Test login at https://curiouskelly.com

If you need to reconfigure:
1. Go to: https://console.cloud.google.com/apis/credentials
2. Find your OAuth 2.0 Client ID
3. Ensure **Authorized redirect URIs** includes:
   - `https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback`
4. Copy **Client ID** and **Client Secret** to Supabase

---

## 2. Apple Sign In (High Priority) 🍎

**Requirements**:
- Apple Developer Account ($99/year)
- Access to https://developer.apple.com

**Step-by-Step**:

### A. Create a Services ID
1. Go to: https://developer.apple.com/account/resources/identifiers/list/serviceId
2. Click **+** button
3. Select **Services IDs**, click Continue
4. Fill in:
   - **Description**: Curious Kelly
   - **Identifier**: `com.curiouskelly.signin` (or your bundle ID + `.signin`)
5. Check **Sign In with Apple**, click Configure
   - This opens a popup for "Web Authentication Configuration"
6. **Primary App ID**: Select your app's ID (or create one)
7. **Domains and Subdomains**: Add `tvjalxxsyryjphkforjv.supabase.co`
8. **Return URLs** (also called "Website URLs"): 
   - Add `https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback`
   - This is where Apple redirects users after authentication
9. Click **Next**, then **Done**, then **Continue**, then **Register**

### B. Create a Key for Client Secret
1. Go to: https://developer.apple.com/account/resources/authkeys/list
2. Click **+** button
3. **Key Name**: Curious Kelly Auth Key
4. Check **Sign In with Apple**, click Configure
5. Select your **Primary App ID**
6. Click Save, then Continue, then Register
7. **IMPORTANT**: Download the `.p8` key file (you can only download once!)
8. Note the **Key ID** (e.g., `ABC123DEFG`)

### C. Get Your Team ID
1. Go to: https://developer.apple.com/account
2. Top right corner shows your **Team ID** (e.g., `XYZ987TEAM`)

### D. Configure in Supabase
1. Go to: https://app.supabase.com/project/tvjalxxsyryjphkforjv/auth/providers
2. Find **Apple**, click to expand
3. Enable it
4. Fill in:
   - **Services ID**: `com.curiouskelly.signin` (from step A)
   - **Team ID**: Your Team ID (from step C)
   - **Key ID**: Your Key ID (from step B)
   - **Private Key**: Open the `.p8` file in a text editor, copy the entire contents including `-----BEGIN PRIVATE KEY-----` and `-----END PRIVATE KEY-----`
5. Click Save

---

## 3. GitHub OAuth (Easy Setup) 🐙

**Step-by-Step**:

1. Go to: https://github.com/settings/developers
2. Click **New OAuth App** (or use existing)
3. Fill in:
   - **Application name**: Curious Kelly
   - **Homepage URL**: `https://curiouskelly.com`
   - **Authorization callback URL**: `https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback`
4. Click **Register application**
5. Copy the **Client ID**
6. Click **Generate a new client secret**, copy the **Client Secret**
7. Go to Supabase: https://app.supabase.com/project/tvjalxxsyryjphkforjv/auth/providers
8. Find **GitHub**, click to expand
9. Enable it, paste **Client ID** and **Client Secret**
10. Click Save

---

## 4. Microsoft/Azure OAuth (Medium Priority) 🔷

**Requirements**:
- Microsoft/Azure account (free)
- Access to https://portal.azure.com

**Step-by-Step**:

1. Go to: https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade
2. Click **New registration**
3. Fill in:
   - **Name**: Curious Kelly
   - **Supported account types**: Accounts in any organizational directory and personal Microsoft accounts
   - **Redirect URI**: Select **Web**, enter `https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback`
4. Click **Register**
5. On the Overview page, copy the **Application (client) ID**
6. Click **Certificates & secrets** in the left sidebar
7. Click **New client secret**
8. **Description**: Curious Kelly Production
9. **Expires**: 24 months (or your preference)
10. Click **Add**, then copy the **Value** (client secret) immediately
11. Go to Supabase: https://app.supabase.com/project/tvjalxxsyryjphkforjv/auth/providers
12. Find **Azure**, click to expand
13. Enable it, paste:
    - **Client ID**: Application (client) ID from step 5
    - **Secret**: Client secret from step 10
14. Click Save

---

## 5. Email Magic Link (No External Setup) ✉️

**Already configured** in Supabase. Just verify:

1. Go to: https://app.supabase.com/project/tvjalxxsyryjphkforjv/auth/templates
2. Ensure **Confirm signup** and **Magic Link** templates are enabled
3. Customize email templates if desired (add branding)

---

## Testing Checklist

After configuring each provider, test:

- [ ] Can initiate login (button click)
- [ ] Redirected to provider's auth page
- [ ] After approval, redirected back to `curiouskelly.com/public/app.html`
- [ ] User info displayed correctly (name, email)
- [ ] Session persists on page reload
- [ ] Can sign out and sign back in

---

## Common Issues & Fixes

### "Invalid redirect URI"
- **Cause**: Redirect URL mismatch
- **Fix**: Ensure Supabase callback URL matches EXACTLY in provider settings

### "User already exists" (for email)
- **Cause**: User signed up with different provider using same email
- **Fix**: Expected behavior - Supabase will link accounts automatically

### Apple shows "Service is not available"
- **Cause**: Services ID not properly configured
- **Fix**: Verify domain and return URL in Apple Developer Console

### GitHub shows "Redirect URI mismatch"
- **Cause**: Wrong callback URL
- **Fix**: Must be `https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback` (no trailing slash)

---

## Priority Order

1. **Google** ✅ (Already done)
2. **GitHub** (5 minutes, easy)
3. **Microsoft/Azure** (10 minutes, medium)
4. **Apple** (20 minutes, requires paid account)

Start with GitHub to test the flow, then move to the others!

