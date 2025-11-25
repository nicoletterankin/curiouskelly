# 🔍 Curious Kelly Platform - Comprehensive Technical Audit

**Date:** December 2025  
**Audit Scope:** Complete codebase analysis for production deployment readiness

---

## 1. PROJECT STRUCTURE

### Directory Organization

```
UI-TARS-desktop/
├── public/                          # Main deployment directory (Netlify)
│   ├── index.html                  # Landing page / Marketing site
│   ├── app.html                    # Main lesson player app
│   ├── player.html                 # Alternative player interface
│   ├── calendar.html               # Lesson calendar view
│   ├── dashboard.html              # User dashboard
│   ├── assets/                     # Static assets (images, Unity builds)
│   │   ├── kelly_canonical/        # Kelly avatar images
│   │   └── unity/                  # Unity WebGL builds
│   │       ├── kelly-v1/           # Unity build v1 (with kbridge.js)
│   │       └── kelly-live/         # Unity build v2 (live version)
│   ├── js/                         # Frontend JavaScript modules
│   │   ├── auth.js                 # Supabase authentication
│   │   ├── api.js                  # API client functions
│   │   ├── calendar.js             # Calendar functionality
│   │   └── neural-link.js          # External provider connections
│   ├── lessons/                    # Lesson content (DNA files)
│   └── unity/                      # Unity integration files
│
├── app/                            # Alternative app shell (unified shell)
│   ├── index.html                  # App entry point
│   ├── unity-bridge.js             # Unity communication bridge
│   └── supabase-service.js         # Supabase service layer
│
├── api/                            # Vercel API routes (TypeScript)
│   ├── stripe-checkout.ts          # Stripe checkout endpoint
│   ├── stripe-session.ts           # Stripe session management
│   ├── waitlist.ts                 # Waitlist signup
│   ├── lead.ts                     # Lead capture
│   └── rum.ts                      # Real User Monitoring
│
├── functions/                      # Serverless function handlers
│   ├── handlers/                   # Core business logic
│   │   ├── stripe-checkout.ts      # Stripe checkout handler
│   │   └── ...
│   ├── netlify/                    # Netlify function wrappers
│   ├── vercel/                     # Vercel function wrappers
│   └── cloudflare/                 # Cloudflare Pages wrappers
│
├── daily-lesson-marketing/         # Astro-based marketing site
│   ├── public/                     # Static assets
│   ├── src/                        # Astro components
│   └── vercel.json                 # Vercel deployment config
│
├── curious-kellly/                 # Legacy lesson player
│   ├── backend/                    # Backend services
│   └── lesson-player-v2/           # Lesson player v2
│
├── digital-kelly/                  # Unity project files
│   └── engines/                    # Unity engine builds
│
├── supabase/                       # Database migrations
│   └── migrations/                 # SQL migration files
│
└── docs/                           # Documentation
```

### Main Entry Points

| File | Purpose | Location |
|------|---------|----------|
| **Landing Page** | Marketing/Login page | `public/index.html` |
| **Main App** | Lesson player interface | `public/app.html` |
| **Alternative App** | Unified shell app | `app/index.html` |
| **Marketing Site** | Astro-based site | `daily-lesson-marketing/` |

### Folder Purposes

- **Marketing Site:** `public/index.html` (Netlify) + `daily-lesson-marketing/` (Astro/Vercel)
- **Player App:** `public/app.html` (main) + `app/index.html` (alternative)
- **Admin Dashboard:** `public/dashboard.html` (basic)
- **Unity Integration:** `public/unity/kelly-v1/` and `public/unity/kelly-live/`
- **API Routes:** `api/` (Vercel) + `functions/` (multi-platform)

---

## 2. DEPLOYMENT & HOSTING

### Hosting Services Configured

#### 1. **Netlify** (Primary - Marketing Site)
- **Config:** `netlify.toml`
- **Build Command:** `git config --global lfs.fetchexclude '*' && git lfs install --skip-smudge || true`
- **Publish Directory:** `public`
- **Functions Directory:** `functions/netlify`
- **Status:** ✅ Configured

**API Redirects:**
- `/api/stripe-checkout` → `/.netlify/functions/stripe-checkout`
- `/api/waitlist` → `/.netlify/functions/waitlist`
- `/api/lead` → `/.netlify/functions/lead`
- `/api/rum` → `/.netlify/functions/rum`

#### 2. **Vercel** (Secondary - Astro Marketing Site)
- **Config:** `daily-lesson-marketing/vercel.json`
- **Build Command:** `npm run build`
- **Output Directory:** `dist`
- **Framework:** Astro
- **Status:** ✅ Configured

**Unity Headers:**
- Cross-Origin-Opener-Policy: `same-origin`
- Cross-Origin-Embedder-Policy: `require-corp`

#### 3. **Railway** (Backend Services)
- **Config:** `curious-kellly/backend/railway.json`
- **Status:** ⚠️ Configured but not primary

### Multiple Deployments

**Yes, there are multiple deployments:**

1. **Netlify:** `public/` directory (static HTML + Netlify Functions)
2. **Vercel:** `daily-lesson-marketing/` (Astro site)
3. **Railway:** `curious-kellly/backend/` (backend services)

**Recommendation:** Consolidate to single deployment strategy to avoid confusion.

---

## 3. ENVIRONMENT VARIABLES & API KEYS

### Required Environment Variables

#### 🔴 CRITICAL SECRETS (Server-side only)

```bash
# Stripe Payment Processing
STRIPE_SECRET_KEY=sk_live_...              # From Stripe Dashboard → Developers → API keys
STRIPE_WEBHOOK_SECRET=whsec_...            # From Stripe Dashboard → Developers → Webhooks
STRIPE_PRICE_MONTHLY=price_...             # Monthly subscription price ID
STRIPE_PRICE_ANNUAL=price_...              # Annual subscription price ID
STRIPE_PRICE_FAMILY=price_...              # Family plan price ID (optional)
STRIPE_PRICE_GIFT=price_...                # Gift purchase price ID (optional)

# Supabase Database (Server-side)
SUPABASE_SERVICE_ROLE_KEY=eyJ...            # From Supabase Dashboard → Settings → API
SUPABASE_DB_URL=postgresql://...           # Database connection string

# Cloudflare R2 (Backups)
CLOUDFLARE_R2_ACCESS_KEY=...
CLOUDFLARE_R2_SECRET_KEY=...
CLOUDFLARE_R2_ENDPOINT=https://...r2.cloudflarestorage.com
CLOUDFLARE_R2_BUCKET=...
```

#### 🟢 PUBLIC SECRETS (Safe to expose to browser)

```bash
# Supabase (Public)
PUBLIC_SUPABASE_URL=https://tvjalxxsyryjphkforjv.supabase.co
PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Site Configuration
PUBLIC_SITE_URL=https://curiouskelly.com

# Analytics (Gated by consent)
PUBLIC_GTM_ID=GTM-...
PUBLIC_GA4_ID=G-...
PUBLIC_META_PIXEL_ID=...
PUBLIC_TIKTOK_PIXEL_ID=...
PUBLIC_TWITTER_PIXEL_ID=...

# Security/Captcha
TURNSTILE_SITE_KEY=...
TURNSTILE_SECRET_KEY=...
PUBLIC_RECAPTCHA_SITE_KEY=...
RECAPTCHA_SECRET_KEY=...
```

### Hardcoded Values Found

⚠️ **CRITICAL:** Supabase credentials are hardcoded in:
- `public/index.html` (lines 599-600)
- `public/app.html` (lines 342-343)
- `public/js/auth.js` (lines 8-9)

**Action Required:** Move to environment variables or config file.

### API Endpoints

**Stripe:**
- Checkout: `/api/stripe-checkout` (POST)
- Session: `/api/stripe-session` (GET)

**Supabase:**
- URL: `https://tvjalxxsyryjphkforjv.supabase.co`
- Auth: Handled via Supabase JS client

**No hardcoded API endpoints found** (except Supabase URLs which should be env vars).

---

## 4. SUPABASE INTEGRATION

### Database Schema

**Location:** `supabase-schema.sql`

#### Tables

1. **`public.users`** (extends `auth.users`)
   - `id` (UUID, references auth.users)
   - `email`, `name`, `age`
   - `subscription_tier` ('free', 'annual', 'gift', 'enterprise')
   - `subscription_status` ('active', 'inactive', 'cancelled', 'expired')
   - `stripe_customer_id`
   - `current_day`, `streak_days`
   - `last_lesson_at`

2. **`public.lessons`**
   - `id`, `day_number` (unique)
   - `title`, `subtitle`
   - `content` (JSONB - PhaseDNA structure)
   - `audio_url`, `duration_seconds`
   - `difficulty`, `tags[]`
   - `is_published`

3. **`public.user_progress`**
   - `user_id`, `lesson_id`
   - `completed`, `progress_percent`
   - `last_position_seconds`, `time_spent_seconds`
   - `completed_at`, `started_at`

4. **`public.affiliates`**
   - `user_id`, `referral_code`
   - `tier` ('scholar', 'fellow', 'ambassador', 'founding')
   - `commission_rate`, `total_referrals`
   - `lifetime_earnings`

5. **`public.referrals`**
   - `affiliate_id`, `referred_user_id`
   - `subscription_value`, `commission_earned`
   - `status` ('pending', 'active', 'cancelled', 'paid')

6. **`public.affiliate_applications`**
   - Application form submissions

7. **`public.enterprise_inquiries`**
   - Enterprise lead capture

8. **`public.newsletter_subscribers`**
   - Email list management

9. **`public.analytics_events`**
   - Event tracking (optional)

### Authentication Flow

**Providers Supported:**
- ✅ Email/Password (Magic Link via OTP)
- ✅ Google OAuth
- ✅ Apple OAuth
- ✅ GitHub OAuth
- ✅ Microsoft/Azure OAuth

**Flow:**
1. User clicks OAuth provider button
2. Supabase redirects to provider
3. Provider redirects back to `/public/app.html`
4. Session stored in browser
5. User profile created in `public.users` table via trigger

**Row Level Security (RLS):**
- ✅ Enabled on all tables
- ✅ Users can only view/update own data
- ✅ Public read access for published lessons

### Data Stored

- **User Profiles:** Email, name, age, subscription status
- **Lesson Variants:** PhaseDNA JSON in `lessons.content`
- **Progress Tracking:** Per-user, per-lesson progress in `user_progress`
- **Streaks:** Calculated via trigger on `user_progress` updates

---

## 5. STRIPE INTEGRATION

### Products/Prices

**Not defined in code** - Must be created in Stripe Dashboard:

1. **Monthly Plan** - `$9.99/month` (referenced in `index.html` line 434)
2. **Annual Plan** - `$99.99/year` (referenced in `index.html` line 431)
3. **Gift Plan** - One-time payment (referenced in gift modal)

**Price IDs Required:**
- `STRIPE_PRICE_MONTHLY`
- `STRIPE_PRICE_ANNUAL`
- `STRIPE_PRICE_GIFT`

### Webhook Handlers

**Location:** `functions/handlers/stripe-checkout.ts`

**Current Implementation:**
- ✅ Checkout session creation
- ✅ Gift purchase support
- ✅ Subscription creation
- ⚠️ **Webhook handler NOT found** - Need to add webhook endpoint

**Missing:**
- Webhook handler for `checkout.session.completed`
- Webhook handler for `customer.subscription.updated`
- Webhook handler for `customer.subscription.deleted`

**Action Required:** Create webhook handler at `/api/stripe-webhook` to:
1. Update `users.subscription_status` on payment success
2. Create Stripe customer record
3. Handle subscription cancellations

### Subscription Tiers

Based on code analysis:
- **Free:** Default tier (no payment)
- **Monthly:** `$9.99/month` (mentioned in UI)
- **Annual:** `$99.99/year` (mentioned in UI)
- **Gift:** One-time payment (gift purchase flow)
- **Enterprise:** Mentioned in schema but no pricing defined

---

## 6. 3D AVATAR / UNITY INTEGRATION

### Unity WebGL Build Files

**Location:** `public/unity/`

#### Build Versions:

1. **kelly-v1** (`public/unity/kelly-v1/`)
   - `kelly-v1.loader.js` ✅ EXISTS
   - `kelly-v1.framework.js` ✅ EXISTS
   - `kelly-v1.wasm` ✅ EXISTS
   - `kelly-v1.data` ✅ EXISTS
   - `kbridge.js` ✅ EXISTS (messaging bridge)
   - `index.html` ✅ EXISTS

2. **kelly-live** (`public/unity/kelly-live/`)
   - `Kelly_Web_Build.loader.js` ✅ EXISTS
   - `Kelly_Web_Build.framework.js.br` ✅ EXISTS (compressed)
   - `Kelly_Web_Build.wasm.br` ✅ EXISTS (compressed)
   - `Kelly_Web_Build.data.br` ✅ EXISTS (compressed)
   - `index.html` ✅ EXISTS
   - ⚠️ **No kbridge.js** in this version

### Web App ↔ Unity Communication

**Method:** `postMessage` API + `kbridge.js`

**Bridge Implementation:**
- **File:** `public/unity/kelly-v1/kbridge.js`
- **Purpose:** Messaging bridge between parent iframe and Unity WebGL

**Message Flow:**
```javascript
// Parent → Unity
window.postMessage({
  destination: 'kelly-webgl',
  type: 'kelly-load',
  payload: { lessonId: 'the-sun' }
}, '*');

// Unity → Parent
window.parent.postMessage({
  source: 'kelly-webgl',
  type: 'kelly-ready',
  status: 'ok'
}, '*');
```

**Unity Bridge Class:**
- **File:** `app/unity-bridge.js`
- **Features:**
  - WebSocket support (optional)
  - PostMessage support (primary)
  - Event routing
  - Connection status tracking

### Audio Files Location

**Lesson Audio:**
- **Path Pattern:** `lessons/audio/{lesson-slug}/{age-range}-{lang}-{phase}.mp3`
- **Example:** `lessons/audio/the-sun/18-35-en-welcome.mp3`

**Found Locations:**
- `lessons/audio/` ✅ EXISTS (multiple lessons)
- `daily-lesson-marketing/public/lessons/audio/` ✅ EXISTS
- `archived/lesson-player-OLD-20251121/lessons/audio/` ✅ EXISTS (archived)

**Audio Generation:**
- Script: `generate_lesson_audio_for_iclone.py`
- Uses ElevenLabs API for synthesis
- Exports WAV files for iClone/AccuLips

### Lesson Playback Trigger

**Flow:**
1. User selects lesson in `app.html`
2. `app.html` loads Unity iframe: `public/unity/kelly-v1/index.html`
3. Unity bridge (`kbridge.js`) sends `kelly-load` message
4. Unity player loads lesson audio from manifest
5. Audio plays with lip sync via Audio2Face

**Manifest Files:**
- Location: `lessons/manifests/{lesson-slug}-manifest.json`
- Contains audio URLs, phases, timing

---

## 7. AUTHENTICATION FLOW

### User Journey Map

```
1. Landing Page (index.html)
   ↓
2. User clicks "Start Learning Now" (Guest Mode)
   OR clicks OAuth provider (Google/Apple/GitHub/Microsoft)
   OR enters email (Magic Link)
   ↓
3. Authentication
   - Guest: localStorage.setItem('guestMode', 'true')
   - OAuth: Supabase redirect → Provider → Redirect back
   - Email: Magic link sent → Click link → Redirect back
   ↓
4. Redirect to app.html
   ↓
5. App Initialization
   - Check session (Supabase)
   - Check guestMode (localStorage)
   - If neither → Redirect to index.html
   ↓
6. Load User Profile
   - Fetch from public.users table
   - Create profile if doesn't exist (trigger)
   ↓
7. Load Today's Lesson
   - Fetch from public.lessons (day_number = user.current_day)
   ↓
8. Initialize Unity Player
   - Load Unity iframe
   - Establish bridge connection
   - Load lesson audio
   ↓
9. Lesson Playback
   - Unity renders Kelly avatar
   - Audio plays with lip sync
   - User interacts via chat input
```

### Pages Requiring Authentication

| Page | Auth Required | Guest Allowed |
|------|--------------|---------------|
| `index.html` | ❌ No | ✅ Yes (landing) |
| `app.html` | ⚠️ Partial | ✅ Yes (guest mode) |
| `dashboard.html` | ✅ Yes | ❌ No |
| `calendar.html` | ⚠️ Partial | ✅ Yes (limited) |
| `player.html` | ⚠️ Partial | ✅ Yes (guest mode) |

**Guest Mode:**
- Allows access to today's lesson only
- No progress tracking
- No streak tracking
- No personalized recommendations

### Google OAuth Configuration

**Status:** ✅ Configured in Supabase

**Redirect URL:**
- Success: `${window.location.origin}/public/app.html`
- Configured in: `public/index.html` line 670

**OAuth Scopes:**
- `access_type: 'offline'`
- `prompt: 'consent'`

**Action Required:** Verify OAuth credentials in Supabase Dashboard → Authentication → Providers → Google

---

## 8. MISSING RESOURCES (CRITICAL)

### 404 Errors Identified

#### ❌ `google_color.svg`
- **Status:** NOT FOUND
- **Referenced In:** Not found in codebase search
- **Action:** Create or remove reference

#### ⚠️ `18-35-en-welcome.mp3` (and other lesson audio)
- **Status:** PARTIALLY MISSING
- **Expected Path:** `lessons/audio/{lesson-slug}/18-35-en-welcome.mp3`
- **Found Locations:**
  - ✅ `lessons/audio/applied-mathematics-math-in-the-real-world/18-35-en-welcome.mp3` EXISTS (complete set)
  - ❌ `lessons/audio/the-sun/` - Only `metadata.json` exists, NO MP3 files
  - ❌ `lessons/audio/poetry/` - Only `metadata.json` exists, NO MP3 files
  - ❌ `lessons/audio/nutrition-science/` - Only `metadata.json` exists, NO MP3 files
  - ❌ `lessons/audio/negotiation-skills/` - Only `metadata.json` exists, NO MP3 files
  - ❌ `lessons/audio/molecular-biology/` - Only `metadata.json` exists, NO MP3 files
  - ❌ `lessons/audio/genetic-engineering-editing-the-code-of-life/` - Only `metadata.json` exists, NO MP3 files
  - ❌ `lessons/audio/dance-expression/` - Only `metadata.json` exists, NO MP3 files
  - ❌ `lessons/audio/creative-writing/` - Only `metadata.json` exists, NO MP3 files
- **Action:** **CRITICAL** - Generate missing audio files using `generate_lesson_audio_for_iclone.py` or update manifests to point to existing audio

#### ✅ `kbridge.js`
- **Status:** EXISTS
- **Location:** `public/unity/kelly-v1/kbridge.js`
- **Action:** ✅ No action needed

#### ⚠️ `kelly-vi.loader.js`
- **Status:** TYPO - Should be `kelly-v1.loader.js`
- **Actual File:** `public/unity/kelly-v1/kelly-v1.loader.js` ✅ EXISTS
- **Action:** Fix any references using incorrect name

#### ✅ `style.css`
- **Status:** EXISTS
- **Location:** `public/unity/kelly-v1/TemplateData/style.css`
- **Also:** `public/unity/kelly-live/TemplateData/style.css`
- **Action:** ✅ No action needed

### Resource Path Issues

**Unity Build Paths:**
- Some references use relative paths that may break in production
- Check `public/app.html` Unity loader configuration
- Verify `buildUrl` variable points to correct Unity build directory

**Audio File Paths:**
- Manifests reference audio files that may not exist yet
- Check: `lessons/manifests/*-manifest.json`
- Verify all referenced audio files exist in `lessons/audio/`

---

## 9. BUILD & DEPENDENCY ISSUES

### Package Managers

**Multiple package managers detected:**
- Root: `pnpm` (via `pnpm-workspace.yaml`)
- `daily-lesson-marketing/`: `npm` (via `package-lock.json`)
- Some subdirectories: `npm`

**Recommendation:** Standardize on single package manager.

### Root Dependencies

**File:** `package.json`

```json
{
  "name": "daily-lesson-platform",
  "packageManager": "pnpm@9.9.0",
  "engines": {
    "node": ">=20.11.1",
    "pnpm": ">=9.9.0"
  }
}
```

**Scripts:**
- `dev`: Runs parallel dev servers
- `build`: Recursive build
- `vercel-build`: Install + recursive build

### Marketing Site Dependencies

**File:** `daily-lesson-marketing/package.json`

**Key Dependencies:**
- `astro`: ^5.0.0
- `@supabase/supabase-js`: ^2.84.0
- `stripe`: ^19.3.1
- `react`: ^18.3.0

**Engines:**
- `node`: >=22.0.0
- `npm`: >=9.0.0

### Conflicting Versions

⚠️ **Node Version Mismatch:**
- Root requires: `>=20.11.1`
- Marketing site requires: `>=22.0.0`

**Action Required:** Align Node version requirements.

### Missing Packages

**No `.env.example` file found** - Should create one with all required variables documented.

**Action Required:** Create `.env.example` based on `SECRETS_MASTER_REFERENCE.md`.

---

## 10. CURRENT BROKEN STATE

### Incomplete Features

1. **Stripe Webhook Handler**
   - Checkout creation works ✅
   - Webhook processing missing ❌
   - Subscription status updates not automated

2. **Guest Mode Limitations**
   - Works for viewing lessons ✅
   - Progress not saved ❌
   - No upgrade prompt ❌

3. **Unity Build Version Confusion**
   - Two builds: `kelly-v1` and `kelly-live`
   - Unclear which is production
   - `kelly-live` missing `kbridge.js`

### Commented-Out Code

**Found in:** `public/app.html`
- Some Unity initialization code may be commented
- Check for `// TODO` or `/* */` blocks

### Multiple index.html Files

**Found:**
1. `public/index.html` - Main landing page ✅
2. `app/index.html` - Alternative app shell ⚠️
3. `daily-lesson-marketing/public/index.html` - Astro site ⚠️

**Issue:** Confusion about which is the "real" entry point.

**Recommendation:** 
- Use `public/index.html` as primary landing
- Document `app/index.html` as alternative/experimental
- Keep Astro site separate (`daily-lesson-marketing/`)

### Mixed Development Branches

**Archived Directories Found:**
- `_archive/` - Old code
- `_archived_legacy/` - Legacy code
- `archived/` - Archived lesson player

**Issue:** May cause confusion about which code is active.

**Recommendation:** Clean up or clearly document archived code.

### Experimental Features

**Found:**
- `app/unity-bridge.js` - Advanced Unity bridge with WebSocket support
- `app/index.html` - Unified shell architecture
- Multiple Unity builds (v1 vs live)

**Status:** Features exist but unclear if production-ready.

---

## 🔧 CRITICAL FIXES REQUIRED

### Priority 1 (Blocking Production)

1. **Move Supabase credentials to environment variables**
   - Remove hardcoded values from `public/index.html` and `public/app.html`
   - Use `PUBLIC_SUPABASE_URL` and `PUBLIC_SUPABASE_ANON_KEY`

2. **Create Stripe webhook handler**
   - Endpoint: `/api/stripe-webhook`
   - Handle `checkout.session.completed`
   - Update `users.subscription_status`

3. **Fix missing audio files** ⚠️ **CRITICAL**
   - **9 out of 10 lessons missing audio files** (only `applied-mathematics` has complete audio)
   - Generate missing audio files using `generate_lesson_audio_for_iclone.py`
   - Or update manifests to use existing audio from `archived/lesson-player-OLD-20251121/lessons/audio/`
   - Affected lessons: the-sun, poetry, nutrition-science, negotiation-skills, molecular-biology, genetic-engineering, dance-expression, creative-writing

4. **Create `.env.example`**
   - Document all required environment variables
   - Include instructions for obtaining each secret

### Priority 2 (Important)

5. **Consolidate Unity builds**
   - Choose one production build (`kelly-v1` or `kelly-live`)
   - Ensure `kbridge.js` exists in chosen build
   - Update all references

6. **Standardize Node version**
   - Align requirements across all `package.json` files
   - Update CI/CD to use consistent version

7. **Document deployment strategy**
   - Choose primary hosting (Netlify vs Vercel)
   - Document which directories deploy where
   - Update deployment guides

### Priority 3 (Nice to Have)

8. **Clean up archived code**
   - Move to separate branch or repository
   - Or clearly mark as archived

9. **Add error handling**
   - Graceful fallbacks for missing resources
   - User-friendly error messages

10. **Add monitoring**
    - Error tracking (Sentry)
    - Performance monitoring
    - User analytics

---

## 📊 SUMMARY

### What's Working ✅

- Supabase authentication (OAuth + Magic Link)
- Stripe checkout session creation
- Unity WebGL builds exist
- Lesson content structure (PhaseDNA)
- Database schema is complete
- Basic app flow works

### What's Broken ❌

- Hardcoded Supabase credentials
- Missing Stripe webhook handler
- Some audio files may be missing
- Multiple deployment configs causing confusion
- Node version mismatch

### What's Unclear ⚠️

- Which Unity build is production (`kelly-v1` vs `kelly-live`)
- Which deployment is primary (Netlify vs Vercel)
- Which `index.html` is the real entry point
- Status of experimental features (`app/` directory)

---

## 🎯 NEXT STEPS

1. **Immediate:** Fix Priority 1 issues (credentials, webhooks, audio)
2. **Short-term:** Consolidate deployments and Unity builds
3. **Long-term:** Clean up codebase, add monitoring, improve error handling

---

**Document Generated:** December 2025  
**Last Updated:** Based on current codebase analysis

