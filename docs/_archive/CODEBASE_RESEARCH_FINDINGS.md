# Codebase Research Findings

**Date:** November 24, 2025

## Project Structure Overview

This codebase contains **TWO separate deployment targets**:

### 1. **Root `public/` Directory** - Static HTML Files
- **Location:** `public/index.html` (and other static HTML files)
- **Purpose:** Standalone landing pages/apps
- **Deployment:** Can be deployed as static files to Vercel/Netlify/Cloudflare
- **API Calls:** Uses `/api/*` endpoints which route to serverless functions

### 2. **`daily-lesson-marketing/` Directory** - Astro Project
- **Location:** `daily-lesson-marketing/src/`
- **Purpose:** Marketing site built with Astro
- **Deployment:** Astro builds to static + serverless functions
- **API Routes:** Uses Astro's `APIRoute` pattern in `src/pages/api/`

---

## API Routing Architecture

### Current Pattern (Root Level)

```
functions/handlers/          → Framework-agnostic handlers
  ├── lead.ts
  ├── rum.ts
  ├── stripe-checkout.ts     ✅ (NEW)
  ├── stripe-session.ts      ✅ (NEW)
  └── waitlist.ts            ✅ (NEW)

functions/vercel/api/        → Vercel-specific wrappers
  ├── lead.ts
  ├── rum.ts
  ├── stripe-checkout.ts     ✅ (NEW)
  ├── stripe-session.ts      ✅ (NEW)
  └── waitlist.ts            ✅ (NEW)

api/                        → Root re-exports (for Vercel)
  ├── lead.ts               → exports from functions/vercel/api/lead
  ├── rum.ts                → exports from functions/vercel/api/rum
  ├── stripe-checkout.ts    ✅ (NEW) → exports from functions/vercel/api/stripe-checkout
  ├── stripe-session.ts     ✅ (NEW) → exports from functions/vercel/api/stripe-session
  └── waitlist.ts           ✅ (NEW) → exports from functions/vercel/api/waitlist
```

### How It Works

1. **Vercel Deployment:**
   - Vercel automatically routes `/api/*` requests to files in `api/`
   - These files re-export Vercel functions from `functions/vercel/api/`
   - Functions use `@vercel/node` types (`VercelRequest`, `VercelResponse`)

2. **Netlify Deployment:**
   - `netlify.toml` redirects `/api/*` to `/.netlify/functions/*`
   - Functions in `functions/netlify/` handle requests
   - **Note:** Need to create Netlify wrappers for new endpoints

3. **Cloudflare Pages:**
   - Functions in `functions/cloudflare/api/` handle requests
   - **Note:** Need to create Cloudflare wrappers for new endpoints

---

## Existing Stripe Integration

### Found in `daily-lesson-marketing/src/pages/api/create-checkout.ts`

**Pattern:** Astro `APIRoute` (different from Vercel functions)

```typescript
import type { APIRoute } from 'astro';
import Stripe from 'stripe';

export const POST: APIRoute = async ({ request }) => {
  // Uses Astro's request/response pattern
  // Environment: import.meta.env.STRIPE_SECRET_KEY
}
```

**Differences from my implementation:**
- Uses `import.meta.env` (Astro) vs `process.env` (Node/Vercel)
- Uses Astro's `APIRoute` type vs Vercel's `VercelRequest/VercelResponse`
- Returns `Response` directly vs Vercel's `res.send()`

---

## Frontend API Calls

### In `public/index.html` (Static HTML)

```javascript
// Calls /api/stripe-checkout
fetch('/api/stripe-checkout', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ planType, customerEmail })
})
```

**This will work IF:**
- ✅ Deployed to Vercel (routes `/api/*` to serverless functions)
- ⚠️ Deployed to Netlify (needs redirect in `netlify.toml`)
- ⚠️ Deployed to Cloudflare (needs function in `functions/cloudflare/api/`)

### In `daily-lesson-marketing` (Astro Project)

Uses Astro API routes at `src/pages/api/create-checkout.ts` - already exists!

---

## Dependencies

### Root `package.json`
- **No Stripe dependency** ❌
- **No `@vercel/node` dependency** ❌
- Only dev dependencies (TypeScript, ESLint, etc.)

### `daily-lesson-marketing/package.json`
- ✅ `stripe: ^19.3.1` (already installed)
- ✅ `@astrojs/vercel: ^8.0.0` (Astro adapter)

---

## What's Missing

### 1. Root Package Dependencies
**Need to add to root `package.json`:**
```json
{
  "dependencies": {
    "stripe": "^19.3.1",
    "@vercel/node": "^3.0.0"
  }
}
```

### 2. Netlify Wrappers
**Need to create:**
- `functions/netlify/stripe-checkout.ts`
- `functions/netlify/stripe-session.ts`
- `functions/netlify/waitlist.ts`

**And update `netlify.toml`:**
```toml
[[redirects]]
from = "/api/stripe-checkout"
to = "/.netlify/functions/stripe-checkout"
status = 200
force = true
```

### 3. Cloudflare Wrappers
**Need to create:**
- `functions/cloudflare/api/stripe-checkout.ts`
- `functions/cloudflare/api/stripe-session.ts`
- `functions/cloudflare/api/waitlist.ts`

### 4. Astro API Routes (Optional)
**If `public/index.html` is served through Astro:**
- Could create `daily-lesson-marketing/src/pages/api/stripe-checkout.ts`
- But `create-checkout.ts` already exists, so might be redundant

---

## Deployment Target Analysis

### For `public/index.html`:

**Option A: Static File Deployment**
- Deploy `public/` folder as static files
- API routes must be serverless functions (Vercel/Netlify/Cloudflare)
- ✅ My Vercel implementation works
- ⚠️ Need Netlify/Cloudflare wrappers

**Option B: Astro Integration**
- Move `public/index.html` into `daily-lesson-marketing/src/pages/`
- Use Astro API routes instead
- Would need to create Astro versions of endpoints

**Current State:** `public/index.html` is standalone static HTML

---

## Recommendations

### Immediate Actions:

1. **Add dependencies to root `package.json`:**
   ```bash
   pnpm add stripe @vercel/node
   ```

2. **Test Vercel deployment:**
   - Deploy `public/` folder to Vercel
   - Test `/api/stripe-checkout` endpoint
   - Verify routing works

3. **Create Netlify wrappers** (if using Netlify):
   - Copy handler pattern from `functions/netlify/lead.ts`
   - Create wrappers for stripe-checkout, stripe-session, waitlist
   - Update `netlify.toml` redirects

4. **Create Cloudflare wrappers** (if using Cloudflare):
   - Copy handler pattern from `functions/cloudflare/api/lead.ts`
   - Create wrappers for stripe-checkout, stripe-session, waitlist

### Long-term Considerations:

1. **Unify API patterns:**
   - Consider if `public/index.html` should be part of Astro project
   - Or keep separate but ensure all deployment targets supported

2. **Environment variables:**
   - Ensure `.env` has all Stripe keys
   - Document which keys needed for which deployment target

---

## Files Created (Status)

✅ **Complete:**
- `functions/handlers/stripe-checkout.ts`
- `functions/handlers/stripe-session.ts`
- `functions/handlers/waitlist.ts`
- `functions/vercel/api/stripe-checkout.ts`
- `functions/vercel/api/stripe-session.ts`
- `functions/vercel/api/waitlist.ts`
- `api/stripe-checkout.ts` (re-export)
- `api/stripe-session.ts` (re-export)
- `api/waitlist.ts` (re-export)

✅ **Complete:**
- Netlify wrappers created
- Cloudflare wrappers created
- `netlify.toml` redirects updated

⚠️ **Still Missing:**
- Root `package.json` dependencies (stripe, @vercel/node, @netlify/functions)

---

## Questions to Resolve

1. **Which deployment target is primary?** (Vercel/Netlify/Cloudflare)
2. **Is `public/index.html` deployed separately or through Astro?**
3. **Should I create Netlify/Cloudflare wrappers now or wait?**
4. **Should I add Stripe to root `package.json` or only in `daily-lesson-marketing/`?**

---

## Next Steps

1. ✅ API structure matches existing pattern
2. ⏳ Add dependencies to root `package.json`
3. ⏳ Test Vercel deployment
4. ⏳ Create Netlify wrappers (if needed)
5. ⏳ Create Cloudflare wrappers (if needed)
6. ⏳ Update `netlify.toml` redirects (if using Netlify)

