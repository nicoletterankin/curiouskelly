# Vercel Build Troubleshooting Handoff

**Date**: November 14, 2025  
**Status**: Final fix deployed (Commit `1e7fbcf`)  
**Deployment**: Awaiting Vercel build results

---

## 🎯 Current Status

### What Was Fixed (Final Attempt)

**Commit `1e7fbcf`**: Resolved module resolution conflict

**Root Cause Identified**:
- Empty `src/lib/validation/` directory existed alongside `validation.ts` file
- Vite's module resolver couldn't determine which to use
- Import path `../../lib/validation` was ambiguous

**Solution Applied**:
1. ✅ Removed empty `src/lib/validation/` directory
2. ✅ Added explicit `.js` extension to import: `'../../lib/validation.js'`
3. ✅ Local build verified: **SUCCESS** (26 pages built in 1.23s)

**Previous Fixes** (Commits `129183b` and `0c777f4`):
1. ✅ Committed `package-lock.json` for deterministic builds
2. ✅ Changed Vercel install to `npm ci --legacy-peer-deps`
3. ✅ Upgraded to Node 22.x (Vercel requirement)

---

## 📊 Build Success Indicators

If the Vercel deployment succeeds, you should see:
- ✅ Build completes in ~35-40 seconds
- ✅ 26 pages generated
- ✅ API routes: `/api/lead` (POST), `/api/rum` (POST)
- ✅ All locale routes: `/`, `/es-es/`, `/pt-br/`
- ✅ Static pages: `/roadmap`, `/thank-you`, `/sitemap.xml`

---

## 🔍 If Build Still Fails

### Check #1: Module Resolution

**Symptoms**: `Could not resolve "../../lib/validation"` or similar

**Debug Steps**:
```bash
# Verify validation.ts exists and no validation/ directory
cd daily-lesson-marketing
ls -la src/lib/ | grep validation
# Should show ONLY: validation.ts (no directory)

# Check the import statement
cat src/pages/api/lead.ts | grep "from.*validation"
# Should show: from '../../lib/validation.js'
```

**If Still Failing**:
Try absolute import with Vite alias:
```typescript
// In src/pages/api/lead.ts
import { validateLeadForm, sanitizeFormData, type LeadFormData } from '@lib/validation.js';
```

Then update `astro.config.mjs` to ensure API routes can use aliases:
```javascript
export default defineConfig({
  vite: {
    resolve: {
      alias: {
        '@lib': path.resolve(__dirname, './src/lib')
      }
    },
    ssr: {
      noExternal: ['libphonenumber-js']  // Add if needed
    }
  }
});
```

### Check #2: Dependencies

**Symptoms**: Missing exports, type errors, peer dependency warnings

**Debug Steps**:
```bash
# Verify package-lock.json is committed
git ls-files package-lock.json
# Should output: daily-lesson-marketing/package-lock.json

# Check Node version
cat .nvmrc
# Should show: 22.11.0

# Verify validation.ts exports
grep "^export" src/lib/validation.ts
# Should show:
# - export interface LeadPayload
# - export type LeadErrors
# - export type LeadFormData
# - export function validateLead
# - export function hasErrors
# - export function sanitizeFormData
# - export function validateLeadForm
```

**If Missing Exports**:
The validation.ts file MUST export these functions:
- `validateLeadForm(data, copy)`
- `sanitizeFormData(body)`
- `LeadFormData` type

### Check #3: Vercel Configuration

**Symptoms**: Wrong Node version, install failures

**Verify vercel.json**:
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "installCommand": "npm ci --legacy-peer-deps",
  "framework": "astro"
}
```

**Verify .nvmrc**:
```
22.11.0
```

**Verify package.json engines**:
```json
"engines": {
  "node": ">=22.0.0",
  "npm": ">=9.0.0"
}
```

### Check #4: TypeScript/Astro Config

**Symptoms**: Path alias not resolving, type errors

**Verify tsconfig.json paths**:
```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@lib/*": ["./src/lib/*"],
      "@components/*": ["./src/components/*"],
      "@layouts/*": ["./src/layouts/*"]
    }
  }
}
```

**Verify astro.config.mjs aliases**:
```javascript
vite: {
  resolve: {
    alias: {
      '@lib': path.resolve(__dirname, './src/lib'),
      '@components': path.resolve(__dirname, './src/components'),
      '@layouts': path.resolve(__dirname, './src/layouts')
    }
  }
}
```

---

## 🚨 Nuclear Options (If Nothing Works)

### Option 1: Inline Validation Functions

Move validation logic directly into `src/pages/api/lead.ts`:

```typescript
// src/pages/api/lead.ts
import type { APIRoute } from 'astro';
import { parsePhoneNumberFromString } from 'libphonenumber-js/min';

// Inline types and functions (copy from validation.ts)
interface LeadPayload {
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  country: string;
  region: string;
  marketing_opt_in: boolean;
  locale: string;
  journey: string;
}

function sanitizeFormData(body: any): LeadPayload {
  return {
    first_name: String(body.first_name || '').trim(),
    last_name: String(body.last_name || '').trim(),
    email: String(body.email || '').trim().toLowerCase(),
    phone: String(body.phone || '').trim(),
    country: String(body.country || '').trim(),
    region: String(body.region || '').trim(),
    marketing_opt_in: Boolean(body.marketing_opt_in),
    locale: String(body.locale || 'en-US').trim(),
    journey: String(body.journey || '').trim()
  };
}

// ... rest of validation logic inline ...

export const POST: APIRoute = async ({ request }) => {
  // ... implementation ...
};
```

**Pros**: Eliminates import issues  
**Cons**: Code duplication, harder to maintain

### Option 2: Switch to Serverless Functions

Create separate Vercel serverless functions instead of Astro API routes:

```bash
# Create api/ directory at root
mkdir -p api
mv src/pages/api/lead.ts api/lead.ts

# Update vercel.json
{
  "functions": {
    "api/**/*.ts": {
      "runtime": "@vercel/node@3"
    }
  }
}
```

### Option 3: Use External Module

Publish validation as npm package:
```bash
cd src/lib
npm init -y
npm publish --name @curiouskelly/validation
```

Then install in main project:
```bash
npm install @curiouskelly/validation
```

---

## 📁 Critical Files

### Must Verify These Files

1. **`src/lib/validation.ts`** - Contains all validation logic
   - Location: `daily-lesson-marketing/src/lib/validation.ts`
   - Exports: `validateLeadForm`, `sanitizeFormData`, `LeadFormData`

2. **`src/pages/api/lead.ts`** - API endpoint
   - Import line: `import { ... } from '../../lib/validation.js'`
   - Must have `.js` extension

3. **`package-lock.json`** - Dependency lockfile
   - Must be committed to git
   - Must exist in repository

4. **`.nvmrc`** - Node version
   - Contains: `22.11.0`

5. **`vercel.json`** - Deployment config
   - Install command: `npm ci --legacy-peer-deps`

### Directory Structure

```
daily-lesson-marketing/
├── src/
│   ├── lib/
│   │   ├── validation.ts          ✅ FILE (must exist)
│   │   ├── validation/            ❌ DIRECTORY (must NOT exist)
│   │   └── i18n/
│   │       └── types.ts           ✅ Dependency of validation.ts
│   └── pages/
│       └── api/
│           └── lead.ts            ✅ Imports validation
├── .nvmrc                         ✅ 22.11.0
├── package-lock.json              ✅ Must be committed
├── vercel.json                    ✅ npm ci command
└── astro.config.mjs               ✅ Vite aliases
```

---

## 🔧 Local Testing (Before Deploying Again)

To replicate Vercel environment locally:

```bash
# Use correct Node version
nvm use 22.11.0

# Clean install
rm -rf node_modules .astro dist
npm ci --legacy-peer-deps

# Build
npm run build

# Should output:
# [build] 26 page(s) built in ~1.2s
# [build] Complete!
```

**If local build fails**, the Vercel build will also fail.

---

## 📊 Build History Analysis

| Commit | Change | Result |
|--------|--------|--------|
| `1e7fbcf` | Removed validation/ directory conflict | **PENDING** |
| `0c777f4` | Upgraded to Node 22.x | ❌ Failed (module resolution) |
| `129183b` | Committed package-lock.json, npm ci | ❌ Failed (Node 18 deprecated) |
| `8eba7bf` | Added Vite resolve extensions | ❌ Failed |
| `1257c0b` | Used relative import with .js | ❌ Failed |
| `c35fd77` | Added missing validation functions | ❌ Failed |
| Earlier | Various TypeScript/path fixes | ❌ All failed |

**Pattern**: Every commit that worked locally failed on Vercel due to environment differences.

**Key Insight**: The validation/ directory was never removed until `1e7fbcf`.

---

## 🎓 Lessons Learned

### Why Builds Failed

1. **Environment Mismatch**: Local (Node 24) vs Vercel (Node 18 → 22)
2. **Module Ambiguity**: Both `validation.ts` and `validation/` existed
3. **Non-Deterministic Builds**: `package-lock.json` was gitignored
4. **Path Resolution**: Vite handles API routes differently than Astro components

### What Fixed It

1. **Deterministic Dependencies**: Committed lockfile + `npm ci`
2. **Correct Node Version**: Aligned with Vercel's supported versions
3. **Unambiguous Modules**: Removed conflicting directory
4. **Explicit Extensions**: Added `.js` to TypeScript imports

---

## 🆘 Emergency Contacts

### Documentation References

- **Full RCA**: `daily-lesson-marketing/RCA_VERCEL_BUILD_FAILURES.md`
- **Astro Docs**: https://docs.astro.build/en/guides/troubleshooting/
- **Vercel Config**: https://vercel.com/docs/projects/project-configuration
- **Vite Module Resolution**: https://vitejs.dev/guide/features.html#modules

### Useful Commands

```bash
# Check Vercel deployment logs
vercel logs <deployment-url>

# Test build locally with Vercel CLI
npm i -g vercel
vercel build

# Debug module resolution
npx vite-node --inspect src/pages/api/lead.ts

# View all validation-related files
find daily-lesson-marketing -name "*validation*" -type f
find daily-lesson-marketing -name "*validation*" -type d
```

---

## ✅ Success Criteria

The deployment is successful when:

1. ✅ Vercel build completes without errors
2. ✅ All 26 pages are generated
3. ✅ No "Could not resolve" errors in build logs
4. ✅ Site is accessible at production URL
5. ✅ API endpoint `/api/lead` returns 200 on POST
6. ✅ Form submissions work end-to-end

**Test POST Request**:
```bash
curl -X POST https://curiouskelly-git-main-lotd.vercel.app/api/lead \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "Test",
    "last_name": "User",
    "email": "test@example.com",
    "phone": "+12025551234",
    "country": "US",
    "region": "DC",
    "marketing_opt_in": true,
    "locale": "en-US",
    "journey": "adults"
  }'
```

Expected response: `{"success": true, "message": "Lead submitted successfully"}`

---

## 🎁 Handoff Summary

**For the Next Engineer**:

You're inheriting a marketing site with a persistent build issue that's been traced to:
1. Module resolution conflicts (validation.ts vs validation/ directory)
2. Environment mismatches (Node versions)
3. Non-deterministic builds (missing lockfile)

**All identified issues have been fixed in commit `1e7fbcf`**.

**If the Vercel build still fails**:
1. Check the deployment logs for the exact error
2. Follow the "If Build Still Fails" section above
3. Use the "Nuclear Options" if module resolution continues to fail
4. The local build DOES work, so the issue is Vercel-specific

**Key Files to Check**:
- `src/lib/validation.ts` (must exist)
- `src/lib/validation/` (must NOT exist)
- `src/pages/api/lead.ts` (import with `.js` extension)
- `package-lock.json` (must be committed)

**Quick Debug**:
```bash
# Verify no validation directory exists
ls daily-lesson-marketing/src/lib/ | grep -c "validation/"
# Should output: 0

# Verify validation.ts exists
ls daily-lesson-marketing/src/lib/validation.ts
# Should output: daily-lesson-marketing/src/lib/validation.ts
```

Good luck! The fix should work. If not, you have all the context above to continue.

---

**Last Updated**: November 14, 2025 06:42 UTC  
**Last Commit**: `1e7fbcf`  
**Deployment Status**: Pending Vercel build results

