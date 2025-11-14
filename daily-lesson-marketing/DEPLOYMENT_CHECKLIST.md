# Deployment Checklist - Vercel Build Fix

## ✅ Pre-Deployment Verification

- [x] Validation functions inlined in `src/pages/api/lead.ts`
- [x] Tests updated and passing (7/7 tests)
- [x] Local build successful (1.26s, 26 pages)
- [x] No linter errors
- [x] No other files importing from `../../lib/validation.js` in API routes

## 🚀 Deployment Steps

### Step 1: Commit Changes
```bash
cd daily-lesson-marketing
git add src/pages/api/lead.ts tests/unit/validation.test.ts VERCEL_BUILD_FIX_SUMMARY.md
git commit -m "fix: inline validation functions in API route to resolve Vercel build errors

- Inlined validateLeadForm and sanitizeFormData directly into lead.ts
- Removed problematic import from '../../lib/validation.js'
- Fixed validation tests to use correct function signatures
- All tests passing (7/7)
- Local build successful

Resolves: Vercel build failure 'Could not resolve ../../lib/validation.js'"
```

### Step 2: Push to Trigger Deployment
```bash
git push origin main
```

### Step 3: Clear Vercel Cache (Recommended)
1. Go to Vercel Dashboard → Project Settings → Caches
2. Click **"Purge CDN Cache"**
3. Click **"Purge Data Cache"**

This ensures Vercel doesn't use stale build artifacts.

### Step 4: Monitor Deployment
1. Go to Vercel Dashboard → Deployments
2. Watch for the new deployment triggered by your push
3. Expected build time: ~35-40 seconds
4. Look for: ✅ "Build succeeded" (not ❌ "Build failed")

## 📊 Expected Results

### Success Indicators:
- ✅ Build completes in ~35-40 seconds
- ✅ 26 pages generated
- ✅ API routes available: `/api/lead` (POST), `/api/rum` (POST)
- ✅ Status: "Build succeeded"
- ✅ No "Could not resolve" errors in logs

### If Build Still Fails:
1. Check the build logs for the specific error
2. Verify the commit was pushed successfully
3. Try clearing cache again
4. Check `VERCEL_BUILD_HANDOFF.md` for additional troubleshooting

## 🔍 Post-Deployment Verification

After successful deployment:
1. Test the API endpoint: `POST /api/lead`
2. Verify form submission works on the site
3. Check that validation errors display correctly
4. Confirm no console errors in browser

## 📝 Files Changed

- `src/pages/api/lead.ts` - Inlined validation functions
- `tests/unit/validation.test.ts` - Fixed test imports
- `VERCEL_BUILD_FIX_SUMMARY.md` - Documentation

## 🎯 What This Fix Does

**Problem**: Vercel couldn't resolve `../../lib/validation.js` during build

**Solution**: Inlined all validation logic directly into the API route file, eliminating the need for module resolution.

**Impact**: 
- ✅ Builds will succeed on Vercel
- ✅ No breaking changes to functionality
- ✅ Same validation logic, just in a different location
- ✅ Component still uses `validation.ts` via `@lib/validation` alias (works fine)

