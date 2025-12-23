# Hybrid Compositor Testing Audit

> **Date:** December 22, 2025  
> **Status:** 🔴 FAILING - Compositor not initializing

---

## 🚨 CRITICAL FINDINGS

### Test Results Summary

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Compositor Script Loaded | ✅ | ✅ v=20251222m | ✅ |
| PixiJS Available | ✅ | ✅ | ✅ |
| Compositor Initialized | ✅ | ❌ FALSE | ❌ |
| Canvas Found | ✅ | ❌ FALSE | ❌ |
| Debug Marker | ✅ | ❌ FALSE | ❌ |
| Mouth Overlay | ✅ | ❌ FALSE | ❌ |
| Eyebrows | ✅ | ❌ FALSE | ❌ |

### Root Cause Analysis

**Problem:** `KellyPixiCompositor.init()` is **never being called**.

**Evidence:**
- Script loads successfully (`v=20251222m`)
- PIXI library available
- `window.KellyPixiCompositor` exists
- But `isInitialized = false`
- `hasApp = false`
- `hasCanvas = false`
- `containerEl = false`

**Why init() isn't called:**
- `init()` is only called inside `playPhaseMedia()` function
- `playPhaseMedia()` is only called when a lesson phase plays
- The TALKING_PHOTO branch requires:
  1. `TALKING_PHOTO = true` (from URL param `?talkingPhoto=1`)
  2. `resolvedSource === 'talking_photo'`
  3. Code path to execute

**Hypothesis:** `playPhaseMedia()` may not be executing, or the TALKING_PHOTO branch isn't being hit.

---

## 🔍 DEBUGGING ADDED

### Enhanced Logging

1. **Compositor Script:**
   - Logs script location and timestamp on load
   - Logs PIXI availability
   - Logs state on initialization

2. **TALKING_PHOTO Branch:**
   - Logs when branch executes
   - Logs compositor availability checks
   - Logs init() call and completion
   - Logs errors with stack traces

3. **Puppeteer Test Suite:**
   - Comprehensive test coverage
   - Captures all console logs
   - Screenshots at key points
   - Checks all component states

---

## 🐛 KNOWN ISSUES

### 1. Vercel Caching (CRITICAL)

**Problem:** Vercel is serving old cached versions.

**Evidence:**
- Git has: `v=20251222n`
- Vercel serves: `v=20251222m`
- learn.html also cached at old version

**Solution:**
- Wait for cache propagation (5-10 minutes)
- Or force cache purge via Vercel dashboard
- Or add aggressive cache-busting

### 2. Init() Not Called

**Problem:** Compositor init() never executes.

**Possible Causes:**
- `playPhaseMedia()` not being called
- TALKING_PHOTO branch condition not met
- Container element (`#kelly-stage`) not found
- Async timing issue

**Next Steps:**
- Verify `playPhaseMedia()` is called
- Check if `#kelly-stage` exists
- Add init() call earlier in page load
- Consider auto-init on DOM ready

---

## 📋 TEST COMMANDS

```bash
# Run full test suite
node tests/hybrid-compositor-test.js

# Run debug script
node tests/debug-compositor.js

# Check deployed versions
curl "https://www.curiouskelly.com/js/kelly-pixi-compositor.js?v=20251222n" | grep "v=20251222"
curl "https://www.curiouskelly.com/learn.html" | grep "kelly-pixi-compositor"
```

---

## 🔧 FIXES NEEDED

1. **Immediate:**
   - Wait for Vercel cache to update
   - Verify new code is deployed
   - Re-run tests

2. **Short-term:**
   - Add fallback init() call on DOM ready
   - Make init() more resilient (retry if container missing)
   - Add health check endpoint

3. **Long-term:**
   - Set up CI/CD test pipeline
   - Add automated deployment verification
   - Monitor compositor initialization rate

---

**Last Updated:** December 22, 2025  
**Next Action:** Wait for deployment, then re-test

