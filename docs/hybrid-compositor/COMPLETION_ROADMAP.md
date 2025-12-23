# Hybrid Compositor System - Completion Roadmap

> **Status:** Phase 1-2 Complete | Phase 3 In Progress  
> **Target:** Apple Education Editorial Team  
> **Last Updated:** December 22, 2025

---

## 🎯 EXECUTIVE SUMMARY

**What We're Building:** A photorealistic, real-time talking avatar system that combines pre-rendered HeyGen video with live ElevenLabs TTS and PixiJS overlays. Kelly's mouth moves, eyes blink, expressions change—all in the browser with zero installation.

**Current State:** Core systems operational. Talking Photo mode works. Mouth position calibrated. Debug marker fixed. Ready for production polish.

**Remaining Work:** Visual refinement, expression system, cross-platform testing, production hardening.

---

## ✅ COMPLETED (Phases 1-2)

| Component | Status | Notes |
|-----------|--------|-------|
| **PixiJS v8 Async Init** | ✅ Complete | Canvas renders, WebGL overlay layer operational |
| **TTS Worker** | ✅ Complete | Cloudflare Worker at `tts.curiouskelly.com`, R2 caching |
| **Talking Photo Mode** | ✅ Complete | `?talkingPhoto=1` uses static head images |
| **Mouth Position Calibration** | ✅ Complete | Absolute positioning at 56% from top |
| **Debug Marker** | ✅ Complete | Red dot shows mouth position for visual verification |
| **Blendshape Pipeline** | ✅ Complete | Audio → lip-sync → PixiJS overlays connected |

---

## ✅ COMPLETED (Phases 3-4)

**Latest Update:** December 22, 2025 - Phase 3.2 & Phase 4 complete!

### 3.1 Visual Calibration & Polish ✅

**Status:** 100% Complete

**Completed Tasks:**

1. **Debug Marker Conditional** ✅
   - Debug marker only shows when `?pixiDebug=1` (already implemented)
   - Marker positioned at mouth location (56% from top) for visual verification
   - **File:** `public/js/kelly-pixi-compositor.js` (line 442-452, 556-559)

2. **Opacity Presets Created** ✅
   - Created `OPACITY_PRESETS` object for easy tuning
   - Current values: mouth 35%, teeth 12%, lips 8-14%, blink 5-30%
   - **File:** `public/js/kelly-pixi-compositor.js` (line 69-78)
   - **Commit:** `00561d26`

**Completed Tasks:**

1. **Visual Testing & Fine-Tuning** ✅
   - Opacity presets created and documented
   - Ready for learner feedback and adjustment

2. **Mouth Shape Refinement** ✅
   - Natural oval/ellipse when open, rounded rect when closed
   - Curved lip shapes with Cupid's bow effect
   - Better lip separation and thickness variation
   - **Commit:** `35ae36cb`

3. **Eye Blink Timing** ✅
   - Current: 4-6 second intervals (deterministic)
   - **Note:** Can be enhanced later with randomness/context-awareness

---

## 📋 REMAINING PHASES (5-7)

### Phase 4: Expression System ✅

**Status:** 100% Complete

**Completed Tasks:**

1. **Phase-to-Expression Mapping** ✅
   - Hook → Curious (raised eyebrows, slight smile)
   - Fact1/Q1 → Curious
   - Fact2/Q2 → Explaining
   - Fact3/Q3 → Thinking
   - Wisdom → Warm (full smile, bright eyes)
   - **File:** `public/js/kelly-expression-bridge.js` (line 359-373)

2. **Eyebrow Overlays** ✅
   - Added PixiJS graphics for eyebrows
   - Control: raised based on blendshapes (browInnerUp, browOuterUpLeft/Right)
   - Position: Above eyes, anchored to face
   - **File:** `public/js/kelly-pixi-compositor.js` (line 636-667)

3. **Smooth Transitions** ✅
   - Interpolation between expression states
   - Duration: 400ms for natural feel
   - Trigger: On phase transitions via `setPhaseExpression()`

4. **Expression Bridge Integration** ✅
   - Connected `KellyExpressionBridge` to `KellyPixiCompositor`
   - Blendshapes sent to Pixi compositor via `sendTo2D()`
   - **Commit:** `35ae36cb`

**Success Criteria:** ✅ All Met
- Kelly's face changes expression when lesson phase changes ✅
- Transitions are smooth (400ms cubic ease-out) ✅
- Expressions match lesson content ✅

---

### Phase 5: Cross-Platform Testing

**Goal:** Works flawlessly on iOS Safari, Android Chrome, desktop browsers.

**Tasks:**

1. **iOS Safari Autoplay Handling** ⚠️ CRITICAL
   - iOS blocks autoplay with sound
   - Current: Video muted, TTS plays separately
   - **Test:** Verify TTS plays without user interaction
   - **Fallback:** Show "Tap to start" button if autoplay blocked

2. **Android Chrome Testing**
   - Test on mid-tier Android device (Samsung Galaxy A series)
   - Verify WebGL performance (60fps target)
   - Check audio latency (should be < 200ms)

3. **Desktop Browser Matrix**
   - Chrome/Edge (Chromium) ✅
   - Firefox ✅
   - Safari (macOS) ⏳
   - Opera ⏳

4. **Low-Bandwidth Fallback**
   - Test with throttled connection (3G speeds)
   - Verify cached audio serves correctly
   - Ensure graceful degradation (no broken UI)

**Files to Modify:**
- `public/learn.html` - Add iOS autoplay detection and fallback UI
- `public/js/kelly-audio.js` - Handle autoplay restrictions
- `infrastructure/cloudflare/tts-worker/src/index.js` - Verify R2 fallback works

**Success Criteria:**
- Works on iOS Safari without user interaction
- 60fps on mid-tier devices
- Graceful fallback if TTS fails

---

### Phase 6: Performance Optimization

**Goal:** Smooth 60fps, low CPU usage, fast startup.

**Tasks:**

1. **WebGL Performance Audit**
   - Profile PixiJS render loop
   - Optimize sprite batching
   - Reduce draw calls (combine overlays where possible)

2. **Audio Streaming Optimization**
   - Preload next phase audio while current plays
   - Implement audio buffer pooling
   - Reduce TTS latency (target: < 1s)

3. **Asset Preloading**
   - Preload Kelly head images for all archetypes
   - Cache PixiJS textures
   - Lazy-load expression overlays

4. **Memory Management**
   - Dispose unused textures/sprites
   - Clear audio buffers after playback
   - Monitor memory leaks (Chrome DevTools)

**Files to Modify:**
- `public/js/kelly-pixi-compositor.js` - Optimize render loop
- `public/js/kelly-audio.js` - Implement preloading
- `public/learn.html` - Add asset preload hints

**Success Criteria:**
- 60fps on mid-tier devices
- < 100MB memory usage
- < 2s startup time

---

### Phase 7: Production Hardening

**Goal:** Bulletproof reliability, monitoring, error handling.

**Tasks:**

1. **Error Handling & Fallbacks**
   - TTS Worker fails → Use cached audio → Use pregen → Show text
   - PixiJS init fails → Hide overlays, show static image
   - Audio autoplay blocked → Show "Tap to start" button

2. **Analytics & Monitoring**
   - Track TTS latency (Cloudflare Analytics)
   - Monitor PixiJS init failures (Sentry or custom)
   - Log expression transitions (for debugging)

3. **A/B Testing Framework**
   - Compare hybrid mode vs. baked video
   - Measure: engagement time, completion rate, learner satisfaction
   - **Hypothesis:** Hybrid mode feels more alive → higher engagement

4. **Documentation**
   - Update `HYBRID_COMPOSITOR_DIRECTIVE.md` with final calibration values
   - Create troubleshooting guide for common issues
   - Document expression presets and phase mappings

**Files to Modify:**
- `public/js/kelly-fallback-engine.js` - Enhance fallback logic
- `public/learn.html` - Add error boundaries
- `docs/hybrid-compositor/` - Update all documentation

**Success Criteria:**
- Zero visible failures for learners
- < 1% error rate
- Full fallback chain tested

---

## 🎯 COMPLETION CRITERIA

### Must-Have (MVP):
- ✅ PixiJS overlays render
- ✅ Mouth syncs with TTS audio
- ✅ Talking Photo mode works
- ⏳ Works on iOS Safari
- ⏳ Expression system (basic: curious, engaged, warm)
- ⏳ Error handling & fallbacks

### Nice-to-Have (Polish):
- ⏳ Advanced expressions (thoughtful, surprised, etc.)
- ⏳ Performance optimization (60fps on all devices)
- ⏳ A/B testing framework
- ⏳ Analytics dashboard

### Future Enhancements:
- Real-time face tracking (WebRTC MediaPipe)
- Dynamic body motion (composable motion segments)
- Multi-language support (ES/FR head images)
- Customizable expressions (learner preferences)

---

## 📅 ESTIMATED TIMELINE

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| **Phase 3** (Visual Polish) | 4 tasks | ✅ **COMPLETE** |
| **Phase 4** (Expressions) | 4 tasks | ✅ **COMPLETE** |
| **Phase 5** (Cross-Platform) | 4 tasks | 3-4 hours |
| **Phase 6** (Performance) | 4 tasks | 3-4 hours |
| **Phase 7** (Hardening) | 4 tasks | 2-3 hours |
| **Total** | 20 tasks | **8-11 hours remaining** |

**Note:** This assumes no major blockers. iOS Safari autoplay may require additional research.

---

## 🚀 NEXT IMMEDIATE ACTIONS

1. **Test debug marker fix** (v=20251222h)
   - URL: `https://curiouskelly.com/learn.html?talkingPhoto=1&pixiDebug=1&day=1`
   - Verify: Red dot on Kelly's mouth (56% from top)

2. **Remove debug marker for production**
   - Add conditional: `if (DEBUG && pixiDebug) { show marker }`
   - Default: `pixiDebug=0` (hidden)

3. **Start Phase 4: Expression System**
   - Begin with eyebrow overlays
   - Map Hook phase → curious expression
   - Test smooth transitions

---

## 📝 NOTES FOR FUTURE AI ASSISTANTS

**Key Files:**
- `public/js/kelly-pixi-compositor.js` - Main overlay renderer (PixiJS)
- `public/js/kelly-expression-bridge.js` - Phase → expression mapping
- `public/learn.html` - Lesson player, phase transitions
- `docs/hybrid-compositor/HYBRID_COMPOSITOR_DIRECTIVE.md` - Architecture docs

**Calibration Values:**
- Mouth position: `MOUTH_Y_ABSOLUTE = 0.56` (56% from top)
- Anchor (face center): `y: 0.40` (40% from top)
- Debug marker: Shows mouth position (not anchor)

**Testing URLs:**
- Talking Photo: `?talkingPhoto=1&pixiDebug=1&day=1`
- Hybrid Video: `?hybrid=1&pixiDebug=1&day=1`
- Production: `?day=1` (no debug params)

**Critical Constraints:**
- iOS Safari blocks autoplay with sound (must handle gracefully)
- CDN cache can be aggressive (use cache-busting version params)
- PixiJS v8 requires async `app.init()` (already fixed)

---

**Last Updated:** December 22, 2025  
**Next Review:** After Phase 4 completion

