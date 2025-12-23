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

## 🚧 IN PROGRESS (Phase 3)

### 3.1 Visual Calibration & Polish

**Status:** 80% Complete

**Remaining Tasks:**

1. **Remove Debug Marker for Production** ⏳
   - Add `?pixiDebug=0` or remove debug mode entirely
   - Keep debug mode available for future calibration
   - **File:** `public/js/kelly-pixi-compositor.js` (line 556-559)

2. **Fine-Tune Overlay Opacity** ⏳
   - Current: 5-35% opacity (may be too subtle or too visible)
   - Test with real learners, adjust based on feedback
   - **Target:** Overlays should be noticeable but not distracting

3. **Mouth Shape Refinement** ⏳
   - Current: Simple rounded rectangle
   - **Enhancement:** Add more natural mouth shapes based on visemes
   - Consider: Upper/lower lip separation, corner curves, teeth visibility

4. **Eye Blink Timing** ⏳
   - Current: 4-6 second intervals (deterministic)
   - **Enhancement:** Add randomness, context-aware blinks (more frequent when thinking)

---

## 📋 REMAINING PHASES (4-7)

### Phase 4: Expression System

**Goal:** Kelly's face reflects lesson content and emotional arc.

**Tasks:**

1. **Phase-to-Expression Mapping**
   - Hook → Curious (raised eyebrows, slight smile)
   - Teach → Engaged (focused eyes, neutral mouth)
   - Review → Thoughtful (slight frown, furrowed brow)
   - Wisdom → Warm (full smile, bright eyes)
   - **File:** `public/js/kelly-expression-bridge.js`

2. **Eyebrow Overlays**
   - Add PixiJS sprite for eyebrows
   - Control: raised, neutral, furrowed
   - Position: Above eyes, anchored to face

3. **Smile/Frown Transitions**
   - Smooth interpolation between expression states
   - Duration: 0.3-0.5 seconds for natural feel
   - Trigger: On phase transitions, not mid-speech

4. **Expression Presets**
   ```javascript
   EXPRESSION_PRESETS = {
     curious: { eyebrows: 'raised', smile: 0.3, eyes: 'wide' },
     engaged: { eyebrows: 'neutral', smile: 0.0, eyes: 'normal' },
     thoughtful: { eyebrows: 'furrowed', smile: -0.2, eyes: 'narrow' },
     warm: { eyebrows: 'neutral', smile: 0.6, eyes: 'bright' }
   };
   ```

**Files to Modify:**
- `public/js/kelly-pixi-compositor.js` - Add eyebrow sprites, expression interpolation
- `public/js/kelly-expression-bridge.js` - Map phases to expressions
- `public/learn.html` - Hook expression changes to phase transitions

**Success Criteria:**
- Kelly's face changes expression when lesson phase changes
- Transitions are smooth (no jarring jumps)
- Expressions match lesson content (curious for questions, warm for wisdom)

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
| **Phase 3** (Visual Polish) | 4 tasks | 2-3 hours |
| **Phase 4** (Expressions) | 4 tasks | 4-6 hours |
| **Phase 5** (Cross-Platform) | 4 tasks | 3-4 hours |
| **Phase 6** (Performance) | 4 tasks | 3-4 hours |
| **Phase 7** (Hardening) | 4 tasks | 2-3 hours |
| **Total** | 20 tasks | **14-20 hours** |

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

