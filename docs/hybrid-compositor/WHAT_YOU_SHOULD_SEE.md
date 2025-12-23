# What You Should See - Final State

**Date:** December 23, 2025

---

## 🎯 THE FINAL STATE

### When You Load: `?hybrid=1&day=1&pixiDebug=1`

**What You Should See:**

1. **HeyGen Video Playing (MUTED)** ✅
   - Kelly's photorealistic face and body
   - Natural head movements, expressions
   - White background
   - **Video has NO audio** (muted)

2. **TTS Audio Playing (LIVE)** ✅
   - ElevenLabs Kelly voice speaking
   - Audio comes from Cloudflare Worker
   - You can HEAR Kelly speaking

3. **Kelly's Mouth MOVING** ✅ **THIS IS WHAT YOU'RE MISSING**
   - **Mouth opens** when she speaks (jawOpen increases)
   - **Mouth closes** during pauses (jawOpen decreases)
   - **Mouth shape changes** based on speech sounds
   - **Lips visible** - brown/tan outline around mouth
   - **Teeth visible** when mouth is wide open
   - **Smooth animation** - no jerky movements

4. **Eyes Blinking** ✅
   - Natural blinks every 4-6 seconds
   - Eyes close briefly, reopen

5. **Red Debug Dot** ✅ (if `pixiDebug=1`)
   - Red dot shows mouth position
   - Should be on Kelly's actual mouth (56% from top)

---

## 🔍 WHY MOUTH ISN'T MOVING

### Root Cause Analysis:

1. **Lip-Sync Not Analyzing Audio** ❌
   - Audio plays but `jawOpen` stays at 0
   - `processFrame()` might not be running
   - Audio element might not be connected to analyser

2. **Mouth Overlay Not Visible** ❌
   - Opacity too low (was 0.35, now increased to 0.6 when open)
   - Overlay might be behind video
   - Blendshapes not reaching compositor

3. **Wrong Base Asset** ❌
   - Using 'explorer' image (doesn't match HeyGen avatar)
   - Should use 'storyteller' or base HeyGen video

---

## ✅ FIXES APPLIED

1. **Correct Avatar** ✅
   - Changed default from 'explorer' to 'storyteller'
   - Matches HeyGen avatar ID

2. **Increased Mouth Visibility** ✅
   - Mouth opacity: 0.35 → 0.6 when jawOpen > 0.1
   - Lip opacity: increased to 0.7 when mouth open
   - Mouth should now be VISIBLE

3. **Always Render** ✅
   - Compositor now renders every frame when enabled
   - Don't skip frames waiting for blendshapes

4. **Debug Logging** ✅
   - Logs jawOpen values when > 5%
   - Can verify lip-sync is working

---

## 🧪 TESTING CHECKLIST

### Visual Test:
- [ ] Load `?hybrid=1&day=1&pixiDebug=1`
- [ ] Video plays (muted)
- [ ] Audio plays (you can hear Kelly)
- [ ] **Mouth opens/closes** as she speaks ← THIS IS THE KEY TEST
- [ ] Red dot is on Kelly's mouth
- [ ] Eyes blink naturally

### Console Test:
- [ ] `🔇 Video MUTED (hybrid mode - using TTS audio)`
- [ ] `🎤 Starting TTS for hybrid mode...`
- [ ] `🔊 ✅ Playing TTS audio`
- [ ] `[KellyLipSync] Audio analysis: { rms: X.XXXX }`
- [ ] `[Pixi] 🎭 Rendering mouth: jawOpen=X.X%` ← Should see this!
- [ ] `👄 Lip-sync ACTIVE: jawOpen=X.XX`

---

## 🎬 THE CORRECT FLOW

```
1. Page loads → ?hybrid=1
2. Video loads → /kelly/videos/001/welcome.mp4 (HeyGen, muted)
3. TTS starts → ElevenLabs audio via Cloudflare Worker
4. Audio plays → KellyLipSync.startFromAudioElement(audio)
5. processFrame() runs → Analyzes audio every frame
6. RMS amplitude > 0 → jawOpen calculated (0-85)
7. setBlendshapes() called → jawOpen varies with speech
8. Compositor renders → PixiJS draws mouth overlay
9. YOU SEE → Kelly's mouth opening/closing! 🎉
```

---

**Status:** Architecture fixed. Mouth visibility increased. Need to verify lip-sync is analyzing audio.

