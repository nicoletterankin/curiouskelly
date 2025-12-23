# Final State - What Kelly Should Look Like

**Date:** December 23, 2025

---

## 🎯 THE FINAL STATE

### What You Should See:

1. **HeyGen Video Playing (MUTED)**
   - Kelly's photorealistic face and body moving naturally
   - White background
   - Body motion, head movements, natural expressions
   - **Video is MUTED** (no audio from video)

2. **TTS Audio Playing (LIVE)**
   - ElevenLabs Kelly voice speaking the lesson script
   - Audio comes from Cloudflare Worker (live generation)
   - Audio drives lip-sync analysis

3. **Kelly's Mouth MOVING (REAL-TIME)**
   - **Mouth opens and closes** in sync with speech
   - **Mouth shape changes** based on audio frequencies
   - **Lips move** naturally as she speaks
   - **Teeth visible** when mouth is open
   - **Smooth transitions** between mouth shapes

4. **Eyes Blinking**
   - Natural blinks every 4-6 seconds
   - Eyes close briefly, then reopen

5. **Expressions**
   - Eyebrows raise/lower based on lesson phase
   - Smile varies with content
   - Natural facial expressions

---

## 🔍 WHY MOUTH ISN'T MOVING

### Current Issues:

1. **Wrong Base Image** ❌
   - Using `kelly_explorer_head.png` (doesn't match HeyGen avatar)
   - Should use the HeyGen video or matching image

2. **Lip-Sync Not Analyzing** ❌
   - Audio is playing but lip-sync isn't detecting it
   - Blendshapes aren't varying (jawOpen stays at 0)

3. **Compositor Not Rendering** ❌
   - Mouth overlay exists but isn't visible
   - Or blendshapes aren't reaching compositor

---

## ✅ WHAT NEEDS TO FIX

1. **Use Correct Base Video**
   - `/kelly/videos/001/welcome.mp4` should be the HeyGen video
   - This video should match one of the 3 curated avatar IDs

2. **Fix Lip-Sync Analysis**
   - Ensure audio element is connected to Web Audio API
   - Verify `processFrame()` is running and analyzing audio
   - Check that RMS amplitude > 0 when audio plays

3. **Fix Mouth Rendering**
   - Ensure `setBlendshapes()` is being called with varying values
   - Verify `_renderOverlaysFromBlendshapes()` is rendering mouth
   - Check that mouth overlay is visible (not hidden behind video)

4. **Use Correct Avatar**
   - Don't default to 'explorer'
   - Use the base/neutral HeyGen avatar ID

---

## 🎬 THE CORRECT FLOW

```
1. Page loads → ?hybrid=1
2. Video loads → /kelly/videos/001/welcome.mp4 (HeyGen video)
3. Video plays → MUTED (no audio)
4. TTS starts → ElevenLabs audio via Cloudflare Worker
5. Audio plays → KellyLipSync analyzes audio frequencies
6. Blendshapes generated → jawOpen varies (0-85) based on audio
7. Compositor receives → setBlendshapes() called with varying values
8. Mouth renders → PixiJS draws mouth overlay on video
9. YOU SEE → Kelly's mouth opening/closing in sync with speech
```

---

**Status:** Need to fix lip-sync analysis and ensure mouth renders correctly.

