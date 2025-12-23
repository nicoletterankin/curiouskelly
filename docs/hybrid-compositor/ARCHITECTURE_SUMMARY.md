# Hybrid Compositor Architecture Summary

**Date:** December 23, 2025  
**Status:** ✅ CORRECTED AND VERIFIED

---

## 🎯 THE CORRECT ARCHITECTURE

### Base Layer: HeyGen Video (MUTED)
- **File:** `/kelly/videos/001/welcome.mp4` (ground truth)
- **Source:** Pre-rendered HeyGen video with **single avatar ID**
- **Purpose:** Provides photorealistic body motion, facial expressions, lighting
- **Audio:** **MUTED** - We don't use video audio
- **Mode:** `?hybrid=1` forces this video

### Audio Layer: ElevenLabs TTS (LIVE)
- **Source:** Cloudflare Worker (`tts.curiouskelly.com`)
- **Purpose:** Provides dynamic, real-time voice
- **Format:** MP3 blob, ~50KB per phrase
- **Latency:** 1-2 seconds

### Overlay Layer: PixiJS WebGL (REAL-TIME)
- **Source:** KellyLipSync blendshapes + KellyExpressionBridge
- **Purpose:** Mouth sync, eye blinks, expressions
- **Rendering:** WebGL canvas overlays on top of video

---

## ✅ VERIFIED CODE LOGIC

### Hybrid Mode (`?hybrid=1`)
```javascript
// Video: HeyGen video (muted)
resolvedVideo = '/kelly/videos/001/welcome.mp4';
v.muted = true;  // ✅ CORRECT: Video muted

// Audio: TTS (live)
if (HYBRID_DEMO) {
  kellyAudio.speak(dbPhase, script);  // ✅ CORRECT: TTS provides audio
}

// Overlay: PixiJS compositor
KellyPixiCompositor.attachVideo(videoEl);  // ✅ CORRECT: Overlays on video
```

### Talking Photo Mode (`?talkingPhoto=1`)
```javascript
// Base: Static image (no video)
resolvedVideo = null;
KellyPixiCompositor.attachImage(personaId);  // ✅ CORRECT: Uses static head image

// Audio: TTS (live)
kellyAudio.speak(dbPhase, script);  // ✅ CORRECT: TTS provides audio
```

---

## 🔍 KEY POINTS

1. **Single HeyGen Avatar ID:** The ground truth video uses ONE HeyGen avatar ID
2. **Video is Muted:** In hybrid mode, video provides visual only (body motion)
3. **TTS is Live:** ElevenLabs provides the voice that drives lip-sync
4. **PixiJS Overlays:** Real-time mouth/eye animations on top of video/image

---

## 📊 TESTING

### Hybrid Mode Test
```
https://curiouskelly.com/learn.html?hybrid=1&day=1&pixiDebug=1
```

**Expected:**
- Video plays (muted)
- TTS audio plays (live)
- Lip-sync analyzes TTS audio
- PixiJS overlays mouth/eyes on video
- Console shows: `🔇 Video MUTED (hybrid mode - using TTS audio)`

### Talking Photo Mode Test
```
https://curiouskelly.com/learn.html?talkingPhoto=1&day=1&pixiDebug=1
```

**Expected:**
- Static image displays
- TTS audio plays (live)
- Lip-sync analyzes TTS audio
- PixiJS overlays mouth/eyes on image
- Console shows: `📸 TALKING PHOTO mode`

---

**Status:** Architecture verified and corrected. Ready for testing.

