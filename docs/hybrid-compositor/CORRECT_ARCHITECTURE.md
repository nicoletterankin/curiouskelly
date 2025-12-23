# Correct Hybrid Compositor Architecture

**Date:** December 23, 2025  
**Status:** ✅ CORRECTED

---

## 🎯 THE CORRECT ARCHITECTURE

### Base Layer: HeyGen Video (MUTED)
- **Source:** Pre-rendered HeyGen video (photorealistic Kelly avatar)
- **Purpose:** Provides body motion, facial expressions, lighting, photorealistic rendering
- **Audio:** **MUTED** - We don't use the video's audio track
- **Example:** `/kelly/videos/001/welcome.mp4` (ground truth HeyGen video)

### Audio Layer: ElevenLabs TTS (LIVE)
- **Source:** Cloudflare Worker → ElevenLabs API
- **Purpose:** Provides dynamic, real-time voice that feels alive
- **Format:** MP3 blob, ~50KB per phrase
- **Latency:** 1-2 seconds

### Overlay Layer: PixiJS WebGL (REAL-TIME)
- **Source:** KellyLipSync blendshapes + KellyExpressionBridge
- **Purpose:** Mouth sync, eye blinks, expressions
- **Rendering:** WebGL canvas overlays on top of video

---

## ✅ CORRECTED CODE LOGIC

### Video Muting Logic
```javascript
// HYBRID mode: HeyGen video is ALWAYS muted (we use TTS)
if (HYBRID_DEMO || needsSeparateTTS) {
  v.muted = true;  // ✅ CORRECT: Video muted, TTS provides audio
}

// Non-hybrid mode: Use video audio if available
else if (videoHasAudio) {
  v.muted = false;  // Video has audio, use it
}
```

### TTS Trigger Logic
```javascript
// CRITICAL: In HYBRID mode, ALWAYS use TTS (video is muted)
if (needsSeparateTTS || HYBRID_DEMO) {
  kellyAudio.speak(dbPhase, script);  // ✅ CORRECT: TTS provides audio
}
```

---

## 🔍 VERIFICATION

### What Should Happen in `?hybrid=1` Mode:

1. **Video loads** → `/kelly/videos/001/welcome.mp4` (or other HeyGen video)
2. **Video plays** → **MUTED** (no audio from video)
3. **TTS starts** → Live ElevenLabs audio via Cloudflare Worker
4. **Lip-sync connects** → Analyzes TTS audio, generates blendshapes
5. **Compositor renders** → PixiJS overlays mouth/eyes on video
6. **Result** → Kelly appears alive, speaking dynamically

### Console Logs to Verify:
```
🔇 Video MUTED (hybrid mode - using TTS audio)
🎤 Starting TTS for hybrid mode (video muted, using live TTS)...
🔊 ✅ Playing TTS audio
👄 Lip-sync connected to audio element
[KellyLipSync] Audio analysis: { rms: 0.0234, ... }
👄 Lip-sync ACTIVE: jawOpen=15.23
```

---

## ⚠️ WHAT WAS WRONG

**Previous Logic (INCORRECT):**
- `videoHasAudio` was false for `HYBRID_DEMO`, so video was muted ✅
- But the comment said "Unmute HeyGen videos" which was confusing
- Logic was backwards: it unmuted videos that had audio, but hybrid mode needs muted video

**Corrected Logic:**
- `HYBRID_DEMO` → Video **MUTED**, TTS **ACTIVE** ✅
- Non-hybrid → Video audio if available, TTS as fallback

---

## 📊 ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────┐
│         HeyGen Video (MUTED)            │
│  - Body motion                          │
│  - Facial expressions                   │
│  - Photorealistic rendering             │
│  - White background                     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      PixiJS WebGL Canvas (OVERLAY)      │
│  - Mouth sync (from TTS audio)         │
│  - Eye blinks                           │
│  - Expressions                          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      ElevenLabs TTS (AUDIO LAYER)       │
│  - Live voice                           │
│  - Drives lip-sync                      │
│  - Cloudflare Worker                    │
└─────────────────────────────────────────┘
```

---

**Status:** Architecture corrected. Hybrid mode now correctly mutes HeyGen video and uses TTS audio.

