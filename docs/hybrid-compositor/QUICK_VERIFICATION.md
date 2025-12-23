# Quick Verification Guide

**For:** `?talkingPhoto=1&pixiDebug=1` mode

## ✅ What You Should See

### Visual Indicators
1. **Red debug marker** on Kelly's mouth ✅ (visible in screenshot)
2. **Compositor initialized** ✅ (console shows "Compositor READY")
3. **PixiJS v8 working** ✅ (console confirms)

### Console Logs to Check

#### 1. Compositor Initialization ✅
```
[Pixi] Compositor READY - Kelly's mouth can now move!
[KellyLipSync] Already initialized
```

#### 2. Audio Playback (MISSING - need to verify)
```
🔊 [Audio] onSpeakStart() called
🔊 [Audio] Audio element: { exists: true, src: "...", paused: false }
```

#### 3. Lip-Sync Connection (MISSING - need to verify)
```
[KellyLipSync] Connected to audio element
[KellyLipSync] Audio already playing - started processFrame loop immediately
```

#### 4. Audio Analysis (MISSING - need to verify)
```
[KellyLipSync] Audio analysis: { rms: 0.0234, maxSample: 0.0456, isActive: true }
👄 Lip-sync ACTIVE: jawOpen=15.23, mouthOpen=12.18
```

## 🔍 Manual Verification Steps

### Step 1: Check Audio Element
Open browser console and run:
```javascript
window.kellyAudio.audio
// Should show: <audio> element with src, not paused, currentTime > 0
```

### Step 2: Check Lip-Sync State
```javascript
window.KellyLipSync.isActive  // Should be true
window.KellyLipSync.analyser  // Should exist
window.kellyAudio.audio._lipSyncConnected  // Should be true
```

### Step 3: Monitor Blendshapes
```javascript
let lastJawOpen = 0;
window.KellyLipSync.onBlendshapesUpdate = (bs) => {
  if (Math.abs(bs.jawOpen - lastJawOpen) > 1) {
    console.log('✅ JawOpen changed:', lastJawOpen.toFixed(2), '->', bs.jawOpen.toFixed(2));
    lastJawOpen = bs.jawOpen;
  }
};
```

### Step 4: Force Audio Playback
If audio isn't playing automatically:
```javascript
// Trigger audio manually
window.playPhaseMedia({
  dbPhase: 'hook',
  script: 'Hello, I am Kelly. This is a test of the hybrid compositor system.'
});
```

## 🎯 Expected Behavior

1. **Page loads** → Compositor initializes ✅
2. **Audio starts** → `onSpeakStart()` fires (check console)
3. **Lip-sync connects** → `startFromAudioElement()` creates analyser connection
4. **Audio analyzed** → RMS amplitude calculated, blendshapes vary
5. **Mouth animates** → Red debug marker moves, mouth overlay changes size

## ⚠️ Common Issues

### Issue: Audio not playing
**Symptoms:** No `🔊 [Audio]` logs in console
**Fix:** 
- Check if autoplay is blocked (click page first)
- Verify TTS endpoint is working
- Check browser console for errors

### Issue: Lip-sync not analyzing
**Symptoms:** No `[KellyLipSync] Audio analysis:` logs
**Fix:**
- Verify `window.KellyLipSync.isActive === true`
- Check `window.KellyLipSync.analyser` exists
- Ensure audio element has `crossOrigin = 'anonymous'`

### Issue: Blendshapes not varying
**Symptoms:** `jawOpen` stays at 0
**Fix:**
- Verify audio is actually playing (check `audio.currentTime`)
- Check CORS headers on audio source
- Verify `processFrame()` loop is running

## 📊 Success Criteria

- ✅ Compositor initialized (visible in screenshot)
- ⏳ Audio playing (need to verify)
- ⏳ Lip-sync analyzing (need to verify)
- ⏳ Blendshapes varying (need to verify)
- ⏳ Mouth animating (need to verify)

**Next:** Run the manual verification steps above to confirm audio analysis is working.

