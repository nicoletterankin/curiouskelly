# Unity Deployment Status - Curious Kelly

**Last Updated:** 2025-01-11  
**Status:** Unity iframe integration complete, awaiting WebGL build deployment

---

## ✅ What's Complete

### 1. Unity Iframe Integration
- ✅ Unity iframe component added to `lesson-player/index.html`
- ✅ Unity messaging bridge integrated in `lesson-player/script.js`
- ✅ Status indicators and UI feedback implemented
- ✅ Lesson loading pipeline connected to Unity
- ✅ CSS styling for Unity container and status display

### 2. Integration Features
- ✅ PostMessage communication between lesson player and Unity
- ✅ Status updates: `loading`, `ready`, `playing`, `stopped`, `error`
- ✅ Automatic lesson loading when Unity becomes ready
- ✅ Pending lesson queue for lessons loaded before Unity is ready
- ✅ Fallback to static images if Unity is unavailable

### 3. File Structure
```
lesson-player/
├── index.html          ✅ Updated with Unity iframe
├── script.js           ✅ Unity integration methods added
└── styles.css          ✅ Unity container styling added

public/unity/kelly-v1/
├── index.html          ⚠️ Placeholder (needs Unity WebGL build)
└── kbridge.js          ✅ Messaging bridge ready
```

---

## ⚠️ What's Pending

### 1. Unity WebGL Build
**Status:** Not deployed  
**Location:** `public/unity/kelly-v1/`  
**Current State:** Placeholder HTML file only

**What's needed:**
1. Build Unity WebGL project using:
   ```powershell
   scripts\build_unity_webgl.ps1 -UnityPath "C:\Program Files\Unity\Hub\Editor\6000.2.1f1\Editor\Unity.exe" -Version kelly-v1
   ```
2. Copy build output to `public/unity/kelly-v1/`
3. Ensure `index.html` includes Unity loader and references `kbridge.js`

### 2. Lesson Assets
**Status:** Need to verify/upload  
**Required files per lesson:**
- Audio: `../lessons/audio/{lessonId}/{ageBucket}-{language}-{phase}.mp3`
- Viseme JSON: `../lessons/audio/{lessonId}/{ageBucket}-{language}-{phase}.a2f.json`
- Expressions (optional): `../lessons/audio/{lessonId}/{ageBucket}-{language}-{phase}.expressions.json`

---

## 🎯 How It Works

### Lesson Player → Unity Communication

1. **Initialization:**
   - Lesson player loads Unity iframe from `/unity/kelly-v1/index.html`
   - Unity sends `kelly-ready` when loaded
   - Lesson player updates status and hides loading indicator

2. **Lesson Loading:**
   - When a lesson is selected, player constructs URLs:
     - Audio: `../lessons/audio/{lessonId}/{ageBucket}-{language}-{phase}.mp3`
     - Viseme JSON: `../lessons/audio/{lessonId}/{ageBucket}-{language}-{phase}.a2f.json`
   - Player sends `kelly-load` message to Unity with URLs
   - Unity loads assets and responds with status updates

3. **Status Flow:**
   ```
   Loading Kelly... → Kelly is ready → Loading lesson... → Playing
   ```

### Message Protocol

**Parent → Unity:**
```javascript
{
  source: 'curiouskelly.com',
  destination: 'kelly-webgl',
  type: 'kelly-load',
  payload: {
    lessonId: 'water-cycle-18-35',
    jsonUrl: '../lessons/audio/.../18-35-en-welcome.a2f.json',
    audioUrl: '../lessons/audio/.../18-35-en-welcome.mp3',
    expressionsUrl: '../lessons/audio/.../18-35-en-welcome.expressions.json',
    offsetMs: 50
  }
}
```

**Unity → Parent:**
```javascript
{
  source: 'kelly-webgl',
  type: 'kelly-ready' | 'kelly-loading' | 'kelly-playing' | 'kelly-stopped' | 'kelly-error',
  status: 'ok',
  lessonId: 'water-cycle-18-35',
  message: 'Error message if type is kelly-error'
}
```

---

## 🚀 Next Steps

1. **Build Unity WebGL:**
   - Run build script from `digital-kelly/engines/kelly_unity_player`
   - Output should go to `Builds/WebGL/kelly-v1/`
   - Copy contents to `public/unity/kelly-v1/`

2. **Verify Assets:**
   - Ensure lesson audio files exist
   - Generate/verify viseme JSON files (Audio2Face output)
   - Test with sample lesson

3. **Test Integration:**
   - Visit `curiouskelly.com/lesson-player`
   - Verify Unity iframe loads
   - Test lesson loading and playback
   - Check status indicators

4. **Deploy:**
   - Deploy updated lesson-player to Cloudflare Pages
   - Deploy Unity build to CDN or `public/unity/`
   - Test on production domain

---

## 📍 Current URL Structure

- **Lesson Player:** `curiouskelly.com/lesson-player`
- **Unity Iframe:** `curiouskelly.com/unity/kelly-v1/index.html`
- **Demo Page:** `curiouskelly.com/demo/avatar/` (uses `UnityIframe.astro` component)

---

## 🔍 Testing Checklist

- [ ] Unity iframe loads without errors
- [ ] Status indicator shows "Kelly is ready"
- [ ] Lesson loads when selected
- [ ] Audio plays in sync with visemes
- [ ] Status updates correctly during playback
- [ ] Fallback to static images works if Unity fails
- [ ] Works across different browsers
- [ ] Mobile responsive

---

## 📝 Notes

- The Unity iframe is currently showing a placeholder
- Once the WebGL build is deployed, Kelly will appear in the iframe
- The messaging bridge (`kbridge.js`) is already in place
- All integration code is complete and ready for testing

---

**Integration Status:** ✅ Complete  
**Build Status:** ⚠️ Pending  
**Deployment Status:** ⏳ Waiting for Unity build




