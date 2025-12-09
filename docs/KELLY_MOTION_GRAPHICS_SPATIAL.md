# 🎬 KELLY MOTION GRAPHICS SPATIAL SYSTEM
## *iPhone Lock Screen Aesthetic for Motion Graphics + VFX*

---

## 🎯 **THE VISION**

```
Kelly is the wallpaper.
Moving. Breathing. Alive.
HD video. 3D models. VFX.

Portrait or landscape.
Time, date, lesson info float on top.
Always readable. Never covering her face.

Swipe up: lesson starts.
Gesture-first. Mobile-native.
Sexy. Smooth. Timeless.
```

---

## 🏗️ **ARCHITECTURE: THREE LAYERS**

### **Layer 1: Background (Kelly)**
- HD lipsync videos (MiniMax + Sync Labs)
- 3D models (Unity Kelly, Model Viewer)
- Dynamic backgrounds (infographics, scenes from Runway)
- **Full-bleed, always visible, never obscured**

### **Layer 2: Safe Zone Intelligence**
- Real-time tracking of Kelly's face/hands
- Dynamic safe zone calculation (60 FPS)
- Synced to video playback or 3D animation
- **Invisible, automatic, bulletproof**

### **Layer 3: Floating UI**
- Time, date, lesson info
- Action buttons, controls
- Popovers, panels
- **Always in safe zones, never on Kelly**

---

## 📐 **THREE IMPLEMENTATION APPROACHES**

### **Approach 1: Pre-Computed Manifests** ⭐ **RECOMMENDED**
**Best for:** HD videos with known content

**How it works:**
1. During video production, run pose detection
2. Generate safe zone manifest JSON
3. At runtime, sync manifest to video playback
4. Update safe zones every frame (60 FPS)

**Pros:**
- ✅ Fast (no runtime ML)
- ✅ Reliable (pre-computed)
- ✅ Works offline
- ✅ Small file size (~50KB per video)

**Cons:**
- ⚠️ Requires pre-processing
- ⚠️ One manifest per video

**Files:**
- `public/js/kelly-video-spatial.js` - Runtime system
- `scripts/generate-video-safe-zones.py` - Manifest generator
- `public/kelly/videos/day-001-phase-01-safe-zones.json` - Example manifest

---

### **Approach 2: Real-Time ML Detection**
**Best for:** 3D models, live content, dynamic scenes

**How it works:**
1. Load MediaPipe Pose or BlazePose in browser
2. Run pose detection every frame
3. Calculate safe zones in real-time
4. Update UI positions dynamically

**Pros:**
- ✅ Works with any content (3D, live, etc.)
- ✅ No pre-processing needed
- ✅ Adapts to unexpected movements

**Cons:**
- ⚠️ Requires ML library (~2MB)
- ⚠️ CPU/GPU intensive
- ⚠️ May lag on low-end devices

**Implementation:**
```javascript
// Load MediaPipe Pose
import { Pose } from '@mediapipe/pose';

const pose = new Pose({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});

pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

// Process video frame
pose.onResults((results) => {
  if (results.poseLandmarks) {
    const face = detectFaceBox(results.poseLandmarks);
    const hands = detectHandBoxes(results.poseLandmarks);
    const safeZones = calculateSafeZones(face, hands);
    updateUI(safeZones);
  }
});

// Send frames to pose detector
const video = document.getElementById('kelly-video');
const camera = new Camera(video, {
  onFrame: async () => {
    await pose.send({image: video});
  },
  width: 1280,
  height: 720
});
camera.start();
```

---

### **Approach 3: Timeline Markers** (Hybrid)
**Best for:** Animated content with known keyframes

**How it works:**
1. During animation (Blender, After Effects), add markers
2. Export markers as JSON timeline
3. At runtime, sync to animation playback
4. Interpolate between keyframes

**Pros:**
- ✅ Artist-controlled (animators set zones)
- ✅ Fast (no ML)
- ✅ Precise (manual annotation)

**Cons:**
- ⚠️ Manual work per video
- ⚠️ Requires animation software integration

**Example Timeline:**
```json
{
  "video_id": "day-001-phase-01",
  "keyframes": [
    {
      "time": 0.0,
      "kelly_face": { "x": 0.45, "y": 0.15, "width": 0.10, "height": 0.15 },
      "kelly_hands": [{ "x": 0.50, "y": 0.50, "gesture": "idle" }]
    },
    {
      "time": 3.5,
      "kelly_face": { "x": 0.45, "y": 0.15, "width": 0.10, "height": 0.15 },
      "kelly_hands": [{ "x": 0.25, "y": 0.50, "gesture": "pointing_left" }]
    }
  ]
}
```

---

## 🎨 **IPHONE LOCK SCREEN AESTHETIC**

### **Design Principles**

1. **Kelly is the Wallpaper**
   - Full-bleed video/3D
   - No borders, no frames
   - Fills entire screen (portrait or landscape)

2. **UI Floats on Top**
   - Frosted glass panels (backdrop-filter: blur(20px))
   - Subtle shadows (0 4px 16px rgba(0,0,0,0.2))
   - Always in safe zones

3. **Gesture-First**
   - Swipe up: start lesson
   - Swipe left/right: navigate phases
   - Tap: pause/play
   - Long press: options

4. **Responsive**
   - Portrait: Kelly center, UI top/bottom
   - Landscape: Kelly left, UI right
   - Adapts to safe zones in both orientations

---

## 📱 **MOBILE-FIRST LAYOUT**

### **Portrait Mode (9:16)**

```
┌─────────────────────────────────────┐
│  Time: 2:34 PM                      │ ← Top safe zone (frosted glass)
│  Date: Tuesday, Dec 9               │
│  Lesson: Sharing What We Have       │
├─────────────────────────────────────┤
│                                     │
│                                     │
│           KELLY                     │ ← Full-bleed video
│         (full-screen)               │
│                                     │
│                                     │
│                                     │
├─────────────────────────────────────┤
│  [Swipe up to start lesson]        │ ← Bottom safe zone
│  💡 📌 ✨ 🎤                         │
└─────────────────────────────────────┘
```

### **Landscape Mode (16:9)**

```
┌──────────────────────────────────────────────────────────────┐
│                                │  Time: 2:34 PM               │
│                                │  Date: Tuesday, Dec 9        │
│                                │  Lesson: Sharing             │
│         KELLY                  │  ────────────────────        │
│      (full-screen)             │  Phase 1 of 5: Hook          │
│                                │                              │
│                                │  💡 Aha  📌 Pin  ✨ Share   │
│                                │  🎤 Talk  📅 Cal  🔍 Search  │
│                                │  ────────────────────        │
│                                │  [Swipe left for next]       │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎬 **PRODUCTION PIPELINE**

### **Step 1: Generate HD Videos**
```bash
# Run HD Golden Lesson Pipeline
cd scripts/kelly-video-factory
npm run generate-hd-video -- --day 1 --phase 1

# Output: public/kelly/videos/day-001-phase-01-hook.mp4
```

### **Step 2: Generate Safe Zone Manifest**
```bash
# Run pose detection on video
python scripts/generate-video-safe-zones.py \
  --video public/kelly/videos/day-001-phase-01-hook.mp4 \
  --output public/kelly/videos/day-001-phase-01-hook-safe-zones.json \
  --sample-rate 30

# Output: day-001-phase-01-hook-safe-zones.json (~50KB)
```

### **Step 3: Upload to Supabase Storage**
```bash
# Upload video + manifest
supabase storage upload videos day-001-phase-01-hook.mp4
supabase storage upload videos day-001-phase-01-hook-safe-zones.json

# CDN URLs:
# https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/videos/day-001-phase-01-hook.mp4
# https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/videos/day-001-phase-01-hook-safe-zones.json
```

### **Step 4: Integrate into learn.html**
```html
<!-- Kelly Video Container -->
<div class="kelly-wallpaper" id="kelly-wallpaper">
  <video 
    id="kelly-video"
    data-video-id="day-001-phase-01-hook"
    src="https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/videos/day-001-phase-01-hook.mp4"
    autoplay
    loop
    muted
    playsinline
  ></video>
</div>

<!-- Floating UI (positioned in safe zones) -->
<div class="floating-ui" id="floating-ui">
  <div class="time-date-panel" id="time-date-panel">
    <div class="time">2:34 PM</div>
    <div class="date">Tuesday, December 9</div>
    <div class="lesson-title">Sharing What We Have</div>
  </div>
  
  <div class="action-dock" id="action-dock">
    <button class="action-btn">💡</button>
    <button class="action-btn">📌</button>
    <button class="action-btn">✨</button>
    <button class="action-btn">🎤</button>
  </div>
</div>

<!-- Kelly Video Spatial System -->
<script src="/js/kelly-video-spatial.js"></script>
<script>
  // Initialize when video loads
  const kellyVideo = document.getElementById('kelly-video');
  kellyVideo.addEventListener('loadedmetadata', async () => {
    const videoId = kellyVideo.dataset.videoId;
    const manifestUrl = `https://tvjalxxsyryjphkforjv.supabase.co/storage/v1/object/public/videos/${videoId}-safe-zones.json`;
    
    await window.kellyVideoSpatial.init(kellyVideo, manifestUrl);
    
    // Position floating UI in safe zones
    positionFloatingUI();
  });
  
  // Listen for safe zone updates (60 FPS)
  window.addEventListener('kelly-safe-zones-updated', (event) => {
    const { safeZones } = event.detail;
    positionFloatingUI(safeZones);
  });
  
  function positionFloatingUI(safeZones) {
    // Position time/date panel in top safe zone
    const timeDatePanel = document.getElementById('time-date-panel');
    window.kellyVideoSpatial.positionPopover(timeDatePanel, 'top-left');
    
    // Position action dock in bottom safe zone
    const actionDock = document.getElementById('action-dock');
    window.kellyVideoSpatial.positionPopover(actionDock, 'bottom-right');
  }
</script>
```

---

## 🎨 **STYLING: FROSTED GLASS AESTHETIC**

```css
/* Kelly Wallpaper (Full-bleed) */
.kelly-wallpaper {
  position: fixed;
  inset: 0;
  z-index: 0;
  overflow: hidden;
}

#kelly-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center 20%;
}

/* Floating UI (iPhone Lock Screen Style) */
.floating-ui {
  position: fixed;
  inset: 0;
  z-index: 10;
  pointer-events: none; /* Let taps through to video */
}

.floating-ui > * {
  pointer-events: auto; /* But panels are interactive */
}

/* Time/Date Panel (Frosted Glass) */
.time-date-panel {
  position: absolute;
  /* Position set by kelly-video-spatial.js */
  
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  
  border-radius: 16px;
  padding: 16px 20px;
  
  box-shadow: 
    0 4px 16px rgba(0, 0, 0, 0.1),
    0 1px 2px rgba(0, 0, 0, 0.05),
    inset 0 1px 0 rgba(255, 255, 255, 0.5);
  
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.time {
  font-size: 48px;
  font-weight: 700;
  color: #000;
  line-height: 1;
  margin-bottom: 4px;
}

.date {
  font-size: 16px;
  font-weight: 500;
  color: rgba(0, 0, 0, 0.7);
  margin-bottom: 12px;
}

.lesson-title {
  font-size: 14px;
  font-weight: 600;
  color: #3B82F6;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* Action Dock (iPhone Style) */
.action-dock {
  position: absolute;
  /* Position set by kelly-video-spatial.js */
  
  display: flex;
  gap: 12px;
  padding: 12px;
  
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
  
  border-radius: 24px;
  
  box-shadow: 
    0 4px 16px rgba(0, 0, 0, 0.1),
    0 1px 2px rgba(0, 0, 0, 0.05);
  
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.action-btn {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  border: none;
  background: rgba(255, 255, 255, 0.9);
  font-size: 24px;
  cursor: pointer;
  transition: all 0.2s ease;
  
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.action-btn:hover {
  transform: scale(1.1);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.action-btn:active {
  transform: scale(0.95);
}

/* Responsive: Portrait */
@media (orientation: portrait) {
  .time-date-panel {
    /* Positioned by JS in top safe zone */
  }
  
  .action-dock {
    /* Positioned by JS in bottom safe zone */
  }
}

/* Responsive: Landscape */
@media (orientation: landscape) {
  .time-date-panel {
    /* Positioned by JS in right safe zone */
  }
  
  .action-dock {
    /* Positioned by JS in right safe zone, below time/date */
  }
}
```

---

## 🚀 **DEPLOYMENT CHECKLIST**

### **Phase 1: Single Video Test**
- [ ] Generate 1 HD video (Day 1, Phase 1)
- [ ] Run pose detection, generate manifest
- [ ] Upload video + manifest to Supabase
- [ ] Test in learn.html with kelly-video-spatial.js
- [ ] Verify safe zones update in real-time
- [ ] Test portrait + landscape orientations

### **Phase 2: Full Day Test**
- [ ] Generate 5 videos (Day 1, all phases)
- [ ] Generate 5 manifests
- [ ] Upload all to Supabase
- [ ] Test phase transitions
- [ ] Verify safe zones adapt to different poses

### **Phase 3: Production Scale**
- [ ] Generate 1,825 videos (365 days × 5 phases)
- [ ] Generate 1,825 manifests
- [ ] Batch upload to Supabase
- [ ] Implement CDN caching
- [ ] Monitor performance (60 FPS target)

---

## 📊 **PERFORMANCE TARGETS**

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Video Load Time** | < 2s | Time to first frame |
| **Manifest Load Time** | < 100ms | JSON fetch + parse |
| **Safe Zone Update Rate** | 60 FPS | requestAnimationFrame |
| **UI Reposition Latency** | < 16ms | Time from zone update to UI move |
| **Memory Usage** | < 100MB | Chrome DevTools Memory |
| **CPU Usage** | < 30% | Chrome DevTools Performance |

---

## 🎯 **SUCCESS CRITERIA**

### **Technical**
- ✅ Safe zones never overlap Kelly's face
- ✅ Safe zones never overlap Kelly's hands when gesturing
- ✅ UI repositions smoothly (no jank)
- ✅ Works in portrait + landscape
- ✅ Works on mobile + desktop
- ✅ 60 FPS maintained

### **Aesthetic**
- ✅ Feels like iPhone lock screen
- ✅ Frosted glass panels look premium
- ✅ Kelly is the star (UI fades to background)
- ✅ Gestures feel natural
- ✅ Transitions are smooth

### **User Experience**
- ✅ Learners say "wow"
- ✅ UI never blocks Kelly
- ✅ Everything is readable
- ✅ Feels magical, not technical
- ✅ "This is AGI" reactions

---

## 🛠️ **TOOLS & DEPENDENCIES**

### **Production (Video Generation)**
- Python 3.9+
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- NumPy (`pip install numpy`)

### **Runtime (Browser)**
- `kelly-video-spatial.js` (no dependencies)
- Modern browser with:
  - `<video>` element
  - `requestAnimationFrame`
  - `fetch` API
  - CSS `backdrop-filter`

### **Optional (Real-Time ML)**
- MediaPipe Pose (browser version)
- TensorFlow.js
- WebGL support

---

## 📝 **NEXT STEPS**

1. **Test Approach 1** (Pre-Computed Manifests)
   - Generate 1 video + manifest
   - Test in browser
   - Measure performance

2. **Refine Safe Zone Algorithm**
   - Adjust zone sizes
   - Tune overlap detection
   - Optimize scoring

3. **Design Floating UI**
   - iPhone lock screen aesthetic
   - Frosted glass panels
   - Smooth animations

4. **Scale to Production**
   - Batch process 1,825 videos
   - Upload to Supabase CDN
   - Monitor performance

5. **Add Gesture Controls**
   - Swipe up: start lesson
   - Swipe left/right: navigate
   - Tap: pause/play
   - Long press: options

---

**Status:** 🟡 Design Complete, Ready for Testing  
**Next:** Generate first video + manifest, test in browser  
**Goal:** iPhone lock screen aesthetic with motion graphics






