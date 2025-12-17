# Kelly at 60 FPS - Setup Guide

## 🚀 Why 60 FPS for Kelly?

### Benefits:
- ✅ **Ultra-smooth motion** - Every facial movement is buttery smooth
- ✅ **Better lip-sync perception** - 16.67ms per frame (vs 33ms at 30 FPS)
- ✅ **More professional look** - High-quality production value
- ✅ **Future-proof** - High frame rate is becoming standard

### Trade-offs:
- ⚠️ **2x render time** - A 10-second clip takes 2x longer
- ⚠️ **Larger file sizes** - Videos are ~1.8-2x bigger
- ⚠️ **More GPU memory** - Need decent graphics card

---

## ⚙️ iClone Setup for 60 FPS

### Step 1: Change Project Frame Rate

```
Top Menu → Edit → Preferences → Project Settings
  
  Default FPS: 60 FPS
  
Click "OK"
```

### Step 2: Set Timeline to 60 FPS

```
In Timeline (bottom panel):
  Right-click on timeline ruler
  → Playback Settings
  → Frame Rate: 60 FPS
```

### Step 3: Verify Current Project

```
Top Menu → Project → Project Settings
  
  Current FPS: 60  ← Should show 60
```

---

## 🎤 Audio Setup (Same Files Work!)

### Good News:
Your existing audio files work perfectly at 60 FPS!

```
kelly_leaves_2-5.wav     ← Works at ANY FPS
kelly_leaves_6-12.wav    ← Works at ANY FPS
kelly_leaves_18-35.wav   ← Works at ANY FPS
...etc
```

### Why?
- Audio is **time-based**, not frame-based
- AccuLips generates smooth animation curves
- Curves interpolate between keyframes automatically
- 60 FPS just samples the curves more frequently

---

## 🎬 AccuLips Workflow (60 FPS)

### Process (Identical to 30 FPS):

1. **Import Audio:**
   ```
   Timeline → Right-click audio track
   → Import Audio
   → Select: kelly_leaves_18-35.wav
   ```

2. **Select Kelly in viewport**

3. **Run AccuLips:**
   ```
   Animation → Facial Animation → AccuLips
   
   Settings:
     Audio: [Your imported track]
     Language: English
     Quality: High  ← Same as before
     
   Click "Apply"
   ```

4. **What Changes at 60 FPS:**
   ```
   Processing time: ~3-5 minutes (slightly longer)
   Animation keyframes: 2x more frames generated
   Result: Smoother mouth movements
   ```

5. **Preview:**
   ```
   Press SPACEBAR
   Watch ultra-smooth lip-sync!
   ```

---

## 📊 Technical Differences

### Frame Precision:

| Frame Rate | ms per Frame | Frames per Second | Lip-Sync Accuracy |
|------------|--------------|-------------------|-------------------|
| 24 FPS | 41.67 ms | 24 | ±42ms |
| 30 FPS | 33.33 ms | 30 | ±33ms |
| **60 FPS** | **16.67 ms** | **60** | **±17ms** |

### Perception:
- Human audio-visual sync tolerance: ±50ms
- 60 FPS is **3x more accurate** than needed
- Result: **Perfectly imperceptible sync**

### Animation Data:

**10-second clip:**
```
30 FPS:
  300 frames
  300 × 53 blendshapes = 15,900 animation values
  
60 FPS:
  600 frames
  600 × 53 blendshapes = 31,800 animation values
  
2x the animation data!
```

---

## 🖥️ Rendering at 60 FPS

### Render Settings:

```
File → Export → Video

Video Settings:
  Resolution: 1920×1080 (or 3840×2160 for 4K)
  Frame Rate: 60 FPS  ← CRITICAL
  Format: MP4
  Codec: H.264
  Quality: High
  
Audio Settings:
  Include Audio: ✓
  Sample Rate: 48000 Hz
  Bitrate: 320 kbps
```

### Render Times (Estimates):

**10-second clip, 1080p:**
- CPU only: ~60-90 minutes
- RTX 2060: ~15-20 minutes
- RTX 3080: ~8-10 minutes
- RTX 4090: ~4-5 minutes

**10-second clip, 4K:**
- CPU only: ~3-4 hours
- RTX 2060: ~45-60 minutes
- RTX 3080: ~20-25 minutes
- RTX 4090: ~10-12 minutes

---

## 📦 File Sizes

### Expected Output Sizes:

**1080p, 10 seconds, H.264:**
```
30 FPS: ~20 MB
60 FPS: ~35-40 MB  ← 1.8-2x larger
```

**4K, 10 seconds, H.264:**
```
30 FPS: ~80 MB
60 FPS: ~140-160 MB
```

### Compression Tips:
```
High Quality (Teaching):
  Bitrate: 10-15 Mbps (larger files, perfect quality)
  
Medium Quality (Web):
  Bitrate: 5-8 Mbps (balanced)
  
Streaming Quality:
  Bitrate: 3-5 Mbps (smaller files, still good)
```

---

## 🌐 Web Delivery at 60 FPS

### HTML5 Video:

Your existing lesson player works perfectly!

```html
<video id="kelly-video" 
       controls 
       preload="auto"
       src="videos/kelly_leaves_18-35_60fps.mp4">
</video>
```

### Browser Support:
- ✅ Chrome/Edge: Full 60 FPS support
- ✅ Firefox: Full 60 FPS support
- ✅ Safari: Full 60 FPS support
- ✅ Mobile: Most devices support 60 FPS playback

### Bandwidth Requirements:

**30 FPS (20 MB / 10 sec):**
- Bitrate: ~16 Mbps
- Good for: 5 Mbps+ connections

**60 FPS (40 MB / 10 sec):**
- Bitrate: ~32 Mbps
- Good for: 10 Mbps+ connections

---

## 🎯 Workflow Summary (60 FPS)

### Complete Pipeline:

```
1. Lesson DNA (JSON)
   ↓
2. ElevenLabs API
   ↓ (No change - same audio files)
3. Audio Files (.wav)
   ↓
4. iClone (60 FPS project)
   ↓ Import audio
5. AccuLips
   ↓ (Generates 600 frames for 10-sec clip)
6. Animation (60 FPS)
   ↓ 53 blendshapes × 600 frames
7. Render (60 FPS, ~2x time)
   ↓
8. MP4 Output (60 FPS, ~2x size)
   ↓
9. Web Delivery (HTML5 video)
```

---

## ✅ Checklist: Converting to 60 FPS

### Before You Start:

- [ ] **GPU Check:** Do you have RTX 2060 or better?
- [ ] **Storage:** Do you have 2x disk space for videos?
- [ ] **Time:** Can you wait 2x longer for renders?
- [ ] **Bandwidth:** Will users have 10+ Mbps connections?

### If Yes to All:

- [ ] Change iClone preferences to 60 FPS
- [ ] Set project to 60 FPS
- [ ] Import audio (same files as before)
- [ ] Run AccuLips (same process)
- [ ] Render at 60 FPS
- [ ] Test playback in browser

---

## 🔧 Optimization Tips

### 1. Render Queue:
```
Batch render all 6 age variants overnight:
  kelly_leaves_2-5.mp4 (60 FPS)
  kelly_leaves_6-12.mp4 (60 FPS)
  kelly_leaves_13-17.mp4 (60 FPS)
  kelly_leaves_18-35.mp4 (60 FPS)
  kelly_leaves_36-60.mp4 (60 FPS)
  kelly_leaves_61-102.mp4 (60 FPS)
```

### 2. Progressive Download:
```html
<!-- In lesson-player -->
<video preload="metadata">
  <!-- Only loads first few frames -->
  <!-- Starts playing faster -->
</video>
```

### 3. Adaptive Streaming (Advanced):
```
Encode multiple bitrates:
  - 60 FPS @ 8 Mbps (high quality)
  - 60 FPS @ 5 Mbps (medium quality)
  - 30 FPS @ 3 Mbps (fallback for slow connections)
  
Use HLS or DASH for automatic switching
```

---

## 🧪 60 FPS Validation Pipeline

Follow this once the scene renders at 60 FPS to guarantee lip-sync quality and regression coverage across all six Kelly personas.

1. **Baseline Perf Capture** – Work through this guide, then profile each persona scene in Unity 2022.3 LTS. Export frame timing metrics to `analytics/Kelly/perf-baseline.csv`.
2. **Voice & Lip-Sync Prep** – Generate 90-second ElevenLabs reference clips for every age bucket, run Audio2Face to create viseme caches, and bind them inside Unity.
3. **Realtime Instrumentation** – Enable latency logging in `curious-kellly/backend` (`VoiceService`, `SessionService`) so every request logs RTT into `analytics/Kelly/voice-latency.csv`.
4. **Automated Perf Tests** – Add a Unity playmode test (e.g. `digital-kelly/engines/tests/Perf60Test.cs`) that fails if any frame drops below 58 FPS and wire it into CI.
5. **Persona QA Checklists** – Run an end-to-end session for each age bucket using the Flutter+Unity app, capturing screen + logs. Store findings in `analytics/Kelly/qa/persona-<bucket>.md`.
6. **Nightly Regression Sweep** – Schedule a batch script to replay the six personas nightly, publish a `analytics/Kelly/daily-report.json`, and alert on latency or FPS regressions.

Keep renders blocked until all six steps are green so the 60 FPS promise stays true in production.

---

## 🎬 The Result

### What You'll See:

**30 FPS Kelly:**
- Smooth lip-sync ✓
- Natural motion ✓
- Standard quality ✓

**60 FPS Kelly:**
- **Buttery smooth lip-sync** ✓✓✓
- **Ultra-fluid facial expressions** ✓✓✓
- **Premium production quality** ✓✓✓
- **Every micro-movement captured** ✓✓✓

### Perception Difference:
- 30 FPS: "This looks professional"
- 60 FPS: "This looks INCREDIBLY professional"

---

## 🚀 Ready to Render?

### Quick Start (iClone):

1. **Edit → Preferences → Project Settings**
   - Default FPS: **60**

2. **Import audio:** `kelly_leaves_18-35.wav`

3. **Animation → AccuLips**
   - Quality: **High**
   - Wait ~3-5 minutes

4. **File → Export → Video**
   - Frame Rate: **60 FPS**
   - Format: **MP4 (H.264)**
   - Quality: **High**

5. **Render overnight** (if doing all 6 variants)

6. **Watch ultra-smooth Kelly!** 🎉

---

## 💡 Pro Tip:

Start with **one age variant** (18-35) at 60 FPS to test:
- Render time on your hardware
- File size for your bandwidth
- Visual quality improvement

If you love it → batch render all 6!
If it's too slow → stick with 30 FPS

---

**60 FPS is ready to go! Your audio files work unchanged.** 🚀

Just change the FPS setting in iClone and render!



