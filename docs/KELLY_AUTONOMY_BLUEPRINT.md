# 🚀 KELLY AUTONOMY BLUEPRINT
> **Created:** December 18, 2025  
> **Goal:** Break free from HeyGen/Sync Labs dependencies  
> **Status:** STRATEGIC PLANNING

---

## 🎯 THE PROBLEM (Deeply Understood)

### Current State
| Service | Cost | Speed | Control |
|---------|------|-------|---------|
| HeyGen | ~$0.05/min | 15-60+ min queue | ❌ None |
| Sync Labs | ~$0.08/min | 2-10 min queue | ❌ None |
| **Target** | **$0** | **< 1 second** | **✅ Full** |

### What We're Paying For
1. **Lip-sync AI** - Mapping audio to mouth movements
2. **Face animation** - Eye blinks, micro-expressions
3. **Motion synthesis** - Head movement, gestures
4. **Video encoding** - Compositing and rendering

### The Insight
**Kelly is ONE fixed character.** We don't need a general-purpose avatar system. We need a Kelly-specific animation system that can be:
- Pre-computed once
- Infinitely reused
- Rendered in real-time

---

## 🧠 DEEP ANALYSIS: How Talking Head Videos Work

### The Pipeline (What HeyGen/Sync Actually Do)

```
Audio → Phoneme Detection → Viseme Mapping → Face Landmarks → Motion Synthesis → Rendering
```

#### Step 1: Phoneme Detection
- Audio is analyzed for phonemes (sounds)
- ~44 phonemes in English
- This is FAST and can run locally (Vosk, Whisper)

#### Step 2: Viseme Mapping
- Phonemes map to visemes (mouth shapes)
- Only ~15 distinct visemes needed
- This is a LOOKUP TABLE, not AI

#### Step 3: Face Landmarks
- 68-point face mesh (standard)
- Kelly's face is FIXED - we can pre-compute this
- Store landmarks for each viseme

#### Step 4: Motion Synthesis
- Head movement, eye blinks, micro-expressions
- HeyGen's "Kling" motions are just pre-recorded patterns
- We can extract these from existing videos

#### Step 5: Rendering
- Blend viseme sprites/deformations
- Apply motion offsets
- Output video frames

### The Key Realization
**Steps 2-5 don't need to happen in real-time per-video.**

If we have:
- Kelly's face texture
- Pre-computed viseme sprites (15 mouth shapes)
- Motion patterns extracted from HeyGen videos
- A simple compositor

...we can generate Kelly videos **locally in seconds**.

---

## 🔬 TECHNICAL APPROACHES (Ordered by Feasibility)

### Approach 1: Sprite-Based Lip-Sync (EASIEST)
**Effort: 2-3 days | Quality: Good | Speed: 10-50 FPS**

```
Pre-render 15 Kelly mouth sprites → Audio → Phoneme → Viseme → Swap sprites → Export
```

**What We Build:**
1. Extract 15 viseme frames from best HeyGen video
2. Simple phoneme-to-viseme mapper (table lookup)
3. FFmpeg-based compositor

**Pros:**
- Dead simple
- Works today
- No AI/GPU needed

**Cons:**
- Slightly "sprite-y" look
- No smooth blending between frames

### Approach 2: Local Wav2Lip (FASTEST PATH TO QUALITY)
**Effort: 1-2 days | Quality: Excellent | Speed: 5-10 sec/video**

```
Kelly image + Audio → Wav2Lip model → Output video
```

**What We Build:**
1. Install Wav2Lip locally (Python + GPU)
2. Use consistent Kelly base image
3. Add motion overlay from HeyGen templates

**Pros:**
- Production quality
- Runs on local GPU
- No queue, no cost per video

**Cons:**
- Requires GPU (RTX 2060+ or M1 Mac)
- Initial setup complexity

### Approach 3: SadTalker (BETTER EXPRESSIONS)
**Effort: 2-3 days | Quality: Excellent+ | Speed: 10-20 sec/video**

```
Kelly image + Audio → SadTalker → Natural expressions + lip-sync
```

**What We Build:**
1. Install SadTalker locally
2. Configure for Kelly's face dimensions
3. Extract expression "drivers" from HeyGen videos

**Pros:**
- More expressive than Wav2Lip
- Includes head motion
- Still local, still free

**Cons:**
- Slightly more GPU-intensive
- More parameters to tune

### Approach 4: Motion Templates + Delta Compression (ADVANCED)
**Effort: 1-2 weeks | Quality: Indistinguishable | Speed: Real-time**

```
Analyze HeyGen videos → Extract pixel deltas → Store as motion templates → Apply in real-time
```

**What We Build:**
1. Motion extractor (optical flow analysis)
2. Delta encoder (what pixels change, when)
3. Real-time player (apply deltas to base frame)

**Pros:**
- Perfect quality (it's literally HeyGen's output, compressed)
- Real-time playback
- Tiny file sizes

**Cons:**
- Complex to build
- Requires good HeyGen samples first

### Approach 5: WebGL Real-Time Puppet (THE ENDGAME)
**Effort: 2-4 weeks | Quality: Custom | Speed: 60 FPS real-time**

```
2D Kelly mesh → Audio analysis → Deform mesh → Render in browser
```

**What We Build:**
1. Kelly face mesh (from best image)
2. Rigged mouth with 15 viseme blend shapes
3. WebGL renderer
4. Audio-reactive driver

**Pros:**
- Zero latency
- No video encoding ever
- Interactive (Kelly responds in real-time)
- Works in browser

**Cons:**
- Significant development
- Different aesthetic (more "animated")

---

## 📊 WHAT WE HAVE (Assets for Autonomy)

### From HeyGen (Completed Videos)
- ✅ 9+ Day 351 videos with Kling motion baked in
- ✅ Consistent Kelly appearance across archetypes
- ✅ High-quality lip-sync reference

### From Our Pipeline
- ✅ Kelly's voice (ElevenLabs, trained)
- ✅ Kelly head images (12 archetypes, 6 ages)
- ✅ Lesson scripts (365 days)

### What We Need to Extract
- [ ] 15 viseme sprite frames
- [ ] Motion patterns (optical flow)
- [ ] Face landmarks per archetype

---

## 🎬 RECOMMENDED ROADMAP

### Phase 1: Motion Library (TODAY)
```powershell
# Download all completed HeyGen videos to permanent storage
npx tsx scripts/backup-heygen-videos.ts
```

Extract from each video:
- Keyframes at 1fps
- Face landmarks (dlib/mediapipe)
- Audio-to-frame alignment

### Phase 2: Local Lip-Sync Setup (THIS WEEK)
```bash
# Option A: Wav2Lip
git clone https://github.com/Rudrabha/Wav2Lip
pip install -r requirements.txt
python inference.py --face kelly.png --audio lesson.mp3

# Option B: SadTalker  
git clone https://github.com/OpenTalker/SadTalker
pip install -r requirements.txt
python inference.py --driven_audio lesson.mp3 --source_image kelly.png
```

### Phase 3: Hybrid Pipeline (NEXT WEEK)
Combine:
- Local lip-sync for mouth
- HeyGen motion templates for body/gestures
- FFmpeg for composition

### Phase 4: Real-Time Kelly (MONTH 2)
Build WebGL puppet system for live interaction.

---

## 💰 COST COMPARISON (365 Days × 12 Archetypes = 4,380 Videos)

| Approach | Per-Video Cost | Total Cost | Time to Complete |
|----------|---------------|------------|------------------|
| HeyGen | ~$5 | **$21,900** | 73 days (at 60 videos/day) |
| Sync Labs | ~$6 | **$26,280** | 36 days (at 120 videos/day) |
| Local Wav2Lip | ~$0.01 (electricity) | **$44** | 5 days (at 1000 videos/day) |
| Local SadTalker | ~$0.02 | **$88** | 7 days |
| Sprite-Based | ~$0 | **$0** | 1 day (real-time) |
| WebGL Real-Time | ~$0 | **$0** | Instant |

---

## 🛠️ IMMEDIATE ACTIONS

### 1. Archive HeyGen Videos NOW
The completed videos have expiring URLs. Archive them:
```powershell
npx tsx scripts/backup-heygen-videos.ts
```

### 2. Test Local Wav2Lip
```bash
# Quick test with one Kelly image + one audio
python inference.py --checkpoint_path wav2lip_gan.pth \
  --face kelly_scientist.png \
  --audio day351_hook.mp3 \
  --outfile test_output.mp4
```

### 3. Extract Viseme Frames
From best HeyGen video, extract 15 mouth positions for sprite fallback.

### 4. Build Simple Compositor
FFmpeg script that swaps mouth sprites based on audio.

---

## 🎯 THE VISION

**Short-term (December 2025):**
- HeyGen for motion base videos (one-time)
- Sync Labs for re-dubs (temporary)
- Local Wav2Lip for overflow

**Medium-term (January 2026):**
- All new videos generated locally
- HeyGen only for new archetype bases
- Zero per-video cost

**Long-term (Q2 2026):**
- Real-time Kelly in browser
- Interactive avatar
- Complete autonomy from all external services

---

## 🔑 KEY INSIGHT

> **HeyGen and Sync Labs are selling us a solved problem.**
>
> Lip-sync is a 2D image transformation problem. The AI models are open-source (Wav2Lip, SadTalker, DINet). The only thing they have is convenience and a queue.
>
> We have something they don't: **ONE character to optimize for.**
>
> A Kelly-specific system will always beat a general-purpose avatar service because:
> 1. We can pre-compute everything Kelly-specific
> 2. We can cache and reuse across all videos
> 3. We control the quality and speed tradeoffs
> 4. We pay once, not per video

---

## 📁 Related Files
- `scripts/backup-heygen-videos.ts` - Archive completed videos
- `scripts/extract-visemes.ts` - Extract mouth shapes (TODO)
- `scripts/local-lipsync.py` - Wav2Lip wrapper (TODO)
- `public/admin/pipeline-monitor.html` - Live dashboard

---

**BOTTOM LINE:** We're paying HeyGen ~$20,000/year for something that can run on a laptop for free. The path to autonomy is clear:

1. ✅ Archive what we've paid for (HeyGen videos)
2. 🔄 Set up local lip-sync (Wav2Lip)
3. 📦 Extract motion patterns
4. 🚀 Build Kelly-specific renderer
5. 🎉 Never pay per-video again
