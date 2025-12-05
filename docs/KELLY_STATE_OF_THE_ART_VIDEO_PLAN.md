# 🎬 Kelly State-of-the-Art Video Generation Pipeline

## Executive Summary

**Current Problem:** SadTalker produces "JibJab" quality - only mouth moves on static image.

**What We Need:** Film-quality animation with:
- Full facial expressions (eyes, brows, micro-expressions)
- Natural hair movement
- Full body animation
- Interaction with teaching elements
- Consistent Kelly identity

**What We Already Have (but aren't using):**
- CC5 + iClone 8.62 pipeline with AccuLips
- Unity 3D Kelly with ARKit blendshapes
- RTX 5090 (32GB VRAM) for 8K rendering
- Audio2Face integration code
- Kelly LoRA for consistent image generation

---

## 🔍 Asset Inventory

### 3D Kelly (Unity/iClone)
| Asset | Location | Capability |
|-------|----------|------------|
| `Kelly_CC5_WebGL.Fbx` | `digital-kelly/engines/` | WebGL-ready with blendshapes |
| `kelly_fbx_v4.fbx` | `Assets/Kelly/Animations/Lessons/` | Lesson animations |
| `ARKitBlendshapeController.cs` | `Assets/Scripts/` | 52+ facial blendshapes |
| `LipSyncController.cs` | `Assets/Scripts/` | Real-time audio → lip-sync |
| `Kelly_RealisticSkin.shader` | `Assets/Shaders/` | SSS skin rendering |
| CC5 Project Files | `projects/Kelly/CC5/` | Source character |

### iClone Pipeline (Documented in Kelly_HD_Pipeline.md)
| Tool | Purpose |
|------|---------|
| Character Creator 5 | Base model + Headshot 2 |
| iClone 8.62 | Animation + Rendering |
| AccuLips | Audio → Phoneme → Lip-sync |
| AccuFACE | Video → Facial expressions |
| Digital Human Shader | Film-quality skin |

### 2D Kelly
| Asset | Count | Resolution |
|-------|-------|------------|
| Pose images | 7 | 3072×5504 |
| Lesson images | 365+ | 1024×1536 |
| Phase images | 1825+ | Various |
| LoRA model | 1 | Consistent identity |

---

## 🎯 Three-Tier Video Architecture

### TIER 1: Pre-Rendered Cinema (Highest Quality)
**Use Case:** Daily lessons, marketing, hero content

```
ElevenLabs TTS → AccuLips (iClone) → 4K Render → CDN
                       ↓
              Full facial animation:
              - AccuFACE brow/eye motion
              - AccuLips lip-sync
              - Hair physics simulation
              - Full body idle animation
              - Environment lighting
```

**Pipeline:**
1. Generate audio with ElevenLabs (Kelly voice)
2. Import to iClone 8.62
3. AccuLips generates phoneme timing
4. AccuFACE adds eye/brow micro-expressions
5. Render at 4K/60fps with RTX 5090
6. Store in CDN for instant playback

**Quality Target:** Pixar-level character animation
**Time:** 10-20 min render per 30s clip
**Cost:** ~$0.05 (ElevenLabs) + render time

---

### TIER 2: AI Video Generation (High Quality, Flexible)
**Use Case:** Dynamic content, custom phrases, real-time-ish

**Option A: DreamActor-M1 (Full Body)**
```
Kelly LoRA Image + ElevenLabs Audio → DreamActor-M1 → Video
```
- Full body animation from single image
- Audio-driven lip-sync
- Natural head/body movement
- 10-30 second generation time

**Option B: LatentSync (High-Fidelity Lip-Sync)**
```
Kelly Image + Audio → LatentSync → Video
```
- Extremely precise lip-sync
- Non-frontal face support
- Multi-language

**Option C: Audio2Face → Video Render**
```
ElevenLabs Audio → Audio2Face → Blendshape Animation → Unity Render → Video
```
- Uses existing Unity Kelly
- Real-time capable
- Can record to video

---

### TIER 3: Real-Time Interactive (Low Latency)
**Use Case:** Live conversations, Q&A, personalized responses

```
User Speech → AI Response → ElevenLabs Stream → Unity Kelly
                                    ↓
                          Audio2Face / LipSyncController
                                    ↓
                          WebGL Blendshape Animation
```

**Existing Code:**
- `LipSyncController.cs` - Spectrum analysis → phonemes
- `ARKitBlendshapeController.cs` - 52 ARKit blendshapes
- `ElevenLabsAudioManager.cs` - TTS + playback
- `KellyWebGLBridge.cs` - Browser integration

**Upgrade Path:**
1. Replace spectrum analysis with Audio2Face SDK
2. Add eye gaze/blink from emotion detection
3. Add body idle animation
4. Add micro-expression triggers

---

## 🛠 Recommended Implementation Order

### Week 1: Activate iClone Pipeline (Tier 1)
**Goal:** Generate first film-quality lesson video

1. **Day 1-2:** Set up CC5 → iClone workflow
   - Load `Kelly_HS2_HD.ccProject`
   - Configure Director's Chair scene
   - Verify AccuLips installation

2. **Day 3-4:** Generate test video
   - ElevenLabs audio for "Welcome to Day 1"
   - AccuLips phoneme generation
   - AccuFACE eye/brow layer
   - 4K render test

3. **Day 5:** Batch automation
   - Script to generate multiple lesson clips
   - CDN upload pipeline

### Week 2: Integrate DreamActor-M1 (Tier 2)
**Goal:** On-demand video generation API

1. Research DreamActor-M1 API access
2. Build generation pipeline:
   ```typescript
   async function generateKellyVideo(text: string): Promise<Buffer> {
     const audio = await generateElevenLabsAudio(text);
     const image = await getKellyLoRAImage(poseType);
     const video = await dreamActorGenerate(image, audio);
     return video;
   }
   ```

### Week 3: Upgrade Unity Real-Time (Tier 3)
**Goal:** Cinema-quality real-time avatar

1. Integrate Audio2Face SDK (now open source)
2. Add eye gaze system
3. Add emotion-to-expression mapping
4. Add body animation blend

---

## 📊 Quality Comparison

| Method | Lip-Sync | Face | Hair | Body | Latency | Cost |
|--------|----------|------|------|------|---------|------|
| SadTalker (current) | 70% | Static | Static | Static | 2-3 min | $0.02 |
| iClone + AccuLips | 99% | Full | Physics | Animated | 10-20 min | $0.05 |
| DreamActor-M1 | 95% | Full | Animated | Full | 10-30s | TBD |
| Unity + Audio2Face | 90% | Full | Static | Animated | Real-time | $0.05 |

---

## 🎨 Visual Quality Targets

### What "Film Quality" Means:
1. **Lips:** Perfect phoneme match, no drift
2. **Eyes:** Micro-saccades, natural blinks (3-4/sec when speaking)
3. **Brows:** Expression-appropriate movement
4. **Cheeks:** Subtle SSS, natural movement
5. **Hair:** Physical simulation or animated flow
6. **Body:** Breathing, weight shifts, gesture timing
7. **Lighting:** Consistent studio look, depth

### Kelly's Signature Traits to Preserve:
- Warm, approachable smile
- Curious eye expressions
- Natural head tilts when listening
- Hand gestures when explaining
- Blue cashmere sweater consistency

---

## 🔧 Technical Integration Points

### iClone → Web Delivery
```
iClone 4K MP4 → FFmpeg compress → CDN → Video Player
```

### DreamActor/LatentSync → Web
```
Generation API → MP4 → Supabase Storage → Signed URL → Video Player
```

### Unity WebGL Real-Time
```
Browser → WebSocket → ElevenLabs Stream → Unity WebGL → Canvas
                                              ↓
                                    Audio2Face Blendshapes
```

---

## 📁 Files to Create/Modify

### New Scripts
- `scripts/iclone-batch-render.py` - Batch lesson rendering
- `scripts/dreamactor-generate.ts` - DreamActor integration
- `scripts/latentsync-generate.ts` - LatentSync integration

### Modify Existing
- `LipSyncController.cs` - Upgrade to Audio2Face
- `kelly-video-player.js` - Support tiered quality
- `kelly-lesson-system.js` - Tier selection logic

---

## 💰 Cost Analysis

### Pre-Rendered Library (365 lessons × 5 phases = 1825 videos)
| Item | Cost |
|------|------|
| ElevenLabs TTS | ~$90 |
| iClone render time | Free (hardware owned) |
| CDN storage (50GB) | ~$5/month |
| **Total** | **~$100 + $5/month** |

### On-Demand Generation (per video)
| Service | Est. Cost |
|---------|-----------|
| ElevenLabs TTS | $0.05 |
| DreamActor-M1 | TBD |
| LatentSync | TBD |

---

## 🚀 Next Steps

### Immediate (Today)
1. [ ] Verify iClone 8.62 + AccuLips is installed
2. [ ] Load Kelly CC5 project
3. [ ] Generate test render with existing audio

### This Week
4. [ ] Set up batch rendering workflow
5. [ ] Research DreamActor-M1 API access
6. [ ] Test LatentSync API

### Next Week
7. [ ] Pre-render launch week lessons (Days 1-7)
8. [ ] Integrate video player tier selection
9. [ ] Begin Audio2Face Unity upgrade

---

## References

- `Kelly_HD_Pipeline.md` - Full iClone workflow
- `KELLY_CORE_ASSET_ROADMAP.md` - Asset specifications
- `digital-kelly/engines/Kelly_Engine_V2/` - Unity project
- `KELLY_ANIMATION_INVENTORY.md` - Available animations
- AccuLips documentation: reallusion.com
- Audio2Face SDK: nvidia.com/audio2face
- DreamActor-M1: dreamactorm-1.com


