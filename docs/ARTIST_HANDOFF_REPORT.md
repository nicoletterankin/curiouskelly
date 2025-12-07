# 🎬 Kelly Video Production — Artist Handoff Report

## Prepared by: Chief Video Officer  
## Date: December 7, 2025  
## Project: Curious Kelly — The Daily Lesson  
## Launch: December 17, 2025

---

## 1. EXECUTIVE SUMMARY

**Curious Kelly** is an AI-powered educational platform delivering daily 8-minute lessons for learners ages 2-102. Kelly is our photorealistic digital human teacher — the face and voice that millions of learners will see every day. **Your work as the CC5/iClone/ZBrush artist is the foundation of Kelly's identity.**

**Current State:** We have a functioning video pipeline producing AI-generated content, but the quality is inconsistent ("JibJab level"). The tools we **already own** (CC5, iClone 8.62, AccuLips) can produce **99% film-quality** results. This handoff document ensures a seamless migration to that standard.

**Immediate Need:** We need **8 base video templates** and a **7-expression library** rendered at 4K/60fps to power 21,900 lesson videos for the full year. Launch content for Days 1-7 (420 videos) must be complete by **December 15, 2025**.

**Your Deliverables:**
1. 8 looping video templates (4K, 60fps, ProRes/PNG sequence)
2. 7 canonical expression images (4096px, PNG with alpha)
3. Unity-compatible FBX with 52 ARKit blendshapes
4. Verified AccuLips configuration for batch rendering

---

## 2. CURRENT PRODUCTION STATUS

### Assets Generated (Database Audit: December 7, 2025)

| Asset Type | Total Count | Days Covered | Cost to Date |
|------------|-------------|--------------|--------------|
| Images | 25 | 5 | $0.00 |
| Animations | 25 | 5 | $1.25 |
| Audio | 924 | 17 | $0.00 |
| Video | 239 | 5 | $4.78 |
| **TOTAL** | **1,213** | **17** | **$6.03** |

### Day-by-Day Coverage

| Day | Images | Animations | Audio | Videos | Status |
|-----|--------|------------|-------|--------|--------|
| 1 | 5 | 5 | 75 | 75 | ✅ Complete |
| 2 | 5 | 5 | 60 | 60 | ✅ Complete |
| 3 | 5 | 5 | 60 | 37 | ⚠️ Partial |
| 4 | 5 | 5 | 60 | 32 | ⚠️ Partial |
| 5 | 5 | 5 | 60 | 35 | ⚠️ Partial |
| 6-15 | 0 | 0 | 60 | 0 | 🔴 Audio only |
| 16-17 | 0 | 0 | 4-5 | 0 | 🔴 Minimal |

### Quality Issues Identified

1. **Face audit scores:** Not populated (face_audit_passed = 0 for all)
2. **Sweater color:** Not validated (sweater_color_check = NULL)
3. **Resolution:** Current videos are 1080p (need 4K)
4. **Frame rate:** Current videos are 25-30fps (need 60fps)
5. **Lip-sync accuracy:** ~70% (SadTalker fallback), need 99% (AccuLips)

### Gap Analysis

**Missing for Launch (Days 1-7):**
- 4K renders of all existing videos
- AccuLips-quality lip-sync
- 60fps frame rate
- Face audit validation

**Missing for Full Year:**
- 358 days without video assets
- ~20,000+ videos to generate

---

## 3. KELLY VISUAL SPECIFICATION

### Character Bible

**Identity:** Kelly — Friendly, approachable AI teacher  
**Age Appearance:** Late 20s (27-29 years)  
**Ethnicity:** Caucasian with warm undertones  
**Build:** Athletic, healthy, approachable

### Facial Features

| Feature | Specification |
|---------|---------------|
| **Face Shape** | Soft oval, friendly proportions |
| **Eyes** | Warm brown, visible catchlights, expressive |
| **Eye Size** | Slightly larger than average (approachable) |
| **Eyebrows** | Natural arch, well-defined |
| **Nose** | Straight bridge, soft tip |
| **Lips** | Full, natural pink, genuine smile lines |
| **Skin** | Fair/medium, warm undertones, subtle freckles allowed |
| **Cheeks** | Soft, natural blush position |

### Hair Specification

| Attribute | Value |
|-----------|-------|
| **Color** | Chestnut brown with subtle caramel highlights |
| **Color RGB** | Base: #5D4037, Highlights: #8D6E63 |
| **Length** | Below shoulders (mid-back when straight) |
| **Style** | Long, wavy, natural movement |
| **Texture** | Healthy shine, not greasy |
| **Physics** | REQUIRED — must move with head motion |

### Wardrobe — CRITICAL

| Item | Specification |
|------|---------------|
| **Top** | Soft powder blue crewneck sweater |
| **Color Name** | Powder Blue / Light Steel Blue |
| **Color RGB** | R: 176, G: 196, B: 222 (#B0C4DE) |
| **Texture** | Soft ribbed knit, visible weave |
| **Fit** | Relaxed, comfortable, not tight |

⚠️ **CRITICAL WARNING — SWEATER COLOR:**
```
✅ CORRECT: Powder blue (#B0C4DE), Light steel blue
❌ WRONG: Teal, turquoise, purple, pink, red, beige, green, yellow
```

The sweater color has been our #1 QA failure. Every render must be validated against the RGB values above.

### Skin Rendering

| Parameter | Value |
|-----------|-------|
| **Shader** | Digital Human Shader (CC5) |
| **SSS Intensity** | 0.25-0.30 |
| **Roughness** | 0.40-0.45 |
| **Pore Detail** | Maximum |
| **Normal Map** | 8K minimum |

### Eye Rendering

| Parameter | Value |
|-----------|-------|
| **Type** | HD Eyes (CC5) |
| **Catchlight** | Two-point studio (soft boxes) |
| **Pupil Dilation** | Dynamic based on emotion |
| **Sclera Veins** | Subtle, not prominent |
| **Moisture** | Visible, not teary |

---

## 4. CANONICAL REFERENCE FILES

### Production-Ready Assets

Location: `C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference\`

| Filename | Purpose | SHA256 Checksum |
|----------|---------|-----------------|
| `Curious Kelly in final pose in Chair...png` | **HERO SHOT** — Primary marketing | `49DDA5AE...` |
| `facing to the left.png` | Side angle, curious pose | `0E223F90...` |
| `head and shoulders without chair.png` | Close-up, headshot | `1067928C...` |
| `neutral face with hair.png` | **TRANSPARENT BG** — Compositing | `99061225...` |
| `close up of face.jpeg` | Detail reference | `561BABC6...` |
| `close up of kellys eyes.png` | Eye detail reference | `39E3BCBC...` |
| `profile of kelly.png` | Profile view | `459AB64E...` |
| `slightly turning her head.png` | 3/4 turn reference | `BEFB4A33...` |

### Video Reference

| File | Location | Purpose |
|------|----------|---------|
| `Avatar IV Video.mp4` | `Ref/` | Motion reference (facial animation style) |

### CC5 Project Files

| File | Location | Status |
|------|----------|--------|
| `Kelly_HS2_HD.ccProject` | `projects/Kelly/CC5/` | Headshot 2 processed |
| `Kelly_HS2_HD_SkinHair.ccProject` | `projects/Kelly/CC5/` | Skin/hair polished |
| `DirectorsChair_Template.iProject` | `projects/Kelly/iClone/` | Scene template |

---

## 5. DELIVERABLES REQUIRED

### A. Video Templates (Priority 1)

**Specifications:**
- Resolution: 3840×2160 (4K UHD)
- Frame Rate: 60 FPS (mandatory)
- Format: ProRes 422 HQ or PNG sequence
- Duration: As specified below
- Audio: None (lip-sync applied separately)
- Background: Director's chair scene

| Template ID | Motion Name | Duration | Loop Point | Description |
|-------------|-------------|----------|------------|-------------|
| T01 | `welcome_walk` | 10s | 8-10s | Kelly walks into frame, sits in chair, waves at camera |
| T02 | `present_explain` | 8s | 6-8s | Leaning forward, hands gesturing while explaining |
| T03 | `curious_examine` | 8s | 6-8s | Head tilt right (15°), examining invisible object with interest |
| T04 | `heartfelt_share` | 8s | 6-8s | Hand on heart, sincere expression, gentle nodding |
| T05 | `excited_discovery` | 8s | 6-8s | Eyes wide, hands raised, genuine excitement |
| T06 | `thoughtful_pause` | 8s | 6-8s | Chin rest on hand, contemplative gaze, slight smile |
| T07 | `celebrating` | 6s | 4-6s | Clapping hands, joyful expression, eyes crinkled |
| T08 | `listening` | 6s | 4-6s | Attentive nod, eye contact, encouraging smile |

**Animation Notes:**
- All templates must include subtle breathing animation
- Blinks: 10-15 per minute, natural variation
- Micro-saccades: 2-4 per second during speaking
- Hair physics: Active throughout
- Loop points must be seamless

### B. Expression Library (Priority 2)

**Specifications:**
- Resolution: 4096×4096 minimum
- Format: PNG with alpha channel
- Color Depth: 16-bit
- Pose: Front-facing, head-and-shoulders

| Expression ID | Name | Use Case | Key Features |
|---------------|------|----------|--------------|
| E01 | `neutral_ready` | Default state | Eyes forward, slight closed-mouth smile |
| E02 | `big_smile` | Welcome, celebration | Teeth showing, crow's feet, genuine joy |
| E03 | `curious_tilt` | Asking questions | Head 15° right, one brow raised |
| E04 | `teaching_engaged` | Explanations | Animated eyes, leaning slightly forward |
| E05 | `proud_encouraging` | Correct answers | Warm approval, soft smile, nodding |
| E06 | `gentle_redirect` | Wrong answers | Kind eyes, NOT disappointed, supportive |
| E07 | `wisdom_serene` | Final wisdom phase | Peaceful, profound, slight mysterious smile |

### C. Technical Package (Priority 3)

**FBX Export Requirements:**

| Attribute | Specification |
|-----------|---------------|
| Format | FBX 2020 or later |
| Textures | Embedded |
| Scale | 1 unit = 1 cm |
| Up Axis | Y-up |
| Skeleton | Standard CC5 rig |
| Blendshapes | 52 ARKit compatible |

**Blendshape Mapping (ARKit → CC5):**

```
eyeBlinkLeft → CC_Base_EyeBlinkLeft
eyeBlinkRight → CC_Base_EyeBlinkRight
eyeLookDownLeft → CC_Base_EyeLookDownLeft
eyeLookDownRight → CC_Base_EyeLookDownRight
eyeLookInLeft → CC_Base_EyeLookInLeft
eyeLookInRight → CC_Base_EyeLookInRight
eyeLookOutLeft → CC_Base_EyeLookOutLeft
eyeLookOutRight → CC_Base_EyeLookOutRight
eyeLookUpLeft → CC_Base_EyeLookUpLeft
eyeLookUpRight → CC_Base_EyeLookUpRight
eyeSquintLeft → CC_Base_EyeSquintLeft
eyeSquintRight → CC_Base_EyeSquintRight
eyeWideLeft → CC_Base_EyeWideLeft
eyeWideRight → CC_Base_EyeWideRight
jawForward → CC_Base_JawForward
jawLeft → CC_Base_JawLeft
jawRight → CC_Base_JawRight
jawOpen → CC_Base_JawOpen
mouthClose → CC_Base_MouthClose
mouthFunnel → CC_Base_MouthFunnel
mouthPucker → CC_Base_MouthPucker
mouthLeft → CC_Base_MouthLeft
mouthRight → CC_Base_MouthRight
mouthSmileLeft → CC_Base_MouthSmileLeft
mouthSmileRight → CC_Base_MouthSmileRight
mouthFrownLeft → CC_Base_MouthFrownLeft
mouthFrownRight → CC_Base_MouthFrownRight
mouthDimpleLeft → CC_Base_MouthDimpleLeft
mouthDimpleRight → CC_Base_MouthDimpleRight
mouthStretchLeft → CC_Base_MouthStretchLeft
mouthStretchRight → CC_Base_MouthStretchRight
mouthRollLower → CC_Base_MouthRollLower
mouthRollUpper → CC_Base_MouthRollUpper
mouthShrugLower → CC_Base_MouthShrugLower
mouthShrugUpper → CC_Base_MouthShrugUpper
mouthPressLeft → CC_Base_MouthPressLeft
mouthPressRight → CC_Base_MouthPressRight
mouthLowerDownLeft → CC_Base_MouthLowerDownLeft
mouthLowerDownRight → CC_Base_MouthLowerDownRight
mouthUpperUpLeft → CC_Base_MouthUpperUpLeft
mouthUpperUpRight → CC_Base_MouthUpperUpRight
browDownLeft → CC_Base_BrowDownLeft
browDownRight → CC_Base_BrowDownRight
browInnerUp → CC_Base_BrowInnerUp
browOuterUpLeft → CC_Base_BrowOuterUpLeft
browOuterUpRight → CC_Base_BrowOuterUpRight
cheekPuff → CC_Base_CheekPuff
cheekSquintLeft → CC_Base_CheekSquintLeft
cheekSquintRight → CC_Base_CheekSquintRight
noseSneerLeft → CC_Base_NoseSneerLeft
noseSneerRight → CC_Base_NoseSneerRight
tongueOut → CC_Base_TongueOut
```

---

## 6. PRODUCTION WORKFLOW

### Step-by-Step iClone Process

**Pre-Production Setup (Once):**

```
1. Launch iClone 8.62
2. Verify AccuLips plugin: Plugins → AccuLips → Check version
3. Open: projects/Kelly/CC5/Kelly_HS2_HD_SkinHair.ccProject
4. Send to iClone: File → Send Character to iClone
5. Load scene: DirectorsChair_Template.iProject
6. Set project FPS: Edit → Preferences → Project Settings → 60 FPS
```

**Per-Video Production (Batch):**

```
1. IMPORT AUDIO
   - Timeline panel → Right-click audio track
   - Import Audio → Select ElevenLabs .wav file
   - Verify waveform visible in timeline

2. RUN ACCULIPS
   - Select Kelly character in viewport
   - Animation → Facial Animation → AccuLips
   - Settings:
     • Audio: [Select imported track]
     • Language: English
     • Quality: High
     • Lip Strength: 1.0
     • Jaw Strength: 0.8
   - Click "Generate Text" → Review transcription
   - Click "Apply to Viseme Track"
   - Wait 2-5 minutes for processing

3. ADD EXPRESSION LAYER (Optional)
   - Plugins → AccuFACE → Video Mode
   - Load reference video (if available)
   - Enable brow/eye channels only
   - Record expression track

4. VERIFY ANIMATION
   - Press SPACEBAR to preview
   - Check: Lip-sync matches audio
   - Check: No phoneme drift
   - Check: Natural blinks present
   - Check: Hair physics active

5. RENDER
   - File → Export → Video
   - Settings:
     • Resolution: 3840×2160
     • Frame Rate: 60 FPS
     • Format: MP4 (H.264) or ProRes
     • Quality: High
     • Include Audio: Yes
   - Output path: renders/Kelly/day_{DAY}_{PHASE}_{ARCHETYPE}.mp4
   - Click Render → Wait for completion
```

### AccuLips Configuration

```yaml
# Optimal AccuLips settings for Kelly

General:
  language: "English (US)"
  quality: "High"
  processing_mode: "Accurate"

Viseme Mapping:
  lip_strength: 1.0
  jaw_strength: 0.8
  tongue_strength: 0.6
  
Timing:
  anticipation: 0.02  # seconds before phoneme
  hold: 0.04          # seconds at peak
  release: 0.03       # seconds after phoneme

Blending:
  smooth_transitions: true
  overlap_amount: 0.3
  
Advanced:
  use_coarticulation: true
  enable_emotion_blending: true
  respect_pauses: true
```

### Render Settings (Export Profile)

```yaml
# iClone Export Profile: Kelly_4K60_Production

Video:
  resolution: "3840x2160"
  frame_rate: 60
  codec: "H.264"
  profile: "High"
  bitrate: "50 Mbps"
  
Audio:
  include: true
  codec: "AAC"
  sample_rate: 48000
  bitrate: "320 kbps"

Quality:
  anti_aliasing: "8x"
  shadow_quality: "Ultra"
  motion_blur: false  # Keep crisp for educational content
  
Output:
  format: "MP4"
  naming: "{project}_{sequence}_{frame}.mp4"
```

---

## 7. QUALITY ASSURANCE

### Automated Checks (Run on Every Export)

| Check | Pass Criteria | Tool |
|-------|---------------|------|
| Face Identity | LoRA similarity > 0.85 | face_audit.py |
| Sweater Color | RGB within ±10 of #B0C4DE | color_check.py |
| Duration | Within ±0.5s of target | ffprobe |
| Resolution | Exactly 3840×2160 | ffprobe |
| Frame Rate | Exactly 60 FPS | ffprobe |
| File Size | > 50MB (no compression artifacts) | stat |
| Audio Sync | Offset < 20ms | sync_check.py |

### Manual Review Points

**Must Pass Human Review:**
- [ ] Kelly's face is recognizable as Kelly (not generic)
- [ ] Expression matches emotional intent
- [ ] Lip-sync feels natural (not robotic)
- [ ] Hair moves naturally with head motion
- [ ] Sweater is clearly powder blue
- [ ] No uncanny valley moments
- [ ] No texture glitches or clipping
- [ ] Lighting is consistent throughout

### Rejection Criteria (Automatic Fail)

```
REJECT IF:
- Face audit score < 0.75
- Sweater color not powder blue
- Any frame below 720p quality
- Lip-sync drift > 100ms
- Missing blendshapes in FBX
- Hair is static (no physics)
- Resolution below 3840×2160
- Frame rate below 60 FPS
```

---

## 8. FILE NAMING & ORGANIZATION

### Naming Convention

**Video Templates:**
```
kelly_T{XX}_{motion_name}_4k60.mp4
kelly_T01_welcome_walk_4k60.mp4
kelly_T02_present_explain_4k60.mp4
```

**Lesson Videos:**
```
kelly_{phase}_{day}_{archetype}.mp4
kelly_hook_001_explorer.mp4
kelly_q1_001_explorer.mp4
kelly_wisdom_001_explorer.mp4
```

**Expression Images:**
```
kelly_E{XX}_{expression_name}.png
kelly_E01_neutral_ready.png
kelly_E02_big_smile.png
```

### Folder Structure

```
renders/
├── templates/
│   ├── kelly_T01_welcome_walk_4k60.mp4
│   ├── kelly_T02_present_explain_4k60.mp4
│   └── ... (8 total)
├── expressions/
│   ├── kelly_E01_neutral_ready.png
│   ├── kelly_E02_big_smile.png
│   └── ... (7 total)
├── lessons/
│   ├── day_001/
│   │   ├── kelly_hook_001_explorer.mp4
│   │   ├── kelly_q1_001_explorer.mp4
│   │   └── ... (60 per day)
│   ├── day_002/
│   └── ... (365 days)
├── exports/
│   └── kelly_unity_arkit.fbx
└── archive/
    └── (previous versions)
```

---

## 9. TIMELINE & MILESTONES

### Critical Path to Launch

```
TODAY (Dec 7) ──────────────────────────────────────────────────────────
│
├── Dec 7-8: Artist reviews this handoff document
│            ✓ Questions resolved
│            ✓ Reference files verified
│
├── Dec 9-10: Template production begins
│             □ T01-T04 rendered (Priority 1)
│             □ Expression library E01-E04
│
├── Dec 11-12: Template production continues
│              □ T05-T08 rendered
│              □ Expression library E05-E07
│              □ FBX export with blendshapes
│
├── Dec 13-14: Days 1-7 lesson videos
│              □ AccuLips batch processing
│              □ Quality review
│              □ Fixes and re-renders
│
├── Dec 15: LAUNCH CONTENT COMPLETE
│           ✓ All Day 1-7 videos approved
│           ✓ CDN upload complete
│
└── Dec 17: LAUNCH DAY ★
            ✓ Platform live
            ✓ First users onboarding
```

### Post-Launch Production (Dec 18 - Mar 2026)

| Week | Days | Videos | Notes |
|------|------|--------|-------|
| Week 1 (Dec 18-24) | 8-14 | 420 | Christmas push |
| Week 2-4 (Dec 25 - Jan 14) | 15-45 | 1,860 | January backlog |
| Feb 2026 | 46-76 | 1,860 | Steady state |
| Mar 2026 | 77-107 | 1,860 | Q1 complete |
| ... | ... | ... | ... |
| Dec 2026 | 336-365 | 1,800 | Year 1 complete |

---

## 10. COMMUNICATION & HANDOFF

### Daily Check-ins

**When:** 10:00 AM PT daily (during active production)  
**Channel:** Slack #kelly-video-production  
**Format:**
```
DAILY STATUS - [DATE]
Completed: [list of renders]
In Progress: [current work]
Blocked: [any issues]
Quality Issues: [any rejections]
```

### Asset Delivery

**Upload Location:** Supabase Storage → `kelly-templates` bucket  
**Notification:** Post in Slack when uploads complete  
**Verification:** Run automated QA suite after upload

### Issue Escalation

| Issue Type | First Contact | Escalation |
|------------|---------------|------------|
| Technical (software) | #kelly-video-production | CTO |
| Creative (look/feel) | CVO | Brand Lead |
| Deadline at risk | CVO immediately | Executive team |
| Quality rejection | Re-render + #quality-review | CVO |

---

## 11. APPENDICES

### A. Reference File Checksums (SHA256)

```
49DDA5AEDE2EE327CD0B499DF4EC4F68F71A4AFE96AFF3E413098F4D433A45A6  Curious Kelly in final pose in Chair - UI elements will go on the side rails - Copy.png
0E223F90F6CB41EDD1F7692023C7CF44C5B21D5A4B49827E50DE2FC2EF540261  facing to the left.png
1067928C67B6A032C5680379CD03DF588B5228E4029332960E0CF38C8B5417B1  head and shoulders without chair.png
9906122502D05FF32255BEA81E30399CA491A60640789F09A967EB21F8102FC4  neutral face with hair.png
561BABC641D67C04BC5E7D43617C91FF86F14FD834B2E8599F9F8A416EBA08F2  close up of face.jpeg
39E3BCBC80775FC7F020A40D1427383347CCDF723665FD0EEDAC558A81BA0398  close up of kellys eyes.png
459AB64ED43D29F973945B24C7C57E969167B60CC0F41EB22220F0D7302E8F34  profile of kelly.png
BEFB4A333E2DB419FD5B20E648C6C7B77AA42FFDA6069F4AB03D8F419A4C34A9  slightly turning her head.png
```

### B. Color Palette (Exact Values)

| Element | Name | HEX | RGB | Notes |
|---------|------|-----|-----|-------|
| Sweater | Powder Blue | #B0C4DE | 176, 196, 222 | ⚠️ CRITICAL |
| Hair Base | Chestnut Brown | #5D4037 | 93, 64, 55 | |
| Hair Highlight | Caramel | #8D6E63 | 141, 110, 99 | |
| Eyes | Warm Brown | #5D4037 | 93, 64, 55 | |
| Lips | Natural Pink | #CC8899 | 204, 136, 153 | |
| Skin Base | Warm Fair | #FFDAB9 | 255, 218, 185 | Approximate |
| Skin Shadow | Warm Shadow | #D4A574 | 212, 165, 116 | Approximate |

### C. Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Character Creator 5 | Latest | Base model |
| iClone 8.62 | 8.62+ | Animation/Rendering |
| AccuLips | Latest | Lip-sync |
| AccuFACE | Latest | Expression capture |
| Headshot 2 | Pro | Photo-to-3D |
| Unity 2022.3 LTS | 2022.3+ | Real-time/WebGL |

### D. Hardware Specifications (Current Render Workstation)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 5090 (32GB VRAM) |
| Driver | 581.29 WHQL |
| CUDA | 13.0 |
| CPU | [Verify] |
| RAM | [Verify] |
| Storage | [Verify] |

---

## SIGN-OFF

### Artist Confirmation

```
□ I have received and reviewed this handoff document
□ I have access to all reference files
□ I understand the deliverables and timeline
□ I have verified the CC5/iClone project files load correctly
□ I have questions about: ________________________________
□ I am ready to begin production on: [DATE]

Artist Name: _______________________
Date: _______________________
Signature: _______________________
```

### CVO Approval

```
□ Handoff document is complete
□ All specifications are accurate
□ Timeline is confirmed
□ Artist questions have been answered
□ Production may begin

CVO Name: Chief Video Officer
Date: December 7, 2025
Status: READY FOR ARTIST REVIEW
```

---

## DOCUMENT HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 7, 2025 | CVO | Initial handoff document |

---

*This document is the single source of truth for Kelly video production.*  
*Any changes require CVO approval and version increment.*

**Contact:** hello@curiouskelly.com  
**Project:** Curious Kelly — The Daily Lesson  
**Company:** Lesson of the Day PBC

---

**🎬 Let's make Kelly the best digital human teacher on the planet.**

