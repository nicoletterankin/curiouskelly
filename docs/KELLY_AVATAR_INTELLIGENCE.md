# 🎭 Kelly Avatar Intelligence Platform

> **Created:** December 20, 2025  
> **Purpose:** Consolidated reference for all Kelly facial detection, animation, and avatar systems  
> **Goal:** Cross the uncanny valley - make Kelly feel ALIVE

---

## 🎯 THE VISION

Kelly communicates with:
- **Her eyes** - Direct gaze that FINDS you
- **Her smile** - Genuine warmth that builds trust  
- **Her lips** - Subtle direction, pointing with intention

When a student looks at Kelly, they should feel:
> "She sees ME. She's teaching ME. She cares about MY learning."

---

## 📊 Current System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KELLY GENERATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Script] → [ElevenLabs Audio] → [HeyGen/SadTalker] → [Video]   │
│                                                                  │
│  Talking Photos:   70 unique Kelly images in HeyGen              │
│  Archetypes:       13 Kelly personas (by head accessory)         │
│  Age variants:     6 (kid, teen, adult, mature, elder, super)    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧬 Kelly Archetypes (13)

| Archetype | Avatar ID | Head Accessory |
|-----------|-----------|----------------|
| Base | `433ad96bf5d647d9964cecf784d008f6` | Animated base |
| Neutral | `7bb18cddacd44333813cc90ffa44f766` | None |
| Survivor | `a2b31ed0b5f84b0fa02d15d411735d3a` | Olive bandana |
| Mystic | `45e5ef8b651846e0b62b7477e552e87b` | White beanie |
| Rebel | `aa8b5eb1d711468a9a6e2085a4f8469c` | Red headband |
| MacGyver | `06b78109ad22489ea2165ebbf180f77b` | Aviator goggles |
| Architect | `9ffd06bd986a4e3086612921f3ac87ea` | Thin glasses |
| Consultant | `e614671b193c40f99772f7de5d1c51f7` | Purple bindi |
| Empath | `b9032c922c6e4e35b58a98abd499d060` | Praying pose |
| **Scientist** | `3f44bd33bfd1494d916d2746808a1a39` | **Round glasses** ⭐ |
| Explorer | `d4eccf6a8d4c427b9313208d640db407` | Goggles |
| Strategist | `4227be1001a3431db2cb4c59f9c25287` | Sunglasses up |
| Provider | `d1d731dcdd5d4bb9af1c020a907671dc` | Dog tags |
| Storyteller | `4f28f8a7e7d44eab99f2cdd0d1530d5f` | Headphones |

⭐ **Scientist** is the default archetype - warm, curious, approachable.

---

## 👀 Eye Behavior System (Uncanny Valley Fix)

### Current Targets (from `Kelly_Uncanny_Blueprint.md`)

| Behavior | Target | Why It Matters |
|----------|--------|----------------|
| **Blink Rate** | 12-18/min | Too few = dead eyes, too many = nervous |
| **Blink Duration** | 120-200ms | Natural rest blink, occasional longer |
| **Saccades** | Every 1-3s | Small gaze shifts = ALIVE |
| **Head Coupling** | 1-2° nod | Eyes lead, head follows slightly |
| **Pupil Dilation** | +3-5% on excitement | Subtle emotional response |

### Implementation Files

- `digital-kelly/engines/Kelly_Engine_V2/onlykelly/Assets/KellyAvatarController.cs`
- `digital-kelly/engines/kelly_unity_player/AVATAR_UPGRADE_GUIDE.md`
- `daily-lesson-marketing/public/lesson-player/js/kelly-avatar-system.js`

### Expression Blendshapes

```csharp
// From KellyAvatarController.cs
{"curious",     { "A01_Brow_Inner_Up": 30f, "A05_Upper_Lid_Raiser": 20f }},
{"explaining",  { "A01_Brow_Inner_Up": 40f, "A06_Cheek_Raiser": 30f }},
{"listening",   { "A02_Brow_Outer_Up_L": 20f, "A02_Brow_Outer_Up_R": 20f }},
{"wisdom",      { "A12_Lip_Corner_Puller": 40f, "A06_Cheek_Raiser": 50f }},
{"celebrating", { "A12_Lip_Corner_Puller": 80f, "A06_Cheek_Raiser": 70f, "A01_Brow_Inner_Up": 50f }}
```

---

## 👄 Lip-Sync Pipeline

### HeyGen (Primary)
- API: `https://api.heygen.com/v2/video/generate`
- Uses `talking_photo_id` + audio URL
- Motion: Kling AI animations

### SadTalker (Alternative)
- API: `fal-ai/sadtalker` via Fal.ai
- Parameters:
  - `still: true` - Subtle motion
  - `enhancer: 'gfpgan'` - Face restoration
  - `preprocess: 'full'` - Full face detection

### Viseme Mapping (15 core shapes)

```
sil → V_Explosive (silence)
PP  → V_Explosive (B/P/M sounds)
FF  → V_Dental_Lip (F/V sounds)
TH  → V_Tight_O (TH sounds)
aa  → V_Wide (A sounds)
oh  → V_Tight_O (O sounds)
```

---

## 🔧 IMPROVEMENT OPPORTUNITIES

### 1. **Eye Gaze Alignment**
**Problem:** Kelly's eyes don't consistently find the camera/learner  
**Solution:** Pre-process talking photos with gaze correction before upload

```javascript
// Potential implementation in avatar generation
const gazeConfig = {
  targetX: 0.5,  // Center of frame
  targetY: 0.45, // Slightly above center (engaging)
  variance: 0.02 // Micro-saccade range
};
```

### 2. **Smile Warmth Optimization**
**Problem:** Some archetypes feel cold or forced  
**Solution:** Audit talking photo selection for natural "Duchenne smile" (eyes + mouth)

**Best archetypes for warmth:** Scientist, Empath, Consultant  
**Need improvement:** Rebel, Survivor, MacGyver

### 3. **Lip Direction ("Pointing with lips")**
**Problem:** Kelly's lips don't guide attention  
**Solution:** Add subtle asymmetric lip movements during emphasis

```
When saying "THIS is important" → Slight pursed/pointed lip
When asking questions → Lip corners up (inviting response)
When transitioning → Neutral, slight smile
```

### 4. **Phase-Matched Expressions**
Map expressions to lesson phases:

| Phase | Expression | Why |
|-------|------------|-----|
| Hook | `curious` + high energy | Capture attention |
| Cliff | `listening` + slight concern | Build tension |
| Facts | `explaining` + confident | Deliver knowledge |
| Wisdom | `wisdom` + warm smile | Connect emotionally |
| Outro | `celebrating` + big smile | Reward completion |

---

## 📁 Key Files Reference

### HeyGen Pipeline
```
scripts/heygen-kelly-production.ts    # Main production pipeline
scripts/heygen-map-kelly-avatars.ts   # Avatar ID mapping
scripts/heygen-phase-generator.ts     # Phase-specific generation
generated-images/kelly-talking-photos.json  # All 70 avatar IDs
```

### SadTalker Integration
```
scripts/test-sadtalker-kelly.ts       # Test script
scripts/generate-sadtalker.ts         # Production generation
scripts/emergency-day354-sadtalker.ts # Fallback generation
```

### Avatar Controllers
```
digital-kelly/engines/Kelly_Engine_V2/onlykelly/Assets/KellyAvatarController.cs
digital-kelly/engines/kelly_unity_player/AVATAR_UPGRADE_GUIDE.md
daily-lesson-marketing/public/lesson-player/js/kelly-avatar-system.js
```

### Uncanny Valley Documentation
```
assets/Ref/Kelly_Uncanny_Blueprint.md
docs/guides/Kelly_Uncanny_Blueprint.md
docs/KELLY_AUTONOMY_BLUEPRINT.md
```

---

## 🎯 NEXT STEPS (Priority Order)

1. **Audit top 5 archetypes** - Which ones connect best? (Run A/B test)
2. **Implement gaze correction** - Pre-process photos before HeyGen upload
3. **Create phase expression map** - Match Kelly's face to lesson flow
4. **Add micro-expression layer** - Subtle breathing, blinks during video
5. **Test SadTalker vs HeyGen** - Quality comparison for cost reduction

---

## 🏗️ Architecture Goal: Kelly Autonomy

From `KELLY_AUTONOMY_BLUEPRINT.md`:

> **Kelly is ONE fixed character.** We don't need a general-purpose avatar system.

**Target:**
| Current | Goal |
|---------|------|
| HeyGen ~$0.05/min, 15-60 min queue | $0, < 1 second |
| Sync Labs ~$0.08/min, 2-10 min queue | Real-time |

**Path:**
1. Pre-compute viseme sprites for all 15 mouth shapes
2. Build real-time blending engine (WebGL/Unity)
3. Store Kelly's face landmarks once, reuse forever
4. Real-time lip-sync with local audio analysis

---

*"Kelly's eyes should feel like they're finding you in a crowded room. Her smile should feel like she's genuinely happy you showed up. Her teaching should feel like she prepared this lesson just for YOU."*
