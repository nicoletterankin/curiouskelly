# 🎬 HeyGen Motion Library Guide

## The Problem: 10-Second Motion Loops

When you upload a photo to HeyGen and create a talking photo avatar:
1. HeyGen applies a **~10 second base motion treatment** (Kling engine)
2. This treatment includes: head movements, blinks, micro-expressions
3. When generating videos **longer than 10 seconds**, this motion **LOOPS**
4. Result: **Uncanny valley** - viewers subconsciously detect the repetition

### Phase Durations vs. Loop Problem

| Phase | Duration | Loops at 10s? | Issue |
|-------|----------|---------------|-------|
| hook | 13s | Yes | Motion repeats at 10s mark |
| cliff | 11s | Yes | Slight loop |
| fact1 | 16s | Yes | **Obvious loop** |
| fact2 | 15s | Yes | **Obvious loop** |
| fact3 | 15s | Yes | **Obvious loop** |
| wisdom | 14s | Yes | Loop visible |
| outro | 9s | No | ✅ Under 10s, no loop |

---

## The Solution: Multi-Motion Scene Stitching

HeyGen's API supports **multiple `video_inputs`** - each is a separate scene with its own avatar.

**Key insight**: If you upload the SAME Kelly face photo but process it 3 times with different motion prompts, you get 3 different avatar_ids, each with unique motion patterns.

Then for a 16-second phase:
```
Scene 1 (0-8s):  avatar_motion_A
Scene 2 (8-16s): avatar_motion_B
```

The viewer sees ONE continuous Kelly, but the motion pattern changes naturally.

---

## Step 1: Create 3 Motion Variants Per Kelly

### In HeyGen UI:

1. **Upload the SAME Kelly head image 3 times**
2. For each upload, use a DIFFERENT motion prompt during the Kling treatment:

#### Motion Variant A: "Contemplative"
```
Slow thoughtful movements. Extended natural blinks. 
Minimal gesture. Centered calm presence. 
Slight head tilt indicating deep thought.
Serene listening posture.
```

#### Motion Variant B: "Engaged"
```
Active conversational energy. Natural eye tracking.
Subtle nods of acknowledgment. Interested lean.
Responsive facial micro-expressions.
Focused attentive gaze.
```

#### Motion Variant C: "Emphatic"
```
Punctuated delivery. Head nods on emphasis.
Eyebrow raise for key points. Forward lean.
Expressive hand gestures. Dynamic energy.
Conclusive confident posture.
```

3. **Save each as a separate talking photo**
4. **Record the 3 avatar_ids**

### Result:
```
Kelly Scientist:
├── scientist_motion_a: abc123 (contemplative)
├── scientist_motion_b: def456 (engaged)
└── scientist_motion_c: ghi789 (emphatic)
```

---

## Step 2: Configure Your .env

```env
# Kelly Motion Variants
KELLY_MOTION_A_ID=abc123
KELLY_MOTION_B_ID=def456
KELLY_MOTION_C_ID=ghi789
```

---

## Step 3: Test with the Multi-Motion Script

```powershell
# Test the multi-scene approach
npx tsx scripts/heygen-multi-motion-test.ts

# Or dry run first
npx tsx scripts/heygen-multi-motion-test.ts --dry-run
```

---

## Step 4: Production Implementation

For a 16-second `fact1` phase, the generation script should:

```typescript
// Split the 16-second script into two ~8-second parts
const [part1, part2] = splitScriptAtMidpoint(fact1Script);

const payload = {
  video_inputs: [
    {
      character: {
        type: 'talking_photo',
        talking_photo_id: KELLY_MOTION_A, // First 8 seconds
      },
      voice: {
        type: 'text',
        input_text: part1,
        voice_id: KELLY_VOICE_ID,
      },
    },
    {
      character: {
        type: 'talking_photo',
        talking_photo_id: KELLY_MOTION_B, // Second 8 seconds
      },
      voice: {
        type: 'text',
        input_text: part2,
        voice_id: KELLY_VOICE_ID,
      },
    },
  ],
  dimension: { width: 1280, height: 720 },
};
```

---

## Motion Library Schema

For full production, create this structure:

```
📁 kelly-motion-library/
├── manifest.json
└── 12 archetypes × 3 motions = 36 avatar_ids

manifest.json:
{
  "scientist": {
    "contemplative": "avatar_id_1",
    "engaged": "avatar_id_2", 
    "emphatic": "avatar_id_3"
  },
  "explorer": {
    "contemplative": "avatar_id_4",
    "engaged": "avatar_id_5",
    "emphatic": "avatar_id_6"
  },
  ...
}
```

---

## Phase-to-Motion Mapping

| Phase Type | Recommended Motion | Why |
|------------|-------------------|-----|
| hook | emphatic | Grab attention, energetic |
| cliff | engaged | Conversational, draws them in |
| fact1-3 | engaged → contemplative | Alternate to avoid patterns |
| wisdom | contemplative | Reflective, profound |
| outro | emphatic | Confident close |

### Smart Splitting for Long Phases

```typescript
function getMotionsForPhase(phase: string, duration: number): string[] {
  if (duration <= 10) {
    // No split needed
    return [MOTION_ENGAGED];
  } else if (duration <= 20) {
    // Split into 2 scenes
    return [MOTION_ENGAGED, MOTION_CONTEMPLATIVE];
  } else {
    // Split into 3 scenes (for 30s+ content)
    return [MOTION_EMPHATIC, MOTION_ENGAGED, MOTION_CONTEMPLATIVE];
  }
}
```

---

## Cost Analysis

### Without Motion Library:
- Each video loops = viewers notice = lower quality
- No reuse of motion investment

### With Motion Library:
- **One-time investment**: Create 36 motion variants (12 archetypes × 3 motions)
- **Credit cost**: ~36 × 10 seconds = 6 minutes = ~6 HeyGen credits
- **Reuse forever**: Every future lesson uses this library
- **Quality**: No visible loops, natural motion variety

---

## Next Steps

1. [ ] Upload 1 Kelly face 3 times with different motion prompts
2. [ ] Record the 3 avatar_ids
3. [ ] Run `heygen-multi-motion-test.ts` to verify scene stitching works
4. [ ] If successful, create full 36-variant motion library
5. [ ] Update generation scripts to use multi-scene approach

---

*Created: December 17, 2025*
*For: Curious Kelly Motion Library*
