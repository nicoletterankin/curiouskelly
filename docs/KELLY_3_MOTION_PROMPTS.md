# 🎬 Kelly's 3 Base Motion Prompts

## The Problem We're Solving

When HeyGen/Kling creates the 10-second base motion treatment:
- At ~9-10 seconds, there's a **"loop seam"** - a subtle head shake designed to ease back to the start
- This is unnatural - no one shakes their head every 10 seconds mid-sentence
- Viewers subconsciously detect this = **uncanny valley**

## The Solution

1. Create **3 different motion treatments** of the same Kelly face
2. Cut scenes at **~8 seconds** (before the head shake)
3. Transition to a **different motion treatment** for the next 8 seconds
4. Result: No loops, no head shakes, natural variety

---

## THE 3 MOTION PROMPTS

These are designed to create **genuinely different movement patterns** while all feeling like the same Kelly. Each should be used when uploading the same Kelly head image to HeyGen.

### Motion A: "Centered Stillness" (Minimal Movement)

```
Subject maintains centered, grounded presence with minimal head movement.
Soft natural blinks every 3-4 seconds.
Eyes stay focused on viewer with warm connection.
Slight micro-movements in eyebrows during emphasis.
No head tilts. No nods. Almost meditative stillness.
Breathing visible but subtle.
Camera static. Background neutral.
```

**Use for:** Wisdom phases, reflective moments, profound statements
**Character:** Calm authority, deep presence

---

### Motion B: "Active Listening" (Moderate Movement)

```
Subject shows engaged, conversational energy with natural head movement.
Occasional small nods of acknowledgment (1-2 per 8 seconds).
Eyes track slightly as if following their own thoughts.
Natural blink pattern, slightly faster than Motion A.
One subtle eyebrow raise during a key point.
Slight forward lean suggesting interest.
Camera static. Background neutral.
```

**Use for:** Fact phases, explaining concepts, educational content
**Character:** Engaged teacher, helpful guide

---

### Motion C: "Expressive Delivery" (Dynamic Movement)

```
Subject speaks with visible enthusiasm and dynamic expression.
Head moves with natural speech emphasis.
Eyes widen slightly for important revelations.
More frequent eyebrow movement showing emotional engagement.
Occasional subtle smile that reaches the eyes.
Energy builds toward the end of the phrase.
Camera static. Background neutral.
```

**Use for:** Hook phases, exciting facts, call-to-action moments
**Character:** Enthusiastic presenter, inspiring speaker

---

## CRITICAL NOTES FOR HEYGEN UPLOAD

When creating each motion variant:

1. **Use the EXACT same Kelly head image** for all 3
2. **Apply each motion prompt during the Kling upscale step**
3. **Let it process the full 10 seconds** (we'll cut at 8 in post)
4. **Save each with a clear naming convention**:
   - `kelly_scientist_motionA` (Centered)
   - `kelly_scientist_motionB` (Listening)
   - `kelly_scientist_motionC` (Expressive)

---

## SCENE CUTTING STRATEGY

To avoid the loop seam (head shake at ~9-10s):

| Phase Duration | Strategy |
|----------------|----------|
| ≤8 seconds | Single scene, Motion B (engaged) |
| 9-16 seconds | 2 scenes: 8s + remainder |
| 17-24 seconds | 3 scenes: 8s + 8s + remainder |

### Example: 15-second fact1 phase

```
0s ────── 8s ────── 15s
[Motion B  ][Motion A   ]
 "When you   "...your motor
 imagine..."  cortex lights up"
```

Split the script at a natural breath point around the 8-second mark.

---

## MOTION ROTATION PATTERNS

To maximize variety and avoid predictability:

### Pattern 1: Contemplative Arc (for wisdom-heavy lessons)
```
Hook:   C (Expressive) → draw them in
Cliff:  B (Listening) → build curiosity  
Fact1:  B → A (Listening → Centered)
Fact2:  A → B (Centered → Listening)
Fact3:  B → A (Listening → Centered)
Wisdom: A (Centered) → profound delivery
Outro:  C (Expressive) → energetic close
```

### Pattern 2: Dynamic Arc (for exciting/discovery lessons)
```
Hook:   C (Expressive)
Cliff:  C → B (Expressive → Listening)
Fact1:  B → C (Listening → Expressive)
Fact2:  C → B (Expressive → Listening)
Fact3:  B → C (Listening → Expressive)
Wisdom: B → A (Listening → Centered)
Outro:  C (Expressive)
```

---

## TESTING CHECKLIST

Before full production, verify:

- [ ] All 3 motion variants uploaded for test archetype
- [ ] Motions A, B, C feel genuinely different
- [ ] 8-second cuts work (no visible head shake)
- [ ] Scene transitions at script break points feel natural
- [ ] Audio flows continuously across scene boundaries
- [ ] Final video plays as one seamless piece

---

## FULL LIBRARY SCOPE

For complete production:

| Archetypes | × | Motions | = | Total Avatar IDs |
|------------|---|---------|---|------------------|
| 12 | × | 3 | = | 36 |

Each archetype (Scientist, Explorer, Rebel, etc.) gets 3 motion treatments.

**One-time investment. Reuse for every lesson forever.**

---

*Created: December 17, 2025*
*Purpose: Eliminate uncanny valley from Kelly videos*
