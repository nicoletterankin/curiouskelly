# Kelly Interaction Template: The Dance Between Learner and Kelly

## Executive Summary

This document defines the **complete artifact structure** needed for every lesson phase so that Kelly can:
1. **Deliver** her teaching script
2. **Present** options with an intro question
3. **Hint** visually toward the best choice (subtly!)
4. **React** emotionally to each choice
5. **Respond** with tailored feedback
6. **Guide** the learner to the next phase

---

## Current Structure (INCOMPLETE)

```json
{
  "script": "Kelly's main teaching content",
  "options": ["Option A text", "Option B text", "Option C text"],
  "responses": {
    "Option A": "Kelly's response when A is chosen",
    "Option B": "Kelly's response when B is chosen",
    "Option C": "Kelly's response when C is chosen"
  }
}
```

### What's Missing:
- ❌ No intro when presenting options ("What do you think?")
- ❌ No quality markers (which answer is "best"?)
- ❌ No hint direction (where should Kelly look?)
- ❌ No emotional reactions (celebrating vs encouraging)
- ❌ No pose/expression mapping per phase
- ❌ No visual cue for "best" option

---

## Enhanced Structure (COMPLETE KELLY DANCE)

```json
{
  // ═══════════════════════════════════════════════════════════════
  // PHASE INTRO: Kelly's Teaching Script
  // ═══════════════════════════════════════════════════════════════
  "script": "Kelly's main teaching content for this phase",
  "kellyPose": "explaining",           // welcome|explaining|thinking|celebrating
  "kellyEmotion": "curious",           // curious|excited|thoughtful|proud

  // ═══════════════════════════════════════════════════════════════
  // OPTIONS PRESENTATION
  // ═══════════════════════════════════════════════════════════════
  "optionIntro": "Which of these interests you most?",
  "optionPose": "thinking",            // Kelly's pose when showing options
  
  "options": [
    {
      "letter": "A",
      "text": "The learner's first choice text",
      "quality": "good",               // best|good|redirect
      "hintCue": "glance-left",        // subtle visual hint direction
      "response": "Kelly's feedback for this choice",
      "responseEmotion": "encouraging", // celebrating|encouraging|redirecting
      "responsePose": "encouraging"
    },
    {
      "letter": "B", 
      "text": "The learner's second choice text",
      "quality": "best",               // ← This is the "most curious" option
      "hintCue": "glance-right",       // Kelly subtly looks toward this
      "response": "Kelly's feedback for this choice",
      "responseEmotion": "celebrating",
      "responsePose": "celebrating"
    },
    {
      "letter": "C",
      "text": "The learner's third choice text", 
      "quality": "redirect",           // Valid but redirects to better path
      "hintCue": null,                 // No hint for redirect options
      "response": "Kelly's feedback, gently steering back",
      "responseEmotion": "thoughtful",
      "responsePose": "thinking"
    }
  ],

  // ═══════════════════════════════════════════════════════════════
  // VISUAL CUES FOR KELLY'S SUBTLE HINTS
  // ═══════════════════════════════════════════════════════════════
  "hintSystem": {
    "enabled": true,
    "intensity": "subtle",             // subtle|medium|obvious
    "bestOption": "B",                 // Which option Kelly hints at
    "hintType": "gaze",                // gaze|point|lean|eyebrow
    "delayMs": 3000                    // Wait 3s before subtle hint
  }
}
```

---

## Kelly's Available Visual States

### Poses (from kelly-production-assets.js)
| State | Image | Use Case |
|-------|-------|----------|
| `hello` | kelly_welcome.png | Welcome, greeting |
| `explaining` | kelly_idle.png | Teaching, presenting facts |
| `thinking` | kelly_hint.png | Considering, before options |
| `pointing-left` | kelly_choice_left.png | Highlighting Option A |
| `pointing-right` | kelly_choice_right.png | Highlighting Option B |
| `encouraging` | kelly_clasp.png | Supportive feedback |
| `celebrating` | kelly_welcome.png | Correct/great answer |
| `listening` | kelly_listening.png | Voice conversation |

### Hint Cues (PROPOSED - NOT YET IMPLEMENTED)
| Cue | Description | Implementation |
|-----|-------------|----------------|
| `gaze-left` | Kelly looks toward Option A | Eye direction in image |
| `gaze-right` | Kelly looks toward Option B | Eye direction in image |
| `lean-left` | Kelly leans slightly left | Body angle |
| `lean-right` | Kelly leans slightly right | Body angle |
| `eyebrow-raise` | Subtle eyebrow for "best" | Expression overlay |
| `smile-hint` | Smile increases near best option | Animation |

---

## Complete Day 1 Example: "Starting Fresh"

### Phase: Hook (The Explorer Archetype)

```json
{
  "phase": "Hook",
  "archetype": "The Explorer",
  
  "script": "Every single day, you wake up with a chance to start over. Fresh starts are not just for January 1st—your brain is wired to embrace new beginnings. Ready to discover why?",
  "kellyPose": "hello",
  "kellyEmotion": "excited",
  
  "optionIntro": "What draws you in most?",
  "optionPose": "thinking",
  
  "options": [
    {
      "letter": "A",
      "text": "I don't believe in fresh starts.",
      "quality": "redirect",
      "hintCue": null,
      "response": "That skepticism is healthy! But research shows our brains actually create mental chapters that help us change. Let us explore the science.",
      "responseEmotion": "encouraging",
      "responsePose": "encouraging"
    },
    {
      "letter": "B",
      "text": "Why does my brain like new beginnings?",
      "quality": "best",
      "hintCue": "gaze-right",
      "response": "Great question! Scientists call it the fresh start effect—our brains use dates like chapters in a book, separating old you from new you.",
      "responseEmotion": "celebrating",
      "responsePose": "celebrating"
    },
    {
      "letter": "C",
      "text": "Can I have a fresh start right NOW?",
      "quality": "good",
      "hintCue": "glance-left",
      "response": "Absolutely! Every moment can be a fresh start. Right now, as you hear this, you are already beginning something new.",
      "responseEmotion": "excited",
      "responsePose": "explaining"
    }
  ],
  
  "hintSystem": {
    "enabled": true,
    "intensity": "subtle",
    "bestOption": "B",
    "hintType": "gaze",
    "delayMs": 2500
  }
}
```

---

## The Kelly Dance Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. PHASE INTRO                                                 │
│     Kelly: [explaining pose] + [curious emotion]                │
│     Speaks: "script" content                                    │
│     Duration: Until audio completes                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. OPTIONS PRESENTATION                                        │
│     Kelly: [thinking pose]                                      │
│     Speaks: "optionIntro" ("What draws you in most?")           │
│     Shows: Three option cards (A, B, C)                         │
│     Wait: hintSystem.delayMs (2.5s)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. SUBTLE HINT (after delay)                                   │
│     Kelly: [gaze toward bestOption]                             │
│     Type: hintSystem.hintType                                   │
│     Intensity: hintSystem.intensity                             │
│     (User may not consciously notice, but feels guided)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. USER HOVERS/CONSIDERS                                       │
│     Kelly: [pointing-left] or [pointing-right]                  │
│     Based on which option user is hovering                      │
│     Creates "dance" - Kelly follows user's exploration          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. USER SELECTS                                                │
│     Kelly: [listening pose] briefly                             │
│     Transition to response...                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. KELLY RESPONDS                                              │
│     Kelly: [responsePose] + [responseEmotion]                   │
│     Speaks: option.response                                     │
│     If quality="best": celebrating, sparkles                    │
│     If quality="good": encouraging, warmth                      │
│     If quality="redirect": thoughtful, gentle steering          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. AUTO-ADVANCE                                                │
│     After response audio completes                              │
│     Move to next phase                                          │
│     Repeat the dance...                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Option Quality Philosophy

### "best" - The Curious Path
- The option that leads to deepest learning
- Shows genuine curiosity/engagement
- Kelly celebrates this choice
- Gets the most enthusiasm in response

### "good" - Valid Engagement  
- Legitimate interest, different angle
- Still moves learning forward
- Kelly encourages and validates
- Response connects back to main thread

### "redirect" - Skepticism/Resistance
- Common objection or doubt
- Kelly acknowledges validity
- Gently provides evidence/reframe
- Invites back to curiosity

**IMPORTANT**: No option is "wrong"! Every choice leads to learning. "redirect" options often produce the most thoughtful responses.

---

## Implementation Requirements

### Database Changes
1. Migrate `lesson_atoms.content` to enhanced structure
2. Ensure backward compatibility (check for old vs new format)
3. Add `hintSystem` field support

### Frontend Changes (learn.html)
1. Parse new `optionIntro` field
2. Implement hint delay timer
3. Add hint visual system (gaze direction)
4. Map `responseEmotion` to Kelly poses
5. Enhance option hover behavior

### New Kelly Assets Needed
1. `kelly_gaze_left.png` - Looking toward Option A
2. `kelly_gaze_right.png` - Looking toward Option B  
3. `kelly_eyebrow_raise.png` - Subtle hint expression
4. Or: Use eye tracking overlay on existing poses

---

## Migration Plan

### Phase 1: Schema Enhancement
- Add new fields to content JSONB
- Keep old format working

### Phase 2: Day 1 Reference Implementation
- Fully implement Day 1 with all new fields
- Test complete dance flow
- Verify visual hints work

### Phase 3: Batch Content Generation
- Use AI to enhance existing atoms with:
  - `optionIntro` generation
  - `quality` classification
  - `responseEmotion` mapping
- Human review for quality

### Phase 4: Visual Asset Creation
- Generate hint pose variations
- Create gaze direction overlays
- Test subtle vs obvious hints

---

## Success Metrics

- **Engagement**: Do users explore options before selecting?
- **Hint Effectiveness**: Do users select "best" more often?
- **Delight**: Do users feel guided, not manipulated?
- **Completion**: Do users finish more lessons?

---

## Key Principle: THE DANCE

Kelly should feel like a **dance partner**, not a quiz master:
- She follows when the learner leads
- She gently guides when the learner hesitates  
- She celebrates exploration, not just "correct" answers
- Every interaction feels natural, not mechanical

The goal is that learners **enjoy** choosing, not stress about "getting it right."



