# The Golden Three Archetypes

## Launch Configuration (v1.0)

Curious Kelly uses **3 maximally distinct archetypes** to personalize every lesson:

| Archetype | Energy | Voice Pattern | Best For |
|-----------|--------|---------------|----------|
| 🧭 **The Explorer** | Wonder & Adventure | "Let's discover... uncharted... expedition..." | Curious learners who love the journey |
| 🔬 **The Scientist** | Evidence & Proof | "Research shows... data confirms... 62%..." | Skeptics who need proof before belief |
| ⚡ **The Rebel** | Edge & Challenge | "They don't want you to know... the system..." | Disengaged/cynical learners, teens |

## Tone → Archetype Mapping

Users select a **tone** in settings. This maps to an archetype:

| Tone Setting | Archetype | Why |
|--------------|-----------|-----|
| 🤔 Curious | The Scientist | Analytical minds want data |
| 😊 Friendly | The Explorer | Warm discovery experience |
| 🎮 Playful | The Explorer | Adventure and fun |
| 🦉 Wise | The Rebel | Deep challenges to assumptions |
| 💪 Coach | The Rebel | Push through resistance |
| 📚 Scholar | The Scientist | Academic precision |

## Content Structure

Each lesson has **15 atoms** (3 archetypes × 5 phases):

```
Day N
├── The Explorer
│   ├── Hook      (adventure invitation)
│   ├── Fact1     (discovery framing)
│   ├── Fact2     (journey metaphor)
│   ├── Fact3     (expedition steps)
│   └── Wisdom    (trail reflection)
├── The Scientist
│   ├── Hook      (hypothesis)
│   ├── Fact1     (research data)
│   ├── Fact2     (neurological mechanism)
│   ├── Fact3     (implementation protocol)
│   └── Wisdom    (evidence summary)
└── The Rebel
    ├── Hook      (challenge status quo)
    ├── Fact1     (expose hidden truth)
    ├── Fact2     (radical reframe)
    ├── Fact3     (subversive strategy)
    └── Wisdom    (liberation call)
```

## Enhanced Atom Schema

Every atom includes Kelly's interaction system:

```json
{
  "script": "Kelly's teaching content",
  "kellyPose": "explaining|hello|thinking",
  "kellyEmotion": "curious|excited|thoughtful|proud|caring",
  "optionIntro": "Kelly's question before options",
  "optionPose": "thinking",
  "hintSystem": {
    "enabled": true,
    "delayMs": 2500,
    "hintType": "gaze",
    "intensity": "subtle",
    "bestOption": "B"
  },
  "options": [
    {
      "letter": "A",
      "text": "Option text",
      "quality": "redirect|good|best",
      "hintCue": null|"gaze-right"|"gaze-left",
      "response": "Kelly's response",
      "responseEmotion": "encouraging|celebrating|excited",
      "responsePose": "encouraging|celebrating|explaining"
    }
  ]
}
```

## Scalability

This template scales to all 365 lessons:

| Topic Example | Explorer | Scientist | Rebel |
|---------------|----------|-----------|-------|
| How Sound Moves | "Sound is a tireless traveler..." | "Sound waves propagate at 343 m/s..." | "They teach sound wrong in school..." |
| Where Lakes Come From | "Imagine discovering a hidden lake..." | "Glacial lake formation requires..." | "Forget what geography class said..." |
| Why Leaves Change Color | "Venture into autumn's mystery..." | "Chlorophyll breakdown reveals..." | "The real reason trees change..." |

## Why These Three?

1. **Maximum Distinction**: Adventure ≠ Data ≠ Challenge
2. **Universal Appeal**: Catches curious + skeptical + disengaged
3. **Scalable Voice**: Any topic fits these three frames
4. **Teen-Friendly**: Rebel voice engages hardest-to-reach learners
5. **Evidence-Based**: Scientist grounds everything in research
6. **Wonder-Preserving**: Explorer maintains the joy of learning

## Future Expansion

Post-launch, we may add:
- **The Storyteller** (narrative learners)
- **The Empath** (emotional connection)
- **The Mystic** (meaning-seekers)

But for v1.0, the Golden Three provide maximum impact with minimum complexity.

---

*Created: December 2024*
*Status: PRODUCTION READY*

