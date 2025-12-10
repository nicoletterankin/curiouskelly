# HeyGen: 12 Kelly Archetypes Plan

## 🎯 THE VISION

12 distinct Kelly Photo Avatars in HeyGen — one per archetype.
When learners switch archetypes, Kelly visually transforms.

---

## 📸 THE 12 KELLY PHOTOS

Each photo uses the SAME base Kelly face but with DIFFERENT:
- Expression (eyes, mouth, eyebrows)
- Head position (tilt, angle)
- Energy/vibe
- Possibly outfit accent

### Base Reference
`C:\iLearnStudio\projects\Kelly\Ref\Best Character Reference\head and shoulders without chair.png`

---

## 🎨 IMAGE GENERATION PROMPTS

Use these prompts with Flux + Kelly LoRA (or similar) to generate consistent variations:

### 1. The Scientist 🔬
```
Kelly, young woman, brown hair, blue sweater, white background, 
EXPRESSION: focused, analytical, one eyebrow slightly raised, 
slight knowing smile, direct eye contact, 
HEAD: straight on, chin slightly up,
ENERGY: confident, evidence-based, "I have the data"
```

### 2. The Explorer 🧭
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: wide eyes with wonder, excited open smile showing teeth,
eyebrows raised in curiosity,
HEAD: tilted slightly right, looking slightly up as if seeing something amazing,
ENERGY: adventurous, "let's discover this together!"
```

### 3. The Rebel ⚡
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: confident smirk, one corner of mouth raised, 
intense direct eye contact, eyebrows slightly furrowed,
HEAD: chin down slightly, looking up through eyebrows,
ENERGY: challenging, edgy, "question everything"
```

### 4. The Architect 🏛️
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: thoughtful, lips pressed together slightly, 
eyes showing concentration, calm confidence,
HEAD: straight, very centered and balanced,
ENERGY: structured, systematic, "let me show you the blueprint"
```

### 5. The Diplomat 🤝
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: warm genuine smile, soft eyes, approachable,
slightly nodding, open expression,
HEAD: tilted slightly, welcoming angle,
ENERGY: balanced, understanding, "I see all perspectives"
```

### 6. The Empath 💗
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: gentle smile, compassionate eyes, 
soft gaze, slightly parted lips as if about to share something caring,
HEAD: tilted with warmth, leaning in slightly,
ENERGY: nurturing, connected, "I feel what you feel"
```

### 7. The MacGyver 🔧
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: practical grin, eyes bright with "I've got an idea",
asymmetrical smile, engaged and ready,
HEAD: tilted forward slightly, action-ready,
ENERGY: hands-on, resourceful, "here's how we can use this"
```

### 8. The Mystic ✨
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: serene, knowing smile, eyes with depth and mystery,
peaceful but profound gaze,
HEAD: slight upward tilt, contemplative angle,
ENERGY: philosophical, transcendent, "there's something deeper here"
```

### 9. The Provider 🛡️
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: warm protective smile, reassuring eyes,
confident but gentle, motherly strength,
HEAD: straight on, grounded, stable,
ENERGY: nurturing, protective, "I'll keep you safe"
```

### 10. The Storyteller 📖
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: animated, expressive eyes, about to share a secret,
dramatic smile, theatrical sparkle,
HEAD: dynamic angle, mid-gesture feeling,
ENERGY: narrative, captivating, "let me tell you a story"
```

### 11. The Strategist 🎯
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: sharp focus, confident knowing look,
slight smile of someone who's figured it out,
HEAD: chin slightly up, commanding angle,
ENERGY: tactical, winning, "here's the smart move"
```

### 12. The Survivor 🏕️
```
Kelly, young woman, brown hair, blue sweater, white background,
EXPRESSION: serious, determined, no-nonsense,
eyes showing resilience and grit,
HEAD: straight on, solid, grounded,
ENERGY: practical, tough, "when things get hard, you'll need this"
```

---

## 🎬 HEYGEN MOTION EXPECTATIONS

HeyGen generates motion from audio. The PHOTO sets the baseline expression.
The SCRIPT content drives the motion intensity.

| Archetype | Base Expression | Motion Style from Audio |
|-----------|-----------------|------------------------|
| Scientist | Focused, analytical | Measured, precise gestures |
| Explorer | Wide-eyed wonder | Animated, excited |
| Rebel | Challenging smirk | Intense, direct |
| Architect | Thoughtful, centered | Methodical, structured |
| Diplomat | Warm, open | Balanced, inclusive |
| Empath | Gentle, soft | Warm, caring |
| MacGyver | Practical grin | Energetic, hands-ready |
| Mystic | Serene, knowing | Calm, profound |
| Provider | Protective warmth | Reassuring, steady |
| Storyteller | Theatrical, expressive | Dramatic, animated |
| Strategist | Sharp, confident | Decisive, commanding |
| Survivor | Determined, serious | Direct, no-nonsense |

---

## 📁 FILE NAMING CONVENTION

```
kelly_archetype_scientist.png
kelly_archetype_explorer.png
kelly_archetype_rebel.png
kelly_archetype_architect.png
kelly_archetype_diplomat.png
kelly_archetype_empath.png
kelly_archetype_macgyver.png
kelly_archetype_mystic.png
kelly_archetype_provider.png
kelly_archetype_storyteller.png
kelly_archetype_strategist.png
kelly_archetype_survivor.png
```

---

## 🚀 WORKFLOW

### Step 1: Generate 12 Kelly Photos
Use Flux + Kelly LoRA with prompts above.
Ensure CONSISTENCY: same face, same sweater, same background.
Only EXPRESSION and HEAD POSITION change.

### Step 2: Upload to HeyGen
For each photo:
1. app.heygen.com → Avatars → Create new
2. Upload the archetype photo
3. Wait for processing
4. Name it clearly: "Kelly - Scientist", "Kelly - Explorer", etc.
5. Copy Avatar ID

### Step 3: Map Avatar IDs
```
const KELLY_ARCHETYPE_AVATARS = {
  "The Scientist": "avatar_id_here",
  "The Explorer": "avatar_id_here",
  "The Rebel": "avatar_id_here",
  // ... etc
};
```

### Step 4: Generate Videos
For each lesson atom:
1. Get the archetype
2. Look up the correct Kelly avatar ID
3. Generate with ElevenLabs audio
4. Upload to Supabase

---

## 📊 SCALE PLANNING

### Phase 1: 12 Archetypes (Launch)
- 12 Photo Avatars
- Visual differentiation by personality

### Phase 2: Age Variants (Post-Launch)
- 12 archetypes × 6 age buckets = 72 avatars
- Same expressions, age-appropriate styling

### Phase 3: Full Matrix (Future)
- 12 × 6 × 3 = 216 avatars
- Complete personalization

---

## ⏱️ TIME ESTIMATE

| Task | Time |
|------|------|
| Generate 12 photos | 1-2 hours |
| Upload to HeyGen | 30 min |
| Wait for processing | 1-2 hours |
| Map avatar IDs | 15 min |
| Test generation | 30 min |
| **Total** | **3-5 hours** |

---

*Created: December 10, 2025*
*Focus: HeyGen Photo Avatars*
*Goal: 12 visually distinct Kellys*

