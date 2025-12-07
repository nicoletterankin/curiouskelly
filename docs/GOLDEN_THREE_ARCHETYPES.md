# The Twelve Archetypes

## Production Configuration (v1.1)

> **Updated:** 2025-12-08 by CAO  
> **Change:** Documented actual 12 archetypes in production database

Curious Kelly uses **12 distinct archetypes** to personalize every lesson. While the original spec called for 3 ("Golden Three"), content was generated for all 12, providing richer personalization options.

---

## The Primary Three (Original "Golden Three")

These have the most content (1,825 atoms each) and are recommended defaults:

| Archetype | Energy | Voice Pattern | Best For |
|-----------|--------|---------------|----------|
| 🧭 **The Explorer** | Wonder & Adventure | "Let's discover... uncharted... expedition..." | Curious learners who love the journey |
| 🔬 **The Scientist** | Evidence & Proof | "Research shows... data confirms... 62%..." | Skeptics who need proof before belief |
| ⚡ **The Rebel** | Edge & Challenge | "They don't want you to know... the system..." | Disengaged/cynical learners, teens |

---

## The Extended Nine (1,820 atoms each)

| Archetype | Energy | Voice Pattern | Best For |
|-----------|--------|---------------|----------|
| 🏛️ **The Architect** | Structure & Design | "The blueprint shows... foundation... framework..." | Systematic thinkers, builders |
| 🤝 **The Diplomat** | Connection & Harmony | "Let's consider all perspectives... shared understanding..." | Collaborative learners, mediators |
| 💗 **The Empath** | Feeling & Connection | "Imagine how it feels... the emotional truth..." | Heart-centered learners, caregivers |
| 🔧 **The MacGyver** | Practical & Resourceful | "Here's how to use this... a tool you can apply..." | Hands-on learners, problem-solvers |
| ✨ **The Mystic** | Meaning & Wonder | "There's something deeper here... the mystery reveals..." | Spiritual seekers, meaning-makers |
| 🛡️ **The Provider** | Care & Protection | "This matters because it keeps us safe... nurtures..." | Parents, caregivers, protectors |
| 📖 **The Storyteller** | Narrative & Memory | "Once upon a time... picture this scene..." | Story-lovers, visual learners |
| 🎯 **The Strategist** | Planning & Winning | "The smart move is... position yourself to..." | Competitive learners, planners |
| 🏕️ **The Survivor** | Resilience & Grit | "When things get tough... you'll need to know..." | Pragmatic learners, preppers |

---

## Tone → Archetype Mapping

Users select a **tone** in settings. Recommended mappings:

| Tone Setting | Primary | Fallback |
|--------------|---------|----------|
| 🤔 Curious | The Scientist | The Explorer |
| 😊 Friendly | The Explorer | The Empath |
| 🎮 Playful | The Explorer | The Storyteller |
| 🦉 Wise | The Mystic | The Rebel |
| 💪 Coach | The Rebel | The Survivor |
| 📚 Scholar | The Scientist | The Architect |
| 🎨 Creative | The Storyteller | The Explorer |
| 🤝 Social | The Diplomat | The Empath |
| 🔧 Practical | The MacGyver | The Strategist |
| 🛡️ Protective | The Provider | The Survivor |

---

## Content Structure

Each lesson has **60 atoms** (12 archetypes × 5 phases):

```
Day N
├── The Explorer (5 phases)
├── The Scientist (5 phases)
├── The Rebel (5 phases)
├── The Architect (5 phases)
├── The Diplomat (5 phases)
├── The Empath (5 phases)
├── The MacGyver (5 phases)
├── The Mystic (5 phases)
├── The Provider (5 phases)
├── The Storyteller (5 phases)
├── The Strategist (5 phases)
└── The Survivor (5 phases)

Each archetype has:
├── Hook      (archetype-specific invitation)
├── Fact1     (first teaching moment)
├── Fact2     (deeper exploration)
├── Fact3     (application/synthesis)
└── Wisdom    (closing reflection)
```

**Exception:** Day 1 uses only the Primary Three (15 atoms) for historical reasons.

---

## Database Statistics

| Metric | Count |
|--------|-------|
| Total Archetypes | 12 |
| Primary Three atoms | 5,475 (3 × 1,825) |
| Extended Nine atoms | 16,380 (9 × 1,820) |
| **Total atoms** | **~21,855** |

---

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

---

## Frontend Implementation

### Recommended Approach

1. **Default to Primary Three** — Explorer, Scientist, Rebel cover most learners
2. **Expose Extended Nine via settings** — Power users can choose specialized archetypes
3. **Archetype quiz option** — Help users discover their best-fit archetype
4. **Fallback gracefully** — If an archetype is missing, use Explorer

### API Query Pattern

```javascript
// Fetch lesson atoms for specific archetype
const atoms = await supabase
  .from('lesson_atoms')
  .select('*')
  .eq('core_lesson_id', lessonId)
  .eq('archetype', userArchetype)
  .order('phase');
```

---

## Example: "How Sound Moves" Across Archetypes

| Archetype | Hook Opening |
|-----------|--------------|
| Explorer | "Sound is a tireless traveler on an invisible journey..." |
| Scientist | "Sound waves propagate at 343 m/s through air..." |
| Rebel | "They teach sound wrong in school—here's the truth..." |
| Architect | "Sound follows a precise blueprint as it moves..." |
| Diplomat | "Sound connects us all, carrying our voices to each other..." |
| Empath | "Have you ever felt sound move through you?..." |
| MacGyver | "Here's how to use sound as a tool..." |
| Mystic | "Sound is the universe speaking in vibration..." |
| Provider | "Understanding sound helps keep your family safe..." |
| Storyteller | "Imagine if you could see sound's journey..." |
| Strategist | "Master sound, and you master communication..." |
| Survivor | "In an emergency, sound can save your life..." |

---

## Why Twelve Works

1. **Comprehensive Coverage** — Catches every learning style
2. **Age Adaptability** — Some archetypes resonate more at different ages
3. **Engagement Diversity** — Re-learners can try new archetypes on repeat lessons
4. **Personalization Depth** — True customization, not just 3 flavors
5. **Future-Proof** — Content exists; UI can expose gradually

---

## Migration Note

The original "Golden Three" spec called for 3 archetypes. During content generation, all 12 were populated. Rather than delete 9 archetypes of work:

- ✅ Keeping all 12 archetypes
- ✅ Primary Three remain the default
- ✅ Extended Nine available for power users
- ✅ Slop cleaned from all 12 (Dec 2025)

---

*Created: December 2024*  
*Updated: December 8, 2025*  
*Status: PRODUCTION READY (12 archetypes)*
