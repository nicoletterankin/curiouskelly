# Integration Plan: Anti's Engine + New Interactive Design

## Decision: Use Anti's Infrastructure + Generate Choices

### What We're Doing

```
Anti's lesson_atoms (TEXT)  →  Add choices layer  →  Interactive learn.html
           ↓                         ↓                        ↓
    "Today we explore X"    +   [A] [B] [C]     =    Full experience
```

---

## Phase 1: Schema Update (Day 1)

### Option A: Add `phase_choices` Table to Supabase

```sql
-- New table for interactive choices
CREATE TABLE phase_choices (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  lesson_atom_id UUID REFERENCES lesson_atoms(id),

  -- Which age group this applies to
  age_group VARCHAR(10) NOT NULL, -- '2-5', '6-12', '13-17', '18-35', '36-60', '61+'

  -- The choices
  choice_a_text TEXT NOT NULL,
  choice_a_response TEXT NOT NULL,
  choice_b_text TEXT NOT NULL,
  choice_b_response TEXT NOT NULL,
  choice_c_text TEXT, -- Optional for difficulty=3
  choice_c_response TEXT,

  -- Language
  language VARCHAR(5) DEFAULT 'en', -- 'en', 'es', 'fr'

  created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX idx_phase_choices_atom ON phase_choices(lesson_atom_id);
CREATE INDEX idx_phase_choices_age ON phase_choices(age_group);
```

### Option B: Add `choices` Field to Existing `lesson_atoms.content`

Update Anti's generator to include choices in the JSONB:

```json
{
  "text": "What does it mean to be a good citizen?",
  "choices": {
    "2-5": {
      "en": [
        {"letter": "A", "text": "Being nice to friends", "response": "That's a great start!"},
        {"letter": "B", "text": "Helping your community", "response": "Wonderful thinking!"}
      ],
      "es": [...],
      "fr": [...]
    },
    "6-12": {...},
    ...
  }
}
```

**Recommendation: Option B** - Less schema change, uses existing content field.

---

## Phase 2: Archetype → Age Group Mapping (Day 1)

### Current Archetypes (Anti's):

```
"The Scientist", "The Explorer", "The Nurturer",
"The Innovator", "The Storyteller", etc.
```

### Required Age Groups (New Design):

```
"2-5", "6-12", "13-17", "18-35", "36-60", "61+"
```

### Solution: Create Mapping Function

```javascript
function archetypeToAgeContent(archetype, targetAge) {
  // Map archetypes to age-appropriate content
  const archetypeAgeMap = {
    'The Explorer': ['2-5', '6-12'], // Curious, discovery
    'The Scientist': ['13-17', '18-35'], // Analytical
    'The Nurturer': ['36-60'], // Family focus
    'The Sage': ['61+'] // Wisdom, reflection
  };

  // Find best archetype for target age
  for (const [archetype, ages] of Object.entries(archetypeAgeMap)) {
    if (ages.includes(targetAge)) {
      return archetype;
    }
  }
  return 'The Scientist'; // Default
}
```

---

## Phase 3: Choice Generation Script (Days 2-3)

### Script: `generate-choices.py` (Uses Anti's Gemini Setup)

```python
"""
Generate interactive choices for existing lesson_atoms
Uses Anti's existing Gemini API connection
"""

from persona_generator import PersonaGenerator  # Anti's existing generator
from supabase_client import supabase  # Anti's existing connection

CHOICE_PROMPT = """
Based on this lesson content about "{topic}":

"{atom_text}"

Generate 2-3 multiple choice options for a {age_group} learner.

Requirements:
- Option A: Surface-level understanding
- Option B: Deeper insight (best answer)
- Option C (optional): Nuanced/challenge answer

For each option, provide:
1. The choice text (age-appropriate)
2. Kelly's response if selected

Output JSON:
{{
  "choices": [
    {{"letter": "A", "text": "...", "response": "..."}},
    {{"letter": "B", "text": "...", "response": "..."}},
    {{"letter": "C", "text": "...", "response": "..."}}
  ]
}}
"""

async def generate_choices_for_atom(atom, age_group, language='en'):
    """Generate choices for a single atom"""
    prompt = CHOICE_PROMPT.format(
        topic=atom['topic'],
        atom_text=atom['content']['text'],
        age_group=age_group
    )

    response = await gemini.generate(prompt)
    return parse_choices(response)


async def bulk_generate_choices():
    """Generate choices for all question-phase atoms"""

    # Get all atoms that need choices (teaching, practice, reflection phases)
    atoms = supabase.from_('lesson_atoms') \
        .select('*') \
        .in_('phase', ['teaching', 'practice', 'reflection']) \
        .execute()

    age_groups = ['2-5', '6-12', '13-17', '18-35', '36-60', '61+']
    languages = ['en', 'es', 'fr']

    for atom in atoms.data:
        for age in age_groups:
            for lang in languages:
                choices = await generate_choices_for_atom(atom, age, lang)

                # Update atom content with choices
                atom['content']['choices'] = atom['content'].get('choices', {})
                atom['content']['choices'][age] = atom['content']['choices'].get(age, {})
                atom['content']['choices'][age][lang] = choices

                # Save back to Supabase
                supabase.from_('lesson_atoms') \
                    .update({'content': atom['content']}) \
                    .eq('id', atom['id']) \
                    .execute()

                await rate_limit_wait()
```

---

## Phase 4: Integration with learn.html (Days 4-5)

### Update loadLesson() Function

```javascript
async function loadLesson(dayNumber, age, language, difficulty) {
  // 1. Get core lesson
  const { data: lesson } = await supabase
    .from('core_lessons')
    .select('*')
    .eq('day_number', dayNumber)
    .single();

  // 2. Get atoms with choices
  const archetype = archetypeToAgeContent(age);
  const { data: atoms } = await supabase
    .from('lesson_atoms')
    .select('*')
    .eq('core_lesson_id', lesson.id)
    .eq('archetype', archetype);

  // 3. Extract choices for this age/language
  const phases = atoms.map((atom) => ({
    type: atom.phase,
    text: atom.content.text,
    choices: atom.content.choices?.[age]?.[language] || null
  }));

  // 4. Filter choices by difficulty (2 or 3)
  phases.forEach((phase) => {
    if (phase.choices && difficulty === 2) {
      phase.choices = phase.choices.slice(0, 2); // Only A and B
    }
  });

  return { lesson, phases };
}
```

---

## Timeline

| Day       | Task                                            | Owner         |
| --------- | ----------------------------------------------- | ------------- |
| **1**     | Update Supabase schema (add choices to content) | You/Anti      |
| **2**     | Create choice generation prompt template        | Me            |
| **3**     | Run choice generation for all atoms             | Anti's engine |
| **4-10**  | Bulk generate choices (use Gemini credits)      | Automated     |
| **11-12** | Validate & fix failed generations               | Both          |
| **13-15** | Integration testing with learn.html             | Me            |
| **16-17** | Final QA & launch prep                          | Both          |

---

## Effort Split

### Anti's Engine Does:

- ✅ Gemini API calls (uses existing credits)
- ✅ Bulk generation infrastructure
- ✅ Write to Supabase
- ✅ Error handling & retries

### I Do (in this codebase):

- ✅ Schema update SQL
- ✅ Choice prompt template
- ✅ learn.html integration code
- ✅ Frontend testing

---

## Cost Estimate

Using Anti's Gemini credits:

- 21,915 atoms × 6 ages × 3 languages = ~394,470 generations
- But only 3 phases need choices: 365 × 3 phases × 12 archetypes × 6 ages × 3 languages = ~236,520
- At ~$0.0001/call = **~$24 in Gemini costs**

---

## Decision for You

**Option 1: I build a standalone generator** (doesn't use Anti's engine)

- Pro: I control everything, can start immediately
- Con: Duplicate work, need to set up API separately

**Option 2: Coordinate with Anti's engine** (recommended)

- Pro: Leverage existing infrastructure, use Gemini credits
- Con: Need to sync on schema changes, coordinate timing

**Option 3: Hybrid**

- Anti generates base content
- I generate choices as a separate layer
- Merge in learn.html

---

## Next Step

Tell Anti:

> "We need to add interactive choices to the lesson_atoms content. Each teaching/practice/reflection phase needs 2-3 choices for each age group (2-5, 6-12, 13-17, 18-35, 36-60, 61+) in 3 languages (EN/ES/FR). The choices need to follow this format: [show JSON format above]. Can you update the PersonaGenerator to include this?"

Or I can create the choice generation script that Anti can plug into their existing pipeline.

**What do you want me to do?**





