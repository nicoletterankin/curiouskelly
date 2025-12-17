# Choice Generation Prompt Template for Anti

## Instructions for Anti

Add this to your `PersonaGenerator` to generate interactive choices for each lesson phase.

---

## Schema Update Required

Before running, add the `choices` field to your atom generation. The `content` JSONB should include:

```json
{
  "text": "The main phase content...",
  "choices": {
    "2-5": { "en": [...], "es": [...], "fr": [...] },
    "6-12": { "en": [...], "es": [...], "fr": [...] },
    "13-17": { "en": [...], "es": [...], "fr": [...] },
    "18-35": { "en": [...], "es": [...], "fr": [...] },
    "36-60": { "en": [...], "es": [...], "fr": [...] },
    "61+": { "en": [...], "es": [...], "fr": [...] }
  }
}
```

---

## The Prompt Template

````python
CHOICE_GENERATION_PROMPT = """
You are Kelly, a warm and intelligent AI teacher creating interactive learning experiences.

## Context
- Topic: {topic}
- Day: {day_number} of 365
- Phase: {phase} (this is a question/interaction moment)
- Universal Truth: {universal_truth}

## Existing Content
{phase_text}

## Your Task
Generate INTERACTIVE MULTIPLE CHOICE options for this learning moment.

Create choices for ALL 6 age groups in ALL 3 languages.

### Age Group Guidelines:
- **2-5 years**: Simple words, playful, focus on feelings and senses
- **6-12 years**: Curious explorer, connections to their world, fun facts
- **13-17 years**: Practical applications, career relevance, real-world examples
- **18-35 years**: Professional depth, efficiency, life applications
- **36-60 years**: Family perspective, community impact, wisdom sharing
- **61+ years**: Reflection, legacy, intergenerational wisdom

### Choice Structure:
- **Choice A**: Surface-level or partially correct understanding
- **Choice B**: Deeper insight - THE BEST ANSWER - shows true understanding
- **Choice C**: Nuanced/challenging perspective (for difficulty mode)

### Response Guidelines:
- Kelly's response to A: Gentle redirection, encouraging
- Kelly's response to B: Celebration, deeper elaboration
- Kelly's response to C: Thoughtful engagement with the complexity

## Output Format (STRICT JSON)

```json
{{
  "2-5": {{
    "en": [
      {{"letter": "A", "text": "...", "response": "That's a good start! Let me show you something even more amazing..."}},
      {{"letter": "B", "text": "...", "response": "Wonderful! You really understand this!"}},
      {{"letter": "C", "text": "...", "response": "Wow, you're thinking so deeply about this!"}}
    ],
    "es": [
      {{"letter": "A", "text": "...", "response": "¡Buen comienzo! Déjame mostrarte algo aún más increíble..."}},
      {{"letter": "B", "text": "...", "response": "¡Maravilloso! ¡Realmente entiendes esto!"}},
      {{"letter": "C", "text": "...", "response": "¡Guau, estás pensando muy profundamente sobre esto!"}}
    ],
    "fr": [
      {{"letter": "A", "text": "...", "response": "C'est un bon début ! Laisse-moi te montrer quelque chose d'encore plus incroyable..."}},
      {{"letter": "B", "text": "...", "response": "Merveilleux ! Tu comprends vraiment cela !"}},
      {{"letter": "C", "text": "...", "response": "Wow, tu réfléchis si profondément à cela !"}}
    ]
  }},
  "6-12": {{
    "en": [...],
    "es": [...],
    "fr": [...]
  }},
  "13-17": {{
    "en": [...],
    "es": [...],
    "fr": [...]
  }},
  "18-35": {{
    "en": [...],
    "es": [...],
    "fr": [...]
  }},
  "36-60": {{
    "en": [...],
    "es": [...],
    "fr": [...]
  }},
  "61+": {{
    "en": [...],
    "es": [...],
    "fr": [...]
  }}
}}
````

## Kelly Constitution Reminders

- Graceful Authority: Confident but humble
- Radical Curiosity: Questions over answers
- Emotional Intelligence: Meet learners where they are
- No judgment: Every answer is a learning moment
- Celebrate effort: Not just correct answers

Generate the choices JSON now:
"""

````

---

## Integration Code for PersonaGenerator

```python
# Add to persona_generator.py

async def generate_choices_for_atom(self, atom: dict, core_lesson: dict) -> dict:
    """Generate interactive choices for a lesson atom."""

    # Only generate choices for question phases
    question_phases = ['teaching', 'practice', 'reflection', 'Fact1', 'Fact2', 'Fact3']
    if atom['phase'] not in question_phases:
        return None

    prompt = CHOICE_GENERATION_PROMPT.format(
        topic=core_lesson['topic'],
        day_number=core_lesson['day_number'],
        phase=atom['phase'],
        universal_truth=core_lesson['universal_truth'],
        phase_text=atom['content'].get('text', atom['content'].get('script', ''))
    )

    response = await self.gemini.generate(prompt)
    choices = self._parse_json_response(response)

    return choices


async def update_atom_with_choices(self, atom_id: str, choices: dict):
    """Update an atom's content with generated choices."""

    # Get current content
    result = self.supabase.from_('lesson_atoms') \
        .select('content') \
        .eq('id', atom_id) \
        .single() \
        .execute()

    content = result.data['content']
    content['choices'] = choices

    # Update
    self.supabase.from_('lesson_atoms') \
        .update({'content': content}) \
        .eq('id', atom_id) \
        .execute()
````

---

## Bulk Generation Script

```python
# generate_all_choices.py

import asyncio
from persona_generator import PersonaGenerator

async def generate_all_choices():
    generator = PersonaGenerator()

    # Get all core lessons
    lessons = generator.supabase.from_('core_lessons') \
        .select('*') \
        .order('day_number') \
        .execute()

    # Priority order: December first (Days 335-365)
    priority_days = list(range(335, 366)) + list(range(1, 335))
    lessons_sorted = sorted(lessons.data, key=lambda x: priority_days.index(x['day_number']))

    for lesson in lessons_sorted:
        print(f"\n📚 Processing Day {lesson['day_number']}: {lesson['topic']}")

        # Get atoms for this lesson
        atoms = generator.supabase.from_('lesson_atoms') \
            .select('*') \
            .eq('core_lesson_id', lesson['id']) \
            .execute()

        for atom in atoms.data:
            # Skip welcome and wisdom (no choices needed)
            if atom['phase'] in ['welcome', 'wisdom', 'Hook', 'Wisdom']:
                continue

            # Check if already has choices
            if atom['content'].get('choices'):
                print(f"  ⏭️  Skipping {atom['phase']} (already has choices)")
                continue

            print(f"  🎯 Generating choices for {atom['phase']}...")

            try:
                choices = await generator.generate_choices_for_atom(atom, lesson)
                if choices:
                    await generator.update_atom_with_choices(atom['id'], choices)
                    print(f"  ✅ Done!")
                else:
                    print(f"  ⚠️  No choices generated")
            except Exception as e:
                print(f"  ❌ Error: {e}")

            # Rate limiting
            await asyncio.sleep(0.5)

    print("\n🎉 COMPLETE!")


if __name__ == "__main__":
    asyncio.run(generate_all_choices())
```

---

## Validation Script

```python
# validate_choices.py

def validate_choices():
    """Check all atoms have valid choices."""

    supabase = get_supabase_client()

    atoms = supabase.from_('lesson_atoms') \
        .select('id, phase, content, core_lesson_id') \
        .in_('phase', ['teaching', 'practice', 'reflection', 'Fact1', 'Fact2', 'Fact3']) \
        .execute()

    missing = []
    invalid = []

    required_ages = ['2-5', '6-12', '13-17', '18-35', '36-60', '61+']
    required_langs = ['en', 'es', 'fr']

    for atom in atoms.data:
        choices = atom['content'].get('choices')

        if not choices:
            missing.append(atom['id'])
            continue

        for age in required_ages:
            if age not in choices:
                invalid.append({'id': atom['id'], 'issue': f'Missing age: {age}'})
                continue

            for lang in required_langs:
                if lang not in choices[age]:
                    invalid.append({'id': atom['id'], 'issue': f'Missing {lang} for {age}'})
                elif len(choices[age][lang]) < 2:
                    invalid.append({'id': atom['id'], 'issue': f'Too few choices for {age}/{lang}'})

    print(f"Total atoms checked: {len(atoms.data)}")
    print(f"Missing choices: {len(missing)}")
    print(f"Invalid choices: {len(invalid)}")

    return {'missing': missing, 'invalid': invalid}
```

---

## Run Order

1. **Test first**: Run for Days 335-340 (5 days)
2. **Review quality**: Check generated choices manually
3. **Full run**: Generate all 365 days
4. **Validate**: Run validation script
5. **Fix failures**: Regenerate any failures

---

## Expected Output

After running, each lesson_atom will have:

```json
{
  "id": "uuid",
  "phase": "teaching",
  "content": {
    "text": "What does it mean to be a good citizen?",
    "choices": {
      "2-5": {
        "en": [
          {"letter": "A", "text": "Following rules", "response": "That's part of it! Being a citizen is also about..."},
          {"letter": "B", "text": "Helping others in your community", "response": "Exactly! Good citizens help make their community better!"},
          {"letter": "C", "text": "Both following rules AND helping others", "response": "Wow, you see the bigger picture!"}
        ],
        "es": [...],
        "fr": [...]
      },
      ...
    }
  }
}
```

---

## Questions for Anti

1. What's the current phase naming? (`Fact1/Fact2/Fact3` or `teaching/practice/reflection`?)
2. How many atoms currently exist per lesson?
3. Can you run a test batch (5 days) today?

---

_Template created: November 28, 2025_











