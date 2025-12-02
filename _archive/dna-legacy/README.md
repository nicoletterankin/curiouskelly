# Legacy DNA Files Archive

**Archived:** 2025-12-01 09:23:06
**Reason:** Supabase is now the single source of truth for lesson content

## Why These Were Archived

DNA files were an earlier content format containing:
- Age-variant lesson scripts
- Multilingual translations
- Voice profiles
- Interactive choices

However, this content was OUT OF SYNC with production Supabase data:
- DNA files: "The Sun", "Habit Stacking", "Planet Earth"
- Supabase: "Starting Fresh", "Three Lives of Water", "Where Clouds Come From"

## Current Architecture

```
SUPABASE (Single Source of Truth)
├── core_lessons (365 topics)
├── lesson_atoms (21,915 content pieces)
└── lesson_shards (38,700 demographic variants)
```

## If You Need This Content

The rich DNA structure (age variants, translations, voice profiles) can be 
regenerated FROM Supabase data if needed. The schema is preserved here for reference.

## Files Archived

- 365_day_dna_metadata.json
- aging_process_dna.json
- applied-mathematics-math-in-the-real-world-dna.json
- applied_mathematics___math_in_the_real_world_dna.json
- creative_writing_dna.json
- dance_expression_dna.json
- genetic_engineering___editing_the_code_of_life_dna.json
- molecular_biology_dna.json
- negotiation_skills_dna.json
- nutrition_science_dna.json
- poetry_dna.json
- the-sun-dna.json
- creative-writing-dna.json
- dance-expression-dna.json
- disruptive_innovation_dna.json
- emotional-intelligence-dna.json
- genetic-engineering-editing-the-code-of-life-dna.json
- habit-stacking-dna.json
- molecular-biology-dna.json
- negotiation-skills-dna.json
- nutrition-science-dna.json
- parasitology_dna.json
- planet-earth-dna.json
- plasma_physics_dna.json
- poetry-dna.json
- simple-machines-dna.json
- stem_cells_dna.json
- the-sun-dna.json
- 365_day_dna_metadata.json
- aging_process_dna.json
- applied-mathematics-math-in-the-real-world-dna.json
- applied_mathematics___math_in_the_real_world_dna.json
- creative_writing_dna.json
- dance_expression_dna.json
- genetic_engineering___editing_the_code_of_life_dna.json
- molecular_biology_dna.json
- negotiation_skills_dna.json
- nutrition_science_dna.json
- poetry_dna.json
- the-sun-dna.json
- creative-writing-dna.json
- dance-expression-dna.json
- disruptive_innovation_dna.json
- genetic-engineering-editing-the-code-of-life-dna.json
- molecular-biology-dna.json
- negotiation-skills-dna.json
- nutrition-science-dna.json
- parasitology_dna.json
- plasma_physics_dna.json
- poetry-dna.json
- stem_cells_dna.json
- the-sun-dna.json
- the-sun-dna.json
